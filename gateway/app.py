"""
PrivateCloudLLM FastAPI gateway.

OpenAI-compatible /v1/chat/completions → mock vLLM (or real vLLM).
Redis-backed conversation + fake-vector memory retrieval.
Prometheus metrics for latency, tokens, failures, memory hits, health.
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
import struct
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

import httpx
import redis
from fastapi import FastAPI, Header, HTTPException, Request, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

VLLM_URL = os.getenv("VLLM_URL", "http://vllm-mock:8001").rstrip("/")
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
MEMORY_TOP_K = int(os.getenv("MEMORY_TOP_K", "3"))
EMBED_DIM = 32  # tiny deterministic fake embeddings — no model weights

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s [%(name)s] request_id=%(request_id)s %(message)s",
)
logger = logging.getLogger("gateway")


class RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "request_id"):
            record.request_id = "-"
        return True


logger.addFilter(RequestIdFilter())

# ---------------------------------------------------------------------------
# Prometheus metrics
# ---------------------------------------------------------------------------

REQ_LATENCY = Histogram(
    "gateway_request_latency_seconds",
    "End-to-end chat completion latency",
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)
TOKENS_TOTAL = Counter(
    "gateway_tokens_total",
    "Tokens processed by role",
    ["kind"],  # prompt | completion | total
)
FAILED_REQUESTS = Counter(
    "gateway_failed_requests_total",
    "Failed chat completion requests",
    ["reason"],
)
MEMORY_RETRIEVALS = Counter(
    "gateway_memory_retrievals_total",
    "Memory retrieval operations",
    ["result"],  # hit | miss | empty
)
HEALTH_GAUGE = Gauge("gateway_up", "Gateway process health (1=up)")

# ---------------------------------------------------------------------------
# Redis + fake embeddings
# ---------------------------------------------------------------------------

redis_client: redis.Redis | None = None


def get_redis() -> redis.Redis:
    global redis_client
    if redis_client is None:
        redis_client = redis.from_url(REDIS_URL, decode_responses=True)
    return redis_client


def fake_embed(text: str, dim: int = EMBED_DIM) -> list[float]:
    """Deterministic hash-seeded unit vector — demo only, not a real embedding model."""
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    vals: list[float] = []
    seed = digest
    while len(vals) < dim:
        for i in range(0, len(seed) - 3, 4):
            if len(vals) >= dim:
                break
            raw = struct.unpack_from(">I", seed, i)[0]
            vals.append((raw / 0xFFFFFFFF) * 2.0 - 1.0)
        seed = hashlib.sha256(seed).digest()
    # L2 normalize
    norm = math.sqrt(sum(v * v for v in vals)) or 1.0
    return [v / norm for v in vals]


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def store_memory(session_id: str, role: str, content: str) -> None:
    r = get_redis()
    mem_id = uuid.uuid4().hex[:12]
    key = f"mem:{session_id}:{mem_id}"
    emb = fake_embed(content)
    pipe = r.pipeline()
    pipe.hset(
        key,
        mapping={
            "role": role,
            "content": content[:2000],
            "embedding": ",".join(f"{x:.6f}" for x in emb),
            "ts": str(time.time()),
        },
    )
    pipe.sadd(f"memidx:{session_id}", key)
    pipe.lpush(f"hist:{session_id}", f"{role}:{content[:500]}")
    pipe.ltrim(f"hist:{session_id}", 0, 49)
    pipe.execute()


def retrieve_memory(session_id: str, query: str, top_k: int = MEMORY_TOP_K) -> list[dict[str, Any]]:
    r = get_redis()
    keys = list(r.smembers(f"memidx:{session_id}"))
    if not keys:
        MEMORY_RETRIEVALS.labels(result="empty").inc()
        return []
    q = fake_embed(query)
    scored: list[tuple[float, dict[str, Any]]] = []
    for key in keys:
        data = r.hgetall(key)
        if not data or "embedding" not in data:
            continue
        emb = [float(x) for x in data["embedding"].split(",")]
        score = cosine(q, emb)
        scored.append(
            (
                score,
                {
                    "role": data.get("role", "unknown"),
                    "content": data.get("content", ""),
                    "score": round(score, 4),
                },
            )
        )
    scored.sort(key=lambda t: t[0], reverse=True)
    hits = [item for _, item in scored[:top_k]]
    MEMORY_RETRIEVALS.labels(result="hit" if hits else "miss").inc()
    return hits


# ---------------------------------------------------------------------------
# API models
# ---------------------------------------------------------------------------


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "mock-llm"
    messages: list[ChatMessage]
    temperature: float = 0.0
    max_tokens: int = 256
    # Non-standard but useful for demo memory scoping
    session_id: str | None = Field(default=None, description="Conversation / memory scope")


# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(_app: FastAPI):
    HEALTH_GAUGE.set(1)
    # Warm Redis connection (non-fatal if Redis is briefly unavailable)
    try:
        get_redis().ping()
        logger.info("Redis connected", extra={"request_id": "-"})
    except Exception as exc:  # noqa: BLE001
        logger.warning("Redis not ready yet: %s", exc, extra={"request_id": "-"})
    yield
    HEALTH_GAUGE.set(0)


app = FastAPI(title="PrivateCloudLLM Gateway", version="0.1.0", lifespan=lifespan)


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
    request.state.request_id = rid
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000
    response.headers["X-Request-Id"] = rid
    response.headers["X-Response-Time-Ms"] = f"{elapsed_ms:.1f}"
    logger.info(
        "%s %s -> %s (%.1fms)",
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
        extra={"request_id": rid},
    )
    return response


@app.get("/health")
def health() -> dict[str, Any]:
    redis_ok = False
    try:
        redis_ok = bool(get_redis().ping())
    except Exception:  # noqa: BLE001
        redis_ok = False
    return {
        "status": "ok" if redis_ok else "degraded",
        "service": "gateway",
        "redis": redis_ok,
        "vllm_url": VLLM_URL,
    }


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/v1/chat/completions")
async def chat_completions(
    req: ChatCompletionRequest,
    request: Request,
    x_session_id: str | None = Header(default=None),
) -> dict[str, Any]:
    rid = getattr(request.state, "request_id", "-")
    session_id = req.session_id or x_session_id or "default"
    last_user = next((m.content for m in reversed(req.messages) if m.role == "user"), "")

    memories = retrieve_memory(session_id, last_user) if last_user else []
    augmented = list(req.messages)
    if memories:
        memory_block = "\n".join(
            f"- ({m['role']}, score={m['score']}): {m['content']}" for m in memories
        )
        augmented = [
            ChatMessage(
                role="system",
                content=f"Retrieved memory for session '{session_id}':\n{memory_block}",
            ),
            *augmented,
        ]

    payload = {
        "model": req.model,
        "messages": [m.model_dump() for m in augmented],
        "temperature": req.temperature,
        "max_tokens": req.max_tokens,
    }

    t0 = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(f"{VLLM_URL}/v1/chat/completions", json=payload)
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPStatusError as exc:
        FAILED_REQUESTS.labels(reason="upstream_http").inc()
        logger.error("Upstream HTTP error: %s", exc, extra={"request_id": rid})
        raise HTTPException(status_code=502, detail=f"vLLM upstream error: {exc.response.status_code}") from exc
    except Exception as exc:  # noqa: BLE001
        FAILED_REQUESTS.labels(reason="upstream_error").inc()
        logger.error("Upstream error: %s", exc, extra={"request_id": rid})
        raise HTTPException(status_code=502, detail="vLLM unreachable") from exc
    finally:
        REQ_LATENCY.observe(time.perf_counter() - t0)

    usage = data.get("usage") or {}
    TOKENS_TOTAL.labels(kind="prompt").inc(usage.get("prompt_tokens", 0))
    TOKENS_TOTAL.labels(kind="completion").inc(usage.get("completion_tokens", 0))
    TOKENS_TOTAL.labels(kind="total").inc(usage.get("total_tokens", 0))

    # Persist user + assistant turns into Redis memory
    try:
        if last_user:
            store_memory(session_id, "user", last_user)
        assistant = (
            (data.get("choices") or [{}])[0].get("message", {}).get("content") or ""
        )
        if assistant:
            store_memory(session_id, "assistant", assistant)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Memory store failed: %s", exc, extra={"request_id": rid})

    data["privatecloudllm"] = {
        "session_id": session_id,
        "request_id": rid,
        "memory_hits": memories,
    }
    return data
