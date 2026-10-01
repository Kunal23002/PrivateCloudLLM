"""
Tiny OpenAI-compatible chat completions stub.
Deterministic replies + token counts — no model weights.
Swap this service for real vLLM later (same /v1/chat/completions contract).
"""

from __future__ import annotations

import hashlib
import time
import uuid
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(title="vLLM Mock", version="0.1.0")


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "mock-llm"
    messages: list[ChatMessage]
    temperature: float = 0.0
    max_tokens: int = 256


def _estimate_tokens(text: str) -> int:
    # Rough ~4 chars/token heuristic for demo metrics
    return max(1, len(text) // 4)


def _deterministic_reply(messages: list[ChatMessage]) -> str:
    last_user = ""
    for m in reversed(messages):
        if m.role == "user":
            last_user = m.content
            break
    digest = hashlib.sha256(last_user.encode("utf-8")).hexdigest()[:8]
    return (
        f"[mock-vllm] Echo digest={digest}. "
        f"You said: {last_user[:400] or '(empty)'}. "
        "This is a deterministic stub — replace with real vLLM for production inference."
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "vllm-mock"}


@app.get("/v1/models")
def list_models() -> dict[str, Any]:
    return {
        "object": "list",
        "data": [
            {
                "id": "mock-llm",
                "object": "model",
                "created": 0,
                "owned_by": "privatecloudllm",
            }
        ],
    }


@app.post("/v1/chat/completions")
def chat_completions(req: ChatCompletionRequest) -> dict[str, Any]:
    reply = _deterministic_reply(req.messages)
    prompt_text = " ".join(m.content for m in req.messages)
    prompt_tokens = _estimate_tokens(prompt_text)
    completion_tokens = _estimate_tokens(reply)
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": reply},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }
