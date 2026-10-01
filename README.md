# PrivateCloudLLM

**Kubernetes-native private LLM platform (resume demo)** — FastAPI gateway, OpenAI-compatible endpoint, Redis conversation memory, Prometheus/Grafana observability.

> Honest scope: this is a **mini demo**, not production. Inference uses a tiny **mock vLLM** stub (no model weights). Real vLLM can be swapped in via the same OpenAI chat completions API.

## What it demonstrates (resume mapping)

| Resume bullet | What this repo shows |
|---|---|
| Kubernetes-native private LLM platform with OpenAI-compatible endpoint via FastAPI gateway | `gateway/` proxies `POST /v1/chat/completions`; `k8s/` Deployments/Services mirror the compose stack |
| Prometheus/Grafana for latency, tokens, failed requests, memory retrieval, pod health | `/metrics` + `observability/` scrape config + Grafana dashboard JSON |
| Redis vector-search memory across sessions | Redis stores snippets + deterministic fake embeddings; gateway does cosine top-k retrieval |

## Architecture

```
Client → FastAPI gateway (:8000)
            ├─ Redis memory (retrieve + store)
            ├─ mock vLLM (:8001)  ← swap for real vLLM
            └─ /metrics → Prometheus → Grafana
```

## Quick start (local)

```bash
docker compose up --build
```

Then:

```bash
# Chat (OpenAI-compatible)
curl -s http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'X-Session-Id: demo-1' \
  -d '{
    "model": "mock-llm",
    "messages": [{"role":"user","content":"Remember that my favorite color is teal."}],
    "session_id": "demo-1"
  }' | jq .

# Second turn — memory retrieval should surface prior snippet
curl -s http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "mock-llm",
    "messages": [{"role":"user","content":"What is my favorite color?"}],
    "session_id": "demo-1"
  }' | jq .

# Health + metrics
curl -s http://localhost:8000/health | jq .
curl -s http://localhost:8000/metrics | head
```

| Service | URL |
|---|---|
| Gateway | http://localhost:8000 |
| Mock vLLM | http://localhost:8001 |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 (admin/admin; anonymous Viewer enabled) |

Dashboard: **PrivateCloudLLM Gateway** (provisioned automatically).

## Kubernetes manifests

Minimal Deployments/Services/ConfigMaps in `k8s/` for the same architecture (resume/demo story — a cluster is optional).

```bash
# Build images for the cluster
docker build -t privatecloudllm/gateway:demo ./gateway
docker build -t privatecloudllm/vllm-mock:demo ./vllm-mock

kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/redis.yaml
kubectl apply -f k8s/vllm-mock.yaml
kubectl apply -f k8s/gateway.yaml
kubectl apply -f k8s/prometheus.yaml
kubectl apply -f k8s/grafana.yaml

kubectl -n privatecloudllm port-forward svc/gateway 8000:8000
```

## Swap in real vLLM later

Point the gateway at a real OpenAI-compatible server:

```bash
# compose override example
VLLM_URL=http://vllm:8000 docker compose up --build
```

Or in Kubernetes, edit `k8s/gateway.yaml` ConfigMap `VLLM_URL` to your vLLM Service. The mock service implements `/v1/chat/completions` and `/v1/models` so the contract stays the same.

## What's mocked vs real

| Component | Status |
|---|---|
| FastAPI OpenAI-compatible gateway | Real |
| Redis conversation + memory index | Real (Redis Alpine) |
| Vector search | Demo: hash-seeded fake embeddings + cosine in gateway (no embedding model / no Redis Stack) |
| LLM inference | **Mock** deterministic stub (`vllm-mock/`) — no GPU, no weights |
| Prometheus scrape + Grafana dashboard | Real configs (demo panels) |
| Kubernetes | Manifest-only demo of the same topology |

## Repo layout

```
gateway/           FastAPI gateway + metrics + Redis memory
vllm-mock/         Tiny OpenAI-compatible chat stub
observability/     Prometheus + Grafana provisioning
k8s/               Deployments / Services / ConfigMaps
docker-compose.yml One-command local demo
```

## Design constraints

- Free / low disk: slim Python images, Redis Alpine, no LLM weights committed
- Resume-demo quality, intentionally simple
- No secrets required for the local demo

## License

MIT — see `LICENSE`.
