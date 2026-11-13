# The Relay

A cited question-answering service over three arXiv papers (*Attention Is All You Need*, *Moshi*,
*H-Net / Dynamic Chunking*), built for Vaani's ML Engineering assignment and then rebuilt as a
production-shaped service after the interview review.

Every answer cites where in the papers it came from, or says the papers don't cover it. The system
is built to stay honest under load: Groq free-tier rate limits, bursts, slow and failing calls.

- **`submission/`** — what was submitted (write-up, `results.csv`, code at the submitted commit).
- **v2 (this branch)** — hybrid retrieval, citation metadata instead of citation parsing, sentence
  streaming, and an HTTP service with Redis. Why each change was made, and what it measured:
  [docs/V2_DESIGN.md](docs/V2_DESIGN.md).

## Quick start

### The service (Docker)

```bash
cp .env.example .env          # add GROQ_API_KEY to use Groq; without it the stand-in model answers
docker compose up --build     # API on http://localhost:8080 (2 replicas behind nginx), Redis, worker
```

```bash
curl -s localhost:8080/v1/ask -H "Authorization: Bearer dev-key-1" -H "Content-Type: application/json" \
     -d '{"question": "What dropout rate does the base Transformer use?"}'
```

Streaming, one verified sentence at a time (Server-Sent Events):

```bash
curl -N localhost:8080/v1/ask/stream -H "Authorization: Bearer dev-key-1" -H "Content-Type: application/json" \
     -d '{"question": "Why use several attention heads instead of one?"}'
```

| Endpoint | What it does |
|---|---|
| `POST /v1/ask` | Whole answer as JSON: text, `claims` (each sentence with its sources and supporting quote), sources, ladder level, timings, cache status |
| `POST /v1/ask/stream` | SSE: `claim` events as each sentence passes its check, then `done` |
| `POST /v1/jobs`, `GET /v1/jobs/{id}` | Async answering through the Redis Streams queue and the worker |
| `GET /healthz`, `/readyz`, `/metrics` | Liveness, readiness (store + index), Prometheus |

`docker compose --profile observability up` adds Prometheus on `:9090`.

### Without Docker

```bash
python -m pip install -r requirements-service.txt
uvicorn service.app:app --port 8000          # in-process store; set REDIS_URL to share state
python ask.py "How does H-Net decide where to cut the byte stream?" --real --stream
```

`ask.py` answers one question from the terminal: retrieval preview only (free), `--standin`, or
`--real` (Groq, spends budget). `--pdf FILE --title "..."` adds a new paper for that run;
`--legacy` runs the submitted pipeline.

## Architecture

```
client ─► nginx ─► API replicas (FastAPI) ──┬─► Redis: answer cache · single-flight locks ·
                        │                   │          fleet-wide token buckets (Lua) · client quotas ·
                        │                   │          job queue (Streams) · events stream
                        ▼                   │
                 relay.pipeline             └─◄ worker (async jobs, at-least-once, XAUTOCLAIM)
   plan (ladder) → router (+HyDE) → hybrid retrieval → decide → claims draft → per-claim check → stream
                        │
                        ▼
              Groq (or the stand-in model service)
```

| Step | v1 (submitted) | v2 (now) |
|---|---|---|
| Retrieval | BM25, 2400-char pieces | BM25 + bge-small embeddings fused by reciprocal rank; search 600-char children, return 2400-char parents; router-written HyDE for the dense query; optional cross-encoder rerank |
| Decide | refuse on weak word overlap | refuse only if word overlap **and** semantic similarity are weak |
| Draft | prose with `[S1]` markers | JSON Lines claims `{text, sources, quote}`: citations are metadata the system renders |
| Check | regex parsing of citations, numbers, quotes | per claim: quote must occur in a cited source, numbers must be in it; a failing claim is dropped, not the whole answer |
| Release | buffered: first token = total | streamed: first token = first verified sentence |
| Limits | one process's limiter | + a Redis bucket per model shared by every replica |
| Repeats | every request drafts | cache + single-flight: a burst of one question costs one draft |

Measured (details and caveats in [docs/V2_DESIGN.md](docs/V2_DESIGN.md)):
- Retrieval, 46 answerable questions, 32 of them in plain engineer wording: the answer is in what the
  drafter reads for **83%** (submitted: 63%); on the plain-wording set **78%** (submitted: 50%).
- Answer quality on the 18 graded questions, real Groq, same drafter and day: on par with the submitted
  pipeline (44–50% vs 44% with gpt-oss-20b; one question is 5.6 points). Claims mode streams the first
  verified sentence in ~1.5 s and uses 23% fewer tokens than prose.
- Service: a burst of 20 requests over 4 distinct questions cost 4 drafts; repeats are served from the
  cache in under 1 ms.

## Tests

```bash
python -m unittest discover -s tests       # 197 tests, no key, no network, no Redis
```

## Repository map

| Path | What |
|---|---|
| `relay/` | the pipeline (`pipeline.py`), rate limiter, retrieval (`retrieve.py` BM25, `dense.py` hybrid), claims (`claims.py`), streaming client, budget ledger, grader |
| `service/` | API (`app.py`), engine (cache, single-flight, fleet gate), store (Redis / memory), worker, metrics |
| `standin/` | the stand-in model service: Groq's API, headers and 429s, calibrated from the real probe |
| `config/` | `pipeline.toml` as submitted and measured; `production.toml` the v2 overrides |
| `scripts/` | probe, evals, load runs, results and write-up builders |
| `data/` | papers, extracted blocks, questions with gold facts, probe and retrieval results |
| `results/` | raw results of every run, `results.csv`, tables |
| `deploy/`, `Dockerfile`, `docker-compose.yml` | containers, nginx, Prometheus |
