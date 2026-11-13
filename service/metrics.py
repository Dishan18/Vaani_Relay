"""Prometheus metrics (GET /metrics). Buckets match the latencies this system actually has: cache hits in
milliseconds, answers in 1-30 s."""
from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

REGISTRY = CollectorRegistry()
LAT = (0.01, 0.05, 0.1, 0.25, 0.5, 1, 1.5, 2, 3, 5, 8, 12, 20, 30, 60)

REQUESTS = Counter("relay_requests_total", "Requests answered", ["kind", "level", "cache"], registry=REGISTRY)
FIRST_TOKEN = Histogram("relay_first_token_seconds", "Arrival to the first released sentence", buckets=LAT,
                        registry=REGISTRY)
TOTAL = Histogram("relay_total_seconds", "Arrival to the complete reply", buckets=LAT, registry=REGISTRY)
TOKENS = Counter("relay_model_tokens_total", "Model tokens", ["model", "direction"], registry=REGISTRY)
MODEL_CALLS = Counter("relay_model_calls_total", "Model calls by outcome", ["model", "step", "status"],
                      registry=REGISTRY)
GLOBAL_DENIED = Counter("relay_global_budget_denied_total", "Calls refused by the fleet-wide rate limit",
                        ["model", "bucket"], registry=REGISTRY)
CLIENT_LIMITED = Counter("relay_client_rate_limited_total", "Requests refused by a client's quota", registry=REGISTRY)
DROPPED_CLAIMS = Counter("relay_claims_dropped_total", "Claims withheld because their quote/number failed",
                         registry=REGISTRY)
INFLIGHT = Gauge("relay_inflight_requests", "Requests being answered by this process", registry=REGISTRY)
JOBS = Counter("relay_jobs_total", "Async jobs by outcome", ["status"], registry=REGISTRY)
