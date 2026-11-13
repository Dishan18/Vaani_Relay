"""Service settings, from environment variables only (twelve-factor), with safe defaults for local runs.

  REDIS_URL               redis://host:6379/0. Unset: in-process MemoryStore (one process only).
  RELAY_LLM_BASE_URL      OpenAI-compatible endpoint. Default Groq; point it at the stand-in
                          (http://standin:8790/openai/v1) to run without a key or budget.
  GROQ_API_KEY            the provider key (or Vaani_Relay/.env). Never logged.
  RELAY_API_KEYS          comma-separated client keys accepted in "Authorization: Bearer <key>".
                          Unset: no auth (local development only).
  RELAY_CLIENT_RPM        requests per minute per client key (token bucket in the store). Default 60.
  RELAY_CACHE_TTL_S       how long a full, checked answer is reused. Default 86400 (the papers don't change).
  RELAY_DEGRADED_TTL_S    TTL for answers produced in a degraded mode (fallback model, lean). Default 120.
  RELAY_SINGLEFLIGHT_S    how long a duplicate request waits for the in-flight original. Default 25.
  RELAY_GLOBAL_LIMIT_FRACTION  share of each model's measured limits the whole fleet may use. Default 0.9.
  RELAY_CONFIG            "production" (default) or "submitted" (the measured v1 pipeline).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env(name: str, default: str | None = None) -> str | None:
    v = os.environ.get(name)
    return v if v not in (None, "") else default


@dataclass(frozen=True)
class Settings:
    redis_url: str | None = field(default_factory=lambda: _env("REDIS_URL"))
    llm_base_url: str = field(default_factory=lambda: _env("RELAY_LLM_BASE_URL", "https://api.groq.com/openai/v1"))
    api_keys: frozenset = field(default_factory=lambda: frozenset(
        k.strip() for k in (_env("RELAY_API_KEYS", "") or "").split(",") if k.strip()))
    client_rpm: float = field(default_factory=lambda: float(_env("RELAY_CLIENT_RPM", "60")))
    cache_ttl_s: float = field(default_factory=lambda: float(_env("RELAY_CACHE_TTL_S", "86400")))
    degraded_ttl_s: float = field(default_factory=lambda: float(_env("RELAY_DEGRADED_TTL_S", "120")))
    singleflight_s: float = field(default_factory=lambda: float(_env("RELAY_SINGLEFLIGHT_S", "25")))
    global_fraction: float = field(default_factory=lambda: float(_env("RELAY_GLOBAL_LIMIT_FRACTION", "0.9")))
    config: str = field(default_factory=lambda: _env("RELAY_CONFIG", "production"))

    @property
    def is_groq(self) -> bool:
        return "groq.com" in self.llm_base_url
