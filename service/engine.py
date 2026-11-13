"""One process's answering engine: the pipeline plus what a fleet of processes needs to share.

    engine = await Engine.create(settings)
    reply, meta = await engine.ask("Why divide by sqrt(d_k)?", on_claim=print)

Per request, in order:
  1. cache      key = normalised question + library version + config. A checked answer is reused
                (24 h; 2 min if it came from a degraded rung, so the full answer replaces it soon).
  2. single-flight  bursts arrive right after an outage, mostly with the same question. The first
                request takes a short lock; duplicates wait for its cached result instead of each
                spending a draft. Our Table 2b burst: 25 people in 5 s, 29% got passages only; every
                duplicate among them would now share one answer.
  3. pipeline   relay.pipeline with config/production.toml (hybrid retrieval, claims, streaming).
  4. fleet-wide rate limit   the in-process limiter predicts waits from Groq's headers; with several
                replicas each sees only its own calls, so before a call is sent it must also take
                1 request and its token estimate from a shared Redis bucket per model (the measured
                limits x RELAY_GLOBAL_LIMIT_FRACTION). Denied -> the pipeline treats the model as busy
                and steps to the next one, exactly as for a predicted wait. After the call the token
                charge is corrected to the real usage.
  5. events     one summary per request on the events stream (analytics; Kafka-mirrorable).
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
from dataclasses import asdict

from relay.dense import build_retriever
from relay.limiter import INPUT_TOKENS, OUTPUT_TOKENS, REQUESTS, TOKENS, Limiter
from relay.llm import Endpoint, LLMClient
from relay.pipeline import FULL, FULL_OTHER, Pipeline, Reply, load_config, load_production_config
from relay.real import groq_api_key
from relay.retrieve import BLOCKS
from service import metrics as M
from service.settings import Settings
from service.store import make_store, new_id


class GlobalBudgetDenied(Exception):
    """The fleet-wide bucket has no room: the call is not sent (the pipeline moves to the next model)."""


def normalise(question: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", question.lower())).strip()


class Engine:
    def __init__(self, settings: Settings, store, pipeline: Pipeline, limiter: Limiter, version: str) -> None:
        self.s, self.store, self.pipe, self.lim, self.version = settings, store, pipeline, limiter, version

    @classmethod
    async def create(cls, settings: Settings | None = None, store=None, chat=None) -> "Engine":
        s = settings or Settings()
        store = store or make_store(s.redis_url)
        cfg = load_production_config() if s.config == "production" else load_config()
        lim = Limiter.from_probe()
        retriever = await asyncio.to_thread(build_retriever, cfg)       # loads/embeds the library once
        # warm the query encoder (and reranker) now: the first real request must not pay model loading
        await asyncio.to_thread(retriever.search, "warm up", 3)
        corpus = hashlib.sha256(BLOCKS.read_bytes()).hexdigest()[:12]
        conf = hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:12]
        eng = cls(s, store, None, lim, f"{corpus}-{conf}")
        if chat is None:
            key = groq_api_key() if s.is_groq else None
            client = LLMClient(Endpoint(s.llm_base_url, api_key=key, source="real" if s.is_groq else "stand-in"))
            chat = eng._gated(client)
        eng.pipe = Pipeline(chat, lim, retriever, cfg)
        return eng

    # ------------------------------------------------------------ fleet-wide rate limit
    def _limits(self, model: str) -> list[tuple[str, float, float, str]]:
        st = self.lim.models.get(model)
        if st is None:
            return []
        f = self.s.global_fraction
        return [(n, b.capacity * f, b.period_s, b.unit) for n, b in st.buckets.items()]

    def _gated(self, client: LLMClient):
        async def chat(payload, *, phase, timeouts=None, on_delta=None):
            model = payload["model"]
            prompt = self.lim.estimate(model, payload["messages"]).tokens
            hold = min(payload.get("max_tokens") or 0, 200)
            want = {REQUESTS: 1.0, TOKENS: float(prompt + hold), INPUT_TOKENS: float(prompt), OUTPUT_TOKENS: float(hold)}
            taken: list[tuple[str, float, float, float, str]] = []
            for name, cap, period, unit in self._limits(model):
                key, amount = f"relay:bucket:{model}:{name}", want.get(unit, 0.0)
                ok, wait = await self.store.take(key, cap, period, amount)
                if not ok:
                    for k, c, p, a, _ in taken:               # give back what this call already took
                        await self.store.take(k, c, p, -a, force=True)
                    M.GLOBAL_DENIED.labels(model, name).inc()
                    raise GlobalBudgetDenied(f"{model} {name}: fleet bucket short, ~{wait:.1f} s")
                taken.append((key, cap, period, amount, unit))
            rec = await client.stream_chat(payload, timeouts=timeouts, on_delta=on_delta)
            u = rec.usage or {}
            pt, ct = u.get("prompt_tokens") or 0, u.get("completion_tokens") or 0
            real = {REQUESTS: 1.0, TOKENS: float(pt + ct), INPUT_TOKENS: float(pt), OUTPUT_TOKENS: float(ct)}
            for k, c, p, a, unit in taken:
                if rec.status == 200 and u:
                    delta = real[unit] - a                       # correct the estimate to what was charged
                elif rec.status is not None and 400 <= rec.status < 500:
                    delta = -a                                   # rejected: Groq charges nothing (probe Q5)
                else:
                    delta = 0.0                                  # timeout / cut off: may have been admitted
                if delta:
                    await self.store.take(k, c, p, delta, force=True)
            M.MODEL_CALLS.labels(model, phase, str(rec.status)).inc()
            if rec.status == 200:
                M.TOKENS.labels(model, "in").inc(pt)
                M.TOKENS.labels(model, "out").inc(ct)
            return rec
        return chat

    # ------------------------------------------------------------ requests
    def cache_key(self, question: str) -> str:
        return "relay:answer:" + hashlib.sha256(f"{self.version}|{normalise(question)}".encode()).hexdigest()[:32]

    async def ask(self, question: str, on_claim=None, request_id: str | None = None,
                  patience_s: float | None = None) -> tuple[dict, dict]:
        """(reply dict, meta). meta["cache"]: hit | shared (waited for an identical in-flight request) | miss."""
        rid = request_id or new_id()
        t0 = time.perf_counter()
        key = self.cache_key(question)
        cached = await self.store.get(key)
        if cached is None:
            # single-flight: one request computes; identical ones arriving meanwhile wait for its result
            if not await self.store.set(key + ":lock", rid, ttl_s=self.s.singleflight_s + 5, nx=True):
                deadline = time.perf_counter() + self.s.singleflight_s
                while time.perf_counter() < deadline:
                    await asyncio.sleep(0.1)
                    cached = await self.store.get(key)
                    if cached is not None or await self.store.get(key + ":lock") is None:
                        break
                if cached is not None:
                    return await self._from_cache(cached, rid, t0, on_claim, "shared")
            else:
                try:
                    return await self._compute(question, key, rid, t0, on_claim, patience_s)
                finally:
                    await self.store.delete(key + ":lock")
            return await self._compute(question, key, rid, t0, on_claim, patience_s)
        return await self._from_cache(cached, rid, t0, on_claim, "hit")

    async def _from_cache(self, cached: str, rid: str, t0: float, on_claim, how: str) -> tuple[dict, dict]:
        d = json.loads(cached)
        d.update(request_id=rid, model_calls=[], spans=[], budget_wait_ms=0.0)
        ms = round((time.perf_counter() - t0) * 1000, 1)
        d["released_ms"] = d["first_token_ms"] = ms
        if on_claim is not None:
            for c in d.get("claims") or []:
                on_claim(c)
        self._observe(d, how)
        await self.store.emit({"request_id": rid, "cache": how, "kind": d["kind"], "level": d["level"], "ms": ms,
                               "tokens": 0, "t": time.time()})
        return d, {"cache": how}

    async def _compute(self, question, key, rid, t0, on_claim, patience_s) -> tuple[dict, dict]:
        M.INFLIGHT.inc()
        try:
            reply: Reply = await self.pipe.answer(question, request_id=rid, patience_s=patience_s, on_claim=on_claim)
        finally:
            M.INFLIGHT.dec()
        d = reply.to_dict()
        d["spans"] = [asdict(s) for s in reply.spans]
        d["model_calls"] = [asdict(c) for c in reply.model_calls]
        if reply.kind in ("answer", "refusal") and reply.level <= 2:
            ttl = self.s.cache_ttl_s if reply.level in (FULL, FULL_OTHER) else self.s.degraded_ttl_s
            if reply.level == FULL_OTHER:
                ttl = self.s.degraded_ttl_s
            await self.store.set(key, json.dumps({k: v for k, v in d.items() if k not in ("spans", "model_calls")},
                                                 default=str), ttl_s=ttl)
        M.DROPPED_CLAIMS.inc(len(reply.checks.get("dropped") or []))
        self._observe(d, "miss")
        await self.store.emit({"request_id": rid, "cache": "miss", "kind": reply.kind, "level": reply.level,
                               "first_token_ms": reply.first_token_ms, "ms": reply.released_ms,
                               "tokens": reply.tokens_in + reply.tokens_out, "t": time.time()})
        return d, {"cache": "miss"}

    @staticmethod
    def _observe(d: dict, cache: str) -> None:
        M.REQUESTS.labels(d["kind"], d.get("level_name") or str(d["level"]), cache).inc()
        if d.get("first_token_ms") is not None:
            M.FIRST_TOKEN.observe(d["first_token_ms"] / 1000)
        if d.get("released_ms") is not None:
            M.TOTAL.observe(d["released_ms"] / 1000)
