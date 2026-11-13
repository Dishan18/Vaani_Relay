"""Shared state for the service: answer cache, single-flight locks, token buckets, job queue, events.

Two implementations with one interface:
  RedisStore   production. Several API replicas and workers share it, so the cache, the in-flight
               dedup, the provider's rate limits and the job queue are global, not per process.
  MemoryStore  same semantics in one process: tests, and `uvicorn service.app:app` without Redis.

Why Redis and not Kafka: the service needs a cache with TTLs, atomic counters (token buckets),
short-lived locks and a small durable work queue with consumer groups and redelivery. Redis does all
five with one dependency (Streams give consumer groups, acks and XAUTOCLAIM redelivery). Kafka is the
right tool when many independent consumers replay a high-volume event log for days (analytics,
training data); the `events` stream here is shaped so it can be mirrored to a Kafka topic when that
need appears, without changing producers.

Token buckets (``take``): capacity C refilling at C / period per second, the same model the probe
measured on Groq (data/probe/findings.md Q1). In Redis it runs as one Lua script, so concurrent
replicas can't both spend the last tokens; time comes from Redis's own clock, not the callers'.
"""
from __future__ import annotations

import asyncio
import json
import time
import uuid

TAKE_LUA = """
local key = KEYS[1]
local cap = tonumber(ARGV[1])
local rate = tonumber(ARGV[2])
local amt = tonumber(ARGV[3])
local force = ARGV[4]
local t = redis.call('TIME')
local now = tonumber(t[1]) + tonumber(t[2]) / 1000000
local v = redis.call('HMGET', key, 'level', 'ts')
local level = tonumber(v[1]) or cap
local ts = tonumber(v[2]) or now
level = math.min(cap, level + math.max(0, now - ts) * rate)
local ok = 0
if amt <= level or force == '1' then
  level = level - amt
  ok = 1
end
redis.call('HSET', key, 'level', tostring(level), 'ts', tostring(now))
redis.call('EXPIRE', key, math.ceil(cap / rate) + 60)
if ok == 1 then return {1, tostring(level)} end
return {0, tostring((amt - level) / rate)}
"""


class MemoryStore:
    def __init__(self) -> None:
        self.kv: dict[str, tuple[float | None, str]] = {}
        self.buckets: dict[str, tuple[float, float]] = {}
        self.jobs: asyncio.Queue | None = None
        self.events: list[dict] = []

    async def ping(self) -> bool:
        return True

    # ---- key/value with TTL (cache, job records)
    async def get(self, key: str) -> str | None:
        v = self.kv.get(key)
        if v is None:
            return None
        exp, val = v
        if exp is not None and exp < time.monotonic():
            self.kv.pop(key, None)
            return None
        return val

    async def set(self, key: str, value: str, ttl_s: float | None = None, nx: bool = False) -> bool:
        if nx and await self.get(key) is not None:
            return False
        self.kv[key] = (time.monotonic() + ttl_s if ttl_s else None, value)
        return True

    async def delete(self, key: str) -> None:
        self.kv.pop(key, None)

    # ---- token bucket
    async def take(self, key: str, capacity: float, period_s: float, amount: float,
                   force: bool = False) -> tuple[bool, float]:
        """(True, level left) or (False, seconds until it would fit). force: charge even if short
        (corrections after a call, which may push the level below zero)."""
        rate = capacity / period_s
        now = time.monotonic()
        level, ts = self.buckets.get(key, (capacity, now))
        level = min(capacity, level + max(0.0, now - ts) * rate)
        if amount <= level or force:
            self.buckets[key] = (level - amount, now)
            return True, level - amount
        self.buckets[key] = (level, now)
        return False, (amount - level) / rate

    # ---- job queue
    def _q(self) -> asyncio.Queue:
        if self.jobs is None:
            self.jobs = asyncio.Queue()
        return self.jobs

    async def enqueue(self, job: dict) -> str:
        await self._q().put(job)
        return job["job_id"]

    async def next_jobs(self, consumer: str, block_ms: int = 1000, count: int = 1) -> list[tuple[str, dict]]:
        try:
            job = await asyncio.wait_for(self._q().get(), block_ms / 1000)
        except asyncio.TimeoutError:
            return []
        return [(job["job_id"], job)]

    async def ack(self, msg_id: str) -> None:
        return None

    async def reclaim(self, consumer: str, min_idle_ms: int = 60000) -> list[tuple[str, dict]]:
        return []

    # ---- events (analytics log)
    async def emit(self, event: dict) -> None:
        self.events.append(event)
        del self.events[:-10000]

    async def close(self) -> None:
        return None


class RedisStore(MemoryStore):
    STREAM, GROUP, EVENTS = "relay:jobs", "relay-workers", "relay:events"

    def __init__(self, url: str) -> None:
        import redis.asyncio as aioredis
        super().__init__()
        # socket_timeout must outlast XREADGROUP's block (1 s), or an idle worker's wait reads as a failure
        self.r = aioredis.from_url(url, decode_responses=True, socket_timeout=30, socket_connect_timeout=5,
                                   health_check_interval=15, retry_on_timeout=True)
        self._take = self.r.register_script(TAKE_LUA)
        self._group_ready = False

    async def ping(self) -> bool:
        return bool(await self.r.ping())

    async def get(self, key: str) -> str | None:
        return await self.r.get(key)

    async def set(self, key: str, value: str, ttl_s: float | None = None, nx: bool = False) -> bool:
        return bool(await self.r.set(key, value, px=int(ttl_s * 1000) if ttl_s else None, nx=nx))

    async def delete(self, key: str) -> None:
        await self.r.delete(key)

    async def take(self, key: str, capacity: float, period_s: float, amount: float,
                   force: bool = False) -> tuple[bool, float]:
        ok, val = await self._take(keys=[key], args=[capacity, capacity / period_s, amount, "1" if force else "0"])
        return bool(int(ok)), float(val)

    async def _group(self) -> None:
        if self._group_ready:
            return
        try:
            await self.r.xgroup_create(self.STREAM, self.GROUP, id="0", mkstream=True)
        except Exception as e:                       # BUSYGROUP: it already exists
            if "BUSYGROUP" not in str(e):
                raise
        self._group_ready = True

    async def enqueue(self, job: dict) -> str:
        await self._group()
        await self.r.xadd(self.STREAM, {"job": json.dumps(job)}, maxlen=100000, approximate=True)
        return job["job_id"]

    async def next_jobs(self, consumer: str, block_ms: int = 1000, count: int = 1) -> list[tuple[str, dict]]:
        await self._group()
        res = await self.r.xreadgroup(self.GROUP, consumer, {self.STREAM: ">"}, count=count, block=block_ms)
        return [(mid, json.loads(f["job"])) for _, msgs in (res or []) for mid, f in msgs]

    async def ack(self, msg_id: str) -> None:
        await self.r.xack(self.STREAM, self.GROUP, msg_id)

    async def reclaim(self, consumer: str, min_idle_ms: int = 60000) -> list[tuple[str, dict]]:
        """Jobs a crashed worker took but never acked, after min_idle_ms: redelivered to this one."""
        await self._group()
        res = await self.r.xautoclaim(self.STREAM, self.GROUP, consumer, min_idle_ms, "0-0", count=10)
        msgs = res[1] if res else []
        return [(mid, json.loads(f["job"])) for mid, f in msgs if f]

    async def emit(self, event: dict) -> None:
        await self.r.xadd(self.EVENTS, {"e": json.dumps(event, default=str)}, maxlen=100000, approximate=True)

    async def close(self) -> None:
        await self.r.aclose()


def make_store(url: str | None):
    return RedisStore(url) if url else MemoryStore()


def new_id() -> str:
    return uuid.uuid4().hex[:16]
