"""Background worker: answers queued jobs (POST /v1/jobs) from the Redis stream.

    python -m service.worker [--concurrency 4]

At-least-once: a job is acked only after its result is stored. A worker that dies mid-job leaves it
pending; another worker reclaims it after 60 s idle (XAUTOCLAIM) and answers it again. Answering twice
is safe: the answer cache and single-flight make the repeat cheap, and the result overwrites in place.
Jobs share the same engine, cache and fleet-wide rate limits as the API, so batch work can't starve
interactive users of budget beyond their shared buckets.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import socket
import time

from service import metrics as M
from service.engine import Engine
from service.settings import Settings

log = logging.getLogger("relay.worker")


async def handle(eng: Engine, msg_id: str, job: dict) -> None:
    key = f"relay:job:{job['job_id']}"
    await eng.store.set(key, json.dumps({"status": "running", "t": time.time()}), ttl_s=86400)
    try:
        reply, meta = await eng.ask(job["question"], request_id=job["job_id"], patience_s=job.get("patience_s"))
        result = {"status": "done", "t": time.time(), "cache": meta["cache"],
                  "result": {k: v for k, v in reply.items() if k not in ("spans",)}}
        M.JOBS.labels("done").inc()
    except Exception as e:
        log.exception("job %s failed", job["job_id"])
        result = {"status": "failed", "t": time.time(), "error": type(e).__name__}
        M.JOBS.labels("failed").inc()
    await eng.store.set(key, json.dumps(result, default=str), ttl_s=86400)
    await eng.store.ack(msg_id)


async def run(concurrency: int = 4, stop: asyncio.Event | None = None, engine: Engine | None = None) -> None:
    eng = engine or await Engine.create(Settings())
    me = f"{socket.gethostname()}-{os.getpid()}"
    sem = asyncio.Semaphore(concurrency)
    stop = stop or asyncio.Event()
    tasks: set[asyncio.Task] = set()
    last_reclaim = 0.0
    log.info("worker %s ready", me)
    backoff = 0.5
    while not stop.is_set():
        batch = []
        try:
            if time.monotonic() - last_reclaim > 30:
                batch += await eng.store.reclaim(me)
                last_reclaim = time.monotonic()
            batch += await eng.store.next_jobs(me, block_ms=1000, count=concurrency)
            backoff = 0.5
        except Exception as e:             # Redis restarting or a network blip: back off, don't die
            log.warning("store unavailable (%s); retrying in %.1f s", type(e).__name__, backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 15.0)
            continue
        for msg_id, job in batch:
            await sem.acquire()
            t = asyncio.create_task(handle(eng, msg_id, job))
            t.add_done_callback(lambda _t: sem.release())
            tasks.add(t)
            t.add_done_callback(tasks.discard)
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--concurrency", type=int, default=int(os.environ.get("RELAY_WORKER_CONCURRENCY", "4")))
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format='{"t":"%(asctime)s","level":"%(levelname)s","msg":"%(message)s"}')
    asyncio.run(run(a.concurrency))


if __name__ == "__main__":
    main()
