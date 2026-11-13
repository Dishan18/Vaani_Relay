"""HTTP API.

    uvicorn service.app:app --host 0.0.0.0 --port 8000

  POST /v1/ask            {"question": "...", "patience_s": 30}  -> the whole reply (JSON)
  POST /v1/ask/stream     same body -> text/event-stream:
                            event: claim   one verified sentence with its sources and quote, as soon as
                                           it passes its check (time to first token = first claim)
                            event: done    the final reply (kind, level, notice, sources, timings)
  POST /v1/jobs           same body -> {"job_id"}; a worker answers it (service/worker.py)
  GET  /v1/jobs/{job_id}  {"status": queued | running | done | failed, "result": ...}
  GET  /healthz           process is up
  GET  /readyz            store reachable and the library index loaded
  GET  /metrics           Prometheus

Client auth: "Authorization: Bearer <key>" with RELAY_API_KEYS set; each key has its own token bucket
(RELAY_CLIENT_RPM), so one noisy client can't spend the provider budget everyone shares.
"""
from __future__ import annotations

import asyncio
import json
import time
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field

from service import metrics as M
from service.engine import Engine
from service.settings import Settings
from service.store import new_id


class Ask(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    patience_s: float | None = Field(default=None, gt=0, le=120)


def create_app(settings: Settings | None = None, engine: Engine | None = None) -> FastAPI:
    s = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.engine = engine or await Engine.create(s)
        yield
        await app.state.engine.store.close()

    app = FastAPI(title="The Relay", version="2.0", lifespan=lifespan)

    async def client(request: Request, authorization: str | None = Header(default=None)) -> str:
        key = "anonymous"
        if s.api_keys:
            if not authorization or not authorization.startswith("Bearer "):
                raise HTTPException(401, "missing bearer token")
            key = authorization.removeprefix("Bearer ").strip()
            if key not in s.api_keys:
                raise HTTPException(403, "unknown key")
        eng: Engine = request.app.state.engine
        ok, wait = await eng.store.take(f"relay:client:{key}", s.client_rpm, 60.0, 1.0)
        if not ok:
            M.CLIENT_LIMITED.inc()
            raise HTTPException(429, "client rate limit", headers={"Retry-After": str(max(1, round(wait)))})
        return key

    @app.post("/v1/ask")
    async def ask(body: Ask, request: Request, _: str = Depends(client)):
        reply, meta = await request.app.state.engine.ask(body.question, patience_s=body.patience_s)
        return JSONResponse({**reply, "cache": meta["cache"]})

    @app.post("/v1/ask/stream")
    async def ask_stream(body: Ask, request: Request, _: str = Depends(client)):
        eng: Engine = request.app.state.engine
        q: asyncio.Queue = asyncio.Queue()

        def on_claim(c):
            q.put_nowait(("claim", {k: v for k, v in c.items() if k != "sid_cites"}))

        async def run():
            try:
                reply, meta = await eng.ask(body.question, on_claim=on_claim, patience_s=body.patience_s)
                q.put_nowait(("done", {**{k: v for k, v in reply.items() if k not in ("spans", "model_calls")},
                                       "cache": meta["cache"]}))
            except Exception as e:                       # never leave the stream hanging
                q.put_nowait(("error", {"error": type(e).__name__}))

        task = asyncio.create_task(run())

        async def events():
            try:
                while True:
                    kind, data = await q.get()
                    yield f"event: {kind}\ndata: {json.dumps(data, default=str)}\n\n"
                    if kind in ("done", "error"):
                        break
            finally:
                if not task.done():                      # the client left: stop spending on it
                    task.cancel()
        return StreamingResponse(events(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.post("/v1/jobs", status_code=202)
    async def submit(body: Ask, request: Request, _: str = Depends(client)):
        eng: Engine = request.app.state.engine
        jid = new_id()
        await eng.store.set(f"relay:job:{jid}", json.dumps({"status": "queued", "t": time.time()}), ttl_s=86400)
        await eng.store.enqueue({"job_id": jid, "question": body.question, "patience_s": body.patience_s})
        M.JOBS.labels("queued").inc()
        return {"job_id": jid}

    @app.get("/v1/jobs/{job_id}")
    async def job(job_id: str, request: Request):
        v = await request.app.state.engine.store.get(f"relay:job:{job_id}")
        if v is None:
            raise HTTPException(404, "no such job")
        return JSONResponse(json.loads(v))

    @app.get("/healthz")
    async def healthz():
        return {"ok": True}

    @app.get("/readyz")
    async def readyz(request: Request):
        eng: Engine | None = getattr(request.app.state, "engine", None)
        try:
            ok = eng is not None and await eng.store.ping()
        except Exception:
            ok = False
        return JSONResponse({"ok": ok, "version": eng.version if eng else None}, status_code=200 if ok else 503)

    @app.get("/metrics")
    async def metrics():
        return Response(generate_latest(M.REGISTRY), media_type=CONTENT_TYPE_LATEST)

    return app


app = create_app()
