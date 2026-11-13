"""The service (service/): API, cache, single-flight, streaming, auth, client quotas, jobs, fleet-wide gate.
In-memory store and a scripted model: no Redis, no key, no network."""
from __future__ import annotations

import asyncio
import json
import unittest

from relay import dense
from relay.llm import CallRecord
from tests.test_claims import StreamingChat, claim_line, first_sentence, script_with

try:
    from fastapi.testclient import TestClient
    HAVE_SERVICE = dense.available()
except ImportError:
    HAVE_SERVICE = False


def draft(user):
    f = first_sentence(user)
    return claim_line(f, 1, " ".join(f.split()[:8])) + "\n"


def make_engine(chat=None, **settings):
    from service.engine import Engine
    from service.settings import Settings
    from service.store import MemoryStore
    s = Settings(**{"redis_url": None, "llm_base_url": "http://scripted", "api_keys": frozenset({"k1"}),
                    "client_rpm": 1000.0, **settings})
    chat = chat or StreamingChat(script_with(draft))
    return asyncio.run(Engine.create(s, store=MemoryStore(), chat=chat)), chat, s


Q = "Why divide the query-key dot products by sqrt(d_k) in scaled dot-product attention?"
AUTH = {"Authorization": "Bearer k1"}


@unittest.skipUnless(HAVE_SERVICE, "pip install -r requirements-service.txt")
class ServiceTests(unittest.TestCase):
    def client(self, **kw):
        from service.app import create_app
        eng, chat, s = make_engine(**kw)
        return TestClient(create_app(s, eng)), eng, chat

    def test_ask_then_cache_hit(self):
        c, eng, chat = self.client()
        with c:
            r1 = c.post("/v1/ask", json={"question": Q}, headers=AUTH).json()
            n = len(chat.calls)
            r2 = c.post("/v1/ask", json={"question": Q.upper() + " "}, headers=AUTH).json()  # same, normalised
        self.assertEqual(r1["kind"], "answer")
        self.assertEqual(r1["cache"], "miss")
        self.assertTrue(r1["claims"] and r1["claims"][0]["quote"])
        self.assertEqual(r2["cache"], "hit")
        self.assertEqual(len(chat.calls), n)                 # no model call for the repeat
        self.assertEqual(r2["text"], r1["text"])

    def test_single_flight_shares_one_answer(self):
        eng, chat, _ = make_engine()

        async def both():
            return await asyncio.gather(eng.ask(Q), eng.ask(Q), eng.ask(Q))
        out = asyncio.run(both())
        caches = sorted(m["cache"] for _, m in out)
        self.assertEqual(caches, ["miss", "shared", "shared"])
        self.assertEqual(sum(1 for step, _ in chat.calls if step == "draft"), 1)

    def test_stream_sends_claims_then_done(self):
        c, _, _ = self.client()
        with c:
            with c.stream("POST", "/v1/ask/stream", json={"question": Q}, headers=AUTH) as r:
                body = "".join(r.iter_text())
        events = [ln.split(": ", 1)[1] for ln in body.splitlines() if ln.startswith("event: ")]
        self.assertEqual(events[0], "claim")
        self.assertEqual(events[-1], "done")
        done = json.loads(body.strip().split("data: ")[-1])
        self.assertEqual(done["kind"], "answer")

    def test_auth(self):
        c, _, _ = self.client()
        with c:
            self.assertEqual(c.post("/v1/ask", json={"question": Q}).status_code, 401)
            self.assertEqual(c.post("/v1/ask", json={"question": Q}, headers={"Authorization": "Bearer nope"}).status_code, 403)

    def test_client_quota(self):
        c, _, _ = self.client(client_rpm=2.0)
        with c:
            codes = [c.post("/v1/ask", json={"question": Q}, headers=AUTH).status_code for _ in range(3)]
        self.assertEqual(codes[:2], [200, 200])
        self.assertEqual(codes[2], 429)

    def test_jobs_roundtrip(self):
        from service.worker import handle
        c, eng, _ = self.client()
        with c:
            jid = c.post("/v1/jobs", json={"question": Q}, headers=AUTH).json()["job_id"]
            self.assertEqual(c.get(f"/v1/jobs/{jid}").json()["status"], "queued")
            msgs = c.portal.call(eng.store.next_jobs, "test", 100, 1)
            c.portal.call(handle, eng, *msgs[0])
            j = c.get(f"/v1/jobs/{jid}").json()
        self.assertEqual(j["status"], "done")
        self.assertEqual(j["result"]["kind"], "answer")

    def test_health_ready_metrics(self):
        c, _, _ = self.client()
        with c:
            self.assertEqual(c.get("/healthz").status_code, 200)
            self.assertTrue(c.get("/readyz").json()["ok"])
            c.post("/v1/ask", json={"question": Q}, headers=AUTH)
            self.assertIn("relay_requests_total", c.get("/metrics").text)

    def test_fleet_gate_denies_then_refunds(self):
        """With the shared bucket empty, the call is not sent and the pipeline moves to another model."""
        from service.store import MemoryStore
        eng, _, _ = make_engine()
        sent = []

        class Client:
            async def stream_chat(self, payload, timeouts=None, on_delta=None):
                sent.append(payload["model"])
                r = CallRecord(call_id="c", source="t", url="u", model=payload["model"], started_at="t",
                               status=200, outcome="ok")
                r.text, r.usage = "ok", {"prompt_tokens": 10, "completion_tokens": 2}
                return r
        gated = eng._gated(Client())
        store: MemoryStore = eng.store
        model = "openai/gpt-oss-120b"
        for name, cap, period, unit in eng._limits(model):
            if name == "TPM":
                asyncio.run(store.take(f"relay:bucket:{model}:{name}", cap, period, cap))   # empty it
        payload = {"model": model, "messages": [{"role": "user", "content": "x" * 400}], "max_tokens": 300}
        from service.engine import GlobalBudgetDenied
        with self.assertRaises(GlobalBudgetDenied):
            asyncio.run(gated(payload, phase="draft"))
        self.assertEqual(sent, [])
        for name, cap, *_ in eng._limits(model):              # whatever it took before TPM was given back
            key = f"relay:bucket:{model}:{name}"
            if name != "TPM" and key in store.buckets:
                self.assertAlmostEqual(store.buckets[key][0], cap, places=3)


if __name__ == "__main__":
    unittest.main()
