"""Offline tests for standin/server.py through the real client (relay.llm) and limiter parser."""
from __future__ import annotations

import asyncio
import copy
import json
import math
import random
import time
import unittest
import urllib.request

from relay.limiter import parse_limit_body
from relay.llm import Endpoint, LLMClient
from standin.server import Replay, StandIn, Switches, fmt_duration, load_calibration

CAL = load_calibration()
M20, M120, MQ = "openai/gpt-oss-20b", "openai/gpt-oss-120b", "qwen/qwen3.8-27b"


def fast_cal():
    c = copy.deepcopy(CAL)
    for m in c["models"].values():
        m["ttfb_ms"].update(mu=math.log(5.0), sigma=0.01)
        m["decode"]["tokens_per_s"] = 20000.0
    return c


def payload(model=M20, text="Say OK.", max_tokens=50):
    return {"model": model, "messages": [{"role": "user", "content": text}], "max_tokens": max_tokens,
            "reasoning_effort": "low" if model != MQ else "none"}


async def call(s: StandIn, p: dict):
    return await LLMClient(Endpoint(s.base_url, source="stand-in")).stream_chat(p)


def run(coro):
    return asyncio.run(coro)


class StreamTests(unittest.TestCase):
    def test_200_stream_headers_usage_reasoning(self):
        async def go():
            async with StandIn(fast_cal(), seed=1) as s:
                return await call(s, payload(max_tokens=400))
        r = run(go())
        self.assertEqual((r.status, r.outcome), (200, "ok"))
        names = [h[0] for h in r.rate_limit_headers]
        for h in ("x-ratelimit-limit-requests", "x-ratelimit-limit-tokens", "x-ratelimit-remaining-requests",
                  "x-ratelimit-remaining-tokens", "x-ratelimit-reset-requests", "x-ratelimit-reset-tokens"):
            self.assertIn(h, names)
        self.assertEqual(r.header("x-ratelimit-limit-tokens"), "8000")
        self.assertEqual(r.header("x-ratelimit-remaining-requests"), "999")
        u = r.usage
        self.assertGreater(u["prompt_tokens"], 0)
        self.assertGreaterEqual(u["completion_tokens"], u.get("completion_tokens_details", {}).get("reasoning_tokens", 0))
        self.assertIn("queue_time", u)
        self.assertTrue(r.text)

    def test_decode_rate_matches_calibration(self):
        c = fast_cal()
        c["models"][MQ]["decode"]["tokens_per_s"] = 500.0

        class R(Replay):
            def lookup(self, messages):
                return "word " * 450                 # ~620 tokens at 3.62 chars/token (~1.2 s)
        async def go():
            async with StandIn(c, replay=R(None), seed=2) as s:
                return await call(s, payload(MQ, max_tokens=2000))
        r = run(go())
        n = r.usage["completion_tokens"]
        dur = (r.total_ms - r.first_visible_ms) / 1000
        self.assertAlmostEqual(n / dur, 500.0, delta=150.0)

    def test_ttfb_lognormal_matches_p50_p95(self):
        s = StandIn(CAL, seed=3)
        for m in (M20, M120, MQ):
            xs = sorted(s.ttfb_s(s.models[m]) * 1000 for _ in range(4000))
            t = CAL["models"][m]["ttfb_ms"]
            self.assertAlmostEqual(xs[len(xs) // 2] / t["p50_target"], 1.0, delta=0.08, msg=m)
            self.assertAlmostEqual(xs[int(len(xs) * 0.95)] / t["p95_target"], 1.0, delta=0.15, msg=m)


class LimitTests(unittest.TestCase):
    def test_rpm_429_body_retry_after_and_not_counted(self):
        async def go():
            async with StandIn(fast_cal(), Switches(limit_scale=0.1, limit_scale_only="RPM"), seed=4) as s:
                ok = [await call(s, payload()) for _ in range(3)]          # RPM 30 * 0.1 = 3
                bad = await call(s, payload())
                bad2 = await call(s, payload())
                return ok, bad, bad2
        ok, bad, bad2 = run(go())
        self.assertTrue(all(r.status == 200 for r in ok))
        self.assertEqual(bad.status, 429)
        info = parse_limit_body(bad.error_body)
        self.assertEqual((info["short"], info["limit"], info["used"], info["requested"]), ("RPM", 3, 3, 1))
        self.assertEqual(int(bad.header("retry-after")), math.ceil(info["wait_s"]))
        self.assertAlmostEqual(info["wait_s"], 20.0, delta=0.5)              # one request refills in 60/3 s
        self.assertIsNone(bad.header("x-ratelimit-remaining-tokens"))       # request 429: request headers only
        self.assertEqual(bad.header("x-ratelimit-remaining-requests"), bad2.header("x-ratelimit-remaining-requests"))

    def test_tpm_429_names_requested_and_hold(self):
        async def go():
            async with StandIn(fast_cal(), seed=5) as s:
                s.models[M20].buckets["TPM"].level = 100.0
                return await call(s, payload(max_tokens=500)), s
        r, s = run(go())
        info = parse_limit_body(r.error_body)
        self.assertEqual(info["short"], "TPM")
        pt = s.models[M20].prompt_tokens(payload()["messages"])
        self.assertEqual(info["requested"], pt + 200)                       # prompt + min(max_tokens, hold_cap)
        self.assertEqual(int(r.header("retry-after")), math.ceil(info["wait_s"]))

    def test_hold_released_to_actual_usage(self):
        async def go():
            async with StandIn(fast_cal(), seed=6) as s:
                b = s.models[M20].buckets["TPM"]
                before = b.sync()
                r = await call(s, payload(max_tokens=1000))
                return before - b.sync(), r
        used, r = run(go())
        self.assertAlmostEqual(used, r.usage["prompt_tokens"] + r.usage["completion_tokens"], delta=3.0)

    def test_halved_limits_show_in_headers(self):
        async def go():
            async with StandIn(fast_cal(), Switches(limit_scale=0.5), seed=7) as s:
                return await call(s, payload())
        r = run(go())
        self.assertEqual(r.header("x-ratelimit-limit-tokens"), "4000")
        self.assertEqual(r.header("x-ratelimit-limit-requests"), "500")

    def test_exhaust_daily_then_refill_rate(self):
        async def go():
            async with StandIn(fast_cal(), seed=8) as s:
                s.control({"exhaust_daily": M120})
                bad = await call(s, payload(M120))
                other = await call(s, payload(M20))
                return bad, other
        bad, other = run(go())
        info = parse_limit_body(bad.error_body)
        self.assertIn(info["short"], ("RPD", "TPD"))
        self.assertGreater(info["wait_s"], 60)
        self.assertEqual(other.status, 200)                                 # limits are per model

    def test_413_when_it_can_never_fit(self):
        async def go():
            async with StandIn(fast_cal(), seed=9) as s:
                return await call(s, payload(text="x " * 40000))
        self.assertEqual(run(go()).status, 413)


class SwitchTests(unittest.TestCase):
    def test_503_and_stall_and_runtime_control(self):
        async def go():
            async with StandIn(fast_cal(), Switches(p503=1.0), seed=10) as s:
                a = await call(s, payload())
                url = f"http://127.0.0.1:{s.port}/control"
                body = json.dumps({"set": {"p503": 0, "stall_rate": 1, "stall_s": 0.4}}).encode()

                def post():
                    rq = urllib.request.Request(url, data=body, method="POST", headers={"content-type": "application/json"})
                    return json.loads(urllib.request.urlopen(rq, timeout=5).read())
                st = await asyncio.to_thread(post)
                b = await call(s, payload())
                return a, st, b
        a, st, b = run(go())
        self.assertEqual(a.status, 503)
        self.assertEqual(st["switches"]["stall_rate"], 1.0)
        self.assertEqual(b.status, 200)
        self.assertGreaterEqual(b.headers_ms, 400 - 20)   # Windows timers can fire up to one ~15.6 ms tick early

    def test_slow_under_load(self):
        s = StandIn(CAL, Switches(slow_under_load=True, load_alpha=0.5), seed=11)
        s.inflight = 5
        self.assertAlmostEqual(s.load_factor(), 3.0)
        s.sw.slow_under_load = False
        self.assertEqual(s.load_factor(), 1.0)


class ReplayTests(unittest.TestCase):
    def test_replay_hit_and_miss(self):
        rep = Replay(None)
        q = "Why divide by sqrt(d_k)?"
        rep.router[q] = '{"papers": ["attention"], "query": "scaled", "alt_query": "softmax"}'
        rep.draft[(q, ("1706.03762v7 p.4 §3.2.1",))] = "Real answer text [S1]."
        user = f"Question: {q}\n\nSOURCES:\n[S1] (Attention Is All You Need; 1706.03762v7 p.4 §3.2.1)\nSome text."
        draft = [{"role": "system", "content": "Answer the question"}, {"role": "user", "content": user}]
        self.assertEqual(rep.lookup(draft), "Real answer text [S1].")
        self.assertEqual(rep.lookup([{"role": "system", "content": "You route questions"},
                                     {"role": "user", "content": q}]), rep.router[q])
        self.assertIsNone(rep.lookup([{"role": "system", "content": "Answer"},
                                      {"role": "user", "content": user.replace("p.4", "p.5")}]))

        async def go():
            async with StandIn(fast_cal(), replay=rep, seed=12) as s:
                return await call(s, {**payload(M120, max_tokens=800), "messages": draft}), s.stats
        r, stats = run(go())
        self.assertEqual(r.text, "Real answer text [S1].")
        self.assertEqual((stats["replay_hits"], stats["replay_misses"]), (1, 0))
        self.assertEqual(r.other_headers.get("x-standin-replay"), "hit")

    def test_duration_format(self):
        self.assertEqual([fmt_duration(x) for x in (2, 2.992, 86.4, 4665.6, 0.45)],
                         ["2s", "2.992s", "1m26.4s", "1h17m45.6s", "450ms"])


if __name__ == "__main__":
    unittest.main()
