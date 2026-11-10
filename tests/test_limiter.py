"""Offline tests for relay.limiter (no network, no Groq budget)."""
from __future__ import annotations

import asyncio
import json
import time
import unittest

from relay.limiter import (REQUESTS, TOKENS, Bucket, Limiter, ModelState, Prompt, PromptEstimator,
                           Unavailable, parse_limit_body)
from relay.llm import CallRecord

M = "m"


class Clock:
    def __init__(self, t: float = 1000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


def body(what: str, short: str, limit: int, used: int, requested: int, wait: str) -> str:
    return json.dumps({"error": {"message": (
        f"Rate limit reached for model `m` in organization `org_REDACTED` service tier `on_demand` on {what} "
        f"({short}): Limit {limit}, Used {used}, Requested {requested}. Please try again in {wait}. Need more"),
        "type": "x", "code": "rate_limit_exceeded"}})


def rec(status=200, headers_ms=100.0, usage=None, headers=(), error_body=None, outcome="ok", text="") -> CallRecord:
    r = CallRecord(call_id="c", source="fake", url="u", model=M, started_at="t", status=status, outcome=outcome)
    r.headers_ms = headers_ms
    r.usage = usage
    r.rate_limit_headers = [list(h) for h in headers]
    r.error_body = error_body
    r.text = text
    return r


def limiter(clock, rpm=(30, 60.0), tpm=(8000, 60.0), rpd=(1000, 86400.0), tpm_level=None, est=None) -> Limiter:
    t = clock()
    b = {}
    if rpm:
        b["RPM"] = Bucket("RPM", REQUESTS, rpm[0], rpm[1], t, rpm[0], "test")
    if tpm:
        b["TPM"] = Bucket("TPM", TOKENS, tpm[0], tpm[1], t, tpm[0] if tpm_level is None else tpm_level, "test")
    if rpd:
        b["RPD"] = Bucket("RPD", REQUESTS, rpd[0], rpd[1], t, rpd[0], "test")
    return Limiter({M: ModelState(M, b)}, est or PromptEstimator(), clock)


class ReserveTests(unittest.TestCase):
    def test_grants_now_then_predicts_rpm_wait(self):
        c = Clock()
        lim = limiter(c)
        for _ in range(30):
            g = lim.reserve(M, 50, 100, c() + 5)
            self.assertEqual(g.wait_s, 0.0)
        t = lim.reserve(M, 50, 100, c() + 10)          # 31st: one request refills in 60/30 = 2 s
        self.assertAlmostEqual(t.wait_s, 2.0, places=3)
        with self.assertRaises(Unavailable) as cm:     # a second one is queued behind it: 4 s > 3 s deadline
            lim.reserve(M, 50, 100, c() + 3)
        self.assertEqual(cm.exception.bucket, "RPM")
        self.assertAlmostEqual(cm.exception.wait_s, 4.0, places=3)

    def test_queue_in_front_is_counted(self):
        c = Clock()
        lim = limiter(c, rpm=(2, 60.0))
        lim.reserve(M, 10, 10, c() + 1)
        lim.reserve(M, 10, 10, c() + 1)
        a = lim.reserve(M, 10, 10, c() + 100)
        b = lim.reserve(M, 10, 10, c() + 100)
        self.assertAlmostEqual(a.wait_s, 30.0, places=3)
        self.assertAlmostEqual(b.wait_s, 60.0, places=3)
        self.assertEqual(b.ahead, 1)

    def test_request_that_can_never_fit(self):
        c = Clock()
        lim = limiter(c)
        with self.assertRaises(Unavailable) as cm:
            lim.reserve(M, 9000, 10, c() + 3600)
        self.assertEqual(cm.exception.bucket, "TPM")
        self.assertIsNone(cm.exception.wait_s)

    def test_admission_charges_prompt_only_and_completion_at_end(self):
        c = Clock()
        lim = limiter(c, rpm=None, rpd=None, tpm_level=1000)
        g = lim.reserve(M, 300, 2000, c() + 1)          # max_tokens 2000 is not held (Q2)
        self.assertAlmostEqual(lim.levels(M)["TPM"], 700, places=3)
        c.t += 1
        g.settle(rec(usage={"prompt_tokens": 200, "completion_tokens": 40}, headers_ms=None))
        # estimate 300 replaced by the real 200, then 40 completion tokens charged at the end
        self.assertAlmostEqual(lim.levels(M)["TPM"], 1000 - 200 + 8000 / 60 - 40, places=3)

    def test_unknown_model_is_unavailable(self):
        lim = limiter(Clock())
        with self.assertRaises(Unavailable):
            lim.reserve("other", 1, 1, 0)


class AdmissionHoldTests(unittest.TestCase):
    def test_capped_hold_reserved_then_released_down_to_usage(self):
        c = Clock()
        lim = limiter(c, rpm=None, rpd=None, tpm_level=4000)
        lim.models[M].hold_cap = 200.0
        g = lim.reserve(M, 300, 2000, c() + 1)           # prompt + min(2000, 200)
        self.assertEqual(g.hold, 200.0)
        self.assertAlmostEqual(lim.levels(M)["TPM"], 3500, places=3)
        g2 = lim.reserve(M, 300, 50, c() + 1)            # prompt + min(50, 200)
        self.assertEqual(g2.hold, 50.0)
        c.t += 1
        g.settle(rec(usage={"prompt_tokens": 280, "completion_tokens": 40}, headers_ms=None))
        # g: 280 prompt + 40 completion in the end; g2 still holds 300 + 50
        self.assertAlmostEqual(lim.levels(M)["TPM"], 4000 - 280 - 40 - 350 + 8000 / 60, places=3)

    def test_hold_cap_learned_from_a_call_alone_in_a_full_bucket(self):
        c = Clock(0.0)
        lim = limiter(c, rpm=None, rpd=None)
        lim.models[M].hold_cap = 200.0
        g = lim.reserve(M, 300, 500, 10)
        c.t = 1.0
        g.settle(rec(headers_ms=200.0, usage={"prompt_tokens": 290, "completion_tokens": 100, "queue_time": 0.1},
                     headers=[("x-ratelimit-remaining-tokens", str(8000 - 290 - 260))]))
        self.assertEqual(lim.models[M].hold_cap, 260.0)
        self.assertEqual(lim.findings_changed[-1]["change"], "admission hold cap raised")

    def test_no_learning_when_another_call_was_in_flight(self):
        c = Clock(0.0)
        lim = limiter(c, rpm=None, rpd=None)
        lim.models[M].hold_cap = 200.0
        g = lim.reserve(M, 300, 500, 10)
        lim.reserve(M, 300, 500, 10)                     # in flight alongside g
        c.t = 1.0
        g.settle(rec(headers_ms=200.0, usage={"prompt_tokens": 290, "completion_tokens": 100},
                     headers=[("x-ratelimit-remaining-tokens", "6000")]))
        self.assertEqual(lim.models[M].hold_cap, 200.0)


class HeaderTests(unittest.TestCase):
    def test_header_overwrites_and_counts_calls_in_flight(self):
        c = Clock(0.0)
        lim = limiter(c, rpm=None, rpd=None)
        rate = 8000 / 60
        g1 = lim.reserve(M, 100, 50, 10)                 # t = 0
        c.t = 1.0
        lim.reserve(M, 500, 50, 10)                      # t = 1, still in flight when g1's headers land
        c.t = 2.0
        # g1's headers arrived at t=0.5 after 0.3 s in Groq's queue: values are as of admission, t=0.2
        g1.settle(rec(headers_ms=500.0, usage={"prompt_tokens": 90, "completion_tokens": 50, "queue_time": 0.3},
                      headers=[("x-ratelimit-limit-tokens", "8000"), ("x-ratelimit-remaining-tokens", "7000")]))
        expect = 7000 + rate * 0.8 - 500 + rate * 1.0 - 50
        self.assertAlmostEqual(lim.levels(M)["TPM"], expect, places=3)

    def test_older_header_does_not_overwrite_newer(self):
        c = Clock(0.0)
        lim = limiter(c, rpm=None, rpd=None)
        g1 = lim.reserve(M, 10, 10, 10)
        g2 = lim.reserve(M, 10, 10, 10)
        c.t = 5.0
        g2.settle(rec(headers_ms=4000.0, usage={"prompt_tokens": 10, "completion_tokens": 0},
                      headers=[("x-ratelimit-remaining-tokens", "6000")]))
        before = lim.levels(M)["TPM"]
        g1.settle(rec(headers_ms=1000.0, usage={"prompt_tokens": 10, "completion_tokens": 0},
                      headers=[("x-ratelimit-remaining-tokens", "100")]))     # older (t=1): ignored
        self.assertAlmostEqual(lim.levels(M)["TPM"], before, places=3)

    def test_rpd_header_anchor(self):
        c = Clock(0.0)
        lim = limiter(c, rpm=None, tpm=None)
        g = lim.reserve(M, 10, 10, 10)
        g.settle(rec(headers_ms=0.0, usage={"prompt_tokens": 10, "completion_tokens": 1},
                     headers=[("x-ratelimit-limit-requests", "1000"), ("x-ratelimit-remaining-requests", "940")]))
        self.assertAlmostEqual(lim.levels(M)["RPD"], 940, places=3)


class Limit429Tests(unittest.TestCase):
    def test_429_releases_and_learns_rpm_only_seen_in_bodies(self):
        c = Clock(0.0)
        lim = limiter(c, rpm=None)                       # like qwen: the probe never hit its RPM
        g = lim.reserve(M, 100, 10, 10)
        g.settle(rec(status=429, outcome="rate_limited", headers_ms=0.0,
                     error_body=body("requests per minute", "RPM", 30, 30, 1, "2s"),
                     headers=[("retry-after", "2"), ("x-ratelimit-limit-requests", "1000"),
                              ("x-ratelimit-remaining-requests", "946")]))
        b = lim.models[M].buckets["RPM"]
        self.assertEqual((b.capacity, b.period_s), (30.0, 60.0))     # period = Limit x "try again in" = 60 s
        lv = lim.levels(M)
        self.assertAlmostEqual(lv["RPM"], 0.0, places=3)
        self.assertAlmostEqual(lv["RPD"], 946, places=3)             # the 429 itself was not counted
        self.assertAlmostEqual(lv["TPM"], 8000, places=3)            # its prompt charge was released
        self.assertEqual(lim.findings_changed[0]["change"], "new limit RPM learned from a 429 body")
        t = lim.reserve(M, 100, 10, 10)
        self.assertAlmostEqual(t.wait_s, 2.0, places=3)

    def test_new_limit_type_from_body(self):
        c = Clock(0.0)
        lim = limiter(c)
        g = lim.reserve(M, Prompt(500, 1500, 2), 100, 10)
        g.settle(rec(status=429, outcome="rate_limited", headers_ms=0.0,
                     error_body=body("tokens per day", "TPD", 200000, 199800, 520, "2m9.6s")))
        b = lim.models[M].buckets["TPD"]
        self.assertEqual((b.unit, b.capacity), (TOKENS, 200000.0))
        self.assertAlmostEqual(lim.levels(M)["TPD"], 200, places=3)
        with self.assertRaises(Unavailable) as cm:
            lim.reserve(M, 500, 100, 60)                 # 300 tokens short at 200000/86400 per s: ~130 s
        self.assertEqual(cm.exception.bucket, "TPD")

    def test_requested_calibrates_estimator(self):
        c = Clock(0.0)
        lim = limiter(c)
        g = lim.reserve(M, Prompt(500, 1500, 2), 200, 10)
        g.settle(rec(status=429, outcome="rate_limited", headers_ms=0.0,
                     error_body=body("tokens per minute", "TPM", 8000, 7900, 560, "3s")))
        self.assertFalse(lim.models[M].admission_includes_max_tokens)   # 560 is nearer 500 than 700
        self.assertGreaterEqual(lim.estimator.estimate(M, 1500, 2), 560)
        self.assertIn("429 Requested", lim.estimator.fits[M]["sources"])

    def test_requested_matching_max_tokens_overturns_q2(self):
        c = Clock(0.0)
        lim = limiter(c)
        g = lim.reserve(M, Prompt(500, 1500, 2), 1000, 10)
        g.settle(rec(status=429, outcome="rate_limited", headers_ms=0.0,
                     error_body=body("tokens per minute", "TPM", 8000, 5000, 1495, "8s")))
        st = lim.models[M]
        self.assertTrue(st.admission_includes_max_tokens)
        self.assertIn("Q2 overturned", lim.findings_changed[-1]["change"])
        before = lim.levels(M)["TPM"]
        lim.reserve(M, 100, 1000, 100)                   # now holds prompt + max_tokens
        self.assertAlmostEqual(before - lim.levels(M)["TPM"], 1100, places=3)

    def test_connect_error_releases_but_timeout_before_headers_keeps_charge(self):
        c = Clock(0.0)
        lim = limiter(c, rpm=(30, 60.0), tpm=None, rpd=None)
        g = lim.reserve(M, 10, 10, 10)
        g.settle(rec(status=None, outcome="connect_error", headers_ms=None))
        self.assertAlmostEqual(lim.levels(M)["RPM"], 30, places=3)
        g = lim.reserve(M, 10, 10, 10)
        g.settle(rec(status=None, outcome="timeout_first_visible", headers_ms=None))
        self.assertAlmostEqual(lim.levels(M)["RPM"], 29, places=3)

    def test_parse_too_large_body(self):
        msg = json.dumps({"error": {"message": "Request too large for model `m` in organization `org_REDACTED` "
                                               "service tier `on_demand` on tokens per minute (TPM): Limit 8000, "
                                               "Requested 8583, please reduce your message size"}})
        p = parse_limit_body(msg)
        self.assertEqual((p["short"], p["unit"], p["limit"], p["requested"], p["used"]), ("TPM", TOKENS, 8000, 8583, None))


class EstimatorTests(unittest.TestCase):
    def test_bound_covers_every_observation_and_adapts(self):
        e = PromptEstimator()
        for chars, msgs, tok in [(36, 1, 85), (470, 2, 205), (500, 2, 210), (4200, 2, 990), (5200, 2, 1280)]:
            e.observe(M, chars, msgs, tok)
        e.misses.clear()                                 # count misses after seeding, as from_probe does
        for chars, msgs, tok in [(36, 1, 85), (470, 2, 205), (500, 2, 210), (4200, 2, 990), (5200, 2, 1280)]:
            self.assertGreaterEqual(e.estimate(M, chars, msgs), tok)
        e.observe(M, 3000, 2, 1100)                      # denser text than anything seen: a miss
        self.assertEqual(e.misses[M], 1)
        self.assertGreaterEqual(e.estimate(M, 3000, 2), 1100)

    def test_default_without_data_is_conservative(self):
        e = PromptEstimator()
        self.assertEqual(e.estimate("x", 300, 2), 128 + 100 + 16)


class FromProbeTests(unittest.TestCase):
    def test_limits_and_estimator_come_from_the_probe(self):
        lim = Limiter.from_probe()
        g20 = lim.models["openai/gpt-oss-20b"].buckets
        self.assertEqual({n: (b.capacity, round(b.period_s)) for n, b in g20.items()},
                         {"TPM": (8000.0, 60), "RPD": (1000.0, 86400), "RPM": (30.0, 60)})
        self.assertNotIn("RPM", lim.models["qwen/qwen3.8-27b"].buckets)   # never hit by the probe
        for st in lim.models.values():
            self.assertFalse(st.admission_includes_max_tokens)
            self.assertFalse(st.count_429)
            self.assertIsNotNone(st.hold_cap)                         # measured capped hold
            self.assertLessEqual(st.hold_cap, 500)
        for m, pts in lim.estimator.points.items():
            for chars, msgs, tok, _ in pts:
                self.assertGreaterEqual(lim.estimator.estimate(m, chars, msgs), tok)


class AsyncTests(unittest.IsolatedAsyncioTestCase):
    async def test_ticket_acquires_when_room_refills(self):
        lim = limiter(time.monotonic, rpm=(1, 0.2), tpm=None, rpd=None)    # 1 request per 0.2 s
        lim.reserve(M, 1, 1, lim.now() + 1)
        t = lim.reserve(M, 1, 1, lim.now() + 1)
        self.assertAlmostEqual(t.wait_s, 0.2, delta=0.02)
        t0 = time.monotonic()
        g = await t.acquire()
        self.assertAlmostEqual(time.monotonic() - t0, 0.2, delta=0.06)
        self.assertGreater(g.waited_s, 0.15)

    async def test_never_waits_past_deadline(self):
        lim = limiter(time.monotonic, rpm=(1, 0.2), tpm=None, rpd=None)
        g = lim.reserve(M, 1, 1, lim.now() + 1)
        deadline = lim.now() + 0.3
        t = lim.reserve(M, 1, 1, deadline)

        async def bad_news():
            await asyncio.sleep(0.05)       # a 429 says the bucket is far emptier than we thought
            g.settle(rec(status=429, outcome="rate_limited", headers_ms=50.0,
                         error_body=body("requests per minute", "RPM", 1, 1, 1, "5s")))
        asyncio.get_running_loop().create_task(bad_news())
        t0 = time.monotonic()
        with self.assertRaises(Unavailable):
            await t.acquire()
        self.assertLess(time.monotonic() - t0, 0.3 + 0.05)

    async def test_cancel_moves_the_queue_up(self):
        lim = limiter(time.monotonic, rpm=(1, 0.3), tpm=None, rpd=None)
        lim.reserve(M, 1, 1, lim.now() + 2)
        a = lim.reserve(M, 1, 1, lim.now() + 2)
        b = lim.reserve(M, 1, 1, lim.now() + 2)
        self.assertAlmostEqual(b.wait_s, 0.6, delta=0.03)
        a.cancel()
        t0 = time.monotonic()
        await b.acquire()
        self.assertLess(time.monotonic() - t0, 0.45)
        with self.assertRaises(Unavailable):
            await a.acquire()


class OutputTokenLimitTests(unittest.TestCase):
    """qwen's OTPM, found in the real drafter step (no header shows it)."""

    def otpm_body(self, used=820, requested=184, wait="240ms"):
        return body("output tokens per minute", "OTPM", 1000, used, requested, wait)

    def test_parse_units(self):
        self.assertEqual(parse_limit_body(self.otpm_body())["unit"], "output_tokens")
        self.assertEqual(parse_limit_body(body("input tokens per minute", "ITPM", 5000, 1, 1, "1s"))["unit"],
                         "input_tokens")

    def test_learned_otpm_charges_output_only_and_never_blocks_a_long_prompt(self):
        c = Clock()
        lim = limiter(c, rpm=None, rpd=None)
        g = lim.reserve(M, 400, 300, c() + 5)
        lim.settle(g, rec(429, error_body=self.otpm_body(), outcome="rate_limited"))
        st = lim.models[M]
        self.assertEqual((st.buckets["OTPM"].unit, st.buckets["OTPM"].capacity, st.out_hold_cap),
                         ("output_tokens", 1000.0, 184.0))
        c.t += 60                                          # OTPM refills
        g2 = lim.reserve(M, 1726, 1200, c() + 5)          # a long prompt: input doesn't count against OTPM
        self.assertEqual(g2.wait_s, 0.0)
        before = lim.levels(M)["OTPM"]
        lim.settle(g2, rec(usage={"prompt_tokens": 1700, "completion_tokens": 250, "queue_time": 0.0}))
        self.assertAlmostEqual(before - lim.levels(M)["OTPM"], 250 - 184, delta=1.0)   # hold -> real completion

    def test_input_limit_charges_prompt_only(self):
        c = Clock()
        lim = limiter(c, rpm=None, rpd=None, tpm=None)
        lim.models[M].buckets["ITPM"] = Bucket("ITPM", "input_tokens", 5000, 60.0, c(), 5000, "test")
        g = lim.reserve(M, 1200, 1000, c() + 5)
        self.assertAlmostEqual(5000 - lim.levels(M)["ITPM"], 1200, delta=1.0)
        lim.release(g)

    def test_from_probe_seeds_runtime_limits(self):
        lim = Limiter.from_probe()
        st = lim.models["qwen/qwen3.8-27b"]
        if "OTPM" in st.buckets:                            # present once a real run has hit it
            self.assertEqual(st.buckets["OTPM"].unit, "output_tokens")
            self.assertIsNotNone(st.out_hold_cap)
            self.assertEqual(lim.predict_wait("qwen/qwen3.8-27b", 1726, 1200), 0.0)


class PredictWaitTests(unittest.TestCase):
    def test_predicts_like_reserve_and_commits_nothing(self):
        c = Clock()
        lim = limiter(c, tpm_level=1000)                     # TPM refills 8000/60 per s
        before = lim.levels(M)
        self.assertEqual(lim.predict_wait(M, 500, 100), 0.0)
        w = lim.predict_wait(M, 3000, 100)
        self.assertAlmostEqual(w, (3000 - 1000) / (8000 / 60), places=3)
        self.assertEqual(lim.levels(M), before)               # nothing reserved
        t = lim.reserve(M, 3000, 100, c() + 60)
        self.assertAlmostEqual(t.wait_s, w, places=3)         # same prediction as a real reservation
        self.assertGreater(lim.predict_wait(M, 10, 10), w)    # counts the ticket queued in front
        self.assertIsNone(lim.predict_wait(M, 9000, 100))     # can never fit
        self.assertIsNone(lim.predict_wait("unknown", 10, 10))

    def test_abandon_keeps_the_admission_charge(self):
        c = Clock()
        lim = limiter(c)
        g = lim.reserve(M, 500, 100, c() + 5)
        after = lim.levels(M)["TPM"]
        lim.abandon(g)
        self.assertEqual(lim.levels(M)["TPM"], after)        # not released
        self.assertNotIn(g.id, lim.models[M].inflight)


if __name__ == "__main__":
    unittest.main()
