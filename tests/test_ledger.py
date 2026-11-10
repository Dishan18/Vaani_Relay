"""Offline tests for the budget ledger and BudgetedClient (no network, no Groq budget)."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from relay.ledger import (Budget, BudgetedClient, BudgetExceeded, Cap, Ledger, ModelCap,
                          estimate_prompt_tokens, load_budget)
from relay.llm import CallRecord, Endpoint, LLMClient, Timeouts
from tests.fake_openai_server import FakeOpenAIServer, delta

M1, M2 = "model-a", "model-b"


def payload(model=M1, content="x" * 30, max_tokens=50):
    return {"model": model, "messages": [{"role": "user", "content": content}], "max_tokens": max_tokens}


def record(outcome="ok", status=200, usage_total=None, headers=()):
    rec = CallRecord(call_id="c", source="real", url="u", model=M1, started_at="t", outcome=outcome, status=status)
    if usage_total is not None:
        rec.usage = {"total_tokens": usage_total}
    rec.rate_limit_headers = [list(h) for h in headers]
    return rec


class Clock:
    def __init__(self, t=1_000_000.0):
        self.t = t

    def __call__(self):
        return self.t


class LedgerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "ledger.jsonl"
        self.clock = Clock()
        self.budget = Budget([
            Cap("account", "*", {M1: ModelCap(10, 10_000), M2: ModelCap(10, 10_000)}),
            Cap("probe", "probe", {M1: ModelCap(3, 1_000), M2: ModelCap(5, 5_000)}, total_requests=6),
        ], window_hours=24, min_remaining_requests_fraction=0.5)

    def tearDown(self):
        self.tmp.cleanup()

    def ledger(self):
        return Ledger(self.budget, self.path, clock=self.clock)

    def test_request_cap_per_model(self):
        with self.ledger() as led:
            for _ in range(3):
                led.settle(led.reserve(payload(), "probe", "p"), record(usage_total=10))
            with self.assertRaisesRegex(BudgetExceeded, "requests 3/3"):
                led.reserve(payload(), "probe", "p")
            led.reserve(payload(M2), "probe", "p")  # other model unaffected

    def test_token_cap_counts_reservation_before_sending(self):
        with self.ledger() as led:
            with self.assertRaisesRegex(BudgetExceeded, "tokens"):
                led.reserve(payload(max_tokens=990), "probe", "p")  # est prompt + 990 > 1000

    def test_inflight_reservations_count(self):
        with self.ledger() as led:
            rs = [led.reserve(payload(), "probe", "p") for _ in range(3)]
            with self.assertRaises(BudgetExceeded):
                led.reserve(payload(), "probe", "p")
            led.release(rs[0])
            led.reserve(payload(), "probe", "p")

    def test_total_request_cap(self):
        with self.ledger() as led:
            for m in (M1, M1, M1, M2, M2, M2):
                led.settle(led.reserve(payload(m), "probe", "p"), record(usage_total=5))
            with self.assertRaisesRegex(BudgetExceeded, "total requests 6/6"):
                led.reserve(payload(M2), "probe", "p")

    def test_account_cap_applies_to_every_scope(self):
        with self.ledger() as led:
            for _ in range(10):
                led.settle(led.reserve(payload(M2), "sweep", "s"), record(usage_total=5))
            with self.assertRaisesRegex(BudgetExceeded, "account"):
                led.reserve(payload(M2), "sweep", "s")
            with self.assertRaisesRegex(BudgetExceeded, "account"):
                led.reserve(payload(M2), "probe", "p")  # probe calls also count against the account

    def test_unknown_model_and_missing_max_tokens_refused(self):
        with self.ledger() as led:
            with self.assertRaisesRegex(BudgetExceeded, "not allowed"):
                led.reserve(payload("model-z"), "probe", "p")
            p = payload()
            del p["max_tokens"]
            with self.assertRaisesRegex(BudgetExceeded, "max_tokens"):
                led.reserve(p, "probe", "p")

    def test_charging_rules(self):
        with self.ledger() as led:
            p = payload(max_tokens=100)
            est = estimate_prompt_tokens(p)
            e = led.settle(led.reserve(p, "probe", "p"), record(usage_total=20))
            self.assertEqual((e["charged_requests"], e["charged_tokens"]), (1, 20))       # what Groq reports
            e = led.settle(led.reserve(p, "probe", "p"), record(usage_total=est + 500))
            self.assertEqual(e["charged_tokens"], est + 500)
            e = led.settle(led.reserve(p, "probe", "p"), record("rate_limited", 429))
            self.assertEqual((e["charged_requests"], e["charged_tokens"]), (1, est))
        with self.ledger() as led:  # fresh caps window not needed: M2 has room
            e = led.settle(led.reserve(payload(M2), "probe", "p"), record("connect_error", None))
            self.assertEqual((e["charged_requests"], e["charged_tokens"]), (0, 0))
            e = led.settle(led.reserve(payload(M2, max_tokens=70), "probe", "p"), record("timeout_stall", 200))
            self.assertEqual(e["charged_tokens"], estimate_prompt_tokens(payload(M2)) + 70)

    def test_old_entries_are_reread_with_the_reported_usage_rule(self):
        from relay.ledger import charge_of
        old = {"status": 200, "outcome": "ok", "usage_total_tokens": 1700, "charged_tokens": 3100,
               "charge_basis": "max(prompt estimate + max_tokens, reported usage)"}
        self.assertEqual(charge_of(old), 1700)
        self.assertEqual(charge_of({**old, "outcome": "timeout_stall"}), 3100)
        self.assertEqual(charge_of({**old, "status": 429, "usage_total_tokens": None, "charged_tokens": 900}), 900)

    def test_inflight_reservations_still_count(self):
        with self.ledger() as led:
            p = payload(max_tokens=100)
            before = led.headroom("probe", payload()["model"])["tokens"]
            r = led.reserve(p, "probe", "p")
            self.assertEqual(before - led.headroom("probe", p["model"])["tokens"], r.reserved_tokens)
            led.release(r)

    def test_rolling_window(self):
        with self.ledger() as led:
            for _ in range(3):
                led.settle(led.reserve(payload(), "probe", "p"), record(usage_total=5))
            self.clock.t += 23.9 * 3600
            with self.assertRaises(BudgetExceeded):
                led.reserve(payload(), "probe", "p")
            self.clock.t += 0.2 * 3600
            led.reserve(payload(), "probe", "p")

    def test_persists_across_processes(self):
        with self.ledger() as led:
            for _ in range(3):
                led.settle(led.reserve(payload(), "probe", "p"), record(usage_total=5))
        lines = self.path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 3)
        self.assertEqual(json.loads(lines[0])["scope"], "probe")
        with self.ledger() as led:
            with self.assertRaises(BudgetExceeded):
                led.reserve(payload(), "probe", "p")

    def test_second_holder_refused(self):
        with self.ledger():
            with self.assertRaisesRegex(BudgetExceeded, "another process"):
                self.ledger()

    def test_server_floor_halts(self):
        with self.ledger() as led:
            led.settle(led.reserve(payload(M2), "probe", "p"),
                       record(usage_total=5, headers=[("x-ratelimit-limit-requests", "1000"),
                                                      ("x-ratelimit-remaining-requests", "499")]))
            self.assertIsNotNone(led.halted)
            with self.assertRaisesRegex(BudgetExceeded, "halted"):
                led.reserve(payload(M2), "probe", "p")

    def test_real_config_matches_probe_brief(self):
        b = load_budget()
        probe = next(c for c in b.caps if c.name == "probe")
        self.assertEqual(probe.total_requests, 220)
        for mc in probe.models.values():
            self.assertLessEqual(mc.requests, 100)
            self.assertLessEqual(mc.tokens, 20_000)
        self.assertEqual(b.window_hours, 24)


class BudgetedClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_end_to_end_writes_ledger_and_raw(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "l.jsonl"
            budget = Budget([Cap("probe", "probe", {"fake-model": ModelCap(1, 1_000)})])
            raw = []
            async with FakeOpenAIServer() as srv:
                srv.scripts.append([("head", 200, [("x-ratelimit-remaining-requests", "999")]),
                                    ("sse", delta(content="hi")),
                                    ("sse", delta(finish="stop", usage={"total_tokens": 9})), ("sse", "[DONE]")])
                with Ledger(budget, path) as led:
                    bc = BudgetedClient(LLMClient(Endpoint(srv.base_url, source="real"),
                                                  Timeouts(2, 2, 5)), led, "probe", raw.append)
                    rec = await bc.stream_chat({"model": "fake-model", "max_tokens": 10,
                                                "messages": [{"role": "user", "content": "q"}]}, phase="P0")
                    self.assertEqual(rec.text, "hi")
                    with self.assertRaises(BudgetExceeded):  # cap of 1 request: refused, nothing sent
                        await bc.stream_chat({"model": "fake-model", "max_tokens": 10,
                                              "messages": [{"role": "user", "content": "q"}]}, phase="P0")
                    self.assertEqual(len(srv.requests), 1)
            entry = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual((entry["phase"], entry["outcome"], entry["hdr_remaining_requests"]), ("P0", "ok", "999"))
            self.assertEqual(raw[0]["record"]["text"], "hi")


if __name__ == "__main__":
    unittest.main()
