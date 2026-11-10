"""Offline tests for scripts/run_load.py (stand-in only)."""
from __future__ import annotations

import asyncio
import copy
import json
import math
import tempfile
import unittest
from pathlib import Path

from relay.retrieve import Retriever
from scripts import run_load
from standin.server import Replay, Switches, load_calibration

QS = [json.loads(x) for x in (Path(__file__).resolve().parents[1] / "data" / "questions.jsonl")
      .read_text(encoding="utf-8").splitlines() if x.strip()]


def fast_cal():
    c = copy.deepcopy(load_calibration())
    for m in c["models"].values():
        m["ttfb_ms"].update(mu=math.log(5.0), sigma=0.01)
        m["decode"]["tokens_per_s"] = 20000.0
    return c


class ArrivalTests(unittest.TestCase):
    def test_deterministic_and_naive_shares_overload_arrivals(self):
        a = run_load.arrivals(run_load.SCENARIOS["overload"], 120, 7, QS)
        b = run_load.arrivals(run_load.SCENARIOS["overload_naive_retry"], 120, 7, QS)
        self.assertEqual(a, b)
        self.assertEqual(a, run_load.arrivals(run_load.SCENARIOS["overload"], 120, 7, QS))
        self.assertTrue(all(0 <= x["t"] < 120 and x["patience_s"] > 0 for x in a))
        burst = run_load.arrivals(run_load.SCENARIOS["burst"], 60, 7, QS)
        self.assertGreaterEqual(sum(1 for x in burst if 20 <= x["t"] <= 25), 25)


class RunTests(unittest.TestCase):
    def test_short_run_rows_detail_give_up_and_naive_retries(self):
        sc = run_load.Scenario("t_naive", 0.0, Switches(p503=1.0), burst=None, naive=True)
        run_load.SCENARIOS["t_naive"] = sc

        def arr(*a, **k):
            return [{"t": 0.1, "qid": "Q2", "question": QS[1]["question"], "patience_s": 30.0},
                    {"t": 0.2, "qid": "Q1", "question": QS[0]["question"], "patience_s": 0.001}]
        old = run_load.arrivals
        run_load.arrivals = arr
        try:
            with tempfile.TemporaryDirectory() as d:
                rows, detail = Path(d) / "r.jsonl", Path(d) / "detail.jsonl"
                res = asyncio.run(run_load.run_scenario(sc, 5, 1, fast_cal(), QS, Retriever.from_blocks(2400), "x",
                                                        Replay(None), rows, detail))
                out = [json.loads(x) for x in rows.read_text(encoding="utf-8").splitlines()]
                reqs = [o for o in out if o["type"] == "answer"]
                self.assertEqual(len(reqs), 2)
                gave = next(o for o in reqs if o["qid"] == "Q1")
                self.assertTrue(gave["gave_up"])
                other = next(o for o in reqs if o["qid"] == "Q2")
                fails = [c for c in other["reply"]["model_calls"] if c["status"] == 503]
                self.assertGreaterEqual(len(fails), 1 + run_load.NAIVE["retries"])   # first try + 3 retries
                self.assertEqual(len(detail.read_text(encoding="utf-8").splitlines()), 2)
                self.assertEqual(res["requests"], 2)
        finally:
            run_load.arrivals = old
            run_load.SCENARIOS.pop("t_naive", None)


if __name__ == "__main__":
    unittest.main()
