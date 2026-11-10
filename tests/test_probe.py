"""Offline tests for the probe's helpers and analysis parsing (no network, no Groq budget)."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import analyze_probe as an  # noqa: E402
import probe_groq as pg  # noqa: E402
from relay.fake_groq import fmt_duration  # noqa: E402
from relay.ledger import Ledger, load_budget  # noqa: E402


class DurationTests(unittest.TestCase):
    def test_groq_formats(self):
        for s, v in [("7.66s", 7.66), ("2m59.56s", 179.56), ("1h2m3s", 3723), ("120ms", 0.12), ("2", 2.0)]:
            self.assertAlmostEqual(pg.parse_duration(s), v, places=3)
            self.assertAlmostEqual(an.parse_duration(s), v, places=3)
        self.assertIsNone(pg.parse_duration(None))
        self.assertIsNone(pg.parse_duration("soon"))

    def test_fake_roundtrip(self):
        for v in (0.25, 7.66, 179.56, 3723.4):
            self.assertAlmostEqual(pg.parse_duration(fmt_duration(v)), v, delta=0.01)


class BodyParseTests(unittest.TestCase):
    def test_rate_limit_body(self):
        msg = ("Rate limit reached for model `openai/gpt-oss-20b` in organization `org_REDACTED` service tier "
               "`on_demand` on requests per minute (RPM): Limit 30, Used 30, Requested 1. Please try again in "
               "1.92s. Need more tokens?")
        m = an.BODY_RE.search(msg).groupdict()
        self.assertEqual((m["short"], m["limit"], m["used"], m["requested"], m["wait"]), ("RPM", "30", "30", "1", "1.92s"))

    def test_too_large_body(self):
        msg = ("Request too large for model `openai/gpt-oss-20b` in organization `org_REDACTED` service tier "
               "`on_demand` on tokens per minute (TPM): Limit 8000, Requested 8591, please reduce")
        m = an.TOO_LARGE_RE.search(msg).groupdict()
        self.assertEqual((m["short"], m["limit"], m["requested"]), ("TPM", "8000", "8591"))


class StatsTests(unittest.TestCase):
    def test_pct_and_ols(self):
        self.assertEqual(an.pct([5, 1, 3, 2, 4], 0.5), 3)
        self.assertEqual(an.pct(list(range(1, 21)), 0.95), 19)
        f = an.ols([0, 1, 2, 3], [1, 3, 5, 7])
        self.assertAlmostEqual(f["intercept"], 1)
        self.assertAlmostEqual(f["slope"], 2)
        self.assertEqual(an.dist([1.0] * 5)["p95_reliable"], False)


class PlanTests(unittest.TestCase):
    def test_allocation_within_brief(self):
        per = {}
        for _, m, _, n in pg.ALLOCATION:
            per[m] = per.get(m, 0) + n
        self.assertTrue(all(v <= 100 for v in per.values()))
        self.assertLessEqual(sum(per.values()), 220)
        # phase code limits agree with the table
        self.assertEqual(sum(k for m, k in pg.P4_STEPS if m == pg.M20), 15)
        self.assertEqual(pg.P3_BURST + len(pg.P3_RECOVERY_OFFSETS), 50)
        self.assertEqual(len(pg.P1_OFFSETS[pg.M20]), 6)

    def test_preflight_fits_at_measured_scale_overhead(self):
        with tempfile.TemporaryDirectory() as tmp:
            with Ledger(load_budget(), Path(tmp) / "l.jsonl") as led:
                ok, text = pg.preflight(led, pg.PHASES, 64)
                self.assertTrue(ok, text)

    def test_nonce_leads_every_prompt(self):
        pr = pg.Prompts()
        for size in ("tiny", "router", "draft"):
            msgs, _ = pr.messages(size, "n:abcd1234")
            self.assertTrue(msgs[0]["content"].startswith("[n:abcd1234]"))


if __name__ == "__main__":
    unittest.main()
