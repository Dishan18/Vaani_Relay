"""The write-up tooling: every number comes in through a placeholder, and the renderer is exact."""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import check_numbers as cn  # noqa: E402
import render_writeup as rw  # noqa: E402

ALLOWED = json.loads((ROOT / "docs" / "writeup" / "allowed_literals.json").read_text(encoding="utf-8"))["patterns"]
FACTS = {"a.ms": {"value": "4622"}, "a.pct": {"value": "72.2"}, "w": {"value": 15.0},
         "reply": {"value": "line one\n[S1] quoted 2410.00037v1"}}
TABLES = "# Tables\n\n## Table 3: retrieval settings\n\nnotes here\n\n| a | b |\n|---|---|\n| 1 | 2 |\n\n## Other\n\nx\n"


class CheckerTests(unittest.TestCase):
    def test_a_typed_number_fails(self):
        bad = cn.untraced_digits("The p95 was 16558 ms and {{a.ms}} at p50.", ALLOWED)
        self.assertEqual([t for _, t, _ in bad], ["16558"])

    def test_names_and_placeholders_pass(self):
        text = ("Q14 on gpt-oss-120b and qwen3.8-27b got a 429 (Table 2b, BM25) in {{a.ms}} ms, "
                "{{quote:reply}} <!-- 123 in a comment -->")
        self.assertEqual(cn.untraced_digits(text, ALLOWED), [])

    def test_a_number_inside_an_identifier_still_fails(self):
        self.assertEqual([t for _, t, _ in cn.untraced_digits("model gpt-oss-999b", ALLOWED)], ["gpt-oss-999b"])


class RenderTests(unittest.TestCase):
    def test_facts_quotes_and_tables(self):
        out = rw.render("Took {{a.ms}} ms ({{a.pct}}%), wait {{w}} s.\n\n{{quote:reply}}\n\n{{table:Table 3}}", FACTS, TABLES)
        self.assertIn("Took 4622 ms (72.2%), wait 15 s.", out)
        self.assertIn("> line one\n> [S1] quoted 2410.00037v1", out)
        self.assertIn("**Table 3: retrieval settings**\n\n| a | b |", out)
        self.assertNotIn("notes here", out)          # only the table goes in; method notes stay in tables.md

    def test_unknown_fact_or_table_stops_the_render(self):
        with self.assertRaises(rw.RenderError):
            rw.render("{{nope}}", FACTS, TABLES)
        with self.assertRaises(rw.RenderError):
            rw.render("{{table:Table 9}}", FACTS, TABLES)


if __name__ == "__main__":
    unittest.main()
