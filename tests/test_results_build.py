"""build_results.py / make_tables.py: stand-in rows are graded only as exact replays, and can't hide."""
from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from relay.retrieve import Retriever
from scripts import build_results, make_tables

R = Retriever.from_blocks(2400)
HEADS = next(p for p in R.pieces if "attention-p05-b006" in p.block_ids)
TEXT = ("Several heads let the model attend to different representation subspaces at different positions [S1]. "
        "The paper uses h = 8 heads with d_k = d_v = 64 [S1]. "
        "The total computational cost is similar to single-head attention with full dimensionality [S1].")


def row(source, rid, text, start):
    reply = {"kind": "answer", "text": text, "level": 0, "released_ms": 1000.0, "tokens_in": 100, "tokens_out": 50,
             "budget_wait_ms": 0.0, "model_calls": [], "checks": {},
             "sources": [{"sid": "S1", "cite": HEADS.cite(), "pid": HEADS.pid, "paper": "attention",
                          "block_ids": list(HEADS.block_ids)}]}
    return {"type": "answer", "run_id": "r", "scenario": "eval:retrieval" if source == "real" else "steady",
            "source": source, "setting": "large", "drafter": "openai/gpt-oss-120b", "qid": "Q2", "request_id": rid,
            "start_time": start, "degraded": False,
            "setting_def": {"piece_chars": 2400, "k": 3, "rerank": False, "router": True}, "reply": reply}


class StandInGrades(unittest.TestCase):
    def test_replay_inherits_grade_placeholder_does_not(self):
        with tempfile.TemporaryDirectory() as d:
            raw, out = Path(d) / "raw", Path(d) / "out"
            raw.mkdir()
            rows = [row("real", "real-1", TEXT, "2026-11-10T10:00:00"),
                    row("stand-in", "si-1", TEXT, "2026-11-10T11:00:00"),
                    row("stand-in", "si-2", "Placeholder first sentence [S1]", "2026-11-10T11:00:01")]
            (raw / "x.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
            build_results.main(["--raw-dir", str(raw), "--out-dir", str(out), "--overrides", str(Path(d) / "none.jsonl")])
            with (out / "results.csv").open(encoding="utf-8", newline="") as f:
                got = {r["request_id"]: r for r in csv.DictReader(f)}
            self.assertEqual(got["real-1"]["answer_grade"], "correct")
            self.assertEqual((got["si-1"]["answer_grade"], got["si-1"]["grade_source"]), ("correct", "replay:real-1"))
            self.assertEqual((got["si-2"]["answer_grade"], got["si-2"]["grade_source"]), ("", "not_graded:stand-in"))
            make_tables.main(["--results", str(out / "results.csv"), "--out", str(out / "tables.md")])
            md = (out / "tables.md").read_text(encoding="utf-8")
            self.assertIn("| steady | stand-in | 2 | 1 | not_graded:stand-in 1 |", md)

    def test_tables_refuse_a_stand_in_grade_without_replay(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "results.csv"
            with p.open("w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=build_results.COLUMNS)
                w.writeheader()
                w.writerow({c: "" for c in build_results.COLUMNS} | {
                    "request_id": "x", "scenario": "steady", "source": "stand-in", "answer_grade": "correct",
                    "grade_source": "auto", "superseded": "0", "tokens_in": "1", "tokens_out": "1"})
            with self.assertRaises(SystemExit):
                make_tables.main(["--results", str(p), "--out", str(Path(d) / "t.md")])


if __name__ == "__main__":
    unittest.main()
