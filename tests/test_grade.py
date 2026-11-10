"""Offline tests for relay.grade with hand-written replies over the real pieces."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from relay.grade import add_override, answer_id, grade, load_overrides, load_questions
from relay.retrieve import Retriever

R = Retriever.from_blocks(2400)
PIECES = {p.pid: {"text": p.text, "paper": p.paper, "block_ids": p.block_ids} for p in R.pieces}
QS = load_questions()


def piece_with(block: str):
    return next(p for p in R.pieces if block in p.block_ids)


def src(sid: str, p) -> dict:
    return {"sid": sid, "cite": p.cite(), "pid": p.pid, "paper": p.paper, "block_ids": list(p.block_ids)}


def reply(text, *pieces, kind="answer"):
    return {"kind": kind, "text": text, "sources": [src(f"S{i}", p) for i, p in enumerate(pieces, 1)]}


HEADS = piece_with("attention-p05-b006")
MOSHI_ABS = piece_with("moshi-p01-b005")
Q2_OK = ("Several heads let the model attend to different representation subspaces at different positions [S1]. "
         "The paper uses h = 8 heads with d_k = d_v = 64 [S1]. "
         "The total computational cost is similar to single-head attention with full dimensionality [S1].")


class AnswerableTests(unittest.TestCase):
    def test_correct_when_every_fact_present_and_cited_to_gold(self):
        g = grade(QS["Q2"], reply(Q2_OK, HEADS), PIECES)
        self.assertEqual((g["label"], g["sure"]), ("correct", True), g["reasons"])
        self.assertTrue(all(f["supported"] for f in g["facts"]))

    def test_wrong_paper_citation_is_wrong_even_with_right_facts(self):
        text = Q2_OK.replace("positions [S1].", "positions [S2].")
        g = grade(QS["Q2"], reply(text, HEADS, MOSHI_ABS), PIECES)
        self.assertEqual(g["label"], "wrong")
        self.assertEqual(g["wrong_paper_citations"][0]["paper"], "moshi")

    def test_fact_cited_to_a_passage_that_does_not_hold_it_is_wrong_unsure(self):
        other = piece_with("attention-p06-b009")          # positional encodings, says nothing about heads
        g = grade(QS["Q2"], reply(Q2_OK.replace("64 [S1]", "64 [S2]"), HEADS, other), PIECES)
        self.assertEqual((g["label"], g["sure"]), ("wrong", False))
        self.assertTrue(any("unsupported" in r for r in g["reasons"]))

    def test_missing_one_fact_is_wrong_but_unsure(self):
        text = Q2_OK.rsplit(" The total", 1)[0]
        g = grade(QS["Q2"], reply(text, HEADS), PIECES)
        self.assertEqual((g["label"], g["sure"]), ("wrong", False))
        self.assertIn("missing 1/3", g["reasons"][0])

    def test_missing_most_facts_is_surely_wrong(self):
        g = grade(QS["Q2"], reply("Heads are useful [S1].", HEADS), PIECES)
        self.assertEqual((g["label"], g["sure"]), ("wrong", True))

    def test_wrong_number_pattern(self):
        table = piece_with("attention-p08-b002")
        g = grade(QS["Q5"], reply("Table 2 gives the big model a training cost of 3.3 · 10^18 FLOPs [S1].", table), PIECES)
        self.assertEqual(g["label"], "wrong")
        self.assertIn("base model", g["reasons"][0])
        ok = grade(QS["Q5"], reply("Table 2 reports 2.3 · 10^19 FLOPs for the big Transformer [S1].", table), PIECES)
        self.assertEqual(ok["label"], "correct", ok["reasons"])

    def test_refusing_an_answerable_question(self):
        g = grade(QS["Q2"], reply("The papers don't say.", kind="refusal"), PIECES)
        self.assertEqual(g["label"], "refused_wrongly")


class UnanswerableTests(unittest.TestCase):
    def test_clean_refusal(self):
        g = grade(QS["Q6"], reply("The papers don't say.", kind="refusal"), PIECES)
        self.assertEqual((g["label"], g["sure"]), ("refused_correctly", True))

    def test_borrowed_number_is_wrong(self):
        g = grade(QS["Q17"], reply("H-Net's latency is about 200 ms in practice [S1].", MOSHI_ABS), PIECES)
        self.assertEqual(g["label"], "wrong")
        self.assertTrue(g["borrowed"])

    def test_answering_an_unanswerable_question_is_wrong(self):
        g = grade(QS["Q18"], reply("Training used a lot of compute [S1].", HEADS), PIECES)
        self.assertEqual(g["label"], "wrong")

    def test_refusal_worded_as_cited_answer_is_unsure(self):
        g = grade(QS["Q6"], reply("The paper reports BLEU and training cost, but inference latency is not reported [S1].",
                                  HEADS), PIECES)
        self.assertEqual((g["label"], g["sure"]), ("refused_correctly", False))

    def test_refusal_with_other_digits_is_unsure(self):
        g = grade(QS["Q18"], reply("The papers don't say. They report 1.3B-parameter models only.", kind="refusal"), PIECES)
        self.assertEqual((g["label"], g["sure"]), ("refused_correctly", False))

    def test_passages_reply_to_unanswerable_is_wrong(self):
        g = grade(QS["Q6"], reply("No written answer this time: here are the most relevant passages.\n"
                                  "[S1] 1706.03762v7 p.5: \"we employ h = 8 parallel attention layers\"", HEADS,
                                  kind="passages"), PIECES)
        self.assertEqual(g["label"], "wrong")

    def test_busy_reply_is_wrong(self):
        self.assertEqual(grade(QS["Q6"], reply("Sorry, the service is busy. Please retry in 10 s.", kind="busy"),
                               PIECES)["label"], "wrong")
        self.assertEqual(grade(QS["Q2"], reply("Sorry, the service is busy. Please retry in 10 s.", kind="busy"),
                               PIECES)["label"], "wrong")


class CitationNormalisationTests(unittest.TestCase):
    def test_fullwidth_citations_count(self):
        g = grade(QS["Q2"], reply(Q2_OK.replace("[S1]", "【S1】"), HEADS), PIECES)
        self.assertEqual(g["label"], "correct", g["reasons"])

    def test_passages_reply_with_gold_facts_is_correct(self):
        text = ("No written answer this time: here are the most relevant passages from the papers.\n"
                "[S1] 1706.03762v7: \"Multi-head attention allows the model to jointly attend to information from "
                "different representation subspaces at different positions.\"\n"
                "[S1] 1706.03762v7: \"In this work we employ h = 8 parallel attention layers, or heads. For each of these "
                "we use d_k = d_v = d_model/h = 64. Due to the reduced dimension of each head, the total computational "
                "cost is similar to that of single-head attention with full dimensionality.\"")
        g = grade(QS["Q2"], reply(text, HEADS, kind="passages"), PIECES)
        self.assertEqual(g["label"], "correct", g["reasons"])


class OverrideTests(unittest.TestCase):
    def test_append_only_latest_wins_and_reason_required(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "o.jsonl"
            add_override("abc", "wrong", "first look", path=p)
            add_override("abc", "correct", "the paraphrase is right", path=p)
            self.assertEqual(load_overrides(p)["abc"]["label"], "correct")
            self.assertEqual(len(p.read_text(encoding="utf-8").splitlines()), 2)
            with self.assertRaises(ValueError):
                add_override("abc", "correct", " ", path=p)
            with self.assertRaises(ValueError):
                add_override("abc", "great", "x", path=p)

    def test_answer_id_depends_on_text_and_sources(self):
        a = reply(Q2_OK, HEADS)
        self.assertEqual(answer_id("Q2", a), answer_id("Q2", dict(a)))
        self.assertNotEqual(answer_id("Q2", a), answer_id("Q2", reply(Q2_OK + " More.", HEADS)))
        self.assertNotEqual(answer_id("Q2", a), answer_id("Q3", a))


if __name__ == "__main__":
    unittest.main()
