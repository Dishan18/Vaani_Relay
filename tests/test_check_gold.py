import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import check_gold  # noqa: E402

BLOCKS = [
    {"block_id": "attention-p1-b1", "paper": "attention", "kind": "body", "text": "we employ h = 8 parallel attention layers"},
    {"block_id": "moshi-p1-b1", "paper": "moshi", "kind": "body", "text": "a theoretical latency of 160ms"},
    {"block_id": "attention-p9-b1", "paper": "attention", "kind": "reference", "text": "latency in references is ignored"},
]


def ans(evs, match=(r"h\s*=\s*8",)):
    return {"qid": "A", "paper": "attention", "arxiv_id": "1706.03762v7", "answerable": True,
            "facts": [{"fact": "h=8", "evidence": evs, "match": list(match)}]}


class CheckGold(unittest.TestCase):
    def errors(self, q):
        return check_gold.check([q], BLOCKS)[1]

    def test_verbatim_quote_passes(self):
        self.assertEqual(self.errors(ans([{"block_id": "attention-p1-b1", "quote": "h = 8 parallel"}])), [])

    def test_grading_pattern_must_match_its_fact_or_evidence(self):
        ev = [{"block_id": "attention-p1-b1", "quote": "h = 8 parallel"}]
        self.assertEqual(self.errors(ans(ev, match=(r"8 parallel",))), [])
        self.assertTrue(self.errors(ans(ev, match=(r"sixteen heads",))))
        self.assertTrue(self.errors(ans(ev, match=())))
        self.assertTrue(self.errors(ans(ev, match=(r"(unclosed",))))

    def test_paraphrased_quote_fails(self):
        self.assertTrue(self.errors(ans([{"block_id": "attention-p1-b1", "quote": "h=8 heads"}])))

    def test_evidence_from_wrong_paper_fails(self):
        e = self.errors(ans([{"block_id": "moshi-p1-b1", "quote": "160ms"}]))
        self.assertTrue(any("is from moshi" in x for x in e))

    def test_fact_without_evidence_fails(self):
        self.assertTrue(self.errors(ans([])))

    def test_refusal_pattern_found_in_paper_fails(self):
        q = {"qid": "R", "paper": "moshi", "arxiv_id": "2410.00037v1", "answerable": False, "absent": ["latenc"]}
        self.assertTrue(self.errors(q))

    def test_refusal_ignores_references_and_other_papers(self):
        q = {"qid": "R", "paper": "attention", "arxiv_id": "1706.03762v7", "answerable": False, "absent": ["latenc"]}
        self.assertEqual(self.errors(q), [])

    def test_real_question_set_passes(self):
        rows, errors = check_gold.check(check_gold.load_jsonl(check_gold.QUESTIONS),
                                        check_gold.load_jsonl(check_gold.BLOCKS))
        self.assertEqual(errors, [])
        self.assertEqual(len(rows), 18)
        self.assertEqual(sum(not r["answerable"] for r in rows), 4)


if __name__ == "__main__":
    unittest.main()
