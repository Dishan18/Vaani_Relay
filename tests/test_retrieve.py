"""Offline tests for relay.retrieve (no model calls)."""
from __future__ import annotations

import unittest

from relay.retrieve import BM25, Retriever, build_pieces, stem, tokenize


def blk(i, text, section="2 Method", kind="body", page=1, paper="attention"):
    return {"block_id": f"{paper}-p{page:02d}-b{i:03d}", "paper": paper, "arxiv_id": "1706.03762v7",
            "page": page, "section": section, "text": text, "kind": kind, "in_appendix": False}


class PieceTests(unittest.TestCase):
    def test_never_crosses_a_section_and_joins_across_pages(self):
        bl = [blk(1, "Alpha beta gamma. " * 5, page=1), blk(2, "Delta epsilon. " * 5, page=2),
              blk(3, "Other section text here. " * 5, section="3 Results", page=2)]
        ps = build_pieces(bl, 2000)
        self.assertEqual([p.section for p in ps], ["2 Method", "3 Results"])
        self.assertEqual(ps[0].pages, (1, 2))
        self.assertEqual(ps[0].cite(), "1706.03762v7 pp.1-2 §2")

    def test_caption_glued_to_next_block_and_table_not_split(self):
        table = "Model BLEU\n" + "row 1 2 3\n" * 200
        bl = [blk(1, "Intro sentence one. Intro sentence two."), blk(2, "Table 2: costs.", kind="caption"),
              blk(3, table, kind="table")]
        ps = build_pieces(bl, 300)
        holder = [p for p in ps if "caption" in p.kinds]
        self.assertEqual(len(holder), 1)
        self.assertIn("table", holder[0].kinds)
        self.assertIn("Table 2: costs.\nModel BLEU", holder[0].text)

    def test_references_and_front_matter_excluded(self):
        bl = [blk(1, "Body text about attention heads."), blk(2, "[1] Some Author. Attention heads.", kind="reference"),
              blk(3, "Title of paper", kind="front_matter")]
        ps = build_pieces(bl, 1000)
        self.assertEqual([p.block_ids for p in ps], [("attention-p01-b001",)])

    def test_long_block_split_at_sentences_and_tiny_tail_merged(self):
        text = " ".join(f"Sentence number {i} is here." for i in range(40))
        ps = build_pieces([blk(1, text), blk(2, "Tiny.")], 300)
        self.assertTrue(all(len(p.text) <= 300 + 10 for p in ps[:-1]))
        self.assertTrue(ps[-1].text.endswith("Tiny."))
        self.assertGreater(len(ps[-1].text), len("Tiny."))       # merged, not a piece of its own
        self.assertTrue(all(p.text.rstrip().endswith((".", "Tiny.")) for p in ps))

    def test_body_subsection_line_opens_a_subsection(self):
        bl = [blk(1, "Text before.", section="3.4 Generative"), blk(2, "3.4.4 Inner Monologue", section="3.4 Generative"),
              blk(3, "Monologue text.", section="3.4 Generative")]
        ps = build_pieces(bl, 1000)
        self.assertEqual([p.section for p in ps], ["3.4 Generative", "3.4.4 Inner Monologue"])
        self.assertTrue(ps[1].cite().endswith("§3.4.4"))

    def test_real_equation_fragments_stay_in_one_piece(self):
        for size in (1200, 2400):
            r = Retriever.from_blocks(size)
            holders = {p.pid for p in r.pieces for b in p.block_ids
                       if b in {f"dynamic_chunking-p06-b0{i}" for i in range(10, 16)}}
            self.assertEqual(len(holders), 1, size)


class BM25Tests(unittest.TestCase):
    def test_stem_and_tokenize(self):
        self.assertEqual(stem("boundaries"), "boundary")
        self.assertEqual(stem("heads"), "head")
        self.assertEqual(stem("class"), "class")
        self.assertEqual(tokenize("What are the heads?"), ["head"])

    def setUp(self):
        bl = [blk(1, "Multi-head attention uses eight heads of dimension 64.", section="3.2 Attention"),
              blk(2, "Positional encodings use sine and cosine functions.", section="3.5 Positional"),
              blk(3, "Training used eight GPUs for three days.", section="5 Training")]
        self.r = Retriever(build_pieces(bl, 1000), 1000)

    def test_exact_topic_ranks_first(self):
        hits = self.r.search("how many attention heads", 3)
        self.assertEqual(hits[0].piece.section, "3.2 Attention")
        self.assertEqual([h.rank for h in hits], list(range(1, len(hits) + 1)))

    def test_paper_filter(self):
        self.assertEqual(self.r.search("attention heads", 3, papers=["moshi"]), [])

    def test_rerank_is_deterministic_and_keeps_candidates(self):
        a = self.r.search("sine cosine positional", 3, use_rerank=True)
        b = self.r.search("sine cosine positional", 3, use_rerank=True)
        self.assertEqual([h.piece.pid for h in a], [h.piece.pid for h in b])
        self.assertEqual(a[0].piece.section, "3.5 Positional")
        self.assertIn("coverage", a[0].features)

    def test_evidence(self):
        hits = self.r.search("sine cosine positional encodings", 3)
        self.assertGreater(self.r.evidence("sine cosine positional encodings", hits)["best"], 0.9)
        self.assertEqual(self.r.evidence("sourdough bread", self.r.search("sourdough bread", 3))["best"], 0.0)

    def test_idf_positive(self):
        idx = BM25(self.r.pieces)
        self.assertTrue(all(v > 0 for v in idx.idf.values()))


if __name__ == "__main__":
    unittest.main()
