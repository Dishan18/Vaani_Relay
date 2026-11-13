"""Hybrid retrieval (relay.dense): BM25 + embeddings, small-to-big. Skipped without fastembed."""
from __future__ import annotations

import unittest

from relay import dense
from relay.retrieve import Retriever


@unittest.skipUnless(dense.available(), "fastembed not installed (pip install -r requirements-service.txt)")
class HybridTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.r = dense.HybridRetriever.build(child_chars=600, parent_chars=2400)

    def test_returns_parent_pieces(self):
        hits = self.r.search("How wide is the hidden layer inside each feed-forward block?", 3)
        self.assertEqual(len(hits), 3)
        self.assertTrue(all(h.piece.pid.split(":")[1] == "2400" for h in hits))
        self.assertEqual(len({h.piece.pid for h in hits}), 3)          # distinct parents
        self.assertTrue(any("2048" in h.piece.text for h in hits))

    def test_paraphrase_bm25_misses_hybrid_finds(self):
        # dev question R14: "unlabelled speech" / "transcripts" vs the paper's "unsupervised audio dataset"
        q = "How many hours of unlabelled speech went into pretraining, and how did they get transcripts for it?"
        gold = "moshi-p17-b007"
        bm25 = Retriever.from_blocks(2400).search(q, 3)
        self.assertFalse(any(gold in h.piece.block_ids for h in bm25))
        hy = self.r.search(q, 3)
        self.assertTrue(any(gold in h.piece.block_ids for h in hy))

    def test_paper_filter_and_dense_query(self):
        hits = self.r.search("attention", 3, papers=["moshi"], dense_query="Moshi depth transformer audio tokens")
        self.assertTrue(all(h.piece.paper == "moshi" for h in hits))

    def test_semantic_evidence_separates_off_topic(self):
        on = self.r.search("What dropout rate does the base Transformer use?", 3)
        off = self.r.search("How do I bake sourdough bread at home?", 3)
        self.assertGreater(self.r.dense_best("What dropout rate does the base Transformer use?", on),
                           self.r.dense_best("How do I bake sourdough bread at home?", off))

    def test_factory(self):
        from relay.pipeline import load_config, load_production_config
        self.assertIsInstance(dense.build_retriever(load_production_config()), dense.HybridRetriever)
        self.assertIsInstance(dense.build_retriever(load_config()), Retriever)


if __name__ == "__main__":
    unittest.main()
