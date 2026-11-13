"""Dense (embedding) retrieval, hybrid fusion with BM25, small-to-big pieces and a cross-encoder rerank.

Why (interview feedback, 2026-11-13, and our own data): BM25 only matches words. Q14 ("how does H-Net
decide where to cut the byte stream?") shares no terms with the routing-module passage and was never
retrieved by any BM25 setting. Embeddings match meaning; BM25 still wins on exact tokens ("d_model",
"Table 2", "P_drop"). So both run and their rankings are fused.

    r = HybridRetriever.build(child_chars=600, parent_chars=2400)      # small-to-big, hybrid
    hits = r.search("how does H-Net decide where to cut the byte stream?", k=3)

Pieces (relay.retrieve.build_pieces, unchanged: section-aware, tables whole):
  child   small pieces (default 600 chars) that are searched: precise matching
  parent  the large piece (default 2400 chars) of the same section holding the child's first block:
          what the drafter reads, so it keeps the surrounding context. This is "small-to-big"
          (parent-child) chunking; with child_chars == parent_chars it is plain hybrid search.
Ranking:
  1. BM25 over children and cosine similarity over child embeddings, each top ``candidates``;
  2. reciprocal rank fusion, score = sum over rankings of w / (60 + rank)   (Cormack et al. 2009);
  3. optional cross-encoder rerank of the fused top ``candidates`` children (query and passage read
     together, so it judges relevance better than either first stage; costs CPU time per query);
  4. children map to parents; the first k distinct parents are returned, in that order.
The returned Hit carries the parent piece, so citations, the gate and the drafter are unchanged, and
its ``bm25`` is the parent's BM25 score for the query (the decide step's lexical threshold).

Embeddings: fastembed (ONNX on CPU, no torch), default BAAI/bge-small-en-v1.5 (384-d, 33M params).
Piece vectors are cached in data/retrieval/emb_cache/ keyed by model and a hash of the texts, so the
library is embedded once. fastembed is optional: without it, ``available()`` is False and callers keep BM25.
"""
from __future__ import annotations

import hashlib
import time
from pathlib import Path

from relay.retrieve import BLOCKS, Hit, Piece, Retriever

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "retrieval" / "emb_cache"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
RERANK_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"
RRF_C = 60


def available() -> bool:
    try:
        import fastembed  # noqa: F401
        import numpy  # noqa: F401
        return True
    except ImportError:
        return False


def piece_text(p: Piece) -> str:
    """What is embedded and reranked: the section title gives a passage its topic."""
    return f"{p.section}\n{p.text}"


class DenseIndex:
    _models: dict[str, object] = {}

    def __init__(self, pieces: list[Piece], model: str = EMBED_MODEL) -> None:
        import numpy as np
        self.np, self.pieces, self.model_name = np, pieces, model
        texts = [piece_text(p) for p in pieces]
        key = hashlib.sha256(("\x00".join(texts) + model).encode("utf-8")).hexdigest()[:16]
        path = CACHE / f"{model.replace('/', '_')}-{len(pieces)}-{key}.npy"
        if path.exists():
            self.vecs = np.load(path)
        else:
            vecs = np.array(list(self.model().passage_embed(texts)), dtype="float32")
            self.vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
            CACHE.mkdir(parents=True, exist_ok=True)
            np.save(path, self.vecs)

    def model(self):
        if self.model_name not in DenseIndex._models:
            from fastembed import TextEmbedding
            DenseIndex._models[self.model_name] = TextEmbedding(self.model_name)
        return DenseIndex._models[self.model_name]

    def query_vec(self, query: str):
        v = self.np.array(list(self.model().query_embed([query]))[0], dtype="float32")
        return v / self.np.linalg.norm(v)

    def ranked(self, query: str, papers: list[str] | None = None, top: int = 50) -> list[tuple[int, float]]:
        """[(piece index, cosine)], best first."""
        sims = self.vecs @ self.query_vec(query)
        order = self.np.argsort(-sims)
        out = []
        for i in order:
            if papers and self.pieces[i].paper not in papers:
                continue
            out.append((int(i), float(sims[i])))
            if len(out) >= top:
                break
        return out


class Reranker:
    _models: dict[str, object] = {}

    def __init__(self, model: str = RERANK_MODEL) -> None:
        from fastembed.rerank.cross_encoder import TextCrossEncoder
        if model not in Reranker._models:
            Reranker._models[model] = TextCrossEncoder(model)
        self.m = Reranker._models[model]

    def scores(self, query: str, texts: list[str]) -> list[float]:
        return [float(s) for s in self.m.rerank(query, texts)]


class HybridRetriever:
    """Same interface the pipeline uses from relay.retrieve.Retriever: search, evidence, ranked,
    pieces, size, index (the parent BM25 index), by_paper_order."""

    def __init__(self, child: Retriever, parent: Retriever, dense: DenseIndex | None, reranker: Reranker | None = None,
                 candidates: int = 30, w_bm25: float = 1.0, w_dense: float = 1.0, keep_bm25: int = 0) -> None:
        self.child, self.parent, self.dense, self.reranker = child, parent, dense, reranker
        self.keep_bm25 = keep_bm25      # the parent-level BM25 top n always make the cut (exact-term questions)
        self.candidates, self.w_bm25, self.w_dense = candidates, w_bm25, w_dense
        # interface of the parent retriever (citations, evidence, the gate's rare terms)
        self.pieces, self.size, self.index = parent.pieces, parent.size, parent.index
        self.by_paper_order = parent.by_paper_order
        self._pos = parent._pos
        # child -> parent: the parent of the same section that holds the child's first block
        by_block: dict[str, int] = {}
        for j, p in enumerate(parent.pieces):
            for b in p.block_ids:
                by_block.setdefault(b, j)
        self.parent_of = [by_block.get(c.block_ids[0]) for c in child.pieces]
        self.last_timing_ms: dict[str, float] = {}

    @classmethod
    def build(cls, child_chars: int = 600, parent_chars: int = 2400, dense: bool = True, rerank: bool = False,
              path: Path = BLOCKS, **kw) -> "HybridRetriever":
        parent = Retriever.from_blocks(parent_chars, path)
        child = parent if child_chars == parent_chars else Retriever.from_blocks(child_chars, path)
        return cls(child, parent, DenseIndex(child.pieces) if dense else None, Reranker() if rerank else None, **kw)

    @classmethod
    def from_pieces(cls, child_pieces: list[Piece], parent_pieces: list[Piece], child_chars: int, parent_chars: int,
                    dense: bool = True, rerank: bool = False, **kw) -> "HybridRetriever":
        parent = Retriever(parent_pieces, parent_chars)
        child = parent if child_pieces is parent_pieces else Retriever(child_pieces, child_chars)
        return cls(child, parent, DenseIndex(child.pieces) if dense else None, Reranker() if rerank else None, **kw)

    def _children(self, query: str, papers: list[str] | None, dense_query: str | None = None) -> list[tuple[int, float]]:
        t0 = time.perf_counter()
        n = self.candidates
        bm = self.child.index.ranked(query, papers)[:n]
        pos = self.child._pos
        score: dict[int, float] = {}
        for h in bm:
            i = pos[h.piece.pid]
            score[i] = score.get(i, 0.0) + self.w_bm25 / (RRF_C + h.rank)
        t1 = time.perf_counter()
        if self.dense is not None:
            for r, (i, _) in enumerate(self.dense.ranked(dense_query or query, papers, n), 1):
                score[i] = score.get(i, 0.0) + self.w_dense / (RRF_C + r)
        t2 = time.perf_counter()
        fused = sorted(score.items(), key=lambda x: -x[1])[:n]
        if self.reranker is not None and fused:
            texts = [piece_text(self.child.pieces[i]) for i, _ in fused]
            rs = self.reranker.scores(dense_query or query, texts)
            fused = [x for _, x in sorted(zip(rs, fused), key=lambda z: -z[0])]
        t3 = time.perf_counter()
        self.last_timing_ms = {"bm25": (t1 - t0) * 1e3, "dense": (t2 - t1) * 1e3, "rerank": (t3 - t2) * 1e3}
        return fused

    def search(self, query: str, k: int, papers: list[str] | None = None, use_rerank: bool = False,
               candidates: int | None = None, dense_query: str | None = None) -> list[Hit]:
        """``query`` drives BM25 (keywords); ``dense_query`` (default: query) drives the embedding search
        and the rerank, e.g. the router's hypothetical answer (HyDE)."""
        from relay.retrieve import tokenize
        out, seen = [], set()
        qtok = tokenize(query)
        if self.keep_bm25:
            # Guaranteed lexical slots: with k = 3, fusion pushed out passages BM25 ranked 2nd-3rd that held
            # the facts of exact-vocabulary questions (Q11, Q16 in data/retrieval/v2.md). BM25's top n
            # parents go first; the fused ranking fills the remaining slots.
            for h in self.parent.index.ranked(query, papers)[: min(self.keep_bm25, k)]:
                j = self._pos[h.piece.pid]
                seen.add(j)
                out.append(Hit(h.piece, h.bm25, len(out) + 1, bm25=h.bm25, features={"via": "bm25"}))
        for i, s in self._children(query, papers, dense_query):
            j = self.parent_of[i]
            if j is None or j in seen:
                continue
            seen.add(j)
            p = self.parent.pieces[j]
            out.append(Hit(p, s, len(out) + 1, bm25=self.parent.index.score(j, qtok),
                           features={"child": self.child.pieces[i].pid}))
            if len(out) >= k:
                break
        return out

    def ranked(self, query: str, papers: list[str] | None = None, use_rerank: bool = False,
               candidates: int | None = None) -> list[Hit]:
        return self.search(query, len(self.parent.pieces), papers)

    def evidence(self, query: str, hits: list[Hit], top: int = 3) -> dict:
        ev = self.parent.evidence(query, hits, top)
        if self.dense is not None and hits:
            ev["dense_best"] = round(self.dense_best(query, hits), 3)
        return ev

    def dense_best(self, query: str, hits: list[Hit]) -> float:
        """Highest cosine between the query and any child of the returned parents."""
        q = self.dense.query_vec(query)
        want = {h.piece.pid for h in hits}
        idx = [i for i, j in enumerate(self.parent_of) if j is not None and self.parent.pieces[j].pid in want]
        return float((self.dense.vecs[idx] @ q).max()) if idx else 0.0


def build_retriever(cfg: dict):
    """The retriever a config asks for. [retrieval] mode = "hybrid" (production) needs fastembed; without
    it, or with mode = "bm25" (the submitted setting), it is the BM25 Retriever over piece_chars pieces."""
    rc = cfg["retrieval"]
    if rc.get("mode", "bm25") == "hybrid" and available():
        return HybridRetriever.build(child_chars=int(rc.get("child_chars", rc["piece_chars"])),
                                     parent_chars=int(rc["piece_chars"]), rerank=bool(rc.get("rerank_model")),
                                     candidates=int(rc.get("candidates", 30)), keep_bm25=int(rc.get("keep_bm25", 0)))
    return Retriever.from_blocks(int(rc["piece_chars"]))
