"""Section-aware pieces built from data/blocks.jsonl, BM25 in the stdlib, optional lexical rerank.

    r = Retriever.from_blocks(size=1200)
    hits = r.search("how does H-Net choose chunk boundaries", k=5, papers=["dynamic_chunking"])
    hits[0].piece.cite()   ->  "2507.07955v2 p.6 §2.2.1"

Pieces (``build_pieces``), from what the extraction audit found (PLAN.md §8, blocks_audit.json):
  * A piece never crosses a section, so every citation names one section. Blocks of the same
    section are joined across page breaks (the piece then cites every page it spans).
  * Moshi keeps its subsections (3.4.1 ... 3.4.4) as body lines, not headings. A short body block
    that starts like "3.4.4 Inner Monologue" opens a new subsection, so citations say §3.4.4.
  * A caption is glued to the block after it (Table 1's rows in the Transformer paper are a body
    block right after their caption), and tables are never split: a table's values are useless
    without its header and caption.
  * Equations come out as many tiny blocks (H-Net's equation 4 is five). Greedy packing within a
    section keeps them together; a tiny tail piece is merged into the one before it.
  * Blocks longer than the piece size are split at sentence ends. A single sentence longer than
    the size stays whole.
  * References and front matter are excluded: bibliography entries match questions by title
    words (PLAN.md §8, trap 2).

BM25: Okapi BM25 (k1, b) over lower-cased alphanumeric tokens, light suffix stemming, a small
stopword list. The section title is indexed with each piece of that section.

Rerank (optional, no model call): BM25's top ``candidates`` are rescored with features BM25
ignores: idf-weighted coverage of the query's terms, matched query bigrams, query terms in the
section title, and proximity (the shortest window holding the matched terms). The weights were
set before measuring and were not tuned on the evaluation questions.
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BLOCKS = ROOT / "data" / "blocks.jsonl"
PAPERS = ("attention", "moshi", "dynamic_chunking")
PAPER_TITLES = {"attention": "Attention Is All You Need",
                "moshi": "Moshi: a speech-text foundation model for real-time dialogue",
                "dynamic_chunking": "H-Net: Dynamic Chunking for End-to-End Hierarchical Sequence Modeling"}
EXCLUDE_KINDS = ("reference", "front_matter")

STOPWORDS = frozenset("""a an and are as at be been being but by can could did do does doing for from had has
have how i if in into is it its itself may might more most much no nor not of on once only or other our out
over own same should so some such than that the their them then there these they this those through to too
under until up very was we were what when where which while who whom why will with would you your per
paper papers authors author say says said use used using also does do vs versus""".split())

_TOKEN = re.compile(r"[a-z0-9]+")
_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[])")
_SUBSECTION = re.compile(r"^(\d+(?:\.\d+)+)\s+[A-Z][^\n]{0,80}$")


def stem(w: str) -> str:
    if len(w) <= 3 or w.isdigit():
        return w
    for suf, rep in (("ational", "ate"), ("ization", "ize"), ("iveness", "ive"), ("fulness", "ful"),
                     ("ies", "y"), ("sses", "ss"), ("ing", ""), ("ed", ""), ("es", ""), ("s", ""), ("ly", "")):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            if suf == "s" and w.endswith(("ss", "us", "is")):
                continue
            if suf == "es" and not w.endswith(("ches", "shes", "xes", "sses", "zes")):
                continue
            return w[: len(w) - len(suf)] + rep
    return w


def tokenize(text: str, keep_stopwords: bool = False) -> list[str]:
    toks = _TOKEN.findall(text.lower())
    return [stem(t) for t in toks if keep_stopwords or t not in STOPWORDS]


# ----------------------------------------------------------------------------- pieces
@dataclass(frozen=True)
class Piece:
    pid: str
    paper: str
    arxiv_id: str
    section: str
    pages: tuple[int, ...]
    block_ids: tuple[str, ...]
    text: str
    in_appendix: bool
    kinds: tuple[str, ...]
    order: int                      # position in the paper, for "adjacent piece" checks

    def cite(self) -> str:
        """Citation granularity decided in M1 (PLAN.md G17): arXiv id + page(s) + section."""
        p = self.pages
        pages = f"p.{p[0]}" if p[0] == p[-1] else f"pp.{p[0]}-{p[-1]}"
        m = re.match(r"^([A-Z]?\.?\d+(?:\.\d+)*)\s", self.section + " ")
        sec = f"§{m.group(1)}" if m else self.section
        return f"{self.arxiv_id} {pages} {sec}" + (" (appendix)" if self.in_appendix else "")


@dataclass
class _Unit:
    text: str
    block_ids: list[str]
    pages: list[int]
    kinds: list[str]
    atomic: bool = False


def _split_long(text: str, size: int) -> list[str]:
    sents = _SENT.split(text)
    out, cur = [], ""
    for s in sents:
        if cur and len(cur) + 1 + len(s) > size:
            out.append(cur)
            cur = s
        else:
            cur = f"{cur} {s}" if cur else s
    if cur:
        out.append(cur)
    return out


def build_pieces(blocks: list[dict], size: int, exclude_kinds: tuple[str, ...] = EXCLUDE_KINDS) -> list[Piece]:
    pieces: list[Piece] = []
    groups: list[tuple[dict, str, list[dict]]] = []     # (first block, section label, blocks)
    for b in blocks:
        if b["kind"] in exclude_kinds:
            continue
        sec = b["section"]
        sub = _SUBSECTION.match(b["text"].strip()) if b["kind"] in ("body", "heading") else None
        prev = groups[-1] if groups else None
        if sub and len(b["text"]) < 90:
            groups.append((b, b["text"].strip(), [b]))
            continue
        if prev and prev[0]["paper"] == b["paper"] and prev[0]["section"] == sec:
            prev[2].append(b)
        else:
            groups.append((b, sec, [b]))
    counters: Counter = Counter()
    for first, label, bl in groups:
        units: list[_Unit] = []
        i = 0
        while i < len(bl):
            b = bl[i]
            if b["kind"] == "heading":
                i += 1                              # the section label carries it
                continue
            if b["kind"] == "caption" and i + 1 < len(bl) and bl[i + 1]["kind"] in ("table", "body"):
                nb = bl[i + 1]
                units.append(_Unit(b["text"] + "\n" + nb["text"], [b["block_id"], nb["block_id"]],
                                   [b["page"], nb["page"]], [b["kind"], nb["kind"]], atomic=True))
                i += 2
                continue
            if b["kind"] == "table" or len(b["text"]) <= size:
                units.append(_Unit(b["text"], [b["block_id"]], [b["page"]], [b["kind"]], atomic=b["kind"] == "table"))
            else:
                for part in _split_long(b["text"], size):
                    units.append(_Unit(part, [b["block_id"]], [b["page"]], [b["kind"]]))
            i += 1
        packed: list[_Unit] = []
        for u in units:
            cur = packed[-1] if packed else None
            if cur and len(cur.text) + 1 + len(u.text) <= size:
                cur.text += "\n" + u.text
                cur.block_ids += [x for x in u.block_ids if x not in cur.block_ids]
                cur.pages += u.pages
                cur.kinds += u.kinds
            else:
                packed.append(_Unit(u.text, list(u.block_ids), list(u.pages), list(u.kinds), u.atomic))
        if len(packed) >= 2 and len(packed[-1].text) < size // 4:      # tiny tail -> previous piece
            t = packed.pop()
            packed[-1].text += "\n" + t.text
            packed[-1].block_ids += [x for x in t.block_ids if x not in packed[-1].block_ids]
            packed[-1].pages += t.pages
            packed[-1].kinds += t.kinds
        for u in packed:
            paper = first["paper"]
            n = counters[paper]
            counters[paper] += 1
            pieces.append(Piece(pid=f"{paper}:{size}:{n:04d}", paper=paper, arxiv_id=first["arxiv_id"],
                                section=label, pages=tuple(sorted(set(u.pages))), block_ids=tuple(u.block_ids),
                                text=u.text, in_appendix=bool(first.get("in_appendix")),
                                kinds=tuple(dict.fromkeys(u.kinds)), order=n))
    return pieces


# ----------------------------------------------------------------------------- BM25
@dataclass
class Hit:
    piece: Piece
    score: float
    rank: int
    bm25: float = 0.0
    features: dict = field(default_factory=dict)


class BM25:
    def __init__(self, pieces: list[Piece], k1: float = 1.2, b: float = 0.75) -> None:
        self.pieces, self.k1, self.b = pieces, k1, b
        self.docs = [tokenize(p.section + " " + p.text) for p in pieces]
        self.tf = [Counter(d) for d in self.docs]
        self.len = [len(d) for d in self.docs]
        self.avgdl = sum(self.len) / max(1, len(self.len))
        df: Counter = Counter()
        for c in self.tf:
            df.update(c.keys())
        n = len(pieces)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def score(self, i: int, q: list[str]) -> float:
        tf, dl, s = self.tf[i], self.len[i], 0.0
        for t in q:
            f = tf.get(t)
            if f:
                s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
        return s

    def ranked(self, query: str, papers: list[str] | None = None) -> list[Hit]:
        q = tokenize(query)
        out = []
        for i, p in enumerate(self.pieces):
            if papers and p.paper not in papers:
                continue
            s = self.score(i, q)
            if s > 0:
                out.append((s, i))
        out.sort(key=lambda x: (-x[0], x[1]))
        return [Hit(self.pieces[i], s, r + 1, bm25=s) for r, (s, i) in enumerate(out)]


def _idf_weight(idf: dict, terms) -> float:
    return sum(idf.get(t, 0.0) for t in terms)


def rerank(index: BM25, query: str, hits: list[Hit], index_of: dict[str, int], weights: dict | None = None) -> list[Hit]:
    """Lexical rerank of BM25 candidates (no model call). Weights fixed a priori."""
    w = {"bm25": 1.0, "coverage": 0.6, "bigrams": 0.4, "heading": 0.3, "proximity": 0.3, **(weights or {})}
    q = tokenize(query)
    qset = list(dict.fromkeys(q))
    qbig = {(a, c) for a, c in zip(q, q[1:])}
    total_idf = _idf_weight(index.idf, qset) or 1.0
    top = max((h.bm25 for h in hits), default=1.0) or 1.0
    out = []
    for h in hits:
        doc = index.docs[index_of[h.piece.pid]]
        dset = set(doc)
        matched = [t for t in qset if t in dset]
        coverage = _idf_weight(index.idf, matched) / total_idf
        dbig = set(zip(doc, doc[1:]))
        bigrams = len(qbig & dbig) / len(qbig) if qbig else 0.0
        head = set(tokenize(h.piece.section))
        heading = _idf_weight(index.idf, [t for t in qset if t in head]) / total_idf
        proximity = 0.0
        if len(matched) >= 2:
            pos = [(j, t) for j, t in enumerate(doc) if t in matched]
            need, best, cnt, left = len(matched), None, Counter(), 0
            for right in range(len(pos)):
                cnt[pos[right][1]] += 1
                while len(cnt) == need:
                    span = pos[right][0] - pos[left][0] + 1
                    best = span if best is None or span < best else best
                    cnt[pos[left][1]] -= 1
                    if not cnt[pos[left][1]]:
                        del cnt[pos[left][1]]
                    left += 1
            proximity = need / best if best else 0.0
        f = {"bm25": h.bm25 / top, "coverage": coverage, "bigrams": bigrams, "heading": heading, "proximity": proximity}
        out.append(Hit(h.piece, sum(w[k] * v for k, v in f.items()), 0, bm25=h.bm25,
                       features={k: round(v, 3) for k, v in f.items()}))
    out.sort(key=lambda x: -x.score)
    for r, x in enumerate(out, 1):
        x.rank = r
    return out


class Retriever:
    def __init__(self, pieces: list[Piece], size: int, k1: float = 1.2, b: float = 0.75) -> None:
        self.size = size
        self.pieces = pieces
        self.index = BM25(pieces, k1, b)
        self._pos = {p.pid: i for i, p in enumerate(pieces)}
        self.by_paper_order = {(p.paper, p.order): p for p in pieces}

    @classmethod
    def from_blocks(cls, size: int, path: Path = BLOCKS, **kw) -> "Retriever":
        with path.open(encoding="utf-8") as f:
            blocks = [json.loads(x) for x in f if x.strip()]
        return cls(build_pieces(blocks, size), size, **kw)

    def search(self, query: str, k: int, papers: list[str] | None = None, use_rerank: bool = False,
               candidates: int = 20) -> list[Hit]:
        hits = self.index.ranked(query, papers)
        if use_rerank:
            hits = self._rerank(query, hits[:candidates]) + hits[candidates:]
            for r, h in enumerate(hits, 1):
                h.rank = r
        return hits[:k]

    def ranked(self, query: str, papers: list[str] | None = None, use_rerank: bool = False,
               candidates: int = 20) -> list[Hit]:
        """The full ranking (for measuring where a passage ranks)."""
        return self.search(query, len(self.pieces), papers, use_rerank, candidates)

    def _rerank(self, query: str, hits: list[Hit]) -> list[Hit]:
        return rerank(self.index, query, hits, self._pos)

    def evidence(self, query: str, hits: list[Hit], top: int = 3) -> dict:
        """How well the best pieces cover the query's content terms, idf-weighted, 0..1.
        ``best``: the single best piece; ``union``: the top ``top`` pieces together."""
        q = list(dict.fromkeys(tokenize(query)))
        total = _idf_weight(self.index.idf, q)
        if not q or not hits or total <= 0:
            return {"best": 0.0, "union": 0.0, "top_bm25": 0.0, "query_terms": len(q)}
        sets = [set(self.index.docs[self._pos[h.piece.pid]]) for h in hits[:top]]
        best = max(_idf_weight(self.index.idf, [t for t in q if t in s]) for s in sets) / total
        union = _idf_weight(self.index.idf, [t for t in q if any(t in s for s in sets)]) / total
        return {"best": round(best, 3), "union": round(union, 3), "top_bm25": round(hits[0].bm25, 3),
                "query_terms": len(q)}
