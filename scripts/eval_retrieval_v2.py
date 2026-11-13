"""Retrieval v2: BM25 vs embeddings vs hybrid vs small-to-big vs cross-encoder rerank. No model calls.

    python scripts/eval_retrieval_v2.py      ->  data/retrieval/v2.md, data/retrieval/v2.json

Question sets (raw question as the query, all three papers searched, k parents = what the drafter reads):
  original  the 14 answerable questions of data/questions.jsonl (gold = the facts' evidence blocks)
  dev       32 more in data/retrieval/dev_questions.jsonl, written in plain engineer language on purpose
            (not the papers' wording), each with the blocks that answer it. Written before any dense
            run was measured; the author of the questions saw the passages, which biases toward findable.
Metrics:
  hit@k        a gold block is inside one of the top-k pieces
  recall@k     original set: share of gold facts with an evidence block in the top k
  MRR          1 / rank of the first piece holding a gold block (0 if not in the top 20)
  ctx chars    characters of the top-k pieces (the drafter's prompt size)
  query ms     p50 time of one search on this CPU (embedding the query, rerank included)
Gate calibration: dense_best (cosine of the best child) for answerable, unanswerable and off-topic
questions, to set the decide step's semantic threshold.
"""
from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from relay.dense import HybridRetriever  # noqa: E402
from relay.retrieve import Retriever  # noqa: E402

OUT = ROOT / "data" / "retrieval"
OFF_TOPIC = ["What is the capital of France?", "How do I bake sourdough bread at home?",
             "Who won the 2022 football World Cup?", "Which Python web framework is the fastest?"]
K = 3


def load():
    qs = [json.loads(x) for x in (ROOT / "data" / "questions.jsonl").open(encoding="utf-8") if x.strip()]
    orig = [{"qid": q["qid"], "question": q["question"], "paper": q["paper"],
             "facts": [{e["block_id"] for e in f["evidence"]} for f in q["facts"]]}
            for q in qs if q["answerable"]]
    unans = [q["question"] for q in qs if not q["answerable"]]
    dev = [json.loads(x) for x in (OUT / "dev_questions.jsonl").open(encoding="utf-8") if x.strip()]
    for d in dev:
        d["facts"] = [set(d["gold_blocks"])]
    return orig, dev, unans


def variants():
    from relay.dense import DenseIndex, Reranker
    p2400, p600 = Retriever.from_blocks(2400), Retriever.from_blocks(600)
    d2400, d600 = DenseIndex(p2400.pieces), DenseIndex(p600.pieces)
    rr = Reranker()

    def hy(child, dense, rerank=False, **kw):
        return HybridRetriever(child, p2400, dense, rr if rerank else None, **kw)
    yield "bm25 2400 (submitted)", p2400, 3
    yield "dense 2400", hy(p2400, d2400, w_bm25=0.0), 3
    yield "hybrid 2400", hy(p2400, d2400), 3
    yield "small-to-big bm25 600->2400", hy(p600, None), 3
    yield "small-to-big dense 600->2400", hy(p600, d600, w_bm25=0.0), 3
    yield "small-to-big hybrid 600->2400", hy(p600, d600), 3
    yield "small-to-big hybrid + rerank", hy(p600, d600, True), 3
    yield "hybrid 2400 + rerank", hy(p2400, d2400, True), 3
    yield "bm25 2400, k=5", p2400, 5
    yield "small-to-big hybrid, k=5", hy(p600, d600), 5
    yield "production: BM25 top 3 kept + small-to-big hybrid, k=5", hy(p600, d600, keep_bm25=3), 5


def score(r, qset, K=3):
    rows = []
    for q in qset:
        t0 = time.perf_counter()
        hits = r.search(q["question"], 20)
        ms = (time.perf_counter() - t0) * 1e3
        top = hits[:K]
        first = next((h.rank for h in hits if any(set(h.piece.block_ids) & f for f in q["facts"])), None)
        got = [any(set(h.piece.block_ids) & f for h in top) for f in q["facts"]]
        rows.append({"qid": q["qid"], "hit": any(got), "recall": sum(got) / len(got), "first": first,
                     "rr": (1 / first) if first else 0.0, "ms": ms,
                     "ctx": sum(len(h.piece.text) for h in top),
                     "top": [h.piece.cite() for h in top]})
    return rows


def main() -> int:
    orig, dev, unans = load()
    results = []
    for name, r, k in variants():
        r.search("warm up", 3)                          # load models outside the timing
        ro, rd = score(r, orig, k), score(r, dev, k)
        allr = ro + rd
        s = {"variant": name, "k": k,
             "orig_hit": sum(x["hit"] for x in ro) / len(ro), "orig_recall": statistics.mean(x["recall"] for x in ro),
             "dev_hit": sum(x["hit"] for x in rd) / len(rd),
             "all_hit": sum(x["hit"] for x in allr) / len(allr), "mrr": statistics.mean(x["rr"] for x in allr),
             "ctx_p50": statistics.median(x["ctx"] for x in allr), "ms_p50": statistics.median(x["ms"] for x in allr),
             "q14": next(x for x in ro if x["qid"] == "Q14")["first"],
             "missed": [x["qid"] for x in allr if not x["hit"]]}
        if hasattr(r, "dense") and r.dense is not None:
            def db(qs):
                return [round(r.dense_best(q, r.search(q, 3)), 3) for q in qs]
            s["dense_best"] = {"answerable": db([q["question"] for q in orig + dev]), "unanswerable": db(unans),
                               "off_topic": db(OFF_TOPIC)}
        results.append((s, ro, rd))
        print(f"{name:34s} k={k} all hit {s['all_hit']:.0%}  orig {s['orig_hit']:.0%}  dev {s['dev_hit']:.0%}  "
              f"MRR {s['mrr']:.2f}  Q14 rank {s['q14']}  {s['ms_p50']:.0f} ms  missed {s['missed']}", flush=True)

    L = ["# Retrieval v2: BM25, embeddings, hybrid, small-to-big, rerank", "",
         f"Generated by `scripts/eval_retrieval_v2.py` (no model calls). Raw question as the query, all papers, "
         f"top k parent pieces of 2400 chars (what the drafter reads; k=3 unless stated). {len(orig)} original + {len(dev)} dev "
         f"answerable questions = {len(orig) + len(dev)}. One question is {100 / (len(orig) + len(dev)):.1f} points.", "",
         "| variant | k | hit@k all | hit@k original (14) | fact recall@k original | hit@k dev (32) | MRR@20 | Q14 first gold rank | "
         "context chars p50 | query ms p50 | missed |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for s, _, _ in results:
        L.append(f"| {s['variant']} | {s['k']} | {s['all_hit']:.0%} | {s['orig_hit']:.0%} | {s['orig_recall']:.0%} | {s['dev_hit']:.0%} | "
                 f"{s['mrr']:.2f} | {s['q14'] or 'not in top 20'} | {s['ctx_p50']:.0f} | {s['ms_p50']:.0f} | "
                 f"{', '.join(s['missed']) or '-'} |")
    L += ["", "## Semantic evidence for the decide step (cosine of the best child)", "",
          "| variant | answerable min / p10 / p50 | unanswerable (each) | off-topic (each) |", "|---|---|---|---|"]
    for s, _, _ in results:
        d = s.get("dense_best")
        if not d:
            continue
        a = sorted(d["answerable"])
        L.append(f"| {s['variant']} | {a[0]} / {a[len(a) // 10]} / {a[len(a) // 2]} | "
                 f"{', '.join(map(str, d['unanswerable']))} | {', '.join(map(str, d['off_topic']))} |")
    (OUT / "v2.md").write_text("\n".join(L) + "\n", encoding="utf-8", newline="\n")
    (OUT / "v2.json").write_text(json.dumps([{"summary": s, "original": ro, "dev": rd} for s, ro, rd in results],
                                            indent=1) + "\n", encoding="utf-8", newline="\n")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    sys.exit(main())
