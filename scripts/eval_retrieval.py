"""Gold-in-top-k for the retrieval settings, no model calls.

    .venv\\Scripts\\python scripts\\eval_retrieval.py   ->  data/retrieval/topk.json, data/retrieval/topk.md

Query = the raw question (no rewrite), all three papers searched. A gold fact counts as retrieved
when one of its evidence quotes (data/questions.jsonl) is inside a top-k piece that holds the
quote's block; a quote cut by a piece boundary counts if both neighbouring pieces are in the top k.
  gold hit@k      share of answerable questions with at least one gold fact in the top k
  all facts@k     share with every gold fact in the top k
  fact recall@k   mean share of gold facts in the top k
  first gold rank where the first piece holding any gold fact ranks in the full ranking (median)
  right paper@1   share whose top piece is from the question's paper
  context chars   characters of the top k pieces (what the drafter would read; tokens ~ chars/4)
Evidence strength (relay.retrieve.Retriever.evidence) is also reported for answerable,
unanswerable and four off-topic controls, to place the "too weak to draft" threshold.
"""
from __future__ import annotations

import json
import re
import tomllib
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from relay.retrieve import Hit, Retriever  # noqa: E402

QUESTIONS = ROOT / "data" / "questions.jsonl"
PIPELINE_CONFIG = ROOT / "config" / "pipeline.toml"
OUT_DIR = ROOT / "data" / "retrieval"
SETTINGS = [  # (name, piece chars, k, rerank)
    ("small", 600, 8, False),
    ("medium", 1200, 5, False),
    ("large", 2400, 3, False),
    ("medium+rerank", 1200, 4, True),
]
OFF_TOPIC = ["What is the capital of France?", "How do I bake sourdough bread at home?",
             "Who won the 2022 football World Cup?", "Which Python web framework is the fastest?"]
_WS = re.compile(r"\s+")


def norm(s: str) -> str:
    return _WS.sub(" ", s).strip()


def load_questions() -> list[dict]:
    with QUESTIONS.open(encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def quote_in(hits: list[Hit], ev: dict, r: Retriever) -> bool:
    q = norm(ev["quote"])
    top = {h.piece.pid for h in hits}
    holders = [h.piece for h in hits if ev["block_id"] in h.piece.block_ids]
    if any(q in norm(p.text) for p in holders):
        return True
    for p in holders:                       # cut by a piece boundary: try the neighbour
        nxt = r.by_paper_order.get((p.paper, p.order + 1))
        if nxt and nxt.pid in top and q in norm(p.text + " " + nxt.text):
            return True
    return False


def fact_in(hits, fact, r) -> bool:
    return any(quote_in(hits, ev, r) for ev in fact["evidence"])


def pct(xs, q):
    xs = sorted(xs)
    if not xs:
        return None
    import math
    return xs[max(0, math.ceil(q * len(xs)) - 1)]


def run_setting(name, size, k, use_rerank, questions):
    r = Retriever.from_blocks(size)
    rows, strength = [], {"answerable": [], "unanswerable": [], "off_topic": []}
    for q in questions:
        full = r.ranked(q["question"], use_rerank=use_rerank)
        top = full[:k]
        ev = r.evidence(q["question"], top)
        strength["answerable" if q["answerable"] else "unanswerable"].append({"qid": q["qid"], **ev})
        if not q["answerable"]:
            continue
        facts = q["facts"]
        got = [fact_in(top, f, r) for f in facts]
        first = None
        for h in full:
            if any(fact_in([h], f, r) for f in facts):
                first = h.rank
                break
        rows.append({"qid": q["qid"], "paper": q["paper"], "trap": q.get("trap") or "",
                     "facts": len(facts), "facts_in_top_k": sum(got), "gold_hit": any(got), "all_facts": all(got),
                     "first_gold_rank": first, "top1_paper": top[0].piece.paper if top else None,
                     "top_k": [{"rank": h.rank, "cite": h.piece.cite(), "pid": h.piece.pid, "score": round(h.score, 3)}
                               for h in top],
                     "context_chars": sum(len(h.piece.text) for h in top)})
    for c in OFF_TOPIC:
        strength["off_topic"].append({"query": c, **r.evidence(c, r.search(c, k))})
    n = len(rows)
    ranks = [x["first_gold_rank"] for x in rows if x["first_gold_rank"]]
    summary = {"setting": name, "piece_chars": size, "k": k, "rerank": use_rerank, "pieces": len(r.pieces),
               "piece_chars_mean": round(statistics.mean(len(p.text) for p in r.pieces)),
               "questions": n,
               "gold_hit_at_k": round(sum(x["gold_hit"] for x in rows) / n, 3),
               "all_facts_at_k": round(sum(x["all_facts"] for x in rows) / n, 3),
               "fact_recall_at_k": round(statistics.mean(x["facts_in_top_k"] / x["facts"] for x in rows), 3),
               "right_paper_at_1": round(sum(x["top1_paper"] == x["paper"] for x in rows) / n, 3),
               "first_gold_rank_p50": pct(ranks, 0.5), "first_gold_rank_max": max(ranks) if ranks else None,
               "never_found": [x["qid"] for x in rows if not x["first_gold_rank"]],
               "missed_at_k": [x["qid"] for x in rows if not x["gold_hit"]],
               "context_chars_p50": pct([x["context_chars"] for x in rows], 0.5),
               "context_chars_max": max(x["context_chars"] for x in rows)}
    return summary, rows, strength


def render(results) -> str:
    L = ["# Retrieval: gold in top k", "",
         "Generated by `scripts/eval_retrieval.py` (no model calls). Query = raw question, all papers, "
         f"{results[0][0]['questions']} answerable questions from `data/questions.jsonl`. "
         "Percentiles are nearest-rank.", "",
         "| setting | piece chars | k | pieces | gold hit@k | all facts@k | fact recall@k | right paper@1 | "
         "first gold rank p50 / max | context chars p50 / max | missed at k |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for s, _, _ in results:
        L.append(f"| {s['setting']} | {s['piece_chars']} | {s['k']} | {s['pieces']} | {s['gold_hit_at_k']:.0%} | "
                 f"{s['all_facts_at_k']:.0%} | {s['fact_recall_at_k']:.0%} | {s['right_paper_at_1']:.0%} | "
                 f"{s['first_gold_rank_p50']} / {s['first_gold_rank_max']} | {s['context_chars_p50']} / "
                 f"{s['context_chars_max']} | {', '.join(s['missed_at_k']) or '-'} |")
    L += ["", "## Per question (facts in top k / gold facts; first gold rank)", "",
          "| qid | trap | " + " | ".join(s["setting"] for s, _, _ in results) + " |",
          "|---|---|" + "---|" * len(results)]
    by = [{x["qid"]: x for x in rows} for _, rows, _ in results]
    for qid in by[0]:
        cells = [f"{b[qid]['facts_in_top_k']}/{b[qid]['facts']}; r{b[qid]['first_gold_rank']}" for b in by]
        L.append(f"| {qid} | {by[0][qid]['trap']} | " + " | ".join(cells) + " |")
    with PIPELINE_CONFIG.open("rb") as f:
        dc = tomllib.load(f)["decide"]
    tb, tm = dc["refuse_if_best_below"], dc["refuse_if_top_bm25_below"]

    def gated(x):
        return x["best"] < tb or x["top_bm25"] < tm
    L += ["", "## Evidence strength (idf-weighted share of question terms in the best piece; top BM25 score)", "",
          f"The no-model refusal in `config/pipeline.toml` [decide] fires when best < {tb} or top BM25 < {tm}.", "",
          "| setting | answerable best min / p50 | answerable top BM25 min | unanswerable best min / max | "
          "off-topic best (each) | refused by the gate: answerable / unanswerable / off-topic |",
          "|---|---|---|---|---|---|"]
    for s, _, st in results:
        a = [x["best"] for x in st["answerable"]]
        ab = [x["top_bm25"] for x in st["answerable"]]
        u = [x["best"] for x in st["unanswerable"]]
        o = [x["best"] for x in st["off_topic"]]
        g = [sum(gated(x) for x in st[k]) for k in ("answerable", "unanswerable", "off_topic")]
        L.append(f"| {s['setting']} | {min(a)} / {pct(a, 0.5)} | {min(ab)} | {min(u)} / {max(u)} | "
                 f"{', '.join(str(x) for x in o)} | {g[0]}/{len(a)} / {g[1]}/{len(u)} / {g[2]}/{len(o)} |")
    return "\n".join(L) + "\n"


def main() -> int:
    questions = load_questions()
    results = [run_setting(*s, questions) for s in SETTINGS]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = {"_generated_by": "scripts/eval_retrieval.py", "settings": [s for s, _, _ in results],
           "per_question": {s["setting"]: rows for s, rows, _ in results},
           "evidence_strength": {s["setting"]: st for s, _, st in results}}
    (OUT_DIR / "topk.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    md = render(results)
    (OUT_DIR / "topk.md").write_text(md, encoding="utf-8", newline="\n")
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
