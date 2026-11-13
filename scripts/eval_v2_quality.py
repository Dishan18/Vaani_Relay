"""Answer quality of the production pipeline on the 18 graded questions, on real Groq.

    python scripts/eval_v2_quality.py [--real] [--settings v2,v2_rerank]
        ->  data/experiments/v2_quality/<run>.jsonl (every reply), result.md

Same conditions as Table 3's "large" row, so the rows compare: drafter fixed (gpt-oss-120b), LLM
verifier off, the ladder waits for budget instead of degrading, graded by relay.grade against the gold
facts with no model call. The claims are streamed (on_claim), so first_token_ms is the time to the first
verified sentence. Differences from the Table 3 run that a reader should know: a different day (model
and queue variance), and the grader's overrides only cover the old answers, so every grade here is
the auto-grader's (unsure ones listed).
"""
from __future__ import annotations

import argparse
import asyncio
import copy
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from relay.dense import build_retriever  # noqa: E402
from relay.grade import grade, load_questions  # noqa: E402
from relay.limiter import Limiter  # noqa: E402
from relay.pipeline import Pipeline, load_production_config  # noqa: E402
from relay.retrieve import Retriever  # noqa: E402

OUT = ROOT / "data" / "experiments" / "v2_quality"
SETTINGS = {
    "v2": {},                                                         # hybrid small-to-big + HyDE + claims
    "v2_rerank": {"retrieval": {"rerank_model": "Xenova/ms-marco-MiniLM-L-6-v2"}},
    "v2_prose": {"draft": {"format": "text"}},                        # new retrieval, the submitted prose + gate
}


def config(name: str, drafter: str = "openai/gpt-oss-120b") -> dict:
    c = copy.deepcopy(load_production_config())
    for k, v in SETTINGS[name].items():
        c[k] = {**c.get(k, {}), **v}
    c["models"]["draft"] = [drafter]
    c["steps"] = {"router": True, "verifier": False}
    c["ladder"].update({"preferred_free_within_s": 600.0, "router_free_within_s": 600.0,
                        "max_wait_s": 600.0, "deadline_s": 900.0})
    return c


async def run(names: list[str], real: bool, drafter: str = "openai/gpt-oss-120b") -> int:
    qs = load_questions()
    pieces = {p.pid: {"text": p.text, "paper": p.paper, "block_ids": list(p.block_ids)}
              for p in Retriever.from_blocks(2400).pieces}
    OUT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    summary = []
    lim = Limiter.from_probe()
    if not real:
        raise SystemExit("quality needs real model text: run with --real (the stand-in has no answers for new prompts)")
    from relay.real import open_real
    raw: list[dict] = []
    with open_real(scope="eval", allow_real=True) as bc:
        async def chat(payload, *, phase, timeouts=None, on_delta=None):
            rec = await bc.stream_chat(payload, phase=phase, timeouts=timeouts, on_delta=on_delta)
            raw.append({"phase": phase, "model": payload["model"], "status": rec.status, "text": rec.text})
            return rec
        for name in names:
            cfg = config(name, drafter)
            name = f"{name}@{drafter.split('/')[-1]}"
            pipe = Pipeline(chat, lim, build_retriever(cfg), cfg)
            path = OUT / f"{name}-{stamp}.jsonl"
            rows = []
            with path.open("w", encoding="utf-8", newline="\n") as f:
                for qid, q in qs.items():
                    t0 = time.perf_counter()
                    raw.clear()
                    reply = await pipe.answer(q["question"], request_id=f"{name}-{stamp}-{qid}", on_claim=lambda c: None)
                    d = reply.to_dict()
                    g = grade(q, d, pieces)
                    row = {"setting": name, "qid": qid, "kind": reply.kind, "level": reply.level,
                           "grade": g["label"], "sure": g["sure"], "reasons": g["reasons"],
                           "first_token_ms": reply.first_token_ms, "total_ms": reply.released_ms,
                           "budget_wait_ms": reply.budget_wait_ms, "tokens": reply.tokens_in + reply.tokens_out,
                           "dropped": len(reply.checks.get("dropped") or []), "text": reply.text,
                           "claims": reply.claims, "dropped_detail": reply.checks.get("dropped") or [],
                           "raw_drafts": [x["text"] for x in raw if x["phase"] == "draft"],
                           "wall_s": round(time.perf_counter() - t0, 2)}
                    f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
                    f.flush()
                    rows.append(row)
                    print(f"  {name:10s} {qid:4s} {reply.kind:9s} {g['label'] or '-':18s} sure={g['sure']} "
                          f"first {reply.first_token_ms:.0f} ms total {reply.released_ms:.0f} ms "
                          f"tokens {row['tokens']} dropped {row['dropped']}", flush=True)
            n = len(rows)

            def pct(xs, p):
                xs = sorted(xs)
                return xs[max(0, -(-int(p * len(xs) * 100) // 100) - 1)] if xs else None
            lab = [r["grade"] for r in rows]
            work_first = [r["first_token_ms"] - r["budget_wait_ms"] for r in rows if r["first_token_ms"] is not None]
            work_total = [r["total_ms"] - r["budget_wait_ms"] for r in rows]
            summary.append({"setting": name, "n": n, "correct": lab.count("correct"), "wrong": lab.count("wrong"),
                            "refused_wrongly": lab.count("refused_wrongly"),
                            "refused_correctly": lab.count("refused_correctly"),
                            "unsure": sum(1 for r in rows if not r["sure"]),
                            "tokens": statistics.mean(r["tokens"] for r in rows),
                            "first_work_p50": statistics.median(work_first),
                            "total_work_p50": statistics.median(work_total),
                            "dropped_claims": sum(r["dropped"] for r in rows), "file": path.name,
                            "per_q": {r["qid"]: r["grade"] for r in rows}})
    L = ["# Production pipeline: answer quality on the 18 graded questions (real Groq)", "",
         f"Generated by `scripts/eval_v2_quality.py`. Drafter {drafter}, LLM verifier off, waits for budget "
         "(as Table 3). Auto-grades only (relay.grade); unsure ones need a look. Latency columns exclude the "
         "budget wait (the eval waits on purpose), so they show the work: router + retrieval + draft + checks.", "",
         "| setting | correct and cited % | wrong % (incl. wrongly refused) | correctly refused % | unsure | "
         "tokens per answer | first verified sentence p50 (ms, excl. budget wait) | complete p50 (ms, excl. wait) | "
         "claims withheld |", "|---|---|---|---|---|---|---|---|---|",
         "| submitted with gpt-oss-120b: large, BM25, prose + gate (Table 3; auto-grades 66.7) | 72.2 | 5.6 | 22.2 | 0 | 1948 | = complete | - | - |",
         "| submitted with gpt-oss-20b (eval:drafter; auto-grades 44.4) | 38.9 | 38.9 | 22.2 | - | - | = complete | - | - |"]
    for s in summary:
        n = s["n"]
        L.append(f"| {s['setting']} | {100 * s['correct'] / n:.1f} | {100 * (s['wrong'] + s['refused_wrongly']) / n:.1f} | "
                 f"{100 * s['refused_correctly'] / n:.1f} | {s['unsure']} | {s['tokens']:.0f} | {s['first_work_p50']:.0f} | "
                 f"{s['total_work_p50']:.0f} | {s['dropped_claims']} |")
    L += ["", "Per question:", "", "| qid | " + " | ".join(s["setting"] for s in summary) + " |",
          "|---|" + "---|" * len(summary)]
    for qid in qs:
        L.append(f"| {qid} | " + " | ".join(str(s["per_q"].get(qid)) for s in summary) + " |")
    (OUT / f"result-{stamp}.md").write_text("\n".join(L) + "\n", encoding="utf-8", newline="\n")
    print("\n".join(L))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--settings", default="v2")
    ap.add_argument("--drafter", default="openai/gpt-oss-120b")
    a = ap.parse_args()
    return asyncio.run(run(a.settings.split(","), a.real, a.drafter))


if __name__ == "__main__":
    sys.exit(main())
