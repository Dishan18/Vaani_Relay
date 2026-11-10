"""Retrieval evaluation: gold-in-top-k offline, and answer quality through the real pipeline.

    python scripts/eval_retrieval.py [topk]                       offline, no model calls
        ->  data/retrieval/topk.json, data/retrieval/topk.md
    python scripts/eval_retrieval.py estimate                      requests/tokens vs the ledger caps
        ->  results/quality/estimate.json (+ printed)
    python scripts/eval_retrieval.py answers [--settings ...] [--drafter M] [--real]
        all 18 questions through the pipeline per setting, drafter fixed, verifier off
        ->  results/raw/answers/<run_id>.jsonl (+ .calls.jsonl: every model call, as the ledger saw it)
    python scripts/eval_retrieval.py drafters --setting S [--drafters a,b] [--real]
        the same at one setting for each drafter (run in parallel: limits are per model)
Without --real the pipeline talks to the stand-in (relay/fake_groq.py) and writes under
--out-dir (required), never to results/ or the real ledger.

Offline gold-in-top-k (topk):

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


def topk_main() -> int:
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


# ============================================================================ the real pipeline
# name -> (piece chars, k, rerank, router). Drafter fixed and verifier off, so only retrieval changes.
ANSWER_SETTINGS = {
    "small": (600, 8, False, True),
    "medium": (1200, 5, False, True),
    "large": (2400, 3, False, True),
    "medium+rerank": (1200, 4, True, True),
    "small_lean": (600, 4, False, False),
}
RAW_ANSWERS = ROOT / "results" / "raw" / "answers"
EST_OUT = ROOT / "results" / "quality" / "estimate.json"
EVAL_SCOPE = "eval"
# Completion tokens per call for the estimate (measured: data/probe/probe_raw.jsonl, P6 drafts with
# 4 passages: gpt-oss-120b 166-314, gpt-oss-20b 105, qwen 130-212; router-sized calls 15-80).
# Expected = a little above the largest seen; worst = max_tokens.
EXPECTED_COMPLETION = {"draft": 400, "understand": 120}
REDRAFT_RATE_EXPECTED = 0.25     # (assumption) share of drafts that fail the gate once


def eval_config(setting: str, drafter: str) -> dict:
    import copy
    from relay.pipeline import load_config
    size, k, rerank, router = ANSWER_SETTINGS[setting]
    c = copy.deepcopy(load_config())
    c["retrieval"].update({"piece_chars": size, "k_full": k, "k_lean": k, "rerank": rerank})
    c["models"]["draft"] = [drafter]
    c["steps"] = {"router": router, "verifier": False}
    # quality, not latency: wait for budget rather than degrade (a degraded reply is flagged)
    c["ladder"].update({"preferred_free_within_s": 600.0, "router_free_within_s": 600.0,
                        "max_wait_s": 600.0, "deadline_s": 900.0})
    return c


def _git_commit() -> str | None:
    import subprocess
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True, timeout=10).stdout.strip() or None
    except Exception:
        return None


def estimate(settings: list[str], step2_drafter: str, step3_drafters: list[str]) -> dict:
    """Requests and tokens per model for step 2 (every setting, one drafter) and step 3 (the
    winning setting, other drafters; the most expensive setting is assumed), against the ledger."""
    from relay.ledger import Ledger, load_budget
    from relay.limiter import Limiter
    from relay.pipeline import ROUTER_SYSTEM, draft_messages, load_config
    lim = Limiter.from_probe()
    qs = load_questions()
    router_model = load_config()["models"]["understand"][0]
    mx = load_config()["max_tokens"]
    per_setting = {}
    for name in settings:
        size, k, rerank, router = ANSWER_SETTINGS[name]
        r = Retriever.from_blocks(size)
        rows = []
        for q in qs:
            hits = r.search(q["question"], k, use_rerank=rerank)
            msgs = draft_messages(q["question"], hits)
            rmsgs = [{"role": "system", "content": ROUTER_SYSTEM}, {"role": "user", "content": q["question"]}]
            rows.append({"draft_prompt": {m: lim.estimate(m, msgs).tokens for m in [step2_drafter] + step3_drafters},
                         "router_prompt": lim.estimate(router_model, rmsgs).tokens if router else 0})
        per_setting[name] = {"router": router, "questions": len(rows),
                             "draft_prompt_mean": {m: round(sum(x["draft_prompt"][m] for x in rows) / len(rows))
                                                   for m in rows[0]["draft_prompt"]},
                             "router_prompt_mean": round(sum(x["router_prompt"] for x in rows) / len(rows)),
                             "rows": rows}

    def cost(name, drafter, worst):
        s = per_setting[name]
        n = s["questions"]
        comp = mx["draft"] if worst else EXPECTED_COMPLETION["draft"]
        drafts = 2.0 if worst else 1.0 + REDRAFT_RATE_EXPECTED
        d_tok = sum(x["draft_prompt"][drafter] + comp for x in s["rows"]) * drafts
        out = {drafter: {"requests": n * drafts, "tokens": d_tok}}
        if s["router"]:
            rc = mx["understand"] if worst else EXPECTED_COMPLETION["understand"]
            r_tok = sum(x["router_prompt"] + rc for x in s["rows"])
            out.setdefault(router_model, {"requests": 0, "tokens": 0})
            out[router_model]["requests"] += n
            out[router_model]["tokens"] += r_tok
        return out

    def add(acc, more):
        for m, v in more.items():
            a = acc.setdefault(m, {"requests": 0.0, "tokens": 0.0})
            a["requests"] += v["requests"]
            a["tokens"] += v["tokens"]
    plan = {}
    for worst in (False, True):
        key = "worst" if worst else "expected"
        step2, step3 = {}, {}
        for name in settings:
            add(step2, cost(name, step2_drafter, worst))
        dear = max(settings, key=lambda n: per_setting[n]["draft_prompt_mean"][step2_drafter])
        for d in step3_drafters:
            add(step3, cost(dear, d, worst))
        total = {}
        add(total, step2)
        add(total, step3)
        plan[key] = {"step2": step2, "step3": step3, "step3_assumed_setting": dear, "total": total}
    with Ledger(load_budget()) as led:
        head = {m: led.headroom(EVAL_SCOPE, m) for m in {step2_drafter, router_model, *step3_drafters}}
    tpm = 8000
    verdict = {}
    for m, h in head.items():
        e = plan["expected"]["total"].get(m, {"requests": 0, "tokens": 0})
        w = plan["worst"]["total"].get(m, {"requests": 0, "tokens": 0})
        verdict[m] = {"headroom_tokens": h["tokens"], "headroom_requests": h["requests"],
                      "expected_tokens": round(e["tokens"]), "worst_tokens": round(w["tokens"]),
                      "expected_requests": round(e["requests"]), "worst_requests": round(w["requests"]),
                      "expected_fits": e["tokens"] <= h["tokens"] and e["requests"] <= h["requests"],
                      "worst_fits": w["tokens"] <= h["tokens"] and w["requests"] <= h["requests"],
                      "minutes_at_tpm_expected": round(e["tokens"] / tpm, 1)}
    for name in settings:
        per_setting[name]["step2_cost"] = {
            k: {m: {"requests": round(v["requests"]), "tokens": round(v["tokens"])}
                for m, v in cost(name, step2_drafter, k == "worst").items()} for k in ("expected", "worst")}
    for s in per_setting.values():
        s.pop("rows")
    return {"_generated_by": "scripts/eval_retrieval.py estimate", "scope": EVAL_SCOPE,
            "step2_drafter": step2_drafter, "step3_drafters": step3_drafters, "settings": per_setting,
            "assumptions": {"completion_tokens_expected": EXPECTED_COMPLETION, "redraft_rate_expected":
                            REDRAFT_RATE_EXPECTED, "worst": "every completion at max_tokens and every draft redrafted",
                            "prompt_tokens": "the limiter's upper-bound estimator on the raw-question retrieval "
                                             "(the router's rewrite changes the pieces, not their size)",
                            "tpm": "8000 per model per minute (measured)"},
            "plan": {k: {kk: ({m: {"requests": round(x["requests"]), "tokens": round(x["tokens"])}
                                   for m, x in vv.items()} if isinstance(vv, dict) else vv)
                         for kk, vv in v.items()} for k, v in plan.items()},
            "per_model": verdict}


def _standin_responder(p: dict) -> str:
    """Dry-run replies shaped like real ones (plumbing only; says nothing about quality)."""
    from relay.pipeline import NOT_IN_PAPERS
    sysmsg, user = p["messages"][0]["content"], p["messages"][-1]["content"]
    if sysmsg.startswith("You route"):
        ql = user.lower()
        paper = ("moshi" if "moshi" in ql else "dynamic_chunking" if "h-net" in ql else "attention")
        return json.dumps({"papers": [paper], "query": user[:80], "alt_query": user[-60:]})
    if sysmsg.startswith("You check"):
        return '{"unsupported": []}'
    q = user.split("\n", 1)[0].lower()
    if any(w in q for w in ("latency", "word error", "energy", "kwh", "p95")):
        return f"{NOT_IN_PAPERS}: the sources don't report this."
    m = re.search(r"\[S1\] \([^\n]*\)\n(.*?)(?:\n\n\[S2\]|\Z)", user, re.S)
    first = re.split(r"(?<=[.!?])\s+", (m.group(1) if m else "").replace("\n", " "))[0]
    return f"{first} [S1]"


async def run_answers(settings: list[str], drafters: list[str], real: bool, out_dir: Path, scenario: str,
                      run_id: str, qids: list[str] | None = None, note: str = "") -> Path:
    import asyncio
    import contextvars
    import datetime as dt
    from relay.fake_groq import FakeGroq
    from relay.ledger import BudgetedClient, Ledger, load_budget
    from relay.limiter import Limiter
    from relay.llm import Endpoint, LLMClient
    from relay.pipeline import FULL, Pipeline
    from relay.real import open_real

    qs = [q for q in load_questions() if not qids or q["qid"] in qids]
    out_dir.mkdir(parents=True, exist_ok=True)
    ans_path = out_dir / f"{run_id}.jsonl"
    calls_path = out_dir / f"{run_id}.calls.jsonl"
    meta = contextvars.ContextVar("meta", default={})
    fa = ans_path.open("a", encoding="utf-8", newline="\n")
    fc = calls_path.open("a", encoding="utf-8", newline="\n")

    def sink(entry: dict) -> None:
        entry = {"type": "call", "run_id": run_id, **meta.get(), **entry}
        fc.write(json.dumps(entry, ensure_ascii=False) + "\n")
        fc.flush()

    fa.write(json.dumps({"type": "run", "run_id": run_id, "scenario": scenario, "source": "real" if real else "stand-in",
                         "settings": {s: ANSWER_SETTINGS[s] for s in settings}, "drafters": drafters,
                         "questions": [q["qid"] for q in qs], "git": _git_commit(), "note": note,
                         "started": dt.datetime.now(dt.timezone.utc).isoformat(timespec="milliseconds")}) + "\n")
    fa.flush()
    lim = Limiter.from_probe()
    retrievers = {}

    async def one_drafter(bc, drafter):
        for setting in settings:
            size = ANSWER_SETTINGS[setting][0]
            r = retrievers.setdefault(size, Retriever.from_blocks(size))

            async def chat(payload, *, phase, timeouts=None):
                return await bc.stream_chat(payload, phase=phase, timeouts=timeouts, extra=dict(meta.get()))
            cfg = eval_config(setting, drafter)
            pipe = Pipeline(chat, lim, r, cfg)
            for q in qs:
                # stop rule: never start a request the ledger might refuse halfway (that reply would
                # come back degraded); worst case = a draft and a redraft at max_tokens, plus the router
                worst = 2 * (2500 + cfg["max_tokens"]["draft"])
                left = bc.ledger.headroom(EVAL_SCOPE, drafter)["tokens"]
                if left < worst:
                    fa.write(json.dumps({"type": "stopped", "run_id": run_id, "setting": setting, "drafter": drafter,
                                         "before_qid": q["qid"], "ledger_tokens_left": left,
                                         "needed_worst": worst}) + "\n")
                    fa.flush()
                    print(f"  STOP {setting} {drafter}: ledger has {left} tokens left, a request may need {worst}")
                    return
                rid = f"{run_id}-{setting}-{drafter.split('/')[-1]}-{q['qid']}"
                token = meta.set({"request_id": rid, "setting": setting, "drafter": drafter, "qid": q["qid"]})
                start = dt.datetime.now(dt.timezone.utc).isoformat(timespec="milliseconds")
                try:
                    reply = await pipe.answer(q["question"], request_id=rid)
                finally:
                    meta.reset(token)
                d = reply.to_dict()
                row = {"type": "answer", "run_id": run_id, "scenario": scenario,
                       "source": "real" if real else "stand-in", "setting": setting,
                       "setting_def": dict(zip(("piece_chars", "k", "rerank", "router"), ANSWER_SETTINGS[setting])),
                       "drafter": drafter, "qid": q["qid"], "request_id": rid, "start_time": start,
                       "degraded": reply.level != FULL, "reply": d}
                fa.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
                fa.flush()
                print(f"  {setting:<14} {drafter.split('/')[-1]:<14} {q['qid']:<4} {reply.kind:<9} "
                      f"level={reply.level} tokens={reply.tokens_in}+{reply.tokens_out} {reply.released_ms:.0f} ms",
                      flush=True)

    try:
        if real:
            with open_real(scope=EVAL_SCOPE, allow_real=True, raw_sink=sink) as bc:
                await asyncio.gather(*(one_drafter(bc, d) for d in drafters))
        else:
            async with FakeGroq(responder=_standin_responder) as fake:
                with Ledger(load_budget(), out_dir / "standin_ledger.jsonl") as led:
                    bc = BudgetedClient(LLMClient(Endpoint(fake.base_url, source="stand-in")), led, EVAL_SCOPE, sink)
                    await asyncio.gather(*(one_drafter(bc, d) for d in drafters))
    finally:
        fa.close()
        fc.close()
    return ans_path


def main(argv: list[str] | None = None) -> int:
    import argparse
    import asyncio
    import datetime as dt
    ap = argparse.ArgumentParser(prog="eval_retrieval.py")
    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("topk")
    e = sub.add_parser("estimate")
    e.add_argument("--settings", default=",".join(ANSWER_SETTINGS))
    e.add_argument("--drafter", default="openai/gpt-oss-120b")
    e.add_argument("--step3-drafters", default="openai/gpt-oss-20b,qwen/qwen3.8-27b")
    for name in ("answers", "drafters"):
        a = sub.add_parser(name)
        a.add_argument("--real", action="store_true", help="real Groq calls through the ledger")
        a.add_argument("--out-dir", type=Path, help="required without --real (dry runs never write results/)")
        a.add_argument("--run-id")
        a.add_argument("--qids", help="comma list (default: all 18)")
        a.add_argument("--note", default="", help="why this run exists (kept in the run header)")
        if name == "answers":
            a.add_argument("--settings", default=",".join(ANSWER_SETTINGS))
            a.add_argument("--drafter", default="openai/gpt-oss-120b")
        else:
            a.add_argument("--setting", required=True, choices=list(ANSWER_SETTINGS))
            a.add_argument("--drafters", default="openai/gpt-oss-20b,qwen/qwen3.8-27b")
    a = ap.parse_args(argv)
    if a.cmd in (None, "topk"):
        return topk_main()
    if a.cmd == "estimate":
        res = estimate(a.settings.split(","), a.drafter, [d for d in a.step3_drafters.split(",") if d])
        EST_OUT.parent.mkdir(parents=True, exist_ok=True)
        EST_OUT.write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8", newline="\n")
        print(json.dumps({"plan": res["plan"], "per_model": res["per_model"],
                          "draft_prompt_mean": {k: v["draft_prompt_mean"] for k, v in res["settings"].items()}},
                         indent=1))
        return 0
    if not a.real and not a.out_dir:
        ap.error("--out-dir is required for a stand-in (dry) run")
    out_dir = RAW_ANSWERS if a.real else a.out_dir
    if a.real and a.out_dir:
        ap.error("real runs always write to results/raw/answers")
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    qids = a.qids.split(",") if a.qids else None
    if a.cmd == "answers":
        settings, drafters, scenario = a.settings.split(","), [a.drafter], "eval:retrieval"
    else:
        settings, drafters, scenario = [a.setting], a.drafters.split(","), "eval:drafter"
    for s in settings:
        if s not in ANSWER_SETTINGS:
            ap.error(f"unknown setting {s}")
    run_id = a.run_id or f"{scenario.split(':')[1]}-{stamp}"
    path = asyncio.run(run_answers(settings, drafters, a.real, out_dir, scenario, run_id, qids, a.note))
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
