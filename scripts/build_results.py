"""Build results/results.csv and the grade files from the raw answer files. Nothing here is typed by hand.

    .venv\\Scripts\\python scripts\\build_results.py [--raw-dir DIR ...] [--out-dir results]
        (default raw dirs: results/raw/answers for the evals, results/raw/runs for load runs)
        ->  <out>/results.csv, <out>/quality/grades.jsonl, <out>/quality/unsure.md

Every answer row in <raw-dir>/*.jsonl is graded by relay.grade against data/questions.jsonl. An
override in results/grade_overrides.jsonl (``python -m relay.grade override ...``) replaces the
auto label for that exact answer, and its reason is kept next to it. results.csv is regenerated
from scratch every time, so a grade is never edited in place.

results.csv: the brief's columns in order, then ours:
  request_id, scenario, source, start_time, first_token_ms, total_ms, status, retries, tokens_in,
  tokens_out, retrieval_setting, answer_grade,
  gave_up, degraded_level, queue_wait_ms, model, calls, grade_sure, answer_id, superseded
  * first_token_ms = total_ms = arrival to release: the reply is buffered until checked (PLAN G2).
  * status: ok for an answer or refusal; rate_limited for a passages-only reply (budget); error for
    busy or a failed request; timeout (gave_up = 1) when the simulated person's patience ran out first:
    then total_ms = how long they waited and first_token_ms is empty. retries = model calls that
    failed (each was followed by another model, a retry, or giving up).
  * answer_grade holds only the brief's four values. Every real reply is graded, passages-only and
    busy replies included (it is what the person got). Stand-in rows always leave it empty (their
    text is a placeholder or a replay of a real one); grade_note says why: "auto", "override: ...",
    or "not graded: stand-in" (plus "; replays real <request_id>, graded <label>" when its answer is
    byte-identical to a graded real answer). make_tables.py stops if a stand-in row carries a grade,
    and shows grading coverage for every scenario and source.
  * degraded rows (level > 0) are graded like any other and keep their degraded_level.
  * excluded: why a run listed in results/excluded_runs.json counts in no table (its rows stay, with
    superseded = 1, and never replace an earlier answer).
  * shadow_cost_usd: the request's model calls priced at Groq's published paid prices
    (data/prices.json, with URL and date); the free tier we ran on costs $0.
  * superseded = 1 when a later run replaces this row: for the evals, a later answer to the same
    (scenario, setting, drafter, question); for load runs, a later run of the same scenario and source
    (the whole earlier run is superseded; the raw files stay, and the newer run's header says why); the
    tables use the latest one.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from relay.grade import OVERRIDES, answer_id, grade, load_overrides, load_questions  # noqa: E402
from relay.retrieve import Retriever  # noqa: E402

COLUMNS = ["request_id", "scenario", "source", "start_time", "first_token_ms", "total_ms", "status", "retries",
           "tokens_in", "tokens_out", "retrieval_setting", "answer_grade",
           "gave_up", "degraded_level", "queue_wait_ms", "model", "calls", "grade_sure", "answer_id", "grade_note",
           "superseded", "shadow_cost_usd", "excluded"]
EXCLUDED = ROOT / "results" / "excluded_runs.json"
PRICES = ROOT / "data" / "prices.json"


def shadow_cost(calls: list[dict], prices: dict) -> float:
    """What this request's model calls would cost at Groq's published paid prices (data/prices.json)."""
    usd = 0.0
    for c in calls:
        pr = prices.get(c.get("model"))
        if pr:
            usd += ((c.get("prompt_tokens") or 0) * pr["input"] + (c.get("completion_tokens") or 0) * pr["output"]) / 1e6
    return usd


def read_answers(raw_dir: Path) -> list[dict]:
    rows = []
    for p in sorted(raw_dir.glob("*.jsonl")):
        if p.name.endswith(".calls.jsonl") or p.name.startswith("standin_ledger"):
            continue
        with p.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    o = json.loads(line)
                    if o.get("type") == "answer":
                        o["_file"] = p.name
                        rows.append(o)
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", type=Path, action="append", default=None)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "results")
    ap.add_argument("--overrides", type=Path, default=OVERRIDES)
    a = ap.parse_args(argv)
    qs = load_questions()
    prices = json.loads(PRICES.read_text(encoding="utf-8"))["models"] if PRICES.exists() else {}
    overrides = load_overrides(a.overrides)
    raw_dirs = a.raw_dir or [ROOT / "results" / "raw" / "answers", ROOT / "results" / "raw" / "runs"]
    rows = [r for d in raw_dirs if d.exists() for r in read_answers(d)]
    excluded = json.loads(EXCLUDED.read_text(encoding="utf-8"))["runs"] if EXCLUDED.exists() else {}
    rows.sort(key=lambda r: (r["source"] != "real", r["start_time"], r["request_id"]))   # real first
    latest = {}
    latest_run: dict[tuple, tuple] = {}              # load runs: (scenario, source) -> (first start, run_id)
    for r in rows:
        if r["run_id"] in excluded:                  # never replaces an earlier answer or run
            continue
        latest[(r["scenario"], r["setting"], r["drafter"], r["qid"])] = r["request_id"]
        if not r["scenario"].startswith("eval:"):
            k, v = (r["scenario"], r["source"]), (r["start_time"], r["run_id"])
            cur = latest_run.get(k)
            if cur is None or (r["run_id"] != cur[1] and v[0] > cur[0] and
                               v[0] > max(x["start_time"] for x in rows if x["run_id"] == cur[1])):
                latest_run[k] = v
    pieces_by_size: dict[int, dict] = {}
    out_rows, grades, unsure = [], [], []
    real_graded: dict[str, tuple[str, str, bool]] = {}      # answer_id -> (label, request_id, sure)
    for r in rows:
        size = r["setting_def"]["piece_chars"]
        if size not in pieces_by_size:
            pieces_by_size[size] = {p.pid: {"text": p.text, "paper": p.paper, "block_ids": p.block_ids}
                                    for p in Retriever.from_blocks(size).pieces}
        reply = r["reply"]
        if r["source"] == "real":
            g = grade(qs[r["qid"]], reply, pieces_by_size[size])
            ov = overrides.get(g["answer_id"])
            final = ov["label"] if ov else g["label"]
            source = (f"override: {ov['reason']}" if ov else ("auto" if final else f"not graded: {g['reasons'][0]}"))
            if final:
                real_graded[g["answer_id"]] = (final, r["request_id"], bool(g["sure"]) or bool(ov))
        else:
            aid = answer_id(r["qid"], reply)
            ov = None
            hit = real_graded.get(aid)
            final = None                                   # stand-in rows are never graded
            source = "not graded: stand-in" + (f"; replays real {hit[1]}, graded {hit[0]}" if hit else "")
            g = {"answer_id": aid, "label": None, "sure": True, "reasons": [source],
                 "facts": [], "wrong_paper_citations": [], "borrowed": []}
        # only the evals answer each question once per setting; load runs repeat questions and every request counts
        if r["run_id"] in excluded:
            superseded = 1
        elif r["scenario"].startswith("eval:"):
            superseded = int(latest[(r["scenario"], r["setting"], r["drafter"], r["qid"])] != r["request_id"])
        else:   # a load run is superseded as a whole by a later run of the same scenario and source
            superseded = int(latest_run[(r["scenario"], r["source"])][1] != r["run_id"])
        rec = {"answer_id": g["answer_id"], "request_id": r["request_id"], "run_id": r["run_id"],
               "scenario": r["scenario"], "setting": r["setting"], "drafter": r["drafter"], "qid": r["qid"],
               "kind": reply["kind"], "degraded": bool(r.get("degraded")), "auto_label": g["label"], "sure": g["sure"],
               "reasons": g["reasons"], "facts": g["facts"], "wrong_paper_citations": g["wrong_paper_citations"],
               "borrowed": g["borrowed"], "final_label": final,
               "override": ({"label": ov["label"], "reason": ov["reason"], "by": ov["by"], "at": ov["at"]} if ov else None),
               "superseded": bool(superseded), "source": r["source"], "grade_note": source}
        grades.append(rec)
        if r["source"] == "real" and not g["sure"] and not ov and not superseded:
            unsure.append((rec, reply))
        calls = reply.get("model_calls") or []
        gave_up = bool(r.get("gave_up"))
        status = "timeout" if gave_up else {"answer": "ok", "refusal": "ok", "passages": "rate_limited"}.get(
            reply["kind"], "error")
        out_rows.append({
            "request_id": r["request_id"], "scenario": r["scenario"], "source": r["source"],
            "start_time": r["start_time"], "first_token_ms": "" if gave_up else reply.get("released_ms"),
            "total_ms": r.get("waited_ms") if gave_up else reply.get("released_ms"), "status": status,
            "retries": sum(1 for c in calls if c.get("status") != 200 or c.get("outcome") != "ok"),
            "tokens_in": reply.get("tokens_in"), "tokens_out": reply.get("tokens_out"),
            "retrieval_setting": r["setting"], "answer_grade": final or "",
            "gave_up": int(gave_up), "degraded_level": reply.get("level"), "queue_wait_ms": reply.get("budget_wait_ms"),
            "model": r["drafter"], "calls": len(calls), "grade_sure": int(bool(g["sure"]) or bool(ov)),
            "answer_id": g["answer_id"], "grade_note": source, "superseded": superseded,
            "shadow_cost_usd": round(shadow_cost(calls, prices), 7), "excluded": excluded.get(r["run_id"], "")})
    (a.out_dir / "quality").mkdir(parents=True, exist_ok=True)
    with (a.out_dir / "results.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(out_rows)
    with (a.out_dir / "quality" / "grades.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for g in grades:
            f.write(json.dumps(g, ensure_ascii=False) + "\n")
    L = ["# Grades the grader isn't sure of", "",
         "Generated by `scripts/build_results.py`. Each needs a person's look. To overrule one:", "",
         "    .venv\\Scripts\\python -m relay.grade override <answer_id> <label> --reason \"...\"", "",
         f"{len(unsure)} unsure (latest runs only; already-overruled ones are not listed).", ""]
    for rec, reply in unsure:
        L += [f"## {rec['qid']} · {rec['setting']} · {rec['drafter'].split('/')[-1]} · auto: **{rec['auto_label']}**",
              "", f"answer_id `{rec['answer_id']}`, request `{rec['request_id']}`", ""]
        L += [f"- {x}" for x in rec["reasons"]]
        L += ["", "> " + (reply.get("text") or "").replace("\n", "\n> "), ""]
        if reply.get("sources"):
            L.append("Sources: " + "; ".join(f"{s['sid']} {s['cite']}" for s in reply["sources"]))
            L.append("")
    (a.out_dir / "quality" / "unsure.md").write_text("\n".join(L) + "\n", encoding="utf-8", newline="\n")
    n_final = sum(1 for g in grades if g["final_label"] and not g["superseded"])
    print(f"{len(out_rows)} rows -> {a.out_dir / 'results.csv'}; graded {n_final} (latest); "
          f"{len(unsure)} unsure; {len(overrides)} overrides applied where they match")
    return 0


if __name__ == "__main__":
    sys.exit(main())
