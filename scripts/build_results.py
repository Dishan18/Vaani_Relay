"""Build results/results.csv and the grade files from the raw answer files. Nothing here is typed by hand.

    .venv\\Scripts\\python scripts\\build_results.py [--raw-dir results/raw/answers] [--out-dir results]
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
    busy. retries = model calls that failed (each was followed by another model or gave up).
  * answer_grade is empty for an ungraded row; grade_source says why. Real rows: "auto" (relay.grade),
    "override" (results/grade_overrides.jsonl), or "not_graded:<kind>" (a passages-only or busy
    reply has no answer to grade). Stand-in rows are NEVER graded by the grader (their text is a
    placeholder or a replay): a stand-in row gets a grade only when its answer is byte-identical to
    a graded real answer (same answer id: question, kind, text, sources), and then grade_source =
    "replay:<real request_id>"; otherwise "not_graded:stand-in". make_tables.py counts and shows
    every ungraded row and refuses a stand-in grade without a replay source.
  * degraded rows (level > 0) are graded like any other and keep their degraded_level.
  * superseded = 1 (eval scenarios only) when a later run answered the same (scenario, setting, drafter,
    question); the
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
           "gave_up", "degraded_level", "queue_wait_ms", "model", "calls", "grade_sure", "answer_id", "grade_source",
           "superseded"]


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
    ap.add_argument("--raw-dir", type=Path, default=ROOT / "results" / "raw" / "answers")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "results")
    ap.add_argument("--overrides", type=Path, default=OVERRIDES)
    a = ap.parse_args(argv)
    qs = load_questions()
    overrides = load_overrides(a.overrides)
    rows = read_answers(a.raw_dir)
    rows.sort(key=lambda r: (r["source"] != "real", r["start_time"], r["request_id"]))   # real first
    latest = {}
    for r in rows:
        latest[(r["scenario"], r["setting"], r["drafter"], r["qid"])] = r["request_id"]
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
            source = "override" if ov else ("auto" if final else f"not_graded:{reply['kind']}")
            if final:
                real_graded[g["answer_id"]] = (final, r["request_id"], bool(g["sure"]) or bool(ov))
        else:
            aid = answer_id(r["qid"], reply)
            ov = None
            hit = real_graded.get(aid)
            final = hit[0] if hit else None
            source = f"replay:{hit[1]}" if hit else "not_graded:stand-in"
            g = {"answer_id": aid, "label": final, "sure": hit[2] if hit else True, "reasons": [source],
                 "facts": [], "wrong_paper_citations": [], "borrowed": []}
        # only the evals answer each question once per setting; load runs repeat questions and every request counts
        superseded = int(r["scenario"].startswith("eval:")
                         and latest[(r["scenario"], r["setting"], r["drafter"], r["qid"])] != r["request_id"])
        rec = {"answer_id": g["answer_id"], "request_id": r["request_id"], "run_id": r["run_id"],
               "scenario": r["scenario"], "setting": r["setting"], "drafter": r["drafter"], "qid": r["qid"],
               "kind": reply["kind"], "degraded": bool(r.get("degraded")), "auto_label": g["label"], "sure": g["sure"],
               "reasons": g["reasons"], "facts": g["facts"], "wrong_paper_citations": g["wrong_paper_citations"],
               "borrowed": g["borrowed"], "final_label": final,
               "override": ({"label": ov["label"], "reason": ov["reason"], "by": ov["by"], "at": ov["at"]} if ov else None),
               "superseded": bool(superseded), "source": r["source"], "grade_source": source}
        grades.append(rec)
        if not g["sure"] and not ov and not superseded:
            unsure.append((rec, reply))
        calls = reply.get("model_calls") or []
        status = {"answer": "ok", "refusal": "ok", "passages": "rate_limited"}.get(reply["kind"], "error")
        out_rows.append({
            "request_id": r["request_id"], "scenario": r["scenario"], "source": r["source"],
            "start_time": r["start_time"], "first_token_ms": reply.get("released_ms"),
            "total_ms": reply.get("released_ms"), "status": status,
            "retries": sum(1 for c in calls if c.get("status") != 200 or c.get("outcome") != "ok"),
            "tokens_in": reply.get("tokens_in"), "tokens_out": reply.get("tokens_out"),
            "retrieval_setting": r["setting"], "answer_grade": final or "",
            "gave_up": 0, "degraded_level": reply.get("level"), "queue_wait_ms": reply.get("budget_wait_ms"),
            "model": r["drafter"], "calls": len(calls), "grade_sure": int(bool(g["sure"]) or bool(ov)),
            "answer_id": g["answer_id"], "grade_source": source, "superseded": superseded})
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
