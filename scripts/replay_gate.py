"""Replay every saved real draft through the citation gate, old rules vs new (no model calls).

    .venv\\Scripts\\python scripts/replay_gate.py

Reads the real answer rows (evals: results/raw/answers; load runs: results/raw/runs) and their draft
calls (the .calls.jsonl files), skipping runs in results/excluded_runs.json. Each draft is checked with
check_citations under the old rules (both [gate] switches off) and the new ones (drop_plain_sentences
and join_formulas on). Writes data/experiments/gate_replay/result.md and result.json:

  * per draft: how many verdicts change;
  * per request: what the person would have got under the new gate, replayed from the drafts that
    were actually made (a request whose first draft passed the old gate but fails the new one would
    need a redraft that was never made: "unknown"); the new answer is graded with relay.grade, so
    the write-up can report the e7aa65f numbers plus this delta. The verifier is not re-run;
  * every sentence the new gate dropped, every unit it let through that the old gate failed (a
    formula joined to its cited sentence), and every draft it newly fails, for checking by hand.
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from relay.grade import grade, load_questions  # noqa: E402
from relay.pipeline import (NOT_IN_PAPERS, check_citations, join_formula_lines, normalize_citations,  # noqa: E402
                            rare_terms, source_list, split_sentences)
from relay.retrieve import Hit, Retriever  # noqa: E402

RAW = [ROOT / "results" / "raw" / "answers", ROOT / "results" / "raw" / "runs"]
EXCLUDED = ROOT / "results" / "excluded_runs.json"
RESULTS = ROOT / "results" / "results.csv"
OUT = ROOT / "data" / "experiments" / "gate_replay"


def load_jsonl(p: Path):
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            yield json.loads(line)


def main() -> int:
    excluded = json.loads(EXCLUDED.read_text(encoding="utf-8"))["runs"] if EXCLUDED.exists() else {}
    qs = load_questions()
    reqs = {}
    for d in RAW:
        for p in sorted(d.glob("*.jsonl")):
            if p.name.endswith(".calls.jsonl"):
                continue
            for o in load_jsonl(p):
                if o.get("type") != "answer" or o.get("source") != "real" or o.get("run_id") in excluded:
                    continue
                rep = o["reply"]
                pids = ((rep.get("checks") or {}).get("retrieval") or {}).get("pids")
                size = (o.get("setting_def") or {}).get("piece_chars")
                if pids and size:
                    reqs[o["request_id"]] = {"qid": o.get("qid"), "scenario": o.get("scenario"), "pids": pids,
                                             "size": size, "kind": rep["kind"], "setting": o.get("setting"),
                                             "drafter": o.get("drafter"), "attempts": [],
                                             "released": (rep.get("text") or "").strip()}
    rows_csv = {}
    if RESULTS.exists():
        with RESULTS.open(encoding="utf-8", newline="") as f:
            rows_csv = {r["request_id"]: r for r in csv.DictReader(f)}
    retrievers: dict[int, Retriever] = {}

    def retr(size):
        if size not in retrievers:
            retrievers[size] = Retriever.from_blocks(size)
        return retrievers[size]

    def hits_for(size, pids):
        by = {p.pid: p for p in retr(size).pieces}
        return [Hit(by[pid], 0.0, i + 1, bm25=0.0) for i, pid in enumerate(pids)]

    for d in RAW:
        for p in sorted(d.glob("*.calls.jsonl")):
            for c in load_jsonl(p):
                rid = (c.get("extra") or {}).get("request_id") or c.get("request_id")
                rec = c.get("record") or {}
                if rid not in reqs or c.get("phase") != "draft" or rec.get("source") != "real":
                    continue
                text = rec.get("text") or ""
                if rec.get("status") != 200 or not text.strip():
                    continue                       # a failed call: the pipeline moved on without a draft
                reqs[rid]["attempts"].append({"model": rec.get("model"), "raw": text,
                                              "text": normalize_citations(text)})

    drafts, requests = [], []
    for rid, r in reqs.items():
        if not r["attempts"]:
            continue
        hits = hits_for(r["size"], r["pids"])
        rare = rare_terms(retr(r["size"]))
        verdicts = []
        for i, a in enumerate(r["attempts"], 1):
            if a["text"].strip().upper().startswith(NOT_IN_PAPERS):
                verdicts.append({"refusal": True})
                continue
            old = check_citations(normalize_citations(a["raw"], strip_invisible=False), hits)   # e7aa65f
            fix = check_citations(a["text"], hits)            # + the zero-width-space fix only
            new = check_citations(a["text"], hits, drop_plain=True, join_formulas=True, rare=rare)
            old_units = set(split_sentences(a["text"]))
            joined = [u for u in join_formula_lines(split_sentences(a["text"])) if u not in old_units]
            new_fail = {f["sentence"] for f in new["failures"]}
            let_through = [u for u in joined if u not in new_fail]
            v = {"refusal": False, "old_ok": old["ok"], "fix_ok": fix["ok"], "new_ok": new["ok"], "dropped": new["dropped"],
                 "let_through": let_through, "old_failures": [f"{f['reason']}: {f['sentence']}" for f in old["failures"]],
                 "new_failures": [f"{f['reason']}: {f['sentence']}" + (f" (terms: {', '.join(map(str, f['terms'][:6]))})"
                                                                      if f.get("terms") else "")
                                  for f in new["failures"]],
                 "new_text": new["text"]}
            verdicts.append(v)
            drafts.append({"request_id": rid, "attempt": i, "model": a["model"], **{k: r[k] for k in ("qid", "scenario")},
                           **v, "text": a["text"]})
        # what the person would have got: the first attempt the new gate passes, from the attempts made
        new_kind, new_text, unknown = "passages", None, False
        for i, v in enumerate(verdicts):
            if v["refusal"]:
                new_kind = "refusal"
                break
            if v["new_ok"]:
                new_kind, new_text = "answer", v["new_text"]
                break
            if v["old_ok"] and i == len(verdicts) - 1:
                unknown = True                     # old gate passed it, so no redraft was ever made
                break
        else:
            if len(verdicts) < 2:
                unknown = True                     # one draft failed both gates but no second was recorded
        row = rows_csv.get(rid, {})
        new_grade = ""
        same = (new_kind == r["kind"] == "answer"
                and (new_text or "").strip() == normalize_citations(r["released"]).strip())
        if same:
            new_grade = row.get("answer_grade", "")          # the same text: keep its recorded grade
        elif new_kind == "answer" and r["qid"] in qs:
            g = grade(qs[r["qid"]], {"kind": "answer", "text": new_text, "sources": source_list(hits)},
                      {p.pid: {"text": p.text, "paper": p.paper, "block_ids": p.block_ids} for p in retr(r["size"]).pieces})
            new_grade = (g["label"] or "") + ("" if g["sure"] else " (unsure)")
        requests.append({"request_id": rid, "qid": r["qid"], "scenario": r["scenario"], "setting": r["setting"],
                         "drafter": r["drafter"], "in_tables": row.get("superseded") == "0" and not row.get("excluded"),
                         "old_kind": r["kind"], "old_grade": row.get("answer_grade", ""),
                         "new_kind": "unknown (a redraft would be needed)" if unknown else new_kind,
                         "new_grade": "" if unknown else new_grade})

    changed = [q for q in requests if q["new_kind"] != q["old_kind"] or
               (q["new_kind"] == "answer" and q["new_grade"].split(" ")[0] != q["old_grade"])]
    d_old = Counter("pass" if x["old_ok"] else "fail" for x in drafts)
    d_new = Counter("pass" if x["new_ok"] else "fail" for x in drafts)
    d_fix = Counter("pass" if x["fix_ok"] else "fail" for x in drafts)
    by_cause = Counter(("zero-width fix" if x["fix_ok"] else "drop/join rules") if x["new_ok"] else "newly failed"
                       for x in drafts if x["old_ok"] != x["new_ok"])
    flips = Counter(f"{'pass' if x['old_ok'] else 'fail'} -> {'pass' if x['new_ok'] else 'fail'}" for x in drafts)
    by_scn = defaultdict(Counter)
    for q in requests:
        if q["in_tables"]:
            by_scn[q["scenario"]]["requests"] += 1
            if q in changed:
                by_scn[q["scenario"]][f"{q['old_kind']}/{q['old_grade'] or '-'} -> {q['new_kind']}/{q['new_grade'] or '-'}"] += 1

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "result.json").write_text(json.dumps({"drafts": drafts, "requests": requests,
                                                 "draft_flips": dict(flips)}, indent=1, ensure_ascii=False) + "\n",
                                     encoding="utf-8")
    L = ["# Gate replay: old rules vs new, on every saved real draft", "",
         "Generated by `scripts/replay_gate.py` (no model calls). Old = the gate as measured at e7aa65f; "
         "fix = old + invisible characters removed from citations (a zero-width space inside \"[S1]\" hid the "
         "citation); new = fix + `drop_plain_sentences` and `join_formulas` on. Real drafts with status 200 from the evals "
         "(results/raw/answers) and load runs (results/raw/runs), excluded runs skipped. The verifier is not re-run.", "",
         "## Drafts", "",
         f"Drafts replayed: {len(drafts)}. Old gate: {d_old['pass']} pass, {d_old['fail']} fail. "
         f"Old + zero-width fix: {d_fix['pass']} pass, {d_fix['fail']} fail. "
         f"New gate: {d_new['pass']} pass, {d_new['fail']} fail.", "",
         "Verdicts that change, by cause: " + (", ".join(f"{k} {v}" for k, v in sorted(by_cause.items())) or "none") + ".",
         "",
         "| old -> new | drafts |", "|---|---|"]
    L += [f"| {k} | {v} |" for k, v in sorted(flips.items())]
    L += ["", "## Requests counted in the tables: what changes", "",
          "Old = what the person got (kind / answer_grade in results.csv). New = the first recorded draft the new "
          "gate passes, as released (dropped sentences removed), graded by relay.grade (no overrides: those are "
          "keyed to the old answers).", "",
          "| scenario | requests | changes (old -> new) |", "|---|---|---|"]
    for scn in sorted(by_scn):
        c = by_scn[scn]
        ch = "; ".join(f"{k}: {v}" for k, v in sorted(c.items()) if k != "requests") or "none"
        L.append(f"| {scn} | {c['requests']} | {ch} |")
    L += ["", "Every changed request, including superseded ones:", "",
          "| request | qid | scenario | in tables | old | new |", "|---|---|---|---|---|---|"]
    for q in changed:
        L.append(f"| `{q['request_id']}` | {q['qid']} | {q['scenario']} | {'yes' if q['in_tables'] else 'no'} | "
                 f"{q['old_kind']} / {q['old_grade'] or '-'} | {q['new_kind']} / {q['new_grade'] or '-'} |")
    dropped = [(x, s) for x in drafts for s in x["dropped"] if x["new_ok"]]
    dropped_failed = [(x, s) for x in drafts for s in x["dropped"] if not x["new_ok"]]
    L += ["", f"## Every sentence the new gate dropped from a draft it passed ({len(dropped)})", "",
          "These would have been removed from the released answer.", ""]
    L += [f"- {x['qid']} · {x['scenario']} · attempt {x['attempt']} · `{x['request_id']}`: \"{s}\"" for x, s in dropped]
    L += ["", f"## Plain sentences in drafts that still failed ({len(dropped_failed)})", "",
          "Classed as droppable, but the draft failed for another reason, so nothing was released from it.", ""]
    L += [f"- {x['qid']} · {x['scenario']} · attempt {x['attempt']}: \"{s}\"" for x, s in dropped_failed]
    lets = [(x, u) for x in drafts for u in x["let_through"]]
    L += ["", f"## Every unit the new gate let through that the old gate failed ({len(lets)})", "",
          "Formula lines joined to the cited sentence they belong to, checked as one (citation, numbers, quotes).", ""]
    L += [f"- {x['qid']} · {x['scenario']} · attempt {x['attempt']} · `{x['request_id']}`:\n\n  ```\n  {u}\n  ```"
          for x, u in lets]
    newly_failed = [x for x in drafts if x["old_ok"] and not x["new_ok"]]
    L += ["", f"## Drafts the new gate fails that the old one passed ({len(newly_failed)})", ""]
    for x in newly_failed:
        L += [f"- {x['qid']} · {x['scenario']} · attempt {x['attempt']} · `{x['request_id']}`: " + "; ".join(x["new_failures"])]
    L += ["", "## Drafts that fail both gates: why the new gate fails them", ""]
    for x in drafts:
        if not x["old_ok"] and not x["new_ok"]:
            L.append(f"- {x['qid']} · {x['scenario']} · attempt {x['attempt']}: " + " | ".join(x["new_failures"])[:600])
    (OUT / "result.md").write_text("\n".join(L) + "\n", encoding="utf-8", newline="\n")
    print(f"{len(drafts)} drafts: old {d_old['pass']} pass / {d_old['fail']} fail; new {d_new['pass']} pass / "
          f"{d_new['fail']} fail; {dict(flips)}; {len(changed)} requests change; {len(dropped)} sentences dropped, "
          f"{len(lets)} units let through -> {OUT / 'result.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
