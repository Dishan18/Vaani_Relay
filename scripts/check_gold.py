"""Check every gold fact in data/questions.jsonl against data/blocks.jsonl.

    .venv\\Scripts\\python scripts\\check_gold.py   ->  data/gold_check.md (exit 1 if anything fails)

For each question:
  * answerable: every gold fact has at least one evidence quote, and every quote appears verbatim
    in the named block, which belongs to the question's paper (right paper = part of "correct").
  * unanswerable: none of the `absent` patterns (case-insensitive regexes) matches any non-reference
    block of the question's paper; each hit is listed so a person can judge it.
  * `wrong_if` and `distractors` quotes are verified too, so the traps point at real text.
  * `review` notes (a proposed gold change waiting for the author) are listed; the set is a draft
    until none is left open. `gold_changed` (what the author changed, from what, and why) and
    `notes` (optional extras, retrieval risks; they never block) are listed too.
Section hints are not checked: the extractor's `section` field is page-layout based (Table 2 of
the Transformer paper sits under "5.4 Regularization"), and the brief treats sections as hints.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BLOCKS = ROOT / "data" / "blocks.jsonl"
QUESTIONS = ROOT / "data" / "questions.jsonl"
OUT = ROOT / "data" / "gold_check.md"
PAPERS = {"attention": "1706.03762v7", "moshi": "2410.00037v1", "dynamic_chunking": "2507.07955v2"}
LABELS = ("correct", "wrong", "refused_correctly", "refused_wrongly")


def load_jsonl(p: Path) -> list[dict]:
    with p.open(encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def check_quote(blocks: dict, paper: str, ev: dict) -> str | None:
    b = blocks.get(ev.get("block_id"))
    if b is None:
        return f"block {ev.get('block_id')} not in blocks.jsonl"
    if b["paper"] != paper:
        return f"block {ev['block_id']} is from {b['paper']}, question is about {paper}"
    if not ev.get("quote") or ev["quote"] not in b["text"]:
        return f"quote not found verbatim in {ev['block_id']}: {ev.get('quote')!r}"
    return None


def check(questions: list[dict], blocks_list: list[dict]) -> tuple[list[dict], list[str]]:
    blocks = {b["block_id"]: b for b in blocks_list}
    rows, errors = [], []
    seen = set()
    for q in questions:
        qid, paper = q.get("qid"), q.get("paper")
        errs: list[str] = []
        if qid in seen:
            errs.append("duplicate qid")
        seen.add(qid)
        if PAPERS.get(paper) != q.get("arxiv_id"):
            errs.append(f"paper/arxiv_id mismatch: {paper} {q.get('arxiv_id')}")
        n_quotes, hits = 0, []
        if q.get("answerable"):
            facts = q.get("facts") or []
            if not facts:
                errs.append("answerable but no gold facts")
            for i, f in enumerate(facts, 1):
                evs = f.get("evidence") or []
                if not evs:
                    errs.append(f"fact {i} has no evidence: {f.get('fact')!r}")
                # grading patterns (relay/grade.py): each must compile and match this fact's own
                # wording or its evidence quotes, so a typo can't silently make a fact unmatchable
                own = " ".join([f.get("fact", "")] + [ev.get("quote", "") for ev in evs])
                if not f.get("match"):
                    errs.append(f"fact {i} has no grading `match` patterns")
                for pat in f.get("match") or []:
                    try:
                        rx = re.compile(pat, re.I)
                    except re.error as e:
                        errs.append(f"fact {i}: bad pattern {pat!r}: {e}")
                        continue
                    if not rx.search(own):
                        errs.append(f"fact {i}: pattern {pat!r} matches neither the fact nor its evidence")
                for ev in evs:
                    n_quotes += 1
                    e = check_quote(blocks, paper, ev)
                    if e:
                        errs.append(f"fact {i}: {e}")
        else:
            if q.get("facts"):
                errs.append("unanswerable but has gold facts")
            pats = q.get("absent") or []
            if not pats:
                errs.append("unanswerable but no `absent` patterns to show it is not reported")
            for pat in pats:
                rx = re.compile(pat, re.I)
                for b in blocks_list:
                    if b["paper"] == paper and b["kind"] != "reference" and rx.search(b["text"]):
                        hits.append(f"/{pat}/ matches {b['block_id']}")
            errs.extend(hits)
        for pat in [w.get("pattern", "") for w in q.get("wrong_match") or []] + list(q.get("distractor_match") or []):
            try:
                re.compile(pat, re.I)
            except re.error as e:
                errs.append(f"bad wrong/distractor pattern {pat!r}: {e}")
        for key in ("wrong_if", "distractors"):
            for item in q.get(key) or []:
                for ev in item.get("evidence") or []:
                    n_quotes += 1
                    # a distractor may sit in another paper (that is the point of borrowing traps)
                    e = check_quote(blocks, blocks.get(ev.get("block_id"), {}).get("paper", paper), ev)
                    if e:
                        errs.append(f"{key}: {e}")
        rows.append({"qid": qid, "paper": paper, "answerable": bool(q.get("answerable")),
                     "trap": q.get("trap") or "", "facts": len(q.get("facts") or []), "quotes": n_quotes,
                     "errors": errs, "review": q.get("review") or "", "notes": q.get("notes") or "",
                     "gold_changed": q.get("gold_changed")})
        errors += [f"{qid}: {e}" for e in errs]
    return rows, errors


def render(rows: list[dict], errors: list[str], questions: list[dict]) -> str:
    n_ans = sum(r["answerable"] for r in rows)
    open_reviews = [r for r in rows if r["review"]]
    L = ["# Gold-fact check", "",
         f"Generated by `scripts/check_gold.py` from `data/questions.jsonl` and `data/blocks.jsonl`.", "",
         f"- questions: {len(rows)} ({n_ans} answerable, {len(rows) - n_ans} must be refused)",
         f"- traps: {sum(1 for r in rows if r['trap'])}",
         f"- failed checks: {len(errors)}",
         f"- open review notes (proposed gold changes, not applied): {len(open_reviews)}"
         + (" -> **draft**, needs the author's decision" if open_reviews else " -> **final**"),
         f"- gold changed by the author: {sum(1 for r in rows if r['gold_changed'])}", "",
         "| qid | paper | answerable | trap | gold facts | quotes verified | result |",
         "|---|---|---|---|---|---|---|"]
    for r in rows:
        res = "PASS" if not r["errors"] else f"FAIL ({len(r['errors'])})"
        L.append(f"| {r['qid']} | {r['paper']} | {'yes' if r['answerable'] else 'no'} | {r['trap']} | "
                 f"{r['facts']} | {r['quotes'] - len([e for e in r['errors'] if 'quote' in e])}/{r['quotes']} | {res} |")
    if errors:
        L += ["", "## Failures", ""] + [f"- {e}" for e in errors]
    if open_reviews:
        L += ["", "## Open review notes", ""]
        for r in open_reviews:
            L.append(f"- **{r['qid']}**: {r['review']}")
    changed = [r for r in rows if r["gold_changed"]]
    if changed:
        L += ["", "## Gold changed by the author", ""]
        for r in changed:
            g = r["gold_changed"]
            trap = f'; trap was "{g["trap_was"]}"' if g.get("trap_was") else ""
            L.append(f'- **{r["qid"]}** ({g.get("by", "")}): gold was "{g.get("gold_was", "")}"{trap}. '
                     f'Why: {g.get("why", "")}')
    noted = [r for r in rows if r["notes"]]
    if noted:
        L += ["", "## Notes (non-blocking)", ""] + [f"- **{r['qid']}**: {r['notes']}" for r in noted]
    return "\n".join(L) + "\n"


def main() -> int:
    blocks = load_jsonl(BLOCKS)
    questions = load_jsonl(QUESTIONS)
    rows, errors = check(questions, blocks)
    OUT.write_text(render(rows, errors, questions), encoding="utf-8", newline="\n")
    print(f"wrote {OUT.relative_to(ROOT)}: {len(rows)} questions, {len(errors)} failed checks, "
          f"{sum(1 for r in rows if r['review'])} open review notes")
    for e in errors:
        print("  FAIL", e)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
