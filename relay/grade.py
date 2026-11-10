"""Grade an answer against the gold facts in data/questions.jsonl, with no model call.

    g = grade(question, reply_dict, pieces)       ->  {"label", "sure", "reasons", ...}
    .venv\\Scripts\\python -m relay.grade override ANSWER_ID LABEL --reason "..."   (append an overrule)

Labels (as in results.csv): correct | wrong | refused_correctly | refused_wrongly.
  correct            every gold fact is present, and each is cited to a passage from the right paper
                     that supports it
  wrong              a wrong fact or invented/borrowed number, a citation to another paper or to a
                     passage that doesn't support the claim (even if the facts are right), an
                     answer to an unanswerable question, or (unsure) a missing gold fact
  refused_correctly  unanswerable, the reply clearly says the papers don't have it, no guessed number
  refused_wrongly    answerable, but refused
Passages-only and busy replies are graded like any answer (it is what the person got): gold facts
present and cited -> correct; an unanswerable question answered without saying it isn't covered -> wrong.
How each check works:
  * a gold fact is present when all of its `match` patterns match the answer (they encode the
    fact; scripts/check_gold.py checks them against the fact's own wording);
  * a present fact is supported when one of the sentences mentioning it cites a passage that holds
    the fact's evidence block, or whose text matches the same patterns;
  * a citation to a passage from another paper makes the answer wrong;
  * `wrong_match` patterns (e.g. the base model's 3.3 x 10^18 for Q5) and, for unanswerable
    questions, `distractor_match` patterns (numbers a reply could borrow) mark wrong answers;
  * every cited sentence's words are compared with its cited passages: low overlap is flagged.
``sure`` is False whenever a pattern could be fooled: a missing fact when most are present, a fact
supported only by an unexpected passage, low overlap, a refusal worded as an answer, digits in a
refusal. Unsure grades are listed for a person to check.

Overrides: a person who disagrees appends a line to results/grade_overrides.jsonl (answer id,
label, reason). The tables apply it to that exact answer (the id hashes the question, the text and
the sources). Grades in results files are never edited by hand.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

from relay.pipeline import cited_ids, normalize_citations, split_sentences
from relay.retrieve import tokenize

ROOT = Path(__file__).resolve().parents[1]
QUESTIONS = ROOT / "data" / "questions.jsonl"
OVERRIDES = ROOT / "results" / "grade_overrides.jsonl"
LABELS = ("correct", "wrong", "refused_correctly", "refused_wrongly")
REFUSAL = re.compile(
    r"NOT_IN_PAPERS|papers? (don't|do not|does not|doesn't) (say|report|give|state|cover|mention|provide|include)|"
    r"(not|n't|never) (reported|report|stated|given|provided|mentioned|included|specified|measured|discussed|covered)|"
    r"no (information|data|mention|figure|number|measurement)s?\b|isn't (in|covered)|not in the (papers|sources)",
    re.I)
_DIGITS = re.compile(r"\d+(?:\.\d+)?")


def load_questions(path: Path = QUESTIONS) -> dict[str, dict]:
    with path.open(encoding="utf-8") as f:
        return {q["qid"]: q for q in (json.loads(x) for x in f if x.strip())}


def answer_id(qid: str, reply: dict) -> str:
    src = [s.get("pid") for s in reply.get("sources") or []]
    h = hashlib.sha256(json.dumps([qid, reply.get("kind"), reply.get("text"), src], ensure_ascii=False).encode("utf-8"))
    return h.hexdigest()[:16]


_SUP = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺", "0123456789-+")
_SUB = str.maketrans("₀₁₂₃₄₅₆₇₈₉ₜ", "0123456789t")
_HYPHENS = re.compile("[\u2010\u2011\u2012\u2013\u2014\u2212\u00ad]")
_SPACES = re.compile("[\u00a0\u2009\u200a\u202f\u2002\u2003\u2007]")


def normalize(text: str) -> str:
    """What the model writes -> plain text the gold patterns can match (added after the first real
    run, which used LaTeX, superscripts and non-breaking hyphens): unicode hyphens and spaces,
    superscript/subscript digits (10¹⁹ -> 10^19), and inline LaTeX (\\(O(n^{2}\\,d)\\) -> O(n^2 d))."""
    t = _HYPHENS.sub("-", _SPACES.sub(" ", normalize_citations(text)))
    t = re.sub(r"([⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺]+)", lambda m: "^" + m.group(1).translate(_SUP), t)
    t = t.translate(_SUB)
    t = re.sub(r"\\[()\[\]]", " ", t)                          # \( \) \[ \]
    t = re.sub(r"\\(?:text|mathrm|operatorname|mathbf|mathit)\{([^{}]*)\}", r"\1", t)
    t = re.sub(r"\\frac\{([^{}]*)\}\{([^{}]*)\}", r"\1/\2", t)
    t = re.sub(r"\\sqrt\{([^{}]*)\}", r"√\1", t)
    t = re.sub(r"\\(?:cdot|times)\b", "·", t)
    t = re.sub(r"\\(?:,|;|!|quad|qquad)", " ", t)
    t = re.sub(r"\\(?:ge|geq)\b", "≥", t)
    t = re.sub(r"\\(?:le|leq)\b", "≤", t)
    t = re.sub(r"([\^_])\{([^{}]*)\}", r"\1\2", t)             # ^{2} -> ^2, _{t-1} -> _t-1
    t = re.sub(r"\\([A-Za-z]+)", r"\1", t)                      # any other \command -> command
    return re.sub(r"[ \t]+", " ", t)


def _all(patterns: list[str], text: str) -> bool:
    return all(re.search(p, text, re.I) for p in patterns)


def _any(patterns: list[str], text: str) -> list[str]:
    return [p for p in patterns if p and re.search(p, text, re.I)]


def _overlap(sentence: str, texts: list[str]) -> float:
    s = set(tokenize(re.sub(r"\[\s*S?\s*\d+(?:\s*[,;]\s*S?\s*\d+)*\s*\]", " ", sentence)))
    if not s:
        return 1.0
    pool = set()
    for t in texts:
        pool |= set(tokenize(t))
    return len(s & pool) / len(s)


def _new_digits(text: str, question: str) -> list[str]:
    q = set(_DIGITS.findall(question))
    return [d for d in _DIGITS.findall(re.sub(r"\[\s*S?\s*\d+[^\]]*\]", " ", text)) if d not in q]


def grade(q: dict, reply: dict, pieces: dict[str, dict]) -> dict:
    """``reply``: Reply.to_dict() (kind, text, sources[{sid, pid, paper, block_ids}]).
    ``pieces``: pid -> {"text", "paper", "block_ids"} for the setting the reply was made with."""
    kind, text = reply.get("kind"), normalize(reply.get("text") or "")
    out = {"qid": q["qid"], "answer_id": answer_id(q["qid"], reply), "label": None, "sure": True,
           "reasons": [], "facts": [], "wrong_paper_citations": [], "borrowed": []}
    # passages-only and busy replies are graded like any answer: it is what the person got
    # (gold facts present and cited -> correct; an unanswerable question without saying it isn't
    # covered -> wrong). Decided by the author, 2026-11-10.
    if kind not in ("answer", "refusal", "passages", "busy"):
        out.update(label=None, sure=False, reasons=[f"no gradable reply (kind={kind})"])
        return out
    refusal_words = bool(REFUSAL.search(text))

    # ------------------------------------------------ unanswerable
    if not q["answerable"]:
        borrowed = _any(q.get("distractor_match") or [], text)
        out["borrowed"] = borrowed
        if borrowed:
            out.update(label="wrong", reasons=[f"borrows a nearby number: /{borrowed[0]}/"])
            return out
        digits = _new_digits(text, q["question"])
        if kind == "refusal":
            out["label"] = "refused_correctly"
            if digits:
                out.update(sure=False, reasons=[f"refusal contains numbers {digits[:5]}: check none is guessed"])
            return out
        if refusal_words:
            out.update(label="refused_correctly", sure=False,
                       reasons=["answered with citations but says the papers don't have it: check it is a clear refusal"]
                       + ([f"contains numbers {digits[:5]}"] if digits else []))
            return out
        out.update(label="wrong", reasons=["answered a question the papers don't answer"])
        return out

    # ------------------------------------------------ answerable
    if kind == "refusal":
        out.update(label="refused_wrongly", reasons=["refused an answerable question"])
        return out
    sources = {s["sid"]: s for s in reply.get("sources") or []}
    # a passages reply tags each quoted excerpt once, at the start of its line: the line is the cited unit
    sents = ([ln for ln in text.splitlines() if cited_ids(ln)] if kind == "passages" else split_sentences(text))
    cites = [[f"S{i}" for i in cited_ids(s)] for s in sents]
    for s, cs in zip(sents, cites):
        for sid in cs:
            src = sources.get(sid)
            if src and src.get("paper") != q["paper"]:
                out["wrong_paper_citations"].append({"sentence": s[:160], "sid": sid, "paper": src["paper"]})
    if out["wrong_paper_citations"]:
        w = out["wrong_paper_citations"][0]
        out["reasons"].append(f"cites {w['sid']} from {w['paper']}, not {q['paper']}")

    def piece_text(sid):
        src = sources.get(sid)
        p = pieces.get(src["pid"]) if src else None
        return normalize(p["text"]) if p else ""

    def piece_blocks(sid):
        src = sources.get(sid)
        return set(src.get("block_ids") or []) if src else set()

    unsure = []
    for i, f in enumerate(q["facts"], 1):
        pats = f["match"]
        present = _all(pats, text)
        rec = {"fact": f["fact"], "present": present, "supported": None, "how": ""}
        if present:
            idx = [j for j, s in enumerate(sents) if _any(pats, s)]
            cited = sorted({sid for j in idx for sid in cites[j]})
            ev_blocks = {ev["block_id"] for ev in f["evidence"]}
            if any(piece_blocks(sid) & ev_blocks for sid in cited):
                rec.update(supported=True, how="cites the gold evidence passage")
            elif any(_all(pats, piece_text(sid)) for sid in cited if sources.get(sid, {}).get("paper") == q["paper"]):
                rec.update(supported=True, how="cited passage states the same fact (not the gold block)")
            else:
                rec.update(supported=False, how=f"sentences mentioning it cite {cited or 'nothing'}, which don't hold it")
                unsure.append(f"fact {i} may be unsupported by its citation: {f['fact']}")
        out["facts"].append(rec)
    missing = [r["fact"] for r in out["facts"] if not r["present"]]
    unsupported = [r["fact"] for r in out["facts"] if r["present"] and r["supported"] is False]
    wrong = [w for w in q.get("wrong_match") or [] if re.search(w["pattern"], text, re.I)]
    low = []
    for s, cs in zip(sents, cites):
        if cs and len(re.findall(r"[A-Za-z]{3,}", s)) >= 5:
            ov = _overlap(s, [piece_text(sid) for sid in cs])
            if ov < 0.5:
                low.append(f"low word overlap ({ov:.2f}) with its citations: {s[:120]}")

    if out["wrong_paper_citations"]:
        out["label"] = "wrong"
        return out
    if wrong:
        out["reasons"].append(f"states {wrong[0]['what']}")
        out["label"] = "wrong"
        if not missing:
            out["sure"] = False
            out["reasons"].append("all gold facts are also present: check it isn't just a comparison")
        return out
    if refusal_words and len(missing) == len(q["facts"]):
        out.update(label="refused_wrongly", sure=False, reasons=["reads as a refusal of an answerable question"])
        return out
    if missing:
        out["label"] = "wrong"
        out["reasons"].append(f"missing {len(missing)}/{len(q['facts'])} gold facts: " + "; ".join(missing))
        if len(missing) <= len(q["facts"]) / 2:
            out["sure"] = False
            out["reasons"].append("most facts present: check the pattern didn't miss a paraphrase")
        out["reasons"] += unsure
        return out
    if unsupported:
        out["label"] = "wrong"
        out["sure"] = False
        out["reasons"] += unsure
        return out
    out["label"] = "correct"
    if low:
        out["sure"] = False
        out["reasons"] += low[:3]
    return out


# ----------------------------------------------------------------------------- overrides
def load_overrides(path: Path = OVERRIDES) -> dict[str, dict]:
    """answer_id -> the latest override for it (the file is append-only; later lines win)."""
    out: dict[str, dict] = {}
    if path.exists():
        with path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    o = json.loads(line)
                    out[o["answer_id"]] = o
    return out


def add_override(answer: str, label: str, reason: str, qid: str | None = None, auto_label: str | None = None,
                 by: str = "author", path: Path = OVERRIDES) -> dict:
    if label not in LABELS:
        raise ValueError(f"label must be one of {LABELS}")
    if not reason or not reason.strip():
        raise ValueError("an override needs a reason")
    o = {"answer_id": answer, "qid": qid, "auto_label": auto_label, "label": label, "reason": reason.strip(),
         "by": by, "at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(o, ensure_ascii=False) + "\n")
    return o


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m relay.grade")
    sub = ap.add_subparsers(dest="cmd", required=True)
    o = sub.add_parser("override", help="overrule one auto grade (appended to results/grade_overrides.jsonl)")
    o.add_argument("answer_id")
    o.add_argument("label", choices=LABELS)
    o.add_argument("--reason", required=True)
    o.add_argument("--by", default="author")
    a = ap.parse_args(argv)
    grades = ROOT / "results" / "quality" / "grades.jsonl"
    known = {}
    if grades.exists():
        with grades.open(encoding="utf-8") as f:
            for line in f:
                g = json.loads(line)
                known[g["answer_id"]] = g
    if a.answer_id not in known:
        print(f"unknown answer id {a.answer_id} (see results/quality/grades.jsonl)", file=sys.stderr)
        return 2
    g = known[a.answer_id]
    rec = add_override(a.answer_id, a.label, a.reason, g.get("qid"), g.get("auto_label"), a.by)
    print(json.dumps(rec, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
