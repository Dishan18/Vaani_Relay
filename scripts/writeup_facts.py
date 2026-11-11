"""Every number and quoted text the write-up uses, computed from files in the repo (rule 3).

    .venv\\Scripts\\python scripts\\writeup_facts.py            -> docs/writeup/facts.json
    .venv\\Scripts\\python scripts\\writeup_facts.py --check    exit 1 if facts.json is stale

Each fact is {"value", "source", "how"}: source is the file(s) it is computed from, how is the
computation in words. Sources are generated files (results.csv, tables, experiment results,
calibration, ledger, raw runs) or the project's own inputs (config/pipeline.toml, data/questions.jsonl,
data/limits_documented.json), marked "input". scripts/render_writeup.py substitutes them into
docs/WRITEUP.tmpl.md; scripts/check_numbers.py fails on any number in the template's own text.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import tomllib
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

OUT = ROOT / "docs" / "writeup" / "facts.json"
RESULTS = ROOT / "results" / "results.csv"
CONFIG = ROOT / "config" / "pipeline.toml"
CAL = ROOT / "data" / "calibration.json"
DOC = ROOT / "data" / "limits_documented.json"
QUESTIONS = ROOT / "data" / "questions.jsonl"
WINNER = ROOT / "results" / "quality" / "winner.json"
TRAPS = ROOT / "data" / "retrieval" / "traps.json"
R4 = ROOT / "data" / "experiments" / "r4_retry_policy" / "result.json"
R4B = ROOT / "data" / "experiments" / "r4b_policy_vs_naive"
REPLAY = ROOT / "data" / "experiments" / "gate_replay" / "result.json"
LEDGER = ROOT / "results" / "ledger" / "groq_calls.jsonl"
EXCLUDED = ROOT / "results" / "excluded_runs.json"
OVERRIDES = ROOT / "results" / "grade_overrides.jsonl"
DETAIL = ROOT / "data" / "results" / "detail.jsonl"
RUNS = ROOT / "results" / "raw" / "runs"
BASELINE = "e7aa65f"                       # Table 3 and the real load runs as measured
REAL_RUNS = {"real_steady": "real_steady-20261111T073012Z", "real_burst": "real_burst-20261111T073012Z"}
DAILY_SCENARIO = "daily_exhausted"                # its run that counts in the tables gives the 3pm reply

facts: dict[str, dict] = {}


def rel(p: Path) -> str:
    return p.relative_to(ROOT).as_posix()


def F(key: str, value, source, how: str) -> None:
    srcs = [source] if isinstance(source, (str, Path)) else list(source)
    facts[key] = {"value": value, "source": [rel(s) if isinstance(s, Path) else s for s in srcs], "how": how}


def load_jsonl(p: Path) -> list[dict]:
    with p.open(encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def nearest_rank(xs: list[float], q: float) -> float | None:
    xs = sorted(xs)
    return xs[max(0, math.ceil(q * len(xs)) - 1)] if xs else None


def pct1(a: int, n: int) -> str:
    return f"{100 * a / n:.1f}" if n else "not measured"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="exit 1 if docs/writeup/facts.json differs from a fresh run")
    a = ap.parse_args(argv)
    build()
    text = json.dumps(facts, indent=1, ensure_ascii=False) + "\n"
    if a.check:
        old = OUT.read_text(encoding="utf-8") if OUT.exists() else ""
        if old != text:
            print("docs/writeup/facts.json is stale: run scripts/writeup_facts.py", file=sys.stderr)
            return 1
        print(f"facts.json up to date ({len(facts)} facts)")
        return 0
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text, encoding="utf-8", newline="\n")
    print(f"{len(facts)} facts -> {rel(OUT)}")
    return 0


def build() -> None:
    cfg = tomllib.loads(CONFIG.read_text(encoding="utf-8"))
    cal = json.loads(CAL.read_text(encoding="utf-8"))["models"]
    doc = json.loads(DOC.read_text(encoding="utf-8"))["models"]
    with RESULTS.open(encoding="utf-8", newline="") as f:
        allrows = list(csv.DictReader(f))
    rows = [r for r in allrows if r["superseded"] == "0" and not r.get("excluded")]

    # ---------------------------------------------------------------- inputs: design
    F("cfg.max_wait_s", cfg["ladder"]["max_wait_s"], CONFIG, "input: [ladder] max_wait_s")
    F("cfg.k_full", cfg["retrieval"]["k_full"], CONFIG, "input: [retrieval] k_full")
    F("cfg.k_lean", cfg["retrieval"]["k_lean"], CONFIG, "input: [retrieval] k_lean")
    F("cfg.piece_chars", cfg["retrieval"]["piece_chars"], CONFIG, "input: [retrieval] piece_chars")
    qs = load_jsonl(QUESTIONS)
    F("q.n", len(qs), QUESTIONS, "input: questions written for the evaluation")
    F("q.unanswerable", sum(q.get("answerable") is False for q in qs), QUESTIONS,
      "input: questions marked answerable = false (the papers don't answer them)")

    # ---------------------------------------------------------------- per scenario outcomes (results.csv)
    by = defaultdict(list)
    for r in rows:
        if not r["scenario"].startswith("eval:"):
            by[(r["scenario"], r["source"])].append(r)
    for (sc, src), rs in by.items():
        n = len(rs)
        k = f"{sc}.{'real' if src == 'real' else 'standin'}"
        lvl = lambda r: r["degraded_level"]
        written = [r for r in rs if r["status"] == "ok" and lvl(r) not in ("3", "4")]
        F(f"{k}.n", n, RESULTS, f"rows of scenario {sc}, source {src}, counted in the tables")
        F(f"{k}.written_pct", pct1(len(written), n), RESULTS, "status ok and degraded_level < 3 (an answer or a "
          "refusal, written by a model), % of requests")
        F(f"{k}.weaker_pct", pct1(sum(1 for r in written if lvl(r) in ("1", "2")), n), RESULTS,
          "written at degraded_level 1 (another model) or 2 (lean), % of requests")
        F(f"{k}.passages_budget_pct", pct1(sum(1 for r in rs if r["status"] == "rate_limited"), n), RESULTS,
          "status rate_limited: cited passages because budget had no drafter in time, %")
        F(f"{k}.passages_gate_pct", pct1(sum(1 for r in rs if r["status"] == "ok" and lvl(r) == "3"), n), RESULTS,
          "status ok at degraded_level 3: cited passages after both drafts failed the citation check, %")
        F(f"{k}.gave_up_pct", pct1(sum(1 for r in rs if r["status"] == "timeout"), n), RESULTS,
          "status timeout (the simulated person left first), %")
        tt = [float(r["first_token_ms"]) for r in rs if r["first_token_ms"]]
        F(f"{k}.ttft_n", len(tt), RESULTS, "requests with a first token")
        for q in (50, 95):
            v = nearest_rank(tt, q / 100)
            F(f"{k}.ttft_p{q}", "not measured" if v is None else f"{v:.0f}", RESULTS, f"first_token_ms p{q}, nearest rank")
        qw = [float(r["queue_wait_ms"]) for r in rs if r["queue_wait_ms"]]
        v = nearest_rank(qw, .95)
        F(f"{k}.queue_wait_p95", "not measured" if v is None else f"{v:.0f}", RESULTS, "queue_wait_ms p95, nearest rank")
    # one combined number for the comparison of real and twin
    for sc in ("real_steady", "real_burst"):
        r_ = facts[f"{sc}.real.written_pct"]["value"]
        s_ = facts[f"{sc}.standin.written_pct"]["value"]
        F(f"{sc}.gap_pts", f"{float(s_) - float(r_):.1f}", RESULTS, "stand-in written % minus real written %")

    # ---------------------------------------------------------------- quality (Table 3 inputs)
    ev = [r for r in rows if r["scenario"] == "eval:retrieval" and r["source"] == "real"]
    by_set = defaultdict(list)
    for r in ev:
        by_set[r["retrieval_setting"]].append(r)
    for s, rs in by_set.items():
        n = len(rs)
        key = s.replace("+", "_")
        F(f"q3.{key}.n", n, RESULTS, f"eval:retrieval rows at setting {s}")
        F(f"q3.{key}.correct_pct", pct1(sum(r["answer_grade"] == "correct" for r in rs), n), RESULTS, "answer_grade correct, %")
        F(f"q3.{key}.refused_pct", pct1(sum(r["answer_grade"] == "refused_correctly" for r in rs), n), RESULTS,
          "answer_grade refused_correctly, %")
        tok = [int(r["tokens_in"] or 0) + int(r["tokens_out"] or 0) for r in rs]
        F(f"q3.{key}.tokens", f"{sum(tok) / n:.0f}", RESULTS, "tokens_in + tokens_out per request, mean")
    win = json.loads(WINNER.read_text(encoding="utf-8"))
    F("q3.chosen", win["setting"]["chosen"], WINNER,
      "the setting make_tables.py picked: most correct, ties within one question broken by fewest tokens")

    # ---------------------------------------------------------------- limits (calibration, documented)
    q = cal["qwen/qwen3.8-27b"]["buckets"]
    F("lim.qwen_otpm", f"{q['OTPM']['capacity']:.0f}", CAL, q["OTPM"]["source"])
    F("lim.qwen_itpm", f"{q['ITPM']['capacity']:.0f}", CAL, q["ITPM"]["source"])
    F("lim.qwen_tpm_doc", doc["qwen/qwen3.8-27b"]["tpm"], DOC, "input: Groq's documented TPM for the model")
    F("lim.otpm_share", f"{100 * q['OTPM']['capacity'] / doc['qwen/qwen3.8-27b']['tpm']:.1f}", [CAL, DOC],
      "hidden output-token limit as % of the documented TPM")
    b = cal["openai/gpt-oss-120b"]
    probe = json.loads((ROOT / "data" / "probe" / "calibration.json").read_text(encoding="utf-8"))
    F("probe.calls", probe["calls"], ROOT / "data" / "probe" / "calibration.json", "real calls the probe logged")
    F("cal.ttfb_n", b["ttfb_ms"]["n"], CAL, "probe calls behind the 120b time-to-first-byte fit")
    F("cal.ttfb_p50", f"{b['ttfb_ms']['p50_target']:.0f}", CAL, "120b server time to first byte p50 (ms), probe")
    F("cal.decode_tps", f"{b['decode']['tokens_per_s']:.0f}", CAL, "120b decode tokens/s p50, probe")
    holds = b.get("admission_extra_samples") or []
    if holds:
        F("cal.hold_max", max(holds), CAL, "largest extra admission hold (tokens) seen in real 120b TPM 429s")

    # ---------------------------------------------------------------- slow request (explain_request)
    import explain_request as ex
    det = [d for d in ex.load(DETAIL) if d.get("run_id") == REAL_RUNS["real_burst"]]
    det.sort(key=lambda d: -(d.get("released_ms") if d.get("released_ms") is not None else d.get("waited_ms") or 0))
    d = det[0]
    s = ex.split(d)
    F("slow.request_id", d["request_id"], DETAIL, "slowest request of the real burst run (scripts/explain_request.py --slowest)")
    F("slow.qid", d.get("qid"), DETAIL, "its question")
    F("slow.total_ms", f"{s['end_ms']:.0f}", DETAIL, "arrival to release, or to when the person left (ms)")
    for part in ("waiting", "working", "retrying", "local"):
        F(f"slow.{part}_ms", f"{s['split_ms'][part]:.0f}", DETAIL, f"time classed as {part} by explain_request.split")
    F("slow.patience_s", f"{d.get('patience_s', 0):.1f}", DETAIL, "the simulated person's patience (s)")
    spans = {x["name"]: x.get("outcome") for x in d.get("spans", [])}
    F("slow.plan", spans.get("plan"), DETAIL, "the ladder level its plan chose (span 'plan')")
    F("slow.check", spans.get("check"), DETAIL, "the citation gate's verdict on its draft (span 'check'; no reason recorded)")
    iv = ex.intervals(d)
    waits = [(b - a) for a, b, kind, _ in iv if kind == "waiting"]
    works = [(b - a) for a, b, kind, _ in iv if kind == "working"]
    F("slow.first_wait_ms", f"{waits[0]:.0f}", DETAIL, "the first waiting interval (budget for the draft)")
    F("slow.last_wait_ms", f"{waits[-1]:.0f}", DETAIL, "the last waiting interval (for a second draft)")
    F("slow.draft_ms", f"{works[0]:.0f}" if works else "0", DETAIL, "the first working interval (the draft call)")
    last_wait = [x for x in iv if x[2] == "waiting"][-1]
    F("slow.last_wait_on", next((m.split("/")[-1] for m in ("openai/gpt-oss-20b", "openai/gpt-oss-120b", "qwen/qwen3.8-27b")
                                 if m in last_wait[3]), "unknown"), DETAIL, "the model the last wait was for (span label)")

    # ---------------------------------------------------------------- retrieval traps (no model calls)
    traps = json.loads(TRAPS.read_text(encoding="utf-8"))
    tq = {t["qid"]: t for t in traps["rows"]}
    size = str(traps["size"])
    for qid in ("Q14", "Q5", "Q17"):
        F(f"trap.{qid}.question", tq[qid]["question"], TRAPS, "the question text")
    F("trap.size", traps["size"], TRAPS, "piece size the traps were measured at (chars)")
    F("trap.Q14.shared", len(tq["Q14"]["shared_terms"]), TRAPS,
      "question terms (after stemming and stopwords) found in the passage that holds the answer")
    F("trap.Q14.terms", len(tq["Q14"]["question_terms"]), TRAPS, "question terms (after stemming and stopwords)")
    F("trap.Q5.rank", tq["Q5"]["ranks"][size]["right_or_distractor_rank"], TRAPS, "rank of the right piece, plain BM25")
    F("trap.Q5.top_cite", tq["Q5"]["top_hits"][0]["cite"], TRAPS, "the piece BM25 puts first")
    F("trap.Q17.wrong_rank", tq["Q17"]["ranks"][size]["first_wrong_paper_rank"], TRAPS,
      "rank of the first piece from another paper (the question is about H-Net)")
    F("trap.Q17.top_cite", tq["Q17"]["top_hits"][0]["cite"], TRAPS, "the piece BM25 puts first")
    F("trap.Q17.top_paper", tq["Q17"]["top_hits"][0]["paper"], TRAPS, "its paper")
    F("trap.Q17.shared_terms", ", ".join(tq["Q17"]["shared_terms"]), TRAPS, "question terms that piece shares")
    F("trap.Q17.distractor_rank", tq["Q17"]["ranks"][size]["right_or_distractor_rank"], TRAPS,
      "rank of the first piece holding a distractor (a number a reply could borrow)")

    ev_all = [r for r in rows if r["scenario"] == "eval:retrieval" and r["source"] == "real"]
    F("trap.Q14.correct", sum(r["answer_grade"] == "correct" for r in ev_all if r["request_id"].endswith("-Q14")), RESULTS,
      "Q14 answers graded correct, over all retrieval settings")
    F("trap.Q14.asked", sum(r["request_id"].endswith("-Q14") for r in ev_all), RESULTS, "Q14 answers, all settings")
    for qid in ("Q12", "Q17"):
        F(f"trap.{qid}.refused", sum(r["answer_grade"] == "refused_correctly" for r in ev_all
                                     if r["request_id"].endswith(f"-{qid}")), RESULTS, f"{qid} correctly refused, all settings")
        F(f"trap.{qid}.asked", sum(r["request_id"].endswith(f"-{qid}") for r in ev_all), RESULTS, f"{qid} asked, all settings")

    # ---------------------------------------------------------------- retries
    r4 = json.loads(R4.read_text(encoding="utf-8"))["summary"]
    for arm in r4:
        for seed, v in r4[arm].items():
            F(f"r4.{arm}.s{seed}.no_written", v["no_written_answer"], R4, f"arm {arm}, seed {seed}: people without a written answer")
            F(f"r4.{arm}.s{seed}.n", v["requests"], R4, f"arm {arm}, seed {seed}: requests")
            F(f"r4.{arm}.s{seed}.ttft_p95", f"{v['ttft_p95_ms']:.0f}", R4, f"arm {arm}, seed {seed}: TTFT p95 (ms)")
    r4b = json.loads((R4B / "result.json").read_text(encoding="utf-8"))["summary"]["current"]
    for seed, v in r4b.items():
        for sc, w in v["by_scenario"].items():
            k = "pipe" if sc == "overload" else "naive"
            F(f"r4b.{k}.s{seed}.no_written", w["no_written_answer"], R4B / "result.json",
              f"seed {seed}, {sc}: people without a written answer")
            F(f"r4b.{k}.s{seed}.n", w["requests"], R4B / "result.json", f"seed {seed}, {sc}: requests")
            F(f"r4b.{k}.s{seed}.http_429", w["http_429"], R4B / "result.json", f"seed {seed}, {sc}: 429s from the stand-in")
    # the worst naive request on seed 7, against the same arrival under the pipeline's policy
    naive = [x for x in load_jsonl(R4B / "rows" / "current-seed7-overload_naive_retry.jsonl") if x.get("type") == "answer"]
    pipe = {x["request_id"].rsplit("-", 1)[1]: x for x in load_jsonl(R4B / "rows" / "current-seed7-overload.jsonl")
            if x.get("type") == "answer"}

    def failed(x):
        return sum(1 for c in x["reply"].get("model_calls") or [] if c.get("status") != 200)
    worst = max((x for x in naive if x["reply"]["kind"] == "passages"), key=lambda x: (failed(x), x["reply"].get("released_ms") or 0))
    twin = pipe[worst["request_id"].rsplit("-", 1)[1]]
    src = [R4B / "rows" / "current-seed7-overload_naive_retry.jsonl", R4B / "rows" / "current-seed7-overload.jsonl"]
    F("naive.qid", worst["qid"], src, "the naive-retry request (seed 7) with the most failed calls that ended in passages")
    F("naive.failed_calls", failed(worst), src, "its calls that got no 200 (429s)")
    F("naive.ms", f"{worst['reply'].get('released_ms'):.0f}", src, "its time to release (ms)")
    F("naive.pipe_kind", twin["reply"]["kind"], src, "what the pipeline's policy gave the same arrival")
    F("naive.pipe_ms", f"{twin['reply'].get('released_ms'):.0f}", src, "and how fast (ms)")
    F("cfg.retry_enabled", "on" if cfg["retry"]["enabled"] else "off", CONFIG, "input: [retry] enabled")
    F("cfg.retry_max", cfg["retry"]["max_per_step"], CONFIG, "input: [retry] max_per_step (the policy tested)")
    F("cfg.retry_budget_pct", f"{100 * cfg['retry']['global_budget_fraction']:.0f}", CONFIG,
      "input: [retry] global_budget_fraction, as %")
    import run_load as rl
    F("naive.retries", rl.NAIVE["retries"], ROOT / "scripts" / "run_load.py", "input: NAIVE retries per call")
    F("naive.backoff_s", rl.NAIVE["backoff_s"], ROOT / "scripts" / "run_load.py", "input: NAIVE first back-off (s)")
    F("load.overload_rate", rl.SCENARIOS["overload"].rate_per_min, ROOT / "scripts" / "run_load.py",
      "input: overload's offered arrivals per minute")

    # ---------------------------------------------------------------- the reply when the daily budget is gone
    daily_ids = [r["request_id"] for r in rows if r["scenario"] == DAILY_SCENARIO and r["source"] == "stand-in"]
    DAILY_RUN = daily_ids[0].rsplit("-", 1)[0]
    daily = [x for x in load_jsonl(RUNS / f"{DAILY_RUN}.jsonl") if x.get("type") == "answer"]
    pick = min(daily, key=lambda x: len(x["reply"]["text"]))
    F("daily.request_id", pick["request_id"], RUNS / f"{DAILY_RUN}.jsonl", "the shortest reply of the daily_exhausted run")
    F("daily.question", pick["reply"]["question"], RUNS / f"{DAILY_RUN}.jsonl", "its question")
    F("daily.reply", pick["reply"]["text"], RUNS / f"{DAILY_RUN}.jsonl", "the reply text the person got, verbatim")
    F("daily.ms", f"{pick['reply'].get('released_ms'):.0f}", RUNS / f"{DAILY_RUN}.jsonl", "its time to release (ms)")
    F("daily.eta_s", pick["reply"].get("eta_s"), RUNS / f"{DAILY_RUN}.jsonl", "the ETA it gave (s)")
    from relay.grade import grade, load_questions
    from relay.retrieve import Retriever
    size = pick["setting_def"]["piece_chars"]
    pieces = {p.pid: {"text": p.text, "paper": p.paper, "block_ids": p.block_ids} for p in Retriever.from_blocks(size).pieces}
    g = grade(load_questions()[pick["qid"]], pick["reply"], pieces)
    F("daily.qid", pick["qid"], RUNS / f"{DAILY_RUN}.jsonl", "its question id")
    F("daily.grade", g["label"] or "not graded", [RUNS / f"{DAILY_RUN}.jsonl", QUESTIONS],
      "relay.grade on this stand-in reply, for the write-up only (stand-in rows carry no grade in results.csv)")

    # ---------------------------------------------------------------- honesty notes
    rep = json.loads(REPLAY.read_text(encoding="utf-8"))
    dr = rep["drafts"]
    F("replay.drafts", len(dr), REPLAY, "real drafts replayed through the old and new gate")
    F("replay.old_pass", sum(x["old_ok"] for x in dr), REPLAY, "drafts the gate as measured at e7aa65f passes")
    F("replay.new_pass", sum(x["new_ok"] for x in dr), REPLAY, "drafts the fixed gate passes")
    F("replay.dropped", sum(len(x["dropped"]) for x in dr if x["new_ok"]), REPLAY, "sentences the fixed gate dropped")
    ch = [q for q in rep["requests"] if q["in_tables"] and
          (q["new_kind"] != q["old_kind"] or (q["new_kind"] == "answer" and q["new_grade"].split(" ")[0] != q["old_grade"]))]
    F("replay.req_changed", len(ch), REPLAY, "requests counted in the tables whose outcome the fixed gate changes")
    F("replay.req_in_tables", sum(q["in_tables"] for q in rep["requests"]), REPLAY, "requests counted in the tables with a draft")
    F("baseline", BASELINE, "git", "the commit the tables and real load runs were measured at")
    graded = [r for r in rows if r["source"] == "real" and r["answer_grade"]]
    F("grades.n", len(graded), RESULTS, "real rows counted in the tables that carry a grade")
    F("grades.unsure", sum(r["grade_sure"] == "0" for r in graded), RESULTS,
      "of those, graded automatically with the grader unsure and not checked by hand (grade_sure 0)")
    F("grades.overrides", len(load_jsonl(OVERRIDES)), OVERRIDES, "grades set by hand (author's overrides)")
    led = load_jsonl(LEDGER)
    F("ledger.cap100_calls", sum(1 for e in led if e.get("account_cap_fraction") == 1.0 and e.get("status") == 200),
      LEDGER, "real calls answered (200) with the account cap at 100% (the two key1 eval jobs)")
    refill = [e for e in led if e.get("ledger_accounting") == "refill"]
    F("ledger.refill_calls", len(refill), LEDGER, "real calls made under refill accounting (the two real load runs)")
    F("ledger.refill_429", sum(e.get("status") == 429 for e in refill), LEDGER, "of those, 429s")
    from relay.ledger import charge_of
    pts = sorted(((e["ts"], charge_of(e)) for e in led if e.get("source") == "real" and e.get("model") == "openai/gpt-oss-120b"
                  and e.get("account", "key1") == "key1"), key=lambda x: x[0])
    peak, j, run = 0, 0, 0
    for t, c in pts:
        run += c
        while pts[j][0] < t - 86400:
            run -= pts[j][1]
            j += 1
        peak = max(peak, run)
    F("ledger.max24h_120b", peak, LEDGER, "most gpt-oss-120b tokens key1 was charged (reported usage) in any 24 h window")
    F("lim.tpd_120b", doc["openai/gpt-oss-120b"]["tpd"], DOC, "input: Groq's documented TPD for gpt-oss-120b")
    bud = tomllib.loads((ROOT / "config" / "budget.toml").read_text(encoding="utf-8"))
    F("budget.window_h", bud["window_hours"], ROOT / "config" / "budget.toml", "input: the ledger's window (hours)")
    F("status.relabelled", sum(1 for r in rows if r["status"] == "ok"
                               and "failed the citation check" in (r.get("degraded_reason") or "")), RESULTS,
      "rows counted in the tables that are status ok after failed checks: passages replies that were status "
      "rate_limited when the tables were measured at the baseline (computed from results.csv alone, no git needed)")
    ex_runs = json.loads(EXCLUDED.read_text(encoding="utf-8"))["runs"]
    F("excluded.key2", sum("key2" in v for v in ex_runs.values()), EXCLUDED, "runs excluded because they ran on a second key")
    F("excluded.bad_key", sum("401" in v for v in ex_runs.values()), EXCLUDED, "runs excluded because every call got 401")


if __name__ == "__main__":
    raise SystemExit(main())
