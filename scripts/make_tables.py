"""Tables from results/results.csv only (C9: reviewers recompute them from that file).

    .venv\\Scripts\\python scripts\\make_tables.py [--results results/results.csv] [--out results/tables.md]
        ->  results/tables.md, results/quality/winner.json

Table 3 (retrieval settings; scenario eval:retrieval) and the drafter comparison (scenario
eval:drafter, plus the eval:retrieval rows of the same setting and drafter, which ran with the same
configuration). Latest answer per question only (superseded = 0).
  correct and cited % = correct / n; wrong % = (wrong + refused_wrongly) / n; correctly refused % =
  refused_correctly / n; the three add to 100 when every row is graded (ungraded rows are counted
  and shown, never folded into a label).
  tokens per answer = tokens_in + tokens_out, every model call in the request, mean.
  TTFT = total here: the reply is released only after its checks. Both include budget waits; the
  eval waits for budget rather than degrade, so latency here is not a serving-latency result.
  Percentiles are nearest-rank, with n.
Quality comes from real rows only. A stand-in row never carries a grade (its reason is in
grade_note); one that does stops this script. Every table that
reports quality shows the ungraded share, and the coverage table lists every scenario and source with
how many rows are graded and why the rest aren't, so an ungraded row can't drop out of sight.
Choosing (the author's rule): only settings with every question answered are eligible; settings within one correct answer of the best are a tie (noise at
18 questions); among tied settings the cheapest (tokens per answer) wins. Drafters: same tie rule;
among tied drafters the one with fewer wrong answers, then the lower total p50, wins.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


CAL = ROOT / "data" / "calibration.json"
PROBE = ROOT / "data" / "probe" / "calibration.json"
LEDGER = ROOT / "results" / "ledger" / "groq_calls.jsonl"
SHORT = {"openai/gpt-oss-20b": "20b", "openai/gpt-oss-120b": "120b", "qwen/qwen3.8-27b": "qwen"}


def rpd_refill(ledger: Path = LEDGER, model: str = "openai/gpt-oss-120b") -> dict | None:
    """The largest rise in x-ratelimit-remaining-requests between two consecutive calls of one key
    (key1: entries without an account label, or labelled key1) on the same UTC day, so a reset at
    00:00 UTC can't explain it, and less than 24 h after the spends, so a rolling window can't either."""
    if not ledger.exists():
        return None
    pts = []
    with ledger.open(encoding="utf-8") as f:
        for line in f:
            e = json.loads(line)
            if (e.get("source") == "real" and e.get("model") == model and e.get("hdr_remaining_requests") is not None
                    and e.get("account", "key1") == "key1"):
                pts.append((e["ts"], int(e["hdr_remaining_requests"]), e["ts_iso"]))
    pts.sort()
    best = None
    for (t0, r0, i0), (t1, r1, i1) in zip(pts, pts[1:]):
        if i0[:10] != i1[:10]:
            continue
        if r1 - r0 > 0 and (best is None or r1 - r0 > best["rise"]):
            best = {"from": r0, "to": r1, "rise": r1 - r0, "hours": round((t1 - t0) / 3600, 2), "at": i0, "until": i1}
    return best


def daily_429s(raw: Path = ROOT / "results" / "raw") -> tuple[int, int]:
    """(429 responses whose body names a per-day limit, all 429 responses) in every saved real call:
    results/raw and the probe's log."""
    daily = total = 0
    for p in [*raw.rglob("*.jsonl"), ROOT / "data" / "probe" / "probe_raw.jsonl"]:
        with p.open(encoding="utf-8") as f:
            for line in f:
                if '"status": 429' not in line:
                    continue
                o = json.loads(line)
                rec = o.get("record") or o
                if rec.get("status") != 429 or rec.get("source", "real") != "real" or o.get("mode", "real") != "real":
                    continue
                total += 1
                body = (rec.get("error_body") or "").lower()
                daily += ("per day" in body)
    return daily, total


def table1(cal_path: Path = CAL, probe_path: Path = PROBE) -> list[str]:
    """Table 1 in the brief's rows: value found, how, confidence."""
    cal = json.loads(cal_path.read_text(encoding="utf-8"))["models"]
    probe = json.loads(probe_path.read_text(encoding="utf-8"))
    ver = probe["verdicts"]
    walls = probe["requests_window_and_pooling"][0]["per_model"]
    first = next(w["first_429"] for w in walls.values() if w.get("first_429"))
    rec = [w["first_recovery_success_s_after_first_429"] for w in walls.values()
           if w.get("first_recovery_success_s_after_first_429") is not None]
    refill = [e["median_refill_per_s"] for m in probe["tokens_window"] for e in (probe["tokens_window"][m] or [])
              if e.get("median_refill_per_s")]
    rr = rpd_refill()
    n_daily, n_429 = daily_429s()
    q = cal["qwen/qwen3.8-27b"]["buckets"]
    p = first["parsed"]

    def span(xs, fmt):
        lo, hi = fmt.format(min(xs)), fmt.format(max(xs))
        return lo if lo == hi else f"{lo}-{hi}"

    def per_model(bucket):          # one value when every model has the same
        vals = {m: f"{c['buckets'][bucket]['capacity']:.0f}" for m, c in cal.items() if bucket in c["buckets"]}
        if len(set(vals.values())) == 1:
            return f"{next(iter(vals.values()))} per model"
        return "; ".join(f"{SHORT[m]} {v}" for m, v in vals.items())
    qwen_rpm_doc = "documented only" in cal["qwen/qwen3.8-27b"]["buckets"]["RPM"]["source"]
    lim_k = "; ".join(f"{SHORT[m]} k={re.search(r'k=(\d+)', v).group(1)}" for m, v in ver["concurrency"].items()
                      if re.search(r"k=(\d+)", v))
    rows = [
        ("Requests per minute", per_model("RPM"), f"429 body \"Limit {p['limit']}\" after "
         f"{walls['openai/gpt-oss-120b']['ok_before_first_429']} successes"
         + ("; qwen: documented" if qwen_rpm_doc else ""), "high" + (" (qwen low)" if qwen_rpm_doc else "")),
        ("Tokens per minute", per_model("TPM") + f"; qwen also out {q['OTPM']['capacity']:.0f}, "
         f"in {q['ITPM']['capacity']:.0f}", "header; qwen's two only in real 429 bodies", "high"),
        ("Requests per day", per_model("RPD"), "header" + (
            f"; refilled {rr['from']} to {rr['to']} in {rr['hours']} h, same day" if rr else ""),
         "high"),
        ("Tokens per day", per_model("TPD") + " (documented)", f"no header; {n_daily} of {n_429} real 429s named a "
         "daily limit", "low"),
        ("Requests in flight at once", f"none found ({lim_k})", "stepwise concurrency", "medium"),
        ("What it returns when a limit is hit", f"429, retry-after {first['retry_after']} s, names the limit "
         f"(\"{p['short']}: Limit {p['limit']}\")", "probe burst; real 429s", "high"),
        ("Time until it recovers", f"{span(rec, '{:.1f}')} s to the next admission; tokens refill "
         f"{span(refill, '{:.0f}')}/s (TPM/60)", "recovery polls; token headers after a draft", "high"),
    ]
    L = ["## Table 1: limits found in the real service", "",
         "Free tier, per model (each model hit its own wall). From data/calibration.json, data/probe/calibration.json "
         "and the ledger's response headers.", "",
         "| limit | value found | how found | confidence |", "|---|---|---|---|"]
    L += [f"| {a} | {b} | {c} | {d} |" for a, b, c, d in rows]
    return L + [""]


def pct(xs, q):
    xs = sorted(xs)
    return xs[max(0, math.ceil(q * len(xs)) - 1)] if xs else None


def summarise(rows: list[dict]) -> dict:
    n = len(rows)
    lab = defaultdict(int)
    for r in rows:
        lab[r["answer_grade"] or "ungraded"] += 1
    tok = [int(r["tokens_in"]) + int(r["tokens_out"]) for r in rows]
    tot = [float(r["total_ms"]) for r in rows if r["total_ms"]]
    ttft = [float(r["first_token_ms"]) for r in rows if r["first_token_ms"]]
    wait = [float(r["queue_wait_ms"] or 0) for r in rows]
    return {"n": n, "correct": lab["correct"], "wrong": lab["wrong"], "refused_wrongly": lab["refused_wrongly"],
            "refused_correctly": lab["refused_correctly"], "ungraded": lab["ungraded"],
            "unsure": sum(1 for r in rows if r["grade_sure"] == "0"),
            "correct_pct": 100 * lab["correct"] / n, "wrong_pct": 100 * (lab["wrong"] + lab["refused_wrongly"]) / n,
            "refused_pct": 100 * lab["refused_correctly"] / n, "ungraded_pct": 100 * lab["ungraded"] / n,
            "tokens_per_answer": sum(tok) / n, "ttft_p50": pct(ttft, .5), "ttft_p95": pct(ttft, .95),
            "total_p50": pct(tot, .5), "total_p95": pct(tot, .95), "wait_p50": pct(wait, .5),
            "degraded": sum(1 for r in rows if r["degraded_level"] not in ("0", "")),
            "retries": sum(int(r["retries"]) for r in rows)}


DETAIL = ROOT / "data" / "results" / "detail.jsonl"


def describe(setting: str) -> str:
    """A retrieval setting as the brief asks for it: piece size, pieces fetched, re-rank yes or no."""
    sys.path.insert(0, str(ROOT / "scripts"))
    from eval_retrieval import ANSWER_SETTINGS
    if setting not in ANSWER_SETTINGS:
        return setting
    size, k, rerank, router = ANSWER_SETTINGS[setting]
    return f"{setting} ({size} / {k} / {'yes' if rerank else 'no'}{'' if router else ', no router'})"


def peak_queue(rows: list[dict]) -> int | None:
    """Most requests waiting for budget at the same moment, computed from results.csv alone: a request is
    queued during each interval in its wait_spans_ms column ("start-end;..." ms after its start_time)."""
    if not rows or not any("wait_spans_ms" in r for r in rows):
        return None
    events = []
    for r in rows:
        t0 = datetime.fromisoformat(r["start_time"]).timestamp()
        for span in filter(None, (r.get("wait_spans_ms") or "").split(";")):
            a, b = (float(x) for x in span.split("-"))
            events += [(t0 + a / 1000, 1), (t0 + b / 1000, -1)]
    if not events:
        return 0
    cur = peak = 0
    for _, step in sorted(events, key=lambda e: (e[0], e[1])):
        cur += step
        peak = max(peak, cur)
    return peak


def pctl(xs, q, n_note=True):
    v = pct(xs, q)
    if v is None:
        return "not measured"
    return f"{v:.0f}" + (" (= max, n<100)" if q == 0.99 and len(xs) < 100 and n_note else "")


def load_tables(latest: list[dict]) -> list[str]:
    from datetime import datetime
    rows = [r for r in latest if not r["scenario"].startswith("eval:")]
    if not rows:
        return []
    by = defaultdict(list)
    for r in rows:
        by[(r["scenario"], r["source"])].append(r)
    L = ["## Table 2a: latency per scenario", "",
         "Load runs only (the evals wait for budget on purpose). TTFT = first checked token = release (replies "
         "are buffered until checked), from arrival, queueing included; people who gave up have no TTFT and count "
         "in total at the time they left. Nearest-rank percentiles; n = requests with a value.", "",
         "| scenario | source | requests sent | duration (s) | offered rate/min | TTFT p50 / p95 / p99 (ms) | "
         "total p50 / p95 / p99 (ms) | n (TTFT / total) |", "|---|---|---|---|---|---|---|---|"]
    T2b = ["## Table 2b: outcomes per scenario", "",
           "Succeeded = status ok (a checked answer or refusal, any degrade level, or cited passages after both "
           "drafts failed the citation check, shown in the next column: not budget); rate-limited = cited passages "
           "only because budget had no drafter in time; timed out or given up = the simulated person left first. Peak queue length "
           "= most requests waiting for budget at once, from results.csv's wait_spans_ms column. Cost per 1,000 "
           "answers: $0 on the free tier we used; shadow = the same calls at Groq's published paid prices "
           "(data/prices.json, with URL and date).", "",
           "| scenario | source | succeeded % | of which passages after a failed check % | rate-limited % | "
           "timed out or given up % | error % | retries per request | peak queue length | tokens per answer | "
           "cost per 1,000 answers (free / shadow) |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    for (sc, src), rs in sorted(by.items()):
        n = len(rs)
        ts = [datetime.fromisoformat(r["start_time"]).timestamp() for r in rs]
        ends = [t + float(r["total_ms"] or 0) / 1000 for t, r in zip(ts, rs)]
        dur = max(ends) - min(ts) if rs else 0.0
        ttft = [float(r["first_token_ms"]) for r in rs if r["first_token_ms"]]
        tot = [float(r["total_ms"]) for r in rs if r["total_ms"]]
        L.append(f"| {sc} | {src} | {n} | {dur:.0f} | {60 * n / dur:.1f} | {pctl(ttft, .5)} / {pctl(ttft, .95)} / "
                 f"{pctl(ttft, .99)} | {pctl(tot, .5)} / {pctl(tot, .95)} / {pctl(tot, .99)} | {len(ttft)} / {len(tot)} |")
        st = defaultdict(int)
        for r in rs:
            st[r["status"]] += 1
        retries = sum(int(r["retries"] or 0) for r in rs) / n
        tokens = sum(int(r["tokens_in"] or 0) + int(r["tokens_out"] or 0) for r in rs) / n
        shadow = sum(float(r.get("shadow_cost_usd") or 0) for r in rs) / n * 1000
        pq = peak_queue(rs)
        gate_fb = sum(1 for r in rs if r["status"] == "ok" and "failed the citation check" in (r.get("degraded_reason") or ""))
        T2b.append(f"| {sc} | {src} | {100 * st['ok'] / n:.1f} | {100 * gate_fb / n:.1f} | {100 * st['rate_limited'] / n:.1f} | "
                   f"{100 * st['timeout'] / n:.1f} | {100 * st['error'] / n:.1f} | {retries:.2f} | "
                   f"{'not measured' if pq is None else pq} | {tokens:.0f} | $0 / ${shadow:.2f} |")
    return L + [""] + T2b + [""]


def ms(x):
    return "not measured" if x is None else f"{x:.0f}"


def pick(stats: dict[str, dict], tiebreak, need: int | None = None) -> tuple[str | None, list[str], int]:
    ok = {k: s for k, s in stats.items() if need is None or s["n"] >= need}
    if not ok:
        return None, [], 0
    best = max(s["correct"] for s in ok.values())
    tied = sorted(k for k, s in ok.items() if s["correct"] >= best - 1)
    win = min(tied, key=tiebreak)
    return win, tied, best


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=ROOT / "results" / "results.csv")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "tables.md")
    a = ap.parse_args(argv)
    with a.results.open(encoding="utf-8", newline="") as f:
        allrows = list(csv.DictReader(f))
    bad = [r["request_id"] for r in allrows if r["source"] != "real" and r["answer_grade"]]
    if bad:
        sys.exit(f"stand-in rows carry a grade (only real rows are graded): {bad[:5]}")
    wrong_value = [r["request_id"] for r in allrows
                   if r["answer_grade"] not in ("", "correct", "wrong", "refused_correctly", "refused_wrongly")]
    if wrong_value:
        sys.exit(f"answer_grade outside the brief's four values: {wrong_value[:5]}")
    latest = [r for r in allrows if r["superseded"] == "0"]
    rows = [r for r in latest if r["source"] == "real"]
    n_questions = 18
    L = ["# Tables", "", f"Generated by `scripts/make_tables.py` from `{a.results.name}` "
         "(latest answer per question). Percentiles nearest-rank. TTFT = total: replies are released "
         "after their checks. Both include budget waits (the eval waits rather than degrade).", ""]
    L += table1()
    winner = {}
    ret = [r for r in rows if r["scenario"] == "eval:retrieval"]
    if ret:
        by = defaultdict(list)
        for r in ret:
            by[(r["retrieval_setting"], r["model"])].append(r)
        stats = {s: summarise(v) for (s, m), v in by.items()}
        drafter = sorted({m for (_, m) in by})
        any_ungraded = any(x["ungraded"] for x in stats.values())
        L += ["## Table 3: retrieval settings", "",
              f"Drafter fixed ({', '.join(drafter)}), verifier off; only retrieval changes.", "",
              # ungraded % only when some setting has ungraded answers (all 0.0 otherwise, and the table is wide)
              "| setting (chars / pieces / re-rank) | questions asked | correct and cited % | wrong % (incl. wrongly refused) "
              "| correctly refused % | " + ("ungraded % | " if any_ungraded else "") +
              "unsure | degraded | tokens per answer | TTFT p50 / p95 (ms) | total p50 / p95 (ms) | budget wait p50 (ms) |",
              "|---|---|---|---|---|" + ("---|" if any_ungraded else "") + "---|---|---|---|---|---|"]
        for s, x in sorted(stats.items(), key=lambda kv: (-kv[1]["n"], -kv[1]["correct"])):
            flag = "" if x["n"] >= n_questions else " (incomplete)"
            L.append(f"| {describe(s)}{flag} | {x['n']} | {x['correct_pct']:.1f} | {x['wrong_pct']:.1f} | {x['refused_pct']:.1f} | "
                     + (f"{x['ungraded_pct']:.1f} | " if any_ungraded else "") +
                     f"{x['unsure']} | {x['degraded']} | {x['tokens_per_answer']:.0f} | "
                     f"{ms(x['ttft_p50'])} / {ms(x['ttft_p95'])} | {ms(x['total_p50'])} / {ms(x['total_p95'])} | "
                     f"{ms(x['wait_p50'])} |")
        win, tied, best = pick(stats, lambda k: stats[k]["tokens_per_answer"], need=n_questions)
        incomplete = sorted(k for k, x in stats.items() if x["n"] < n_questions)
        if win:
            why = (f"best correct = {best} (complete settings only); within one of it (tie): {', '.join(tied)}; "
                   f"cheapest tied: {win} ({stats[win]['tokens_per_answer']:.0f} tokens per answer)"
                   + (f"; not eligible yet (incomplete): {', '.join(incomplete)}" if incomplete else ""))
            L += ["", f"**Chosen setting so far: {win}.** {why}.", ""]
        else:
            why = "no setting has every question answered yet"
            L += ["", f"**No setting chosen:** {why}.", ""]
        if any(x["ungraded"] or x["unsure"] for x in stats.values()):
            L += ["Unsure or ungraded rows exist: the choice is provisional until they are checked "
                  "(results/quality/unsure.md).", ""]
        winner["setting"] = {"chosen": win, "tied": tied, "best_correct": best, "why": why,
                             "provisional": any(x["ungraded"] or x["unsure"] for x in stats.values()),
                             "stats": stats}
    dr = [r for r in rows if r["scenario"] == "eval:drafter"]
    if dr:
        setting = sorted({r["retrieval_setting"] for r in dr})
        same = [r for r in ret if r["retrieval_setting"] in setting]
        by = defaultdict(list)
        for r in dr + same:
            by[r["model"]].append(r)
        stats = {m: summarise(v) for m, v in by.items()}
        L += [f"## Drafters at setting {', '.join(setting)}", "",
              "Same configuration for each (verifier off). Rows for the fixed drafter come from the retrieval sweep.", "",
              "| drafter | questions asked | correct and cited % | wrong % (incl. wrongly refused) | correctly refused % | ungraded % | unsure | "
              "tokens per answer | total p50 / p95 (ms) | budget wait p50 (ms) | retries |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
        for m, x in sorted(stats.items(), key=lambda kv: -kv[1]["correct"]):
            L.append(f"| {m} | {x['n']} | {x['correct_pct']:.1f} | {x['wrong_pct']:.1f} | {x['refused_pct']:.1f} | "
                     f"{x['ungraded_pct']:.1f} | {x['unsure']} | {x['tokens_per_answer']:.0f} | {ms(x['total_p50'])} / "
                     f"{ms(x['total_p95'])} | {ms(x['wait_p50'])} | {x['retries']} |")
        win, tied, best = pick(stats, lambda k: (stats[k]["wrong"] + stats[k]["refused_wrongly"],
                                                 stats[k]["total_p50"] or 1e18), need=n_questions)
        why = (f"best correct = {best}; within one of it (tie): {', '.join(tied)}; among those, fewest wrong "
               f"then lowest total p50: {win}")
        L += ["", f"**First drafter: {win}.** {why}.", ""]
        winner["drafter"] = {"chosen": win, "tied": tied, "best_correct": best, "why": why, "stats": stats}
    gate = [r for r in latest if r["scenario"] == "eval:q14_decide_gate" and r["source"] == "real"]
    if gate:
        L += ["## Q14 decide gate: before / after (separate from Table 3)", "",
              "Before = the retrieval sweep's answer (old gate: evidence scored against the question only); after = "
              "the same question and setting with the gate scoring the router's rewrite too. The router prompt is "
              "unchanged, so Table 3 stays as measured.", "",
              "| qid | setting | drafter | before | after |", "|---|---|---|---|---|"]
        for g in gate:
            qid = g["request_id"].rsplit("-", 1)[-1]           # request ids end with -<qid>
            before = next((r for r in latest if r["scenario"] == "eval:retrieval" and r["source"] == "real"
                           and r["retrieval_setting"] == g["retrieval_setting"] and r["model"] == g["model"]
                           and r["request_id"].endswith("-" + qid)), None)
            L.append(f"| {qid} | {g['retrieval_setting']} | {g['model']} | "
                     f"{before['answer_grade'] if before else 'not measured'} | {g['answer_grade']} |")
        L.append("")
    L += load_tables(latest)
    cov = defaultdict(lambda: defaultdict(int))
    for r in latest:
        key = (r["scenario"], r["source"])
        cov[key]["rows"] += 1
        cov[key]["graded" if r["answer_grade"] else (r.get("grade_note") or "not graded").split(";")[0]] += 1
    L += ["## Grading coverage (every scenario and source)", "",
          "| scenario | source | rows | graded | not graded, by reason |", "|---|---|---|---|---|"]
    for (sc, src), c in sorted(cov.items()):
        why = ", ".join(f"{k} {v}" for k, v in sorted(c.items()) if k not in ("rows", "graded")) or "-"
        L.append(f"| {sc} | {src} | {c['rows']} | {c['graded']} | {why} |")
    L.append("")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("\n".join(L) + "\n", encoding="utf-8", newline="\n")
    wpath = a.out.parent / "quality" / "winner.json"
    wpath.parent.mkdir(parents=True, exist_ok=True)
    wpath.write_text(json.dumps({"_generated_by": "scripts/make_tables.py", **winner}, indent=1) + "\n",
                     encoding="utf-8", newline="\n")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    sys.exit(main())
