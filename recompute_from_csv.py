"""Recompute the write-up's Tables 2a, 2b and 3 from results.csv alone, with no repo code.

    python recompute_from_csv.py [path\to\results.csv]

Rules (the same ones scripts/make_tables.py uses):
  * keep only rows with superseded == 0 (latest run / latest answer)
  * load tables: every scenario that doesn't start with "eval:", grouped by (scenario, source)
  * Table 3: scenario == "eval:retrieval", source == "real", grouped by retrieval_setting
  * percentiles are nearest-rank: sorted values, index ceil(p * n) - 1
"""
import csv
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime

path = sys.argv[1] if len(sys.argv) > 1 else r"D:\Vaani\Vaani_Relay\submission\results.csv"
with open(path, encoding="utf-8", newline="") as f:
    rows = list(csv.DictReader(f))


def pct(xs, q):
    xs = sorted(xs)
    return round(xs[max(0, math.ceil(q * len(xs)) - 1)]) if xs else None


def peak_queue(rs):
    ev = []
    for r in rs:
        t0 = datetime.fromisoformat(r["start_time"]).timestamp()
        for span in filter(None, r["wait_spans_ms"].split(";")):
            a, b = (float(x) for x in span.split("-"))
            ev += [(t0 + a / 1000, 1), (t0 + b / 1000, -1)]
    cur = peak = 0
    for _, s in sorted(ev):
        cur += s
        peak = max(peak, cur)
    return peak


print(f"{len(rows)} rows in file; {sum(r['superseded'] == '0' for r in rows)} current (superseded = 0)\n")
latest = [r for r in rows if r["superseded"] == "0"]

# ---------------- Tables 2a + 2b
by = defaultdict(list)
for r in latest:
    if not r["scenario"].startswith("eval:"):
        by[(r["scenario"], r["source"])].append(r)

print("TABLE 2a / 2b")
print(f"{'scenario':22}{'source':9}{'n':>4}{'dur s':>7}{'rate/m':>7}  {'TTFT p50/p95/p99':20}{'total p50/p95/p99':22}"
      f"{'ok%':>6}{'gateFB%':>8}{'rl%':>6}{'gaveup%':>8}{'retr/req':>9}{'peakQ':>6}{'tok/ans':>8}{'$/1k':>6}")
for (sc, src), rs in sorted(by.items()):
    n = len(rs)
    starts = [datetime.fromisoformat(r["start_time"]).timestamp() for r in rs]
    ends = [t + float(r["total_ms"] or 0) / 1000 for t, r in zip(starts, rs)]
    dur = max(ends) - min(starts)
    ttft = [float(r["first_token_ms"]) for r in rs if r["first_token_ms"]]
    tot = [float(r["total_ms"]) for r in rs if r["total_ms"]]
    st = Counter(r["status"] for r in rs)
    gate_fb = sum(r["status"] == "ok" and "failed the citation check" in r["degraded_reason"] for r in rs)
    retries = sum(int(r["retries"] or 0) for r in rs) / n
    tok = sum(int(r["tokens_in"] or 0) + int(r["tokens_out"] or 0) for r in rs) / n
    cost = sum(float(r["shadow_cost_usd"] or 0) for r in rs) / n * 1000
    t3 = f"{pct(ttft, .5)}/{pct(ttft, .95)}/{pct(ttft, .99)}"
    a3 = f"{pct(tot, .5)}/{pct(tot, .95)}/{pct(tot, .99)}"
    print(f"{sc:22}{src:9}{n:>4}{dur:>7.0f}{60 * n / dur:>7.1f}  {t3:20}{a3:22}"
          f"{100 * st['ok'] / n:>6.1f}{100 * gate_fb / n:>8.1f}{100 * st['rate_limited'] / n:>6.1f}"
          f"{100 * st['timeout'] / n:>8.1f}{retries:>9.2f}{peak_queue(rs):>6}{tok:>8.0f}{cost:>6.2f}")

# ---------------- Table 3
print("\nTABLE 3 (real eval:retrieval rows)")
by3 = defaultdict(list)
for r in latest:
    if r["scenario"] == "eval:retrieval" and r["source"] == "real":
        by3[r["retrieval_setting"]].append(r)
print(f"{'setting':14}{'n':>4}{'correct%':>10}{'wrong%':>8}{'refused%':>10}{'tok/ans':>9}  {'TTFT p50/p95':14}")
for s, rs in sorted(by3.items(), key=lambda kv: -sum(r["answer_grade"] == "correct" for r in kv[1])):
    n = len(rs)
    g = Counter(r["answer_grade"] for r in rs)
    tok = sum(int(r["tokens_in"]) + int(r["tokens_out"]) for r in rs) / n
    ttft = [float(r["first_token_ms"]) for r in rs if r["first_token_ms"]]
    print(f"{s:14}{n:>4}{100 * g['correct'] / n:>10.1f}{100 * (g['wrong'] + g['refused_wrongly']) / n:>8.1f}"
          f"{100 * g['refused_correctly'] / n:>10.1f}{tok:>9.0f}  {pct(ttft, .5)}/{pct(ttft, .95)}")
