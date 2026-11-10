"""Analyse the probe's raw log into Table 1, a calibration file and a findings report.

    .venv\\Scripts\\python scripts\\analyze_probe.py                      # real: data/probe/probe_raw.jsonl
    .venv\\Scripts\\python scripts\\analyze_probe.py --raw data/probe/dryrun/<truth>/probe_raw.jsonl \\
                                                    --out data/probe/dryrun/<truth>

Writes (all numbers computed here from the raw log; nothing typed by hand):
  <out>/limits_table.md   Table 1: documented vs observed limits per model
  <out>/calibration.json  per model: first-token fit vs prompt tokens, TTFB p50/p95, tokens/s, chunk
                          gaps, every limit seen (incl. ones only in 429 bodies), what admission
                          charges, prompt-token estimate fit
  <out>/findings.md       each probe question: answer + quoted header values and 429 bodies; every
                          observation that contradicts the documentation is flagged
Stdlib only.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MODELS = ("openai/gpt-oss-20b", "openai/gpt-oss-120b", "qwen/qwen3.8-27b")
SHORT = {"openai/gpt-oss-20b": "gpt-oss-20b", "openai/gpt-oss-120b": "gpt-oss-120b", "qwen/qwen3.8-27b": "qwen3.8-27b"}
P95_MIN_N = 20

BODY_RE = re.compile(r"on (?P<what>[a-z ]+?) \((?P<short>[A-Z]+)\): Limit (?P<limit>\d+), Used (?P<used>\d+), "
                     r"Requested (?P<requested>\d+)\. Please try again in (?P<wait>(?:[0-9.]+(?:ms|h|m|s))+)")
TOO_LARGE_RE = re.compile(r"on (?P<what>[a-z ]+?) \((?P<short>[A-Z]+)\): Limit (?P<limit>\d+), Requested (?P<requested>\d+)")


# ----------------------------------------------------------------------------- small stats
def pct(xs: list[float], q: float) -> float | None:
    """Nearest-rank percentile."""
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    return xs[max(0, math.ceil(q * len(xs)) - 1)]


def dist(xs: list[float | None], nd: int = 1) -> dict[str, Any]:
    xs = [x for x in xs if x is not None]
    n = len(xs)
    if not n:
        return {"n": 0}
    r = lambda v: None if v is None else round(v, nd)  # noqa: E731
    return {"n": n, "p50": r(pct(xs, 0.5)), "p95": r(pct(xs, 0.95)), "max": r(max(xs)), "min": r(min(xs)),
            "p95_reliable": n >= P95_MIN_N}


def ols(xs: list[float], ys: list[float]) -> dict[str, Any] | None:
    pts = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    n = len(pts)
    if n < 3:
        return None
    mx = sum(p[0] for p in pts) / n
    my = sum(p[1] for p in pts) / n
    sxx = sum((p[0] - mx) ** 2 for p in pts)
    if sxx == 0:
        return None
    b = sum((p[0] - mx) * (p[1] - my) for p in pts) / sxx
    a = my - b * mx
    ss_res = sum((p[1] - (a + b * p[0])) ** 2 for p in pts)
    ss_tot = sum((p[1] - my) ** 2 for p in pts)
    resid = [abs(p[1] - (a + b * p[0])) for p in pts]
    return {"n": n, "intercept": round(a, 3), "slope": round(b, 6),
            "r2": round(1 - ss_res / ss_tot, 4) if ss_tot else None,
            "abs_residual_p95": round(pct(resid, 0.95), 2), "x_range": [min(p[0] for p in pts), max(p[0] for p in pts)]}


def parse_duration(s: str | None) -> float | None:
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        pass
    parts = re.findall(r"([0-9]*\.?[0-9]+)(ms|h|m|s)", s)
    return sum(float(n) * {"ms": 0.001, "h": 3600, "m": 60, "s": 1}[u] for n, u in parts) if parts else None


def iso(t: float | None) -> str | None:
    return None if t is None else datetime.fromtimestamp(t, timezone.utc).isoformat(timespec="milliseconds")


# ----------------------------------------------------------------------------- loading
class Call:
    def __init__(self, line: dict) -> None:
        self.raw = line
        r = line["record"]
        self.r = r
        self.run_id = line["run_id"]
        self.phase, self.model, self.size, self.variant = line["phase"], line["model"], line["size"], line["variant"]
        self.step = line.get("step") or {}
        self.t0 = line["t_start_unix"]
        self.status, self.outcome = r["status"], r["outcome"]
        self.ok = self.status == 200 and self.outcome == "ok"
        self.h = {k.lower(): v for k, v in r.get("rate_limit_headers") or []}
        u = r.get("usage") or {}
        self.usage = u
        self.prompt_tokens = u.get("prompt_tokens")
        self.completion_tokens = u.get("completion_tokens")
        self.prompt_chars = line["prompt_chars"]
        self.n_messages = len(line["payload"]["messages"])
        self.max_tokens = line["payload"].get("max_tokens")
        self.est = line["est_prompt_tokens"]
        self.connect_ms = r.get("connect_ms")
        self.headers_ms = r.get("headers_ms")
        ft = [x for x in (r.get("first_reasoning_ms"), r.get("first_visible_ms")) if x is not None]
        self.first_token_ms = min(ft) if ft else None
        self.first_visible_ms = r.get("first_visible_ms")
        self.total_ms = r.get("total_ms")
        self.chunks = r.get("chunks") or []

    def hi(self, name: str) -> int | None:
        v = self.h.get(name)
        try:
            return int(v) if v is not None else None
        except ValueError:
            return None

    def hs(self, name: str) -> float | None:
        return parse_duration(self.h.get(name))

    @property
    def t_headers(self) -> float | None:
        return None if self.headers_ms is None else self.t0 + self.headers_ms / 1000

    @property
    def t_adm(self) -> float | None:
        """When the server admitted the call: header arrival minus the reported queue_time. Groq sends
        headers after queueing, but (we find) computes their values at admission."""
        th = self.t_headers
        qt = self.usage.get("queue_time")
        return th - qt if th is not None and isinstance(qt, (int, float)) and self.ok else th

    @property
    def t_end(self) -> float:
        return self.t0 + (self.total_ms or 0) / 1000

    def after_connect(self, ms: float | None) -> float | None:
        return None if ms is None or self.connect_ms is None else ms - self.connect_ms

    def token_chunks(self) -> list[float]:
        return [c[0] for c in self.chunks if c[1] in ("reasoning", "content", "mixed") and c[2] > 0]

    def body(self) -> dict | None:
        b = self.r.get("error_body")
        if not b:
            return None
        try:
            return json.loads(b)
        except json.JSONDecodeError:
            return {"_unparsed": b}

    def body_message(self) -> str | None:
        b = self.body()
        if not b:
            return None
        e = b.get("error", b)
        return e.get("message") if isinstance(e, dict) else str(e)

    def parsed_limit(self) -> dict | None:
        msg = self.body_message() or ""
        m = BODY_RE.search(msg)
        if m:
            d = m.groupdict()
            return {"what": d["what"], "short": d["short"], "limit": int(d["limit"]), "used": int(d["used"]),
                    "requested": int(d["requested"]), "try_again_in": d["wait"],
                    "try_again_in_s": parse_duration(d["wait"]), "form": "rate_limit_reached"}
        m = TOO_LARGE_RE.search(msg)
        if m:
            d = m.groupdict()
            return {"what": d["what"], "short": d["short"], "limit": int(d["limit"]),
                    "requested": int(d["requested"]), "form": "request_too_large"}
        return None


def load(raw: Path) -> tuple[list[Call], list[dict]]:
    calls, events = [], []
    with raw.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            o = json.loads(line)
            if o.get("type") == "call":
                calls.append(Call(o))
            elif o.get("type") == "event":
                events.append(o)
    calls.sort(key=lambda c: c.t0)
    return calls, events


# ----------------------------------------------------------------------------- analyses
def sequential(c: Call) -> bool:
    """Calls not competing with our own other calls (for latency and fits)."""
    if c.phase == "P3":
        return False
    if c.phase == "P4":
        return c.step.get("k") == 1
    return True


def per_call_rates(c: Call) -> dict[str, Any]:
    toks = c.token_chunks()
    out: dict[str, Any] = {}
    if c.completion_tokens and len(toks) >= 2 and toks[-1] - toks[0] > 50:
        out["client_tps"] = c.completion_tokens / ((toks[-1] - toks[0]) / 1000)
    ct = c.usage.get("completion_time")
    if c.completion_tokens and ct:
        out["server_tps"] = c.completion_tokens / ct
    out["gaps"] = [b - a for a, b in zip(toks, toks[1:])]
    return out


def calibration(calls: list[Call], doc: dict) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for m in MODELS:
        cs = [c for c in calls if c.model == m]
        ok = [c for c in cs if c.ok]
        seq = [c for c in ok if sequential(c)]
        if not cs:
            continue
        rates = [per_call_rates(c) for c in seq]
        gaps = [g for r in rates for g in r["gaps"]]
        fit_ttft = ols([c.prompt_tokens for c in seq], [c.after_connect(c.first_token_ms) for c in seq])
        fit_vis = ols([c.prompt_tokens for c in seq], [c.after_connect(c.first_visible_ms) for c in seq])
        fit_prompt = ols([c.prompt_chars for c in ok if c.n_messages == 1], [c.prompt_tokens for c in ok if c.n_messages == 1])
        fit_prompt2 = ols([c.prompt_chars for c in ok if c.n_messages == 2], [c.prompt_tokens for c in ok if c.n_messages == 2])
        ratios = [c.est / c.prompt_tokens for c in ok if c.prompt_tokens]
        q_time = [c.usage.get("queue_time") for c in seq if c.usage.get("queue_time") is not None]
        by_size = {}
        for size in sorted({c.size for c in seq}):
            ss = [c for c in seq if c.size == size]
            by_size[size] = {
                "headers_ms": dist([c.headers_ms for c in ss]),
                "server_ttfb_ms": dist([c.after_connect(c.headers_ms) for c in ss]),
                "first_token_after_connect_ms": dist([c.after_connect(c.first_token_ms) for c in ss]),
                "first_visible_after_connect_ms": dist([c.after_connect(c.first_visible_ms) for c in ss]),
                "total_ms": dist([c.total_ms for c in ss]),
                "prompt_tokens": dist([c.prompt_tokens for c in ss], 0),
                "completion_tokens": dist([c.completion_tokens for c in ss], 0),
                "finish_reason_length": sum(1 for c in ss if c.r.get("finish_reason") == "length"),
            }
        out[m] = {
            "calls": len(cs), "ok": len(ok), "sequential_ok": len(seq),
            "outcomes": dict(sorted(_count(f"{c.status}:{c.outcome}" for c in cs).items())),
            "connect_ms": dist([c.connect_ms for c in ok]),
            "ttfb": {
                "_what": "headers_ms = time from call start (incl. new TCP+TLS connection) to response headers; "
                         "server_ttfb_ms = the same minus connect_ms (what a kept-alive connection would see). "
                         "All successful calls of this model, any size, any phase except P3 bursts and P4 k>1.",
                "headers_ms": dist([c.headers_ms for c in seq]),
                "server_ttfb_ms": dist([c.after_connect(c.headers_ms) for c in seq]),
            },
            "first_token_fit_vs_prompt_tokens": {
                "_what": "OLS: first token (reasoning or visible) after connect, ms = intercept + slope * prompt_tokens",
                "first_token": fit_ttft, "first_visible": fit_vis},
            "by_size": by_size,
            "tokens_per_s": {"client_side": dist([r.get("client_tps") for r in rates]),
                             "server_reported": dist([r.get("server_tps") for r in rates])},
            "chunk_gaps_ms": dist(gaps),
            "server_queue_time_s": dist(q_time, 4),
            "prompt_token_estimate": {
                "_what": "prompt_tokens reported by the server vs prompt characters; our ledger estimate is "
                         "ceil(chars/3) + 8*messages + overhead, and must never be below the real count",
                "one_message_fit": fit_prompt, "two_message_fit": fit_prompt2,
                "estimate_over_actual": dist(ratios, 3),
                "estimate_below_actual_calls": sum(1 for r in ratios if r < 1),
            },
            "latency_vs_in_flight": p4_table(calls, m),
            "limits_seen": limits_seen(cs, m, doc),
            "admission": admission(calls, m),
            "token_accounting_fit": token_accounting(calls, m),
            "server_ttfb_vs_queue_time": {
                "_what": "OLS: server_ttfb_ms = intercept + slope * queue_time_s (queue_time from x_groq.usage)",
                "fit": ols([c.usage.get("queue_time") for c in seq], [c.after_connect(c.headers_ms) for c in seq]),
                "queue_time_share_of_server_ttfb_p50": pct([1000 * c.usage["queue_time"] / c.after_connect(c.headers_ms)
                                                            for c in seq if c.usage.get("queue_time") is not None
                                                            and (c.after_connect(c.headers_ms) or 0) > 0], 0.5)},
            "reasoning_p0": [{"variant": c.variant, "status": c.status, "settings": c.raw.get("settings"),
                              "reasoning_chars": c.r.get("reasoning_chars"), "visible_text": c.r.get("text"),
                              "completion_tokens": c.completion_tokens,
                              "completion_tokens_details": c.usage.get("completion_tokens_details"),
                              "first_visible_after_connect_ms": c.after_connect(c.first_visible_ms),
                              "error": c.body_message()} for c in cs if c.phase == "P0"],
        }
    return out


def _count(it) -> dict:
    d: dict = defaultdict(int)
    for x in it:
        d[x] += 1
    return dict(d)


def p4_table(calls: list[Call], m: str) -> list[dict]:
    rows = []
    for k in sorted({c.step.get("k") for c in calls if c.model == m and c.phase == "P4"}):
        ss = [c for c in calls if c.model == m and c.phase == "P4" and c.step.get("k") == k]
        ok = [c for c in ss if c.ok]
        rows.append({"k": k, "calls": len(ss), "ok": len(ok), "non_200": [c.status for c in ss if c.status != 200],
                     "headers_ms_p50": pct([c.headers_ms for c in ok], 0.5),
                     "server_ttfb_ms_p50": pct([c.after_connect(c.headers_ms) for c in ok], 0.5),
                     "first_token_after_connect_ms_p50": pct([c.after_connect(c.first_token_ms) for c in ok], 0.5),
                     "total_ms_p50": pct([c.total_ms for c in ok], 0.5),
                     "total_ms_max": max((c.total_ms for c in ok), default=None),
                     "queue_time_s_p50": pct([c.usage.get("queue_time") for c in ok], 0.5)})
    return rows


def limits_seen(cs: list[Call], m: str, doc: dict) -> dict:
    hdr = defaultdict(set)
    for c in cs:
        for k in ("x-ratelimit-limit-requests", "x-ratelimit-limit-tokens"):
            if k in c.h:
                hdr[k].add(c.h[k])
    bodies = []
    for c in cs:
        if c.status and c.status >= 400:
            pl = c.parsed_limit()
            bodies.append({"phase": c.phase, "variant": c.variant, "status": c.status,
                           "retry_after": c.h.get("retry-after"), "parsed": pl,
                           "type": (c.body() or {}).get("error", {}).get("type") if isinstance((c.body() or {}).get("error"), dict) else None,
                           "code": (c.body() or {}).get("error", {}).get("code") if isinstance((c.body() or {}).get("error"), dict) else None})
    from_bodies = {}
    for b in bodies:
        if b["parsed"]:
            from_bodies.setdefault(b["parsed"]["short"], set()).add(b["parsed"]["limit"])
    return {"headers": {k: sorted(v) for k, v in hdr.items()},
            "from_error_bodies": {k: sorted(v) for k, v in from_bodies.items()},
            "documented": {k: doc["models"][m][k] for k in ("rpm", "rpd", "tpm", "tpd")},
            "error_responses": bodies}


def admission(calls: list[Call], m: str) -> dict | None:
    """P2: baseline (last P0 header) -> A (max_tokens 2000) -> B (tiny)."""
    res = []
    for run in sorted({c.run_id for c in calls}):
        rc = [c for c in calls if c.run_id == run and c.model == m]
        a = next((c for c in rc if c.phase == "P2" and c.variant == "A"), None)
        b = next((c for c in rc if c.phase == "P2" and c.variant == "B"), None)
        if not a or not a.ok:
            continue
        base = next((c for c in reversed(rc) if c.phase == "P0" and c.ok and c.t0 < a.t0), None)
        lim = a.hi("x-ratelimit-limit-tokens")
        if base is None or lim is None:
            continue
        rate = lim / 60
        dt_a = a.t_adm - base.t_adm
        drop_raw = base.hi("x-ratelimit-remaining-tokens") - a.hi("x-ratelimit-remaining-tokens")
        # base call's own unused max_tokens may have been refunded between base and A
        base_unused = (base.max_tokens or 0) - (base.completion_tokens or 0)
        full_res = (a.prompt_tokens or 0) + a.max_tokens
        d = {"run_id": run, "limit_tokens": lim,
             "baseline": {"call": base.variant, "remaining_tokens": base.h.get("x-ratelimit-remaining-tokens"),
                          "reset_tokens": base.h.get("x-ratelimit-reset-tokens"), "at": iso(base.t_headers)},
             "A": {"max_tokens": a.max_tokens, "prompt_tokens": a.prompt_tokens, "completion_tokens": a.completion_tokens,
                   "remaining_tokens": a.h.get("x-ratelimit-remaining-tokens"),
                   "reset_tokens": a.h.get("x-ratelimit-reset-tokens"), "at": iso(a.t_headers)},
             "seconds_between_baseline_and_A_admission": round(dt_a, 3),
             "drop_at_A_raw": drop_raw,
             # expected drop between the two admissions, with refill at limit/60 per second
             "expect_if_max_tokens_held": round(full_res - base_unused - rate * dt_a, 1),
             "expect_if_usage": round((a.prompt_tokens or 0) + (base.completion_tokens or 0) - rate * dt_a, 1),
             "expect_if_max_tokens_held_no_refill": full_res - base_unused,
             "expect_if_usage_no_refill": (a.prompt_tokens or 0) + (base.completion_tokens or 0)}
        d_adm = min(abs(drop_raw - d["expect_if_max_tokens_held"]), abs(drop_raw - d["expect_if_max_tokens_held_no_refill"]))
        d_use = min(abs(drop_raw - d["expect_if_usage"]), abs(drop_raw - d["expect_if_usage_no_refill"]))
        gap = abs(d["expect_if_max_tokens_held_no_refill"] - d["expect_if_usage_no_refill"])
        d["verdict"] = ("max_tokens reserved at admission" if d_adm < d_use and d_adm < 0.4 * gap
                        else "only the prompt charged at admission" if d_use < d_adm and d_use < 0.4 * gap
                        else "inconclusive")
        if b is not None and b.hi("x-ratelimit-remaining-tokens") is not None:
            dt_b = b.t_adm - a.t_adm
            rise = b.hi("x-ratelimit-remaining-tokens") - a.hi("x-ratelimit-remaining-tokens")
            unused = a.max_tokens - (a.completion_tokens or 0)
            b_charge = (b.prompt_tokens or 0) + (b.max_tokens if d["verdict"].startswith("max_tokens") else 0)
            d["B"] = {"remaining_tokens": b.h.get("x-ratelimit-remaining-tokens"), "at": iso(b.t_headers),
                      "seconds_after_A_admission": round(dt_b, 3), "rise_since_A": rise,
                      "A_unused_max_tokens": unused, "B_own_charge_est": b_charge,
                      "refill_if_bucket": round(rate * dt_b, 1)}
            if d["verdict"].startswith("max_tokens"):
                refunded = rise + b_charge
                refill = rate * dt_b
                # refunded within the window if the rise exceeds what refill alone could explain
                d["B"]["refund_verdict"] = ("unused max_tokens refunded when A finished"
                                            if refunded - refill > 0.5 * unused or (refunded > 0.5 * unused and refill < 0.25 * unused)
                                            else "no refund seen")
        res.append(d)
    ov = [c for c in calls if c.model == m and c.phase == "P2-overflow"]
    return {"pairs": res, "overflow": [{"status": c.status, "max_tokens": c.max_tokens, "body": c.body_message(),
                                        "parsed": c.parsed_limit(), "retry_after": c.h.get("retry-after"),
                                        "headers_ms": c.headers_ms, "total_ms": c.total_ms,
                                        "completion_tokens": c.completion_tokens} for c in ov]}


def charge_model(cal_m: dict | None) -> dict:
    """How a call moves x-ratelimit-remaining-tokens, from P2's verdicts (defaults: prompt only)."""
    pairs = ((cal_m or {}).get("admission") or {}).get("pairs") or []
    v = pairs[-1] if pairs else {}
    admission = (v.get("verdict") or "").startswith("max_tokens")
    refund = (v.get("B") or {}).get("refund_verdict", "").startswith("unused")
    return {"admission_max_tokens": admission, "refund": refund, "from": v.get("verdict") or "no P2 verdict"}


def token_accounting(calls: list[Call], m: str) -> dict | None:
    """Fit x-ratelimit-remaining-tokens on every pair of consecutive, non-overlapping calls of one
    model to two accounting models, on two time bases:
      usage:      prompt charged at admission, completion charged when the call ends
      admission:  prompt + max_tokens held at admission, unused max_tokens refunded at the end
      time base:  header arrival, or admission (= header arrival - usage.queue_time)
    Both assume a bucket refilling at limit/60 per second, capped at the limit."""
    cs = sorted([c for c in calls if c.model == m and c.ok and c.phase != "P3"
                 and c.hi("x-ratelimit-remaining-tokens") is not None], key=lambda c: c.t0)
    pairs = []
    for p, q in zip(cs, cs[1:]):
        if p.t_end > q.t0 + 0.05:
            continue      # overlapping calls (P4 steps): order of admission unknown
        lim = q.hi("x-ratelimit-limit-tokens")
        rate = lim / 60
        rp, rq = p.hi("x-ratelimit-remaining-tokens"), q.hi("x-ratelimit-remaining-tokens")
        ct_p, pt_q = p.completion_tokens or 0, q.prompt_tokens or 0
        row = {"p": f"{p.phase}/{p.size}/{p.variant}", "q": f"{q.phase}/{q.size}/{q.variant}",
               "max_tokens_p": p.max_tokens, "max_tokens_q": q.max_tokens, "remaining_p": rp, "remaining_q": rq}
        for base, tp, tq in (("admission", p.t_adm, q.t_adm), ("headers", p.t_headers, q.t_headers)):
            dt = tq - tp
            lvl_u = min(lim, rp - ct_p + rate * dt)
            lvl_a = min(lim, rp + ((p.max_tokens or 0) - ct_p) + rate * dt)
            row[f"dt_{base}_s"] = round(dt, 3)
            row[f"saturated_{base}"] = rp - ct_p + rate * dt >= lim
            row[f"res_usage_{base}"] = round(rq - (lvl_u - pt_q), 1)
            row[f"res_admission_{base}"] = round(rq - (lvl_a - pt_q - (q.max_tokens or 0)), 1)
        pairs.append(row)
    if not pairs:
        return None
    summary = {}
    for model in ("usage", "admission"):
        for base in ("admission", "headers"):
            r = [abs(x[f"res_{model}_{base}"]) for x in pairs if not x[f"saturated_{base}"]]
            summary[f"{model}@{base}"] = {"n_unsaturated": len(r), "median_abs_residual": pct(r, 0.5),
                                          "p90_abs_residual": pct(r, 0.9), "max_abs_residual": max(r) if r else None}
    best = min((k for k in summary if summary[k]["n_unsaturated"]),
               key=lambda k: summary[k]["median_abs_residual"], default=None)
    by_max = defaultdict(list)
    for x in pairs:
        if not x["saturated_admission"]:
            by_max[f"{x['max_tokens_p']}->{x['max_tokens_q']}"].append(x["res_usage_admission"])
    return {"pairs": pairs, "fit": summary, "best": best,
            "usage_residual_by_max_tokens_pair": {k: {"n": len(v), "median": pct(v, 0.5), "min": min(v), "max": max(v)}
                                                  for k, v in sorted(by_max.items())}}


def tokens_window(calls: list[Call], m: str, cm: dict) -> list[dict] | None:
    """P1 polls after a draft call: does remaining-tokens rise steadily (bucket) or stay flat until a
    minute boundary (window)? Each interval p -> q is corrected for q's own admission charge and for
    what p's completion did (refund of unused max_tokens, or usage charged at the end)."""
    out = []
    for run in sorted({c.run_id for c in calls}):
        pts = [c for c in calls if c.run_id == run and c.model == m and c.phase == "P1" and c.ok
               and c.hi("x-ratelimit-remaining-tokens") is not None]
        if len(pts) < 3:
            continue
        lim = pts[0].hi("x-ratelimit-limit-tokens")
        rate = lim / 60
        series = [{"t_admission": iso(c.t_adm), "offset_s": c.step.get("actual_offset_s"), "after": c.step.get("after"),
                   "queue_time_s": c.usage.get("queue_time"),
                   "remaining_tokens": c.hi("x-ratelimit-remaining-tokens"), "reset_tokens": c.h.get("x-ratelimit-reset-tokens"),
                   "wall_second": round(c.t_adm % 60, 2)} for c in pts]
        incs = []
        for p, q in zip(pts, pts[1:]):
            dt = q.t_adm - p.t_adm
            own_q = (q.prompt_tokens or 0) + ((q.max_tokens or 0) if cm["admission_max_tokens"] else 0)
            if cm["admission_max_tokens"]:
                post_p = -((p.max_tokens or 0) - (p.completion_tokens or 0)) if cm["refund"] else 0
            else:
                post_p = p.completion_tokens or 0
            if p.hi("x-ratelimit-remaining-tokens") + rate * dt - post_p >= lim - 50:
                continue   # the bucket could have filled up in between: no information about refill
            refill = q.hi("x-ratelimit-remaining-tokens") - p.hi("x-ratelimit-remaining-tokens") + own_q + post_p
            incs.append({"dt_s": round(dt, 3), "refill_est": refill, "bucket_expect": round(rate * dt, 1),
                         "minute_boundary_crossed": int(q.t_adm // 60) != int(p.t_adm // 60),
                         "corrections": {"own_admission_q": own_q, "after_p_completed": post_p}})
        info = [i for i in incs if i["bucket_expect"] > 30]
        rising = [i for i in info if i["refill_est"] > 0.4 * i["bucket_expect"]]
        flat_no_boundary = [i for i in info if not i["minute_boundary_crossed"] and i["refill_est"] < 0.2 * i["bucket_expect"]]
        slopes = [i["refill_est"] / i["dt_s"] for i in incs if i["dt_s"] > 0.5]
        if len(info) >= 2 and len(rising) >= 0.7 * len(info):
            verdict = "refilling bucket"
        elif len(info) >= 2 and len(flat_no_boundary) >= 0.5 * len(info):
            verdict = "fixed window (no refill between polls inside a minute)"
        elif len(info) == 1 and rising and abs(info[0]["refill_est"] / info[0]["bucket_expect"] - 1) < 0.15:
            verdict = "refilling bucket (one informative interval only; it matches TPM/60 within 15%)"
        else:
            verdict = f"inconclusive ({len(info)} informative intervals)"
        reset_ratio = [c.hs("x-ratelimit-reset-tokens") / ((lim - c.hi("x-ratelimit-remaining-tokens")) / rate)
                       for c in calls if c.model == m and c.ok and c.hs("x-ratelimit-reset-tokens")
                       and c.hi("x-ratelimit-remaining-tokens") is not None and lim - c.hi("x-ratelimit-remaining-tokens") > 100]
        out.append({"run_id": run, "limit_tokens": lim, "bucket_refill_per_s_if_tpm_over_60": round(rate, 2),
                    "charge_model_used": cm, "series": series, "intervals": incs,
                    "median_refill_per_s": round(statistics.median(slopes), 2) if slopes else None,
                    "reset_tokens_over_time_to_refill_at_tpm_per_60": dist(reset_ratio, 3),
                    "verdict": verdict})
    return out or None


def requests_window_and_pooling(calls: list[Call], events: list[dict]) -> list[dict]:
    """P3: where the RPM wall fell per model, whether qwen was refused too, and how recovery went."""
    out = []
    for run in sorted({c.run_id for c in calls if c.phase == "P3"}):
        bs = next((e for e in events if e["run_id"] == run and e["event"] == "burst_start"), None)
        t_burst = datetime.fromisoformat(bs["wall"]).timestamp() if bs else None
        per = {}
        for m in MODELS:
            cs = [c for c in calls if c.run_id == run and c.model == m and c.phase == "P3"]
            if not cs:
                continue
            burst = sorted([c for c in cs if c.variant in ("burst", "bystander")], key=lambda c: c.t_headers or c.t0)
            first429 = next((c for c in burst if c.status == 429), None)
            # successes among requests SENT before the first refused one (a fast 429 can overtake
            # slower successes that were admitted earlier, so header arrival order would undercount)
            first429_sent = min((c.t0 for c in burst if c.status == 429), default=None)
            ok_before = sum(1 for c in burst if c.ok and (first429_sent is None or c.t0 < first429_sent))
            # requests of this model admitted in the 60 s before the burst (should be zero by design)
            prior = [c for c in calls if c.model == m and c.ok and c.t0 < (t_burst or 0) and c.t0 > (t_burst or 0) - 60]
            rec = sorted([c for c in cs if c.variant in ("recovery", "bystander_recovery")], key=lambda c: c.t0)
            first_ok = next((c for c in rec if c.ok), None)
            last_fail = None
            for c in rec:
                if c.ok:
                    break
                last_fail = c
            d: dict[str, Any] = {
                "burst_calls": len(burst), "ok_before_first_429": ok_before,
                "ok_in_burst": sum(1 for c in burst if c.ok),
                "admitted_in_60s_before_burst": len(prior),
                "statuses": [c.status for c in burst],
                "first_429": None if first429 is None else {
                    "at": iso(first429.t_headers), "s_after_burst_start": round((first429.t_headers or 0) - (t_burst or 0), 3),
                    "retry_after": first429.h.get("retry-after"), "reset_requests": first429.h.get("x-ratelimit-reset-requests"),
                    "remaining_requests": first429.h.get("x-ratelimit-remaining-requests"),
                    "remaining_tokens": first429.h.get("x-ratelimit-remaining-tokens"),
                    "body": first429.body_message(), "parsed": first429.parsed_limit()},
                "recovery": [{"offset_s": c.step.get("actual_offset_s", c.step.get("offset_s")), "status": c.status,
                              "wall": iso(c.t0), "retry_after": c.h.get("retry-after"),
                              "try_again_in": (c.parsed_limit() or {}).get("try_again_in")} for c in rec],
            }
            if first429 is not None and first_ok is not None:
                t429 = first429.t_headers or first429.t0
                d["first_recovery_success_s_after_first_429"] = round(first_ok.t0 - t429, 3)
                d["minute_boundary_between_429_and_recovery"] = int(first_ok.t0 // 60) != int(t429 // 60)
                d["first_recovery_success_s_after_burst_start"] = round(first_ok.t0 - (t_burst or first_ok.t0), 3)
                if last_fail is not None:
                    d["last_failure_s_after_first_429"] = round(last_fail.t0 - t429, 3)
                    d["minute_boundary_between_last_failure_and_success"] = int(first_ok.t0 // 60) != int(last_fail.t0 // 60)
                ra = parse_duration(first429.h.get("retry-after"))
                d["retry_after_s"] = ra
                d["try_again_in_s"] = (first429.parsed_limit() or {}).get("try_again_in_s")
                if d["first_recovery_success_s_after_first_429"] <= 5.5:
                    d["verdict"] = "refilling bucket (a request was readmitted within seconds)"
                elif d.get("minute_boundary_between_last_failure_and_success") or (last_fail is None and d["minute_boundary_between_429_and_recovery"]):
                    d["verdict"] = "fixed window aligned to the wall-clock minute"
                elif t_burst and 55 <= first_ok.t0 - t_burst <= 70:
                    d["verdict"] = "sliding 60 s window (recovered ~60 s after the burst began)"
                else:
                    d["verdict"] = "inconclusive"
            elif first429 is not None:
                d["verdict"] = "no recovery seen within the polling schedule"
            # RPD accounting across the burst: did 429s decrement remaining-requests?
            seq_rr = [(c.status, c.hi("x-ratelimit-remaining-requests")) for c in burst + rec if c.hi("x-ratelimit-remaining-requests") is not None]
            d["remaining_requests_sequence"] = seq_rr
            per[m] = d
        a, b, q = (per.get(MODELS[0]), per.get(MODELS[1]), per.get(MODELS[2]))
        if a and b and q:
            q429 = sum(1 for s in q["statuses"] if s == 429)
            walls = [x["ok_before_first_429"] for x in (a, b) if x["first_429"]]
            if q429 == 0 and walls and min(walls) >= 25:
                pooled = "per model (each model hit its own wall; the bystander model was never refused)"
            elif q429 > 0 or (walls and min(walls) < 20):
                pooled = "pooled or partly pooled (walls came early or the bystander was refused)"
            elif not walls:
                pooled = "not measured (neither gpt-oss model reached an RPM wall)"
            else:
                pooled = "inconclusive"
        else:
            pooled = "not measured"
        out.append({"run_id": run, "burst_start": bs["wall"] if bs else None, "per_model": per, "pooling_verdict": pooled})
    return out


def rpd_429_counting(calls: list[Call]) -> list[dict]:
    """Does a 429 count toward the daily request allowance? The lowest x-ratelimit-remaining-requests
    ever reported (on any response, 429s included) is the count after every admission, whatever order
    concurrent responses came back in. Compare what it says was used with what we sent. Other users
    of the shared key can only make the used count larger."""
    out = []
    for m in MODELS:
        cs = [c for c in calls if c.model == m]
        vals = [(c.hi("x-ratelimit-remaining-requests"), c) for c in cs if c.hi("x-ratelimit-remaining-requests") is not None]
        lims = {c.hi("x-ratelimit-limit-requests") for c in cs if c.hi("x-ratelimit-limit-requests") is not None}
        if not vals or len(lims) != 1:
            continue
        lim = lims.pop()
        low, low_call = min(vals, key=lambda v: v[0])
        n200 = sum(1 for c in cs if c.status == 200)
        n429 = sum(1 for c in cs if c.status == 429)
        nother = sum(1 for c in cs if c.status not in (200, 429))
        used = lim - low
        first_seen = max(v for v, _ in vals)
        v429 = [c.hi("x-ratelimit-remaining-requests") for c in sorted(cs, key=lambda c: c.t0)
                if c.status == 429 and c.hi("x-ratelimit-remaining-requests") is not None]
        ok_vals = [c.hi("x-ratelimit-remaining-requests") for c in cs if c.status == 200
                   and c.hi("x-ratelimit-remaining-requests") is not None]
        dupes = len(ok_vals) - len(set(ok_vals))
        # Direct test: adjacent 429s with no success sent between them. If a 429 counted, the second
        # would report one fewer remaining request than the first.
        pairs429 = drops429 = 0
        seq = sorted(cs, key=lambda c: c.t0)
        for x, y in zip(seq, seq[1:]):
            if x.status == 429 and y.status == 429 and None not in (x.hi("x-ratelimit-remaining-requests"), y.hi("x-ratelimit-remaining-requests")):
                pairs429 += 1
                drops429 += y.hi("x-ratelimit-remaining-requests") < x.hi("x-ratelimit-remaining-requests")
        if pairs429 >= 2:
            verdict = (f"429s not counted ({drops429} of {pairs429} back-to-back 429 pairs showed a drop)"
                       if drops429 <= 0.2 * pairs429 else
                       f"429s counted ({drops429} of {pairs429} back-to-back 429 pairs showed a drop)"
                       if drops429 >= 0.8 * pairs429 else f"ambiguous ({drops429} of {pairs429} pairs dropped)")
        elif not v429:
            verdict = "no 429s for this model"
        else:
            verdict = "too few back-to-back 429s to tell"
        out.append({"model": m, "limit_requests": lim, "highest_remaining_seen": first_seen,
                    "lowest_remaining_seen": low, "lowest_seen_at": iso(low_call.t0), "used_per_header": used,
                    "our_200s": n200, "our_429s": n429, "our_other_responses": nother,
                    "expect_if_429_counted": n200 + n429, "expect_if_not": n200,
                    "used_before_our_first_call": lim - 1 - first_seen,
                    "remaining_on_429s_in_order": v429, "duplicate_values_among_successes": dupes,
                    "header_count_minus_our_count": used - n200,
                    "verdict": verdict})
    return out


def _rpd_429_local(calls: list[Call]) -> list[dict]:
    """Does a 429 decrement x-ratelimit-remaining-requests? Compare the successes just before and
    just after a run of 429s."""
    out = []
    for m in MODELS:
        cs = [c for c in calls if c.model == m and c.phase == "P3" and c.hi("x-ratelimit-remaining-requests") is not None]
        cs.sort(key=lambda c: c.t_headers or c.t0)
        for i, c in enumerate(cs):
            if c.status != 429:
                continue
            prev_ok = next((x for x in reversed(cs[:i]) if x.ok), None)
            n429 = 0
            j = i
            while j < len(cs) and cs[j].status == 429:
                n429 += 1
                j += 1
            next_ok = cs[j] if j < len(cs) and cs[j].ok else None
            if prev_ok and next_ok:
                ok_between = 0  # the next success itself is charged once
                drop = prev_ok.hi("x-ratelimit-remaining-requests") - next_ok.hi("x-ratelimit-remaining-requests")
                out.append({"model": m, "prev_ok_remaining": prev_ok.hi("x-ratelimit-remaining-requests"),
                            "n_429_between": n429, "next_ok_remaining": next_ok.hi("x-ratelimit-remaining-requests"),
                            "drop": drop, "expect_if_429_counted": n429 + 1 + ok_between,
                            "expect_if_not": 1 + ok_between,
                            "429_remaining_values": [x.hi("x-ratelimit-remaining-requests") for x in cs[i:j]]})
            break
    return out


# ----------------------------------------------------------------------------- report
def q(v: Any) -> str:
    return "`" + str(v).replace("`", "'") + "`"


def run_findings(calls, events, doc, cal, tokwin, p3, rpd429, mode, truths) -> tuple[str, list[str], dict]:
    L: list[str] = []
    contra: list[str] = []
    verdicts: dict[str, Any] = {}
    runs = sorted({c.run_id for c in calls})
    L.append("# Probe findings")
    L.append("")
    L.append(f"Generated by `scripts/analyze_probe.py` from {len(calls)} logged calls in {len(runs)} run(s) "
             f"({', '.join(runs)}); mode: **{mode}**. Documentation claims: `data/limits_documented.json` "
             f"(retrieved {doc['retrieved_at']}). Every number below is computed from the raw log.")
    if mode != "real":
        L.append("")
        L.append(f"**This is a dry-run against a local fake whose behaviour we chose ({', '.join(truths)}). "
                 "Nothing here describes Groq.**")
    L.append("")

    # Q1
    L.append("## Q1. Fixed windows or refilling buckets?")
    L.append("")
    L.append("**Tokens (P1 polls after a draft call).**")
    v_tok = {}
    for m in MODELS:
        tw = tokwin.get(m)
        if not tw:
            continue
        for r in tw:
            v_tok[m] = r["verdict"]
            L.append(f"- **{SHORT[m]}: {r['verdict']}.** Median refill {r['median_refill_per_s']} tokens/s vs "
                     f"{r['bucket_refill_per_s_if_tpm_over_60']}/s if the TPM ({r['limit_tokens']}) refilled evenly. "
                     f"`x-ratelimit-remaining-tokens` by seconds after the draft call: "
                     + ", ".join(f"+{p['offset_s']}: {q(p['remaining_tokens'])}" for p in r["series"])
                     + f". `reset-tokens` / time-to-refill-at-TPM/60: p50 {r['reset_tokens_over_time_to_refill_at_tpm_per_60'].get('p50')} "
                     f"(n={r['reset_tokens_over_time_to_refill_at_tpm_per_60'].get('n')}).")
    if not v_tok:
        L.append("- Not measured.")
    L.append("")
    L.append("**Requests (recovery after the RPM wall in P3).**")
    v_req = {}
    for run in p3:
        for m, d in run["per_model"].items():
            if m == MODELS[2] and not d.get("first_429"):
                continue
            if not d.get("first_429"):
                L.append(f"- **{SHORT[m]}:** no 429 in {d['burst_calls']} burst requests.")
                continue
            v_req[m] = d.get("verdict")
            rec = "; ".join(f"+{r['offset_s']}s {r['status']}" for r in d["recovery"])
            L.append(f"- **{SHORT[m]}: {d.get('verdict')}.** First 429 at {d['first_429']['at']} "
                     f"(wall second {round(datetime.fromisoformat(d['first_429']['at']).timestamp() % 60, 1)}); "
                     f"`retry-after: {d['first_429']['retry_after']}`; body says try again in "
                     f"{q((d['first_429']['parsed'] or {}).get('try_again_in'))}. Recovery polls: {rec}. "
                     f"First success {d.get('first_recovery_success_s_after_first_429')} s after the first 429.")
    if not v_req:
        L.append("- Not measured.")
    verdicts["tokens_window"] = v_tok
    verdicts["requests_window"] = v_req
    L.append("")

    # Q2
    L.append("## Q2. Charged on max_tokens at admission, or on actual usage?")
    L.append("")
    v_adm = {}
    for m in MODELS:
        a = cal.get(m, {}).get("admission") or {}
        for p in a.get("pairs", []):
            v_adm[m] = p["verdict"]
            L.append(f"- **{SHORT[m]}: {p['verdict']}.** Baseline `x-ratelimit-remaining-tokens: {p['baseline']['remaining_tokens']}`; "
                     f"call A (max_tokens {p['A']['max_tokens']}, prompt {p['A']['prompt_tokens']} tokens, used "
                     f"{p['A']['completion_tokens']} completion tokens) got `{p['A']['remaining_tokens']}` in its headers, "
                     f"{p['seconds_between_baseline_and_A_admission']} s later (admission to admission): a drop of {p['drop_at_A_raw']}. "
                     f"Expected {p['expect_if_max_tokens_held']} if max_tokens is held at admission, "
                     f"{p['expect_if_usage']} if only usage is charged (refill at TPM/60 included)."
                     + (f" Call B, {p['B']['seconds_after_A_admission']} s later: `{p['B']['remaining_tokens']}` "
                        f"(rise {p['B']['rise_since_A']}; A left {p['B']['A_unused_max_tokens']} max_tokens unused) "
                        + (f"**{p['B']['refund_verdict']}**." if p['B'].get('refund_verdict') else "(no max_tokens hold to refund).")
                        if p.get("B") else ""))
        for o in a.get("overflow", []):
            L.append(f"- **{SHORT[m]} P2-overflow** (max_tokens {o['max_tokens']} > TPM): HTTP {o['status']} after "
                     f"{o['headers_ms'] and round(o['headers_ms'])} ms; body: {q(o['body'])}; `retry-after: {o['retry_after']}`.")
    L.append("")
    L.append("**All consecutive non-overlapping calls** (same model), fitted to two accounting rules on two time "
             "bases. Median |residual| in tokens between the observed `x-ratelimit-remaining-tokens` and the "
             "prediction (bucket refilling at TPM/60 per second, capped at the limit; saturated pairs excluded):")
    L.append("")
    L.append("| model | pairs | usage @ admission time | usage @ header time | max_tokens held @ admission time | max_tokens held @ header time | best |")
    L.append("|---|---|---|---|---|---|---|")
    for m in MODELS:
        ta = cal.get(m, {}).get("token_accounting_fit")
        if not ta:
            continue
        f = ta["fit"]
        cell = lambda k: f"{_r(f[k]['median_abs_residual'])} (n={f[k]['n_unsaturated']})"  # noqa: E731
        L.append(f"| {SHORT[m]} | {len(ta['pairs'])} | {cell('usage@admission')} | {cell('usage@headers')} | "
                 f"{cell('admission@admission')} | {cell('admission@headers')} | {ta['best']} |")
    L.append("")
    for m in MODELS:
        ta = cal.get(m, {}).get("token_accounting_fit")
        if not ta:
            continue
        big = {k: v for k, v in ta["usage_residual_by_max_tokens_pair"].items() if abs(v["median"]) > 60}
        if big:
            L.append(f"- {SHORT[m]}: under the best-fitting rule, residuals larger than 60 tokens cluster on these "
                     f"max_tokens transitions (previous -> next call): "
                     + "; ".join(f"{k}: median {v['median']} (n={v['n']}, {v['min']}..{v['max']})" for k, v in big.items())
                     + ". A positive residual means more tokens were left than the rule predicts.")
    ev = [e for e in events if e["event"] == "skipped" and e["phase"] == "P2-overflow"]
    for e in ev:
        L.append(f"- P2-overflow skipped: {e['reason']}.")
    # reasoning tokens in usage
    for m in MODELS:
        cs = [c for c in calls if c.model == m and c.ok and c.completion_tokens and c.r.get("reasoning_chars")]
        if cs:
            vis = sum(len(c.r.get("text") or "") for c in cs)
            rch = sum(c.r.get("reasoning_chars") or 0 for c in cs)
            ct = sum(c.completion_tokens for c in cs)
            L.append(f"- **{SHORT[m]} reasoning in usage:** over {len(cs)} calls with streamed reasoning, "
                     f"completion_tokens total {ct} vs {vis} visible chars and {rch} reasoning chars "
                     f"(≈{round(ct / max(1, (vis + rch)) , 3)} tokens/char overall): "
                     + ("reasoning tokens are counted in completion_tokens." if ct > 1.5 * vis / 4 + 5 else
                        "completion_tokens look like visible text only."))
    verdicts["admission"] = v_adm
    L.append("")

    # Q3
    L.append("## Q3. Per model or pooled?")
    L.append("")
    v_pool = None
    for run in p3:
        v_pool = run["pooling_verdict"]
        L.append(f"- **{run['pooling_verdict']}.** Burst began {run['burst_start']}.")
        for m, d in run["per_model"].items():
            L.append(f"  - {SHORT[m]}: {d['burst_calls']} burst requests; {d['ok_before_first_429']} of those sent before "
                     f"the first refused one succeeded ({d['ok_in_burst']} in the whole burst); statuses {q(' '.join(str(s) for s in d['statuses']))}; "
                     f"{d['admitted_in_60s_before_burst']} of our calls to it in the 60 s before."
                     + (f" First 429 body: {q(d['first_429']['body'])}" if d.get("first_429") else ""))
    if not p3:
        L.append("- Not measured (P3 did not run).")
    # cross-model header evidence
    for run in sorted({c.run_id for c in calls}):
        c120 = next((c for c in calls if c.run_id == run and c.model == MODELS[1] and c.phase == "P0" and c.ok), None)
        last20 = next((c for c in reversed(calls) if c.run_id == run and c.model == MODELS[0] and c.ok and c120 and c.t0 < c120.t0), None)
        if c120 and last20:
            L.append(f"- Header cross-check: the first {SHORT[MODELS[1]]} call, {round(c120.t0 - last20.t_end, 1)} s after "
                     f"{SHORT[MODELS[0]]}'s last call (which saw `remaining-tokens: {last20.h.get('x-ratelimit-remaining-tokens')}`), "
                     f"saw `remaining-tokens: {c120.h.get('x-ratelimit-remaining-tokens')}` and "
                     f"`remaining-requests: {c120.h.get('x-ratelimit-remaining-requests')}` "
                     f"(vs {SHORT[MODELS[0]]} `{last20.h.get('x-ratelimit-remaining-requests')}`).")
    firsts = {m: next((c for c in calls if c.model == m and c.status == 200), None) for m in MODELS}
    if all(firsts.values()):
        L.append("- `x-ratelimit-remaining-requests` on each model's first successful call: "
                 + ", ".join(f"{SHORT[m]} `{c.h.get('x-ratelimit-remaining-requests')}` ({iso(c.t0)})" for m, c in firsts.items())
                 + ". Each model starts its own daily count, so the daily request allowance is per model too.")
    verdicts["pooling"] = v_pool
    L.append("")

    # Q4
    L.append("## Q4. Is there a concurrency limit? Latency vs requests in flight")
    L.append("")
    L.append("| model | k | calls | ok | non-200 | server TTFB p50 ms | first token p50 ms | total p50 ms | total max ms |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    conc_429 = []
    v_conc = {}
    for m in MODELS:
        rows = cal.get(m, {}).get("latency_vs_in_flight") or []
        for r in rows:
            L.append(f"| {SHORT[m]} | {r['k']} | {r['calls']} | {r['ok']} | {r['non_200'] or ''} | {_r(r['server_ttfb_ms_p50'])} | "
                     f"{_r(r['first_token_after_connect_ms_p50'])} | {_r(r['total_ms_p50'])} | {_r(r['total_ms_max'])} |")
        bad = [c for c in calls if c.model == m and c.phase == "P4" and c.status != 200]
        conc_429 += bad
        if rows:
            k1 = next((r for r in rows if r["k"] == 1), None)
            kmax = rows[-1]
            growth = (kmax["server_ttfb_ms_p50"] / k1["server_ttfb_ms_p50"]) if k1 and k1["server_ttfb_ms_p50"] and kmax["server_ttfb_ms_p50"] else None
            v_conc[m] = (f"no limit up to k={kmax['k']}" if not bad else f"{len(bad)} non-200 responses at k≤{kmax['k']}") + \
                        (f"; server TTFB p50 x{round(growth, 2)} from k=1 to k={kmax['k']}" if growth else "")
    L.append("")
    for m, v in v_conc.items():
        L.append(f"- **{SHORT[m]}:** {v}.")
    for c in conc_429:
        L.append(f"- P4 {SHORT[c.model]} k={c.step.get('k')}: HTTP {c.status} body {q(c.body_message())}")
    verdicts["concurrency"] = v_conc
    L.append("")

    # Q5
    L.append("## Q5. What 429 bodies say that headers don't")
    L.append("")
    seen = set()
    for c in calls:
        if not c.status or c.status < 400:
            continue
        pl = c.parsed_limit()
        key = (c.model, c.status, (pl or {}).get("short"), (pl or {}).get("form"), c.variant if c.status == 400 else "")
        if key in seen:
            continue
        seen.add(key)
        hdrs = ", ".join(f"`{k}: {v}`" for k, v in c.h.items())
        L.append(f"- **{SHORT.get(c.model, c.model)} HTTP {c.status}** ({c.phase} {c.variant}). Body: {q(c.body_message())}. "
                 f"Headers on the same response: {hdrs or 'none'}.")
        if pl:
            L.append(f"  - The body names the limit ({pl['what']}, {pl['short']}) and gives Limit {pl['limit']}"
                     + (f", Used {pl['used']}" if 'used' in pl else "") + f", Requested {pl['requested']}"
                     + (f", try again in {pl['try_again_in']}" if pl.get("try_again_in") else "")
                     + ". No header names which limit was hit"
                     + ("; no header reports RPM at all." if pl["short"] == "RPM" else "."))
    if not seen:
        L.append("- No error responses were logged.")
    if not any((c.parsed_limit() or {}).get("short") == "TPM" for c in calls):
        L.append("- **No TPM 429 or 413 was produced**, so the token-limit body is not measured. P2-overflow is the free "
                 "way to get one, and it runs only when admission holds max_tokens. Buying a TPM wall with real tokens "
                 "was ruled out in the plan (§3).")
    hdr429 = sorted({k for c in calls if c.status == 429 for k in c.h})
    if hdr429:
        L.append(f"- Headers present on 429 responses: {', '.join('`' + k + '`' for k in hdr429)}. "
                 + ("No token headers (`x-ratelimit-*-tokens`) on a 429." if not any(k.endswith('-tokens') for k in hdr429) else ""))
    L.append("")
    if rpd429:
        L.append("**Do 429s count toward the daily request allowance?**")
        for r in rpd429:
            L.append(f"- {SHORT[r['model']]}: lowest `x-ratelimit-remaining-requests` seen `{r['lowest_remaining_seen']}` "
                     f"of `{r['limit_requests']}` (highest `{r['highest_remaining_seen']}`, on our first call), so "
                     f"{r['used_per_header']} requests used today; we sent {r['our_200s']} that were admitted and "
                     f"{r['our_429s']} that got 429. The 429s, in order, reported {r['remaining_on_429s_in_order'] or 'nothing'} "
                     f"→ **{r['verdict']}**. Header count minus our count of admitted calls: "
                     f"{r['header_count_minus_our_count']}; {r['duplicate_values_among_successes']} successes reported a "
                     f"value another success also reported, so the header count is approximate under concurrency.")
        L.append("")

    # Q6
    L.append("## Q6. Latency per model, router-sized and draft-sized calls")
    L.append("")
    L.append(f"Sequential calls only (P0–P2, P1, P4 k=1, P6). Times from call start include a new TCP+TLS connection "
             f"(connect p50 shown separately); 'after connect' is what a kept-alive connection would see. "
             f"p95 is marked * where n < {P95_MIN_N} (it is then close to the maximum, not a real tail estimate).")
    L.append("")
    L.append("| model | size | n | prompt tok p50 | connect p50 ms | server TTFB p50 / p95 ms | first visible after connect p50 / p95 ms | total p50 / p95 ms | max ms | cut at max_tokens |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for m in MODELS:
        c = cal.get(m)
        if not c:
            continue
        for size in ("router", "draft", "tiny"):
            s = c["by_size"].get(size)
            if not s:
                continue
            tt, fv, tot = s["server_ttfb_ms"], s["first_visible_after_connect_ms"], s["total_ms"]
            L.append(f"| {SHORT[m]} | {size} | {tot['n']} | {s['prompt_tokens'].get('p50')} | {c['connect_ms'].get('p50')} | "
                     f"{_pp(tt)} | {_pp(fv)} | {_pp(tot)} | {tot.get('max')} | {s['finish_reason_length']} |")
    L.append("")
    for m in MODELS:
        c = cal.get(m)
        if not c:
            continue
        f = c["first_token_fit_vs_prompt_tokens"]["first_token"]
        tps = c["tokens_per_s"]
        qf = c["server_ttfb_vs_queue_time"]
        qd = c["server_queue_time_s"]
        if qf["fit"]:
            L.append(f"- **{SHORT[m]} queueing:** server-reported `queue_time` p50 {qd.get('p50')} s, p95 {qd.get('p95')} s, "
                     f"max {qd.get('max')} s (n={qd.get('n')}). Server TTFB ≈ {qf['fit']['intercept']} ms + "
                     f"{round(qf['fit']['slope'])} ms per second of queue_time (R²={qf['fit']['r2']}); queue_time is "
                     f"{round(qf['queue_time_share_of_server_ttfb_p50'] * 100) if qf['queue_time_share_of_server_ttfb_p50'] else '?'}% "
                     f"of server TTFB at the median.")
        L.append(f"- **{SHORT[m]}:** first token ≈ {f and f['intercept']} ms + {f and round(f['slope'] * 1000, 3)} ms per 1K prompt "
                 f"tokens (n={f and f['n']}, R²={f and f['r2']}); decode {tps['client_side'].get('p50')} tokens/s client-side, "
                 f"{tps['server_reported'].get('p50')} server-reported (p50); chunk gaps p50 {c['chunk_gaps_ms'].get('p50')} ms, "
                 f"p95 {c['chunk_gaps_ms'].get('p95')} ms.")
    L.append("")

    # contradictions
    for m in MODELS:
        c = cal.get(m)
        if not c:
            continue
        ls = c["limits_seen"]
        d = ls["documented"]
        for v in ls["headers"].get("x-ratelimit-limit-requests", []):
            if int(v) != d["rpd"]:
                contra.append(f"{SHORT[m]}: `x-ratelimit-limit-requests: {v}`, docs say RPD {d['rpd']}.")
        for v in ls["headers"].get("x-ratelimit-limit-tokens", []):
            if int(v) != d["tpm"]:
                contra.append(f"{SHORT[m]}: `x-ratelimit-limit-tokens: {v}`, docs say TPM {d['tpm']}.")
        for short, vals in ls["from_error_bodies"].items():
            key = {"RPM": "rpm", "TPM": "tpm", "RPD": "rpd", "TPD": "tpd"}.get(short)
            for v in vals:
                if key and v != d[key]:
                    contra.append(f"{SHORT[m]}: an error body reports {short} Limit {v}, docs say {d[key]}.")
        for b in ls["error_responses"]:
            if b["status"] == 429 and b["retry_after"] is None:
                contra.append(f"{SHORT[m]}: a 429 in {b['phase']} had no `retry-after` header (docs: set on 429).")
                break
        if c["prompt_token_estimate"]["estimate_below_actual_calls"]:
            contra.append(f"{SHORT[m]}: our prompt estimate was below the real prompt_tokens on "
                          f"{c['prompt_token_estimate']['estimate_below_actual_calls']} calls (ledger reservation not an upper bound).")
        for p in c["reasoning_p0"]:
            if p["variant"] == "effort_none" and p["status"] != 200:
                contra.append(f"{SHORT[m]}: `reasoning_effort: none` was refused ({p['status']}: {p['error']}); docs list it.")
            if p["variant"] == "effort_default_parsed" and p["status"] == 200 and (p["reasoning_chars"] or 0) > 0:
                contra.append(f"{SHORT[m]}: `reasoning_effort: default` returned {p['reasoning_chars']} reasoning chars; "
                              f"docs say no reasoning tokens are returned at default.")
    for run in p3:
        for m, dd in run["per_model"].items():
            if m == MODELS[2]:
                continue
            doc_rpm = doc["models"][m]["rpm"]
            if dd.get("first_429") is None and dd["burst_calls"] > doc_rpm:
                contra.append(f"{SHORT[m]}: {dd['burst_calls']} requests inside one minute, no 429 (docs RPM {doc_rpm}).")
            elif dd.get("first_429") and dd["admitted_in_60s_before_burst"] == 0 and abs(dd["ok_before_first_429"] - doc_rpm) > 2:
                contra.append(f"{SHORT[m]}: the RPM wall fell after {dd['ok_before_first_429']} requests, docs say {doc_rpm}.")
    L.append("## Contradictions with the documentation")
    L.append("")
    L += [f"- ⚠ {x}" for x in contra] or ["- None found in what was measured."]
    L.append("")
    return "\n".join(L), contra, verdicts


def _r(v):
    return "" if v is None else round(v)


def _pp(d: dict) -> str:
    if not d.get("n"):
        return ""
    return f"{_r(d['p50'])} / {_r(d['p95'])}{'' if d['p95_reliable'] else '*'}"


def limits_table(cal: dict, tokwin: dict, p3: list[dict], doc: dict, mode: str) -> str:
    L = ["# Table 1. Groq free-tier limits: documented vs observed", ""]
    if mode != "real":
        L += ["**Dry-run against a local fake. Not Groq.**", ""]
    L += ["Generated by `scripts/analyze_probe.py`. Documented values from `data/limits_documented.json`; "
          "observed values from response headers and 429/413 bodies in the probe log.", ""]
    L.append("| | " + " | ".join(SHORT[m] for m in MODELS) + " |")
    L.append("|---|" + "---|" * len(MODELS))

    def row(label, fn):
        cells = []
        for m in MODELS:
            try:
                v = fn(m)
            except Exception:
                v = None
            cells.append("—" if v in (None, "", []) else str(v))
        L.append(f"| {label} | " + " | ".join(cells) + " |")

    row("Status (docs)", lambda m: doc["models"][m]["status"])
    row("RPM documented", lambda m: doc["models"][m]["rpm"])
    row("RPM observed (429 body Limit)", lambda m: ", ".join(map(str, cal[m]["limits_seen"]["from_error_bodies"].get("RPM", []))))
    row("RPM wall: successes sent before the first 429", lambda m: next((r["per_model"][m]["ok_before_first_429"] for r in p3
                                                                if m in r["per_model"] and r["per_model"][m].get("first_429")), None))
    row("RPD documented", lambda m: doc["models"][m]["rpd"])
    row("RPD observed (`x-ratelimit-limit-requests`)", lambda m: ", ".join(cal[m]["limits_seen"]["headers"].get("x-ratelimit-limit-requests", [])))
    row("TPM documented", lambda m: doc["models"][m]["tpm"])
    row("TPM observed (`x-ratelimit-limit-tokens`)", lambda m: ", ".join(cal[m]["limits_seen"]["headers"].get("x-ratelimit-limit-tokens", [])))
    row("TPM observed (429/413 body Limit)", lambda m: ", ".join(map(str, cal[m]["limits_seen"]["from_error_bodies"].get("TPM", []))))
    row("TPD documented (not probed)", lambda m: doc["models"][m]["tpd"])
    row("Token limit behaviour (P1)", lambda m: "; ".join(r["verdict"] for r in tokwin.get(m) or []))
    row("Request limit behaviour (P3)", lambda m: next((r["per_model"][m].get("verdict") for r in p3
                                                        if m in r["per_model"] and r["per_model"][m].get("first_429")), None))
    row("Admission charge (P2)", lambda m: "; ".join(p["verdict"] + (f", {p['B']['refund_verdict']}" if p.get("B", {}).get("refund_verdict") else "")
                                                     for p in cal[m]["admission"]["pairs"]))
    row("Concurrency (P4)", lambda m: (lambda rows: f"no non-200 up to k={rows[-1]['k']}" if rows and not any(r['non_200'] for r in rows)
                                       else (f"non-200 at k={[r['k'] for r in rows if r['non_200']]}" if rows else None))(cal[m]["latency_vs_in_flight"]))
    row("Per model or pooled (P3)", lambda m: p3[-1]["pooling_verdict"] if p3 else None)
    row("Calls logged / ok", lambda m: f"{cal[m]['calls']} / {cal[m]['ok']}")
    L.append("")
    return "\n".join(L)


def truth_check(truths: list[str], verdicts: dict) -> list[str]:
    """Dry-run only: does the analysis recover the behaviour the fake was told to have?"""
    out = []
    for t in truths:
        window, charge, pool = t.split("-")
        exp_w = "bucket" if window == "bucket" else "window"
        for kind in ("tokens_window", "requests_window"):
            for m, v in verdicts.get(kind, {}).items():
                ok = (exp_w in (v or "")) if exp_w == "bucket" else ("window" in (v or "") and "bucket" not in (v or ""))
                out.append(f"- {'PASS' if ok else 'FAIL'} {kind} {SHORT[m]}: expected {exp_w}, got {v!r}")
        exp_a = "max_tokens" if charge.startswith("admission") else "only the prompt"
        for m, v in verdicts.get("admission", {}).items():
            out.append(f"- {'PASS' if v and v.startswith(exp_a) else 'FAIL'} admission {SHORT[m]}: expected {exp_a!r}, got {v!r}")
        pv = verdicts.get("pooling") or ""
        exp_p = "pooled" if pool == "pooled" else "per model"
        out.append(f"- {'PASS' if pv.startswith(exp_p) else 'FAIL'} pooling: expected {exp_p!r}, got {pv!r}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Analyse probe_raw.jsonl")
    ap.add_argument("--raw", default=str(ROOT / "data" / "probe" / "probe_raw.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "data" / "probe"))
    args = ap.parse_args()
    raw, out = Path(args.raw), Path(args.out)
    doc = json.loads((ROOT / "data" / "limits_documented.json").read_text(encoding="utf-8"))
    calls, events = load(raw)
    if not calls:
        print(f"no calls in {raw}")
        return 1
    modes = sorted({c.raw["mode"] for c in calls})
    if len(modes) > 1:
        print(f"mixed modes in one raw file: {modes}")
        return 1
    mode = modes[0]
    truths = sorted({c.raw.get("truth") for c in calls if c.raw.get("truth")})
    cal = calibration(calls, doc)
    tokwin = {m: tokens_window(calls, m, charge_model(cal.get(m))) for m in MODELS}
    p3 = requests_window_and_pooling(calls, events)
    rpd429 = rpd_429_counting(calls)
    findings, contra, verdicts = run_findings(calls, events, doc, cal, tokwin, p3, rpd429, mode, truths)
    if mode != "real":
        tc = truth_check(truths, verdicts)
        findings += "\n## Dry-run check: did the analysis recover the fake's behaviour?\n\n" + "\n".join(tc) + "\n"
    else:
        tc = []
    out.mkdir(parents=True, exist_ok=True)
    calib = {"_generated_by": "scripts/analyze_probe.py", "_raw": str(raw.relative_to(ROOT)) if raw.is_relative_to(ROOT) else str(raw),
             "mode": mode, "runs": sorted({c.run_id for c in calls}), "calls": len(calls),
             "models": cal, "tokens_window": tokwin, "requests_window_and_pooling": p3,
             "rpd_429_counting": rpd429, "verdicts": verdicts, "contradictions_with_docs": contra}
    if tc:
        calib["dry_run_truth_check"] = tc
    (out / "calibration.json").write_text(json.dumps(calib, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    (out / "limits_table.md").write_text(limits_table(cal, tokwin, p3, doc, mode), encoding="utf-8", newline="\n")
    (out / "findings.md").write_text(findings, encoding="utf-8", newline="\n")
    print(f"wrote {out / 'limits_table.md'}, {out / 'calibration.json'}, {out / 'findings.md'}")
    for line in tc:
        print(line.encode("ascii", "replace").decode())
    return 1 if any(l.startswith("- FAIL") for l in tc) else 0


if __name__ == "__main__":
    sys.exit(main())
