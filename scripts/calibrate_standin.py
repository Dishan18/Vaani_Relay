"""Stand-in parameters from the real probe (R1). No model calls.

    .venv\\Scripts\\python scripts\\calibrate_standin.py   ->  data/calibration.json

Per model, each value names where it came from ("measured": a file and field, n), or says it is
documented-only or an assumption. standin/server.py reads nothing else.
  ttfb        lognormal fitted to server_ttfb_ms p50/p95 (calibration.json ttfb.server_ttfb_ms):
              mu = ln p50, sigma = (ln p95 - ln p50) / 1.645
  queue       server_ttfb = intercept + slope * queue_time (calibration.json server_ttfb_vs_queue_time),
              used to report usage.queue_time consistently with the drawn TTFB
  decode      tokens/s p50 (tokens_per_s.server_reported); chars per completion token from probe_raw
  prompt      prompt_tokens = a + b*chars + c*messages, least squares on probe_raw 200s with the
              reasoning settings the pipeline uses (the central fit, not the limiter's upper bound)
  reasoning   reasoning_tokens observed on those calls (sampled empirically)
  buckets     RPM/TPM/RPD from the probe, plus limits later real runs found in 429 bodies (qwen's
              OTPM, Limiter.from_probe); TPD and qwen's RPM documented only
  hold_cap    admission holds prompt + min(max_tokens, hold_cap) (findings Q2; Limiter.from_probe)
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from relay.limiter import Limiter  # noqa: E402

CAL = ROOT / "data" / "probe" / "calibration.json"
RAW = ROOT / "data" / "probe" / "probe_raw.jsonl"
DOC = ROOT / "data" / "limits_documented.json"
OUT = ROOT / "data" / "calibration.json"
Z95 = 1.6448536


def ols3(rows):
    """Least squares y = a + b*x1 + c*x2 by normal equations (stdlib)."""
    import itertools
    X = [(1.0, x1, x2) for x1, x2, _ in rows]
    y = [r[2] for r in rows]
    A = [[sum(X[k][i] * X[k][j] for k in range(len(X))) for j in range(3)] for i in range(3)]
    b = [sum(X[k][i] * y[k] for k in range(len(X))) for i in range(3)]
    # Gauss-Jordan
    M = [A[i] + [b[i]] for i in range(3)]
    for c in range(3):
        p = max(range(c, 3), key=lambda r: abs(M[r][c]))
        if abs(M[p][c]) < 1e-12:
            return None
        M[c], M[p] = M[p], M[c]
        for r in range(3):
            if r != c:
                f = M[r][c] / M[c][c]
                M[r] = [M[r][i] - f * M[c][i] for i in range(4)]
    return [M[i][3] / M[i][i] for i in range(3)]


def admission_extras(hold_caps: dict) -> dict:
    """Per model: tokens Groq held at admission beyond prompt + hold_cap, measured from real TPM 429s whose
    request was retried with the identical prompt (Requested - that retry's prompt_tokens - hold_cap)."""
    import re
    out: dict[str, list] = {}
    runs = ROOT / "results" / "raw" / "answers"
    for p in sorted(runs.glob("*.calls.jsonl")) if runs.exists() else []:
        calls = [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
        for i, c in enumerate(calls):
            r = c["record"]
            if r.get("source") != "real" or r.get("status") != 429 or "(TPM)" not in (r.get("error_body") or ""):
                continue
            m = re.search(r"Requested (\d+)", r["error_body"])
            same = [d for d in calls[i + 1:] if d.get("request_id") == c.get("request_id") and d["phase"] == c["phase"]
                    and d["record"]["status"] == 200 and d["record"]["prompt_chars"] == r["prompt_chars"]]
            if m and same:
                pt = same[0]["record"]["usage"]["prompt_tokens"]
                extra = int(m.group(1)) - pt - (hold_caps.get(r["model"]) or 0)
                out.setdefault(r["model"], []).append({"extra": extra, "max_tokens": r["max_tokens"], "prompt_tokens": pt,
                                                       "requested": int(m.group(1)), "file": p.name})
    return out


def main() -> int:
    cal = json.loads(CAL.read_text(encoding="utf-8"))
    doc = json.loads(DOC.read_text(encoding="utf-8"))["models"]
    lim = Limiter.from_probe()
    raw = [json.loads(x) for x in RAW.read_text(encoding="utf-8").splitlines() if x.strip()]
    calls = [c for c in raw if c.get("type") == "call" and c.get("mode") == "real" and c["record"]["status"] == 200]
    keys = ("reasoning_effort", "reasoning_format", "include_reasoning")
    models = {}
    extras = admission_extras({m: st.hold_cap for m, st in lim.models.items()})
    for m, v in cal["models"].items():
        want = lim.settings.get(m, {})
        mine = [c for c in calls if c["model"] == m and all(c["payload"].get(k) == want.get(k) for k in keys)
                and (c["record"].get("usage") or {}).get("prompt_tokens")]
        t = v["ttfb"]["server_ttfb_ms"]
        mu = math.log(t["p50"])
        sigma = max(0.05, (math.log(t["p95"]) - mu) / Z95)
        qf = v["server_ttfb_vs_queue_time"]["fit"]
        tps = v["tokens_per_s"]["server_reported"]
        ct = sum(c["record"]["usage"]["completion_tokens"] for c in calls if c["model"] == m)
        cc = sum(len(c["record"].get("text") or "") + (c["record"].get("reasoning_chars") or 0)
                 for c in calls if c["model"] == m)
        fit = ols3([(c["prompt_chars"], len(c["payload"]["messages"]), c["record"]["usage"]["prompt_tokens"]) for c in mine])
        reasoning = sorted((c["record"]["usage"].get("completion_tokens_details") or {}).get("reasoning_tokens") or 0
                           for c in mine)
        st = lim.models[m]
        buckets = {}
        for name, b in st.buckets.items():
            buckets[name] = {"capacity": b.capacity, "period_s": b.period_s, "unit": b.unit,
                             "source": f"measured ({b.source})"}
        d = doc[m]
        if "RPM" not in buckets:
            buckets["RPM"] = {"capacity": d["rpm"], "period_s": 60.0, "unit": "requests",
                              "source": "documented only (the probe never hit this model's RPM)"}
        buckets["TPD"] = {"capacity": d["tpd"], "period_s": 86400.0, "unit": "tokens",
                          "source": "documented only (no header shows TPD; never hit)"}
        models[m] = {
            "settings": want,
            "ttfb_ms": {"dist": "lognormal", "mu": round(mu, 5), "sigma": round(sigma, 5),
                        "p50_target": t["p50"], "p95_target": t["p95"], "n": t["n"],
                        "source": "measured: data/probe/calibration.json ttfb.server_ttfb_ms"},
            "queue_fit": {"intercept_ms": qf["intercept"], "slope_ms_per_s": qf["slope"], "r2": qf["r2"],
                          "source": "measured: server_ttfb_vs_queue_time"},
            "decode": {"tokens_per_s": tps["p50"], "n": tps["n"], "chars_per_token": round(cc / ct, 3),
                       "source": "measured: tokens_per_s.server_reported p50; chars/token from probe_raw"},
            "chunk_gap_ms": {k: v["chunk_gaps_ms"][k] for k in ("p50", "p95", "max", "n")},
            "prompt_tokens_fit": {"a": round(fit[0], 3), "b_per_char": round(fit[1], 5), "c_per_message": round(fit[2], 3),
                                  "n": len(mine), "source": "least squares on probe_raw 200s with these settings"}
            if fit else None,
            "reasoning_tokens_samples": reasoning,
            "buckets": buckets,
            "hold_cap": st.hold_cap,
            "out_hold_cap": st.out_hold_cap,       # output-token limits (qwen OTPM): from real-run 429 bodies
            # measured from real TPM 429s (Requested - the identical retry's prompt - hold_cap): admission held
            # more than prompt + hold_cap on long drafts; the stand-in samples one of these per admission
            "admission_extra_samples": sorted(x["extra"] for x in extras.get(m, [])),
            "admission_extra_source": extras.get(m, []),
            "count_429_toward_rpd": False,
            "concurrency_limit": None,
            "latency_vs_in_flight": v.get("latency_vs_in_flight"),
        }
    out = {"_generated_by": "scripts/calibrate_standin.py", "_inputs": [str(p.relative_to(ROOT)) for p in (CAL, RAW, DOC)],
           "findings": {"limits": "refilling buckets per model (findings Q1, Q3)",
                        "admission": "prompt + min(max_tokens, hold_cap) held at admission, released to usage at the end (Q2)",
                        "429": "not counted toward RPD (Q5); RPM visible only in 429 bodies; retry-after = ceil(wait)",
                        "concurrency": "no limit found up to 8 in flight",
                        "load": "no consistent slowdown with in-flight calls up to 8 (latency_vs_in_flight)"},
           "models": models}
    OUT.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n")
    for m, v in models.items():
        print(f"{m:<22} ttfb p50 {v['ttfb_ms']['p50_target']} p95 {v['ttfb_ms']['p95_target']} "
              f"sigma {v['ttfb_ms']['sigma']}  tps {v['decode']['tokens_per_s']}  "
              f"prompt fit {v['prompt_tokens_fit']}  buckets {sorted(v['buckets'])}  hold {v['hold_cap']}")
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
