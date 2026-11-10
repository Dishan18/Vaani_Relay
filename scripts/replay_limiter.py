"""Replay the real probe log through relay.limiter and score its predictions against Groq's headers.

    .venv\\Scripts\\python scripts\\replay_limiter.py   ->  data/probe/limiter_replay.json (+ printed table)

Every real call is replayed at its real times (fake clock): granted when it was sent (the probe
didn't use this limiter, so nothing is refused), then settled with its real CallRecord when it
finished. Just before each response's headers are applied, the limiter's predicted level at the
call's admission time is compared with the value Groq reported. Only forecasts are scored: headers
newer than the bucket's last anchor (older ones arrive out of order and are ignored by design), and
not each model's first response (the state before it is unknown). Ignored headers are counted.
  * TPM vs x-ratelimit-remaining-tokens; RPD vs x-ratelimit-remaining-requests
  * "with headers": the limiter as built (each response overwrites the estimate)
  * "model only": the same limiter with header overwrite switched off, as a baseline
And for the P3 burst: what the limiter's RPM bucket said when each call was sent, versus whether
Groq accepted it.

Caveat: the estimator is seeded from this same log (from_probe), so the TPM numbers check the
bucket + charging + overwrite mechanics, not the estimator's accuracy on unseen prompts.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from relay.limiter import TOKENS, Limiter, Prompt  # noqa: E402
from relay.llm import CallRecord  # noqa: E402

RAW = ROOT / "data" / "probe" / "probe_raw.jsonl"
OUT = ROOT / "data" / "probe" / "limiter_replay.json"


class Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


def record(d: dict) -> CallRecord:
    r = CallRecord(**{k: v for k, v in d.items() if k in CallRecord.__dataclass_fields__ and k != "chunks"})
    return r


def pct(xs, q):
    xs = sorted(xs)
    if not xs:
        return None
    import math
    return xs[max(0, math.ceil(q * len(xs)) - 1)]


def run(use_headers: bool) -> dict:
    clock = Clock()
    lim = Limiter.from_probe(clock=clock)
    if not use_headers:
        lim._apply_headers = lambda *a, **k: None          # baseline: our model of the buckets only
    calls = [json.loads(x) for x in RAW.read_text(encoding="utf-8").splitlines() if x.strip()]
    calls = [c for c in calls if c.get("type") == "call" and c.get("mode") == "real"]
    timeline = []
    for i, c in enumerate(calls):
        t0 = c["t_start_unix"]
        timeline.append((t0, 0, i))
        timeline.append((t0 + (c["record"]["total_ms"] or 0) / 1000, 1, i))
    timeline.sort()
    grants: dict[int, object] = {}
    errs = {"TPM": [], "RPD": []}
    stale = {"TPM": 0, "RPD": 0}
    seen: set[tuple[str, str]] = set()
    rpm_rows = []
    for t, kind, i in timeline:
        clock.t = t
        c = calls[i]
        m = c["model"]
        st = lim.models[m]
        if kind == 0:
            msgs = c["payload"]["messages"]
            p = lim.estimate(m, msgs)
            lv_rpm = st.buckets["RPM"].level(st.events, t) if "RPM" in st.buckets else None
            g = lim._grant(st, p, p.tokens, c["payload"]["max_tokens"], t, 0.0)   # replay: never refuse
            grants[i] = g
            if c["phase"] == "P3" and lv_rpm is not None:
                rpm_rows.append({"model": m, "status": c["record"]["status"], "rpm_level_before": round(lv_rpm, 3)})
            continue
        g = grants[i]
        rec = record(c["record"])
        if rec.headers_ms is not None:
            t_hdr = c["t_start_unix"] + rec.headers_ms / 1000
            qt = (rec.usage or {}).get("queue_time")
            t_adm = max(g.t_granted, t_hdr - qt if rec.status == 200 and isinstance(qt, (int, float)) else t_hdr)
            pt = (rec.usage or {}).get("prompt_tokens")
            for name, hdr in (("TPM", "x-ratelimit-remaining-tokens"), ("RPD", "x-ratelimit-remaining-requests")):
                v = rec.header(hdr)
                b = st.buckets.get(name)
                if v is None or b is None:
                    continue
                if (m, name) not in seen:
                    seen.add((m, name))
                    continue
                if t_adm < b.anchor_t:
                    stale[name] += 1
                    continue
                pred = b.level(st.events, t_adm)
                if name == "TPM" and pt is not None:
                    pred += g.prompt_tokens - pt        # Groq charged the real prompt, we held our estimate
                if name == "RPD" and rec.status == 429:
                    pred += 1                           # a 429 isn't counted (Q5)
                errs[name].append({"model": m, "phase": c["phase"], "err": pred - int(v)})
        lim.settle(g, rec, t_sent=c["t_start_unix"])
    out = {}
    for name, rows in errs.items():
        for m in sorted({r["model"] for r in rows}):
            e = [abs(r["err"]) for r in rows if r["model"] == m]
            signed = [r["err"] for r in rows if r["model"] == m]
            out.setdefault(name, {})[m] = {"n": len(e), "abs_err_p50": round(pct(e, 0.5), 1),
                                           "abs_err_p90": round(pct(e, 0.9), 1), "abs_err_max": round(max(e), 1),
                                           "over_predicted_by_more_than_100": sum(1 for x in signed if x > 100)}
    rpm = {}
    for m in sorted({r["model"] for r in rpm_rows}):
        rows = [r for r in rpm_rows if r["model"] == m]
        rpm[m] = {"p3_calls": len(rows),
                  "groq_429_and_limiter_said_no_room": sum(1 for r in rows if r["status"] == 429 and r["rpm_level_before"] < 1),
                  "groq_429_but_limiter_said_room": sum(1 for r in rows if r["status"] == 429 and r["rpm_level_before"] >= 1),
                  "groq_200_but_limiter_said_no_room": sum(1 for r in rows if r["status"] == 200 and r["rpm_level_before"] < 1)}
    return {"prediction_error": out, "stale_headers_ignored": stale, "rpm_in_p3_burst": rpm,
            "hold_cap_after_replay": {m: st.hold_cap for m, st in lim.models.items()},
            "findings_changed": lim.findings_changed}


def main() -> int:
    res = {"_generated_by": "scripts/replay_limiter.py", "_raw": str(RAW.relative_to(ROOT)),
           "with_headers": run(True), "model_only": run(False)}
    OUT.write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {OUT}")
    for name in ("TPM", "RPD"):
        print(f"\n{name}: |predicted - Groq header| p50 / p90 / max (n)")
        for m, a in res["with_headers"]["prediction_error"].get(name, {}).items():
            b = res["model_only"]["prediction_error"][name][m]
            print(f"  {m:<20} with headers {a['abs_err_p50']:>7} / {a['abs_err_p90']:>7} / {a['abs_err_max']:>7} (n={a['n']})"
                  f"   model only {b['abs_err_p50']:>7} / {b['abs_err_p90']:>7} / {b['abs_err_max']:>7}")
    print("\nheaders ignored as out of order (with headers):", res["with_headers"]["stale_headers_ignored"])
    print("hold cap after replay:", res["with_headers"]["hold_cap_after_replay"])
    print("\nP3 burst, RPM bucket when each call was sent:")
    for m, r in res["with_headers"]["rpm_in_p3_burst"].items():
        print(f"  {m:<20} {r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
