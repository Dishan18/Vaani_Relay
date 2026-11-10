"""Budget plan for the next 24 h per model, from the ledger and measured costs. No model calls.

    .venv\\Scripts\\python scripts\\budget_plan.py   ->  results/quality/budget_plan.{md,json}

Two accountings, side by side:
  groq    what Groq is measured to charge: real usage (findings Q2), against daily buckets that
          refill continuously at capacity/86400 per second (measured for requests per day;
          ASSUMED the same for tokens per day, which no header shows). Capacity here = our 90%
          share of the documented daily limit.
  ledger  what relay/ledger.py enforces today: charged_tokens = max(prompt estimate + max_tokens,
          usage), summed over a rolling 24 h window, so each spend comes back all at once 24 h later.
Job costs come from measured tokens per request (part 1 of the retrieval sweep, per model), scaled
to other settings with the estimate's prompt sizes (results/quality/estimate.json). Load-run sizes
are assumptions, stated below.
"""
from __future__ import annotations

import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

LEDGER = ROOT / "results" / "ledger" / "groq_calls.jsonl"
ANSWERS = ROOT / "results" / "raw" / "answers"
EST = ROOT / "results" / "quality" / "estimate.json"
DOC = ROOT / "data" / "limits_documented.json"
OUT = ROOT / "results" / "quality" / "budget_plan"
FRACTION = 0.9                       # config/budget.toml account cap
M120, M20, MQ = "openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b"
# Load runs (assumptions): production config (router qwen, drafter 120b first, verifier on when it
# has headroom), the chosen setting.
LOAD = {"real_steady": {"requests": 30, "how": "2 requests/min for 15 min"},
        "real_burst": {"requests": 20, "how": "20 requests within 10 s (inside RPM 30)"}}


def load_jsonl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def measured_costs() -> dict:
    """Per (setting, step, model): mean real tokens per call and calls per request, from part-1 answers."""
    per = defaultdict(list)
    reqs = defaultdict(int)
    for p in sorted(ANSWERS.glob("*.jsonl")):
        if p.name.endswith(".calls.jsonl"):
            continue
        for o in load_jsonl(p):
            if o.get("type") != "answer" or o.get("source") != "real":
                continue
            reqs[o["setting"]] += 1
            for c in o["reply"].get("model_calls") or []:
                per[(o["setting"], c["step"], c["model"])].append(c["prompt_tokens"] + c["completion_tokens"])
    out = {}
    for (s, step, m), xs in per.items():
        out.setdefault(s, {})[f"{step}:{m}"] = {"calls_per_request": round(len(xs) / reqs[s], 3),
                                                 "tokens_per_call": round(statistics.mean(xs), 1),
                                                 "tokens_per_request": round(sum(xs) / reqs[s], 1)}
    return {"by_setting": out, "requests": dict(reqs)}


def bucket_level(events: list[tuple[float, float]], cap: float, period: float, now: float) -> float:
    """Level of a refilling bucket (full at the first event's window start) after these spends."""
    rate = cap / period
    level, t = cap, (events[0][0] if events else now)
    for ts, amount in sorted(events):
        level = min(cap, level + (ts - t) * rate) - amount
        t = ts
    return min(cap, level + (now - t) * rate)


def main() -> int:
    now = time.time()
    doc = json.loads(DOC.read_text(encoding="utf-8"))["models"]
    led = [e for e in load_jsonl(LEDGER) if e.get("source") == "real"]
    est = json.loads(EST.read_text(encoding="utf-8"))
    mc = measured_costs()
    models = [M120, M20, MQ]
    state = {}
    for m in models:
        cap_t, cap_r = FRACTION * doc[m]["tpd"], FRACTION * doc[m]["rpd"]
        ev = [e for e in led if e["model"] == m]
        actual = [(e["ts"], float(e.get("usage_total_tokens") or 0)) for e in ev]
        reqs = [(e["ts"], 1.0) for e in ev if e.get("status") != 429]
        lv_t = bucket_level(actual, cap_t, 86400.0, now)
        lv_r = bucket_level(reqs, cap_r, 86400.0, now)
        win = [e for e in ev if e["ts"] > now - 86400]
        charged = sum(e["charged_tokens"] for e in win)
        # ledger: when does each spend leave the window
        state[m] = {"groq_tokens_available_now": round(lv_t), "groq_requests_available_now": round(lv_r),
                    "groq_tokens_refill_per_h": round(cap_t / 24), "groq_full_again_in_h": round((cap_t - lv_t) / (cap_t / 86400) / 3600, 1),
                    "groq_actual_used_24h": round(sum(a for ts, a in actual if ts > now - 86400)),
                    "ledger_charged_24h": charged, "ledger_tokens_available_now": round(cap_t - charged),
                    "ledger_last_spend_returns_at": (time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime(max(e["ts"] for e in win) + 86400))
                                                     if win else None),
                    "cap_tokens": cap_t, "cap_requests": cap_r}
    # ---- job costs (actual tokens), from measured per-request costs
    big = mc["by_setting"]
    ref = "large" if "large" in big else next(iter(big))
    ref_prompt = est["settings"][ref]["draft_prompt_mean"][M120]

    def draft_cost(setting, model):
        """Measured draft tokens per request at the reference setting, scaled by prompt size."""
        d = big[ref].get(f"draft:{M120}", {"tokens_per_request": 0})["tokens_per_request"]
        p = est["settings"][setting]["draft_prompt_mean"].get(model, ref_prompt)
        return d * (p + 300) / (ref_prompt + 300)
    router = big[ref].get(f"understand:{MQ}", {"tokens_per_request": 0})["tokens_per_request"]
    jobs = []
    done_small = mc["requests"].get("small", 0)
    if done_small < 18:
        n = 18 - done_small
        jobs.append({"job": f"sweep part 1 remainder: small ({n} questions)", "requests": {M120: n, MQ: n},
                     "tokens": {M120: draft_cost("small", M120) * n, MQ: router * n}, "priority": 0})
    for s in ("medium+rerank", "small_lean"):
        cost = {M120: draft_cost(s, M120) * 18}
        if est["settings"][s]["router"]:
            cost[MQ] = router * 18
        jobs.append({"job": f"sweep part 2: {s}", "requests": {M120: 18, **({MQ: 18} if MQ in cost else {})},
                     "tokens": cost, "priority": 1})
    jobs.append({"job": "drafter step: winner with 20b and qwen", "requests": {M20: 18, MQ: 18 + 36},
                 "tokens": {M20: draft_cost(ref, M20) * 18, MQ: draft_cost(ref, MQ) * 18 + router * 36}, "priority": 2})
    per_req_draft = draft_cost(ref, M120)
    verify = per_req_draft * 0.8          # (assumption) the verifier reads the sources plus the answer
    for name, spec in LOAD.items():
        n = spec["requests"]
        jobs.append({"job": f"{name} ({spec['how']})", "requests": {M120: n, MQ: 2 * n},
                     "tokens": {M120: per_req_draft * n, MQ: (router + verify) * n}, "priority": 3})
    # ---- schedule: each job in priority order, at the earliest time every model it uses has room
    # (groq: bucket level; ledger: rolling-window headroom at 2.0x charge), never before the previous
    # job ends. A job lasts at least its tokens / TPM (8000 per model per minute), and 5 min.
    TPM, LEDGER_FACTOR = 8000.0, 2.0
    g_events = {m: [(e["ts"], float(e.get("usage_total_tokens") or 0)) for e in led if e["model"] == m] for m in models}
    l_events = {m: [(e["ts"], float(e["charged_tokens"])) for e in led if e["model"] == m] for m in models}
    horizon = now + 86400

    def groq_room(m, t):
        return bucket_level(g_events[m], state[m]["cap_tokens"], 86400.0, t)

    def ledger_room(m, t):
        return state[m]["cap_tokens"] - sum(x for ts, x in l_events[m] if t - 86400 < ts <= t)

    def earliest(room, need, start):
        t = start
        while t <= horizon:
            if all(room(m, t) >= need[m] for m in need):
                return t
            t += 300                                       # 5-minute steps
        return None
    t_g = t_l = now
    for j in sorted(jobs, key=lambda x: x["priority"]):
        dur = max(300.0, max(v / TPM * 60 for v in j["tokens"].values()))
        sg = earliest(groq_room, j["tokens"], t_g)
        sl = earliest(ledger_room, {m: v * LEDGER_FACTOR for m, v in j["tokens"].items()}, t_l)
        fmt = lambda t: time.strftime("%m-%d %H:%M UTC", time.gmtime(t)) if t else "not within 24 h"
        j["start_groq"], j["start_ledger"] = fmt(sg), fmt(sl)
        j["fits_groq"], j["fits_ledger_now"] = sg is not None, sl is not None and sl <= now + 600
        if sg is not None:
            for m, v in j["tokens"].items():
                g_events[m].append((sg + dur / 2, v))
            t_g = sg + dur
        if sl is not None:
            for m, v in j["tokens"].items():
                l_events[m].append((sl + dur / 2, v * LEDGER_FACTOR))
            t_l = sl + dur
        j["tokens"] = {m: round(t) for m, t in j["tokens"].items()}
    jobs.sort(key=lambda x: x["priority"])
    out = {"_generated_by": "scripts/budget_plan.py", "now": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime(now)),
           "accounting": {"groq": "real usage vs refilling daily bucket (tokens: assumed)", "ledger": "max(est+max_tokens, usage), rolling 24 h"},
           "ledger_overcharge_note": "jobs' ledger cost assumed 2.0x real usage (measured ratio for 120b drafts in part 1)",
           "measured_costs": mc, "models": state, "jobs": jobs, "load_assumptions": LOAD}
    OUT.with_suffix(".json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n")
    L = ["# Budget plan, next 24 h", "", f"Generated by `scripts/budget_plan.py` at {out['now']}. "
         "Groq accounting = real usage against a daily bucket refilling at cap/24 per hour "
         "(measured for daily requests, assumed for daily tokens); cap = 90% of documented. "
         "Ledger = what relay/ledger.py enforces today (max(estimate + max_tokens, usage), rolling 24 h).", "",
         "| model | real tokens used (24 h) | available now (Groq) | refill per hour | full again in | "
         "ledger charged (24 h) | available now (ledger) | ledger's last spend returns |",
         "|---|---|---|---|---|---|---|---|"]
    for m in models:
        s = state[m]
        L.append(f"| {m} | {s['groq_actual_used_24h']} | {s['groq_tokens_available_now']} | {s['groq_tokens_refill_per_h']} | "
                 f"{s['groq_full_again_in_h']} h | {s['ledger_charged_24h']} | {s['ledger_tokens_available_now']} | "
                 f"{s['ledger_last_spend_returns_at']} |")
    L += ["", "Jobs in priority order, each at the earliest time every model it uses has room (and not before the "
          "previous job ends). Ledger column: charges assumed 2.0x real usage, as measured for 120b drafts.", "",
          "| job | tokens per model (real usage) | earliest start, Groq accounting | earliest start, today's ledger |",
          "|---|---|---|---|"]
    for j in jobs:
        L.append(f"| {j['job']} | " + ", ".join(f"{m.split('/')[-1]} {t}" for m, t in j["tokens"].items())
                 + f" | {j['start_groq']} | {j['start_ledger']} |")
    OUT.with_suffix(".md").write_text("\n".join(L) + "\n", encoding="utf-8", newline="\n")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    sys.exit(main())
