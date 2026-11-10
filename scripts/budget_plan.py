"""Budget plan for the next 24 h per model, on the ledger's own accounting. No model calls.

    .venv\\Scripts\\python scripts\\budget_plan.py   ->  results/quality/budget_plan.{md,json}

Accounting = relay/ledger.py as it runs: each call counts the usage Groq reported (relay.ledger.
charge_of; calls with no usage keep their conservative charge), summed over a rolling 24 h window
against 90% of the documented daily limit. No refill is assumed: Groq never shows a daily-token
bucket, so a spend counts until exactly 24 h after it was made (the author's rule: the conservative
choice on a shared key). Each job must also leave the eval runner's stop-rule margin (7400 tokens,
a draft and a redraft at max_tokens) free on its drafter.
Job costs come from measured tokens per request in the real runs so far (per model and step),
scaled to other settings by the estimate's prompt sizes (results/quality/estimate.json).
Load-run sizes are assumptions, stated below.
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

from relay.ledger import charge_of  # noqa: E402

LEDGER = ROOT / "results" / "ledger" / "groq_calls.jsonl"
ANSWERS = ROOT / "results" / "raw" / "answers"
EST = ROOT / "results" / "quality" / "estimate.json"
DOC = ROOT / "data" / "limits_documented.json"
OUT = ROOT / "results" / "quality" / "budget_plan"
FRACTION = 0.9                       # config/budget.toml account cap
STOP_MARGIN = 7400                   # scripts/eval_retrieval.py stop rule (2 x (2500 + 1200))
TPM = 8000.0                         # measured, per model per minute: bounds how fast a job can spend
M120, M20, MQ = "openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b"
MODELS = [M120, M20, MQ]
# Load runs (assumptions): production config (router qwen, drafter 120b first, verifier on when it
# has headroom; the verifier reads the sources plus the answer, assumed 0.8 x a draft), setting large.
LOAD = {"real_steady": {"requests": 30, "how": "2 requests/min for 15 min"},
        "real_burst": {"requests": 20, "how": "20 requests within 10 s (inside RPM 30)"}}
VERIFY_FACTOR = 0.8


def load_jsonl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def measured() -> tuple[dict, dict]:
    """(setting -> 'step:model' -> tokens per request, setting -> requests answered) from real answers
    (latest per question)."""
    latest = {}
    for p in sorted(ANSWERS.glob("*.jsonl")):
        if p.name.endswith(".calls.jsonl"):
            continue
        for o in load_jsonl(p):
            if o.get("type") == "answer" and o.get("source") == "real" and o["scenario"] == "eval:retrieval":
                latest[(o["setting"], o["qid"])] = o
    per, n = defaultdict(lambda: defaultdict(float)), defaultdict(int)
    for (s, _), o in latest.items():
        n[s] += 1
        for c in o["reply"].get("model_calls") or []:
            per[s][f"{c['step']}:{c['model']}"] += c["prompt_tokens"] + c["completion_tokens"]
    return {s: {k: v / n[s] for k, v in d.items()} for s, d in per.items()}, dict(n)


def main() -> int:
    now = time.time()
    doc = json.loads(DOC.read_text(encoding="utf-8"))["models"]
    led = [e for e in load_jsonl(LEDGER) if e.get("source") == "real"]
    est = json.loads(EST.read_text(encoding="utf-8"))
    per, answered = measured()
    cap = {m: FRACTION * doc[m]["tpd"] for m in MODELS}
    events = {m: [(e["ts"], float(charge_of(e))) for e in led if e["model"] == m] for m in MODELS}

    def used(m, t):
        return sum(x for ts, x in events[m] if t - 86400 < ts <= t)

    def room(m, t):
        return cap[m] - used(m, t)

    ref = "large"
    ref_prompt = est["settings"][ref]["draft_prompt_mean"][M120]
    draft_ref = per[ref].get(f"draft:{M120}", 0.0)
    router = per[ref].get(f"understand:{MQ}", 0.0)

    def draft(setting, model):
        p = est["settings"][setting]["draft_prompt_mean"].get(model, ref_prompt)
        return draft_ref * (p + 300) / (ref_prompt + 300)

    jobs = [{"job": "reruns after the gate fix (large Q8, Q15; medium Q7, Q8, Q11)",
             "tokens": {M120: draft(ref, M120) * 5, MQ: router * 5}}]
    left_small = 18 - answered.get("small", 0)
    if left_small > 0:
        jobs.append({"job": f"finish small ({left_small} questions)",
                     "tokens": {M120: draft("small", M120) * left_small, MQ: router * left_small}})
    jobs.append({"job": "drafter step at the winner (20b, qwen; 120b reused)",
                 "tokens": {M20: draft(ref, M20) * 18, MQ: draft(ref, MQ) * 18 + router * 36}})
    jobs.append({"job": "sweep part 2: medium+rerank", "tokens": {M120: draft("medium+rerank", M120) * 18, MQ: router * 18}})
    jobs.append({"job": "sweep part 2: small_lean", "tokens": {M120: draft("small_lean", M120) * 18}})
    for name, spec in LOAD.items():
        k = spec["requests"]
        jobs.append({"job": f"{name} ({spec['how']})",
                     "tokens": {M120: draft(ref, M120) * k, MQ: (router + VERIFY_FACTOR * draft(ref, M120)) * k}})

    horizon = now + 86400
    t = now
    for j in jobs:
        need = {m: v + (STOP_MARGIN if m != MQ or len(j["tokens"]) == 1 else 0) for m, v in j["tokens"].items()}
        dur = max(300.0, max(v / TPM * 60 for v in j["tokens"].values()))
        start = None
        s = t
        while s <= horizon:
            if all(room(m, s) >= need[m] for m in need):
                start = s
                break
            s += 300
        j["start"] = time.strftime("%m-%d %H:%M UTC", time.gmtime(start)) if start else "not within 24 h"
        j["fits_now"] = start is not None and start <= now + 600
        if start is not None:
            for m, v in j["tokens"].items():
                events[m].append((start + dur / 2, v))
            t = start + dur
        j["tokens"] = {m: round(v) for m, v in j["tokens"].items()}

    state = {}
    for m in MODELS:
        win = sorted((ts, x) for ts, x in ((e["ts"], float(charge_of(e))) for e in led if e["model"] == m)
                     if ts > now - 86400)
        state[m] = {"used_24h": round(sum(x for _, x in win)), "room_now": round(cap[m] - sum(x for _, x in win)),
                    "cap": cap[m],
                    "first_spend_returns": time.strftime("%m-%d %H:%M UTC", time.gmtime(win[0][0] + 86400)) if win else None,
                    "all_back": time.strftime("%m-%d %H:%M UTC", time.gmtime(win[-1][0] + 86400)) if win else "now"}
        # when this model has room again for each load run (on today's spend only)
    load_need = {name: {M120: draft(ref, M120) * s["requests"] + STOP_MARGIN} for name, s in LOAD.items()}
    back = {}
    for name, need in load_need.items():
        s = now
        while s <= now + 2 * 86400 and room(M120, s) < need[M120]:
            s += 300
        back[name] = time.strftime("%m-%d %H:%M UTC", time.gmtime(s))
    out = {"_generated_by": "scripts/budget_plan.py", "now": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime(now)),
           "accounting": "ledger: reported usage, rolling 24 h, no refill assumed", "measured_tokens_per_request": per,
           "answered": answered, "models": state, "jobs": jobs, "load_assumptions": LOAD,
           "load_run_120b_room_after_all_jobs": back}
    OUT.with_suffix(".json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n")
    L = ["# Budget plan, next 24 h", "", f"Generated by `scripts/budget_plan.py` at {out['now']}. Accounting = the "
         "ledger as it runs: reported usage per call, summed over a rolling 24 h window against 90% of the "
         "documented daily limit; a spend returns exactly 24 h after it was made (no refill assumed: Groq shows no "
         "daily-token bucket). Jobs keep the stop rule's 7400-token margin free on their drafter.", "",
         "| model | used, last 24 h | room now | first spend returns | everything back |", "|---|---|---|---|---|"]
    for m in MODELS:
        s = state[m]
        L.append(f"| {m} | {s['used_24h']} | {s['room_now']} | {s['first_spend_returns']} | {s['all_back']} |")
    L += ["", "Jobs in the order they will run, each at the earliest time its models have room:", "",
          "| job | tokens per model | earliest start |", "|---|---|---|"]
    for j in jobs:
        L.append(f"| {j['job']} | " + ", ".join(f"{m.split('/')[-1]} {v}" for m, v in j["tokens"].items())
                 + f" | {j['start']} |")
    L += ["", "Load-run sizes are assumptions (real_steady: 30 requests at 2/min; real_burst: 20 within 10 s), with "
          "120b drafting and the verifier on. Their 120b cost scales with the drafter decision."]
    OUT.with_suffix(".md").write_text("\n".join(L) + "\n", encoding="utf-8", newline="\n")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    sys.exit(main())
