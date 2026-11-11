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
LOAD = {"real_steady": {"requests": 30, "how": "2 requests/min for 15 min", "like": "steady"},
        "real_burst": {"requests": 20, "how": "20 requests within 10 s (inside RPM 30)", "like": "burst"}}
RUNS = ROOT / "results" / "raw" / "runs"
# Eval jobs still to run before the load runs. small_lean Q12-Q18 and the Q14 rerun ran on 2026-11-11 07:24 UTC
# (key1, account cap 100%: docs/findings_and_decisions.md), so none are left.
SCHEDULED: list[dict] = []


def load_run_cost(scenario: str) -> tuple[dict, int, str]:
    """Tokens per request per model, every attempt, from the latest stand-in run of ``scenario`` (the current
    pipeline: router, drafter fallback, verifier with headroom, degrade ladder). The stand-in counts prompt
    tokens with the fit to real usage, so these approximate what the same requests would cost on Groq."""
    files = sorted(RUNS.glob(f"{scenario}-*.jsonl"))
    if not files:
        return {}, 0, ""
    rows = [o for o in load_jsonl(files[-1]) if o.get("type") == "answer"]
    tot: dict[str, float] = {}
    for o in rows:
        for c in o["reply"].get("model_calls") or []:
            tot[c["model"]] = tot.get(c["model"], 0.0) + (c.get("prompt_tokens") or 0) + (c.get("completion_tokens") or 0)
    return {m: v / len(rows) for m, v in tot.items()}, len(rows), files[-1].name


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


def main(argv=None) -> int:
    import argparse
    from relay.ledger import LEGACY_ACCOUNT
    ap = argparse.ArgumentParser()
    ap.add_argument("--account", default=None, help="ledger account to plan for (default: the key in .env)")
    a = ap.parse_args(argv)
    if a.account is None:
        from relay.real import account_label
        a.account = account_label()
    now = time.time()
    doc = json.loads(DOC.read_text(encoding="utf-8"))["models"]
    led = [e for e in load_jsonl(LEDGER) if e.get("source") == "real"
           and e.get("account", LEGACY_ACCOUNT) == a.account]
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

    # only what is still to do: each sweep setting until all 18 questions are answered (latest per question)
    jobs = []
    pending = {"small_lean"} if any("small_lean" in j["setting"] for j in SCHEDULED) else set()
    for j in SCHEDULED:
        cost = {M120: draft(j["setting"], M120) * j["n"]}
        if est["settings"][j["setting"]]["router"]:
            cost[MQ] = router * j["n"]
        jobs.append({"job": j["job"], "tokens": cost})
    for setting in ("large", "medium", "small", "medium+rerank", "small_lean"):
        if setting in pending:
            continue
        left = 18 - answered.get(setting, 0)
        if left > 0:
            cost = {M120: draft(setting, M120) * left}
            if est["settings"][setting]["router"]:
                cost[MQ] = router * left
            jobs.append({"job": f"sweep: {setting}, {left} questions left", "tokens": cost})
    load_basis = {}
    for name, spec in LOAD.items():
        k = spec["requests"]
        per, n_rows, src = load_run_cost(spec["like"])
        load_basis[name] = {"stand_in_run": src, "stand_in_requests": n_rows,
                            "tokens_per_request": {m: round(v) for m, v in per.items()}}
        jobs.append({"job": f"{name} ({spec['how']})", "tokens": {m: v * k for m, v in per.items()}})

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
    load_need = {name: {M120: load_run_cost(s["like"])[0].get(M120, 0.0) * s["requests"] + STOP_MARGIN}
                 for name, s in LOAD.items()}
    back = {}
    for name, need in load_need.items():
        s = now
        while s <= now + 2 * 86400 and room(M120, s) < need[M120]:
            s += 300
        back[name] = time.strftime("%m-%d %H:%M UTC", time.gmtime(s))
    out = {"_generated_by": "scripts/budget_plan.py", "now": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime(now)),
           "accounting": "ledger: reported usage, rolling 24 h, no refill assumed", "measured_tokens_per_request": per,
           "answered": answered, "models": state, "jobs": jobs, "load_assumptions": LOAD,
           "load_run_120b_room_after_all_jobs": back, "load_cost_basis": load_basis}
    OUT.with_suffix(".json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n")
    L = ["# Budget plan, next 24 h", "", f"Generated by `scripts/budget_plan.py` at {out['now']} for ledger account "
         f"{a.account} (one key: the brief allows one budget). Accounting = the "
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
    L += ["", "Load-run sizes are assumptions (real_steady: 30 requests at 2/min; real_burst: 20 within 10 s). Their "
          "tokens per request per model are measured on the stand-in with the current pipeline:"]
    for name, b in load_basis.items():
        L.append(f"- {name}: like `{b['stand_in_run']}` ({b['stand_in_requests']} requests): "
                 + ", ".join(f"{m.split('/')[-1]} {v}" for m, v in b["tokens_per_request"].items()) + " tokens per request")
    OUT.with_suffix(".md").write_text("\n".join(L) + "\n", encoding="utf-8", newline="\n")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    sys.exit(main())
