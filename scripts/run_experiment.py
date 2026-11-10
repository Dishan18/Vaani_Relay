"""One iterate round on the stand-in: a baseline and candidate arms, same seeds and arrivals, one change.

    .venv\\Scripts\\python scripts\\run_experiment.py --name r1_token_margin \\
        --arm baseline: --arm margin_560:limiter.token_margin=560 \\
        [--seeds 7,8] [--scenarios steady,burst,overload,chaos] [--duration-s 180]
        ->  data/experiments/<name>/{result.md, result.json, rows/, detail.jsonl}
            and one line in docs/findings_and_decisions.md ("Iterate rounds")

Every arm runs every scenario with every seed (identical arrivals, questions and patience per seed),
side by side, each on its own stand-in. No Groq calls. The first arm is the baseline.
Decision rule (fixed before running):
  primary   people without a written answer = passages-only + gave up + error, summed over the
            scenarios, per seed
  better    a candidate is better when, on every seed, its primary count is at least 2 lower than the
            baseline's (more than one request of noise); or, when it is within 1 on every seed, its
            TTFT p95 (released replies, all scenarios) is at least 10% lower on every seed
  keep      the best candidate that is better than the baseline; otherwise keep the baseline
The slowest request of each arm (first seed) is explained with scripts/explain_request.py.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from relay.pipeline import load_config  # noqa: E402
from relay.retrieve import Retriever  # noqa: E402
from scripts import explain_request, run_load  # noqa: E402
from standin.server import Replay, load_calibration  # noqa: E402

EXP = ROOT / "data" / "experiments"
LOG = ROOT / "docs" / "findings_and_decisions.md"


def parse_arm(text: str) -> tuple[str, dict]:
    name, _, rest = text.partition(":")
    over = {}
    for kv in filter(None, rest.split(",")):
        k, v = kv.split("=", 1)
        try:
            v = json.loads(v)
        except json.JSONDecodeError:
            pass
        over[k] = v
    return name, over


def pct(xs, q):
    xs = sorted(xs)
    return xs[max(0, math.ceil(q * len(xs)) - 1)] if xs else None


def fmt(x) -> str:
    return "-" if x is None else f"{x:.0f}"


def metrics(rows: list[dict]) -> dict:
    n = len(rows)
    no_written = sum(1 for d in rows if d["gave_up"] or d["kind"] in ("passages", "error", "busy"))
    ttft = [d["released_ms"] for d in rows if d.get("released_ms") is not None and not d["gave_up"]]
    calls = [c for d in rows for c in d.get("model_calls") or []]
    return {"requests": n, "no_written_answer": no_written, "gave_up": sum(1 for d in rows if d["gave_up"]),
            "passages": sum(1 for d in rows if d["kind"] == "passages"),
            "written_pct": round(100 * (n - no_written) / n, 1) if n else None,
            "ttft_p50_ms": pct(ttft, .5), "ttft_p95_ms": pct(ttft, .95),
            "calls": len(calls), "failed_calls": sum(1 for c in calls if c["status"] != 200 or c["outcome"] != "ok"),
            "http_429": sum(1 for c in calls if c["status"] == 429)}


async def run(a) -> dict:
    arms = [parse_arm(x) for x in a.arm]
    seeds = [int(x) for x in a.seeds.split(",")]
    scen = a.scenarios.split(",")
    out = EXP / a.name
    (out / "rows").mkdir(parents=True, exist_ok=True)
    with (ROOT / "data" / "questions.jsonl").open(encoding="utf-8") as f:
        questions = [json.loads(x) for x in f if x.strip()]
    retriever = Retriever.from_blocks(load_config()["retrieval"]["piece_chars"])
    cal = load_calibration()
    replay = Replay(ROOT / "results" / "raw" / "answers")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    jobs = []
    for arm, over in arms:
        for seed in seeds:
            for sc in scen:
                path = out / "rows" / f"{arm}-seed{seed}-{sc}.jsonl"
                jobs.append((arm, seed, sc, run_load.run_scenario(
                    run_load.SCENARIOS[sc], a.duration_s, seed, cal, questions, retriever, stamp, replay, path,
                    out / "detail.jsonl", overrides=over, label=f"{arm}-seed{seed}")))
    results = await asyncio.gather(*(j[3] for j in jobs))
    table: dict = {}
    for (arm, seed, sc, _), r in zip(jobs, results):
        table.setdefault(arm, {}).setdefault(seed, []).append(r)
    summary = {}
    for arm, by_seed in table.items():
        summary[arm] = {}
        for seed, rs in by_seed.items():
            rows = [d for r in rs for d in r["detail_rows"]]
            m = metrics(rows)
            m["by_scenario"] = {r["scenario"]: metrics(r["detail_rows"]) for r in rs}
            m["stand_in_429"] = sum(r["stand_in"]["429"] for r in rs)
            summary[arm][seed] = m
    base = arms[0][0]
    verdicts = {}
    for arm, _ in arms[1:]:
        lower = all(summary[arm][s]["no_written_answer"] <= summary[base][s]["no_written_answer"] - 2 for s in seeds)
        tie = all(abs(summary[arm][s]["no_written_answer"] - summary[base][s]["no_written_answer"]) <= 1 for s in seeds)
        faster = all((summary[arm][s]["ttft_p95_ms"] or 1e18) <= 0.9 * (summary[base][s]["ttft_p95_ms"] or 1e18)
                     for s in seeds)
        verdicts[arm] = "better (fewer people without a written answer)" if lower else (
            "better (same service, TTFT p95 at least 10% lower)" if tie and faster else "not better")
    better = [arm for arm, v in verdicts.items() if v.startswith("better")]
    keep = min(better, key=lambda arm: sum(summary[arm][s]["no_written_answer"] for s in seeds)) if better else base
    slow = {}
    det = explain_request.load(out / "detail.jsonl")
    for arm, _ in arms:
        rows = [d for d in det if d["run_id"].endswith(f"-{arm}-seed{seeds[0]}")]
        if rows:
            d = max(rows, key=lambda d: d.get("released_ms") if d.get("released_ms") is not None else d.get("waited_ms") or 0)
            slow[arm] = {"request_id": d["request_id"], **{k: v for k, v in explain_request.split(d).items() if k != "timeline"}}
    res = {"_generated_by": "scripts/run_experiment.py", "name": a.name, "stamp": stamp, "arms": dict(arms),
           "seeds": seeds, "scenarios": scen, "duration_s": a.duration_s, "summary": summary,
           "verdicts": verdicts, "keep": keep, "slowest": slow,
           "rule": "better = no-written-answer count at least 2 lower on every seed, or within 1 on every seed and "
                   "TTFT p95 at least 10% lower on every seed"}
    (out / "result.json").write_text(json.dumps(res, indent=1, default=str) + "\n", encoding="utf-8", newline="\n")
    L = [f"# Iterate round {a.name}", "", f"Generated by `scripts/run_experiment.py` ({stamp}); stand-in only. "
         f"Scenarios {', '.join(scen)}, {a.duration_s:.0f} s each, seeds {', '.join(map(str, seeds))}. "
         f"Rule: {res['rule']}.", "",
         "| arm | change | seed | requests | no written answer | gave up | passages | written % | TTFT p50 / p95 (ms) | 429s | failed calls |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for arm, over in arms:
        for s in seeds:
            m = summary[arm][s]
            L.append(f"| {arm} | {json.dumps(over) if over else '(baseline)'} | {s} | {m['requests']} | {m['no_written_answer']} | "
                     f"{m['gave_up']} | {m['passages']} | {m['written_pct']} | {fmt(m['ttft_p50_ms'])} / {fmt(m['ttft_p95_ms'])} | "
                     f"{m['http_429']} | {m['failed_calls']} |")
    L += ["", "| arm | verdict |", "|---|---|"] + [f"| {k} | {v} |" for k, v in verdicts.items()]
    L += ["", f"**Keep: {keep}.**", "", "Slowest request per arm (first seed), from scripts/explain_request.py:", ""]
    for arm, s in slow.items():
        sp = s["split_ms"]
        L.append(f"- {arm}: `{s['request_id']}` {s['end_ms']:.0f} ms = waiting {sp['waiting']:.0f} + working "
                 f"{sp['working']:.0f} + retrying {sp['retrying']:.0f} + local {sp['local']:.0f} ms; all before the first token")
    (out / "result.md").write_text("\n".join(L) + "\n", encoding="utf-8", newline="\n")
    with LOG.open("a", encoding="utf-8", newline="\n") as f:
        f.write(f"- **{a.name}** ({stamp}, stand-in, seeds {','.join(map(str, seeds))}): "
                + "; ".join(f"{k} {v}" for k, v in verdicts.items()) + f". Kept: {keep}. "
                f"`data/experiments/{a.name}/result.md`\n")
    print("\n".join(L))
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--arm", action="append", required=True, help="name:key=value,key=value (first = baseline)")
    ap.add_argument("--seeds", default="7,8")
    ap.add_argument("--scenarios", default="steady,burst,overload,chaos")
    ap.add_argument("--duration-s", type=float, default=180.0)
    a = ap.parse_args(argv)
    asyncio.run(run(a))
    return 0


if __name__ == "__main__":
    sys.exit(main())
