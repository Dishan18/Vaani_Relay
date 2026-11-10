"""Load runs against the stand-in only (no Groq calls, no ledger): R5-R9 on the stand-in.

    .venv\\Scripts\\python scripts\\run_load.py [--scenarios steady,burst,...] [--duration-s 300] [--seed 7]
        ->  results/raw/runs/<scenario>-<utc>.jsonl   one row per request (read by build_results.py)
            data/results/detail.jsonl               per-request step spans and every model call

There is no --real: real load runs (real_steady, real_burst) wait until 120b's budget is back.

Each scenario gets its own stand-in (standin/server.py, calibrated from the probe, replaying the
recorded real router and draft texts where the request matches), its own client-side limiter
(Limiter.from_probe) and the production pipeline (config/pipeline.toml: router, drafter fallback
order, verifier with headroom, degrade ladder). Scenarios run side by side, each on its own port.
Arrivals are open-loop (a request arrives when its time comes, whatever the backlog), so queueing
isn't hidden. Every simulated person has a patience the pipeline can't see; when it runs out they
leave: the request is cancelled (the limiter keeps what admission charged) and recorded as
status timeout, gave_up 1.

Scenarios (sizes and rates are assumptions, chosen against the measured limits: per model 30 RPM
and 8000 TPM, so about 3-4 drafts a minute per drafter):
  steady                Poisson 3/min
  burst                 1/min background + 25 arrivals within 5 s at t = 20 s
  overload              Poisson 15/min (well above what three drafters can serve)
  overload_naive_retry  the same arrivals, questions and patience as overload, but no limiter and a
                        naive retry: up to 3 retries on the same model, 1 s apart, ignoring retry-after;
                        one router and one drafter model, no fallback
  rpm_halved            Poisson 6/min; the stand-in halves RPM and TPM; the client is not told
  daily_exhausted       Poisson 3/min; every model's daily budget is used up at the start
  chaos                 Poisson 4/min + 10 arrivals at mid-run; 10% 503s, 5% 8 s stalls, slow-down under load
Patience: lognormal, median 25 s, sigma 0.6 (p95 about 67 s), an assumption (PLAN.md G3).
Questions: drawn uniformly from the 18 in data/questions.jsonl.
"""
from __future__ import annotations

import argparse
import asyncio
import contextvars
import copy
import hashlib
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from relay.limiter import Grant, Limiter, Prompt  # noqa: E402
from relay.llm import Endpoint, LLMClient  # noqa: E402
from relay.pipeline import Pipeline, load_config  # noqa: E402
from relay.retrieve import Retriever  # noqa: E402
from standin.server import Replay, StandIn, Switches, load_calibration  # noqa: E402

RUNS = ROOT / "results" / "raw" / "runs"
DETAIL = ROOT / "data" / "results" / "detail.jsonl"
ANSWERS = ROOT / "results" / "raw" / "answers"
PATIENCE = {"median_s": 25.0, "sigma": 0.6}
NAIVE = {"retries": 3, "backoff_s": 1.0}


@dataclass
class Scenario:
    name: str
    rate_per_min: float
    switches: Switches
    burst: tuple[float, int, float] | None = None      # (at s, n, within s)
    exhaust_daily: bool = False
    naive: bool = False
    arrivals_from: str | None = None                   # reuse another scenario's arrivals


SCENARIOS = {
    "steady": Scenario("steady", 3.0, Switches()),
    "burst": Scenario("burst", 1.0, Switches(), burst=(20.0, 25, 5.0)),
    "overload": Scenario("overload", 15.0, Switches()),
    "overload_naive_retry": Scenario("overload_naive_retry", 15.0, Switches(), naive=True, arrivals_from="overload"),
    "rpm_halved": Scenario("rpm_halved", 6.0, Switches(limit_scale=0.5, limit_scale_only="RPM,TPM")),
    "daily_exhausted": Scenario("daily_exhausted", 3.0, Switches(), exhaust_daily=True),
    "chaos": Scenario("chaos", 4.0, Switches(p503=0.10, stall_rate=0.05, stall_s=8.0, slow_under_load=True),
                      burst=None),
}


def arrivals(sc: Scenario, duration_s: float, seed: int, questions: list[dict]) -> list[dict]:
    """[{t, qid, question, patience_s}] for one scenario (open loop), deterministic per seed."""
    name = sc.arrivals_from or sc.name
    rng = random.Random(f"{seed}:{name}")
    base = SCENARIOS[name]
    ts, t = [], 0.0
    rate = base.rate_per_min / 60.0
    while rate > 0:
        t += rng.expovariate(rate)
        if t >= duration_s:
            break
        ts.append(t)
    if base.burst:
        at, n, within = base.burst
        ts += [at + rng.uniform(0, within) for _ in range(n)]
    if name == "chaos":
        ts += [duration_s / 2 + rng.uniform(0, 3.0) for _ in range(10)]
    ts.sort()
    out = []
    for t in ts:
        q = rng.choice(questions)
        patience = math.exp(math.log(PATIENCE["median_s"]) + PATIENCE["sigma"] * rng.gauss(0, 1))
        out.append({"t": t, "qid": q["qid"], "question": q["question"], "patience_s": round(patience, 2)})
    return out


# ----------------------------------------------------------------------------- naive client
class NaiveLimiter:
    """No limiter: every reservation is granted at once. Only the estimator and settings are real."""

    def __init__(self, real: Limiter) -> None:
        self._real = real
        self.estimator = real.estimator
        self.settings = real.settings
        self._n = 0

    def now(self) -> float:
        return time.monotonic()

    def estimate(self, model: str, messages: list[dict]) -> Prompt:
        return self._real.estimate(model, messages)

    def reserve(self, model, prompt_tokens, max_tokens, deadline):
        self._n += 1
        pt = prompt_tokens.tokens if isinstance(prompt_tokens, Prompt) else int(prompt_tokens)
        return Grant(self, self._n, model, None, pt, max_tokens, self.now(), 0.0)

    def predict_wait(self, *a, **k) -> float:
        return 0.0

    def capacity(self, *a, **k):
        return None

    def levels(self, *a, **k) -> dict:
        return {}

    def settle(self, g, rec, t_sent=None) -> None:
        g.settled = True

    def release(self, g) -> None:
        g.settled = True

    def abandon(self, g) -> None:
        g.settled = True


def naive_retry(chat):
    async def wrapped(payload, *, phase, timeouts=None):
        rec = await chat(payload, phase=phase, timeouts=timeouts)
        for _ in range(NAIVE["retries"]):
            if rec.status not in (429, 500, 502, 503):
                break
            await asyncio.sleep(NAIVE["backoff_s"])
            rec = await chat(payload, phase=phase, timeouts=timeouts)
        return rec
    return wrapped


# ----------------------------------------------------------------------------- one scenario
CURRENT = contextvars.ContextVar("current_request", default=None)
NOTE = contextvars.ContextVar("run_note", default="")


async def run_scenario(sc: Scenario, duration_s: float, seed: int, calibration: dict, questions: list[dict],
                       retriever: Retriever, run_stamp: str, replay: Replay, rows_path: Path,
                       detail_path: Path) -> dict:
    sw = copy.deepcopy(sc.switches)
    stable = int(hashlib.sha256(f"{seed}:{sc.name}".encode()).hexdigest()[:8], 16)   # hash() is salted per process
    async with StandIn(calibration, sw, replay, seed=stable) as s:
        if sc.exhaust_daily:
            s.control({"exhaust_daily": "all"})
        base_chat_client = LLMClient(Endpoint(s.base_url, source="stand-in"))

        async def chat(payload, *, phase, timeouts=None):
            cur = CURRENT.get()
            t0 = time.perf_counter()
            rec = await base_chat_client.stream_chat(payload, timeouts=timeouts)
            if cur is not None:
                u = rec.usage or {}
                cur["calls"].append({"step": phase, "model": payload["model"], "status": rec.status,
                                     "outcome": rec.outcome,
                                     "start_ms": round((t0 - cur["t0"]) * 1000, 1),
                                     "end_ms": round((time.perf_counter() - cur["t0"]) * 1000, 1),
                                     "prompt_tokens": u.get("prompt_tokens") or 0,
                                     "completion_tokens": u.get("completion_tokens") or 0,
                                     "replay": rec.other_headers.get("x-standin-replay")})
            return rec
        cfg = load_config()
        real_lim = Limiter.from_probe()
        if sc.naive:
            cfg = copy.deepcopy(cfg)
            cfg["models"]["understand"] = cfg["models"]["understand"][:1]
            cfg["models"]["draft"] = cfg["models"]["draft"][:1]
            lim, chat_fn = NaiveLimiter(real_lim), naive_retry(chat)
        else:
            lim, chat_fn = real_lim, chat
        pipe = Pipeline(chat_fn, lim, retriever, cfg)
        arr = arrivals(sc, duration_s, seed, questions)
        run_id = f"{sc.name}-{run_stamp}"
        t_start = time.perf_counter()
        wall0 = datetime.now(timezone.utc)
        rows_f = rows_path.open("w", encoding="utf-8", newline="\n")
        rows_f.write(json.dumps({"type": "run", "run_id": run_id, "scenario": sc.name, "source": "stand-in",
                                 "duration_s": duration_s, "seed": seed, "arrivals": len(arr),
                                 "switches": sw.__dict__, "naive": sc.naive, "patience": PATIENCE,
                                 "naive_retry": NAIVE if sc.naive else None, "exhaust_daily": sc.exhaust_daily,
                                 "note": NOTE.get(),
                                 "started": wall0.isoformat(timespec="milliseconds")}) + "\n")
        detail_rows = []

        async def person(i: int, a: dict) -> None:
            delay = a["t"] - (time.perf_counter() - t_start)
            if delay > 0:
                await asyncio.sleep(delay)
            arrived = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
            rid = f"{run_id}-{i:04d}"
            cur = {"t0": time.perf_counter(), "calls": []}
            token = CURRENT.set(cur)
            task = asyncio.ensure_future(pipe.answer(a["question"], request_id=rid))
            CURRENT.reset(token)
            gave_up, reply, err = False, None, None
            try:
                reply = await asyncio.wait_for(asyncio.shield(task), timeout=a["patience_s"])
            except asyncio.TimeoutError:
                gave_up = True
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass
            except Exception as e:           # the pipeline turns internal failures into "busy"; this is a bug
                err = f"{type(e).__name__}: {e}"
            waited_ms = round((time.perf_counter() - cur["t0"]) * 1000, 1)
            attempts = [{"step": c["step"], "model": c["model"], "status": c["status"], "outcome": c["outcome"],
                         "prompt_tokens": c["prompt_tokens"], "completion_tokens": c["completion_tokens"]}
                        for c in cur["calls"]]
            if reply is not None:
                d = reply.to_dict()
                # every HTTP attempt (a naive retry is several attempts inside one pipeline call), so
                # results.csv's retries column counts them; the pipeline's own list is kept beside it
                d["pipeline_model_calls"], d["model_calls"] = d["model_calls"], attempts
            else:
                d = {"kind": "gave_up" if gave_up else "error", "text": "", "sources": [], "level": None,
                     "released_ms": None, "budget_wait_ms": None, "notice": err or "the person left",
                     "model_calls": attempts,
                     "tokens_in": sum(c["prompt_tokens"] for c in cur["calls"]),
                     "tokens_out": sum(c["completion_tokens"] for c in cur["calls"]), "spans": [], "checks": {}}
            row = {"type": "answer", "run_id": run_id, "scenario": sc.name, "source": "stand-in", "setting": "large",
                   "setting_def": {"piece_chars": cfg["retrieval"]["piece_chars"], "k": cfg["retrieval"]["k_full"],
                                   "rerank": cfg["retrieval"]["rerank"], "router": True},
                   "drafter": cfg["models"]["draft"][0], "qid": a["qid"], "request_id": rid, "start_time": arrived,
                   "arrival_offset_s": round(a["t"], 3), "patience_s": a["patience_s"], "gave_up": gave_up,
                   "waited_ms": waited_ms, "degraded": bool(d.get("level")), "reply": d}
            rows_f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
            rows_f.flush()
            detail_rows.append({"request_id": rid, "run_id": run_id, "scenario": sc.name, "qid": a["qid"],
                                "arrival_offset_s": round(a["t"], 3), "patience_s": a["patience_s"],
                                "gave_up": gave_up, "waited_ms": waited_ms, "kind": d["kind"], "level": d.get("level"),
                                "released_ms": d.get("released_ms"), "budget_wait_ms": d.get("budget_wait_ms"),
                                "notice": d.get("notice"), "spans": d.get("spans") or [], "model_calls": cur["calls"]})
        try:
            await asyncio.gather(*(person(i, a) for i, a in enumerate(arr)))
        finally:
            rows_f.close()
        with detail_path.open("a", encoding="utf-8", newline="\n") as f:
            for d in detail_rows:
                f.write(json.dumps(d, ensure_ascii=False, default=str) + "\n")
        kinds = {}
        for d in detail_rows:
            k = "gave_up" if d["gave_up"] else d["kind"]
            kinds[k] = kinds.get(k, 0) + 1
        return {"scenario": sc.name, "requests": len(detail_rows), "outcomes": kinds, "stand_in": s.stats,
                "wall_s": round(time.perf_counter() - t_start, 1)}


async def main_async(a) -> int:
    with (ROOT / "data" / "questions.jsonl").open(encoding="utf-8") as f:
        questions = [json.loads(x) for x in f if x.strip()]
    retriever = Retriever.from_blocks(load_config()["retrieval"]["piece_chars"])
    calibration = load_calibration(a.calibration)
    replay = Replay(ANSWERS)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    a.out_dir.mkdir(parents=True, exist_ok=True)
    a.detail.parent.mkdir(parents=True, exist_ok=True)
    names = a.scenarios.split(",")
    for n in names:
        if n not in SCENARIOS:
            raise SystemExit(f"unknown scenario {n}; choose from {', '.join(SCENARIOS)}")
    # rows are written under a .partial name and renamed when the scenario ends, so a reader never
    # sees a half-written file
    paths = {n: a.out_dir / f"{n}-{stamp}.jsonl" for n in names}
    results = await asyncio.gather(*(run_scenario(SCENARIOS[n], a.duration_s, a.seed, calibration, questions,
                                                  retriever, stamp, replay, paths[n].with_suffix(".jsonl.partial"),
                                                  a.detail) for n in names))
    for n in names:
        paths[n].with_suffix(".jsonl.partial").replace(paths[n])
    summary = {"_generated_by": "scripts/run_load.py", "stamp": stamp, "duration_s": a.duration_s, "seed": a.seed,
               "replay": {"router": len(replay.router), "drafts": len(replay.draft)}, "scenarios": results}
    (a.out_dir / f"summary-{stamp}.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8",
                                                    newline="\n")
    for r in results:
        print(f"{r['scenario']:<22} {r['requests']:>4} requests in {r['wall_s']} s  {r['outcomes']}  "
              f"stand-in: ok {r['stand_in']['ok']} 429 {r['stand_in']['429']} 503 {r['stand_in']['503']} "
              f"stalls {r['stand_in']['stalls']} replay hits {r['stand_in']['replay_hits']}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", default=",".join(SCENARIOS))
    ap.add_argument("--duration-s", type=float, default=300.0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out-dir", type=Path, default=RUNS)
    ap.add_argument("--detail", type=Path, default=DETAIL)
    ap.add_argument("--calibration", type=Path, default=ROOT / "data" / "calibration.json")
    ap.add_argument("--note", default="", help="why this run exists (kept in each run header)")
    a = ap.parse_args(argv)
    NOTE.set(a.note)
    return asyncio.run(main_async(a))


if __name__ == "__main__":
    sys.exit(main())
