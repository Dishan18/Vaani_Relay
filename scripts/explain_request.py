"""Where one request's time went: waiting / working / retrying, before or after the first token.

    .venv\\Scripts\\python scripts\\explain_request.py --request-id ID [--detail data/results/detail.jsonl]
    .venv\\Scripts\\python scripts\\explain_request.py --slowest [--scenario overload] [--run-id ...]

Reads one request's spans and model calls from a detail file (scripts/run_load.py writes them) and
splits its time, from arrival to release (or to when the person left), into:
  waiting   inside a wait:* span: waiting for budget (the client-side limiter's queue)
  retrying  an attempt that follows a failed attempt of the same step, the gap before it (a back-off or
            a wait), and anything else inside the same pipeline call span after the first failure
  working   model calls that were the step's first try, and the gaps the stand-in spent streaming
  local     everything else: retrieval, the decide step, the citation gate, our own overhead
Before / after the first token: replies are released only after their checks (buffered), so all of a
request's time is before its first token; a person who gave up never got one.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DETAIL = ROOT / "data" / "results" / "detail.jsonl"
PRIORITY = {"retrying": 3, "waiting": 2, "working": 1}


def load(detail: Path) -> list[dict]:
    with detail.open(encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def intervals(d: dict) -> list[tuple[float, float, str, str]]:
    out = []
    for sp in d.get("spans") or []:
        if sp["name"].startswith("wait:") and sp.get("end_ms") is not None:
            out.append((sp["start_ms"], sp["end_ms"], "waiting", f"{sp['name']} {sp.get('model') or ''} {sp.get('outcome', '')}"))
    by_step: dict[str, list[dict]] = {}
    for c in d.get("model_calls") or []:
        if c.get("start_ms") is None:
            continue
        by_step.setdefault(c["step"], []).append(c)
    for step, cs in by_step.items():
        cs.sort(key=lambda c: c["start_ms"])
        failed_end = None
        for c in cs:
            ok = c["status"] == 200 and c["outcome"] == "ok"
            what = f"{step} {c['model'].split('/')[-1]} {c['status']}:{c['outcome']}"
            if failed_end is not None:
                out.append((failed_end, c["start_ms"], "retrying", f"{step}: gap before the retry"))
                out.append((c["start_ms"], c["end_ms"], "retrying", what))
            else:
                out.append((c["start_ms"], c["end_ms"], "working", what))
            if not ok:
                failed_end = c["end_ms"]
    return out


def split(d: dict) -> dict:
    end = d.get("released_ms") if d.get("released_ms") is not None else d.get("waited_ms") or 0.0
    iv = [x for x in intervals(d) if x[1] > x[0]]
    points = sorted({0.0, float(end)} | {p for a, b, *_ in iv for p in (a, b) if 0 <= p <= end})
    total = {"waiting": 0.0, "retrying": 0.0, "working": 0.0, "local": 0.0}
    for a, b in zip(points, points[1:]):
        mid = (a + b) / 2
        cover = [x[2] for x in iv if x[0] <= mid < x[1]]
        kind = max(cover, key=lambda k: PRIORITY[k]) if cover else "local"
        total[kind] += b - a
    return {"end_ms": end, "split_ms": {k: round(v, 1) for k, v in total.items()},
            "before_first_token_ms": round(end, 1), "after_first_token_ms": 0.0,
            "got_first_token": d.get("released_ms") is not None and not d.get("gave_up"),
            "timeline": sorted(iv)}


def render(d: dict, s: dict) -> str:
    L = [f"request {d['request_id']}  ({d['scenario']}, {d.get('qid')}, arrived at +{d.get('arrival_offset_s')} s, "
         f"patience {d.get('patience_s')} s)",
         f"outcome: {d['kind']} level={d.get('level')} gave_up={d.get('gave_up')}  notice: {d.get('notice') or '-'}",
         f"total {s['end_ms']:.0f} ms, all before the first token (replies are released after their checks)"
         + ("" if s["got_first_token"] else "; no first token: the person left"), ""]
    for k in ("waiting", "working", "retrying", "local"):
        v = s["split_ms"][k]
        L.append(f"  {k:<9} {v:>9.0f} ms  {100 * v / s['end_ms']:5.1f}%" if s["end_ms"] else f"  {k:<9} {v:>9.0f} ms")
    L += ["", "timeline (ms from arrival):"]
    for a, b, kind, what in s["timeline"]:
        L.append(f"  {a:>8.0f} - {b:>8.0f}  {kind:<9} {what}")
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--detail", type=Path, default=DETAIL)
    ap.add_argument("--request-id")
    ap.add_argument("--slowest", action="store_true")
    ap.add_argument("--scenario")
    ap.add_argument("--run-id")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    rows = load(a.detail)
    if a.request_id:
        rows = [d for d in rows if d["request_id"] == a.request_id]
    else:
        rows = [d for d in rows if (not a.scenario or d["scenario"] == a.scenario) and (not a.run_id or d["run_id"] == a.run_id)]
        rows.sort(key=lambda d: -(d.get("released_ms") if d.get("released_ms") is not None else d.get("waited_ms") or 0))
        rows = rows[:1]
    if not rows:
        print("no such request", file=sys.stderr)
        return 2
    d = rows[0]
    s = split(d)
    print(json.dumps({"request_id": d["request_id"], **{k: v for k, v in s.items() if k != "timeline"}}) if a.json
          else render(d, s))
    return 0


if __name__ == "__main__":
    sys.exit(main())
