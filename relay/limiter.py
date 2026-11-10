"""Client-side rate limiter for Groq, built from the probe's measurements (data/probe/), not the docs.

What the probe found, and how each finding shapes this module (data/probe/findings.md):
  * Q1: every limit refills continuously like a bucket.
      - tokens: TPM/60 per second; x-ratelimit-reset-tokens = deficit / that rate
      - requests: a request was readmitted 2 s after the RPM wall (30/60 per second)
      - daily requests: x-ratelimit-reset-requests = deficit / (1000/86400) on every response
    So every limit here is a ``Bucket(capacity, period_s)`` with refill capacity/period per second.
  * Q3: limits are per model -> one set of buckets per model, no sharing.
  * Q2: admission holds the prompt plus a capped part of max_tokens, not max_tokens in full: on
    calls admitted alone into a full bucket the hold was at most min(max_tokens, 200), and
    max_tokens 2000 held no more than 256 did. The hold is released when the call ends and the
    real completion is charged. Header values are computed at admission (header arrival minus
    usage.queue_time).
      -> ``reserve`` charges 1 request and prompt estimate + min(max_tokens, hold_cap).
      -> ``settle`` replaces the estimate with the real prompt_tokens and, at the end, releases
         the hold down to the real completion_tokens.
      -> ``hold_cap`` starts at the probe's largest observed hold and is learned online from calls
         admitted alone into a full bucket.
  * Q5: 429s don't count toward the request allowance -> a 429 releases its charges. RPM is
    visible only in 429 bodies (Limit/Used/Requested/"try again in"); no header shows it.
  * Limits a model's probe never hit (e.g. qwen's RPM) are not invented from the docs. The first
    429 that names a limit creates that bucket at runtime (``_learn_from_body``), for any limit
    type: RPM, RPD, TPM, TPD, ...

Header overwrite, accounting for calls in flight
  Each bucket is an *anchor* (time, level) plus the model's log of our own charge events. Level
  now = replay the events after the anchor, with refill, capped at capacity. When a response
  arrives, its headers become the new anchor at its admission time:
    x-ratelimit-remaining-tokens   -> TPM anchor
    x-ratelimit-remaining-requests -> RPD anchor
    429 body "Limit L, Used U"     -> anchor L - U for the named limit
  Our calls admitted after that moment, and completions that ended after it, are events after the
  anchor, so they are applied on top. Calls still in flight are counted exactly once, whatever
  order the responses come back in. An anchor older than the current one is ignored.

Prompt-token estimator (``PromptEstimator``)
  Groq charges prompt_tokens, which we can't count client-side: there is a hidden chat template.
  The probe showed tokens ≈ a + b*chars + c*messages fits very well (R² ≥ 0.99). Per model:
    1. Keep the last N observations (chars, messages, tokens). Two sources:
       - usage.prompt_tokens from every successful call (exact);
       - the "Requested" value of every token-limit 429 body. Groq's own count of what the call
         needed; available even though the call failed.
    2. Least-squares fit of a, b, c (dropping c, then b, if the data can't separate them).
    3. Scale up by the worst relative under-prediction in the window, plus a margin (1% and
       4 tokens), so the estimate is at or above every observation in the window. The bound is
       relative because tokens per character vary with the text: math-heavy passages are denser.
  ``from_probe`` seeds it with every probe call. Each settled call and each new token 429 adds a
  point and refits, so a new prompt shape that undershoots raises the bound on the next call. If
  a token 429's Requested matches prompt + max_tokens better than prompt alone, admission does
  hold max_tokens after all (Q2 overturned). That model then switches to charging it, and the
  switch is logged in ``findings_changed``.

API
  r = limiter.reserve(model, prompt, max_tokens, deadline)     # never blocks
      -> Grant     granted now (charges recorded)
      -> Ticket    predicted wait (r.wait_s), counting every queued request in front, FIFO per model
      raises Unavailable(reason, wait_s, bucket) if the prediction misses the deadline, or the
      request can never fit
  grant = await ticket.acquire()      # never waits past the deadline; raises Unavailable instead
  ticket.cancel()                     # caller chose to degrade instead
  grant.settle(call_record)           # headers overwrite, usage replaces estimates, 429s release
``deadline`` is an absolute time on the limiter's clock (``limiter.now() + seconds``).
"""
from __future__ import annotations

import asyncio
import itertools
import json
import math
import re
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
CALIBRATION = ROOT / "data" / "probe" / "calibration.json"
PROBE_RAW = ROOT / "data" / "probe" / "probe_raw.jsonl"

REQUESTS, TOKENS = "requests", "tokens"
EPS = 1e-6
COMPACT_AFTER_S = 300.0     # events older than this are folded into the anchor; older headers ignored

LIMIT_RE = re.compile(r"on (?P<what>[a-z ]+?) \((?P<short>[A-Z]+)\): Limit (?P<limit>\d+), Used (?P<used>\d+), "
                      r"Requested (?P<requested>\d+)\. Please try again in (?P<wait>(?:[0-9.]+(?:ms|h|m|s))+)")
TOO_LARGE_RE = re.compile(r"on (?P<what>[a-z ]+?) \((?P<short>[A-Z]+)\): Limit (?P<limit>\d+), Requested (?P<requested>\d+)")


class Unavailable(Exception):
    """The request can't be served before its deadline (or ever). The caller should degrade."""

    def __init__(self, model: str, reason: str, wait_s: float | None = None, bucket: str | None = None):
        super().__init__(f"{model}: {reason}" + (f" (predicted wait {wait_s:.1f} s)" if wait_s is not None else ""))
        self.model, self.reason, self.wait_s, self.bucket = model, reason, wait_s, bucket


def parse_duration(s: str | None) -> float | None:
    """'7.66s' / '2m59.56s' / '1h2m3s' / '120ms' / '2' -> seconds."""
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        pass
    parts = re.findall(r"([0-9]*\.?[0-9]+)(ms|h|m|s)", s)
    return sum(float(n) * {"ms": 0.001, "h": 3600, "m": 60, "s": 1}[u] for n, u in parts) if parts else None


def parse_limit_body(text: str | None) -> dict | None:
    """Groq's 429/413 message -> {short, unit, period_s, limit, used?, requested, wait_s?}."""
    if not text:
        return None
    try:
        obj = json.loads(text)
        msg = (obj.get("error") or {}).get("message") or ""
    except (json.JSONDecodeError, AttributeError):
        msg = text
    m = LIMIT_RE.search(msg) or TOO_LARGE_RE.search(msg)
    if not m:
        return None
    d = m.groupdict()
    what = d["what"]
    unit = TOKENS if "token" in what else REQUESTS if "request" in what else what
    period = 60.0 if "minute" in what else 3600.0 if "hour" in what else 86400.0 if "day" in what else None
    if period is None:
        period = {"M": 60.0, "H": 3600.0, "D": 86400.0}.get(d["short"][-1:], 60.0)
    return {"short": d["short"], "unit": unit, "period_s": period, "limit": int(d["limit"]),
            "used": int(d["used"]) if d.get("used") else None, "requested": int(d["requested"]),
            "wait_s": parse_duration(d.get("wait"))}


# ----------------------------------------------------------------------------- estimator
def _solve(a: list[list[float]], b: list[float]) -> list[float] | None:
    """Gaussian elimination with partial pivoting; None if (near) singular."""
    n = len(b)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(m[r][c]))
        if abs(m[p][c]) < 1e-9 * max(1.0, max(abs(x) for row in m for x in row[:n])):
            return None
        m[c], m[p] = m[p], m[c]
        for r in range(n):
            if r != c:
                f = m[r][c] / m[c][c]
                for k in range(c, n + 1):
                    m[r][k] -= f * m[c][k]
    return [m[i][n] / m[i][i] for i in range(n)]


@dataclass(frozen=True)
class Prompt:
    """A prompt-token estimate plus what it was computed from (so settle can calibrate)."""
    tokens: int
    chars: int
    messages: int


class PromptEstimator:
    def __init__(self, default_per_char: float = 1 / 3, default_per_message: float = 8.0,
                 default_overhead: float = 128.0, margin_tokens: float = 4.0, margin_frac: float = 0.01,
                 max_points: int = 400) -> None:
        self.default = (default_overhead, default_per_char, default_per_message)
        self.margin_tokens, self.margin_frac = margin_tokens, margin_frac
        self.points: dict[str, deque] = {}
        self.fits: dict[str, dict] = {}
        self.misses: dict[str, int] = {}           # observations above the estimate at the time
        self.max_points = max_points

    def observe(self, model: str, chars: int, messages: int, tokens: int, source: str = "usage") -> None:
        if self.estimate(model, chars, messages) < tokens:
            self.misses[model] = self.misses.get(model, 0) + 1
        self.points.setdefault(model, deque(maxlen=self.max_points)).append((chars, messages, tokens, source))
        self._refit(model)

    def _refit(self, model: str) -> None:
        pts = list(self.points.get(model) or [])
        a, b, c = self.default
        kind = "default"
        if len(pts) >= 3:
            xs = [(1.0, p[0], p[1]) for p in pts]
            ys = [float(p[2]) for p in pts]
            for cols, name in (((0, 1, 2), "a+b*chars+c*messages"), ((0, 1), "a+b*chars"), ((0,), "a")):
                ata = [[sum(x[i] * x[j] for x in xs) for j in cols] for i in cols]
                aty = [sum(x[i] * y for x, y in zip(xs, ys)) for i in cols]
                sol = _solve(ata, aty)
                if sol is not None:
                    coef = dict(zip(cols, sol))
                    a = coef.get(0, 0.0)
                    b = coef.get(1, 0.0 if name == "a" else self.default[1])
                    c = coef.get(2, 0.0)
                    if name == "a":
                        b = 0.0
                    kind = name
                    break
        preds = [a + b * p[0] + c * p[1] for p in pts]
        resid = [p[2] - q for p, q in zip(pts, preds)]
        # relative bound: tokens per character vary with the text (math-heavy passages are denser), so
        # the worst under-prediction is kept as a fraction of the prediction, not a fixed number
        frac = max([0.0] + [r / q for r, q in zip(resid, preds) if q >= 1.0])
        absb = max([0.0] + [r for r, q in zip(resid, preds) if q < 1.0])
        self.fits[model] = {"intercept": a, "per_char": b, "per_message": c, "form": kind, "n": len(pts),
                            "frac_bound": frac, "abs_bound": absb, "max_residual": max([0.0] + resid),
                            "sources": sorted({p[3] for p in pts})}

    def estimate(self, model: str, chars: int, messages: int) -> int:
        f = self.fits.get(model)
        if f is None:
            a, b, c = self.default
            return math.ceil(a + b * chars + c * messages)
        pred = f["intercept"] + f["per_char"] * chars + f["per_message"] * messages
        return math.ceil(max(pred, 0.0) * (1 + f["frac_bound"] + self.margin_frac) + f["abs_bound"] + self.margin_tokens)

    def prompt(self, model: str, messages: list[dict]) -> Prompt:
        chars = sum(len(m.get("content") or "") for m in messages if isinstance(m, dict))
        return Prompt(self.estimate(model, chars, len(messages)), chars, len(messages))


# ----------------------------------------------------------------------------- buckets
@dataclass
class Event:
    t: float
    unit: str
    amount: float
    grant_id: int
    kind: str          # "request" | "prompt" | "completion" | "correction"
    seq: int = 0       # order of our own charges; anchors record the last one they already include


@dataclass
class Bucket:
    name: str              # "RPM", "TPM", "RPD", "TPD", ...
    unit: str              # REQUESTS | TOKENS
    capacity: float
    period_s: float
    anchor_t: float
    anchor_level: float
    source: str            # where capacity/period came from
    after_seq: int = 0     # our events with seq <= this are already inside anchor_level

    @property
    def rate(self) -> float:
        return self.capacity / self.period_s

    def level(self, events: list[Event], now: float) -> float:
        lvl, t = self.anchor_level, self.anchor_t
        for e in events:
            if e.seq <= self.after_seq or e.unit != self.unit:
                continue
            if e.t > now:
                break
            te = max(t, e.t)
            lvl = min(self.capacity, lvl + self.rate * (te - t)) - e.amount
            t = te
        return min(self.capacity, lvl + self.rate * max(0.0, now - t))


@dataclass
class ModelState:
    model: str
    buckets: dict[str, Bucket]
    admission_includes_max_tokens: bool = False     # Q2 says False; a token 429 can overturn it
    hold_cap: float | None = None                   # admission holds prompt + min(max_tokens, hold_cap)
    hold_obs: deque = field(default_factory=lambda: deque(maxlen=50))
    inflight: dict = field(default_factory=dict)    # grant id -> Grant
    count_429: bool = False                         # Q5 says 429s don't count
    events: list[Event] = field(default_factory=list)
    holds: dict[int, float] = field(default_factory=dict)     # expected completion tokens in flight
    queue: deque = field(default_factory=deque)
    changed: asyncio.Event | None = None


class Grant:
    def __init__(self, limiter: "Limiter", gid: int, model: str, prompt: Prompt | None, prompt_tokens: int,
                 max_tokens: int, t: float, waited_s: float) -> None:
        self.limiter, self.id, self.model = limiter, gid, model
        self.prompt, self.prompt_tokens, self.max_tokens = prompt, prompt_tokens, max_tokens
        self.t_granted, self.waited_s = t, waited_s
        self.wait_s = 0.0
        self.settled = False
        self.hold = 0.0          # tokens held at admission beyond the prompt
        self.alone = True        # no other call of this model in flight while this one was admitted

    def settle(self, rec, t_sent: float | None = None) -> None:
        self.limiter.settle(self, rec, t_sent)

    def release(self) -> None:
        """The call was never sent."""
        self.limiter.release(self)


class Ticket:
    def __init__(self, limiter: "Limiter", tid: int, model: str, prompt: Prompt | None, prompt_tokens: int,
                 max_tokens: int, deadline: float, wait_s: float, ahead: int, t: float) -> None:
        self.limiter, self.id, self.model = limiter, tid, model
        self.prompt, self.prompt_tokens, self.max_tokens = prompt, prompt_tokens, max_tokens
        self.deadline, self.wait_s, self.ahead, self.t_created = deadline, wait_s, ahead, t
        self.done = False

    async def acquire(self) -> Grant:
        return await self.limiter._acquire(self)

    def cancel(self) -> None:
        self.limiter._drop(self)


# ----------------------------------------------------------------------------- limiter
class Limiter:
    def __init__(self, models: dict[str, ModelState], estimator: PromptEstimator | None = None,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.models = models
        self.estimator = estimator or PromptEstimator()
        self.clock = clock
        self._ids = itertools.count(1)
        self._seq = itertools.count(1)
        self.findings_changed: list[dict] = []
        self.settings: dict[str, dict] = {}     # reasoning settings the estimator was calibrated for
        self.completions: dict[tuple[str, int], deque] = {}    # (model, max_tokens) -> recent completion_tokens

    def now(self) -> float:
        return self.clock()

    # ---------------------------------------------------------------- construction
    @classmethod
    def from_probe(cls, calibration: Path = CALIBRATION, raw: Path = PROBE_RAW,
                   clock: Callable[[], float] = time.monotonic) -> "Limiter":
        """Buckets, charging rules and the estimator's seed, all from the probe's outputs."""
        cal = json.loads(calibration.read_text(encoding="utf-8"))
        verdicts = cal.get("verdicts", {})
        rpd429 = {r["model"]: r["verdict"] for r in cal.get("rpd_429_counting", [])}
        now = clock()
        models: dict[str, ModelState] = {}
        rpd_period = _rpd_period_from_raw(raw)
        for m, v in cal["models"].items():
            ls = v["limits_seen"]
            buckets: dict[str, Bucket] = {}
            tpm = ls["headers"].get("x-ratelimit-limit-tokens")
            if tpm:
                ratio = ((cal.get("tokens_window", {}).get(m) or [{}])[0]
                         .get("reset_tokens_over_time_to_refill_at_tpm_per_60", {}).get("p50")) or 1.0
                cap = float(max(int(x) for x in tpm))
                buckets["TPM"] = Bucket("TPM", TOKENS, cap, 60.0 * ratio, now, cap,
                                        f"probe: x-ratelimit-limit-tokens; period 60 s x reset ratio {ratio}")
            rpd = ls["headers"].get("x-ratelimit-limit-requests")
            if rpd:
                cap = float(max(int(x) for x in rpd))
                buckets["RPD"] = Bucket("RPD", REQUESTS, cap, rpd_period, now, cap,
                                        f"probe: x-ratelimit-limit-requests; period from reset-requests ({rpd_period:.0f} s)")
            for e in ls["error_responses"]:
                p = e.get("parsed")
                if not p or p.get("form") != "rate_limit_reached" or p["short"] in buckets:
                    continue
                unit = TOKENS if "token" in p["what"] else REQUESTS
                # "Used = Limit, try again in w" means one unit refills in w seconds -> period = Limit * w
                period = (p["limit"] * p["try_again_in_s"] / max(1, p["requested"])
                          if p.get("try_again_in_s") and p.get("used") == p["limit"] else
                          {"M": 60.0, "H": 3600.0, "D": 86400.0}.get(p["short"][-1:], 60.0))
                buckets[p["short"]] = Bucket(p["short"], unit, float(p["limit"]), period, now, float(p["limit"]),
                                             f"probe: 429 body Limit {p['limit']}, try again in {p.get('try_again_in')}")
            adm = (verdicts.get("admission") or {}).get(m, "")
            ex = ((v.get("token_accounting_fit") or {}).get("admission_extra_after_idle") or {})
            obs = [x for k in (ex.get("by_max_tokens") or {}).values() for x in k["values"]]
            st = ModelState(m, buckets, admission_includes_max_tokens=adm.startswith("max_tokens reserved"),
                            count_429=rpd429.get(m, "").startswith("429s counted"),
                            hold_cap=float(max(obs)) if obs else None)
            st.hold_obs.extend(float(x) for x in obs)
            models[m] = st
        lim = cls(models, PromptEstimator(), clock)
        lim.settings = {}
        if raw.exists():
            lines = [json.loads(x) for x in raw.read_text(encoding="utf-8").splitlines() if x.strip()]
            # the reasoning settings each model runs with (probe P0 decision): settings change the hidden
            # chat template (qwen + reasoning_effort low adds ~27 prompt tokens), so seed only from those
            for o in lines:
                if o.get("type") == "event" and o.get("event") == "settings" and o.get("mode") == "real":
                    lim.settings[o["model"]] = o["settings"]
            keys = ("reasoning_effort", "reasoning_format", "include_reasoning")
            for o in lines:
                if o.get("type") != "call" or o.get("mode") != "real" or o["model"] not in models:
                    continue
                want = lim.settings.get(o["model"])
                if want is not None and {k: o["payload"].get(k) for k in keys} != {k: want.get(k) for k in keys}:
                    continue
                u = o["record"].get("usage") or {}
                if o["record"].get("status") == 200 and u.get("prompt_tokens") is not None:
                    lim.estimator.observe(o["model"], o["prompt_chars"], len(o["payload"]["messages"]),
                                          u["prompt_tokens"], "probe usage")
                    if u.get("completion_tokens") is not None:
                        lim._note_completion(o["model"], o["payload"].get("max_tokens"), u["completion_tokens"])
        lim.estimator.misses.clear()      # count misses from now on, not while seeding
        return lim

    # ---------------------------------------------------------------- state helpers
    def _state(self, model: str) -> ModelState:
        st = self.models.get(model)
        if st is None:
            raise Unavailable(model, "no measured limits for this model")
        if st.changed is None:
            st.changed = asyncio.Event()
        return st

    def _notify(self, st: ModelState) -> None:
        if st.changed is not None:
            st.changed.set()
            st.changed = asyncio.Event()

    def _event(self, st: ModelState, t: float, unit: str, amount: float, gid: int, kind: str) -> None:
        st.events.append(Event(t, unit, amount, gid, kind, next(self._seq)))

    def _anchor(self, st: ModelState, b: Bucket, t: float, level: float, own: int | None) -> None:
        """New truth for ``b`` at time ``t``. It already includes our events before ``t`` and the
        call's own admission charges (the server counted them before reporting)."""
        inc = [e.seq for e in st.events if e.t < t or (own is not None and e.grant_id == own
                                                      and e.kind in ("request", "prompt"))]
        b.anchor_t, b.anchor_level, b.after_seq = t, level, max(inc + [b.after_seq])

    def _compact(self, st: ModelState, now: float) -> None:
        horizon = now - COMPACT_AFTER_S
        for b in st.buckets.values():
            if b.anchor_t < horizon:
                lvl = b.level(st.events, horizon)
                b.anchor_t, b.anchor_level = horizon, lvl
                b.after_seq = max([e.seq for e in st.events if e.t <= horizon] + [b.after_seq])
        keep = min((b.after_seq for b in st.buckets.values()), default=0)
        st.events = [e for e in st.events if e.seq > keep]

    def levels(self, model: str) -> dict[str, float]:
        st = self._state(model)
        now = self.now()
        self._compact(st, now)
        return {n: b.level(st.events, now) for n, b in st.buckets.items()}

    @staticmethod
    def admission_hold(st: ModelState, max_tokens: int) -> float:
        """Tokens Groq holds at admission beyond the prompt (findings Q2)."""
        if st.admission_includes_max_tokens:
            return float(max_tokens)
        return float(min(max_tokens, st.hold_cap or 0.0))

    def _needs(self, st: ModelState, prompt_tokens: int, max_tokens: int) -> dict[str, float]:
        tok = prompt_tokens + self.admission_hold(st, max_tokens)
        return {n: (1.0 if b.unit == REQUESTS else float(tok) if b.unit == TOKENS else 0.0)
                for n, b in st.buckets.items()}

    def _expected_completion(self, model: str, max_tokens: int) -> float:
        d = self.completions.get((model, max_tokens))
        if d and len(d) >= 3:
            s = sorted(d)
            return float(min(max_tokens, s[max(0, math.ceil(0.9 * len(s)) - 1)]))
        return float(max_tokens)

    def _hold(self, model: str, max_tokens: int) -> float:
        """What a call in flight will still take when it ends: expected completion minus what
        admission already holds (for predictions only; Groq charges it at the end, so it doesn't
        block admission, but it will lower the bucket)."""
        st = self.models[model]
        return max(0.0, self._expected_completion(model, max_tokens) - self.admission_hold(st, max_tokens))

    def _note_completion(self, model: str, max_tokens: int | None, completion: int) -> None:
        if max_tokens:
            self.completions.setdefault((model, max_tokens), deque(maxlen=100)).append(completion)

    def _predict(self, st: ModelState, items: list[tuple[int, int]], now: float,
                 with_holds: bool = True) -> list[tuple[float, str | None]]:
        """Earliest start time of each (prompt_tokens, max_tokens) item, FIFO, given current levels
        and refill. ``with_holds`` also subtracts the expected completions of calls in flight (and of
        each item ahead), which Groq will charge when they end: the realistic prediction, used to
        refuse early. Without it: the earliest possible time, used to decide when to look again.
        Returns (time, limiting bucket) per item."""
        lv = {n: b.level(st.events, now) for n, b in st.buckets.items()}
        held = sum(st.holds.values()) if with_holds else 0.0
        for n, b in st.buckets.items():
            if b.unit == TOKENS:
                lv[n] -= held
        t = now
        out = []
        for pt, mx in items:
            need = self._needs(st, pt, mx)
            start, limiting = t, None
            for n, b in st.buckets.items():
                short = need[n] - lv[n]
                if short > EPS:
                    s = t + short / b.rate
                    if s > start:
                        start, limiting = s, n
            for n, b in st.buckets.items():
                lv[n] = min(b.capacity, lv[n] + b.rate * (start - t)) - need[n]
                if with_holds and b.unit == TOKENS:
                    lv[n] -= self._hold(st.model, mx)
            t = start
            out.append((start, limiting))
        return out

    def _feasible_now(self, st: ModelState, prompt_tokens: int, max_tokens: int, now: float) -> bool:
        need = self._needs(st, prompt_tokens, max_tokens)
        return all(b.level(st.events, now) + EPS >= need[n] for n, b in st.buckets.items())

    def _grant(self, st: ModelState, prompt: Prompt | None, prompt_tokens: int, max_tokens: int, now: float,
               waited: float) -> Grant:
        gid = next(self._ids)
        hold = self.admission_hold(st, max_tokens)
        self._event(st, now, REQUESTS, 1.0, gid, "request")
        self._event(st, now, TOKENS, float(prompt_tokens + hold), gid, "prompt")
        st.holds[gid] = self._hold(st.model, max_tokens)
        g = Grant(self, gid, st.model, prompt, prompt_tokens, max_tokens, now, waited)
        g.hold = hold
        g.alone = not st.inflight
        for other in st.inflight.values():
            other.alone = False
        st.inflight[gid] = g
        return g

    # ---------------------------------------------------------------- public API
    def estimate(self, model: str, messages: list[dict]) -> Prompt:
        return self.estimator.prompt(model, messages)

    def reserve(self, model: str, prompt_tokens: int | Prompt, max_tokens: int, deadline: float) -> Grant | Ticket:
        st = self._state(model)
        now = self.now()
        self._compact(st, now)
        prompt = prompt_tokens if isinstance(prompt_tokens, Prompt) else None
        pt = prompt.tokens if prompt else int(prompt_tokens)
        need = self._needs(st, pt, max_tokens)
        for n, b in st.buckets.items():
            if need[n] > b.capacity:
                raise Unavailable(model, f"needs {need[n]:.0f} {b.unit} but {n} holds at most {b.capacity:.0f}", None, n)
        if not st.queue and self._feasible_now(st, pt, max_tokens, now):
            return self._grant(st, prompt, pt, max_tokens, now, 0.0)
        items = [(t.prompt_tokens, t.max_tokens) for t in st.queue] + [(pt, max_tokens)]
        start, limiting = self._predict(st, items, now)[-1]
        if start > deadline:
            raise Unavailable(model, f"{limiting or 'queue'} would not have room before the deadline "
                                     f"({len(st.queue)} queued in front)", start - now, limiting)
        t = Ticket(self, next(self._ids), model, prompt, pt, max_tokens, deadline, start - now, len(st.queue), now)
        st.queue.append(t)
        return t

    async def _acquire(self, t: Ticket) -> Grant:
        st = self._state(t.model)
        while True:
            now = self.now()
            if t.done:
                raise Unavailable(t.model, "ticket cancelled")
            if st.queue and st.queue[0] is t and self._feasible_now(st, t.prompt_tokens, t.max_tokens, now):
                st.queue.popleft()
                t.done = True
                g = self._grant(st, t.prompt, t.prompt_tokens, t.max_tokens, now, now - t.t_created)
                self._notify(st)
                return g
            idx = list(st.queue).index(t)
            items = [(x.prompt_tokens, x.max_tokens) for x in list(st.queue)[: idx + 1]]
            start, limiting = self._predict(st, items, now)[-1]
            if start > t.deadline or now >= t.deadline:
                self._drop(t)
                raise Unavailable(t.model, f"{limiting or 'queue'} would not have room before the deadline",
                                  max(0.0, start - now), limiting)
            t.wait_s = start - now
            earliest = self._predict(st, items, now, with_holds=False)[-1][0]
            ev = st.changed or asyncio.Event()
            st.changed = ev
            # wake when room may exist, on any change (a settle, a release), or at the deadline
            timeout = min(max(earliest - now, 0.005), t.deadline - now)
            try:
                await asyncio.wait_for(ev.wait(), max(0.0, timeout))
            except asyncio.TimeoutError:
                pass

    def _drop(self, t: Ticket) -> None:
        st = self.models.get(t.model)
        t.done = True
        if st and t in st.queue:
            st.queue.remove(t)
            self._notify(st)

    def release(self, g: Grant) -> None:
        st = self._state(g.model)
        for e in st.events:
            if e.grant_id == g.id:
                e.amount = 0.0
        st.holds.pop(g.id, None)
        st.inflight.pop(g.id, None)
        g.settled = True
        self._notify(st)

    def settle(self, g: Grant, rec, t_sent: float | None = None) -> None:
        """Apply a finished call (relay.llm.CallRecord). ``t_sent`` is when the request went out on
        this limiter's clock (default: when it was granted)."""
        if g.settled:
            return
        st = self._state(g.model)
        now = self.now()
        t0 = g.t_granted if t_sent is None else t_sent
        t_hdr = t0 + rec.headers_ms / 1000 if rec.headers_ms is not None else None
        st.holds.pop(g.id, None)
        st.inflight.pop(g.id, None)
        g.settled = True
        status = rec.status
        mine = [e for e in st.events if e.grant_id == g.id]
        full_before = self._bucket_full_before(st, g, rec, t_hdr)

        def adjust(kind: str, new_amount: float) -> None:
            # Buckets whose anchor already includes the event (a header counted the real charge)
            # are unaffected; the others see the corrected amount. An event compacted away gets a
            # correction event instead.
            for e in mine:
                if e.kind != kind:
                    continue
                if e in st.events:
                    e.amount = new_amount
                else:
                    self._event(st, now, e.unit, new_amount - e.amount, g.id, "correction")

        if rec.outcome == "connect_error" or (status is not None and 400 <= status < 500
                                              and not (status == 429 and st.count_429)):
            # never reached Groq, or rejected (Q5: 429s don't count). A timeout before headers keeps
            # its charge: Groq may already have admitted the call.
            adjust("request", 0.0)
            adjust("prompt", 0.0)
        if status is not None and status >= 400:
            info = parse_limit_body(rec.error_body)
            if info:
                self._learn_from_body(st, g, info, t_hdr if t_hdr is not None else now)
        elif status == 200:
            u = rec.usage or {}
            pt = u.get("prompt_tokens")
            ct = u.get("completion_tokens")
            if pt is not None:
                adjust("prompt", float(pt) + g.hold)
                if g.prompt is not None:
                    self.estimator.observe(g.model, g.prompt.chars, g.prompt.messages, int(pt), "usage")
            if ct is None:   # cut off without usage: guess from what arrived (cancelled-stream billing unmeasured)
                ct = math.ceil((len(rec.text or "") + (rec.reasoning_chars or 0)) / 3)
            else:
                self._note_completion(g.model, g.max_tokens, int(ct))
            # release the admission hold down to the real completion (negative = tokens given back)
            self._event(st, now, TOKENS, float(ct) - g.hold, g.id, "completion")
            rem = _int(rec.header("x-ratelimit-remaining-tokens"))
            if full_before and pt is not None and rem is not None and "TPM" in st.buckets:
                self._learn_hold(st, g, st.buckets["TPM"].capacity - pt - rem)
        self._apply_headers(st, rec, t_hdr, g)
        self._notify(st)

    # ---------------------------------------------------------------- learning
    def _t_adm(self, rec, t_hdr: float, g: Grant) -> float:
        qt = (rec.usage or {}).get("queue_time")
        t_adm = t_hdr - qt if rec.status == 200 and isinstance(qt, (int, float)) else t_hdr
        return max(t_adm, g.t_granted)

    def _bucket_full_before(self, st: ModelState, g: Grant, rec, t_hdr: float | None) -> bool:
        """Was the TPM bucket full just before this call was admitted, with nothing else of ours in
        flight? Then its header shows exactly prompt + hold below the limit."""
        b = st.buckets.get("TPM")
        if b is None or t_hdr is None or not g.alone or rec.status != 200:
            return False
        others = [e for e in st.events if e.grant_id != g.id]
        return b.level(others, self._t_adm(rec, t_hdr, g)) >= b.capacity - 1.0

    def _learn_hold(self, st: ModelState, g: Grant, observed: float) -> None:
        observed = max(0.0, min(float(g.max_tokens), observed))
        st.hold_obs.append(observed)
        cap = max(st.hold_obs)
        if st.hold_cap is None or cap > st.hold_cap * 1.05 + 1:
            self.findings_changed.append({"model": st.model, "change": "admission hold cap raised",
                                          "was": st.hold_cap, "now": cap, "max_tokens": g.max_tokens})
        st.hold_cap = max(cap, st.hold_cap or 0.0)

    def _apply_headers(self, st: ModelState, rec, t_hdr: float | None, g: Grant) -> None:
        if t_hdr is None:
            return
        t_adm = self._t_adm(rec, t_hdr, g)
        own = g.id
        for hdr_rem, hdr_lim, hdr_reset, name, unit in (
                ("x-ratelimit-remaining-tokens", "x-ratelimit-limit-tokens", "x-ratelimit-reset-tokens", "TPM", TOKENS),
                ("x-ratelimit-remaining-requests", "x-ratelimit-limit-requests", "x-ratelimit-reset-requests", "RPD", REQUESTS)):
            rem, lim = _int(rec.header(hdr_rem)), _int(rec.header(hdr_lim))
            if rem is None:
                continue
            b = st.buckets.get(name)
            if b is None and lim:
                reset = parse_duration(rec.header(hdr_reset))
                period = (lim * reset / (lim - rem)) if reset and lim > rem else (60.0 if unit == TOKENS else 86400.0)
                b = st.buckets[name] = Bucket(name, unit, float(lim), period, t_adm, float(rem),
                                              f"runtime: {hdr_lim} header")
                self._anchor(st, b, t_adm, float(rem), own)
                continue
            if b is None:
                continue
            if lim and lim != b.capacity:
                b.capacity, b.source = float(lim), f"runtime: {hdr_lim} header"
            if t_adm >= b.anchor_t:
                self._anchor(st, b, t_adm, float(rem), own)

    def _learn_from_body(self, st: ModelState, g: Grant, info: dict, t: float) -> None:
        name = info["short"]
        b = st.buckets.get(name)
        if b is None:
            period = (info["limit"] * info["wait_s"] / max(1, info["requested"])
                      if info.get("wait_s") and info.get("used") == info["limit"] else info["period_s"])
            b = st.buckets[name] = Bucket(name, info["unit"], float(info["limit"]), period, t, float(info["limit"]),
                                          f"runtime: 429 body Limit {info['limit']}")
            self._anchor(st, b, t, float(info["limit"]), g.id)
            self.findings_changed.append({"model": st.model, "change": f"new limit {name} learned from a 429 body",
                                          "limit": info["limit"], "period_s": period})
        else:
            if info["limit"] != b.capacity:
                self.findings_changed.append({"model": st.model, "change": f"{name} capacity from a 429 body",
                                              "was": b.capacity, "now": info["limit"]})
                b.capacity, b.source = float(info["limit"]), f"runtime: 429 body Limit {info['limit']}"
            # Used == Limit and "try again in w": `requested` units refill in w seconds
            if info.get("wait_s") and info.get("used") == info["limit"]:
                period = info["limit"] * info["wait_s"] / max(1, info["requested"])
                if abs(period / b.period_s - 1) > 0.2:
                    self.findings_changed.append({"model": st.model, "change": f"{name} refill period from a 429 body",
                                                  "was_s": b.period_s, "now_s": period})
                    b.period_s = period
        if info.get("used") is not None and t >= b.anchor_t:
            self._anchor(st, b, t, float(info["limit"] - info["used"]), g.id)
        if info["unit"] == TOKENS and g.prompt is not None:
            # "Requested" is Groq's admission need for this call: prompt + hold
            req = info["requested"]
            as_capped = abs(req - (g.prompt_tokens + g.hold))
            as_full = abs(req - (g.prompt_tokens + g.max_tokens))
            if (as_full < as_capped and not st.admission_includes_max_tokens
                    and g.max_tokens - g.hold > 50):
                st.admission_includes_max_tokens = True
                self.findings_changed.append({"model": st.model, "change": "admission holds max_tokens in full (Q2 overturned)",
                                              "requested": req, "prompt_estimate": g.prompt_tokens,
                                              "max_tokens": g.max_tokens, "hold_cap_was": st.hold_cap})
            prompt_part = req - self.admission_hold(st, g.max_tokens)
            if prompt_part > 0:
                self.estimator.observe(st.model, g.prompt.chars, g.prompt.messages, prompt_part, "429 Requested")


def _int(v: str | None) -> int | None:
    try:
        return int(v) if v is not None else None
    except ValueError:
        return None


def _rpd_period_from_raw(raw: Path) -> float:
    """Daily-request refill period from x-ratelimit-reset-requests = deficit * period / limit."""
    vals = []
    if raw.exists():
        with raw.open(encoding="utf-8") as f:
            for line in f:
                o = json.loads(line)
                if o.get("type") != "call" or o.get("mode") != "real":
                    continue
                h = {k.lower(): v for k, v in o["record"].get("rate_limit_headers") or []}
                rem, lim = _int(h.get("x-ratelimit-remaining-requests")), _int(h.get("x-ratelimit-limit-requests"))
                reset = parse_duration(h.get("x-ratelimit-reset-requests"))
                if rem is not None and lim and lim > rem and reset:
                    vals.append(reset * lim / (lim - rem))
    if not vals:
        return 86400.0
    vals.sort()
    return vals[len(vals) // 2]
