"""The shared daily Groq budget, enforced in code.

Every real Groq call goes through ``BudgetedClient``:
  1. ``Ledger.reserve`` checks every cap that applies, counting calls still in flight. It refuses
     (raises ``BudgetExceeded``) before anything is sent if the call could break a cap.
  2. The call runs on ``relay.llm.LLMClient``.
  3. ``Ledger.settle`` appends one line to ``results/ledger/groq_calls.jsonl`` (flushed and
     fsynced). This happens for every call, including timeouts and cancellations.

Caps come from ``config/budget.toml`` as fractions of the documented daily allowance
(``data/limits_documented.json``):
  * an account-wide safety net over all real calls (scope "*");
  * per-run caps, e.g. the probe's 10% (scope "probe").
They are counted over a **rolling 24 h window**. That is at least as strict as a fixed daily
reset at any hour, and we don't yet know which one Groq uses.

Charging is conservative until the probe tells us how Groq really counts:
  * connect error (the server was never reached): 0 requests, 0 tokens
  * HTTP 429: 1 request, estimated prompt tokens. Assumes a rejected call isn't charged its
    max_tokens; probe phase P2 checks this.
  * anything else: 1 request, max(estimated prompt + max_tokens, reported total usage).
    Correct whether Groq charges max_tokens at admission or actual usage.
A call without ``max_tokens`` is refused: unbounded output can't be reserved.

The key is shared, so other users may be spending the same allowance. After each call the
server's own ``x-ratelimit-remaining-requests`` (RPD) is compared with a floor. Below it, the
ledger halts all further real calls with a reason.

Only one process may hold the ledger at a time (OS file lock on ``groq_calls.lock``), so two
scripts can't both pass the same cap.

Status:  .venv\\Scripts\\python -m relay.ledger
"""
from __future__ import annotations

import itertools
import json
import math
import os
import sys
import time
import tomllib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from relay.llm import CONNECT_ERROR, CallRecord, LLMClient

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LEDGER = ROOT / "results" / "ledger" / "groq_calls.jsonl"
DEFAULT_BUDGET = ROOT / "config" / "budget.toml"
DOCUMENTED = ROOT / "data" / "limits_documented.json"


class BudgetExceeded(Exception):
    """Raised instead of sending a call that could break a cap."""


@dataclass(frozen=True)
class ModelCap:
    requests: int
    tokens: int


@dataclass(frozen=True)
class Cap:
    name: str
    scope: str                       # "*" = every real call; otherwise only calls in this scope
    models: dict[str, ModelCap]      # a model not listed here is refused under this cap
    total_requests: int | None = None

    def applies(self, scope: str) -> bool:
        return self.scope == "*" or self.scope == scope


@dataclass
class Budget:
    caps: list[Cap]
    window_hours: float = 24.0
    min_remaining_requests_fraction: float | None = None


def load_budget(budget_path: Path = DEFAULT_BUDGET, documented_path: Path = DOCUMENTED) -> Budget:
    cfg = tomllib.loads(budget_path.read_text(encoding="utf-8"))
    doc = json.loads(documented_path.read_text(encoding="utf-8"))["models"]
    caps = []
    for c in cfg["cap"]:
        frac = float(c["fraction_of_documented"])
        models = {m: ModelCap(math.floor(doc[m]["rpd"] * frac), math.floor(doc[m]["tpd"] * frac))
                  for m in c["models"]}
        caps.append(Cap(c["name"], c.get("scope", "*"), models, c.get("total_requests")))
    floor = cfg.get("server_floor", {}).get("min_remaining_requests_fraction")
    return Budget(caps, float(cfg.get("window_hours", 24)), floor)


def estimate_prompt_tokens(payload: dict[str, Any]) -> int:
    """Over-estimate on purpose (~3 chars/token; English is ~4) plus per-message overhead.
    Settlement uses reported usage whenever it is larger."""
    msgs = [m for m in payload.get("messages") or [] if isinstance(m, dict)]
    chars = sum(len(m.get("content") or "") for m in msgs)
    return math.ceil(chars / 3) + 8 * len(msgs) + 16


def max_output_tokens(payload: dict[str, Any]) -> int:
    mx = payload.get("max_completion_tokens") or payload.get("max_tokens")
    if not isinstance(mx, int) or mx <= 0:
        raise BudgetExceeded("budgeted calls must set max_tokens (unbounded output cannot be reserved)")
    return mx


@dataclass
class Reservation:
    id: int
    model: str
    scope: str
    phase: str
    est_prompt_tokens: int
    max_tokens: int
    reserved_tokens: int


def _lock(path: Path):
    f = open(path, "a+b")
    try:
        if os.name == "nt":
            import msvcrt
            f.seek(0)
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        f.close()
        raise BudgetExceeded(f"another process holds {path.name}; only one budgeted process at a time")
    return f


def _unlock(f) -> None:
    try:
        if os.name == "nt":
            import msvcrt
            f.seek(0)
            msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
    finally:
        f.close()


def _int_header(rec: CallRecord, name: str) -> int | None:
    v = rec.header(name)
    try:
        return int(v) if v is not None else None
    except ValueError:
        return None


class Ledger:
    def __init__(self, budget: Budget, path: Path = DEFAULT_LEDGER,
                 clock: Callable[[], float] = time.time) -> None:
        self.budget = budget
        self.path = path
        self.clock = clock
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lockfile = _lock(path.with_suffix(".lock"))
        self.entries: list[dict[str, Any]] = []
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    self.entries.append(json.loads(line))
        self.inflight: dict[int, Reservation] = {}
        self._ids = itertools.count(1)
        self.halted: str | None = None

    # --------------------------------------------------------------- lifecycle
    def close(self) -> None:
        if self._lockfile is not None:
            _unlock(self._lockfile)
            self._lockfile = None

    def __enter__(self) -> "Ledger":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # --------------------------------------------------------------- accounting
    def _in_window(self) -> list[dict[str, Any]]:
        cutoff = self.clock() - self.budget.window_hours * 3600
        return [e for e in self.entries if e["ts"] >= cutoff]

    def usage(self, cap: Cap) -> dict[str, Any]:
        """Requests/tokens charged in the window under ``cap``, plus calls still in flight."""
        per = {m: {"requests": 0, "tokens": 0} for m in cap.models}
        total = 0
        for e in self._in_window():
            if not cap.applies(e["scope"]):
                continue
            p = per.setdefault(e["model"], {"requests": 0, "tokens": 0})
            p["requests"] += e["charged_requests"]
            p["tokens"] += e["charged_tokens"]
            total += e["charged_requests"]
        for r in self.inflight.values():
            if not cap.applies(r.scope):
                continue
            p = per.setdefault(r.model, {"requests": 0, "tokens": 0})
            p["requests"] += 1
            p["tokens"] += r.reserved_tokens
            total += 1
        return {"models": per, "total_requests": total}

    def reserve(self, payload: dict[str, Any], scope: str, phase: str) -> Reservation:
        if self.halted:
            raise BudgetExceeded(f"ledger halted: {self.halted}")
        model = payload.get("model")
        est = estimate_prompt_tokens(payload)
        mx = max_output_tokens(payload)
        need = est + mx
        applicable = [c for c in self.budget.caps if c.applies(scope)]
        if not applicable:
            raise BudgetExceeded(f"no cap covers scope {scope!r}; refusing an unbudgeted real call")
        for cap in applicable:
            if model not in cap.models:
                raise BudgetExceeded(f"model {model!r} is not allowed under cap {cap.name!r}")
            u = self.usage(cap)
            mc = cap.models[model]
            mu = u["models"][model]
            if mu["requests"] + 1 > mc.requests:
                raise BudgetExceeded(f"cap {cap.name!r}: {model} requests {mu['requests']}/{mc.requests}")
            if mu["tokens"] + need > mc.tokens:
                raise BudgetExceeded(f"cap {cap.name!r}: {model} tokens {mu['tokens']}+{need} > {mc.tokens}")
            if cap.total_requests is not None and u["total_requests"] + 1 > cap.total_requests:
                raise BudgetExceeded(f"cap {cap.name!r}: total requests {u['total_requests']}/{cap.total_requests}")
        r = Reservation(next(self._ids), model, scope, phase, est, mx, need)
        self.inflight[r.id] = r
        return r

    def release(self, r: Reservation) -> None:
        """Drop a reservation for a call that was never sent."""
        self.inflight.pop(r.id, None)

    def settle(self, r: Reservation, rec: CallRecord, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        self.inflight.pop(r.id, None)
        usage_total = (rec.usage or {}).get("total_tokens")
        if rec.outcome == CONNECT_ERROR:
            req, tok, basis = 0, 0, "never reached the server"
        elif rec.status == 429:
            req, tok, basis = 1, r.est_prompt_tokens, "429: prompt estimate only (assumption; probe P2 checks)"
        else:
            req = 1
            tok = max(r.reserved_tokens, int(usage_total or 0))
            basis = "max(prompt estimate + max_tokens, reported usage)"
        now = self.clock()
        entry = {
            "ts": now,
            "ts_iso": datetime.fromtimestamp(now, timezone.utc).isoformat(timespec="milliseconds"),
            "scope": r.scope,
            "phase": r.phase,
            "model": r.model,
            "call_id": rec.call_id,
            "source": rec.source,
            "outcome": rec.outcome,
            "status": rec.status,
            "est_prompt_tokens": r.est_prompt_tokens,
            "max_tokens": r.max_tokens,
            "reserved_tokens": r.reserved_tokens,
            "usage_total_tokens": usage_total,
            "charged_requests": req,
            "charged_tokens": tok,
            "charge_basis": basis,
            "hdr_remaining_requests": rec.header("x-ratelimit-remaining-requests"),
            "hdr_remaining_tokens": rec.header("x-ratelimit-remaining-tokens"),
            "extra": extra or {},
        }
        with self.path.open("a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        self.entries.append(entry)
        self._check_server_floor(rec)
        return entry

    def _check_server_floor(self, rec: CallRecord) -> None:
        frac = self.budget.min_remaining_requests_fraction
        if frac is None:
            return
        remaining = _int_header(rec, "x-ratelimit-remaining-requests")
        limit = _int_header(rec, "x-ratelimit-limit-requests")
        if remaining is not None and limit and remaining < frac * limit:
            self.halted = (f"server reports {remaining}/{limit} requests left today for {rec.model}, "
                           f"below the floor of {frac:.0%}; the shared key is being used elsewhere "
                           f"or the cap is wrong")

    def status(self) -> dict[str, Any]:
        out = {"window_hours": self.budget.window_hours, "halted": self.halted, "caps": {}}
        for cap in self.budget.caps:
            u = self.usage(cap)
            out["caps"][cap.name] = {
                "scope": cap.scope,
                "total_requests": f"{u['total_requests']}/{cap.total_requests}" if cap.total_requests else u["total_requests"],
                "models": {m: {"requests": f"{u['models'][m]['requests']}/{mc.requests}",
                               "tokens": f"{u['models'][m]['tokens']}/{mc.tokens}"}
                           for m, mc in cap.models.items()},
            }
        return out


class BudgetedClient:
    """The only way scripts should make real calls: reserve -> call -> settle."""

    def __init__(self, client: LLMClient, ledger: Ledger, scope: str,
                 raw_sink: Callable[[dict[str, Any]], None] | None = None) -> None:
        self.client = client
        self.ledger = ledger
        self.scope = scope
        self.raw_sink = raw_sink

    async def stream_chat(self, payload: dict[str, Any], *, phase: str, on_delta=None, timeouts=None,
                          extra: dict[str, Any] | None = None) -> CallRecord:
        r = self.ledger.reserve(payload, self.scope, phase)
        settled = False

        def sink(rec: CallRecord) -> None:
            nonlocal settled
            entry = self.ledger.settle(r, rec, extra)
            settled = True
            if self.raw_sink is not None:
                self.raw_sink({"phase": phase, "extra": extra or {}, "ledger": entry, "record": rec.to_dict()})

        try:
            return await self.client.stream_chat(payload, on_delta=on_delta, record_sink=sink, timeouts=timeouts)
        finally:
            if not settled:
                self.ledger.release(r)


def main() -> int:
    with Ledger(load_budget()) as led:
        print(json.dumps(led.status(), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
