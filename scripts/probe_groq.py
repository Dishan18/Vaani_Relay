"""Probe Groq's free-tier limits and latency, as planned in docs/probe_plan.md.

    .venv\\Scripts\\python scripts\\probe_groq.py --preflight            # plan vs caps, no calls
    .venv\\Scripts\\python scripts\\probe_groq.py --dry-run              # whole probe vs a local fake
    .venv\\Scripts\\python scripts\\probe_groq.py --dry-run --truth window,usage,pooled
    .venv\\Scripts\\python scripts\\probe_groq.py --real                 # real Groq calls (budgeted)
    .venv\\Scripts\\python scripts\\probe_groq.py --real --phases P0,P6  # e.g. latency again on a later day

Every call goes through relay.ledger (probe cap: 10% of each model's daily allowance, 220 requests
in total; account cap 90%). Every call, its exact payload (prompt text, prompt chars, settings), its
ledger entry and its full CallRecord are appended to data/probe/probe_raw.jsonl (real runs) or to
data/probe/dryrun/<truth>/probe_raw.jsonl (dry-runs, which use their own throwaway ledger). Waits,
skips, refusals and decisions are logged in the same file as ``"type": "event"`` lines.

Phase order (walls last): P0 -> P2 (+ P2-overflow) -> P1 per model, then P4, P6, P3.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import re
import secrets
import shutil
import sys
import threading
import time
from collections import deque
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from relay.ledger import (BudgetedClient, BudgetExceeded, Ledger, load_budget,  # noqa: E402
                          max_output_tokens, prompt_chars)
from relay.llm import CallRecord, Endpoint, LLMClient, Timeouts  # noqa: E402

M20, M120, MQ = "openai/gpt-oss-20b", "openai/gpt-oss-120b", "qwen/qwen3.8-27b"
MODELS = (M20, M120, MQ)
GPT_OSS = (M20, M120)
SCOPE = "probe"
REAL_RAW = ROOT / "data" / "probe" / "probe_raw.jsonl"
DRYRUN_DIR = ROOT / "data" / "probe" / "dryrun"
TIMEOUTS = Timeouts(first_visible_s=60, stall_s=30, deadline_s=120)   # measure tails, don't cut them
SOFT_RPM = 25            # our own per-model requests/min outside P3 (docs say 30; no header shows it)
OVERHEAD_MARGIN = 8      # added to the measured template overhead
TINY_TEXT = "Reply with exactly: OK"

# ----------------------------------------------------------------------------- call sizes
MAX_TOKENS = {"tiny": 16, "p0": 256, "p2a": 2000, "overflow": 8500, "router": 200, "draft": 500}

ROUTER_SYSTEM = (
    'You route questions for a QA system over three papers: "attention" = Attention Is All You Need '
    '(1706.03762v7); "moshi" = Moshi, a speech-text foundation model for real-time dialogue '
    '(2410.00037v1); "dynamic_chunking" = Dynamic Chunking for End-to-End Hierarchical Sequence '
    'Modeling, H-Net (2507.07955v2). Reply with JSON only: {"papers": [...], '
    '"kind": "fact|numeric|compare|unanswerable"}.')

DRAFT_SYSTEM = (
    "Answer the question using only the passages below. Cite every claim with the passage id in "
    "square brackets, e.g. [attention-p04-b012]. If the passages do not contain the answer, say "
    "so plainly. Be concise.")

# (question, papers it targets). Used round-robin by router and draft calls.
QUESTIONS = [
    ("What is the dimensionality of the keys and values per attention head in the base Transformer?", ["attention"]),
    ("How does Moshi model its own speech and the user's speech at the same time?", ["moshi"]),
    ("How does H-Net decide where to place chunk boundaries?", ["dynamic_chunking"]),
    ("Why are dot products scaled by one over the square root of d_k?", ["attention"]),
    ("What frame rate does the Mimi codec run at, and why does it matter for latency?", ["moshi"]),
    ("What is the smoothing module in dynamic chunking for?", ["dynamic_chunking"]),
    ("What BLEU does the big Transformer reach on WMT 2014 English-to-German?", ["attention"]),
    ("What is Inner Monologue in Moshi?", ["moshi"]),
    ("How does H-Net compare with a BPE-tokenized Transformer at matched compute?", ["dynamic_chunking"]),
    ("Compare how the Transformer and H-Net handle long sequences.", ["attention", "dynamic_chunking"]),
    ("Which of the papers report results on speech, and which on byte-level language modeling?",
     ["moshi", "dynamic_chunking"]),
    ("What learning-rate schedule does the Transformer use?", ["attention"]),
]


def load_passages(path: Path = ROOT / "data" / "blocks.jsonl", target_chars: int = 1100,
                  max_chars: int = 1300) -> list[dict]:
    """Body blocks merged within a section into ~target_chars passages (a stand-in for retrieval
    chunks; the real chunker comes in M1b)."""
    passages: list[dict] = []
    cur: dict | None = None
    with path.open(encoding="utf-8") as f:
        for line in f:
            b = json.loads(line)
            if b["kind"] != "body":
                cur = None
                continue
            if cur and cur["paper"] == b["paper"] and cur["section"] == b["section"] and len(cur["text"]) < target_chars:
                cur["text"] += " " + b["text"]
                continue
            cur = {"id": b["block_id"], "paper": b["paper"], "section": b["section"], "text": b["text"]}
            passages.append(cur)
    for p in passages:     # similar sizes keep the time-to-first-token vs prompt-size fit clean
        if len(p["text"]) > max_chars:
            cut = p["text"].rfind(". ", 0, max_chars)
            p["text"] = p["text"][: cut + 1 if cut > max_chars // 2 else max_chars]
    return passages


WORD = re.compile(r"[a-z0-9]+")
STOP = set("the a an of in on and or to is are for by with what how does do which why at as it its be "
           "and from that this their his her they we our".split())


def pick_passages(passages: list[dict], question: str, papers: list[str], n: int = 4) -> list[dict]:
    q = {w for w in WORD.findall(question.lower()) if w not in STOP}
    scored = []
    for i, p in enumerate(passages):
        if p["paper"] not in papers:
            continue
        words = set(WORD.findall(p["text"].lower()))
        scored.append((-len(q & words), i, p))
    scored.sort(key=lambda t: (t[0], t[1]))
    return [p for _, _, p in scored[:n]]


class Prompts:
    def __init__(self) -> None:
        self.passages = load_passages()
        self.qi = {"router": 0, "draft": 0}

    def messages(self, size: str, nonce: str) -> tuple[list[dict], dict]:
        tag = f"[{nonce}]"
        if size in ("tiny", "p0", "p2a", "overflow"):
            return [{"role": "user", "content": f"{tag} {TINY_TEXT}"}], {}
        i = self.qi[size] % len(QUESTIONS)
        self.qi[size] += 1
        q, papers = QUESTIONS[i]
        if size == "router":
            return ([{"role": "system", "content": f"{tag} {ROUTER_SYSTEM}"},
                     {"role": "user", "content": q}], {"question_index": i})
        ps = pick_passages(self.passages, q, papers)
        body = "QUESTION: " + q + "\n\nPASSAGES:\n" + "\n\n".join(f"[{p['id']}] ({p['section']}) {p['text']}" for p in ps)
        return ([{"role": "system", "content": f"{tag} {DRAFT_SYSTEM}"},
                 {"role": "user", "content": body}], {"question_index": i, "passage_ids": [p["id"] for p in ps]})


# ----------------------------------------------------------------------------- allocation
# The contract: the most calls each phase may make per model, by size. The runner refuses to exceed
# a row (a bug, not a budget event). Pre-flight sums worst-case charges from this table.
ALLOCATION: list[tuple[str, str, str, int]] = [
    ("P0", M20, "p0", 2), ("P0", M120, "p0", 2), ("P0", MQ, "p0", 3),
    ("P2", M20, "p2a", 1), ("P2", M20, "tiny", 1),
    ("P2", M120, "p2a", 1), ("P2", M120, "tiny", 1),
    ("P2", MQ, "p2a", 1), ("P2", MQ, "tiny", 1),
    ("P2-overflow", M20, "overflow", 1),
    ("P1", M20, "tiny", 6), ("P1", M120, "tiny", 3),
    ("P4", M20, "router", 15), ("P4", M120, "router", 13), ("P4", MQ, "router", 5),
    ("P6", M20, "draft", 1), ("P6", M120, "draft", 2), ("P6", MQ, "draft", 3),
    ("P6", M20, "router", 7), ("P6", M120, "router", 5), ("P6", MQ, "router", 12),
    ("P3", M20, "tiny", 50), ("P3", M120, "tiny", 50), ("P3", MQ, "tiny", 12),
]
# P1 polls follow each model's first P6 draft call: its prompt + answer (~1.5K tokens) is real usage,
# so the deficit can't be refunded away and refill (or its absence) is visible for ~10 s.
P1_OFFSETS = {M20: [1, 3, 6, 10, 15, 30], M120: [1, 4, 8]}
P4_STEPS = [(M20, 1), (M120, 1), (MQ, 1), (M20, 2), (M120, 4), (MQ, 4), (M20, 4), (M120, 8), (M20, 8)]
P6_DRAFTS = [M20, M120, MQ, M120, MQ, MQ]
P6_ROUTERS = {M20: 7, M120: 5, MQ: 12}
P3_BURST = 40              # stop issuing after 3 429s; 40 leaves room for a bucket refilling mid-burst
P3_STOP_AFTER_429 = 3
P3_RECOVERY_OFFSETS = [2, 4, 7, 11, 16, 22, 30, 40, 52, 65]
P3_BYSTANDER = 10
P3_BYSTANDER_RECOVERY = [30, 65]
PHASES = ("P0", "P2", "P2-overflow", "P1", "P4", "P6", "P3")   # P1 runs inside P6, after a draft
DEFAULT_SETTINGS = {M20: {"reasoning_effort": "low"}, M120: {"reasoning_effort": "low"},
                    MQ: {"reasoning_effort": "none"}}


class ProbeStop(Exception):
    pass


def parse_duration(s: str | None) -> float | None:
    """'7.66s' / '2m59.56s' / '1h2m3s' / '120ms' -> seconds."""
    if not s:
        return None
    s = s.strip()
    try:
        return float(s)
    except ValueError:
        pass
    total, matched = 0.0, False
    for num, unit in re.findall(r"([0-9]*\.?[0-9]+)(ms|h|m|s)", s):
        matched = True
        total += float(num) * {"ms": 0.001, "h": 3600, "m": 60, "s": 1}[unit]
    return total if matched else None


def hdr_int(rec: CallRecord, name: str) -> int | None:
    v = rec.header(name)
    try:
        return int(v) if v is not None else None
    except ValueError:
        return None


# ----------------------------------------------------------------------------- governor
@dataclass
class Obs:
    t: float            # perf_counter when the headers arrived
    remaining: int
    reset_s: float
    limit: int


class Governor:
    """Waits (never spends) until the last-seen headers say a step fits, and keeps our own RPM
    below SOFT_RPM because no header reports RPM."""

    def __init__(self, probe: "Probe", model: str) -> None:
        self.p = probe
        self.model = model
        self.obs: Obs | None = None
        self.starts: deque[float] = deque()
        self.last_start = -1e9

    def observe(self, rec: CallRecord, t_start: float) -> None:
        rem = hdr_int(rec, "x-ratelimit-remaining-tokens")
        lim = hdr_int(rec, "x-ratelimit-limit-tokens")
        if rem is None or lim is None or rec.headers_ms is None:
            return
        t = t_start + rec.headers_ms / 1000
        if self.obs is None or t >= self.obs.t:
            self.obs = Obs(t, rem, parse_duration(rec.header("x-ratelimit-reset-tokens")) or 0.0, lim)

    def started(self) -> None:
        now = time.perf_counter()
        self.starts.append(now)
        self.last_start = now

    async def gate(self, need_tokens: int, n_requests: int, phase: str) -> None:
        o = self.obs
        if o is not None:
            now = time.perf_counter()
            full_at = o.t + o.reset_s
            need = min(need_tokens, o.limit)
            if now < full_at and o.remaining < need:
                wait = min(full_at - now + 0.5, 65.0)
                self.p.event(phase, self.model, "wait_tokens", wait_s=round(wait, 2), need=need_tokens,
                             last_remaining=o.remaining, last_reset_s=o.reset_s)
                await asyncio.sleep(wait)
        while True:
            now = time.perf_counter()
            while self.starts and now - self.starts[0] > 60:
                self.starts.popleft()
            if len(self.starts) + n_requests <= SOFT_RPM:
                return
            wait = 60 - (now - self.starts[0]) + 0.2
            self.p.event(phase, self.model, "wait_rpm", wait_s=round(wait, 2), sent_last_60s=len(self.starts))
            await asyncio.sleep(wait)


# ----------------------------------------------------------------------------- probe
class Probe:
    def __init__(self, bc: BudgetedClient, raw_path: Path, mode: str, truth: str | None,
                 phases: tuple[str, ...]) -> None:
        self.bc = bc
        self.ledger: Ledger = bc.ledger
        self.raw_path = raw_path
        self.mode = mode
        self.truth = truth
        self.phases = phases
        self.run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + secrets.token_hex(3)
        self.prompts = Prompts()
        self.gov = {m: Governor(self, m) for m in MODELS}
        self.settings = {m: dict(v) for m, v in DEFAULT_SETTINGS.items()}
        self.used: dict[tuple[str, str, str], int] = {}
        self.allowed = {(ph, m, sz): n for ph, m, sz, n in ALLOCATION}
        self.admission_verdict: dict[str, dict] = {}
        self.raw_path.parent.mkdir(parents=True, exist_ok=True)
        self._raw = self.raw_path.open("a", encoding="utf-8", newline="\n")

    def close(self) -> None:
        self._raw.flush()
        os.fsync(self._raw.fileno())
        self._raw.close()

    # ---------------------------------------------------------------- logging
    def _write(self, obj: dict) -> None:
        self._raw.write(json.dumps(obj, ensure_ascii=False) + "\n")
        self._raw.flush()      # fsync only on close: it blocks the event loop for ~30-60 ms here

    def event(self, phase: str | None, model: str | None, event: str, **kw) -> None:
        self._write({"type": "event", "run_id": self.run_id, "mode": self.mode, "truth": self.truth,
                     "ts": time.time(), "phase": phase, "model": model, "event": event, **kw})
        brief = " ".join(f"{k}={v}" for k, v in kw.items() if not isinstance(v, (dict, list)))
        print(f"  [{phase or '-'}] {model or ''} {event} {brief}"[:200], flush=True)

    # ---------------------------------------------------------------- budget helpers
    def build(self, model: str, size: str, overrides: dict | None = None) -> tuple[dict, dict, str]:
        nonce = "n:" + secrets.token_hex(4)
        msgs, meta = self.prompts.messages(size, nonce)
        settings = {**self.settings[model], **(overrides or {})}
        settings = {k: v for k, v in settings.items() if v is not None}
        payload = {"model": model, "messages": msgs, "max_tokens": MAX_TOKENS[size], **settings}
        return payload, meta, nonce

    def p3_reserve(self, model: str) -> tuple[int, int]:
        """Worst-case requests/tokens P3 still needs for ``model`` (protected from earlier phases)."""
        if "P3" not in self.phases:
            return 0, 0
        left = self.allowed.get(("P3", model, "tiny"), 0) - self.used.get(("P3", model, "tiny"), 0)
        payload, _, _ = self.build(model, "tiny")
        return left, left * self.ledger.worst_charge(payload)

    def p3_total_requests(self) -> int:
        return sum(self.p3_reserve(m)[0] for m in MODELS)

    # ---------------------------------------------------------------- one call
    async def call(self, phase: str, model: str, size: str, *, variant: str = "main",
                   overrides: dict | None = None, step: dict | None = None, gate: bool = True) -> CallRecord | None:
        key = (phase, model, size)
        if self.used.get(key, 0) >= self.allowed.get(key, 0):
            raise RuntimeError(f"allocation exceeded for {key}: the phase code has a bug")
        self.used[key] = self.used.get(key, 0) + 1
        payload, meta, nonce = self.build(model, size, overrides)
        worst = self.ledger.worst_charge(payload)
        if phase != "P3":
            req_left, tok_left = self.p3_reserve(model)
            h = self.ledger.headroom(SCOPE, model)
            if (h["tokens"] - worst < tok_left or h["requests"] - 1 < req_left
                    or h["total_requests"] - 1 < self.p3_total_requests()):
                self.event(phase, model, "skip_protect_p3", size=size, variant=variant, worst=worst,
                           headroom=h, p3_needs={"requests": req_left, "tokens": tok_left})
                return None
        if gate:
            await self.gov[model].gate(worst, 1, phase)
        line_meta = {"type": "call", "run_id": self.run_id, "mode": self.mode, "truth": self.truth,
                     "phase": phase, "model": model, "size": size, "variant": variant, "nonce": nonce,
                     "step": step or {}, "prompt_meta": meta, "settings": {k: v for k, v in payload.items()
                                                                          if k not in ("messages", "model")},
                     "payload": payload, "prompt_chars": prompt_chars(payload),
                     "prompt_chars_by_message": [len(m["content"]) for m in payload["messages"]],
                     "est_prompt_tokens": self.ledger.estimate(payload),
                     "prompt_overhead_used": self.ledger.overhead_for(model)}

        def sink(d: dict) -> None:
            self._write({**line_meta, "t_start_unix": t_unix, "ledger": d["ledger"], "record": d["record"]})

        bc = BudgetedClient(self.bc.client, self.ledger, SCOPE, sink)
        t_unix, t0 = time.time(), time.perf_counter()
        self.gov[model].started()
        try:
            rec = await bc.stream_chat(payload, phase=phase, timeouts=TIMEOUTS,
                                       extra={"run_id": self.run_id, "size": size, "variant": variant})
        except BudgetExceeded as e:
            self.event(phase, model, "refused_by_ledger", size=size, variant=variant, reason=str(e))
            if self.ledger.halted:
                raise ProbeStop(f"ledger halted: {self.ledger.halted}")
            return None
        self.gov[model].observe(rec, t0)
        u = rec.usage or {}
        print(f"  [{phase}] {model:<20} {size:<8} {variant:<22} {rec.outcome:<14} {rec.status} "
              f"hdr={rec.headers_ms and round(rec.headers_ms)}ms vis={rec.first_visible_ms and round(rec.first_visible_ms)}ms "
              f"tot={round(rec.total_ms or 0)}ms rem_tok={rec.header('x-ratelimit-remaining-tokens')} "
              f"usage={u.get('prompt_tokens')}/{u.get('completion_tokens')}", flush=True)
        if rec.status in (401, 403, 404):
            raise ProbeStop(f"HTTP {rec.status} from {model}: {rec.error_body}")
        if rec.status == 400 and not (phase == "P0" and variant != "main"):
            raise ProbeStop(f"HTTP 400 on the main settings for {model}: {rec.error_body}")
        if u.get("prompt_tokens") is not None and u["prompt_tokens"] > self.ledger.estimate(payload):
            self.event(phase, model, "estimate_violation", prompt_tokens=u["prompt_tokens"],
                       estimate=self.ledger.estimate(payload))
        return rec

    # ---------------------------------------------------------------- phases
    async def p0(self, m: str) -> list[CallRecord]:
        recs: list[CallRecord] = []
        if m in GPT_OSS:
            r = await self.call("P0", m, "p0", variant="main")
            if r is None:
                raise ProbeStop(f"P0 main call for {m} was not sent")
            if r.status == 400:
                raise ProbeStop(f"400 on main settings for {m}: {r.error_body}")
            recs.append(r)
            r2 = await self.call("P0", m, "p0", variant="include_reasoning_false",
                                 overrides={"include_reasoning": False})
            recs += [r2] if r2 else []
        else:
            variants = [("effort_none", {"reasoning_effort": "none"}),
                        ("effort_default_parsed", {"reasoning_effort": "default", "reasoning_format": "parsed"}),
                        ("effort_low_parsed", {"reasoning_effort": "low", "reasoning_format": "parsed"})]
            ok = {}
            for name, ov in variants:
                self.settings[m] = {}
                r = await self.call("P0", m, "p0", variant=name, overrides=ov)
                if r is not None:
                    recs.append(r)
                    ok[name] = (r.status == 200 and r.outcome == "ok", ov)
            for name in ("effort_none", "effort_default_parsed"):   # lowest accepted effort wins
                if ok.get(name, (False,))[0]:
                    self.settings[m] = {"reasoning_effort": ok[name][1]["reasoning_effort"]}
                    break
            else:
                raise ProbeStop(f"no accepted low-reasoning setting for {m}")
        if not any(r.status == 200 and r.usage for r in recs):
            raise ProbeStop(f"no usage reported by {m} on any successful P0 call; the analysis needs it")
        self.event("P0", m, "settings", settings=self.settings[m])
        # measured template overhead -> tighter, still-conservative reservations
        over = []
        for r in recs:
            pt = (r.usage or {}).get("prompt_tokens")
            if r.status == 200 and pt is not None:
                chars = r.prompt_chars
                over.append(pt - (math.ceil(chars / 3) + 8))
        if over:
            ov = max(0, max(over)) + OVERHEAD_MARGIN
            self.ledger.set_prompt_overhead(m, ov)
            self.event("P0", m, "prompt_overhead", measured=over, used=ov, configured=self.ledger.budget.prompt_overhead_tokens)
        return recs

    async def p2(self, m: str, baseline: CallRecord | None) -> CallRecord | None:
        a = await self.call("P2", m, "p2a", variant="A")
        b = await self.call("P2", m, "tiny", variant="B", gate=False)
        base = hdr_int(baseline, "x-ratelimit-remaining-tokens") if baseline else None
        rem_a = hdr_int(a, "x-ratelimit-remaining-tokens") if a else None
        verdict = {"baseline_remaining": base, "A_remaining": rem_a,
                   "B_remaining": hdr_int(b, "x-ratelimit-remaining-tokens") if b else None}
        if base is not None and rem_a is not None:
            drop = base - rem_a
            verdict["drop_at_A"] = drop
            verdict["admission_charges_max_tokens"] = drop >= MAX_TOKENS["p2a"] / 2
        self.admission_verdict[m] = verdict
        self.event("P2", m, "admission_quick_verdict", **verdict)
        return a

    async def p1(self, m: str, t_end: float, after: str) -> None:
        for off in P1_OFFSETS.get(m, []):
            wait = t_end + off - time.perf_counter()
            if wait > 0:
                await asyncio.sleep(wait)
            await self.call("P1", m, "tiny", variant="poll", step={"offset_s": off, "after": after,
                            "actual_offset_s": round(time.perf_counter() - t_end, 3)}, gate=False)

    async def p2_overflow(self) -> None:
        v = self.admission_verdict.get(M20, {})
        if not v.get("admission_charges_max_tokens"):
            self.event("P2-overflow", M20, "skipped", reason="P2 did not show admission charging, so an "
                       "8500-token call could really generate 8500 tokens", verdict=v)
            return
        await self.call("P2-overflow", M20, "overflow", variant="max_tokens_over_tpm")

    async def p4(self) -> None:
        last: dict[str, float] = {}
        for i, (m, k) in enumerate(P4_STEPS):
            if m in last:
                wait = last[m] + 20 - time.perf_counter()
                if wait > 0:
                    await asyncio.sleep(wait)
            payload, _, _ = self.build(m, "router")
            await self.gov[m].gate(k * self.ledger.worst_charge(payload), k, "P4")
            last[m] = time.perf_counter()
            step_id = f"P4-{i}"
            await asyncio.gather(*[self.call("P4", m, "router", variant=f"k{k}", gate=False,
                                             step={"k": k, "step_id": step_id, "slot": j}) for j in range(k)])

    async def p6(self) -> None:
        seq: list[tuple[str, str]] = [(m, "draft") for m in P6_DRAFTS]
        left = dict(P6_ROUTERS)
        while any(left.values()):
            for m in MODELS:
                if left[m]:
                    seq.append((m, "router"))
                    left[m] -= 1
        last = -1e9
        polled: set[str] = set()
        for i, (m, size) in enumerate(seq):
            wait = last + 3 - time.perf_counter()
            if wait > 0:
                await asyncio.sleep(wait)
            last = time.perf_counter()
            r = await self.call("P6", m, size, variant="sequential", step={"seq": i})
            if (size == "draft" and r is not None and r.status == 200 and "P1" in self.phases
                    and m in P1_OFFSETS and m not in polled):
                polled.add(m)
                await self.p1(m, time.perf_counter(), after=f"P6 draft {r.call_id}")
                last = time.perf_counter()

    async def p3(self) -> None:
        # clean slate: >= 65 s since the last call to any model, then start at :15 past the minute
        since = max(g.last_start for g in self.gov.values())
        wait = since + 65 - time.perf_counter()
        if wait > 0:
            self.event("P3", None, "wait_clean_slate", wait_s=round(wait, 1))
            await asyncio.sleep(wait)
        sec = datetime.now(timezone.utc).second + datetime.now(timezone.utc).microsecond / 1e6
        align = (15 - sec) % 60
        self.event("P3", None, "wait_align", wait_s=round(align, 1), start_second=15)
        await asyncio.sleep(align)
        t_burst = time.perf_counter()
        self.event("P3", None, "burst_start", wall=datetime.now(timezone.utc).isoformat(timespec="milliseconds"))

        async def burst(m: str) -> None:
            st = {"sent": 0, "n429": 0, "first_429": None}

            async def worker(w: int) -> None:
                while st["sent"] < P3_BURST and st["n429"] < P3_STOP_AFTER_429:
                    st["sent"] += 1
                    idx = st["sent"]
                    r = await self.call("P3", m, "tiny", variant="burst", gate=False,
                                        step={"i": idx, "worker": w, "t_since_burst_s": round(time.perf_counter() - t_burst, 3)})
                    if r is not None and r.status == 429:
                        st["n429"] += 1
                        if st["first_429"] is None:
                            st["first_429"] = time.perf_counter()

            await asyncio.gather(*[worker(w) for w in range(4)])
            self.event("P3", m, "burst_done", sent=st["sent"], n429=st["n429"],
                       first_429_after_s=None if st["first_429"] is None else round(st["first_429"] - t_burst, 3))
            if st["first_429"] is None:
                return
            ok_run = 0
            for off in P3_RECOVERY_OFFSETS:
                wait = st["first_429"] + off - time.perf_counter()
                if wait > 0:
                    await asyncio.sleep(wait)
                r = await self.call("P3", m, "tiny", variant="recovery", gate=False,
                                    step={"offset_s": off, "actual_offset_s": round(time.perf_counter() - st["first_429"], 3),
                                          "wall_second": datetime.now(timezone.utc).second})
                ok_run = ok_run + 1 if (r is not None and r.status == 200) else 0
                if ok_run >= 3:
                    break

        async def bystander() -> None:
            first_429 = None
            pending = []
            for i in range(P3_BYSTANDER):
                async def one(i=i):
                    return await self.call("P3", MQ, "tiny", variant="bystander", gate=False,
                                           step={"i": i + 1, "t_since_burst_s": round(time.perf_counter() - t_burst, 3)})
                pending.append(asyncio.create_task(one()))
                if len(pending) >= 2:
                    done = await pending.pop(0)
                    if done is not None and done.status == 429 and first_429 is None:
                        first_429 = time.perf_counter()
                await asyncio.sleep(0.4)
            for t in pending:
                r = await t
                if r is not None and r.status == 429 and first_429 is None:
                    first_429 = time.perf_counter()
            if first_429 is None:
                return
            for off in P3_BYSTANDER_RECOVERY:
                wait = first_429 + off - time.perf_counter()
                if wait > 0:
                    await asyncio.sleep(wait)
                await self.call("P3", MQ, "tiny", variant="bystander_recovery", gate=False, step={"offset_s": off})

        await asyncio.gather(burst(M20), burst(M120), bystander())

    # ---------------------------------------------------------------- driver
    def restore_previous_settings(self) -> None:
        """For runs without P0 (e.g. P6 again on a later day): reuse the last real run's decisions."""
        if not self.raw_path.exists():
            return
        with self.raw_path.open(encoding="utf-8") as f:
            for line in f:
                o = json.loads(line)
                if o.get("type") == "event" and o.get("mode") == self.mode:
                    if o["event"] == "settings":
                        self.settings[o["model"]] = o["settings"]
                    elif o["event"] == "prompt_overhead":
                        self.ledger.set_prompt_overhead(o["model"], o["used"])

    async def run(self) -> int:
        self.event(None, None, "run_start", phases=list(self.phases), allocation=ALLOCATION,
                   timeouts=vars(TIMEOUTS) if hasattr(TIMEOUTS, "__dict__") else str(TIMEOUTS),
                   soft_rpm=SOFT_RPM, overhead_default=self.ledger.budget.prompt_overhead_tokens,
                   ledger_status=self.ledger.status())
        if "P0" not in self.phases:
            self.restore_previous_settings()
        rc = 0
        try:
            last_rec: dict[str, CallRecord | None] = {}
            for m in MODELS:
                if "P0" in self.phases:
                    recs = await self.p0(m)
                    last_rec[m] = next((r for r in reversed(recs) if r.status == 200), None)
                if "P2" in self.phases:
                    await self.p2(m, last_rec.get(m))
                    if m == M20 and "P2-overflow" in self.phases:
                        await self.p2_overflow()
            if "P4" in self.phases:
                await self.p4()
            if "P6" in self.phases:
                await self.p6()
            if "P3" in self.phases:
                await self.p3()
        except ProbeStop as e:
            self.event(None, None, "stop", reason=str(e))
            rc = 2
        self.event(None, None, "run_end", rc=rc, ledger_status=self.ledger.status(),
                   calls_by_phase={f"{k[0]}|{k[1]}|{k[2]}": v for k, v in self.used.items()})
        return rc


# ----------------------------------------------------------------------------- pre-flight
def preflight(ledger: Ledger, phases: tuple[str, ...], overhead_assumed: int | None = None) -> tuple[bool, str]:
    """Worst-case charge of every planned call vs what the caps still allow. Pure arithmetic."""
    prompts = Prompts()
    rows: dict[str, dict[str, int]] = {m: {"requests": 0, "tokens": 0, "flex_tokens": 0} for m in MODELS}
    saved = dict(ledger.prompt_overhead)
    if overhead_assumed is not None:
        for m in MODELS:
            ledger.set_prompt_overhead(m, overhead_assumed)
    lines = []
    for ph, m, size, n in ALLOCATION:
        if ph not in phases:
            continue
        msgs, _ = prompts.messages(size, "n:00000000")
        payload = {"model": m, "messages": msgs, "max_tokens": MAX_TOKENS[size], **DEFAULT_SETTINGS[m]}
        est = ledger.estimate(payload)
        worst = est + max_output_tokens(payload)
        if size == "overflow":
            charge = est        # a 4xx rejection is charged its prompt estimate; needs `worst` headroom
        else:
            charge = worst
        rows[m]["requests"] += n
        if ph == "P6" and size == "router":
            rows[m]["flex_tokens"] += n * charge     # P6 routers stop when only P3's reserve is left
        else:
            rows[m]["tokens"] += n * charge
        lines.append(f"  {ph:<12} {m:<20} {size:<8} x{n:<3} est_prompt={est:<5} worst/call={worst:<5} "
                     f"charged<={n * charge}")
    ledger.prompt_overhead = saved
    ok = True
    total = sum(r["requests"] for r in rows.values())
    out = [f"pre-flight (template overhead assumed: {overhead_assumed if overhead_assumed is not None else ledger.budget.prompt_overhead_tokens} tokens/call)"]
    out += lines
    for m in MODELS:
        h = ledger.headroom(SCOPE, m)
        r = rows[m]
        fits_fixed = r["tokens"] <= h["tokens"] and r["requests"] <= h["requests"]
        fits_all = r["tokens"] + r["flex_tokens"] <= h["tokens"]
        ok &= fits_fixed
        out.append(f"  {m:<20} requests {r['requests']:>3}/{h['requests']:<4} fixed tokens {r['tokens']:>6} "
                   f"+ flexible P6 routers {r['flex_tokens']:>5} vs headroom {h['tokens']:>6}: "
                   f"{'fits' if fits_all else ('fixed fits, P6 routers trimmed' if fits_fixed else 'DOES NOT FIT')}")
    tot_h = min(ledger.headroom(SCOPE, m)["total_requests"] for m in MODELS)
    ok &= total <= tot_h
    out.append(f"  total requests {total}/{tot_h}: {'fits' if total <= tot_h else 'DOES NOT FIT'}")
    return ok, "\n".join(out)


# ----------------------------------------------------------------------------- main
@contextmanager
def fake_in_thread(truth):
    """Run the fake on its own event loop in a thread, so its work never delays our timestamps."""
    from relay.fake_groq import FakeGroq
    ready = threading.Event()
    box: dict[str, Any] = {}

    def runner() -> None:
        async def serve() -> None:
            async with FakeGroq(truth) as fake:
                box["fake"], box["loop"], box["stop"] = fake, asyncio.get_running_loop(), asyncio.Event()
                ready.set()
                await box["stop"].wait()
        asyncio.run(serve())

    th = threading.Thread(target=runner, daemon=True)
    th.start()
    ready.wait(10)
    try:
        yield box["fake"]
    finally:
        box["loop"].call_soon_threadsafe(box["stop"].set)
        th.join(10)


def parse_truth(s: str):
    from relay.fake_groq import Truth
    window, charge, pool = (s.split(",") + ["", "", ""])[:3]
    return Truth(window=window or "bucket", charge=charge or "admission_refund", pooled=(pool == "pooled"))


async def amain(args) -> int:
    phases = tuple(p for p in PHASES if p in set(args.phases.split(","))) if args.phases else PHASES
    budget = load_budget()
    if args.preflight:
        with Ledger(budget) as led:
            for assumed in (None, 64):
                ok, text = preflight(led, phases, assumed)
                print(text)
        return 0
    if args.dry_run:
        truth = parse_truth(args.truth)
        out = DRYRUN_DIR / truth.label()
        if out.exists():
            shutil.rmtree(out)          # simulated output, regenerated on every dry-run
        out.mkdir(parents=True)
        with fake_in_thread(truth) as fake:
            with Ledger(budget, out / "ledger.jsonl") as led:
                ok, text = preflight(led, phases)
                print(text)
                ok2, text2 = preflight(led, phases, 64)
                print(text2)
                client = LLMClient(Endpoint(fake.base_url, source="fake"), TIMEOUTS)
                probe = Probe(BudgetedClient(client, led, SCOPE), out / "probe_raw.jsonl", "dry-run",
                              truth.label(), phases)
                try:
                    probe.event(None, None, "preflight", ok=ok, text=text, ok_at_64=ok2, text_at_64=text2)
                    rc = await probe.run()
                finally:
                    probe.close()
                print(json.dumps(led.status(), indent=2))
                print(f"raw: {out / 'probe_raw.jsonl'}")
                return rc
    if not args.real:
        print("nothing to do: pass --preflight, --dry-run or --real")
        return 1
    from relay.real import open_real
    with ExitStack() as stack:
        bc = stack.enter_context(open_real(SCOPE, allow_real=True, timeouts=TIMEOUTS))
        ok, text = preflight(bc.ledger, phases)
        ok2, text2 = preflight(bc.ledger, phases, 64)
        print(text)
        print(text2)
        if not (ok or ok2):
            print("pre-flight failed: the planned calls cannot fit the remaining caps; nothing sent")
            return 3
        probe = Probe(bc, REAL_RAW, "real", None, phases)
        try:
            probe.event(None, None, "preflight", ok=ok, text=text, ok_at_64=ok2, text_at_64=text2)
            rc = await probe.run()
        finally:
            probe.close()
        print(json.dumps(bc.ledger.status(), indent=2))
        print(f"raw: {REAL_RAW}")
        return rc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--preflight", action="store_true", help="print the plan vs the caps; no calls")
    g.add_argument("--dry-run", action="store_true", help="run every phase against a local fake Groq")
    g.add_argument("--real", action="store_true", help="make real, budgeted Groq calls")
    ap.add_argument("--truth", default="bucket,admission_refund,per_model",
                    help="dry-run only: window|bucket, admission_refund|admission|usage, per_model|pooled")
    ap.add_argument("--phases", default="", help=f"comma list from {','.join(PHASES)} (default: all)")
    args = ap.parse_args()
    return asyncio.run(amain(args))


if __name__ == "__main__":
    sys.exit(main())
