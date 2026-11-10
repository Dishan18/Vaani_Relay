"""The stand-in: an OpenAI-compatible streaming chat server that behaves like Groq as measured.

    .venv\\Scripts\\python -m standin.server serve [--port 8790] [--replay results/raw/answers] [--seed 1]
                                                [--p503 0] [--stall-rate 0] [--stall-s 8] [--slow-under-load]
                                                [--limit-scale 1]
    .venv\\Scripts\\python -m standin.server ctl [--port 8790] show
    .venv\\Scripts\\python -m standin.server ctl [--port 8790] set p503=0.05 stall_rate=0.02 limit_scale=0.5
    .venv\\Scripts\\python -m standin.server ctl [--port 8790] exhaust-daily [MODEL|all]
    .venv\\Scripts\\python -m standin.server ctl [--port 8790] reset

No Groq calls. Every parameter comes from data/calibration.json (scripts/calibrate_standin.py,
fitted on the real probe), so the client code path is the same as for Groq: base URL
http://127.0.0.1:<port>/openai/v1.

Behaviour, and where each piece comes from
  * Limits: per-model refilling buckets (RPM, TPM, RPD, TPD, and qwen's OTPM, which no header shows
    and a real 429 revealed: output tokens only, holding min(max_tokens, 184) at admission), refilled continuously at
    capacity/period per second (probe findings Q1, Q3). Admission takes 1 request from RPM and RPD
    and prompt + min(max_tokens, hold_cap) tokens from TPM and TPD; at the end the hold is
    released down to the real completion (Q2). A 429 takes nothing (Q5).
  * Headers: x-ratelimit-limit/remaining/reset-requests (= RPD) and -tokens (= TPM) on every 200,
    values as of admission, reset = deficit / refill rate in Groq's duration format. A 429 carries
    retry-after = ceil(this request's wait) and the same headers (an RPM 429 from Groq carried only
    the request ones, which is what the stand-in does for request limits).
  * 429 body: Groq's exact message: "Rate limit reached for model `m` in organization `org_...`
    service tier `on_demand` on <what> (<RPM|RPD|TPM|TPD>): Limit L, Used U, Requested R. Please try
    again in <wait>. Need more tokens? ...", type requests|tokens, code rate_limit_exceeded. The
    limit named is the one with the longest wait for this request.
  * TTFB: lognormal fitted to each model's server-side p50/p95; usage.queue_time is derived from
    it with the measured ttfb = intercept + slope * queue_time line.
  * Output: gpt-oss streams `reasoning` deltas first (token counts sampled from the probe), then
    content, at the measured tokens/s, in ~20 ms slices (Windows timers tick at ~15.6 ms; Groq's
    own chunks are 1-3 ms apart). prompt_tokens from the fitted a + b*chars + c*messages.
  * Text: with --replay, a request that matches a recorded real call (the router for the same
    question; the drafter for the same question and the same cited sources) gets the real text
    back, and the response says x-standin-replay: hit. Anything else gets a shape-only placeholder
    (x-standin-replay: miss), which must never be graded as an answer.
Switches (CLI flags at start, or `ctl set` / POST /control at runtime)
  p503             share of requests answered 503 before admission (nothing charged)
  stall_rate       share of admitted requests that wait stall_s (8 s) before the first byte
  slow_under_load  TTFB and decode slow by (1 + load_alpha * (in flight - 1)); off by default
                   because the probe saw no consistent slowdown up to 8 in flight
  limit_scale      multiplies every bucket's capacity and refill rate (0.5 = "per-minute limit halved"
                   and every other limit halved too; use limit_scale_only to pick buckets)
  exhaust_daily    empties RPD and TPD now for a model (or all); they then refill at the daily rate

Where the stand-in is least like Groq (most important first)
  1. Daily limits. TPD (200,000) and qwen's RPM are documented only: never measured. Whether a day
     is a fixed window or a refilling bucket, and what the daily 429 says, are unknown; the stand-in
     assumes a refilling bucket and reuses the per-minute message format. The "daily budget
     exhausted" scenario rests on this.
  2. Answer text. Replay covers only recorded requests; any other prompt gets placeholder text, so
     quality can only ever come from real calls (or exact replays of them).
  3. Latency tail. The lognormal is fitted to n = 13-21 sequential calls per model; the real p95 is
     the sample max at that n, and 120b's tail (6.8 s, all server queue) may be rarer or worse.
     Latency is independent per request; Groq's queue is shared and bursts correlate.
  4. Token 429s. The probe never hit a TPM limit, so the token 429's headers and wording are copied
     from the RPM one. Request too large returns 413 (unmeasured).
  5. Load. Off by default (no slowdown measured up to 8 in flight); the slowdown factor when on is
     an assumption. There is no concurrency limit, as measured up to 8; above 8 is unknown.
  6. Failure shapes. 503 bodies and stalls are invented shapes; Groq's real failures were not
     observed in the probe. A client that leaves mid-stream is charged its admission hold (how
     Groq bills cancelled streams is unmeasured, H10).
  7. Transport. Plain HTTP on localhost: no TLS (Groq's connect was ~90 ms), no Cloudflare, no
     HTTP/2. Real wall-clock reset headers depend on our clock only.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
import re
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CALIBRATION = ROOT / "data" / "calibration.json"
WHAT = {"RPM": "requests per minute", "RPD": "requests per day", "TPM": "tokens per minute",
        "TPD": "tokens per day", "OTPM": "output tokens per minute", "ITPM": "input tokens per minute"}


def fmt_duration(s: float) -> str:
    """Groq's style: 2s, 2.992s, 1m26.4s, 1h17m45.6s, 450ms."""
    if s < 1:
        return f"{max(0, round(s * 1000))}ms"
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    sec = round(sec, 3)
    out = ""
    if h >= 1:
        out += f"{int(h)}h"
    if m >= 1 or h >= 1:
        out += f"{int(m)}m"
    txt = f"{sec:.3f}".rstrip("0").rstrip(".")
    return out + f"{txt}s"


class Bucket:
    def __init__(self, capacity: float, period_s: float, unit: str, clock) -> None:
        self.base_capacity, self.period_s, self.unit, self.clock = float(capacity), float(period_s), unit, clock
        self.scale = 1.0
        self.level = self.capacity
        self.t = clock()

    @property
    def capacity(self) -> float:
        return self.base_capacity * self.scale

    @property
    def rate(self) -> float:
        return self.capacity / self.period_s

    def sync(self) -> float:
        now = self.clock()
        self.level = min(self.capacity, self.level + (now - self.t) * self.rate)
        self.t = now
        return self.level

    def set_scale(self, scale: float) -> None:
        self.sync()
        frac = self.level / self.capacity if self.capacity else 1.0
        self.scale = scale
        self.level = frac * self.capacity

    def wait_for(self, n: float) -> float:
        short = n - self.sync()
        return 0.0 if short <= 1e-9 else short / self.rate

    def take(self, n: float) -> None:
        self.sync()
        self.level -= n

    def give(self, n: float) -> None:
        self.sync()
        self.level = min(self.capacity, self.level + n)

    def reset_s(self) -> float:
        return max(0.0, (self.capacity - self.sync()) / self.rate)


@dataclass
class Switches:
    p503: float = 0.0
    stall_rate: float = 0.0
    stall_s: float = 8.0
    slow_under_load: bool = False
    load_alpha: float = 0.15
    limit_scale: float = 1.0
    limit_scale_only: str = ""        # comma list of bucket names; empty = all


class Model:
    def __init__(self, name: str, cal: dict, clock) -> None:
        self.name, self.cal = name, cal
        self.buckets = {k: Bucket(b["capacity"], b["period_s"], b["unit"], clock) for k, b in cal["buckets"].items()}
        self.hold_cap = float(cal.get("hold_cap") or 0.0)
        self.out_hold_cap = float(cal.get("out_hold_cap") or self.hold_cap)

    def prompt_tokens(self, messages: list[dict]) -> int:
        f = self.cal.get("prompt_tokens_fit")
        chars = sum(len(m.get("content") or "") for m in messages)
        if not f:
            return math.ceil(chars / 4) + 8 * len(messages)
        return max(1, round(f["a"] + f["b_per_char"] * chars + f["c_per_message"] * len(messages)))


# ----------------------------------------------------------------------------- replay
_Q = re.compile(r"^Question: (.*?)\n\nSOURCES:\n", re.S)
_CITE = re.compile(r"^\[S\d+\] \([^;]*; ([^)]*)\)$", re.M)


class Replay:
    """Real texts from results/raw/answers: router replies by question, drafts by (question, cited
    sources). Built only from real (source=real) answer rows."""

    def __init__(self, raw_dir: Path | None) -> None:
        self.router: dict[str, str] = {}
        self.draft: dict[tuple, str] = {}
        if raw_dir is None or not raw_dir.exists():
            return
        cites: dict[int, dict[str, str]] = {}          # piece size -> pid -> citation (refusals list pids only)

        def cite_of(size: int, pid: str) -> str | None:
            if size not in cites:
                from relay.retrieve import Retriever
                cites[size] = {p.pid: p.cite() for p in Retriever.from_blocks(size).pieces}
            return cites[size].get(pid)
        for p in sorted(raw_dir.glob("*.jsonl")):
            if p.name.endswith(".calls.jsonl"):
                continue
            for line in p.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                o = json.loads(line)
                if o.get("type") != "answer" or o.get("source") != "real":
                    continue
                rep = o["reply"]
                q = rep["question"]
                rt = (rep.get("checks") or {}).get("router")
                if rt and "failed" not in rt:
                    query = rt.get("query") or ""
                    alt = rt.get("alt_query") or ""
                    strip = lambda s: s[: -len(q) - 1] if s.endswith(" " + q) else s
                    self.router[q] = json.dumps({"papers": rt.get("papers") or [], "query": strip(query),
                                                 "alt_query": strip(alt)})
                if rep["kind"] == "answer" and rep.get("sources"):
                    self.draft[(q, tuple(s["cite"] for s in rep["sources"]))] = rep["text"]
                elif rep["kind"] == "refusal" and rep["text"].startswith("The papers don't say"):
                    # the drafter refused: key it by the sources it was shown, like any draft
                    rest = rep["text"][len("The papers don't say."):].strip()
                    pids = ((rep.get("checks") or {}).get("retrieval") or {}).get("pids")
                    size = (o.get("setting_def") or {}).get("piece_chars")
                    if pids and size:
                        cs = tuple(cite_of(size, pid) for pid in pids)
                        if all(cs):
                            self.draft[(q, cs)] = f"NOT_IN_PAPERS: {rest or 'not covered.'}"

    def lookup(self, messages: list[dict]) -> str | None:
        sysmsg = messages[0].get("content") or "" if messages else ""
        user = messages[-1].get("content") or "" if messages else ""
        if sysmsg.startswith("You route"):
            return self.router.get(user)
        m = _Q.match(user)
        if m and "failed the check" not in user:
            return self.draft.get((m.group(1), tuple(_CITE.findall(user))))
        return None


def placeholder(messages: list[dict]) -> str:
    """Shape-only text for a request with no recorded reply. Never a gradable answer."""
    sysmsg = messages[0].get("content") or "" if messages else ""
    user = messages[-1].get("content") or "" if messages else ""
    if sysmsg.startswith("You route"):
        ql = user.lower()
        paper = "moshi" if "moshi" in ql else "dynamic_chunking" if ("h-net" in ql or "chunk" in ql) else "attention"
        return json.dumps({"papers": [paper], "query": user, "alt_query": user})
    if sysmsg.startswith("You check"):
        return '{"unsupported": []}'
    m = re.search(r"\[S1\] \([^\n]*\)\n(.*?)(?:\n\n\[S2\] \(|\Z)", user, re.S)
    if m:
        # a real sentence (not a paragraph label like "Metrics."), so the shape is a plausible answer
        sents = re.split(r"(?<=[.!?])\s+", m.group(1).replace("\n", " "))
        good = [s for s in sents if len(re.findall(r"[A-Za-z]{2,}", s)) >= 6]
        return f"{(good or sents)[0]} [S1]"
    return "OK"


# ----------------------------------------------------------------------------- server
class StandIn:
    def __init__(self, calibration: dict, switches: Switches | None = None, replay: Replay | None = None,
                 seed: int | None = None, clock=time.monotonic) -> None:
        self.clock = clock
        self.models = {m: Model(m, c, clock) for m, c in calibration["models"].items()}
        self.sw = switches or Switches()
        self.replay = replay or Replay(None)
        self.rng = random.Random(seed)
        self.inflight = 0
        self.stats = {"requests": 0, "ok": 0, "429": 0, "503": 0, "413": 0, "stalls": 0, "replay_hits": 0,
                      "replay_misses": 0}
        self._server: asyncio.base_events.Server | None = None
        self.port = 0
        self.apply_scale()

    # ------------------------------------------------------------ control
    def apply_scale(self) -> None:
        only = {x.strip() for x in self.sw.limit_scale_only.split(",") if x.strip()}
        for m in self.models.values():
            for name, b in m.buckets.items():
                b.set_scale(self.sw.limit_scale if (not only or name in only) else 1.0)

    def control(self, body: dict) -> dict:
        for k, v in (body.get("set") or {}).items():
            if not hasattr(self.sw, k):
                raise ValueError(f"unknown switch {k}")
            cur = getattr(self.sw, k)
            setattr(self.sw, k, (str(v).lower() in ("1", "true", "yes", "on")) if isinstance(cur, bool)
                    else type(cur)(v))
        if "set" in body and ({"limit_scale", "limit_scale_only"} & set(body["set"])):
            self.apply_scale()
        ex = body.get("exhaust_daily")
        if ex:
            for m in (self.models.values() if ex == "all" else [self.models[ex]]):
                for name in ("RPD", "TPD"):
                    if name in m.buckets:
                        m.buckets[name].sync()
                        m.buckets[name].level = 0.0
        if body.get("reset"):
            for m in self.models.values():
                for b in m.buckets.values():
                    b.sync()
                    b.level = b.capacity
        return self.state()

    def state(self) -> dict:
        return {"switches": self.sw.__dict__, "inflight": self.inflight, "stats": self.stats,
                "levels": {m: {n: round(b.sync(), 1) for n, b in mo.buckets.items()} for m, mo in self.models.items()},
                "capacity": {m: {n: b.capacity for n, b in mo.buckets.items()} for m, mo in self.models.items()}}

    # ------------------------------------------------------------ latency model
    def ttfb_s(self, m: Model) -> float:
        t = m.cal["ttfb_ms"]
        ms = math.exp(t["mu"] + t["sigma"] * self.rng.gauss(0.0, 1.0))
        return ms / 1000 * self.load_factor()

    def load_factor(self) -> float:
        return 1.0 + self.sw.load_alpha * max(0, self.inflight - 1) if self.sw.slow_under_load else 1.0

    def queue_time_s(self, m: Model, ttfb_s: float) -> float:
        q = m.cal["queue_fit"]
        return max(0.0, (ttfb_s * 1000 - q["intercept_ms"]) / q["slope_ms_per_s"])

    # ------------------------------------------------------------ http
    async def __aenter__(self) -> "StandIn":
        await self.start()
        return self

    async def __aexit__(self, *exc) -> None:
        await self.stop()

    async def start(self, host: str = "127.0.0.1", port: int = 0) -> None:
        self._server = await asyncio.start_server(self._handle, host, port)
        self.port = self._server.sockets[0].getsockname()[1]

    async def stop(self) -> None:
        if self._server:
            self._server.close()
            await self._server.wait_closed()

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/openai/v1"

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            head = await reader.readuntil(b"\r\n\r\n")
            lines = head.decode("latin-1").split("\r\n")
            method, path, _ = lines[0].split(" ", 2)
            hdr = {k.strip().lower(): v.strip() for k, v in (l.split(":", 1) for l in lines[1:] if ":" in l)}
            body = await reader.readexactly(int(hdr.get("content-length", "0") or 0))
            if path.rstrip("/").endswith("/control"):
                try:
                    out = self.state() if method == "GET" else self.control(json.loads(body or b"{}"))
                    await self._send(writer, 200, [("content-type", "application/json")], json.dumps(out).encode())
                except (ValueError, KeyError) as e:
                    await self._send(writer, 400, [("content-type", "application/json")],
                                     json.dumps({"error": str(e)}).encode())
                return
            if method != "POST" or not path.rstrip("/").endswith("/chat/completions"):
                await self._send(writer, 404, [("content-type", "application/json")], b'{"error":"not found"}')
                return
            await self._chat(json.loads(body), writer)
        except (asyncio.IncompleteReadError, ConnectionError):
            pass
        except Exception as e:                       # never crash the server on one bad request
            try:
                await self._send(writer, 500, [("content-type", "application/json")],
                                 json.dumps({"error": {"message": f"stand-in error: {type(e).__name__}"}}).encode())
            except Exception:
                pass
        finally:
            try:
                writer.close()
            except Exception:
                pass

    async def _send(self, writer, status: int, hdrs: list[tuple[str, str]], body: bytes | None = None) -> None:
        reason = {200: "OK", 400: "Bad Request", 404: "Not Found", 413: "Payload Too Large", 429: "Too Many Requests",
                  500: "Internal Server Error", 503: "Service Unavailable"}.get(status, "OK")
        lines = [f"HTTP/1.1 {status} {reason}", "server: relay-standin", f"x-request-id: req_{uuid.uuid4().hex[:20]}"]
        lines += [f"{k}: {v}" for k, v in hdrs]
        if body is None:
            lines.append("transfer-encoding: chunked")
            writer.write(("\r\n".join(lines) + "\r\n\r\n").encode("latin-1"))
        else:
            lines.append(f"content-length: {len(body)}")
            writer.write(("\r\n".join(lines) + "\r\n\r\n").encode("latin-1") + body)
        await writer.drain()

    async def _sse(self, writer, obj) -> None:
        data = obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False)
        payload = f"data: {data}\n\n".encode("utf-8")
        writer.write(f"{len(payload):x}\r\n".encode() + payload + b"\r\n")
        await writer.drain()

    @staticmethod
    def _rate_headers(m: Model, request_only: bool = False) -> list[tuple[str, str]]:
        rpd, tpm = m.buckets.get("RPD"), m.buckets.get("TPM")
        h = []
        if rpd:
            h += [("x-ratelimit-limit-requests", f"{rpd.capacity:.0f}")]
        if tpm and not request_only:
            h += [("x-ratelimit-limit-tokens", f"{tpm.capacity:.0f}")]
        if rpd:
            h += [("x-ratelimit-remaining-requests", f"{max(0, math.floor(rpd.sync())):d}")]
        if tpm and not request_only:
            h += [("x-ratelimit-remaining-tokens", f"{max(0, math.floor(tpm.sync())):d}")]
        if rpd:
            h += [("x-ratelimit-reset-requests", fmt_duration(rpd.reset_s()))]
        if tpm and not request_only:
            h += [("x-ratelimit-reset-tokens", fmt_duration(tpm.reset_s()))]
        return h

    async def _chat(self, p: dict, writer) -> None:
        self.stats["requests"] += 1
        model = p.get("model")
        m = self.models.get(model)
        if m is None:
            await self._send(writer, 404, [("content-type", "application/json")], json.dumps(
                {"error": {"message": f"The model `{model}` does not exist or you do not have access to it.",
                           "type": "invalid_request_error", "code": "model_not_found"}}).encode())
            return
        if self.rng.random() < self.sw.p503:
            self.stats["503"] += 1
            await self._send(writer, 503, [("content-type", "application/json")], json.dumps(
                {"error": {"message": "Service Unavailable", "type": "internal_server_error"}}).encode())
            return
        msgs = p.get("messages") or []
        pt = m.prompt_tokens(msgs)
        mx = int(p.get("max_tokens") or p.get("max_completion_tokens") or 1024)
        hold = min(mx, m.hold_cap)
        out_hold = min(mx, m.out_hold_cap)
        per_unit = {"requests": 1.0, "tokens": float(pt + hold), "input_tokens": float(pt), "output_tokens": float(out_hold)}
        need = {n: per_unit.get(b.unit, float(pt + hold)) for n, b in m.buckets.items()}
        for n, b in m.buckets.items():
            if need[n] > b.capacity:
                self.stats["413"] += 1
                await self._send(writer, 413, [("content-type", "application/json")], json.dumps(
                    {"error": {"message": f"Request too large for model `{model}` in organization `org_STANDIN` "
                                          f"service tier `on_demand` on {WHAT.get(n, n)} ({n}): Limit "
                                          f"{b.capacity:.0f}, Requested {need[n]:.0f}, please reduce your message size "
                                          "and try again.", "type": "tokens", "code": "rate_limit_exceeded"}}).encode())
                return
        waits = {n: b.wait_for(need[n]) for n, b in m.buckets.items()}
        worst = max(waits, key=lambda n: waits[n])
        if waits[worst] > 0:
            self.stats["429"] += 1
            b = m.buckets[worst]
            kind = "requests" if b.unit == "requests" else "tokens"
            used = max(0, round(b.capacity - b.sync()))
            w = waits[worst]
            body = {"error": {"message": (
                f"Rate limit reached for model `{model}` in organization `org_STANDIN` service tier `on_demand` on "
                f"{WHAT.get(worst, worst)} ({worst}): Limit {b.capacity:.0f}, Used {used}, Requested {need[worst]:.0f}. "
                f"Please try again in {fmt_duration(w)}. Need more tokens? Upgrade to Dev Tier today at "
                "https://console.groq.com/settings/billing"),
                "type": kind, "code": "rate_limit_exceeded"}}
            hdrs = [("retry-after", str(max(1, math.ceil(w))))] + self._rate_headers(m, request_only=b.unit == "requests")
            await self._send(writer, 429, [("content-type", "application/json")] + hdrs, json.dumps(body).encode())
            return
        # admitted: charge at admission, snapshot headers as of admission (as Groq computes them)
        for n, b in m.buckets.items():
            b.take(need[n])
        hdrs = self._rate_headers(m)
        self.inflight += 1
        released = False

        def settle(completion: int) -> None:
            nonlocal released
            if released:
                return
            released = True
            for b in m.buckets.values():
                if b.unit == "tokens":
                    b.give(hold)
                    b.take(completion)
                elif b.unit == "output_tokens":
                    b.give(out_hold)
                    b.take(completion)
        try:
            ttfb = self.ttfb_s(m)
            if self.rng.random() < self.sw.stall_rate:
                self.stats["stalls"] += 1
                ttfb += self.sw.stall_s
            t0 = time.perf_counter()
            await asyncio.sleep(ttfb)
            text = self.replay.lookup(msgs)
            hit = text is not None
            self.stats["replay_hits" if hit else "replay_misses"] += 1
            if not hit:
                text = placeholder(msgs)
            await self._send(writer, 200, [("content-type", "text/event-stream")] + hdrs +
                             [("x-standin-replay", "hit" if hit else "miss")])
            rid = f"chatcmpl-{uuid.uuid4().hex[:24]}"
            base = {"id": rid, "object": "chat.completion.chunk", "created": int(time.time()), "model": model}
            await self._sse(writer, {**base, "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""},
                                                          "finish_reason": None}]})
            dec = m.cal["decode"]
            tps = dec["tokens_per_s"]
            cpt = dec["chars_per_token"]
            samples = m.cal.get("reasoning_tokens_samples") or [0]
            gpt_oss = model.startswith("openai/gpt-oss")
            effort = p.get("reasoning_effort")
            reasoning = self.rng.choice(samples) if (gpt_oss or effort not in (None, "none")) else 0
            reasoning = min(reasoning, mx)
            show = p.get("include_reasoning", True) if gpt_oss else p.get("reasoning_format") == "parsed"
            answer_tokens = max(1, math.ceil(len(text) / cpt))
            out_tokens = min(answer_tokens, mx - reasoning)
            visible = text[: max(0, round(out_tokens * cpt))] if out_tokens < answer_tokens else text
            finish = "stop" if out_tokens >= answer_tokens else "length"
            # pace against an absolute schedule: Windows sleeps overshoot by up to a 15.6 ms tick, so
            # sleeping a fixed slice each time would drift well below the measured tokens/s
            slice_s = 0.02
            clock = time.perf_counter
            due = clock()

            async def pace(tokens: float) -> None:
                nonlocal due
                due += tokens / (tps / self.load_factor())
                d = due - clock()
                if d > 0:
                    await asyncio.sleep(d)
            left = reasoning
            while left > 0:
                n = min(max(1, int(tps / self.load_factor() * slice_s)), left)
                left -= n
                if show:
                    await self._sse(writer, {**base, "choices": [{"index": 0, "delta": {"reasoning": "." * n},
                                                                  "finish_reason": None}]})
                await pace(n)
            i = 0
            while i < len(visible):
                n_tok = max(1, int(tps / self.load_factor() * slice_s))
                piece = visible[i: i + max(1, round(n_tok * cpt))]
                i += len(piece)
                await self._sse(writer, {**base, "choices": [{"index": 0, "delta": {"content": piece},
                                                              "finish_reason": None}]})
                await pace(len(piece) / cpt)
            completion = reasoning + max(0, out_tokens)
            total = time.perf_counter() - t0
            usage = {"queue_time": round(self.queue_time_s(m, ttfb), 6), "prompt_tokens": pt, "prompt_time": 0.002,
                     "completion_tokens": completion, "completion_time": round(completion / tps, 6),
                     "total_tokens": pt + completion, "total_time": round(total, 6)}
            if reasoning and gpt_oss:
                usage["completion_tokens_details"] = {"reasoning_tokens": reasoning}
            await self._sse(writer, {**base, "choices": [{"index": 0, "delta": {}, "finish_reason": finish}],
                                     "x_groq": {"id": f"req_{uuid.uuid4().hex[:20]}", "usage": usage}})
            await self._sse(writer, "[DONE]")
            writer.write(b"0\r\n\r\n")
            await writer.drain()
            settle(completion)
            self.stats["ok"] += 1
        except (ConnectionError, asyncio.CancelledError):
            # the client left mid-stream: keep what admission charged (unmeasured on Groq, H10)
            released = True
            raise
        finally:
            self.inflight -= 1


# ----------------------------------------------------------------------------- cli
def load_calibration(path: Path = CALIBRATION) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


async def _serve(a) -> None:
    sw = Switches(p503=a.p503, stall_rate=a.stall_rate, stall_s=a.stall_s, slow_under_load=a.slow_under_load,
                  limit_scale=a.limit_scale)
    s = StandIn(load_calibration(a.calibration), sw, Replay(a.replay), seed=a.seed)
    await s.start(a.host, a.port)
    print(f"stand-in on {s.base_url}  (replay: {len(s.replay.router)} router, {len(s.replay.draft)} drafts)", flush=True)
    try:
        while True:
            await asyncio.sleep(3600)
    finally:
        await s.stop()


def _ctl(a) -> int:
    import urllib.request
    url = f"http://127.0.0.1:{a.port}/control"
    if a.action == "show":
        req = urllib.request.Request(url, method="GET")
    else:
        body: dict[str, Any] = {}
        if a.action == "set":
            body["set"] = dict(kv.split("=", 1) for kv in a.args)
        elif a.action == "exhaust-daily":
            body["exhaust_daily"] = (a.args[0] if a.args else "all")
        elif a.action == "reset":
            body["reset"] = True
        req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                     headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as r:
        print(json.dumps(json.loads(r.read()), indent=1))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m standin.server")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8790)
    s.add_argument("--calibration", type=Path, default=CALIBRATION)
    s.add_argument("--replay", type=Path, default=None, help="results/raw/answers: replay recorded real texts")
    s.add_argument("--seed", type=int, default=None)
    s.add_argument("--p503", type=float, default=0.0)
    s.add_argument("--stall-rate", type=float, default=0.0)
    s.add_argument("--stall-s", type=float, default=8.0)
    s.add_argument("--slow-under-load", action="store_true")
    s.add_argument("--limit-scale", type=float, default=1.0)
    c = sub.add_parser("ctl")
    c.add_argument("--port", type=int, default=8790)
    c.add_argument("action", choices=["show", "set", "exhaust-daily", "reset"])
    c.add_argument("args", nargs="*")
    a = ap.parse_args(argv)
    if a.cmd == "serve":
        try:
            asyncio.run(_serve(a))
        except KeyboardInterrupt:
            pass
        return 0
    return _ctl(a)


if __name__ == "__main__":
    sys.exit(main())
