"""A local simulator of Groq's OpenAI-compatible endpoint, with rate limits whose behaviour we choose.

Used for probe dry-runs (scripts/probe_groq.py --dry-run) and tests. Because we pick the "truth"
(fixed windows or refilling buckets, admission or usage charging, per-model or pooled limits), a
dry-run checks that the probe and its analysis would *recover* that truth before any real budget is
spent. Nothing here is a claim about how Groq behaves; the numbers are illustrative defaults.

Simulated:
  * limits per model (or one pool): RPM and TPM as a bucket or a wall-clock-minute window; RPD as
    a counter; TPM charged on prompt + max_tokens at admission (refunded at the end, or not) or
    on actual usage at the end
  * x-ratelimit-* headers (RPD for requests, TPM for tokens), retry-after on 429
  * 429 bodies in Groq's wording; 413 when a request alone exceeds TPM
  * reasoning parameters per model family (gpt-oss: low/medium/high, include_reasoning;
    qwen: none/default/low/medium/high, reasoning_format), 400 on invalid combinations
  * streaming with reasoning deltas, content deltas and x_groq.usage in the last event
  * latency that grows with prompt size and with requests in flight; optional concurrency limit
"""
from __future__ import annotations

import asyncio
import json
import math
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

ORG = "org_01fakeorgid000000000000"


@dataclass
class Truth:
    window: str = "bucket"            # "bucket" | "window" (applies to RPM and TPM)
    charge: str = "admission_refund"  # "admission_refund" | "admission" | "usage"
    pooled: bool = False              # one set of RPM/TPM counters for every model
    rpm: int = 30
    rpd: int = 1000
    tpm: int = 8000
    rpd_used_at_start: int = 37       # someone else has used the shared key today
    max_concurrency: int | None = None
    count_429_toward_rpd: bool = False
    oversize_status: int = 413        # a request whose own need exceeds TPM

    def label(self) -> str:
        return f"{self.window}-{self.charge}-{'pooled' if self.pooled else 'per_model'}"


@dataclass
class ModelSim:
    family: str            # "gpt-oss" | "qwen"
    template_tokens: int   # hidden chat-template tokens
    decode_tps: float      # tokens per second after the first token
    base_ttft_s: float
    per_prompt_token_s: float


MODELS = {
    "openai/gpt-oss-20b": ModelSim("gpt-oss", 70, 900.0, 0.12, 0.00008),
    "openai/gpt-oss-120b": ModelSim("gpt-oss", 70, 450.0, 0.20, 0.00015),
    "qwen/qwen3.8-27b": ModelSim("qwen", 14, 350.0, 0.25, 0.00012),
}


def fmt_duration(s: float) -> str:
    """Groq-style durations: '7.66s', '2m59.56s', '1h2m3.4s', '120ms'."""
    if s < 1:
        return f"{max(0, int(s * 1000))}ms"
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    out = ""
    if h:
        out += f"{int(h)}h"
    if h or m:
        out += f"{int(m)}m"
    return out + f"{sec:.2f}".rstrip("0").rstrip(".") + "s"


class _Counter:
    """RPM or TPM counter, either a refilling bucket or a wall-clock-minute window."""

    def __init__(self, capacity: int, kind: str, clock) -> None:
        self.cap = capacity
        self.kind = kind
        self.clock = clock
        self.level = float(capacity)   # bucket: tokens available
        self.used = 0.0                # window: used in the current minute
        self.minute = int(clock() // 60)

    def _sync(self) -> None:
        now = self.clock()
        if self.kind == "bucket":
            last = getattr(self, "_t", now)
            self.level = min(self.cap, self.level + (now - last) * self.cap / 60.0)
            self._t = now
        else:
            m = int(now // 60)
            if m != self.minute:
                self.minute, self.used = m, 0.0

    def available(self) -> float:
        self._sync()
        return self.level if self.kind == "bucket" else self.cap - self.used

    def take(self, n: float) -> None:
        self._sync()
        if self.kind == "bucket":
            self.level -= n
        else:
            self.used += n

    def give(self, n: float) -> None:
        self._sync()
        if self.kind == "bucket":
            self.level = min(self.cap, self.level + n)
        else:
            self.used = max(0.0, self.used - n)

    def wait_for(self, n: float) -> float:
        """Seconds until ``n`` is available (0 if now)."""
        self._sync()
        short = n - self.available()
        if short <= 0:
            return 0.0
        if self.kind == "bucket":
            return short / (self.cap / 60.0)
        return 60 - (self.clock() % 60)

    def reset_s(self) -> float:
        """What Groq's reset header would say: time until the counter is full again."""
        self._sync()
        if self.kind == "bucket":
            return (self.cap - self.level) / (self.cap / 60.0)
        return 0.0 if self.used == 0 else 60 - (self.clock() % 60)


class FakeGroq:
    def __init__(self, truth: Truth = Truth(), clock=time.time) -> None:
        self.truth = truth
        self.clock = clock
        self.req: dict[str, _Counter] = {}
        self.tok: dict[str, _Counter] = {}
        self.rpd_used: dict[str, int] = {}
        self.inflight = 0
        self.requests: list[dict] = []
        self._server: asyncio.base_events.Server | None = None
        self.port = 0

    # ----------------------------------------------------------------- limits
    def _key(self, model: str) -> str:
        return "pool" if self.truth.pooled else model

    def _counters(self, model: str) -> tuple[_Counter, _Counter]:
        k = self._key(model)
        if k not in self.req:
            self.req[k] = _Counter(self.truth.rpm, self.truth.window, self.clock)
            self.tok[k] = _Counter(self.truth.tpm, self.truth.window, self.clock)
            self.rpd_used[k] = self.truth.rpd_used_at_start
        return self.req[k], self.tok[k]

    def _headers(self, model: str) -> list[tuple[str, str]]:
        rq, tk = self._counters(model)
        k = self._key(model)
        now = datetime.now(timezone.utc)
        midnight = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        return [
            ("x-ratelimit-limit-requests", str(self.truth.rpd)),
            ("x-ratelimit-limit-tokens", str(self.truth.tpm)),
            ("x-ratelimit-remaining-requests", str(self.truth.rpd - self.rpd_used[k])),
            ("x-ratelimit-remaining-tokens", str(max(0, int(tk.available())))),
            ("x-ratelimit-reset-requests", fmt_duration((midnight - now).total_seconds())),
            ("x-ratelimit-reset-tokens", fmt_duration(tk.reset_s())),
        ]

    @staticmethod
    def _body(model: str, what: str, short: str, limit: int, used: int, requested: int, wait: float,
              kind: str) -> bytes:
        msg = (f"Rate limit reached for model `{model}` in organization `{ORG}` service tier `on_demand` "
               f"on {what} ({short}): Limit {limit}, Used {used}, Requested {requested}. "
               f"Please try again in {fmt_duration(wait)}. Need more tokens? Upgrade to Dev Tier today "
               f"at https://console.groq.com/settings/billing")
        return json.dumps({"error": {"message": msg, "type": kind, "code": "rate_limit_exceeded"}}).encode()

    # ----------------------------------------------------------------- server
    async def __aenter__(self) -> "FakeGroq":
        self._server = await asyncio.start_server(self._handle, "127.0.0.1", 0)
        self.port = self._server.sockets[0].getsockname()[1]
        return self

    async def __aexit__(self, *exc) -> None:
        assert self._server is not None
        self._server.close()
        await self._server.wait_closed()

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/openai/v1"

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            head = await reader.readuntil(b"\r\n\r\n")
            headers = {}
            for ln in head.decode("latin-1").split("\r\n")[1:]:
                if ":" in ln:
                    k, v = ln.split(":", 1)
                    headers[k.strip().lower()] = v.strip()
            body = json.loads(await reader.readexactly(int(headers.get("content-length", "0"))) or b"{}")
            self.requests.append(body)
            await self._respond(body, writer)
        except (ConnectionError, asyncio.IncompleteReadError):
            pass
        finally:
            try:
                writer.close()
            except Exception:
                pass

    async def _send(self, writer, status: int, hdrs: list[tuple[str, str]], body: bytes | None = None) -> None:
        reason = {200: "OK", 400: "Bad Request", 413: "Payload Too Large", 429: "Too Many Requests"}.get(status, "X")
        lines = [f"HTTP/1.1 {status} {reason}", "Connection: close", "Content-Type: "
                 + ("text/event-stream" if body is None else "application/json"),
                 f"x-request-id: req_{uuid.uuid4().hex[:20]}"]
        lines += [f"{k}: {v}" for k, v in hdrs]
        if body is None:
            lines.append("Transfer-Encoding: chunked")
        else:
            lines.append(f"Content-Length: {len(body)}")
        writer.write(("\r\n".join(lines) + "\r\n\r\n").encode("latin-1") + (body or b""))
        await writer.drain()

    async def _chunk(self, writer, obj) -> None:
        data = f"data: {obj if isinstance(obj, str) else json.dumps(obj)}\n\n".encode()
        writer.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
        await writer.drain()

    def _bad(self, msg: str) -> bytes:
        return json.dumps({"error": {"message": msg, "type": "invalid_request_error"}}).encode()

    async def _respond(self, p: dict, writer) -> None:
        model = p.get("model", "")
        sim = MODELS.get(model)
        if sim is None:
            return await self._send(writer, 404, [], self._bad(f"The model `{model}` does not exist"))
        effort = p.get("reasoning_effort")
        fmt = p.get("reasoning_format")
        if sim.family == "gpt-oss":
            if fmt is not None:
                return await self._send(writer, 400, [], self._bad("reasoning_format is not supported with this model"))
            if effort not in (None, "low", "medium", "high"):
                return await self._send(writer, 400, [], self._bad("`reasoning_effort` must be one of `low`, `medium`, `high`"))
        else:
            if effort not in (None, "none", "default", "low", "medium", "high"):
                return await self._send(writer, 400, [], self._bad("invalid reasoning_effort"))
            if fmt is not None and "include_reasoning" in p:
                return await self._send(writer, 400, [], self._bad("include_reasoning cannot be used with reasoning_format"))

        msgs = p.get("messages") or []
        chars = sum(len(m.get("content") or "") for m in msgs)
        prompt_tokens = math.ceil(chars / 4) + 4 * len(msgs) + sim.template_tokens
        max_tokens = int(p.get("max_tokens") or p.get("max_completion_tokens") or 1024)
        rq, tk = self._counters(model)
        k = self._key(model)
        t = self.truth
        need = prompt_tokens + (max_tokens if t.charge != "usage" else 0)

        if t.max_concurrency is not None and self.inflight >= t.max_concurrency:
            body = json.dumps({"error": {"message": "Too many concurrent requests", "type": "requests",
                                         "code": "rate_limit_exceeded"}}).encode()
            return await self._send(writer, 429, self._headers(model) + [("retry-after", "1")], body)
        if need > t.tpm:
            body = json.dumps({"error": {"message":
                f"Request too large for model `{model}` in organization `{ORG}` service tier `on_demand` on "
                f"tokens per minute (TPM): Limit {t.tpm}, Requested {need}, please reduce your message size "
                f"and try again.", "type": "tokens", "code": "rate_limit_exceeded"}}).encode()
            return await self._send(writer, t.oversize_status, self._headers(model), body)
        if rq.available() < 1:
            wait = rq.wait_for(1)
            if t.count_429_toward_rpd:
                self.rpd_used[k] += 1
            body = self._body(model, "requests per minute", "RPM", t.rpm, int(t.rpm - rq.available()), 1, wait, "requests")
            return await self._send(writer, 429, self._headers(model) + [("retry-after", str(math.ceil(wait)))], body)
        if tk.available() < need:
            wait = tk.wait_for(need)
            body = self._body(model, "tokens per minute", "TPM", t.tpm, int(t.tpm - tk.available()), need, wait, "tokens")
            return await self._send(writer, 429, self._headers(model) + [("retry-after", str(math.ceil(wait)))], body)

        rq.take(1)
        tk.take(need)
        self.rpd_used[k] += 1
        hdrs = self._headers(model)
        self.inflight += 1
        try:
            await self._stream(writer, p, sim, prompt_tokens, max_tokens, hdrs, tk, need)
        finally:
            self.inflight -= 1

    async def _stream(self, writer, p, sim: ModelSim, prompt_tokens: int, max_tokens: int, hdrs, tk, held) -> None:
        t0 = time.perf_counter()
        queue = 0.01 + 0.04 * max(0, self.inflight - 1)
        await asyncio.sleep(queue + sim.base_ttft_s + prompt_tokens * sim.per_prompt_token_s)
        await self._send(writer, 200, hdrs)
        rid = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        base = {"id": rid, "object": "chat.completion.chunk", "model": p["model"]}
        await self._chunk(writer, {**base, "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""},
                                                        "finish_reason": None}]})
        effort = p.get("reasoning_effort")
        user = (p["messages"][-1].get("content") or "")
        system = (p["messages"][0].get("content") or "") if len(p["messages"]) > 1 else ""
        if sim.family == "gpt-oss":
            reasoning_tokens = {"low": 24, "medium": 90, "high": 300}.get(effort or "medium", 90)
            show_reasoning = p.get("include_reasoning", True)
        else:
            reasoning_tokens = {"none": 0, "default": 0, "low": 40}.get(effort or "default", 120)
            show_reasoning = p.get("reasoning_format") == "parsed"
        if "JSON" in system:
            answer = '{"papers": ["attention"], "kind": "fact"}'
        elif "PASSAGES" in user:
            answer = ("The model uses scaled dot-product attention [attention p.4 §3.2.1]. " * 14).strip()
        else:
            answer = "OK"
        answer_tokens = math.ceil(len(answer) / 4)
        reasoning_tokens = min(reasoning_tokens, max_tokens)
        answer_tokens_out = min(answer_tokens, max_tokens - reasoning_tokens)
        finish = "stop" if answer_tokens_out == answer_tokens else "length"
        # emit in ~20 ms slices (Windows timers tick at ~15.6 ms)
        per_slice = max(1, int(sim.decode_tps * 0.02))
        r_left = reasoning_tokens
        while r_left > 0:
            n = min(per_slice, r_left)
            r_left -= n
            if show_reasoning:
                await self._chunk(writer, {**base, "choices": [{"index": 0, "delta": {"reasoning": "think " * n},
                                                                "finish_reason": None}]})
            await asyncio.sleep(n / sim.decode_tps)
        a = answer[: answer_tokens_out * 4]
        for i in range(0, len(a), per_slice * 4):
            await self._chunk(writer, {**base, "choices": [{"index": 0, "delta": {"content": a[i:i + per_slice * 4]},
                                                            "finish_reason": None}]})
            await asyncio.sleep(min(per_slice, answer_tokens_out) / sim.decode_tps)
        completion = reasoning_tokens + answer_tokens_out
        total_time = time.perf_counter() - t0
        usage = {"queue_time": round(queue, 4), "prompt_tokens": prompt_tokens, "prompt_time": 0.01,
                 "completion_tokens": completion, "completion_time": round(completion / sim.decode_tps, 4),
                 "total_tokens": prompt_tokens + completion, "total_time": round(total_time, 4)}
        if self.truth.charge == "admission_refund":
            tk.give(held - (prompt_tokens + completion))
        elif self.truth.charge == "usage":
            tk.take(completion)          # the prompt was taken at admission
        await self._chunk(writer, {**base, "choices": [{"index": 0, "delta": {}, "finish_reason": finish}],
                                   "x_groq": {"id": "req_x", "usage": usage}})
        await self._chunk(writer, "[DONE]")
        writer.write(b"0\r\n\r\n")
        await writer.drain()
