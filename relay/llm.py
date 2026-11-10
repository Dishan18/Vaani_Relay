"""One async streaming client for any OpenAI-compatible chat endpoint (Groq now, the stand-in later).

Stdlib only (asyncio streams + ssl): a hand-rolled HTTP/1.1 reader lets us timestamp every
network read ourselves, which is the whole point of this module. One connection per call
(``Connection: close``); the TCP+TLS setup time is recorded separately as ``connect_ms`` so it
never hides inside "time to headers".

Every call produces a ``CallRecord`` (times in ms from the moment the call starts):
  connect_ms          TCP (+TLS) connection established
  headers_ms          full response header block received
  first_byte_ms       first response-body byte received
  first_reasoning_ms  first hidden-reasoning delta (``delta.reasoning`` / ``reasoning_content`` /
                      text inside ``<think>...</think>``)
  first_visible_ms    first content delta outside reasoning that contains a non-space character
  total_ms            call finished (success, error, timeout or cancellation)
  chunks              every SSE event: arrival time, kind, characters, bytes
  rate_limit_headers  every ``x-ratelimit-*`` and ``retry-after`` header, verbatim, in order
  error_body          the whole body of any non-200 response (organisation ids redacted)
  usage               token usage as reported (``usage`` or ``x_groq.usage``)

Timeouts (each optional; None disables):
  first_visible_s  no visible token this long after the call started -> cancel the call
  stall_s          no bytes for this long once the body has started -> cancel
  deadline_s       the whole call takes longer than this -> cancel
Cancelling aborts the TCP connection, so the server sees the client go away.

There are **no retries** here. Retrying is a policy decision and lives with the caller, which can
see the record of the failed attempt (``record_sink`` is called for every call, including calls
cancelled by the caller).
"""
from __future__ import annotations

import asyncio
import json
import re
import ssl
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable
from urllib.parse import urlsplit

USER_AGENT = "relay-llm-client/0.1"
ORG_ID = re.compile(r"org_[A-Za-z0-9]+")
KEPT_HEADERS = ("x-request-id", "date", "content-type", "server", "via", "cf-ray", "x-groq-region")

OK = "ok"
RATE_LIMITED = "rate_limited"
HTTP_ERROR = "http_error"
STREAM_ERROR = "stream_error"
TIMEOUT_FIRST_VISIBLE = "timeout_first_visible"
TIMEOUT_STALL = "timeout_stall"
TIMEOUT_DEADLINE = "timeout_deadline"
CONNECT_ERROR = "connect_error"
PROTOCOL_ERROR = "protocol_error"
CANCELLED = "cancelled"


def redact(text: str) -> str:
    return ORG_ID.sub("org_REDACTED", text)


@dataclass(frozen=True)
class Endpoint:
    base_url: str                 # e.g. "https://api.groq.com/openai/v1"
    api_key: str | None = None
    source: str = "real"          # "real" or "stand-in"; copied into every record

    def __repr__(self) -> str:   # never print the key
        return f"Endpoint(base_url={self.base_url!r}, source={self.source!r}, api_key={'***' if self.api_key else None})"


@dataclass(frozen=True)
class Timeouts:
    first_visible_s: float | None = 20.0
    stall_s: float | None = 10.0
    deadline_s: float | None = 60.0


@dataclass
class Chunk:
    t_ms: float
    kind: str          # reasoning | content | mixed | usage | done | error | other
    chars: int
    bytes: int


@dataclass
class CallRecord:
    call_id: str
    source: str
    url: str
    model: str | None
    started_at: str                      # UTC ISO-8601, wall clock
    request_bytes: int = 0
    prompt_chars: int = 0
    max_tokens: int | None = None
    outcome: str = ""
    status: int | None = None
    error: str | None = None
    connect_ms: float | None = None
    headers_ms: float | None = None
    first_byte_ms: float | None = None
    first_reasoning_ms: float | None = None
    first_visible_ms: float | None = None
    total_ms: float | None = None
    bytes_received: int = 0
    chunks: list[Chunk] = field(default_factory=list)
    rate_limit_headers: list[list[str]] = field(default_factory=list)
    other_headers: dict[str, str] = field(default_factory=dict)
    error_body: str | None = None
    usage: dict[str, Any] | None = None
    finish_reason: str | None = None
    response_id: str | None = None
    response_model: str | None = None
    text: str = ""
    reasoning_chars: int = 0

    def header(self, name: str) -> str | None:
        name = name.lower()
        for k, v in self.rate_limit_headers:
            if k.lower() == name:
                return v
        return None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        for k, v in d.items():
            if k.endswith("_ms") and isinstance(v, float):
                d[k] = round(v, 1)
        d["chunks"] = [[round(c.t_ms, 1), c.kind, c.chars, c.bytes] for c in self.chunks]
        return d


class _Timeout(Exception):
    def __init__(self, outcome: str):
        super().__init__(outcome)
        self.outcome = outcome


class _ChunkedDecoder:
    """Incremental decoder for ``Transfer-Encoding: chunked``."""

    def __init__(self) -> None:
        self.buf = b""
        self.remaining = 0        # bytes left in the current chunk
        self.need_crlf = False
        self.done = False

    def feed(self, data: bytes) -> bytes:
        self.buf += data
        out = bytearray()
        while not self.done:
            if self.need_crlf:
                if len(self.buf) < 2:
                    break
                if self.buf[:2] != b"\r\n":
                    raise ValueError("bad chunk terminator")
                self.buf = self.buf[2:]
                self.need_crlf = False
            if self.remaining == 0:
                i = self.buf.find(b"\r\n")
                if i < 0:
                    break
                size = int(self.buf[:i].split(b";")[0].strip() or b"0", 16)
                self.buf = self.buf[i + 2:]
                if size == 0:
                    self.done = True
                    break
                self.remaining = size
            take = self.buf[: self.remaining]
            out += take
            self.buf = self.buf[len(take):]
            self.remaining -= len(take)
            if self.remaining:
                break
            self.need_crlf = True
        return bytes(out)


class _ThinkSplitter:
    """Splits streamed content into ('reasoning', text) / ('content', text) around
    <think>...</think>, holding back a possible partial tag at the end of a piece."""

    OPEN, CLOSE = "<think>", "</think>"

    def __init__(self) -> None:
        self.in_think = False
        self.carry = ""

    def feed(self, text: str, final: bool = False) -> list[tuple[str, str]]:
        s = self.carry + text
        self.carry = ""
        out: list[tuple[str, str]] = []
        while s:
            tag = self.CLOSE if self.in_think else self.OPEN
            i = s.find(tag)
            if i >= 0:
                if i:
                    out.append(("reasoning" if self.in_think else "content", s[:i]))
                s = s[i + len(tag):]
                self.in_think = not self.in_think
                continue
            hold = 0
            if not final:
                for n in range(min(len(tag) - 1, len(s)), 0, -1):
                    if tag.startswith(s[-n:]):
                        hold = n
                        break
            emit, self.carry = (s[:-hold], s[-hold:]) if hold else (s, "")
            if emit:
                out.append(("reasoning" if self.in_think else "content", emit))
            break
        return out


class LLMClient:
    def __init__(self, endpoint: Endpoint, timeouts: Timeouts = Timeouts(),
                 ssl_context: ssl.SSLContext | None = None) -> None:
        self.endpoint = endpoint
        self.timeouts = timeouts
        u = urlsplit(endpoint.base_url.rstrip("/") + "/chat/completions")
        if u.scheme not in ("http", "https"):
            raise ValueError(f"unsupported scheme in {endpoint.base_url!r}")
        self._scheme = u.scheme
        self._host = u.hostname or ""
        self._port = u.port or (443 if u.scheme == "https" else 80)
        self._path = u.path or "/"
        self._url = f"{u.scheme}://{u.netloc}{self._path}"
        self._ssl = (ssl_context or ssl.create_default_context()) if u.scheme == "https" else None

    async def stream_chat(
        self,
        payload: dict[str, Any],
        *,
        on_delta: Callable[[str, str], None] | None = None,
        record_sink: Callable[[CallRecord], None] | None = None,
        timeouts: Timeouts | None = None,
    ) -> CallRecord:
        """Send one streaming chat request. Always returns (or, if the caller cancels, raises
        CancelledError after) a finished CallRecord, which is also passed to ``record_sink``."""
        to = timeouts or self.timeouts
        payload = {**payload, "stream": True}
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        messages = payload.get("messages") or []
        rec = CallRecord(
            call_id=uuid.uuid4().hex[:16],
            source=self.endpoint.source,
            url=self._url,
            model=payload.get("model"),
            started_at=datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            request_bytes=len(body),
            prompt_chars=sum(len(m.get("content") or "") for m in messages if isinstance(m, dict)),
            max_tokens=payload.get("max_tokens") or payload.get("max_completion_tokens"),
        )
        t0 = time.perf_counter()
        state: dict[str, Any] = {"writer": None}
        try:
            await self._run(rec, body, to, t0, state, on_delta)
        except asyncio.CancelledError:
            rec.outcome = CANCELLED
            raise
        finally:
            w = state["writer"]
            if w is not None:
                try:
                    w.transport.abort()  # hard close: the server sees the client leave
                except Exception:
                    pass
            rec.total_ms = (time.perf_counter() - t0) * 1000
            if record_sink is not None:
                record_sink(rec)
        return rec

    # ------------------------------------------------------------------ internals

    async def _run(self, rec: CallRecord, body: bytes, to: Timeouts, t0: float,
                   state: dict[str, Any], on_delta) -> None:
        def now_ms() -> float:
            return (time.perf_counter() - t0) * 1000

        last_byte = [None]  # perf_counter of the last body byte, for stall detection
        stage = ["connect"]  # connect -> send -> headers -> body; named in timeout errors

        def budget() -> tuple[float, str]:
            elapsed = time.perf_counter() - t0
            cands: list[tuple[float, str]] = []
            if to.deadline_s is not None:
                cands.append((to.deadline_s - elapsed, TIMEOUT_DEADLINE))
            if to.first_visible_s is not None and rec.first_visible_ms is None:
                cands.append((to.first_visible_s - elapsed, TIMEOUT_FIRST_VISIBLE))
            if to.stall_s is not None and last_byte[0] is not None:
                cands.append((to.stall_s - (time.perf_counter() - last_byte[0]), TIMEOUT_STALL))
            if not cands:
                return (None, "")  # type: ignore[return-value]
            return min(cands)

        async def timed(coro):
            secs, why = budget()
            if secs is not None and secs <= 0:
                coro.close()
                raise _Timeout(why)
            try:
                return await asyncio.wait_for(coro, secs)
            except asyncio.TimeoutError:
                raise _Timeout(why) from None

        try:
            # --- connect + send
            try:
                reader, writer = await timed(asyncio.open_connection(
                    self._host, self._port, ssl=self._ssl,
                    server_hostname=self._host if self._ssl else None))
            except _Timeout:
                raise
            except (OSError, ssl.SSLError) as e:
                rec.outcome, rec.error = CONNECT_ERROR, f"{type(e).__name__}: {e}"
                return
            state["writer"] = writer
            rec.connect_ms = now_ms()
            stage[0] = "send"
            host_hdr = self._host if self._port in (80, 443) else f"{self._host}:{self._port}"
            head = [
                f"POST {self._path} HTTP/1.1",
                f"Host: {host_hdr}",
                f"User-Agent: {USER_AGENT}",
                "Content-Type: application/json",
                "Accept: text/event-stream",
                "Accept-Encoding: identity",
                "Connection: close",
                f"Content-Length: {len(body)}",
            ]
            if self.endpoint.api_key:
                head.append(f"Authorization: Bearer {self.endpoint.api_key}")
            writer.write(("\r\n".join(head) + "\r\n\r\n").encode("latin-1") + body)
            await timed(writer.drain())

            # --- status line + headers
            stage[0] = "headers"
            buf = b""
            while b"\r\n\r\n" not in buf:
                data = await timed(reader.read(65536))
                if not data:
                    rec.outcome, rec.error = PROTOCOL_ERROR, "connection closed before headers"
                    return
                buf += data
            raw_head, rest = buf.split(b"\r\n\r\n", 1)
            rec.headers_ms = now_ms()
            stage[0] = "body"
            lines = raw_head.decode("latin-1").split("\r\n")
            m = re.match(r"HTTP/\d(?:\.\d)?\s+(\d{3})", lines[0])
            if not m:
                rec.outcome, rec.error = PROTOCOL_ERROR, f"bad status line {lines[0][:80]!r}"
                return
            rec.status = int(m.group(1))
            headers: dict[str, str] = {}
            for ln in lines[1:]:
                if ":" not in ln:
                    continue
                k, v = ln.split(":", 1)
                k, v = k.strip(), v.strip()
                headers[k.lower()] = v
                kl = k.lower()
                if kl.startswith("x-ratelimit-") or kl == "retry-after":
                    rec.rate_limit_headers.append([k, redact(v)])
                elif kl in KEPT_HEADERS:
                    rec.other_headers[kl] = redact(v)

            chunked = "chunked" in headers.get("transfer-encoding", "").lower()
            length = int(headers["content-length"]) if "content-length" in headers and not chunked else None
            decoder = _ChunkedDecoder() if chunked else None
            received = 0

            async def body_pieces():
                nonlocal received
                pending = rest
                while True:
                    if pending:
                        data, pending = pending, b""
                    else:
                        if decoder is not None and decoder.done:
                            return
                        if length is not None and received >= length:
                            return
                        data = await timed(reader.read(65536))
                        if not data:
                            return
                    t = time.perf_counter()
                    if rec.first_byte_ms is None:
                        rec.first_byte_ms = (t - t0) * 1000
                    last_byte[0] = t
                    rec.bytes_received += len(data)
                    received += len(data)
                    yield decoder.feed(data) if decoder is not None else data

            # --- non-200: keep the whole body
            if rec.status != 200:
                chunks = []
                async for piece in body_pieces():
                    chunks.append(piece)
                rec.error_body = redact(b"".join(chunks).decode("utf-8", "replace"))
                rec.outcome = RATE_LIMITED if rec.status == 429 else HTTP_ERROR
                rec.error = f"HTTP {rec.status}"
                return

            # --- 200: server-sent events
            splitter = _ThinkSplitter()
            sse_buf = ""
            saw_done = False

            def emit(kind: str, text: str, t_ms: float) -> None:
                if not text:
                    return
                if kind == "reasoning":
                    rec.reasoning_chars += len(text)
                    if rec.first_reasoning_ms is None:
                        rec.first_reasoning_ms = t_ms
                else:
                    rec.text += text  # kept as it arrives, so a timeout still has the partial answer
                    if rec.first_visible_ms is None and text.strip():
                        rec.first_visible_ms = t_ms
                if on_delta is not None:
                    on_delta(kind, text)

            async for piece in body_pieces():
                t_ms = now_ms()
                sse_buf += piece.decode("utf-8", "replace") if piece else ""
                sse_buf = sse_buf.replace("\r\n", "\n")
                while "\n\n" in sse_buf:
                    event, sse_buf = sse_buf.split("\n\n", 1)
                    data_lines = [ln[5:].lstrip() for ln in event.split("\n") if ln.startswith("data:")]
                    if not data_lines:
                        continue
                    data = "\n".join(data_lines)
                    ev_bytes = len(event.encode("utf-8")) + 2
                    if data.strip() == "[DONE]":
                        saw_done = True
                        rec.chunks.append(Chunk(t_ms, "done", 0, ev_bytes))
                        continue
                    try:
                        obj = json.loads(data)
                    except json.JSONDecodeError:
                        rec.chunks.append(Chunk(t_ms, "other", 0, ev_bytes))
                        continue
                    if isinstance(obj, dict) and "error" in obj:
                        rec.chunks.append(Chunk(t_ms, "error", 0, ev_bytes))
                        rec.outcome = STREAM_ERROR
                        rec.error_body = redact(json.dumps(obj["error"], ensure_ascii=False))
                        rec.error = "error event in stream"
                        return
                    rec.response_id = rec.response_id or obj.get("id")
                    rec.response_model = rec.response_model or obj.get("model")
                    usage = obj.get("usage") or (obj.get("x_groq") or {}).get("usage")
                    if usage:
                        rec.usage = usage
                    kinds: set[str] = set()
                    chars = 0
                    for choice in obj.get("choices") or []:
                        delta = choice.get("delta") or {}
                        r = delta.get("reasoning") or delta.get("reasoning_content")
                        if r:
                            emit("reasoning", r, t_ms)
                            kinds.add("reasoning")
                            chars += len(r)
                        c = delta.get("content")
                        if c:
                            for kind, part in splitter.feed(c):
                                emit(kind, part, t_ms)
                                kinds.add(kind)
                                chars += len(part)
                        if choice.get("finish_reason"):
                            rec.finish_reason = choice["finish_reason"]
                    kind = kinds.pop() if len(kinds) == 1 else ("mixed" if kinds else ("usage" if usage else "other"))
                    rec.chunks.append(Chunk(t_ms, kind, chars, ev_bytes))
                if saw_done:
                    break

            for kind, part in splitter.feed("", final=True):
                emit(kind, part, now_ms())
            if saw_done or rec.finish_reason:
                rec.outcome = OK
            else:
                rec.outcome, rec.error = PROTOCOL_ERROR, "stream ended without [DONE] or finish_reason"
        except _Timeout as t:
            rec.outcome = t.outcome
            rec.error = f"{t.outcome} during {stage[0]}"
        except (OSError, ssl.SSLError, ValueError) as e:
            rec.outcome, rec.error = PROTOCOL_ERROR, f"{type(e).__name__}: {e}"
