"""A tiny scripted HTTP/1.1 server that speaks just enough OpenAI-style SSE to test relay.llm offline.

Each request pops the next script from a queue. A script is a list of steps:
  ("head", status, [(name, value), ...])   status line + headers (adds Connection: close)
  ("sse", obj_or_"[DONE]")                  one server-sent event
  ("body", bytes)                           raw body bytes (for error responses)
  ("sleep", seconds)
  ("close",)                                close without finishing
Set ``chunked=True`` on the server to frame the body with Transfer-Encoding: chunked.

The server records every request (method, path, headers, JSON body) and whether the client
went away before the script finished (``client_gone``), so tests can check that a timeout
really cancels the call.
"""
from __future__ import annotations

import asyncio
import json


class FakeOpenAIServer:
    def __init__(self, chunked: bool = False) -> None:
        self.chunked = chunked
        self.scripts: list[list[tuple]] = []
        self.requests: list[dict] = []
        self.client_gone = asyncio.Event()
        self._server: asyncio.base_events.Server | None = None
        self.port = 0

    async def __aenter__(self) -> "FakeOpenAIServer":
        self._server = await asyncio.start_server(self._handle, "127.0.0.1", 0)
        self.port = self._server.sockets[0].getsockname()[1]
        return self

    async def __aexit__(self, *exc) -> None:
        assert self._server is not None
        self._server.close()
        await self._server.wait_closed()

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1"

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            head = await reader.readuntil(b"\r\n\r\n")
            lines = head.decode("latin-1").split("\r\n")
            method, path, _ = lines[0].split(" ", 2)
            headers = {}
            for ln in lines[1:]:
                if ":" in ln:
                    k, v = ln.split(":", 1)
                    headers[k.strip().lower()] = v.strip()
            body = await reader.readexactly(int(headers.get("content-length", "0")))
            self.requests.append({"method": method, "path": path, "headers": headers,
                                  "json": json.loads(body or b"{}")})
            script = self.scripts.pop(0) if self.scripts else [("head", 500, []), ("body", b"no script")]
            await self._play(script, writer)
        except (ConnectionError, asyncio.IncompleteReadError):
            self.client_gone.set()
        finally:
            try:
                writer.close()
            except Exception:
                pass

    async def _write(self, writer: asyncio.StreamWriter, data: bytes) -> None:
        if self.chunked and data:
            data = f"{len(data):x}\r\n".encode() + data + b"\r\n"
        writer.write(data)
        await writer.drain()
        # On a loopback socket a write to a peer that has closed often succeeds once; a
        # zero-length read tells us the peer is gone.
        if writer.is_closing():
            raise ConnectionResetError

    async def _play(self, script: list[tuple], writer: asyncio.StreamWriter) -> None:
        for step in script:
            op = step[0]
            if op == "head":
                _, status, hdrs = step
                reason = {200: "OK", 429: "Too Many Requests", 500: "Internal Server Error"}.get(status, "X")
                lines = [f"HTTP/1.1 {status} {reason}", "Connection: close"]
                lines += [f"{k}: {v}" for k, v in hdrs]
                lines.append("Transfer-Encoding: chunked" if self.chunked else "")
                writer.write(("\r\n".join(l for l in lines if l) + "\r\n\r\n").encode("latin-1"))
                await writer.drain()
            elif op == "sse":
                payload = step[1] if isinstance(step[1], str) else json.dumps(step[1])
                await self._write(writer, f"data: {payload}\n\n".encode("utf-8"))
            elif op == "body":
                await self._write(writer, step[1])
            elif op == "sleep":
                await asyncio.sleep(step[1])
            elif op == "close":
                await asyncio.sleep(0.05)  # let what was written reach the client first
                writer.transport.abort()
                return
        if self.chunked:
            writer.write(b"0\r\n\r\n")
            await writer.drain()


def delta(content: str | None = None, reasoning: str | None = None, finish: str | None = None,
          usage: dict | None = None) -> dict:
    d: dict = {}
    if content is not None:
        d["content"] = content
    if reasoning is not None:
        d["reasoning"] = reasoning
    obj: dict = {"id": "chatcmpl-test", "model": "fake-model",
                 "choices": [{"index": 0, "delta": d, "finish_reason": finish}]}
    if usage is not None:
        obj["x_groq"] = {"usage": usage}
    return obj
