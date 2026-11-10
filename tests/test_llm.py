"""Offline tests for relay.llm against a local scripted server (no network, no Groq budget).

Run:  .venv\\Scripts\\python -m unittest discover -s tests -t .
"""
from __future__ import annotations

import asyncio
import socket
import unittest

from relay.llm import (CANCELLED, CONNECT_ERROR, OK, PROTOCOL_ERROR, RATE_LIMITED, STREAM_ERROR,
                       TIMEOUT_DEADLINE, TIMEOUT_FIRST_VISIBLE, TIMEOUT_STALL, Endpoint,
                       LLMClient, Timeouts, _ThinkSplitter)
from tests.fake_openai_server import FakeOpenAIServer, delta

RATE_HEADERS = [
    ("x-ratelimit-limit-requests", "1000"),
    ("x-ratelimit-limit-tokens", "6000"),
    ("x-ratelimit-remaining-requests", "999"),
    ("x-ratelimit-remaining-tokens", "5880"),
    ("x-ratelimit-reset-requests", "1m26.4s"),
    ("x-ratelimit-reset-tokens", "1.2s"),
    ("x-request-id", "req_test123"),
]
USAGE = {"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15}
PAYLOAD = {"model": "fake-model", "messages": [{"role": "user", "content": "hi there"}], "max_tokens": 16}
FAST = Timeouts(first_visible_s=2.0, stall_s=2.0, deadline_s=5.0)
# asyncio timers run on time.monotonic, which ticks every ~15.6 ms on Windows, while records use
# perf_counter. A timeout can therefore fire up to one tick early as seen in total_ms.
TICK_MS = 20


def ok_script():
    return [
        ("head", 200, [("Content-Type", "text/event-stream")] + RATE_HEADERS),
        ("sleep", 0.05),
        ("sse", delta(reasoning="Let me think.")),
        ("sleep", 0.05),
        ("sse", delta(content="Hello")),
        ("sse", delta(content=" world")),
        ("sse", delta(finish="stop", usage=USAGE)),
        ("sse", "[DONE]"),
    ]


class LLMClientTests(unittest.IsolatedAsyncioTestCase):

    async def _client(self, srv, timeouts=FAST):
        return LLMClient(Endpoint(srv.base_url, api_key="sk-test", source="stand-in"), timeouts)

    async def test_stream_ok_records_every_timestamp(self):
        for chunked in (False, True):
            with self.subTest(chunked=chunked):
                async with FakeOpenAIServer(chunked=chunked) as srv:
                    srv.scripts.append(ok_script())
                    deltas, sunk = [], []
                    rec = await (await self._client(srv)).stream_chat(
                        PAYLOAD, on_delta=lambda k, t: deltas.append((k, t)), record_sink=sunk.append)
                self.assertEqual(rec.outcome, OK, rec.error)
                self.assertEqual(rec.text, "Hello world")
                self.assertEqual(rec.status, 200)
                self.assertEqual(rec.finish_reason, "stop")
                self.assertEqual(rec.usage, USAGE)
                self.assertEqual(rec.reasoning_chars, len("Let me think."))
                # ordering of the milestones
                self.assertLessEqual(rec.connect_ms, rec.headers_ms)
                self.assertLessEqual(rec.headers_ms, rec.first_byte_ms)
                self.assertLess(rec.first_byte_ms, rec.first_visible_ms)
                self.assertLess(rec.first_reasoning_ms, rec.first_visible_ms)
                self.assertGreaterEqual(rec.first_visible_ms - rec.first_reasoning_ms, 40)  # the 50 ms sleep
                self.assertLessEqual(rec.first_visible_ms, rec.total_ms)
                # every chunk recorded, in order
                self.assertEqual([c.kind for c in rec.chunks], ["reasoning", "content", "content", "usage", "done"])
                self.assertEqual([c.t_ms for c in rec.chunks], sorted(c.t_ms for c in rec.chunks))
                # headers verbatim, in order; non-rate-limit headers kept separately
                self.assertEqual(rec.rate_limit_headers, [list(h) for h in RATE_HEADERS[:6]])
                self.assertEqual(rec.header("x-ratelimit-reset-requests"), "1m26.4s")
                self.assertEqual(rec.other_headers["x-request-id"], "req_test123")
                self.assertEqual(deltas, [("reasoning", "Let me think."), ("content", "Hello"), ("content", " world")])
                self.assertEqual(sunk, [rec])
                self.assertEqual(rec.source, "stand-in")

    async def test_request_shape(self):
        async with FakeOpenAIServer() as srv:
            srv.scripts.append(ok_script())
            await (await self._client(srv)).stream_chat(PAYLOAD)
            req = srv.requests[0]
        self.assertEqual((req["method"], req["path"]), ("POST", "/v1/chat/completions"))
        self.assertEqual(req["headers"]["authorization"], "Bearer sk-test")
        self.assertEqual(req["headers"]["accept"], "text/event-stream")
        self.assertIs(req["json"]["stream"], True)
        self.assertEqual(req["json"]["messages"], PAYLOAD["messages"])

    async def test_api_key_never_in_record(self):
        async with FakeOpenAIServer() as srv:
            srv.scripts.append(ok_script())
            client = await self._client(srv)
            rec = await client.stream_chat(PAYLOAD)
        self.assertNotIn("sk-test", repr(rec.to_dict()))
        self.assertNotIn("sk-test", repr(client.endpoint))

    async def test_429_keeps_headers_and_whole_body_with_org_redacted(self):
        body = (b'{"error":{"message":"Rate limit reached for model `fake-model` in organization '
                b'`org_01abcDEF` service tier `on_demand` on tokens per minute (TPM): Limit 6000, '
                b'Used 5990, Requested 120. Please try again in 1.1s.","type":"tokens","code":"rate_limit_exceeded"}}')
        async with FakeOpenAIServer() as srv:
            srv.scripts.append([("head", 429, [("retry-after", "2"), ("Content-Type", "application/json"),
                                               ("Content-Length", str(len(body)))] + RATE_HEADERS[:6]),
                                ("body", body)])
            rec = await (await self._client(srv)).stream_chat(PAYLOAD)
        self.assertEqual(rec.outcome, RATE_LIMITED)
        self.assertEqual(rec.status, 429)
        self.assertEqual(rec.header("retry-after"), "2")
        self.assertIn("org_REDACTED", rec.error_body)
        self.assertNotIn("org_01abcDEF", rec.error_body)
        self.assertIn("Please try again in 1.1s.", rec.error_body)
        self.assertEqual(len(rec.error_body), len(body.decode()) - len("org_01abcDEF") + len("org_REDACTED"))
        self.assertIsNone(rec.first_visible_ms)

    async def test_first_visible_timeout_cancels_call(self):
        # Reasoning keeps arriving (so it is not a stall) but no visible token ever comes.
        script = [("head", 200, [])] + [s for _ in range(40) for s in
                                        (("sse", delta(reasoning="hmm ")), ("sleep", 0.05))]
        async with FakeOpenAIServer() as srv:
            srv.scripts.append(script)
            rec = await (await self._client(srv, Timeouts(first_visible_s=0.3, stall_s=1.0, deadline_s=5.0))
                         ).stream_chat(PAYLOAD)
            await asyncio.wait_for(srv.client_gone.wait(), 2.0)  # the server saw us leave
        self.assertEqual(rec.outcome, TIMEOUT_FIRST_VISIBLE)
        self.assertIsNotNone(rec.first_reasoning_ms)
        self.assertIsNone(rec.first_visible_ms)
        self.assertGreaterEqual(rec.total_ms, 300 - TICK_MS)
        self.assertLess(rec.total_ms, 800)

    async def test_first_visible_timeout_before_headers(self):
        async with FakeOpenAIServer() as srv:
            srv.scripts.append([("sleep", 2.0), ("head", 200, [])])
            rec = await (await self._client(srv, Timeouts(first_visible_s=0.2, stall_s=None, deadline_s=5.0))
                         ).stream_chat(PAYLOAD)
        self.assertEqual(rec.outcome, TIMEOUT_FIRST_VISIBLE)
        self.assertEqual(rec.error, "timeout_first_visible during headers")
        self.assertIsNone(rec.headers_ms)

    async def test_stall_timeout(self):
        async with FakeOpenAIServer() as srv:
            srv.scripts.append([("head", 200, []), ("sse", delta(content="Partial")), ("sleep", 2.0),
                                ("sse", delta(content=" never"))])
            rec = await (await self._client(srv, Timeouts(first_visible_s=2.0, stall_s=0.2, deadline_s=5.0))
                         ).stream_chat(PAYLOAD)
        self.assertEqual(rec.outcome, TIMEOUT_STALL)
        self.assertEqual(rec.text, "Partial")  # partial answer kept
        self.assertLess(rec.total_ms, 1000)

    async def test_deadline(self):
        # Slow but steady: never stalls, visible token early, still too slow overall.
        script = [("head", 200, [])] + [s for _ in range(60) for s in
                                        (("sse", delta(content="x")), ("sleep", 0.05))]
        async with FakeOpenAIServer() as srv:
            srv.scripts.append(script)
            rec = await (await self._client(srv, Timeouts(first_visible_s=1.0, stall_s=1.0, deadline_s=0.4))
                         ).stream_chat(PAYLOAD)
        self.assertEqual(rec.outcome, TIMEOUT_DEADLINE)
        self.assertGreaterEqual(rec.total_ms, 400 - TICK_MS)
        self.assertLess(rec.total_ms, 900)
        self.assertTrue(rec.text.startswith("x"))

    async def test_caller_cancellation_still_reports_record(self):
        script = [("head", 200, []), ("sse", delta(content="Hi")), ("sleep", 5.0)]
        sunk = []
        async with FakeOpenAIServer() as srv:
            srv.scripts.append(script)
            client = await self._client(srv)
            task = asyncio.create_task(client.stream_chat(PAYLOAD, record_sink=sunk.append))
            await asyncio.sleep(0.2)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        self.assertEqual(len(sunk), 1)
        self.assertEqual(sunk[0].outcome, CANCELLED)
        self.assertEqual(sunk[0].text, "Hi")
        self.assertIsNotNone(sunk[0].total_ms)

    async def test_connect_error(self):
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
        s.close()  # nothing listens here now
        # Windows retries a refused SYN to localhost for ~2 s before giving up, so allow more time.
        rec = await LLMClient(Endpoint(f"http://127.0.0.1:{port}/v1"),
                              Timeouts(first_visible_s=10, stall_s=None, deadline_s=10)).stream_chat(PAYLOAD)
        self.assertEqual(rec.outcome, CONNECT_ERROR)
        self.assertIsNone(rec.connect_ms)

    async def test_eof_without_done_is_protocol_error(self):
        async with FakeOpenAIServer() as srv:
            srv.scripts.append([("head", 200, []), ("sse", delta(content="cut")), ("close",)])
            rec = await (await self._client(srv)).stream_chat(PAYLOAD)
        self.assertEqual(rec.outcome, PROTOCOL_ERROR)
        self.assertEqual(rec.text, "cut")

    async def test_error_event_mid_stream(self):
        async with FakeOpenAIServer() as srv:
            srv.scripts.append([("head", 200, []), ("sse", delta(content="a")),
                                ("sse", {"error": {"message": "overloaded for org_99zz", "type": "server_error"}})])
            rec = await (await self._client(srv)).stream_chat(PAYLOAD)
        self.assertEqual(rec.outcome, STREAM_ERROR)
        self.assertIn("org_REDACTED", rec.error_body)

    async def test_think_tags_split_across_chunks(self):
        async with FakeOpenAIServer() as srv:
            srv.scripts.append([("head", 200, []),
                                ("sse", delta(content="<thi")), ("sleep", 0.03),
                                ("sse", delta(content="nk>plan it</th")), ("sleep", 0.03),
                                ("sse", delta(content="ink>\n\n")), ("sleep", 0.05),
                                ("sse", delta(content="Answer")),
                                ("sse", delta(finish="stop")), ("sse", "[DONE]")])
            rec = await (await self._client(srv)).stream_chat(PAYLOAD)
        self.assertEqual(rec.outcome, OK)
        self.assertEqual(rec.reasoning_chars, len("plan it"))
        self.assertEqual(rec.text, "\n\nAnswer")
        answer_chunk = [c for c in rec.chunks if c.kind == "content"][-1]
        self.assertEqual(rec.first_visible_ms, answer_chunk.t_ms)  # whitespace is not "visible"
        self.assertLess(rec.first_reasoning_ms, rec.first_visible_ms)


class ThinkSplitterTests(unittest.TestCase):
    def test_partial_tags(self):
        s = _ThinkSplitter()
        out = []
        for piece in ["a<", "think", ">b</", "think>c<x"]:
            out += s.feed(piece)
        out += s.feed("", final=True)
        joined = {}
        for k, t in out:
            joined[k] = joined.get(k, "") + t
        self.assertEqual(joined, {"content": "ac<x", "reasoning": "b"})


if __name__ == "__main__":
    unittest.main()
