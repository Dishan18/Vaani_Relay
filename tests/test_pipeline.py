"""Offline tests for relay.pipeline: scripted model replies, real retriever, real limiter logic.

Limiter buckets are built per test (real clock, short refill periods) so each rung of the degrade
ladder is reached deterministically from the limiter's predicted wait.
"""
from __future__ import annotations

import asyncio
import copy
import json
import re
import time
import unittest

from relay.fake_groq import FakeGroq
from relay.limiter import REQUESTS, TOKENS, Bucket, Limiter, ModelState, PromptEstimator
from relay.llm import CallRecord, Endpoint, LLMClient
from relay.pipeline import (BUSY, FULL, FULL_OTHER, LEAN, PASSAGES, Pipeline, check_citations, draft_messages, load_config,
                            plain_chat, split_sentences)
from relay.retrieve import Retriever

Q120, Q20, QW = "openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b"
RETRIEVER = Retriever.from_blocks(2400)
QUESTION = "Why divide the query-key dot products by sqrt(d_k) in scaled dot-product attention?"


def cfg(**ladder):
    c = copy.deepcopy(load_config())
    c["ladder"].update({"preferred_free_within_s": 0.3, "max_wait_s": 1.5, "deadline_s": 10.0,
                        "router_free_within_s": 0.3, **ladder})
    return c


def make_limiter(tpm_level: dict | None = None, period: float = 8.0) -> Limiter:
    """Each model: RPM 30/60 s, TPM 8000 per `period` s (fast refill for tests), RPD 1000/day."""
    t = time.monotonic()
    models = {}
    for m in (Q120, Q20, QW):
        lv = (tpm_level or {}).get(m, 8000)
        models[m] = ModelState(m, {"RPM": Bucket("RPM", REQUESTS, 30, 60.0, t, 30, "test"),
                                   "TPM": Bucket("TPM", TOKENS, 8000, period, t, lv, "test"),
                                   "RPD": Bucket("RPD", REQUESTS, 1000, 86400.0, t, 1000, "test")}, hold_cap=200)
    return Limiter(models, PromptEstimator(), time.monotonic)


def sources_of(user: str) -> dict[int, str]:
    out = {}
    for m in re.finditer(r"\[S(\d+)\] \([^\n]*\)\n(.*?)(?=\n\n\[S\d+\] \(|\Z)", user, re.S):
        out[int(m.group(1))] = m.group(2)
    return out


def cited_answer(user: str) -> str:
    """A draft that passes the gate: the first sentence of S1, cited."""
    s1 = sources_of(user)[1]
    first = re.split(r"(?<=[.!?])\s+", s1.replace("\n", " "))[0]
    return f"{first} [S1]"


class ScriptedChat:
    """chat(payload, phase=...) -> CallRecord, with replies chosen by ``script(step, model, payload)``,
    which returns text, or (status, text)."""

    def __init__(self, script):
        self.script = script
        self.calls: list[tuple[str, str]] = []

    async def __call__(self, payload, *, phase, timeouts=None):
        self.calls.append((phase, payload["model"]))
        out = self.script(phase, payload["model"], payload)
        status, text = out if isinstance(out, tuple) else (200, out)
        chars = sum(len(m["content"]) for m in payload["messages"])
        r = CallRecord(call_id="c", source="scripted", url="u", model=payload["model"], started_at="t",
                       status=status, outcome="ok" if status == 200 else "rate_limited")
        r.headers_ms, r.total_ms = 5.0, 20.0
        if status == 200:
            r.text = text
            r.usage = {"prompt_tokens": chars // 4, "completion_tokens": len(text) // 4 + 1, "queue_time": 0.0}
        else:
            r.error_body = json.dumps({"error": {"message": (
                f"Rate limit reached for model `{payload['model']}` in organization `org_REDACTED` service tier "
                "`on_demand` on requests per minute (RPM): Limit 30, Used 30, Requested 1. "
                "Please try again in 2s. Need more"), "type": "requests", "code": "rate_limit_exceeded"}})
        await asyncio.sleep(0)
        return r


def default_script(step, model, payload):
    user = payload["messages"][-1]["content"]
    if step == "understand":
        return json.dumps({"papers": ["attention"], "query": "scaled dot-product attention softmax gradients",
                           "alt_query": "scaling factor dot products"})
    if step == "draft":
        return cited_answer(user)
    if step == "verify":
        return '{"unsupported": []}'
    raise AssertionError(step)


def run(coro):
    return asyncio.run(coro)


class LadderTests(unittest.TestCase):
    def test_full_path_router_draft_gate_verifier(self):
        chat = ScriptedChat(default_script)
        r = run(Pipeline(chat, make_limiter(), RETRIEVER, cfg()).answer(QUESTION))
        self.assertEqual((r.kind, r.level), ("answer", FULL), r.notice)
        self.assertEqual(chat.calls, [("understand", QW), ("draft", Q120), ("verify", QW)])
        self.assertTrue(r.checks["gate"]["ok"])
        self.assertEqual(r.checks["verifier"]["ok"], True)
        names = [s.name for s in r.spans]
        for n in ("plan", "understand", "wait:understand", "call:understand", "retrieve", "decide", "draft",
                  "wait:draft", "call:draft", "check", "verify", "send"):
            self.assertIn(n, names)
        self.assertTrue(all(s.end_ms is not None and s.end_ms >= s.start_ms for s in r.spans))
        self.assertEqual(r.tokens_in, sum(c.prompt_tokens for c in r.model_calls))
        self.assertEqual(len(r.model_calls), 3)
        self.assertTrue(all(s["paper"] == "attention" for s in r.sources))
        self.assertEqual(r.notice, "")

    def test_preferred_busy_uses_other_model_full(self):
        chat = ScriptedChat(default_script)
        lim = make_limiter({Q120: 0}, period=60.0)          # 120b refills too slowly to be free in time
        r = run(Pipeline(chat, lim, RETRIEVER, cfg()).answer(QUESTION))
        self.assertEqual((r.kind, r.level), ("answer", FULL_OTHER))
        self.assertIn(("draft", Q20), chat.calls)
        self.assertNotIn(("draft", Q120), chat.calls)
        self.assertIn("fallback model", r.notice)

    def test_all_backed_up_gives_lean_without_router_or_verifier(self):
        chat = ScriptedChat(default_script)
        probe = make_limiter()
        hits = RETRIEVER.search(QUESTION, 3)
        rate = 8000 / 60.0
        # each drafter's TPM sits 1 s of refill short of the lean draft (a full one is many s away)
        lv = {m: probe.estimate(m, draft_messages(QUESTION, hits[:2])).tokens + 200 - rate * 1.0
              for m in (Q120, Q20, QW)}
        lim = make_limiter(lv, period=60.0)
        r = run(Pipeline(chat, lim, RETRIEVER, cfg()).answer(QUESTION))
        self.assertEqual((r.kind, r.level), ("answer", LEAN), r.notice)
        self.assertEqual([c[0] for c in chat.calls], ["draft"])
        self.assertLessEqual(len(r.sources), 2)
        self.assertIn("shorter answer", r.notice)
        self.assertGreater(r.budget_wait_ms, 0)

    def test_nothing_free_gives_passages_with_eta_and_no_model_call(self):
        chat = ScriptedChat(default_script)
        lim = make_limiter({Q120: 0, Q20: 0, QW: 0}, period=60.0)   # lean needs ~10 s of refill
        r = run(Pipeline(chat, lim, RETRIEVER, cfg()).answer(QUESTION))
        self.assertEqual((r.kind, r.level), ("passages", PASSAGES))
        self.assertEqual(chat.calls, [])
        self.assertGreater(r.eta_s, 1.5)
        self.assertIn("about", r.text)
        self.assertIn("[S1] 1706.03762v7", r.text)
        self.assertEqual(r.tokens_in + r.tokens_out, 0)

    def test_429_on_preferred_drafter_falls_back(self):
        def script(step, model, payload):
            if step == "draft" and model == Q120:
                return 429, ""
            return default_script(step, model, payload)
        chat = ScriptedChat(script)
        r = run(Pipeline(chat, make_limiter(), RETRIEVER, cfg()).answer(QUESTION))
        self.assertEqual((r.kind, r.level), ("answer", FULL_OTHER))
        self.assertEqual([c for c in chat.calls if c[0] == "draft"], [("draft", Q120), ("draft", Q20)])
        self.assertEqual(r.model_calls[1].prompt_tokens, 0)        # a 429 costs no tokens

    def test_internal_failure_is_busy(self):
        class Broken(Retriever):
            def search(self, *a, **k):
                raise RuntimeError("index gone")
        broken = Broken(RETRIEVER.pieces[:5], 2400)
        r = run(Pipeline(ScriptedChat(default_script), make_limiter(), broken, cfg()).answer(QUESTION))
        self.assertEqual((r.kind, r.level), ("busy", BUSY))
        self.assertRegex(r.text, r"retry in \d+ s")


class DecideAndCheckTests(unittest.TestCase):
    def test_off_topic_refused_by_router_without_draft(self):
        def script(step, model, payload):
            if step == "understand":
                return '{"papers": [], "query": "capital of France", "alt_query": "France capital city"}'
            raise AssertionError("no other call expected")
        chat = ScriptedChat(script)
        r = run(Pipeline(chat, make_limiter(), RETRIEVER, cfg()).answer("What is the capital of France?"))
        self.assertEqual(r.kind, "refusal")
        self.assertEqual(chat.calls, [("understand", QW)])

    def test_no_matching_passage_refused_without_model_call_when_router_fails(self):
        def script(step, model, payload):
            if step == "understand":
                return "sorry, no JSON"
            raise AssertionError("no draft expected")
        chat = ScriptedChat(script)
        r = run(Pipeline(chat, make_limiter(), RETRIEVER, cfg()).answer("How do I bake sourdough bread at home?"))
        self.assertEqual(r.kind, "refusal")
        self.assertEqual([c[0] for c in chat.calls], ["understand"])
        self.assertEqual(r.checks["evidence"]["best"], 0.0)

    def test_decide_gate_uses_the_router_rewrite_too(self):
        """Q14: the question's words miss the routing passage; the router's rewrite doesn't."""
        q = "How does H-Net decide where to cut the byte stream into pieces?"

        def script(step, model, payload):
            if step == "understand":
                return json.dumps({"papers": ["dynamic_chunking"],
                                   "query": "routing module boundary probability cosine similarity chunking layer",
                                   "alt_query": "dynamic chunking boundaries"})
            if step == "draft":
                return cited_answer(payload["messages"][-1]["content"])
            return default_script(step, model, payload)
        chat = ScriptedChat(script)
        r = run(Pipeline(chat, make_limiter(), Retriever.from_blocks(1200), cfg()).answer(q))
        self.assertIn(("draft", Q120), chat.calls)               # not refused at the gate
        decide = next(s for s in r.spans if s.name == "decide")
        self.assertIsNotNone(decide.detail["rewrite"])

    def test_uncited_draft_gets_one_redraft(self):
        def script(step, model, payload):
            user = payload["messages"][-1]["content"]
            if step == "draft" and "failed the check" not in user:
                return "Scaling keeps the softmax gradients large."
            return default_script(step, model, payload)
        chat = ScriptedChat(script)
        r = run(Pipeline(chat, make_limiter(), RETRIEVER, cfg()).answer(QUESTION))
        self.assertEqual(r.kind, "answer")
        self.assertEqual(r.checks["attempts"], 2)
        self.assertEqual([c for c in chat.calls if c[0] == "draft"], [("draft", Q120), ("draft", Q120)])

    def test_fullwidth_citations_pass_the_gate_and_are_normalised(self):
        def script(step, model, payload):
            if step == "draft":
                return cited_answer(payload["messages"][-1]["content"]).replace("[S1]", "【S1】")
            return default_script(step, model, payload)
        chat = ScriptedChat(script)
        r = run(Pipeline(chat, make_limiter(), RETRIEVER, cfg()).answer(QUESTION))
        self.assertEqual((r.kind, r.checks["attempts"]), ("answer", 1))
        self.assertIn("[S1]", r.text)
        self.assertNotIn("【", r.text)

    def test_invented_number_twice_falls_back_to_passages(self):
        def script(step, model, payload):
            if step == "draft":
                return "The paper uses 9999 attention heads [S1]."
            return default_script(step, model, payload)
        r = run(Pipeline(ScriptedChat(script), make_limiter(), RETRIEVER, cfg()).answer(QUESTION))
        self.assertEqual(r.kind, "passages")
        self.assertIn("failed the citation check", r.notice)

    def test_verifier_rejection_triggers_redraft(self):
        n = {"verify": 0}

        def script(step, model, payload):
            if step == "verify":
                n["verify"] += 1
                return '{"unsupported": [1]}' if n["verify"] == 1 else '{"unsupported": []}'
            return default_script(step, model, payload)
        r = run(Pipeline(ScriptedChat(script), make_limiter(), RETRIEVER, cfg()).answer(QUESTION))
        self.assertEqual((r.kind, r.checks["attempts"]), ("answer", 2))

    def test_not_in_papers_is_a_refusal_and_drops_guessed_numbers(self):
        def script(step, model, payload):
            if step == "draft":
                return "NOT_IN_PAPERS: the paper reports BLEU, but a p95 latency of 123 ms is not given."
            return default_script(step, model, payload)
        r = run(Pipeline(ScriptedChat(script), make_limiter(), RETRIEVER, cfg()).answer(
            "What was the Transformer's 95th percentile inference latency on WMT 2014 English-German?"))
        self.assertEqual(r.kind, "refusal")
        self.assertEqual(r.text, "The papers don't say.")

    def test_verifier_skipped_without_headroom(self):
        lim = make_limiter({QW: 1000, Q20: 1000}, period=600.0)   # verifiers far below 30% TPM
        chat = ScriptedChat(default_script)
        r = run(Pipeline(chat, lim, RETRIEVER, cfg()).answer(QUESTION))
        self.assertEqual(r.kind, "answer")
        self.assertNotIn("verify", [c[0] for c in chat.calls])
        self.assertIn("citations only", r.notice)


class GateTests(unittest.TestCase):
    def setUp(self):
        self.hits = RETRIEVER.search("Transformer big training cost FLOPs Table 2", 3, ["attention"])

    def test_number_must_be_in_a_cited_source(self):
        i = next(n for n, h in enumerate(self.hits, 1) if "2.3 · 10^19" in h.piece.text)
        self.assertTrue(check_citations(f"The big model cost 2.3 · 10^19 FLOPs to train [S{i}].", self.hits)["ok"])
        bad = check_citations(f"The big model cost 4.1 · 10^19 FLOPs to train [S{i}].", self.hits)
        self.assertFalse(bad["ok"])
        self.assertIn("number", bad["failures"][0]["reason"])

    def test_missing_or_invalid_citation_fails(self):
        self.assertEqual(check_citations("The big model was trained for days.", self.hits)["failures"][0]["reason"],
                         "no citation")
        self.assertIn("wasn't given", check_citations("The big model was trained [S9].", self.hits)["failures"][0]["reason"])

    def test_quote_must_exist(self):
        r = check_citations('The paper says "attention is a magic trick for everything" here [S1].', self.hits)
        self.assertFalse(r["ok"])

    def test_sentence_split_keeps_citations_with_their_sentence(self):
        s = split_sentences("First claim here [S1]. Second claim follows [S2][S3].\n- Third claim in a list [S1].")
        self.assertEqual(s, ["First claim here [S1].", "Second claim follows [S2][S3].", "Third claim in a list [S1]."])


class HttpIntegrationTest(unittest.TestCase):
    def test_pipeline_over_http_against_the_fake_server(self):
        def responder(p):
            sysmsg, user = p["messages"][0]["content"], p["messages"][-1]["content"]
            if sysmsg.startswith("You route"):
                return '{"papers": ["attention"], "query": "scaled dot-product attention", "alt_query": "softmax"}'
            if sysmsg.startswith("You check"):
                return '{"unsupported": []}'
            return cited_answer(user)

        async def go():
            async with FakeGroq(responder=responder) as fake:
                client = LLMClient(Endpoint(fake.base_url, source="stand-in"))
                pipe = Pipeline(plain_chat(client), make_limiter(), RETRIEVER, cfg(deadline_s=20.0))
                return await pipe.answer(QUESTION)
        r = run(go())
        self.assertEqual(r.kind, "answer", (r.notice, [s.outcome for s in r.spans]))
        self.assertEqual([c.step for c in r.model_calls], ["understand", "draft", "verify"])
        self.assertTrue(all(c.status == 200 and not c.tokens_estimated for c in r.model_calls))


if __name__ == "__main__":
    unittest.main()
