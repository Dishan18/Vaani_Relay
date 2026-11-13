"""Claims mode: citation metadata carried by the drafter, checked per claim, streamed (no model calls)."""
from __future__ import annotations

import asyncio
import copy
import json
import re
import unittest

from relay.claims import ClaimStream, check_claim, numbers, render, words
from relay.pipeline import FULL, PASSAGES, Pipeline, load_config, load_production_config
from tests.test_pipeline import QUESTION, RETRIEVER, ScriptedChat, make_limiter, sources_of


def claims_cfg():
    c = copy.deepcopy(load_production_config())
    c["retrieval"]["mode"] = "bm25"        # these tests are about claims, not retrieval
    c["ladder"].update({"preferred_free_within_s": 0.3, "max_wait_s": 1.5, "deadline_s": 10.0,
                        "router_free_within_s": 0.3})
    return c


def claim_line(text, src, quote):
    return json.dumps({"text": text, "sources": [src], "quote": quote})


def first_sentence(user: str, i: int = 1) -> str:
    s = sources_of(user)[i]
    return re.split(r"(?<=[.!?])\s+", s.replace("\n", " "))[0]


class StreamingChat(ScriptedChat):
    """Feeds content to on_delta in small pieces, like a real stream."""
    async def __call__(self, payload, *, phase, timeouts=None, on_delta=None):
        r = await super().__call__(payload, phase=phase, timeouts=timeouts)
        if on_delta is not None and r.status == 200:
            for i in range(0, len(r.text), 7):
                on_delta("content", r.text[i:i + 7])
                await asyncio.sleep(0)
        return r


def script_with(draft_fn):
    def script(step, model, payload):
        user = payload["messages"][-1]["content"]
        if step == "understand":
            return json.dumps({"papers": ["attention"], "query": "scaled dot-product attention softmax",
                               "alt_query": "scaling factor", "hyde": "dividing by sqrt(d_k) keeps the softmax gradients"})
        if step == "draft":
            return draft_fn(user)
        if step == "verify":
            return '{"unsupported": []}'
        raise AssertionError(step)
    return script


class CheckClaimTests(unittest.TestCase):
    hits = RETRIEVER.search(QUESTION, 3)

    def test_quote_must_be_in_a_cited_source(self):
        p = self.hits[0].piece.text
        q = " ".join(p.split()[:10])
        self.assertTrue(check_claim({"text": "A claim about attention.", "sources": [1], "quote": q}, self.hits).ok)
        bad = check_claim({"text": "A claim about attention.", "sources": [1], "quote": "words that are nowhere in it at all"},
                          self.hits)
        self.assertFalse(bad.ok)
        self.assertIn("quote not found", bad.reason)

    def test_quote_ignores_case_spacing_and_hyphenation(self):
        p = self.hits[0].piece.text
        q = " ".join(p.split()[:10]).upper().replace(" ", "   ")
        self.assertTrue(check_claim({"text": "A claim about attention.", "sources": [1], "quote": q}, self.hits).ok)

    def test_unknown_source_and_missing_number(self):
        q = " ".join(self.hits[0].piece.text.split()[:8])
        self.assertIn("wasn't given", check_claim({"text": "A claim here.", "sources": [9], "quote": q}, self.hits).reason)
        c = check_claim({"text": "It uses 987654 heads.", "sources": [1], "quote": q}, self.hits)
        self.assertEqual(c.reason, "number 987654 is not in the cited sources")

    def test_whole_numbers_not_digits(self):
        # the first real run rejected "0.1" because the check compared "0" and "1"
        self.assertEqual(numbers("a rate of P_drop = 0.1 and 36,000 steps"), ["0.1", "36000"])

    def test_model_written_markers_are_stripped(self):
        q = " ".join(self.hits[0].piece.text.split()[:8])
        c = check_claim({"text": "Attention is computed. [S1]", "sources": ["S1"], "quote": q}, self.hits)
        self.assertEqual((c.ok, c.text, c.sources), (True, "Attention is computed.", [1]))
        self.assertEqual(render([c]), "Attention is computed. [S1]")

    def test_stream_releases_each_complete_line(self):
        q = " ".join(self.hits[0].piece.text.split()[:8])
        s = ClaimStream(self.hits)
        line = claim_line("One claim about it.", 1, q)
        self.assertEqual(s.feed(line[:20]), [])
        got = s.feed(line[20:] + "\n" + line[:5])
        self.assertEqual(len(got), 1)
        self.assertTrue(got[0].ok)

    def test_refusal_line(self):
        s = ClaimStream(self.hits)
        s.feed('{"not_in_papers": "no latency figures"}\n', final=True)
        self.assertEqual(s.refusal, "no latency figures")

    def test_objects_on_one_line_and_braces_in_strings(self):
        # gpt-oss-20b wrote several claims on one line ("}{"); a quote may contain braces
        q = " ".join(self.hits[0].piece.text.split()[:8])
        a = claim_line("First claim here.", 1, q)
        b = json.dumps({"text": "Second {with braces} claim.", "sources": [1], "quote": q})
        s = ClaimStream(self.hits)
        got = s.feed(a + b[:30])
        self.assertEqual(len(got), 1)
        got += s.feed(b[30:], final=True)
        self.assertEqual([c.text for c in got], ["First claim here.", "Second {with braces} claim."])

    def test_glued_words_ellipsis_and_table_quotes(self):
        from relay.claims import quote_found
        src = ("whereF represents the fraction of vectors actually selected, G denotes the average boundary "
               "probability. Self-Attention O(n^2 · d) O(1) O(1) Recurrent O(n · d^2) O(n) O(n)")
        self.assertTrue(quote_found("F represents the fraction of vectors actually selected", src))
        self.assertTrue(quote_found("F represents the fraction... G denotes the average boundary probability", src))
        self.assertTrue(quote_found("Self-Attention O(1) ... Recurrent O(n)", src))
        self.assertFalse(quote_found("F represents the fraction of tokens kept by the router", src))
        self.assertFalse(quote_found("Recurrent O(1) Self-Attention O(n) is faster", src))

    def test_words(self):
        self.assertEqual(words("Multi-\n head  ATTENTION"), ["multihead", "attention"])


class ClaimsPipelineTests(unittest.TestCase):
    def test_claims_answer_with_metadata(self):
        def draft(user):
            f = first_sentence(user)
            return claim_line(f, 1, " ".join(f.split()[:8])) + "\n"
        r = asyncio.run(Pipeline(ScriptedChat(script_with(draft)), make_limiter(), RETRIEVER, claims_cfg()).answer(QUESTION))
        self.assertEqual((r.kind, r.level), ("answer", FULL), r.notice)
        self.assertEqual(len(r.claims), 1)
        self.assertEqual(r.claims[0]["sources"], ["S1"])
        self.assertTrue(r.text.endswith("[S1]"))

    def test_bad_claim_dropped_good_kept(self):
        def draft(user):
            f = first_sentence(user)
            return (claim_line(f, 1, " ".join(f.split()[:8])) + "\n"
                    + claim_line("An invented claim.", 1, "this quote is not in any source text") + "\n")
        r = asyncio.run(Pipeline(ScriptedChat(script_with(draft)), make_limiter(), RETRIEVER, claims_cfg()).answer(QUESTION))
        self.assertEqual(r.kind, "answer")
        self.assertEqual(len(r.claims), 1)
        self.assertEqual(len(r.checks["dropped"]), 1)
        self.assertNotIn("invented", r.text)
        self.assertIn("removed", r.notice)

    def test_no_good_claim_redraft_then_passages(self):
        chat = ScriptedChat(script_with(lambda u: "not json at all\n"))
        r = asyncio.run(Pipeline(chat, make_limiter(), RETRIEVER, claims_cfg()).answer(QUESTION))
        self.assertEqual((r.kind, r.level), ("passages", PASSAGES))
        self.assertEqual([c for c in chat.calls if c[0] == "draft"].__len__(), 2)

    def test_streaming_releases_claims_before_the_end(self):
        def draft(user):
            f1, f2 = first_sentence(user, 1), first_sentence(user, 2)
            return (claim_line(f1, 1, " ".join(f1.split()[:8])) + "\n" + claim_line(f2, 2, " ".join(f2.split()[:8])) + "\n")
        seen = []
        r = asyncio.run(Pipeline(StreamingChat(script_with(draft)), make_limiter(), RETRIEVER, claims_cfg())
                        .answer(QUESTION, on_claim=lambda c: seen.append(c["text"])))
        self.assertEqual(len(seen), 2)
        self.assertEqual(r.kind, "answer")
        self.assertIsNotNone(r.first_token_ms)
        self.assertLessEqual(r.first_token_ms, r.released_ms)
        self.assertIn("streamed", r.notice)

    def test_refusal_in_claims_mode(self):
        r = asyncio.run(Pipeline(ScriptedChat(script_with(lambda u: '{"not_in_papers": "no such figure"}\n')),
                                 make_limiter(), RETRIEVER, claims_cfg()).answer(QUESTION))
        self.assertEqual(r.kind, "refusal")

    def test_submitted_config_is_unchanged(self):
        self.assertNotIn("draft", load_config())
        self.assertEqual(load_config()["retrieval"]["piece_chars"], 2400)
        self.assertEqual(load_production_config()["draft"]["format"], "claims")


if __name__ == "__main__":
    unittest.main()
