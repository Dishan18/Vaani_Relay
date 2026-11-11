# The Relay

<!-- Template. Every number comes from docs/writeup/facts.json ({{key}}) or results/tables.md ({{table:...}}).
     Render: scripts/render_writeup.py. Check: scripts/check_numbers.py (fails on any digit typed here). -->

## What I built and why

Python/asyncio on Groq's free tier: a router (qwen3.8-27b) picks the paper and rewrites the question; BM25 fetches {{cfg.k_full}} pieces of {{cfg.piece_chars}} characters; a decide step refuses on weak evidence; gpt-oss-120b drafts with citations; a deterministic gate checks every sentence's citation, numbers and quotes, and a verifier on another model checks support when budget allows. Nothing is released before the checks.

**The decision that shaped it most: an answer ladder with a no-model floor.** A limiter seeded with the measured limits predicts whether a model has room within {{cfg.max_wait_s}} s; if not, the request steps down instead of queueing: another model, a lean answer ({{cfg.k_lean}} pieces, no router), then cited passages picked with no model call plus an estimate of when a written answer is possible. The floor costs no budget, so budget alone never leaves a person without a reply: with the daily budget gone it answered in {{daily_exhausted.standin.ttft_p50}} ms at p50 (stand-in).

## The real service and my stand-in

{{table:Table 1}}

Found by a probe of {{probe.calls}} real calls (headers, a burst past the minute wall, recovery polls, rising concurrency) and the bodies of real 429s.

The stand-in speaks Groq's API, headers and 429 wording with the same per-model buckets, draws first-byte time from a lognormal fitted to {{cal.ttfb_n}} probe calls (gpt-oss-120b p50 {{cal.ttfb_p50}} ms), streams at {{cal.decode_tps}} tokens/s, slows under load, stalls and fails on demand, and replays real model replies. **Least like Groq:** it replays drafts that already passed the check, so it never shows a gate failure. On identical arrivals the real steady run gave {{real_steady.real.written_pct}}% a written answer and its twin {{real_steady.standin.written_pct}}%, the gap being passages after failed checks; in the burst, {{real_burst.real.written_pct}}% against {{real_burst.standin.written_pct}}%, with real p95 first token {{real_burst.real.ttft_p95}} ms against {{real_burst.standin.ttft_p95}} ms.

## Results

{{table:Table 2a}}

{{table:Table 2b}}

Rows marked real are real runs; the rest are the stand-in; percentiles are nearest-rank. Rate-limited means budget caused the fallback; passages after failed checks have their own column. **The number we could not have guessed:** qwen3.8-27b also caps output at {{lim.qwen_otpm}} tokens a minute, {{lim.otpm_share}}% of its documented {{lim.qwen_tpm_doc}} TPM, and input at {{lim.qwen_itpm}}; neither is documented or in a header, both surfaced only in real 429 bodies. Quality on {{q.n}} questions I wrote ({{q.unanswerable}} unanswerable from the papers): {{q3.large.correct_pct}}% correct and cited, {{q3.large.refused_pct}}% correctly refused, {{q3.large.tokens}} tokens per answer (Table 3).

## Where the time goes

{{slow.request_id}} ({{slow.qid}}, real burst) took {{slow.total_ms}} ms, all before the first token, and never got one: the person's patience was {{slow.patience_s}} s. scripts/explain_request.py splits its recorded spans into waiting {{slow.waiting_ms}} ms, working {{slow.working_ms}} ms, retrying {{slow.retrying_ms}} ms, local {{slow.local_ms}} ms: it waited {{slow.first_wait_ms}} ms for budget on gpt-oss-120b, drafted in {{slow.draft_ms}} ms, the citation check returned {{slow.check}} (no reason recorded), and a second draft waited {{slow.last_wait_ms}} ms for {{slow.last_wait_on}} until the person left. No call failed, so the split counts that as waiting; in effect it was a retry, and the person got nothing.

## When a limit is hit

**Per minute** (stand-in overload, {{load.overload_rate}} arrivals a minute): {{overload.standin.written_pct}}% got a written answer ({{overload.standin.weaker_pct}}% from a fallback model or the lean step), {{overload.standin.passages_budget_pct}}% got passages, {{overload.standin.gave_up_pct}}% gave up. Delayed: whoever meets an empty bucket, up to the ladder's limit (queue wait p95 {{overload.standin.queue_wait_p95}} ms); dropped: nobody, though some leave first. Real burst: {{real_burst.real.written_pct}}% written, {{real_burst.real.passages_budget_pct}}% passages for budget, {{real_burst.real.passages_gate_pct}}% after failed checks, {{real_burst.real.gave_up_pct}}% gave up.

**Per day** (stand-in, daily budget gone): {{daily_exhausted.standin.written_pct}}% written, {{daily_exhausted.standin.passages_budget_pct}}% passages: with no daily room anywhere the ladder goes straight to the floor rather than hold people until a reset.

## Retrieval: quality versus cost

{{table:Table 3}}

I chose **{{q3.chosen}}**: most correct ({{q3.large.correct_pct}}%) at {{q3.large.tokens}} tokens per answer. Re-ranking kept medium's correctness ({{q3.medium_rerank.correct_pct}}%) at fewer tokens ({{q3.medium_rerank.tokens}} against {{q3.medium.tokens}}); small_lean is cheapest ({{q3.small_lean.tokens}}) but {{q3.small_lean.correct_pct}}% correct. Near the limit, the lean step fetches {{cfg.k_lean}} pieces and skips the router; at the floor the passages are the reply.

## How this library will mislead a naive retriever

Plain BM25 on the raw question, {{trap.size}}-character pieces (data/retrieval/traps.md):

- **Different words for the same idea.** "{{trap.Q14.question}}" shares {{trap.Q14.shared}} of its {{trap.Q14.terms}} terms with the passage that answers it; BM25 never ranks it, and the router's rewrite has not fixed it ({{trap.Q14.correct}} of {{trap.Q14.asked}} settings correct).
- **Text about a table outranks the table.** "{{trap.Q5.question}}": BM25 ranks {{trap.Q5.top_cite}} first, which discusses that table but lacks the figure; the piece holding the table is rank {{trap.Q5.rank}}.
- **Unanswerable questions that look answerable.** "{{trap.Q17.question}}": the top hit is from the other paper ({{trap.Q17.top_paper}}, {{trap.Q17.top_cite}}), sharing only "{{trap.Q17.shared_terms}}", and a piece with a number a reply could borrow is at rank {{trap.Q17.distractor_rank}}. The decide gate refused it in {{trap.Q17.refused}} of {{trap.Q17.asked}} settings.

## Retries

**Ships** (the stricter policy below is {{cfg.retry_enabled}}): a step with no predicted room moves to the next model before sending anything; after a sent call fails (429, 5xx, timeout) it tries the next model in the fallback order, each model at most once per step, then steps down the ladder. **Tested and rejected:** {{cfg.retry_max}} retry per step on a different model, never after a daily-limit 429, within {{cfg.retry_budget_pct}}% of recent calls. On identical stand-in arrivals (steady, burst, overload, chaos; two seeds) it left {{r4.retry_on.s7.no_written}} and {{r4.retry_on.s8.no_written}} people without a written answer, against {{r4.retry_off.s7.no_written}} and {{r4.retry_off.s8.no_written}}. **Retrying made it worse:** a naive client (no limiter; {{naive.retries}} retries on the same model, {{naive.backoff_s}} s apart) on the same stand-in overload arrivals left {{r4b.naive.s7.no_written}} and {{r4b.naive.s8.no_written}} of {{r4b.naive.s7.n}} without a written answer, against {{r4b.pipe.s7.no_written}} and {{r4b.pipe.s8.no_written}}, drawing {{r4b.naive.s7.http_429}} and {{r4b.naive.s8.http_429}} rejections against {{r4b.pipe.s7.http_429}} and {{r4b.pipe.s8.http_429}}. One {{naive.qid}} arrival made {{naive.failed_calls}} rejected calls over {{naive.ms}} ms and got passages; under our policy the same arrival got a written {{naive.pipe_kind}} in {{naive.pipe_ms}} ms.

## What the person sees

With the daily budget gone, a person who asked "{{daily.question}}" gets this after {{daily.ms}} ms (stand-in, {{daily.request_id}}):

{{quote:daily.reply}}

Honest and immediate, not always useful: retrieved without the router, these passages miss the figures asked for (graded {{daily.grade}}), and the estimate assumes a refilling daily bucket, which we never measured directly (next section).

## What I'm unsure about, what I left out, what I learned

**Least certain:** how Groq's daily token limit works. No header shows it and no 429 named it. We assumed it refills like the daily request counter, which we saw refill within hours (Table 1), and ran the real load runs on that ({{ledger.refill_calls}} calls, {{ledger.refill_429}} rejection, not a daily one). One thing we know: our key was charged {{ledger.max24h_120b}} gpt-oss-120b tokens within one {{budget.window_h}} h window, over the documented {{lim.tpd_120b}}, and nothing was refused, so it is not a rolling sum of that size. If it instead resets at a fixed hour, the estimate in the reply above is wrong by hours, and a lower hidden cap would refuse without warning.

**Not built, on purpose:** sentence-by-sentence release as each sentence passes the check (we buffer, so first token equals total time); and a model-based grader (it would spend the shared budget). **Next:** a router prompt for Q14's vocabulary gap; fetching the neighbouring piece when a draft's number is missing from the retrieved ones (Q2's model width fails the gate this way).

**One new thing, in my own words:** [AUTHOR: to write.]

**Honesty notes.** All runs are as measured at {{baseline}}; since then {{status.relabelled}} rows in the tables moved from rate-limited to succeeded (passages after failed checks, now their own column). The gate was fixed afterwards; and replaying all {{replay.drafts}} saved real drafts it passes {{replay.new_pass}} instead of {{replay.old_pass}}, drops {{replay.dropped}} sentences, and changes {{replay.req_changed}} of {{replay.req_in_tables}} requests in the tables (data/experiments/gate_replay/result.md). The grader changed after I saw answers (results/quality/grade_changes_v1_v2.md, grade_changes_v2_v3.md) and one Q13 gold fact became optional; {{grades.overrides}} grades were set by hand and {{grades.unsure}} of {{grades.n}} graded real rows were never checked by hand. To run without waiting, two eval jobs capped at the documented daily limit instead of our lower cap ({{ledger.cap100_calls}} calls) and the real load runs used refill accounting. {{excluded.key2}} runs on a second API key are excluded (the brief allows one budget), as are {{excluded.bad_key}} where a mistyped key got only 401s.

## Your feedback on Vaani

[AUTHOR: to write after visiting vaani.dev: what stood out, what confused you, one thing you would change.]

## How I worked with the assistant

[AUTHOR: to write. Full conversation history attached. Three prompts that mattered and why; one thing it got wrong that you caught and how; one decision where you overruled it; what is distinctly yours.]
