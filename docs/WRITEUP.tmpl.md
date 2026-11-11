# The Relay

<!-- Template. Every number comes from docs/writeup/facts.json ({{key}}) or results/tables.md ({{table:...}}).
     Render: scripts/render_writeup.py. Check: scripts/check_numbers.py (fails on any digit typed here). -->

## What I built and why

Python/asyncio on Groq's free tier: qwen3.8-27b routes, BM25 fetches {{cfg.k_full}} pieces of {{cfg.piece_chars}} characters, a decide step refuses weak evidence, gpt-oss-120b drafts with citations, and a gate checks every sentence before release. **The decision that shaped it most: an answer ladder with a no-model floor.** If the limiter predicts no room within {{cfg.max_wait_s}} s, a request steps down (another model, a lean answer, then cited passages with no model call and an estimate of when to retry) instead of queueing. The floor needs no budget: with the daily budget gone it replied in {{daily_exhausted.standin.ttft_p50}} ms at p50 (stand-in).

## The real service and my stand-in

{{table:Table 1}}

From a probe of {{probe.calls}} real calls and real 429 bodies. The stand-in speaks Groq's API, headers and 429 wording with the same buckets, first-byte and streaming rates fitted per model to the probe (gpt-oss-120b: {{cal.ttfb_n}} calls, {{cal.decode_tps}} tokens/s), load slowdown, stalls and failures, and replays real replies. **Least like Groq:** it replays only drafts that passed the check, so it never shows a gate failure: on identical arrivals the real steady run gave {{real_steady.real.written_pct}}% a written answer, its twin {{real_steady.standin.written_pct}}%.

## Results

{{table:Table 2a}}

{{table:Table 2b}}

Real rows are real runs; the rest are the stand-in; percentiles nearest-rank. **The number we could not have guessed:** qwen3.8-27b also caps output at {{lim.qwen_otpm}} tokens a minute ({{lim.otpm_share}}% of its documented TPM) and input at {{lim.qwen_itpm}}, in no document or header, only in 429 bodies. Quality on {{q.n}} questions I wrote ({{q.unanswerable}} unanswerable): {{q3.large.correct_pct}}% correct and cited, {{q3.large.refused_pct}}% correctly refused, {{q3.large.tokens}} tokens per answer.

## Where the time goes

{{slow.request_id}} ({{slow.qid}}) took {{slow.total_ms}} ms and the person left at {{slow.patience_s}} s, before any token. explain_request.py on its spans: waiting {{slow.waiting_ms}} ms, working {{slow.working_ms}}, retrying {{slow.retrying_ms}}. It waited {{slow.first_wait_ms}} ms for gpt-oss-120b, drafted in {{slow.draft_ms}} ms, failed the citation check, then waited {{slow.last_wait_ms}} ms for a second draft on {{slow.last_wait_on}}: in effect a retry, and the person got nothing.

## When a limit is hit

**Per minute** (stand-in overload): {{overload.standin.written_pct}}% written ({{overload.standin.weaker_pct}}% from a fallback model or the lean step, the weaker answers), {{overload.standin.passages_budget_pct}}% passages, {{overload.standin.gave_up_pct}}% gave up; others are delayed up to the ladder's limit (queue wait p95 {{overload.standin.queue_wait_p95}} ms); nobody is turned away. **Per day:** {{daily_exhausted.standin.written_pct}}% written, {{daily_exhausted.standin.passages_budget_pct}}% passages: straight to the floor rather than holding people until a reset.

## Retrieval: quality versus cost

{{table:Table 3}}

**{{q3.chosen}}**: most correct at {{q3.large.tokens}} tokens per answer; re-ranking saved tokens without adding correct answers; small_lean is cheapest but {{q3.small_lean.correct_pct}}% correct. Near the limit the lean step fetches {{cfg.k_lean}} pieces and skips the router.

## How this library will mislead a naive retriever

Plain BM25, raw question (data/retrieval/traps.md):
- **Different words:** "{{trap.Q14.question}}" shares {{trap.Q14.shared}} of {{trap.Q14.terms}} terms with its answer; {{trap.Q14.correct}} of {{trap.Q14.asked}} settings got it right.
- **Text about a table outranks it:** for "{{trap.Q5.question}}" the table is rank {{trap.Q5.rank}}.
- **Unanswerable but plausible:** "{{trap.Q17.question}}" tops with {{trap.Q17.top_paper}} on "{{trap.Q17.shared_terms}}"; the decide gate refused it in {{trap.Q17.refused}} of {{trap.Q17.asked}} settings.

## Retries

**Ships:** a failed call (429, 5xx, timeout) moves to the next model in the fallback order, each at most once per step, then down the ladder; no resend to the same model. **Rejected:** {{cfg.retry_max}} retry per step within {{cfg.retry_budget_pct}}% of recent calls left {{r4.retry_on.s7.no_written}} and {{r4.retry_on.s8.no_written}} people without a written answer against {{r4.retry_off.s7.no_written}} and {{r4.retry_off.s8.no_written}} (stand-in, two seeds). **Worse:** a naive client ({{naive.retries}} same-model retries, no limiter; stand-in) left {{r4b.naive.s7.no_written}} and {{r4b.naive.s8.no_written}} of {{r4b.naive.s7.n}} without one against {{r4b.pipe.s7.no_written}} and {{r4b.pipe.s8.no_written}}, drawing {{r4b.naive.s7.http_429}} and {{r4b.naive.s8.http_429}} rejections.

## What the person sees

Daily budget gone; "{{daily.question}}" gets, after {{daily.ms}} ms (stand-in):

{{quote:daily.reply}}

Honest and immediate but, without the router, the passages miss the figures asked for (graded {{daily.grade}}).

## What I'm unsure about, what I left out, what I learned

**Least certain:** Groq's daily token limit, never shown in a header or 429. Evidence says it refills: our key was charged {{ledger.max24h_120b}} gpt-oss-120b tokens in one rolling {{budget.window_h}} h window, over the documented {{lim.tpd_120b}}, with no daily 429, and the request counter refills (Table 1). If it resets at a fixed hour instead, the estimate above is wrong by hours. **Not built:** sentence-by-sentence release (we buffer, so first token equals total); a model-based grader (it spends shared budget). **Next:** a router prompt for Q14; fetching the neighbouring piece when a draft's number is missing (Q2).

**One new thing, in my own words:** One thing I learned is that a response starting quickly does not mean the system is fast. Time to first token is how long the person waits before seeing anything; total time is how long until the whole answer has arrived. At the model the two differ: the first piece arrives quickly and the rest streams after it. In my pipeline they are the same, because every reply is checked before it is released, so the person sees nothing until the checked answer is ready. That moved the question from "how fast does the model start" to "where does the waiting happen". Splitting one slow request into waiting, working and retrying showed it spent almost all its time waiting for budget, not generating. An average would have hidden that; percentiles and a per-request split did not.

**Honesty notes.** Tables from runs measured at {{baseline}}; {{status.relabelled}} rows later moved from rate-limited to succeeded (passages after failed checks). The fixed gate passes {{replay.new_pass}} of {{replay.drafts}} saved real drafts instead of {{replay.old_pass}}, changing {{replay.req_changed}} requests in the tables; the replay is its measurement (one real rerun would be anecdotal). The grader changed after I saw answers (results/quality/grade_changes_*.md) and a Q13 gold fact became optional; {{grades.unsure}} of {{grades.n}} graded real rows were never checked by hand. To avoid waiting, two eval jobs capped at the documented limit, and the real load runs and a last Q14 check used refill accounting. Runs on a second key ({{excluded.key2}}, one budget) and a mistyped key ({{excluded.bad_key}}, 401s) are excluded.

## Your feedback on Vaani

What I liked about Vaani was that it brings different parts of building a voice agent together in one platform. Instead of focusing only on speech recognition or voice generation, it also covers agent setup, tool integrations and call analysis. I think this makes it useful for developers looking to build practical voice-based applications.

One thing I found a little confusing was the latency information across the website. Different pages mention different response times, but it was not very clear what exactly was being measured or how the numbers compare.

One change I would suggest is adding a simple end-to-end example showing how someone can create an agent, connect it to tools, make a call and analyse the result. This would make the platform easier to understand, especially for someone exploring it for the first time.

## How I worked with the assistant

My prompts from the main working session are attached verbatim (prompts.md).

**Three prompts that mattered.** First, the ground rules I set before any code: measure before designing, never make a person wait for budget that isn't coming, and every reported number must come from a file a script produced. They became the ledger with its caps, the answer ladder with a no-model floor, and the number checker behind this page. Second, "before any load testing, measure answer quality on real groq": quality became the first measurement, so the retrieval setting and the drafter were chosen on graded answers, not on speed. Third, the gate fix: "dont accept uncited sentences", and replay every saved real draft through the old and new gate before any real call. That turned a change I could only have shown with one anecdotal rerun into a measurement over every draft we had.

**What it got wrong, and how I caught it.** While grading, the assistant said a retrieved passage did not contain "discarding the prediction of the model for them" and treated an answer as unsupported. I rebuilt the pieces, found the phrase in the source, and asked it to recheck how it had decided; the answer stayed correct. I caught it by reading the source instead of the assistant's summary of it.

**Where I overruled it.** It proposed letting the citation gate accept an uncited opening or closing sentence. I refused: no sentence reaches a person without a citation; filler can be dropped, anything else fails the draft. Earlier it suggested counting the daily budget as a refilling bucket; I kept the conservative rolling sum until there was evidence, and when I later used refill accounting to avoid waiting, the honesty notes say so.

**What is distinctly mine.** The ground rules; the questions and their gold facts; the grading decisions and overrides; going back to one API key after re-reading the brief; and the call to stop spending budget once the replay had measured the fix.
