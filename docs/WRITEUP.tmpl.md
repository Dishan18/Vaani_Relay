# The Relay

<!-- Template. Every number comes from docs/writeup/facts.json ({{key}}) or results/tables.md ({{table:...}}).
     Render: scripts/render_writeup.py. Check: scripts/check_numbers.py (fails on any digit typed here). -->

## What I built and why

I built a Python pipeline on Groq's free tier. A router (qwen3.8-27b) picks the paper and rewrites the question, a search tool (BM25) fetches {{cfg.k_full}} pieces of {{cfg.piece_chars}} characters, a decide step refuses weak evidence, gpt-oss-120b drafts with a citation on every sentence, and a check verifies each citation, number and quote before release, rewriting a failed draft once.

**The decision that shaped it most: an answer ladder with a floor that needs no model.** If the limiter predicts no room within {{cfg.max_wait_s}} s, the request steps down instead of queueing: another model, then a lean answer, then cited passages with no model call, flagged as unchecked. The floor costs no budget, so nobody is left without a reply: with the daily budget gone it answered in {{daily_exhausted.standin.ttft_p50}} ms at the median (stand-in).

## The real service and my stand-in

{{table:Table 1}}

Found by a probe of {{probe.calls}} real calls and from real 429 messages. The stand-in speaks Groq's API, headers and 429 wording with the same limits, uses timings fitted to the probe (gpt-oss-120b: {{cal.ttfb_n}} calls, {{cal.decode_tps}} tokens a second), streams replies in chunks, and can slow down under load, stall, fail and refuse. **Where it is least like Groq:** it replays only drafts that passed the check, so it never shows a failed one. On the same arrivals, the real steady run gave {{real_steady.real.written_pct}}% of people a written answer and its twin {{real_steady.standin.written_pct}}%.

## Results

{{table:Table 2a}}

{{table:Table 2b}}

Percentiles are nearest-rank. Each reply's first token arrives with its last, since nothing is shown before the check; the two time columns differ only because people who gave up count in total time but have no first token (see n). **The number I could not have guessed:** qwen3.8-27b also caps output at {{lim.qwen_otpm}} tokens a minute ({{lim.otpm_share}}% of its documented limit) and input at {{lim.qwen_itpm}}. Neither is documented anywhere. On {{q.n}} questions I wrote ({{q.unanswerable}} unanswerable), the chosen setting gave {{q3.large.correct_pct}}% correct, cited answers and {{q3.large.refused_pct}}% correct refusals, at {{q3.large.tokens}} tokens per answer.

## Where the time goes

{{slow.request_id}} ({{slow.qid}}) took {{slow.total_ms}} ms and the person left at {{slow.patience_s}} s, before any token. explain_request.py splits its spans into waiting {{slow.waiting_ms}} ms, working {{slow.working_ms}} ms, retrying {{slow.retrying_ms}} ms and local work {{slow.local_ms}} ms. It waited {{slow.first_wait_ms}} ms for gpt-oss-120b, drafted in {{slow.draft_ms}} ms, failed the citation check, then waited {{slow.last_wait_ms}} ms for a second draft on {{slow.last_wait_on}}. A retry in all but name, and the person got nothing.

## When a limit is hit

**Per minute** (stand-in overload): {{overload.standin.written_pct}}% got a written answer ({{overload.standin.weaker_pct}}% a weaker one, from a fallback model or the lean step), {{overload.standin.passages_budget_pct}}% got passages and {{overload.standin.gave_up_pct}}% gave up waiting; the system rejected nobody outright. Anyone meeting an empty bucket waits at most until the ladder's limit (p95 queue wait {{overload.standin.queue_wait_p95}} ms). **Per day:** {{daily_exhausted.standin.written_pct}}% written, {{daily_exhausted.standin.passages_budget_pct}}% passages, because the ladder goes straight to the floor instead of holding people until a reset.

## Retrieval: quality versus cost

{{table:Table 3}}

I chose **{{q3.chosen}}**: the most correct answers, at {{q3.large.tokens}} tokens each. Re-ranking saved tokens but added no correct answers, and small_lean was cheapest but only {{q3.small_lean.correct_pct}}% correct. Near the limit, the lean step fetches {{cfg.k_lean}} pieces and skips the router.

## How this library will mislead a naive retriever

Plain BM25 on the raw question (data/retrieval/traps.md):
- **Different words.** "{{trap.Q14.question}}" shares {{trap.Q14.shared}} of {{trap.Q14.terms}} terms with its answer, so it is never found; {{trap.Q14.correct}} of {{trap.Q14.asked}} settings got it right.
- **Text about a table beats the table.** For "{{trap.Q5.question}}" the table itself is rank {{trap.Q5.rank}}.
- **Unanswerable but plausible.** "{{trap.Q17.question}}" pulls up the other paper ({{trap.Q17.top_paper}}) on the word "{{trap.Q17.shared_terms}}". The decide step refused it in {{trap.Q17.refused}} of {{trap.Q17.asked}} settings.

## Retries

**What ships:** a failed call (429, server error or timeout) moves to the next model in the fallback order, each tried once, then down the ladder. Nothing is resent to the same model. **Tested and rejected:** {{cfg.retry_max}} retry per step within {{cfg.retry_budget_pct}}% of recent calls left {{r4.retry_on.s7.no_written}} and {{r4.retry_on.s8.no_written}} people without a written answer, against {{r4.retry_off.s7.no_written}} and {{r4.retry_off.s8.no_written}} (stand-in, two seeds). **Worse:** a naive client ({{naive.retries}} retries on the same model, no limiter) left {{r4b.naive.s7.no_written}} and {{r4b.naive.s8.no_written}} of {{r4b.naive.s7.n}} without one, against {{r4b.pipe.s7.no_written}} and {{r4b.pipe.s8.no_written}}, and drew {{r4b.naive.s7.http_429}} and {{r4b.naive.s8.http_429}} rejections (stand-in).

## What the person sees

At 3pm, with the daily budget gone, "{{daily.question}}" gets this after {{daily.ms}} ms (stand-in):

{{quote:daily.reply}}

It is immediate, gives no time estimate it cannot back, and flags the passages as unchecked; here they miss the figures asked for (graded {{daily.grade}}), which only a model could tell.

## What I'm unsure about, what I left out, what I learned

**Least certain:** Groq's daily token limit. Documented: {{lim.tpd_120b}} a day. Observed: my key was charged {{ledger.max24h_120b}} gpt-oss-120b tokens in one rolling {{budget.window_h}}-hour window with no daily-limit rejection, so it seems to refill or reset rather than sum strictly, but I never saw the limit itself (confidence low). If it is lower than the limiter assumes, a call gets a 429 and falls to the next rung. **Not built:** sentence-by-sentence release (I buffer, so first token equals total) and a model-based grader (it would spend shared budget). **Next:** a router prompt for Q14, and fetching the neighbouring piece when a draft's number is missing (Q2).

**One new thing, in my own words:** One thing I learned is that a response starting quickly does not mean the system is fast. Time to first token is how long the person waits before seeing anything; total time is how long until the whole answer has arrived. At the model the two differ: the first piece arrives quickly and the rest streams after it. In my pipeline they are the same, because every reply is checked before it is released, so the person sees nothing until the checked answer is ready. That moved the question from "how fast does the model start" to "where does the waiting happen". Splitting one slow request into waiting, working and retrying showed it spent almost all its time waiting for budget, not generating. An average would have hidden that; percentiles and a per-request split did not.

**Honesty notes.** All tables are generated from the submitted results.csv. The real runs date from commit {{baseline}}; later I reclassified {{status.relabelled}} rows as succeeded (passages after a failed check, not budget) and reran the daily-budget scenario on the stand-in after changing that reply. The gate fix is a separate replay of all {{replay.drafts}} saved drafts ({{replay.new_pass}} pass instead of {{replay.old_pass}}; {{replay.req_changed}} requests would change), not applied to the tables. I changed the grader after seeing answers (results/quality/grade_changes_*.md) and made a Q13 gold fact optional; {{grades.unsure}} of {{grades.n}} graded real rows were never checked by hand. To avoid waiting, two eval jobs capped at the documented limit and the last real runs used refill accounting. Excluded runs and their reasons are in results/excluded_runs.json.

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
