# Findings and decisions

What the real runs taught us, what we changed and why, and what is waiting for the iterate rounds.
Numbers here point at the generated files that hold them (rule 3); nothing is typed from memory.

## Candidates for the iterate rounds (after the retrieval sweep is complete)

1. **The decide gate turns a vocabulary mismatch into a refusal (Q14).** `relay/pipeline.py`
   scores "evidence too weak" against the question's own words. Q14 ("How does H-Net decide where
   to cut the byte stream into pieces?") shares none of its terms with the routing passage, so at
   the medium setting the gate refused an answerable question before any draft, even though the
   router had rewritten the query (`results/quality/traps.md`, Q14). Candidate fix: score the
   evidence against the question plus the router's rewrite, or only refuse on near-zero overlap
   with both. Left unchanged until the sweep is done, so every setting runs the same pipeline
   (author's decision, 2026-11-10).
2. **The router's rewrite did not rescue Q14 either.** Its queries ("dynamic chunking boundary
   prediction ...") never retrieved §2.2.1 (`traps.md`, "gold in sources: no"). Candidate: ask the
   router for the mechanism's likely name ("routing module", "boundary probability"), or a second
   search on the rewrite alone without the question's words appended.
3. **Admission margin in the limiter.** Medium Q5's first draft got a real TPM 429 the limiter
   didn't predict: "Limit 8000, Used 5916, Requested 2099 ... try again in 112.5 ms", i.e. 15
   tokens short (`results/raw/answers/retrieval-part1-20261110.calls.jsonl`). The limit is one the
   limiter models (TPM), not a hidden one. Requested (2099) minus the same prompt's reported
   prompt_tokens (1795, from the retry) is 304, more than the 200-token hold the probe measured; one
   429 can't tell a bigger hold from a higher admission-time prompt count. Across the 45
   non-overlapping 120b draft pairs in part 1, admission matched reported prompts (median residual
   1 token, spread -511..+109), so header-based levels are noisy by a few hundred tokens and the
   limiter admits with no margin. Candidate: admit only with level >= need + a margin from that
   spread. Not changed mid-sweep.
4. **Unsure grades.** `results/quality/unsure.md` lists grades the auto-grader flagged; the author's
   overrides go to `results/grade_overrides.jsonl`. Q9 and Q10 (medium) were confirmed wrong by the
   author (2026-11-10); the reasons are in the overrides file.

## Found in the real runs

- **qwen has an undocumented output-token limit.** The drafter step got "output tokens per minute
  (OTPM): Limit 1000, Used 820, Requested 184" (`results/raw/answers/drafter-large-20261110.calls.jsonl`).
  No header shows it. The limiter created the bucket at runtime but charged it like total tokens
  (prompt + hold), so a 1726-token qwen draft "could never fit" in a 1000 bucket and Q6-Q18 of
  qwen's drafter run fell back to passages without a call. Fixed: limits on input tokens are charged
  the prompt only, limits on output tokens an output hold at admission (learned from the 429's
  Requested; 184 for max_tokens 300) released to the real completion; limits found in real-run 429
  bodies now seed `Limiter.from_probe` and the stand-in's calibration. qwen's drafter step Q6-Q18
  was rerun (run header says why); its first Q1-Q5 stand. The rerun then met qwen's second hidden
  limit, "input tokens per minute (ITPM): Limit 7000, Used 5101, Requested 1918"
  (`drafter-large-qwen-rerun-otpm-20261111.calls.jsonl`); the fixed limiter made it an input-only
  bucket and the request stepped down to lean instead of being locked out. Both now seed the limiter
  and the stand-in. 120b's only 429s were TPM, a limit the limiter models (see candidate 3: they keep
  recurring near a full bucket).
- **Stand-in replay gaps.** The first stand-in load run (`results/raw/runs/*-20261110T181851Z.jsonl`)
  never replayed refusals (stored under a different key than the lookup), and its placeholder could
  copy a paragraph label ("Metrics."), which the citation gate rightly rejects; it also lacked qwen's
  OTPM. All seven scenarios were rerun on the fixed stand-in; the newer run supersedes the older in
  results.csv and the old files stay.
- **small_lean is incomplete** (11/18): the stop rule halted it when 120b's 24 h ledger window was
  full. The rest waits for 120b's spends to leave the window (`results/quality/budget_plan.md`).

## One budget: the key2 runs are excluded (2026-11-11)

The author briefly switched .env to a second API key ("key2", which showed 999/1000 daily requests on
each model; that does not by itself show a separate organization, see the next section) and the
remaining jobs ran on it: small_lean Q12-Q18, the Q14
decide-gate rerun, real_steady (31 requests) and real_burst (20). The brief allows one budget, so the
author put the original key back and every reported real result uses key1 only. The key2 runs stay
in the raw files and in results.csv (superseded, with the reason in the `excluded` column; list in
`results/excluded_runs.json`) but count in no table. The ledger now keeps one account per key
(entries before labels = key1; a fingerprint -> label map stays in a gitignored local file, no key
material is committed), so key2's calls never counted against key1's caps. The key1 jobs wait for
key1's 24 h window (`results/quality/budget_plan.md`, `--account key1`).

## Mistyped key, 401s, and Groq's daily request counter (2026-11-11)

* **The two key1 eval jobs ran early with the account cap at 100%** of the documented limits
  (author's decision: run now instead of waiting for 14:06 UTC). The override is the environment
  variable `RELAY_ACCOUNT_CAP_FRACTION`, set for those two runs only; every ledger line written under
  it carries `account_cap_fraction`. The load runs stay on the 90% cap.
* **First attempt: every call got HTTP 401 Invalid API Key.** The key in .env had been mistyped. The
  two runs (`retrieval-part2-lean-rest-key1-20261111`, `q14-decide-gate-large-key1-20261111`) are
  excluded (`results/excluded_runs.json`): no model call ran, so every reply is the passages fallback.
  The ledger had charged those 16 calls 17,398 tokens of 120b (the 4xx rule: prompt estimate). A 401
  identifies no account, so nothing was spent: `charge_of`/`requests_of` now count a 401 as 0 (old
  entries re-read, file untouched), and the first 401 halts the ledger so a bad key costs one call,
  not 16. Reruns: `...-key1b-20261111`, on the corrected key (the author confirms it is the key1
  used for the last 24 h).
* **Groq's daily request counter refills; it is not a rolling 24 h window** (measured, from the
  `x-ratelimit-remaining-requests` headers saved in the ledger for gpt-oss-120b on key1): 942 at
  11-10 14:03 UTC, back to 999 by 17:49 UTC (57 requests returned in under 4 h; RPD 1000/24 h refills
  ~42 per hour); 915 at 18:26 UTC, 999 at 11-11 07:23 UTC. So "fresh" (999/1000) only means the
  bucket is full, not that a key is unused or in another organization. Groq exposes no daily-token
  header, so the daily-token side is still unmeasured. The ledger keeps the rolling 24 h sum, at the
  author's request: it is at least as strict as any refill rule.

## Real load runs early, on refill accounting (2026-11-11)

The author asked for real_steady and real_burst now instead of at 17:55 UTC. Under the rolling 24 h
sum key1 had no 120b room even at a 100% cap (18.8K tokens; the two runs need ~52.7K), so the author
chose **refill accounting for these runs**: `RELAY_LEDGER_ACCOUNTING=refill` makes each model's cap
(still 90% of the documented daily limit) drain continuously at cap / 24 h, the behaviour measured for
Groq's daily request counter (previous section). That left 113K of 120b room. Every ledger line of the
runs carries `ledger_accounting: refill`. Caveat: Groq shows no daily-token header, so whether its
daily-token limit refills the same way is unmeasured; a daily-limit 429 would be handled by the ladder
(never retried) and would show in the run. Rolling stays the default for everything else.

## What the real load runs showed (2026-11-11, key1, refill accounting)

Runs `real_steady-20261111T073012Z` and `real_burst-20261111T073012Z`; numbers in `results/tables.md`.
* **real_steady's passages replies were the citation gate, not budget.** Each of the 4 had zero budget
  wait and three model calls: router, draft, and a redraft that also failed the gate. Replaying the
  saved drafts through `check_citations`:
  * topic or summary sentences without a citation ("Moshi's architecture is built around three core
    modules.");
  * display formulas split by the sentence splitter into uncited lines (Q3's PE equations);
  * d_model = 512 (Q2), which is in the paper but not in the 3 retrieved pieces, so the number check
    is right to block it.

  real_burst's two 3-call passages replies (both Q2) are the same pattern. The rest of real_burst's
  passages were the ladder: no call, an instant reply when 120b's minute budget couldn't serve them
  within 15 s.
* **The stand-in missed this.** It replays the eval's answers, which passed the gate, so it never
  produces a fresh draft that fails. In its real_steady twin every person got a written answer; in the
  real run 4 of 31 got passages after failed checks (Table 2b, "of which passages after a failed check").
  Fresh-draft gate failures are a gap in the stand-in's realism.
* **No daily-limit 429 under refill accounting.** The runs got one TPM 429, on gpt-oss-20b.
* What was done about it: see "Gate fix and how it is reported" below. (I first proposed letting the
  gate accept an uncited framing sentence; the author ruled that out: an uncited sentence is never
  released.)

## Budget rules bent for the last runs, and why (2026-11-11)

Both overrides were the author's decisions, for these runs only. Every ledger line they wrote records
the override, and the defaults (90% cap, rolling 24 h sum) stay for everything else.
* **Account cap raised to 100% for the two key1 eval jobs** (small_lean Q12-Q18, Q14 on large;
  `RELAY_ACCOUNT_CAP_FRACTION=1.0`, lines carry `account_cap_fraction`).
  * Why: the author asked to run them now rather than at 14:06 UTC. At 90% key1 had 6,630 tokens of
    120b room against about 9.6K needed plus the stop rule's 7,400-token margin.
  * The 100% cap is Groq's documented limit itself, so nothing beyond the published allowance was
    used.
* **Refill accounting for real_steady and real_burst** (`RELAY_LEDGER_ACCOUNTING=refill`, 90% cap
  kept, lines carry `ledger_accounting`).
  * Why: the author asked for the load runs now rather than at 17:55 UTC. Under the rolling sum key1
    had about 18.8K of 120b room even at 100%, against about 52.7K needed.
  * Refill (cap / 24 h) is the behaviour measured on Groq's own daily request counter (section
    above). It gave 113K of room.
  * Risk accepted: the daily-token side is unmeasured. No daily-limit 429 came back.

## Gate fix and how it is reported (2026-11-11)

**The baseline is e7aa65f.** Table 3 and the real load runs stay as measured there, on the old gate.
The write-up reports them as that commit plus the replay delta below; nothing is re-run to replace
them.

**The fix** (`[gate]` in config/pipeline.toml, now on). Every released sentence stays cited.
* `drop_plain_sentences`: an uncited sentence with no number (digits or words), quotation or
  technical term is dropped from the reply. Anything else uncited fails the draft as before.
  * Technical terms: a name or acronym, a hyphenated or underscored compound, LaTeX, or a word that is
    rare in the corpus and occurs in the given sources.
* `join_formulas`: a display-formula line, and a line leading into one ("... is computed as"), is
  checked together with the cited sentence it belongs to. This applies only when the joined lines
  contain math, so an uncited list item never borrows a later citation.
* **Bug found by the replay, fixed:** gpt-oss sometimes writes a zero-width space inside the brackets
  ("[<zero-width space>S1]"), and the gate did not read it as a citation. Invisible characters are now
  stripped when citations are normalised.
* The passages notice said "no model call" even after two drafts. It now says "no written answer"
  and gives the reason.
* **Status:** a passages reply after failed checks is now `ok`, with degraded_level 3 and the reason
  in `degraded_reason`. `rate_limited` means budget caused the fallback. 48 status cells changed;
  answers and grades did not. Table 2b adds the column "of which passages after a failed check".

**Replay** (`scripts/replay_gate.py` -> `data/experiments/gate_replay/result.md`): every saved real
draft, old gate vs new, with every changed verdict, dropped sentence and joined formula listed.
* The drop rule removed nothing. Every uncited sentence in the real drafts names something or states
  a number, so those drafts still fail, as the rule intends.
* The changes come from formula joining and the zero-width fix.
* At request level the changes are few: in real_steady one person (Q3) would have got a correct
  written answer instead of passages.
* Still failing, correctly: Q2's "d_model = 512", which is not in the retrieved pieces.
* Q14's "{0,1}" used to be read as the number "0,1", a gate false positive. Fixed before the Q14
  medium run, so that run tests the decide gate, not this bug: a comma left between digits after
  thousands separators are removed now separates two numbers. The replay's "old" column keeps the
  e7aa65f parse.

**Waiting for 120b budget** (rolling 24 h on key1; `results/quality/budget_plan.md`):
* Q14 on medium: the gate refused it there.
* One real_steady rerun on the fixed gate, reported next to the e7aa65f run, not replacing it.

## Q14 on medium, and the real_steady rerun (2026-11-11)

* **Q14 on medium**, on the fixed gate (aefeb5a), run early at the author's request, key1, 90% cap,
  refill accounting (`q14-medium-fixed-gate-key1-20261111`).
  * The decide gate now lets it through. In the sweep it was refused_wrongly; now it is answered, and
    graded wrong, sure: missing all 4 gold facts (the cosine-similarity routing rule).
  * Its cited pieces do not hold that passage. The remaining problem is retrieval: Q14's words never
    match the routing-module passage. That is the router-prompt next step.
* **The rerun did not fit right away.** The budget plan priced the real_steady rerun from the stand-in
  at about half what the real run was charged on gpt-oss-120b (`results/quality/budget_plan.md`).
  * The cause: real drafts fail the check and get redrafted, which the stand-in never shows.
  * Started now on 50K of refill room, the rerun would have run out of 120b partway, been drafted on
    other models, and not be comparable to the e7aa65f run.
  * So it starts automatically once refill room covers the real cost plus 10%.
  * `scripts/budget_plan.py` now prices a load run from its latest real run when one exists.

## No more real calls: the real_steady rerun is cancelled (2026-11-11, author's decision)

The rerun of real_steady on the fixed gate is cancelled and no further real calls will be made. Why:
* **The gate fix is measured by the replay** (data/experiments/gate_replay/result.md): every saved real
  draft from the evals and load runs, through the old and new gate. One rerun against one run would be
  an anecdote: arrivals meet different minute budgets and the models answer differently each time.
* **Groq's daily token bucket refills (evidence).** In the ledger, key1 shows more gpt-oss-120b tokens
  in one rolling 24 h window (`ledger.max24h_120b` in docs/writeup/facts.json) than the documented TPD
  of 200,000, with no daily 429 (Table 1: none of the saved real 429s named a per-day limit).
  * A rolling 24 h sum at the documented size would have refused those calls.
  * Taken with the request counter, which we watched refill within one UTC day, this is evidence the
    daily bucket refills.
  * A reset at 00:00 UTC would also fit the token numbers alone; the request counter's same-day
    refill is what points to refilling.
  * Another run would add to that total without adding to the evidence.

**Stopping background jobs left orphans.** Every earlier stop of a waiter script ended the tool's task
but left its bash process alive, sleeping until its start time. Five were found when the rerun was
cancelled:
* the 14:06 small_lean/Q14 waiter;
* the 17:56 load-run waiter;
* two copies of the 18:01/18:21 waiter;
* the refill waiter.

All five were killed before any of them reached a real call (the ledger's last real call is the Q14
medium run). From here, a stopped waiter is checked in the process list, not only in the tool.

## Decisions taken (2026-11-10)

- **Ledger charges what Groq reports.** A completed call counts its reported usage (findings Q2:
  Groq releases what admission held down to real usage); a call with no usage keeps the
  conservative max(estimate + max_tokens, usage); in-flight reservations still count. The cap stays
  a rolling 24 h sum, not a refilling bucket: Groq never shows the daily-token bucket, so its refill
  is unmeasured, and the rolling sum is the conservative choice on a shared key. Old ledger lines
  are not edited; they are re-read with the new rule (`relay.ledger.charge_of`). Why: the old rule
  charged ~2.0x real usage on 1200-max_tokens drafts and stopped part 1 at small Q4.
- **Citation gate reads 【S#】.** gpt-oss writes 【S3】 instead of [S3], mostly on redrafts; the old
  gate rejected those drafts, causing all four passages-only replies in part 1. Citations are now
  normalised to [S#] before the gate and before release; the grader does the same. The drafting
  prompt was not changed mid-sweep. Only the five requests whose drafts used 【S#】 were rerun (large
  Q8, Q15; medium Q7, Q8, Q11); drafts that passed the old gate pass the new one unchanged, so the
  rest of part 1 stands. The old rows stay in `results/raw/answers/`; the reruns' run headers say why.
- **Grader v2** (after part 1): normalises LaTeX, superscripts and non-breaking hyphens before
  matching and widens four patterns; every changed grade is in `results/quality/grade_changes_v1_v2.md`.
  The author checked all 9 changes (2026-11-10): each answer has the gold facts, and v1 missed
  them on formatting.
- **Passages-only and busy replies are graded like any answer**: it is what the person got.
- **answer_grade holds only the brief's four values.** Stand-in rows leave it empty; the reason is
  in `grade_note` (after the brief's columns), including when a stand-in answer replays a real one.
- **No real_steady / real_burst until 120b's budget is back** (`results/quality/budget_plan.md`).

## Iterate rounds (stand-in only; one change per round, 2 seeds, kept only if better)

- **r1_token_margin** (20261110T190658Z, stand-in, seeds 7,8): margin_560 not better. Kept: baseline. `data/experiments/r1_token_margin/result.md`
- **r2_max_wait** (20261110T191008Z, stand-in, seeds 7,8): wait_3s not better; wait_6s not better; wait_15s better (fewer people without a written answer). Kept: wait_15s. `data/experiments/r2_max_wait/result.md`
- **r3_first_token_timeout** (20261110T191322Z, stand-in, seeds 7,8): first_token_3s not better; first_token_6s not better. Kept: baseline_15s. `data/experiments/r3_first_token_timeout/result.md`
- **r4_retry_policy** (20261110T191647Z, stand-in, seeds 7,8): retry_on not better. Kept: retry_off. `data/experiments/r4_retry_policy/result.md`
- **r4b_policy_vs_naive** (20261110T192004Z, stand-in, seeds 7,8): policy not better. Kept: current. `data/experiments/r4b_policy_vs_naive/result.md`
