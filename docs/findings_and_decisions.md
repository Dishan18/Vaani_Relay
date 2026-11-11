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
