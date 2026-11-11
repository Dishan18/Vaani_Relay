# Write-up audit (2026-11-11)

Every claim in docs/WRITEUP.md, the evidence behind it, and what changed when the evidence did not
back the draft. Numbers are not repeated here: each is a fact in docs/writeup/facts.json (key in
`backticks`) or a table cell from results/tables.md, and scripts/check_numbers.py fails on any number
typed into the template.

## Corrections made (the draft said X, the evidence said Y)

| # | section | draft said | evidence | now says |
|---|---|---|---|---|
| 1 | built | the floor "always answers" | relay/pipeline.py: a "busy" reply exists for internal failures (it never fired: Table 2b error 0.0) | budget alone never leaves a person without a reply |
| 2 | built | daily-gone reply time, unlabelled | `daily_exhausted.standin.ttft_p50` is a stand-in run | labelled "(stand-in)" |
| 3 | Table 1 | RPD refill: largest rise in remaining requests | that pair crossed midnight UTC, so a daily reset could explain it | largest rise within one UTC day (make_tables.py `rpd_refill`) |
| 4 | Table 1 | "no daily-token 429 was seen" (typed) | not computed | counted from every saved real call, probe log included (`daily_429s`) |
| 5 | stand-in / results | the unguessable number sat under heading 2 | the brief asks for it under Results | moved to Results |
| 6 | time | waited for gpt-oss-120b's "minute budget" | the span records only "granted", not which limit | "budget on gpt-oss-120b" |
| 7 | time | I was about to give the {0,1} parse bug as the reason the draft failed | that failure was a different request (…-0004); this one gave up, its row has no saved retrieval, the span records no reason | "the citation check returned fail (no reason recorded)" |
| 8 | time | "a retry that made things worse" | counterfactual, not measured | "in effect it was a retry, and the person got nothing" |
| 9 | naive retriever | Q14 "0 of 5 correct" | the 5 are settings, and small_lean has no router | "0 of 5 settings correct" |
| 10 | naive retriever | heading "a table split from its caption" | my label; traps.json measures a passage about Table 2 outranking the piece that holds it (checked: the top piece lacks the figure, the right piece has it) | "Text about a table outranks the table" |
| 11 | naive retriever | Q17's top hit is "a Moshi latency passage with numbers to borrow" | it is Moshi's introduction, no latency figures, sharing only "latency"; the first piece with a borrowable number is further down (`trap.Q17.distractor_rank`) | described as the file shows |
| 12 | retries | "nothing is resent after a 429, a 5xx or a timeout" | config/pipeline.toml [retry]: off = "try every model in the fallback order"; relay/pipeline.py `_try_models` | after a failed call it tries the next model, each at most once per step |
| 13 | retries | naive client "back-off from 1 s" | scripts/run_load.py `naive_retry`: a fixed pause, same model, and NaiveLimiter has no limiter | "no limiter; N retries on the same model, 1 s apart"; labelled stand-in |
| 14 | retries | my earlier example (Q3, retried ten times, pipeline answered in half a second) | never saved to a file | replaced by a rule-picked example from data/experiments/r4b_policy_vs_naive/rows (`naive.*`) |
| 15 | person sees | "the estimate assumes Groq's daily tokens refill, which is unmeasured" | partly measured, see 16 | "never measured directly" |
| 16 | unsure | daily token limit "unmeasured"; "if it is a rolling sum, a busy day runs out hours early" | ledger: our key was charged more gpt-oss-120b tokens in one 24 h window (`ledger.max24h_120b`) than the documented TPD (`lim.tpd_120b`) and nothing was refused | not a rolling sum of the documented size; what remains unknown is refill vs fixed reset vs a different size |
| 17 | unsure | next: "fetching table rows with their captions" | the replay's remaining number failure is Q2's model width, a number in a neighbouring piece | "fetching the neighbouring piece when a draft's number is missing (Q2)" |
| 18 | honesty | "tables as measured at e7aa65f" | Table 2b relabelled rows since (`status.relabelled`, computed against git e7aa65f) | "runs as measured at e7aa65f; since then N rows moved from rate-limited to succeeded" |
| 19 | honesty | key2 and mistyped-key runs excluded "because the brief allows one budget" | the mistyped-key runs are excluded because every call got 401 | the two reasons given separately |
| 20 | tooling | a fallback `else 0` for Q14's shared terms in writeup_facts.py | a typed number | removed; computed from traps.json |

## External review (2026-11-11): points raised and what the data said

| # | point raised | what the data showed | done |
|---|---|---|---|
| 36 | tables may not match the final results.csv | they do: an independent recompute matches every cell; the honesty note made them sound stale | note rewritten: all tables are generated from the submitted results.csv; the runs date from e7aa65f, 11 rows were later reclassified, and the gate replay is a separate experiment not applied to the tables |
| 37 | first token differs from total in some rows, so maybe tokens leak before the check | for all 510 replies first_token_ms = total_ms; rows differ only because people who gave up count in total time but have no first token | stated under Results |
| 38 | the 3pm reply quoted irrelevant numbers and a time estimate | the evidence test scores those passages at the maximum, so no lexical filter can catch them; the estimate assumed the daily refill we never measured | code change: a daily limit gives no time estimate and the passages are flagged as unchecked (relay/pipeline.py, limiter daily_short, tests); daily_exhausted rerun on the stand-in, no real calls; the write-up shows that run's reply |
| 39 | daily token limit stated too strongly | Table 1 already said documented, confidence low | the write-up now gives documented vs observed and says confidence is low |
| 40 | second API key | the author asked that it not be discussed | the write-up points to results/excluded_runs.json for every excluded run and its reason; the raw files are unchanged |
| 41 | stand-in streaming not mentioned | it streams in chunks (standin/server.py) | mentioned |
| 42 | "nobody is turned away" | error 0.0 everywhere, 5.1% gave up | "rejected nobody outright", with the give-ups |
| 43 | slow-request split missing 3 ms | explain_request.py reports local work | added |
| 44 | Table 3 continued on page 2 without headers | | headers repeat across pages; the always-zero "ungraded %" column is shown only when non-zero, and Table 3 now fits on page 1 |
| 45 | "three unanswerable questions" | data/questions.jsonl has four with answerable = false (Q6, Q12, Q17, Q18), matching Table 3's 22.2% correctly refused (4 of 18) | no change |

## Pre-upload review (2026-11-11): checked against every line of the brief

| # | where | problem | evidence | fix |
|---|---|---|---|---|
| 30 | Table 3 | the setting column gave names only; the brief asks for piece size, pieces fetched and re-rank yes or no | the brief's Table 3 header | each setting now reads "name (piece chars / pieces / re-rank)" from scripts/eval_retrieval.py ANSWER_SETTINGS |
| 31 | Table 3 | "wrong %" counted wrongly refused answers too, without saying so; an independent recompute from results.csv disagreed on four rows | make_tables.py summarise() | header now "wrong % (incl. wrongly refused)"; recompute matches |
| 32 | Table 2b | peak queue length came from data/results/detail.jsonl, so it could not be recomputed from results.csv as the brief requires | the brief: "every figure in the tables must be computable from this file" | new results.csv column wait_spans_ms; make_tables.py computes peak queue from it; every value unchanged |
| 33 | built | the brief's "at least one tool call, one branch or loop" was not pointed out | relay/pipeline.py | the search tool, the decide step and the ladder (branches), and the one rewrite of a failed draft (loop) are named |
| 34 | honesty | "tables come from runs measured at e7aa65f" | Table 1 comes from the probe, not those runs | "the load and retrieval tables are as measured at e7aa65f" |
| 35 | all prose | clipped, list-like sentences after the length cuts | | rewritten in plain sentences; no em or en dashes anywhere in the write-up or README |

An independent recompute (nearest-rank percentiles and status shares straight from results.csv, without
make_tables.py) now matches every cell of Tables 2a, 2b and 3.

## Final pass (2026-11-11, before submission): further corrections

| # | section | draft said | evidence | now says |
|---|---|---|---|---|
| 21 | stand-in | "a first-byte lognormal fitted to N probe calls, M tokens/s" as if for the stand-in as a whole | those are gpt-oss-120b's fit (data/calibration.json); each model has its own | "fitted per model to the probe (gpt-oss-120b: ...)" |
| 22 | retries | the naive client's numbers, unlabelled | data/experiments/r4b_policy_vs_naive is stand-in only | labelled stand-in |
| 23 | naive retriever | Q14: "0 of 5 settings answered it" | it was answered, wrongly, in some settings | "0 of 5 settings got it right" |
| 24 | honesty | "runs as measured at e7aa65f" | the Q14-on-medium check ran later (q14-medium-fixed-gate-key1-20261111) | "tables from runs measured at e7aa65f" |
| 25 | honesty | refill accounting only for the real load runs | the Q14 check also used it (its ledger lines carry ledger_accounting) | "the real load runs and a last Q14 check used refill accounting" |
| 26 | unsure | daily token limit "unmeasured" | ledger: `ledger.max24h_120b` in a rolling 24 h window against the documented `lim.tpd_120b`, no daily 429 (Table 1) | "evidence says it refills"; the remaining doubt is a fixed-hour reset |
| 27 | honesty | no reason given for not rerunning real_steady | author's decision (docs/findings_and_decisions.md) | "the replay is its measurement (one real rerun would be anecdotal)" |
| 28 | limits | the real burst split | only repeated Table 2b | deleted |
| 29 | Table 1 | long "how found" cells | same sources, shorter wording; values unchanged | generated by make_tables.py |

Checked and kept this pass: the decide gate refused Q17 in every setting (refusal spans); nobody turned
away (Table 2b error 0.0); the request counter refills (Table 1, same UTC day); the 3pm request skipped
the router (its spans). Page budget: scripts/build_writeup.py prints the page count and, with
--simulate-author-words, how much room the author's sections have.

## Trimmed for length (2026-11-11): claims no longer in the write-up

Removed rather than shortened, so they need no evidence now; each is still backed where it lives:
the stand-in's burst comparison and p95 first token (`real_burst.*`); the probe's phases
(data/probe/findings.md); the verifier being on another model (config); Q5's top-ranked piece and Q17's
distractor rank (data/retrieval/traps.json); the slow request's local time; the stricter retry
policy's daily-429 rule, the pipeline's 429 counts and the Q8 example (data/experiments/r4*); the
reply estimate's assumption; the replay's dropped-sentence count and the override count. Table 1's
concurrency cell keeps only the limit found per model (make_tables.py).

## Every claim and its evidence

| section | claim | evidence | status |
|---|---|---|---|
| built | router qwen3.8-27b picks paper, rewrites | config/pipeline.toml [models] understand; relay/pipeline.py ROUTER_SYSTEM | backed |
| built | BM25, `cfg.k_full` pieces of `cfg.piece_chars` | config/pipeline.toml [retrieval] | backed (input) |
| built | decide step refuses on weak evidence | span "refuse: evidence too weak" on Q17, results/raw/answers | backed |
| built | gpt-oss-120b drafts | config [models] draft | backed (input) |
| built | gate checks citation, numbers, quotes | relay/pipeline.py check_citations | backed (code) |
| built | verifier on another model when budget allows | config [models] verify; pipeline "skipped (no headroom)" | backed (code) |
| built | released only after checks | results.csv first_token_ms = total_ms | backed |
| built | ladder: other model, lean (`cfg.k_lean`, no router), passages with no model call and an ETA | relay/pipeline.py; detail span "understand skipped (lean)" | backed |
| built | budget never leaves anyone without a reply; `daily_exhausted.standin.ttft_p50` | results.csv; Table 2b error 0.0 | corrected (1, 2) |
| stand-in | Table 1 | make_tables.py from data/calibration.json, data/probe/calibration.json, ledger, raw calls | corrected (3, 4) |
| stand-in | probe of `probe.calls` calls | data/probe/calibration.json | backed |
| results | qwen OTPM `lim.qwen_otpm`, ITPM `lim.qwen_itpm`, share `lim.otpm_share` of documented TPM | data/calibration.json; 429 bodies in results/raw/answers/drafter-large-20261110.calls.jsonl and drafter-large-qwen-rerun-otpm-20261111.calls.jsonl ("output tokens per minute (OTPM)", "input tokens per minute (ITPM)"); data/limits_documented.json | backed |
| results | neither in a header | those calls' headers: only x-ratelimit-{limit,remaining,reset}-{requests,tokens} | backed |
| stand-in | API, headers, 429 wording, buckets, lognormal first byte over `cal.ttfb_n` calls, `cal.decode_tps` tokens/s, slows, stalls, fails, replays | standin/server.py (Switches, Replay, decode); data/calibration.json | backed |
| stand-in | replays only drafts that passed the check | standin/server.py Replay (answers and refusals only; redrafts get a placeholder); Table 2b failed-check column 0.0 on every stand-in row | backed |
| stand-in | real vs twin written %, burst p95 first token | `real_steady.*`, `real_burst.*` from results.csv | backed |
| results | real rows are real runs; nearest-rank percentiles | Table 2a/2b; make_tables.py | backed |
| results | `q.n` questions I wrote, `q.unanswerable` unanswerable | data/questions.jsonl (answerable = false) | backed; authorship is the author's statement |
| results | quality at the chosen setting | `q3.large.*`, Table 3 | backed |
| time | slow request and its split | `slow.*`: scripts/explain_request.py on data/results/detail.jsonl | corrected (6, 7, 8) |
| limits | overload: written, weaker, passages, gave up, queue p95 | `overload.standin.*` | backed |
| limits | dropped: nobody | Table 2b error 0.0 | backed |
| limits | real burst split | `real_burst.real.*` | backed |
| limits | per day: straight to the floor | notices in results/raw/runs/daily_exhausted-20261110T192348Z.jsonl ("TPD would not have room") | backed |
| retrieval | chose `q3.chosen`; re-rank; small_lean | results/quality/winner.json; Table 3 | backed |
| retrieval | near the limit: lean step | config [retrieval] k_lean; lean span | backed |
| naive | Q14 shares `trap.Q14.shared` of `trap.Q14.terms` terms, never ranked | data/retrieval/traps.json (shared_terms against the gold piece) | corrected (9) |
| naive | Q5 | traps.json; checked the top and right pieces' text | corrected (10) |
| naive | Q17; decide gate refused it `trap.Q17.refused` of `trap.Q17.asked` | traps.json; refusal spans in results/raw/answers (Q12, by contrast, was refused by the drafter) | corrected (11) |
| retries | what ships | config [retry]; relay/pipeline.py `_try_models` | corrected (12) |
| retries | stricter policy and its numbers | config [retry]; data/experiments/r4_retry_policy/result.json | backed |
| retries | naive client and its numbers, Q8 example | scripts/run_load.py; data/experiments/r4b_policy_vs_naive/result.json and rows | corrected (13, 14) |
| person sees | the reply, verbatim, `daily.ms` | results/raw/runs/daily_exhausted-20261110T192348Z.jsonl | backed |
| person sees | retrieved without the router | its spans: plan, retrieve, send (no understand step) | backed |
| person sees | misses the figures asked for (`daily.grade`) | relay.grade on that reply, for the write-up only (stand-in rows carry no grade in results.csv) | backed, labelled |
| unsure | daily token limit | Table 1; `ledger.max24h_120b`; `ledger.refill_*` | corrected (15, 16) |
| unsure | not built: sentence release; model grader | design (relay/pipeline.py buffers; relay/grade.py is deterministic) | backed |
| unsure | next steps | data/experiments/gate_replay/result.md (Q2); results.csv (Q14) | corrected (17) |
| honesty | replay numbers | `replay.*`: data/experiments/gate_replay/result.json | backed |
| honesty | grader changed after seeing answers; Q13 optional | results/quality/grade_changes_v1_v2.md, grade_changes_v2_v3.md; commit a6ede60 | backed |
| honesty | overrides, unchecked grades | `grades.*`: results/grade_overrides.jsonl, results.csv grade_sure | backed |
| honesty | cap at the documented limit, refill accounting | `ledger.cap100_calls`, `ledger.refill_calls`: ledger lines carry the override | backed; "why" is the author's decision (docs/findings_and_decisions.md) |
| honesty | exclusions | `excluded.*`: results/excluded_runs.json | corrected (19) |

## Not traceable to a generated file (left as is, for the author to judge)

- Authorship ("questions I wrote") and the reasons for the budget overrides are the author's own statements.
- Counts written as words, not digits ("three papers", "two seeds", "each model at most once"): structural, not measured. The checker looks at digits only.
- Length: the prose is over two pages before the author's sections (scripts/check_numbers.py prints the word count).
