# The Relay — plan

Source: `The_Relay_Assignment_Brief.pdf` (4 pages). Ground rules: `CLAUDE.md`.
Assumed received 2026-11-10, so the deadline is **2026-11-14** ("within 4 days"). Please confirm.

No code exists yet. Groq limits below are written as **hypotheses to measure**, never as facts.

---

## 1. Deliverables

### 1.1 Code

| # | Deliverable | Brief requirement |
|---|---|---|
| C1 | **Pipeline** in Python (asyncio) | At least 4 steps, at least 1 tool call, at least 1 branch or loop; one step retrieves passages from the library. The brief's story: understand the question → retrieve → decide → draft a cited reply → check the reply before sending. |
| C2 | **Library fetcher + index** | Fetch exactly 1706.03762**v7**, 2410.00037**v1**, 2507.07955**v2**; split into pieces; index them for retrieval. Every answer cites where in the papers it came from. |
| C3 | **Real model client** (Groq free tier, own key) | Streaming, so time to first token can be measured. Records every rate-limit header and every 429 body. |
| C4 | **Stand-in model service** | Can be slow, can fail, can refuse at a limit, gets slower under load, streams its reply a piece at a time. Calibrated from real measurements. |
| C5 | **Probe** of the real service | A modest number of real requests: latency spread, limits, behaviour at a limit. |
| C6 | **Load generator** | At least 3 load shapes in the text. Table 2a requires **5 named scenarios** (see gap G1). |
| C7 | **Evaluation**: at least 15 own questions + grader | Answer quality and tokens per answer. Must include questions the papers don't answer. |
| C8 | **Retrieval sweep** | At least 3 settings (piece size, pieces fetched, re-rank yes/no). |
| C9 | **Table generator** | Tables 1, 2a, 2b, 3 computed from `results.csv`. Reviewers recompute them, so they must match exactly. |
| C10 | **Write-up renderer + number checker** | Enforces rule 3: every number in the write-up comes from generated output. |

### 1.2 Runs

| # | Run | Real or stand-in |
|---|---|---|
| R1 | Probe: latency sample, RPM/TPM ceilings, 429 shape, recovery time, concurrency | Real |
| R2 | Stand-in validation: repeat R1 against the stand-in and compare | Stand-in |
| R3 | Retrieval sweep: at least 3 settings × at least 15 questions | Real (quality has to come from the real model) |
| R4 | Quality eval at the chosen setting (may reuse R3's chosen row) | Real |
| R5 | Scenario **steady** | Real (small) + stand-in |
| R6 | Scenario **burst** | Real (inside RPM) + stand-in (beyond) |
| R7 | Scenario **sustained overload** | Stand-in |
| R8 | Scenario **per-minute limit halved** | Stand-in (configured), or real behind a self-imposed half-rate limiter |
| R9 | Scenario **daily budget exhausted** | Stand-in, plus one real exhaustion at the end of a day to see the real 429 and its recovery |
| R10 | Slow-request trace for heading 4 | Picked from R5–R9 by a script (for example the slowest request with a retry) |

### 1.3 Raw files (all kept, all committed)

- `results/results.csv`: one row per request. Required columns, **in this order**:
  `request_id, scenario, source, start_time, first_token_ms, total_ms, status, retries, tokens_in, tokens_out, retrieval_setting, answer_grade`
  - `source` ∈ {real, stand-in}; `status` ∈ {ok, rate_limited, timeout, error};
    `answer_grade` ∈ {correct, wrong, refused_correctly, refused_wrongly}.
  - Proposed extra columns **appended after** the required ones (see G4/G5): `gave_up`, `degraded_level`, `queue_wait_ms`, `model`, `calls`.
- `data/probe/probe_raw.jsonl`: every probe call with its exact payload, full response headers, 429 bodies and timestamps.
- `results/raw/runs/<scenario>-<utc>.jsonl`: per-request span traces (queued / working / retry-wait, first token, each model call and tool call).
- `results/raw/answers/*.jsonl`: full answer text and citations for every graded answer.
- `results/ledger/groq_calls.jsonl`: every real Groq call made by anything, for budget accounting.
- `data/calibration.json`: stand-in parameters, produced by a script from R1.
- `data/limits_documented.md`: what Groq's docs claim, with URL and date. These are the hypotheses.
- Conversation history with the assistant (export of the real session transcripts).

### 1.4 Tables (generated, then copied into the write-up)

- **Table 1** Limits found: RPM, TPM, RPD, TPD, requests in flight, what it returns at a limit, time to recover. Each row has value, how found, confidence.
- **Table 2a** Latency per scenario: real/stand-in, requests sent, duration (s), offered rate/min, TTFT p50/p95/p99, total p50/p95/p99 (ms).
- **Table 2b** Outcomes per scenario: succeeded %, rate-limited %, timed out or given up %, retries per request, peak queue length, tokens per answer, cost per 1,000 answers.
- **Table 3** Retrieval settings: questions asked, correct and cited %, wrong %, correctly refused %, tokens per answer, TTFT p50/p95, total p50/p95.
- Format rules: ms for time, one decimal for percentages, "not measured" where we didn't measure.

### 1.5 Write-up: at most 2 pages, headings in this exact order

1. **What I built and why**: design, and the one decision that shaped it most.
2. **The real service and my stand-in**: limits found and how; stand-in behaviour, calibration, where it differs most. *(Table 1)*
3. **Results**: TTFT and total at median and slow end (state percentiles and n), failures, given-up, per scenario; real vs stand-in; one number nobody could have guessed; quality on at least 15 questions; tokens per answer. *(Tables 2a, 2b)*
4. **Where the time goes**: one slow request split into waiting / working / retrying, before or after first token, and how we know.
5. **When a limit is hit**: at the per-minute limit and at the per-day limit: who is delayed, dropped, given a weaker answer, and why.
6. **Retrieval: quality versus cost**: at least 3 settings, the choice and why, what retrieval does when the limit is close. *(Table 3)*
7. **How this library will mislead a naive retriever**: 3 ways, each with an example question.
8. **Retries** (brief's heading 6): policy, and one case where retrying made things worse.
9. **What the person sees** (brief's 7): the reply at 3pm when the daily budget is gone.
10. **What I'm unsure about, what I left out, what I learned** (brief's 8): least certain assumption and what breaks; two things deliberately not built; one new thing, explained in the author's own words.
11. **Your feedback on Vaani**: placement assumed (G9). **Human-only**: needs the author's own visit to vaani.dev.
12. **How I worked with the assistant** (must be last): real transcript (attached, outside the page limit), then half a page: three prompts that mattered and why, one thing the assistant got wrong and how it was caught, one decision where the author overruled it, what is distinctly theirs. **Human-authored.**

---

## 2. Gaps and contradictions in the brief

| # | Issue | What we'll do |
|---|---|---|
| G1 | Text says "at least three load shapes"; Table 2a says "at least: steady, burst, sustained overload, per-minute limit halved, daily budget exhausted" (five). | Run all five. The table is what gets checked. |
| G2 | **The pipeline "checks the reply before sending it", but we must report time to first token.** If the check runs on the full draft, the first token reaches the person only after drafting and checking finish, so TTFT ≈ total time. Streaming the unchecked draft risks sending the "wrong or embarrassing answer" the company forbids. | Define TTFT as **first checked token delivered to the person, measured from arrival (including queueing)**. Candidate design: check claim by claim and release each sentence once its citation checks out. Decide after measuring how much latency the check adds. Never count a "thinking…" placeholder as a first token. |
| G3 | "Fast" isn't defined, and neither is how soon people give up. | Choose a latency target and a patience distribution for simulated users, and state both as assumptions. Section 8 of the write-up names the least certain one. |
| G4 | `status` has no value for **given up** or **weaker answer**, yet Table 2b has "timed out or given up %" and heading 5 asks who gets a weaker answer. | Map give-ups to `status=timeout` with `gave_up=1` in an appended column. Degraded answers are `status=ok` with `degraded_level>0`. Document the mapping in the write-up. |
| G5 | "Every figure in the tables must be computable from this file", but **peak queue length** can't be computed from the required columns, and neither can how a slow request splits into waiting/working/retrying. | Append `queue_wait_ms`: a request is queued from `start_time` to `start_time + queue_wait_ms`, so a sweep over rows gives peak queue length. Span-level detail lives in the raw traces, keyed by `request_id`. |
| G6 | `answer_grade` has no "not graded" value, but rate-limited, failed and synthetic load-test rows have no answer to grade. | Leave the field empty for ungraded rows and say so. To grade stand-in rows, the stand-in can replay real recorded answers for the same question, so the grade carries over. Label replayed rows as such. |
| G7 | **Cost per 1,000 answers** on a free tier is $0. | Report $0 actual plus a shadow cost from Groq's published paid price for the same model (constant recorded in a file with URL and date). Tokens are in the CSV, so the cost is recomputable. |
| G8 | Percentile **method** isn't specified, but reviewers recompute. With small n, methods disagree, and a p99 from 40 requests is just the maximum. | Use nearest-rank and state it plus n next to every percentile. Mark p99 as "≈ max (n<100)" where that applies. |
| G9 | "One more heading: Your feedback on Vaani" has no stated position. "How I worked with the assistant" must be last. | Put Vaani after heading 8 and before the assistant section. Confirm if you prefer otherwise. |
| G10 | 2 pages for 12 headings plus 4 tables plus half a page on the assistant is very tight. Unclear whether tables count toward the 2 pages. | Assume they count. Keep tables compact, aim for about 1.5 pages of prose, and ask the reviewer if possible. |
| G11 | "Answer quality" needs judgement, but rule 3 says no hand-typed numbers. | Gold answers (key facts + expected paper/section, or "not in papers") are **inputs** written by the author. A script grades against them, and the percentages come from the script. Disagreements found during manual spot checks are logged to a file, never patched into the numbers. An LLM judge would spend the shared budget, so it's optional and only used if measurements show we can afford it. |
| G12 | "At least 15 questions **you wrote yourself**." | The author writes or at least owns the questions. The assistant may propose candidates, and the final set is the author's. Aim for about 20: about 12 answerable, about 5 unanswerable, plus a few aimed at the retriever traps in §7 of the write-up. |
| G13 | "Requests in flight at once" is in Table 1 but nowhere in the brief's text, and may not be documented by Groq. | Probe it carefully (stepwise concurrency) or report "not measured" with a reason. |
| G14 | "Per-minute limit halved" can't be done to the real service. | Use the stand-in config, or our own limiter at half the measured RPM/TPM in front of the real service. Label which. |
| G15 | Real **daily exhaustion** costs the rest of that day's budget, and the recovery time depends on whether the daily window is fixed or rolling, which is unknown. | Do it as the last real action of a day. Observe recovery the next morning. If it's too expensive, report "not measured" and fall back on header-derived values with low confidence. |
| G16 | "Use Groq's free tier", but limits are **per model**. Is spending a second model's quota for degraded answers legitimate routing or a "trick to get around its limits"? | **Decision needed from you** (D1). Default: one primary model; a smaller model only as a disclosed degradation step. |
| G17 | "Cite where in the papers" doesn't fix the granularity (page? section? quote?). | **Decided (M1):** arXiv id with version + page + section heading (appendix flagged) + block id, e.g. `1706.03762v7 p.8 §6.1`. The PDFs give pages and bold headings reliably, and blocks never span a page. |
| G18 | Groq probably offers no embeddings endpoint (hypothesis), and rule 4 limits dependencies. | BM25 in pure Python is the baseline. Any re-ranking is lexical or LLM-based, and LLM re-ranking costs budget. That's a real quality-vs-cost axis for Table 3. |
| G19 | "Keep the raw results for both" plus "one file named results.csv": one file, or one per source? | One `results.csv` with a `source` column covering every run. Per-run raw traces stay separately. |
| G20 | A row is "one row per request", but a request makes several model calls. | One row per **user request**. `tokens_*` sum every call including retries. `retries` counts model-call retries within the request. Probe calls get rows with `scenario=probe`. |

---

## 3. Hypotheses register (rule 1): measured in M3, never assumed

| H | Hypothesis | How we test it | Cost |
|---|---|---|---|
| H1 | RPM and TPM limits as documented for the chosen model | Read headers on every call; short burst to the first 429 | low |
| H2 | RPD and TPD as documented | Headers (`x-ratelimit-*`) on every call; real exhaustion once (G15) | header: free; exhaustion: expensive |
| H3 | Shape of a 429: status, body, `retry-after`, which headers | Captured in R1 burst | low |
| H4 | Recovery time after RPM/TPM 429 | Retry at increasing gaps after a 429 | low |
| H5 | Concurrency limit (if any) | Stepwise parallel requests 1, 2, 4, 8… | medium |
| H6 | TTFT and total latency distributions; long tail | About 30–50 sequential small streamed calls, spread over time | low |
| H7 | Latency grows with prompt size and output length | Varied prompt and `max_tokens` sizes | low |
| H8 | Rate-limit headers arrive on streaming responses; usage arrives in the final chunk | Inspect R1 raw | free |
| H9 | Rejected (429) calls do or don't count toward RPD | Compare remaining counters before and after | free |
| H10 | A stream cancelled by a give-up still bills its full tokens | Cancel mid-stream; compare counters | low |
| H11 | Daily window is fixed (UTC midnight) or rolling | `reset` headers over the day | free |

---

## 4. Proposed design (to be confirmed by measurements)

Pipeline steps, each with its own timeout and span trace:

1. **Understand**: classify and rewrite the question. A cheap model call, or heuristics when budget is short. **Branch:** obviously out of scope → polite refusal, no further spend.
2. **Retrieve** (**tool call**): `search_papers(query, k)` over the BM25 index, with an injected-fault mode for tool timeouts.
3. **Decide**: is the evidence enough? → draft / search again with a reformulated query (**loop**, at most 2) / refuse ("the papers don't say").
4. **Draft**: streamed, cited answer constrained to the retrieved pieces.
5. **Check**: deterministic first (every claim cites a retrieved chunk id; quoted spans exist in that chunk), optional model check if budget allows. On failure: one redraft, else refuse.

**Admission and degradation ladder** (rule 2). One budget governor sees the remaining RPM/TPM/RPD/TPD and the person's patience estimate, and picks the highest level that can finish in time:
L0 full pipeline → L1 skip model in Understand/Check (deterministic only) → L2 fewer, smaller pieces → L3 smaller model (if D1 allows) → L4 **extractive answer**: top cited passages, no model call, labelled as such → L5 honest "budget gone until ~T" message plus the extractive passages. The level is written to `degraded_level`.

**Stand-in**: a separate local HTTP server speaking the same OpenAI-compatible streaming protocol and returning the same header and 429 formats as measured, so the client code path is identical. Latency is drawn from the fitted distributions, slowed by in-flight load, with per-minute/per-day token buckets, injected errors, and "slow success" responses.

**Load generator**: open-loop arrivals (Poisson / burst / ramp), so queueing isn't hidden by coordinated omission. Simulated users have sampled patience and cancel when it runs out.

---

## 5. Repo layout

```
Vaani_Relay/
├── CLAUDE.md                ground rules
├── PLAN.md                  this file
├── README.md                how to run everything (PowerShell commands)
├── requirements.txt         pinned, each justified below
├── .env.example             GROQ_API_KEY=   (real .env is gitignored)
├── config/
│   ├── default.toml         all knobs: models, timeouts, budget caps, ladder thresholds
│   └── scenarios/*.toml     one file per load shape
├── relay/                   the package
│   ├── llm/        client.py (streaming, headers), limiter.py (RPM/TPM/RPD/TPD buckets), ledger.py
│   ├── retrieval/  chunk.py, bm25.py, rerank.py
│   ├── pipeline/   steps.py, tools.py, governor.py (admission + ladder), run.py
│   ├── standin/    server.py, model.py (latency/fault/limit behaviour)
│   ├── load/       shapes.py, users.py (patience), runner.py
│   ├── eval/       grade.py
│   └── trace.py    span timing → raw jsonl
├── scripts/
│   ├── fetch_papers.py        exact versions + sha256 manifest
│   ├── build_index.py
│   ├── probe_groq.py          R1 (real; --dry-run vs relay/fake_groq.py)
│   ├── analyze_probe.py       R1 -> data/probe/{limits_table.md,calibration.json,findings.md}
│   ├── calibrate_standin.py   R1 raw → data/calibration.json
│   ├── validate_standin.py    R2
│   ├── run_scenario.py        R5–R9
│   ├── run_eval.py            R3/R4
│   ├── make_tables.py         results.csv → results/tables.md
│   ├── pick_slow_request.py   R10
│   ├── render_writeup.py      docs/WRITEUP.tmpl.md + generated numbers → docs/WRITEUP.md
│   └── check_secrets.py       run before every push
├── data/
│   ├── papers/                downloaded sources + MANIFEST (sha256)
│   ├── index/                 built index (regenerable)
│   ├── questions.jsonl        author's questions + gold facts/citations
│   ├── limits_documented.md   doc claims = hypotheses
│   └── calibration.json       script output
├── results/
│   ├── results.csv            THE raw file
│   ├── tables.md              generated
│   ├── ledger/groq_calls.jsonl
│   └── raw/{probe,runs,answers}/
├── docs/
│   ├── WRITEUP.tmpl.md        prose with {{placeholders}} for every number
│   ├── WRITEUP.md             rendered
│   └── conversation/          exported assistant transcripts
└── tests/                     stdlib unittest, all against the stand-in
```

**Dependencies** (target: 1–2):

| Package | Why we need it | Stdlib alternative and why not |
|---|---|---|
| ~~`aiohttp`~~ *(dropped in M2)* | Planned for the async HTTP client and the stand-in server. | **Replaced by stdlib** `asyncio` streams + `ssl`. HTTP/1.1 POST with chunked/SSE parsing is about 150 lines and tested offline. Owning the read loop means every network read gets its own timestamp, which is exactly what the latency measurements need. The stand-in server will use `asyncio.start_server` the same way. |
| `pypdf==7.0.0` *(added in M1)* | The brief names PDF versions. Its text visitor gives each run's font, size and position. Without those, `2.3·10^19` flattens to `1019`, `d_model` to `dmodel`, and headings and running headers can't be told from body text. | No stdlib PDF parser. arXiv HTML doesn't exist for all three versions, and LaTeX source would need its own parser and loses page numbers for citations. |

Everything else is stdlib: `tomllib`, `csv`, `json`, `statistics`, `asyncio`, `unittest`, `hashlib`, `re`. No numpy or pandas; percentiles are computed with an explicit nearest-rank function so reviewers can reproduce them.

---

## 6. Build order under one shared daily Groq budget

**Budget discipline**
- Budget is expressed as a **share of the measured daily limit**, whichever of RPD/TPD binds first, because RAG prompts are large and TPD may bind before RPD. We don't know the absolute numbers until M3.
- A self-imposed daily cap of **90%** of the measured limit; the 10% left over is a reserve for re-runs and the write-up's last-minute checks.
- Real calls only via `--real`. Tests, development and debugging always hit the stand-in. Every real call is in the ledger, so the remaining budget is always known.
- Expensive real work runs **right after the daily reset** so that a failure can be fixed and re-run the same day.

**Milestones** (each ends with commit + push)

| M | Day | What | Real budget |
|---|---|---|---|
| M0 | 11-10 | Repo, `CLAUDE.md`, `PLAN.md`, `.gitignore`, remote | 0 |
| M1 | 11-10 | Fetch papers at exact versions (+sha256); inspect PDF vs HTML vs LaTeX; choose parser and citation granularity (G17); chunker + BM25; offline retrieval sanity checks. **M1a done:** `fetch_papers.py`, `extract_papers.py`, `audit_blocks.py` (see §8). **M1b next:** chunker + BM25. | 0 |
| M2 | 11-10 | Streaming client, ledger, hard cap, limiter; stand-in v0 with placeholder parameters; trace spans; results.csv writer. **Client done:** `relay/llm.py` + offline tests (§9). | 0 |
| M3 | 11-10 / 11-11 | Record documented limits as hypotheses → **probe (R1)**: latency sample, size sweep, RPM burst to 429, recovery, concurrency steps. **Measurement-first gate: no ladder thresholds, timeouts or retry policy get fixed before this.** | ≤ 15% |
| M4 | 11-11 | Calibrate stand-in from R1; validate (R2) with quantile comparison; write down where it differs | 0 |
| M5 | 11-11 | Full pipeline + governor + ladder against the stand-in; author writes questions + gold; grader; unit tests | 0 |
| M6 | 11-12 (after reset) | **Retrieval sweep (R3)** at ≥3 settings, then lock the chosen setting; R4 quality eval at that setting. Save every answer for stand-in replay (G6). | ≤ 55% |
| M7 | 11-12 | Real slices: steady + burst within limits, halved-RPM via own limiter (R5, R6, R8 real rows). All five scenarios on the stand-in (R5–R9). | ≤ 20% |
| M8 | 11-12 (end of day) | Real **daily exhaustion** (R9 real), capture the 429 and headers | remainder |
| M9 | 11-13 | Observe daily recovery (H2/H11); `make_tables.py`; slow-request pick (R10); re-run any gaps | ≤ 50% |
| M10 | 11-13 | Write-up template + render + number check; export conversation; secret scan; final push | ≤ 10% reserve |
| — | 11-14 | Buffer / submission | — |

If M3 shows the daily budget is smaller than expected, the first things to shrink are: the number of settings in the sweep (down to the minimum of 3), then questions per setting (down to 15), then real load slices. Stand-in runs are unaffected.

---

## 7. Decisions needed from you

- **D1** (G16): one Groq model only, or a second, smaller model as a disclosed degradation step?
- **D2** (G9, G10): Vaani heading placement OK? Do tables count toward the 2 pages?
- **D3**: confirm the deadline is 2026-11-14.
- **D4**: you'll write the evaluation questions and the human-only sections (Vaani feedback, "what I learned", assistant section). I'll draft candidates only if you ask.

---

## 8. Library extraction (M1a): what the PDFs taught us

Pipeline: `scripts/fetch_papers.py` → `data/papers/MANIFEST.json` (versions checked against the
arXiv API, sha256-pinned) → `scripts/extract_papers.py` → `data/blocks.jsonl` (+
`data/blocks_extract_stats.json`) → `scripts/audit_blocks.py` → `data/blocks_audit.json`.
The audit exits non-zero while any extractor bug it can detect remains. All counts live in
`data/blocks_audit.json`; none are copied here (rule 3).

Decisions that came from looking at the PDFs, not from assumptions:
- **Positioned runs, not `extract_text()`.** Plain extraction flattens super- and subscripts, so
  exponents in Attention Table 2 turn into wrong numbers. Runs are grouped into lines in
  **content-stream order**. Sorting by y interleaved the side-by-side captions in
  dynamic_chunking.
- **Embedded figures are skipped.** Their text comes with figure-local coordinates and private
  glyph encodings (`4UBHF` = "Stage") and was leaking into headings. Consequence for retrieval:
  anything stated only inside a figure is unanswerable and must be refused.
- **Every vocabulary-based fix is checked against the papers' own words.** This covers
  hyphenation, kerning (`T ext`) and digit-word merges, so suspended hyphens in the source
  ("content- and context- dependent") stay as written.
- **TeX big-operator fonts get a mapping.** They have no Unicode map: CMEX `P` and txexs
  `U+02DD` are ∑.
- **Each block carries extra fields beyond the requested schema.** `kind` is one of
  body/heading/caption/table/footnote/reference/front_matter, and `in_appendix` marks appendix
  text. The retriever needs both.

Future retrieval bugs the extractor can't fix (counts in `blocks_audit.json →
future_retrieval_bugs`). These seed write-up §7, "How this library will mislead a naive
retriever":
1. **Table cells lose their column.** A row with an empty cell no longer says which column a
   number belongs to. Keep each table, its caption and its header rows in one chunk. Never read
   a value off a ragged row.
2. **Bibliography entries match questions by title words.** Exclude `kind=reference` by default.
3. **Figure-only facts are gone.** Captions are their own blocks and must be attached to their
   table or figure.
4. **Paragraphs split at page breaks.** The chunker joins same-section body blocks across a page
   break and cites both pages.
5. **Some glyphs have no Unicode mapping, and inline footnote or citation markers remain.**
   These are harmless for retrieval, but an answer must not invent a symbol.

---

## 9. Streaming client (M2): `relay/llm.py`

- **What one class does.** One `LLMClient` handles any OpenAI-compatible endpoint. The Groq and
  stand-in endpoints differ only in base URL and the `source` label written into each record.
- **What every call records.** Every call returns a `CallRecord`, and `record_sink` sees it even
  when the caller cancels. It holds:
  - times: connect, headers, first body byte, first reasoning delta, first visible token, total;
  - every SSE chunk;
  - all `x-ratelimit-*` and `retry-after` headers, verbatim and in order;
  - the whole body of any non-200 response, with `org_…` redacted;
  - usage, finish reason, and the partial text kept so far.
- **Timeouts and retries.** Three timeouts, each of which aborts the TCP connection: no first
  visible token in time, a stall once the body has started, and an overall deadline. A timeout
  record names the stage it fired in: connect, send, headers or body. There are no retries inside
  the client; retrying is the caller's policy.
- **What counts as "visible".** Content outside `<think>…</think>` and outside the
  `reasoning`/`reasoning_content` fields, containing a non-space character. Whitespace after
  `</think>` is not a first token.
- **Offline tests.** `tests/test_llm.py` runs against the scripted local server in
  `tests/fake_openai_server.py`. It covers normal and chunked framing, 429 bodies, all three
  timeouts (including that the server sees the client leave), caller cancellation, connect
  errors, truncated streams, mid-stream error events, and think-tags split across chunks.
  Run: `.venv\Scripts\python -m unittest discover -s tests -t .`

Windows facts found while testing. All matter for the measurement design:
- **Timer granularity.** asyncio timers run on `time.monotonic`, which ticks about every
  15.6 ms on Windows, so a timeout can fire up to one tick early. Recorded times use
  `perf_counter` and are unaffected. Never set timeouts finer than about 20 ms.
- **Refused connections.** A refused TCP connect to localhost takes about 2 s on Windows
  (SYN retries). Stand-in failure modes should answer with HTTP errors, not by refusing
  connections, or the 2 s will pollute latency results.
- **fsync blocks the event loop.** One `os.fsync` on the data disk takes about 30–60 ms. Done
  per call it delayed the timestamps of every call in flight. The ledger and raw logs flush per
  line and fsync on close.

**Ledger + caps done** (`relay/ledger.py`, `config/budget.toml`, `relay/real.py`): rolling-24 h caps
derived from `data/limits_documented.json`, reservation before send, conservative charging,
server-floor halt for the shared key, one-process file lock, `--real` opt-in. Every 4xx is
charged its prompt estimate only. The hidden chat-template overhead is measured per model.

**Probe done (R1)** (2026-11-10), as designed in `docs/probe_plan.md`; §4 there lists what changed
while implementing.
- **Scripts.** `scripts/probe_groq.py` runs it. `scripts/analyze_probe.py` writes
  `data/probe/limits_table.md` (Table 1), `data/probe/calibration.json` and
  `data/probe/findings.md`, which flags anything that contradicts the docs.
- **Raw data.** `data/probe/probe_raw.jsonl`. The spend is in `results/ledger/groq_calls.jsonl`.
- **Validation.** Before any real call, the whole probe ran against `relay/fake_groq.py` under
  three chosen behaviours: bucket/admission/per-model, window/usage/pooled and
  window/admission/per-model. The analysis recovered each one (`data/probe/dryrun/*/findings.md`,
  "Dry-run check" section).
- **What the findings feed (next).** The client-side limiter, the token counts each call
  reserves, whether falling back to another model adds capacity, the in-flight cap, how a 429 is
  read, and the latency targets and timeouts.
- **Not settled.** p95 latency per call size needs P6 repeated on later days:
  `probe_groq.py --real --phases P6`, which reuses the measured settings.

Still open in M2: client-side limiter (built on the probe findings), stand-in v0, results.csv writer.
