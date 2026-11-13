# Code guide: what every file does, what you might change, and how to answer new questions live

Repo: `D:\Vaani\Vaani_Relay`. Interview helpers `ask.py` and `recompute_from_csv.py` sit in this folder (not committed). Run everything from here.

---

## v2 additions (production branch)

| File | What it does |
|---|---|
| `relay/dense.py` | Hybrid retrieval: bge-small embeddings + BM25 fused by reciprocal rank, small-to-big (600 → 2400 chars), BM25 top-3 kept, optional cross-encoder rerank; `build_retriever(cfg)` |
| `relay/claims.py` | Citation metadata: JSON-Lines claims `{text, sources, quote}`, per-claim checks (quote in source, numbers), streaming parser, renderer |
| `config/production.toml` | v2 settings merged over the submitted `pipeline.toml` |
| `service/app.py` | FastAPI: `/v1/ask`, `/v1/ask/stream` (SSE), `/v1/jobs`, health, metrics |
| `service/engine.py` | cache, single-flight, fleet-wide Redis rate limit around each model call |
| `service/store.py` | Redis (Lua token buckets, Streams queue) and an in-memory twin |
| `service/worker.py` | job consumer, at-least-once, reclaims dead workers' jobs |
| `scripts/eval_retrieval_v2.py`, `scripts/eval_v2_quality.py` | retrieval comparison (free), answer quality on real Groq |
| `docs/V2_DESIGN.md` | the critique of every step and what each change measured |

## Part 1. The map in one picture

```
data/papers/*.pdf ──extract_papers.py──► data/blocks.jsonl ──retrieve.py──► pieces + BM25 index
                                                                              │
question ──► relay/pipeline.py (plan → router → retrieve → decide → draft → check → send)
                │            uses relay/limiter.py   to predict waits (ladder)
                │            uses relay/llm.py       to stream from an OpenAI-style endpoint
                │            through relay/ledger.py (real calls only: budget caps, log)
                ▼
          Groq (relay/real.py reads the key)   or   standin/server.py (fake Groq)
                ▼
results/raw/*.jsonl ──build_results.py (+ relay/grade.py)──► results/results.csv ──make_tables.py──► tables.md ──► write-up
```

---

## Part 2. Every file, what it does

### `relay/` — the system itself (this is what they'll ask about)

| File | Lines | What it does | Key names to remember |
|---|---|---|---|
| `pipeline.py` | 959 | **The whole request flow.** Steps, ladder, prompts, citation gate, verifier, passages fallback. | `Pipeline._answer` (~555) the flow · `_plan` (~679) picks the ladder level · `_understand` (~864) router · `_retrieve` (~741) + `_fuse` (~759) RRF · decide block (~606) · `_draft` (~879) · `check_citations` (~325) the gate · `_verify` (~897) · `_passages` (~934) floor reply · `_call` (~776) reserve→call→settle · `_try_models` (~817) fallback/retry policy · prompts `ROUTER_SYSTEM`, `DRAFT_SYSTEM`, `VERIFY_SYSTEM` (~377–401) |
| `limiter.py` | 873 | **Client-side copy of Groq's rate limits.** Per-model refilling buckets, header re-anchoring, FIFO queue, wait prediction, learns new limits from 429 bodies. | `Bucket.level` · `predict_wait` · `reserve` → `Grant`/`Ticket`/`Unavailable` · `settle` (applies headers/usage) · `_learn_from_body` · `PromptEstimator` (a + b·chars + c·messages) · `daily_short` |
| `retrieve.py` | 318 | **The retrieval tool.** Turns blocks into section-aware pieces (never crosses a section, keeps tables whole, captions glued to tables), BM25 index, optional lexical rerank, evidence score. | `PAPERS`, `PAPER_TITLES` (the library list) · `build_pieces` · `BM25` · `Retriever.search` · `Retriever.evidence` · `Piece.cite()` → "1706.03762v7 p.4 §3.2.1" |
| `llm.py` | 490 | **Streaming HTTP client**, stdlib only, hand-rolled HTTP/1.1 so it can timestamp every network read (headers, first byte, first visible token). Timeouts for first token, stalls, deadline. Splits gpt-oss reasoning from content. | `LLMClient.stream_chat` · `CallRecord` (status, headers, usage, timings) · `Timeouts` · `Endpoint` (never prints the key) |
| `ledger.py` | 489 | **Budget guard for real calls.** Every real call reserves against caps (90% of documented daily per model, rolling 24 h), logs to `results/ledger/groq_calls.jsonl`, halts on 401 or if Groq's own counter is low. | `BudgetedClient.stream_chat` · `Ledger.reserve/settle` · `charge_of` |
| `real.py` | 95 | **The only place the API key is read** (`GROQ_API_KEY` from env or `.env`). `open_real(...)` refuses unless the caller passes `allow_real=True` (the `--real` flag). | `open_real` · `groq_api_key` |
| `grade.py` | 307 | **Auto-grader, no model.** Checks an answer against gold facts in `data/questions.jsonl` (regex matches + citation to the right blocks), refusal patterns; manual overrides. | `grade()` · `python -m relay.grade override ...` |
| `fake_groq.py` | 356 | A simpler fake Groq used for probe dry-runs and tests (you choose the "truth": buckets vs windows etc.). Not the stand-in. | |

### `standin/server.py` (627) — the fake Groq used for load tests
Asyncio HTTP server with Groq's API, headers and exact 429 text. Per-model buckets with measured limits; lognormal latency; streams reasoning then content at measured tokens/s; switches `p503`, `stall_rate`, `slow_under_load`, `limit_scale`, `exhaust_daily`; replays recorded real answers (`Replay`). `_chat` (~409) is the whole request handling.

### `config/`
| File | What it controls |
|---|---|
| `pipeline.toml` | **Every tunable knob**, each with where its number came from: retrieval (piece size, k, rerank), decide thresholds, model order per step, max_tokens, ladder waits, timeouts, gate options, retry policy. |
| `budget.toml` | Ledger caps (fraction of documented daily allowance, probe cap). |

### `data/`
| Path | What it is |
|---|---|
| `papers/*.pdf`, `papers/MANIFEST.json` | The 3 PDFs at exact versions + sha256 |
| `blocks.jsonl` | 1,171 extracted blocks: `block_id, paper, arxiv_id, page, section, text, kind, in_appendix` |
| `questions.jsonl` | Your 18 questions with gold facts, evidence quotes and match patterns |
| `probe/` | Probe raw log, findings, Table 1, calibration from real Groq |
| `calibration.json` | Stand-in parameters fitted from the probe |
| `limits_documented.json`, `prices.json` | What Groq's docs claim (with URL/date), paid prices for shadow cost |
| `retrieval/` | Top-k retrieval eval and the naive-retriever traps |
| `experiments/` | The iterate rounds (max_wait, retry policy, token margin, first-token timeout, naive retry) |
| `results/detail.jsonl` | Per-request spans of the load runs (for `explain_request.py`, peak queue) |

### `scripts/` — grouped by job
| Group | Scripts |
|---|---|
| **Build the library** | `fetch_papers.py` (download exact arXiv versions, verify), `extract_papers.py` (PDF → blocks, fixing superscripts, ligatures, headers), `audit_blocks.py` (extraction checks), `check_gold.py` (every gold fact is really in the blocks) |
| **Measure Groq** | `probe_groq.py` (the 171-call probe), `analyze_probe.py` (→ findings, Table 1, calibration), `replay_limiter.py` (score limiter predictions vs real headers), `calibrate_standin.py` (→ `data/calibration.json`), `check_key.py` |
| **Run things** | `eval_retrieval.py` (top-k offline; answers through the pipeline, `--real`), `run_load.py` (load scenarios; `SCENARIOS` dict ~line 78), `run_experiment.py` (A/B arms on 2 seeds), `run_m6_sequence.py`, `budget_plan.py` |
| **Analyse** | `explain_request.py` (waiting/working/retrying split), `replay_gate.py` (old vs new gate over saved drafts), `retriever_traps.py`, `trap_report.py`, `grade_diff.py` |
| **Report** | `build_results.py` (raw → results.csv + grades), `make_tables.py` (csv → tables.md), `writeup_facts.py`, `render_writeup.py`, `check_numbers.py`, `build_writeup.py` (→ PDF) |
| **Hygiene** | `scan_secrets.py` (no key leaves the machine), `extract_prompts.py` (your prompts → prompts.md) |

### `tests/` — 168 offline tests, no key needed
`test_pipeline.py` (each ladder rung, gate, refusals), `test_limiter.py`, `test_standin.py`, `test_llm.py` (against `fake_openai_server.py`), `test_ledger.py`, `test_retrieve.py`, `test_grade.py`, `test_run_load.py`, `test_results_build.py`, `test_scan_secrets.py`, `test_writeup.py`, `test_probe.py`, `test_check_gold.py`.
Run: `python -m unittest discover -s tests`

### Docs
`PLAN.md` (plan, deliverables, hypotheses), `CLAUDE.md` (your ground rules), `docs/findings_and_decisions.md` (what changed and why), `docs/probe_plan.md`, `docs/WRITEUP.md` (generated), `docs/writeup/` (facts + audit).

---

## Part 3. Areas you might be asked to change

| Request | File · place | Change | Then |
|---|---|---|---|
| Bigger/smaller pieces, more/fewer pieces | `config/pipeline.toml` `[retrieval]` | `piece_chars`, `k_full`, `k_lean` | `python ask.py "<q>"` to see new retrieval |
| Turn on rerank | same | `rerank = true` | |
| Refuse more/less eagerly | `[decide]` | `refuse_if_best_below`, `refuse_if_top_bm25_below` | |
| Wait longer before degrading | `[ladder]` | `max_wait_s`, `preferred_free_within_s`, `deadline_s` | `python scripts/run_load.py --scenarios burst,overload --duration-s 120` |
| Different drafter / fallback order | `[models]` | reorder lists | |
| Shorter/longer answers | `relay/pipeline.py` `DRAFT_SYSTEM` | rule 5 "At most 6 sentences" | |
| Turn off router or verifier | `[steps]` | `router = false` / `verifier = false` | |
| Retry on | `[retry]` | `enabled = true` | logic in `_try_models` |
| Stricter/looser gate | `[gate]` + `check_citations` | e.g. `drop_plain_sentences = false` | `python scripts/replay_gate.py` |
| Harsher test conditions | `scripts/run_load.py` `SCENARIOS` | `Switches(p503=0.2, stall_rate=0.1, slow_under_load=True)` | run that scenario |
| A new load shape | same dict | `Scenario("spike", 1.0, Switches(), burst=(10.0, 50, 2.0))` | `--scenarios spike` |
| Cache repeated questions | `Pipeline.answer` (~525) | dict keyed by normalised question; store only `kind == "answer"` | sketch in the prep PDF, section 7 |
| Stream checked sentences | `_call` → `self.chat(...)` | pass `on_delta`, split sentences, gate each | |
| Priority users | `limiter.py` `st.queue` in `reserve`/`_acquire` | heap instead of FIFO deque | |

Rule of thumb to say out loud: *"Config first; code only if config can't express it; then run the stand-in to see the effect, and the real service only if quality is the question — because new prompts don't match the stand-in's replays."*

---

## Part 4. If he asks a question about the papers, live

Use the helper `ask.py` (tested this morning). Three modes:

```bash
python ask.py "What is the dropout rate used in the base Transformer?"
```
**Retrieval only — free, instant.** Prints the evidence score, what the decide step would do (draft or refuse), and the top 3 pieces with citations and the best excerpt. Use this first: it shows *where the answer comes from*.

```bash
python ask.py "What is the dropout rate used in the base Transformer?" --standin
```
**Whole pipeline on the stand-in — free.** Shows every step with timings (plan, router, retrieve, decide, draft, check, verify, send) and each model call. Text is a real replay only if the question is one of your 18; otherwise it's placeholder text — say so.

```bash
python ask.py "What is the dropout rate used in the base Transformer?" --real
```
**Whole pipeline on Groq — spends budget (about 2–4 calls, 2–5k tokens).** Your `.env` has a key, so this works. Gives the real cited answer or refusal. Ask before using it ("this spends the shared daily budget — fine?"), then use it.

Options: `--paper attention|moshi|dynamic_chunking` to restrict the search, `--size 1200 --k 5` to try another retrieval setting.

**How to talk through the result:** "The router mapped the question to paper X and rewrote it as …; BM25 found §… with evidence 83%; the decide step let it through; 120b drafted 4 sentences with citations; the gate checked every number is in the cited piece; the verifier on qwen agreed; released at 1.5 s."

**If the answer is wrong:** don't panic, diagnose like an engineer with retrieval-only mode: is the right section in the top 3? If not → retrieval problem (vocabulary mismatch, table vs text: your three traps). If yes but the answer's wrong → drafting problem. If it refused → check evidence score vs thresholds.

---

## Part 5. If he gives you a NEW file

### Fast path (in the interview): `ask.py --pdf`
```bash
python ask.py "What does the paper propose?" --pdf C:\path\new.pdf --key newpaper --title "Title of the new paper"
```
Add `--standin` or `--real` for the full pipeline. What it does, all in memory (repo untouched):
1. Extracts the PDF with plain pypdf: one block per paragraph, section from numbered headings ("3.2 Training").
2. Adds the blocks to the existing 1,171, rebuilds pieces and the BM25 index.
3. Registers the paper everywhere the code names the library: `PAPERS` and `PAPER_TITLES` (retrieve.py), the router prompt and `_PAPER_WORDS` (pipeline.py). Without the router change, the router would say "about none of the papers" and the pipeline would refuse.

Tested with the assignment brief as the "new file": "What columns must results.csv have?" → the right piece ranked #1.

**Say the limitations honestly:** the quick extractor doesn't fix superscripts, ligatures, tables or headers like the real one; a PDF without numbered headings gets one section ("text"), so citations say page only; a scanned PDF (images) has no text layer and would need OCR.

### Proper path (what you'd do for real, ~30 min) — know the steps
1. **Fetch:** add `{"key", "arxiv_id", "version", "title"}` to `PAPERS` in `scripts/fetch_papers.py`, run it (downloads, checks the version, writes sha256 to `MANIFEST.json`). Not on arXiv → put the PDF in `data/papers/` and add a manifest entry by hand.
2. **Extract:** `python scripts/extract_papers.py` → rewrites `data/blocks.jsonl`. It requires an arXiv version stamp on page 1 matching the manifest, so a non-arXiv PDF needs that check relaxed (`extract_paper`, ~line 422).
3. **Audit:** `python scripts/audit_blocks.py`, read a few blocks: headings, tables, equations.
4. **Register:** `PAPERS` + `PAPER_TITLES` in `relay/retrieve.py`; add a line to `ROUTER_SYSTEM` and keywords to `_PAPER_WORDS` in `relay/pipeline.py`; the "three papers" refusal text in `_answer`.
5. **Evaluate:** write questions with gold facts into `data/questions.jsonl` (including unanswerable ones), `python scripts/check_gold.py`, `python scripts/eval_retrieval.py` (offline top-k, free), then answers with `--real`.
6. **Watch the cost:** more pieces don't change per-request tokens (still k pieces), but a 4th paper makes routing and wrong-paper retrieval harder; the gate's "only use a source for claims about its own paper" rule matters more.

### If the file isn't a PDF
- `.txt` / `.md`: same idea — split into paragraphs, make blocks with `paper, arxiv_id (any id), page (1), section, text, kind="body"`. The rest of the pipeline only needs that schema.
- Web page: save as text first.

---

## Part 6. Things to know before you screen-share

- `git status` in `Vaani_Relay` shows `results/ledger/groq_calls.jsonl` modified: that's the 2 real calls from this morning's test of `ask.py --real` (H-Net p95 question → correctly refused). The ledger is append-only by your own rule, so leave it, or say "that's a test call from this morning" if asked.
- Never open `.env` on screen.
- Open a **new** terminal so `python` is 3.12.
- All 168 tests pass: `python -m unittest discover -s tests`.
