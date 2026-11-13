# The Relay v2: critique of every step, what changed, what it measured

Written after the interview review (2026-11-13). The reviewer's points were:
- why no embeddings;
- why BM25 alone;
- why no hierarchical chunking;
- why the constraints weren't stated up front;
- why parse citations out of text instead of carrying them as metadata.

Each is answered below with a change and a measurement, or with the reason it wasn't changed.

The submitted pipeline is untouched and still reproduces every number in the write-up:
`config/pipeline.toml` is as submitted. v2 is `config/production.toml` merged over it, using the same
code.

---

## 0. The constraints, stated up front

The brief left them out on purpose. They should have opened the write-up, each with a number:

| Goal | Constraint I hold the system to | How it is enforced |
|---|---|---|
| Never wrong or embarrassing | No sentence reaches a person unless it is pinned to text in a cited source: every number in a cited source, every quote found in it. Unanswerable means refuse, never guess | per-claim check (v2), citation gate (v1), `NOT_IN_PAPERS` |
| Fast | First sentence within **3 s at p50** when budget allows; nobody waits more than **15 s** for budget; whole request ≤ **30 s** | streaming claims; ladder `max_wait_s`, `deadline_s` |
| Nobody turned away | Every request gets a reply: a full answer, a weaker answer, or cited passages. Never an error page or a hang | the answer ladder; its floor uses no model |
| Cheap | Free tier: $0. Shadow budget ≤ $2 per 1,000 answers at paid prices; ≤ ~3.5k tokens per answer | token accounting per request; cache |
| Inside the provider's limits | Never more than the measured RPM/TPM/RPD per model, **across all replicas**; no retry storms | client limiter (headers) + fleet-wide Redis buckets; failover, not retries |
| Honest under pressure | A degraded reply says it is degraded and why; a daily-budget reply gives no time estimate it can't back | `notice` on every reply |

## 1. Evaluation first: the test set was biased toward BM25

**Problem.** I wrote the 14 answerable questions myself, after reading the papers, mostly in the
papers' own words. BM25 finds the answer for 93% of them. Real users don't phrase questions that way.

**Change.** Added 32 retrieval questions (`data/retrieval/dev_questions.jsonl`) in plain engineer
wording ("how wide is the hidden layer inside each feed-forward block?"), each tied to the blocks
that answer it. All 46 answerable questions together: one question is 2.2 points, instead of 7 points
with 14.

**Result.** The submitted retrieval finds the answer for **50%** of the plain-wording questions, against
93% of the originals. That gap is the reviewer's point, measured.

**Still weak:** the 32 were written by me, after seeing the passages, which biases them toward findable
answers. A question set written by someone who hasn't read the passages is the next step.

## 2. Ingestion (PDF → blocks): kept, with one defect that matters now

The extractor (`scripts/extract_papers.py`) keeps superscripts, sections, captions with tables, and
removes running headers. **Defect found in v2:** it sometimes glues a variable to the word before it
("whereF represents", "andN controls"). That didn't hurt BM25 much, but it made verbatim-quote checks
reject true claims (section 5).

**Fixed downstream:** quotes are compared as letters and digits with spacing removed.

**Not fixed upstream:**
- Equations come out as fragments: Q14's gold passage is seven tiny blocks ("1 2", "1−", "q_t^⊤k_t−1…").
- Tables are flattened text.

A layout-aware parser (Nougat or GROBID for equations and tables) is the right fix, and the largest
remaining source of retrieval misses on math-heavy questions.

## 3. Chunking: hierarchical, small-to-big

**Problem.** One fixed piece size forces a trade-off. Small pieces match precisely but lose context:
600 characters gave 27.8% correct in Table 3. Large pieces keep context but match coarsely.

**Change.** Small-to-big (parent-child), `relay/dense.py`. The system searches **600-character
children**, maps each to its **2,400-character parent** in the same section, and hands the drafter the
distinct parents. Pieces still never cross a section, and tables stay whole.

## 4. Retrieval: hybrid BM25 + embeddings, HyDE, guaranteed lexical slots, optional rerank

**Changes** (all in `relay/dense.py`):
- **Embeddings:** `BAAI/bge-small-en-v1.5` (384-d), run with ONNX on CPU through fastembed, so no torch and
  no GPU. The library has ~1,800 children, so search is one matrix product (4–8 ms). A vector database
  would add a service for nothing at this size; it's the next step when the library reaches ~10⁶ chunks.
- **Hybrid fusion:** BM25 and dense rankings are merged by reciprocal rank fusion (1 ÷ (60 + rank)).
  BM25 still wins on exact tokens ("d_model", "Table 2"); embeddings win on paraphrase.
- **HyDE:** the router (already a model call) also writes one hypothetical answer sentence in the paper's
  vocabulary. It drives the dense search only and is never shown, so a wrong guess costs only a worse
  search, and it adds no model call.
- **Guaranteed lexical slots:** with 3 slots, fusion pushed out passages BM25 ranked 2nd–3rd, which held
  the facts of exact-vocabulary questions (Q11, Q16). Now BM25's top 3 parents always make the cut, and
  fusion fills up to k = 5.
- **Cross-encoder rerank** (`ms-marco-MiniLM-L-6`): available, off by default. It adds ~4 points but takes
  ~1.7 s per query on CPU. On a GPU host it's ~20 ms and should be on.
- Retrieval now runs off the event loop (`asyncio.to_thread`): an inline 1.7 s rerank would freeze every
  other request in the process.

**Measured** (`scripts/eval_retrieval_v2.py` → `data/retrieval/v2.md`, no model calls, raw question as
the query):

| Retrieval | k | All 46: hit@k | Original 14: hit / fact recall | Plain-wording 32: hit | Context chars |
|---|---|---|---|---|---|
| BM25 2400 (submitted) | 3 | 63% | 93% / 84% | 50% | 5.6k |
| dense only | 3 | 67% | 71% / 50% | 66% | 5.3k |
| hybrid 2400 | 3 | 70% | 71% / 61% | 69% | 5.5k |
| small-to-big hybrid | 3 | 76% | 71% / 66% | 78% | 5.9k |
| hybrid 2400 + rerank | 3 | 80% | 86% / 77% | 78% | 5.5k (1.8 s/query) |
| BM25 2400 | 5 | 72% | 93% / 86% | 62% | 9.3k |
| **production: BM25 top 3 + small-to-big hybrid** | **5** | **83%** | **93% / 88%** | **78%** | **9.3k** |

The production setting loses nothing on the original questions, gains 28 points on plain wording, and
costs ~65% more context, about +900 prompt tokens per draft.

**Q14 is still not retrieved by any variant.** "Where does H-Net cut the byte stream?" sits among dozens
of passages about chunking, and its gold passage is equation fragments (section 2). The full pipeline
now answers it partly: the router's HyDE finds the routing-module introduction, so the answer names
the routing module and boundary prediction, but not the cosine-similarity rule.

**Semantic similarity can't detect unanswerable questions.** Their best cosine (0.68–0.79) sits *above*
the answerable median. So refusing on similarity would refuse good questions. Refusals stay with the
router (off-topic questions) and the drafter (`not_in_papers`). The decide step now refuses only when
word overlap **and** semantic similarity are both weak, so a paraphrased answerable question is no
longer refused before drafting.

## 5. Citations as metadata (the reviewer's question)

**Problem.** The v1 drafter wrote prose with `[S1]` markers, and a regex gate parsed them back out.
Every real failure of that gate was a parsing failure, not a grounding failure:
- `【S3】` brackets;
- a zero-width space inside `[S1]`;
- display formulas split into uncited lines;
- `{0,1}` read as one number.

**Change** (`relay/claims.py`). The drafter answers in JSON Lines, one claim per line, and never
writes citation markup:

```json
{"text": "The base model uses a dropout rate of 0.1.", "sources": [1], "quote": "For the base model, we use a rate of P_drop = 0.1"}
{"not_in_papers": "the papers report no p95 latency"}
```

The citation is **metadata**: source IDs plus the exact supporting words, carried from retrieval to the
reply. The system renders the `[S1]` markers itself. Each claim is checked with plain string
operations: its sources were given, its quote occurs in a cited source, and its numbers appear there.

**What metadata removes:** all citation parsing, and its whole class of bugs. Because each line is a
complete claim, claims can be checked and **released one at a time while the model is still writing**
(section 6). A bad claim is dropped on its own instead of failing the whole answer, and the verifier's
verdict can drop single sentences.

**What metadata doesn't remove:** verification. A model can attach the wrong source, or a real quote
that doesn't support the sentence. The quote turns the first case into a free string check; the second
still needs the LLM verifier, which runs when the answer isn't streamed.

**What went wrong on the way, measured on real Groq (gpt-oss-120b, 18 graded questions):**

| Run | Correct + cited | Why |
|---|---|---|
| submitted (BM25, prose + gate) | 72.2% (auto-grades: 66.7%) | — |
| v2, first run | 16.7% | answers grounded but incomplete: terse claims, and true claims withheld |
| v2, run 2 (BM25 slots, k=5, "answer every part") | 33.3% | still 14 claims withheld |
| diagnosis | — | the PDF glues words ("whereF"), so correct quotes failed a word-by-word match; "…"-joined quotes failed |

Fixed: quotes are compared as letters and digits with spacing ignored, each "…" part separately, and a
trimmed quote counts if 8+ consecutive words match. A fabricated quote still fails (`tests/test_claims.py`).
Same-day comparison after the fix: section 9.

## 6. Streaming: time to first token is no longer total time

**Problem.** v1 buffered the whole checked answer, so first token = total. For voice that is dead air.

**Change.** With `on_claim`, each verified claim is released the moment its line completes: over SSE
in the API, straight to TTS in a voice product. The trade-off: the LLM verifier needs the whole answer,
so a streamed answer relies on the per-claim quote check, and the reply says so. A claim already shown
is never retracted, so after the first release the pipeline never restarts on another model.

**Measured, real Groq, excluding budget wait:** first verified sentence p50 **≈ 1.4–1.5 s**; complete
answer p50 ≈ 1.6 s (v2 runs, gpt-oss-120b).

## 7. Router

The router is a model call (~0.4 s, ~300 tokens) on the critical path of every full-level request,
on qwen, which is the most expensive model per token at paid prices.
- **Kept, and made to earn its place:** it now also writes the HyDE sentence.
- **Seen in v2:** qwen once returned an unparseable reply. The pipeline falls back to the raw question,
  as designed.

**Next:** use JSON mode for the router; pick the paper with an embedding classifier (no model call) and
keep the LLM only for rewriting; run the router and a first retrieval in parallel.

## 8. Rate limits, cache and serving: from a script to a service

| Problem in v1 | Change in v2 |
|---|---|
| The limiter only knows its own process's calls; two replicas would each spend the whole budget | **Fleet-wide token buckets in Redis** (one Lua script per take, Redis's clock), per model, from the measured limits × 0.9. A call must take its share before it is sent; if denied, the pipeline steps to the next model, as for a predicted wait. After the call, the charge is corrected to real usage; a 4xx is refunded. The in-process limiter still predicts waits from Groq's headers, which reflect global usage, so the two agree. |
| A burst of 25 people asking the same thing after an outage costs 25 drafts (Table 2b burst: 29% got only passages) | **Answer cache** (normalised question + library version + config; 24 h, 2 min for degraded answers) and **single-flight**: the first request takes a lock, duplicates wait for its result. Measured on the stack: 10 simultaneous identical questions → 1 draft, 9 shared; repeats served in < 1 ms. |
| One shared provider budget, no protection from a noisy client | API keys and a **per-client token bucket** (`RELAY_CLIENT_RPM`); 429 with Retry-After |
| No way to queue work | **Redis Streams job queue** + worker: consumer group, ack after the result is stored, `XAUTOCLAIM` redelivers a dead worker's jobs. At-least-once is safe because the cache makes a repeat cheap. |
| Scripts only | **FastAPI**: `/v1/ask`, `/v1/ask/stream` (SSE), `/v1/jobs`, `/healthz`, `/readyz`, `/metrics`. Two replicas behind nginx (SSE buffering off), Docker Compose, non-root image with the embedding model baked in, `HF_HUB_OFFLINE=1`. |
| Spans per request, no fleet view | **Prometheus**: first-token and total histograms, requests by kind/level/cache, model calls by status, tokens, fleet denials, client limits, withheld claims, in-flight. An events stream holds one summary per request. |

**Why Redis and not Kafka.** The service needs a cache with TTLs, atomic counters, short locks and a
small durable queue with consumer groups. Redis does all five with one dependency. Kafka is the right
tool when many independent consumers replay a high-volume event log for days (analytics, training
data). The events stream is shaped so it can be mirrored to a Kafka topic then, without touching the
producers.

**Found by running the stack:**
- The embedding model wasn't actually baked into the image (the build reused cached vectors), so each
  replica downloaded it on its first request: 16–18 s. Fixed: the build loads the model, the runtime is
  offline, and startup warms the encoder. First request is now 1.1–1.8 s (stand-in model).
- An idle worker's blocking read hit the Redis client's socket timeout and crashed. Fixed: timeouts,
  plus back-off on store errors. Verified by restarting Redis under the worker: it logged, backed off,
  and finished the next job.

## 9. Answer quality after the fixes (real Groq, same day, same drafter)

`scripts/eval_v2_quality.py` → `data/experiments/v2_quality/` (every run, raw drafts and every withheld
claim kept). Same 18 questions and setup as Table 3: verifier off, waits for budget, auto-grades.
One question = 5.6 points.

**Same day, same drafter (gpt-oss-20b), after all fixes:**

| Pipeline | Correct + cited | Wrong (incl. wrongly refused) | Correctly refused | Tokens/answer | First sentence p50* | Complete p50* | Claims withheld |
|---|---|---|---|---|---|---|---|
| submitted (BM25 k=3, prose + gate), auto-grades | 44.4% | 33.3% | 22.2% | – | = complete | – | – |
| new retrieval + prose + gate | **50.0%** | 27.8% | 22.2% | 3,797 | 1,795 ms | 1,795 ms | – |
| **production: new retrieval + claims** | 44.4% | 33.3% (1 wrongly refused) | 22.2% | **2,935** | **1,459 ms** | 1,648 ms | 5 |

\* p50, nearest-rank, excluding budget wait (the eval waits on purpose): router + retrieval + draft + checks.

**What this does and doesn't show.**
- **Answer quality is on par, not better, on these 18 questions.** The differences are one question
  either way, which is within noise.
  - The 18 are mostly phrased in the papers' own words, where BM25 was already strong (93% hit). The
    retrieval gain is on plain-wording questions (50% → 78% hit@k), and those have no gold-fact
    grading yet. Grading answers on them is the next measurement.
- **Claims vs prose (same retrieval, same day):** claims got one question fewer right (Q1 too terse,
  Q16 wrongly refused; Q15 was right only with claims). In exchange:
  - the first sentence arrives ~340 ms sooner and the answer streams;
  - it uses 23% fewer tokens (no citation-format redrafts);
  - there's no citation parsing left to break;
  - single bad claims are withheld instead of the whole answer.
- **gpt-oss-120b wasn't re-run after the quote-matching fixes** (daily budget). Its two earlier runs
  (16.7%, 33.3%) measured the bugs that were then fixed, not the final design.
  - **Before relying on claims mode with 120b, re-run `scripts/eval_v2_quality.py --real` on it.**
  - `[draft] format = "text"` in `config/production.toml` switches back to prose with everything else
    kept.

**Default chosen: claims.** On equal measured quality, it streams, it's cheaper, and it removes a
whole class of failures. The evidence for that is thin (18 questions, one drafter), and the switch above
is one line.

## 10. What I'd do next, in order

1. **Ingestion:** a layout-aware parser for equations and tables (section 2). That is the root of Q14
   and of the quote-matching problems.
2. **A held-out question set written by someone else**, plus a sampled LLM judge on live traffic, since
   regex gold facts miss paraphrases.
3. **Rerank on GPU**, on by default (+4–8 points of retrieval hit rate for ~20 ms).
4. **The router:** JSON mode, an embedding paper classifier, parallel first retrieval.
5. **Run it on Kubernetes:** HPA on in-flight requests, Redis with a replica, and OpenTelemetry traces.
   The spans already exist per request.
6. **A semantic cache** (embedding-similar questions share an answer, behind a strict threshold).
