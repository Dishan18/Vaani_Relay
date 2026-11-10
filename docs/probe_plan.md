# Probe plan: what Groq's free tier really does

**Status: approved 2026-11-10 and implemented in `scripts/probe_groq.py`.** Changes made while
implementing, each forced by a dry-run against the local fake, are listed in §4.

**Models.** `openai/gpt-oss-20b`, `openai/gpt-oss-120b` (production) and `qwen/qwen3.8-27b`
(preview).

**Documented limits.** All three: 30 RPM · 1K RPD · 8K TPM · 200K TPD. Source:
`data/limits_documented.json`, retrieved 2026-11-10. These are hypotheses.

**Budget.**
- At most 10% of each model's daily allowance: ≤ 100 requests and ≤ 20,000 tokens per model.
- At most 220 requests overall.
- The key is shared, so other usage can eat the same allowance.

---

## 0. How the budget is enforced (code, not discipline)

| Layer | Where | What it does |
|---|---|---|
| Pre-flight | `scripts/probe_groq.py --dry-run` (to write) | Sums the worst-case reservation of every planned call from the phase table below. Refuses to start if any per-model or total cap could be exceeded. |
| Runtime caps | `relay/ledger.py` + `config/budget.toml` (**built and tested**) | Every real call reserves `estimated prompt + max_tokens` before it is sent. Reservations count calls still in flight. Caps: probe = 10% of documented RPD/TPD per model plus 220 total; account safety net = 90%. Both over a rolling 24 h window. A call that could break a cap is refused with `BudgetExceeded`, and nothing is sent. |
| Charging | `Ledger.settle` | 1 request, plus `max(reservation, reported usage)` tokens. That is correct whatever Groq's rule turns out to be. A 429 is charged its prompt estimate. A connect error is charged nothing. |
| Shared key | `server_floor` | If Groq's own `x-ratelimit-remaining-requests` drops below 50% of the daily limit, every further real call halts. Someone else is using the key, and our measurements would be confounded. |
| Opt-in | `relay/real.py` | Real calls need `--real`. Without it everything runs against the stand-in. |
| No retries | `relay/llm.py` | A failed probe call is a data point, never silently repeated. |

Prompts start with a unique nonce. Groq says cached tokens don't count toward limits, and an
identical repeated prompt could be cached and distort both cost and latency.

---

## 1. What the documentation does not tell us

| # | Unknown | Why it matters to the design | Answered by |
|---|---|---|---|
| U1 | Whether RPM and TPM are **fixed windows** or **refilling buckets**, and what `x-ratelimit-reset-tokens` actually measures. | Decides whether a burst waits seconds or until the minute boundary, and therefore whether a person can be made to wait or must get a smaller answer (rule 2). | P1 |
| U2 | Whether TPM/TPD are charged on **`max_tokens` at admission** or on **actual usage**. Do reasoning tokens count? | If admission is charged, a generous `max_tokens` wastes real budget, so it must be sized per step. | P2 |
| U3 | Whether a **429-rejected** request counts toward RPD or TPM. Is a **cancelled stream** billed in full? | Retry storms and give-ups would cost budget even when nothing was answered. | P2, P3 |
| U4 | Whether limits are **per model or pooled** across the organisation. The docs say "organization level" yet list limits per model. | Whether falling back to another model actually adds capacity. | P3 |
| U5 | Whether there is a **concurrency limit**, and how latency grows with requests in flight. | Sizes our own in-flight cap and the stand-in's load-slowdown curve. | P4 |
| U6 | What a **429 body** says that headers don't: which limit, Limit/Used/Requested, "try again in" versus `retry-after`. | Whether the governor can tell an RPM wall from a TPM wall, and how long it really has to wait. | P3, P2-overflow |
| U7 | Latency distributions per model for **router-sized** and **draft-sized** calls; server-side `queue_time` vs `completion_time` (if usage reports them). | Sizes timeouts and the latency target, calibrates the stand-in, and separates waiting from working. | P6, plus every call |
| U8 | **No header exposes RPM or TPD.** RPM is invisible until a 429, and TPD is not shown at all. | The governor must track both itself. | P3 (RPM); TPD not probed (§3) |
| U9 | Whether the **daily window** is fixed (which time zone) or rolling. | What "the budget is gone until …" can honestly tell the person. | Passive: `reset-requests` on every call (§3) |
| U10 | Whether **headers on a stream** are admission-time values, since they arrive before generation. | How to read U2's evidence. | P2 |
| U11 | Which **reasoning controls** each model accepts (`reasoning_effort`, `reasoning_format`/`include_reasoning`), and how many hidden tokens a minimal call burns. | Reasoning tokens consume `max_tokens` and TPM. If a cap is too tight the answer is cut off before any visible text. | P0 |
| U12 | How stable a **preview** model's limits and availability are. | Whether qwen can be relied on at all. | Not answerable in one session; noted as a risk |

---

## 2. Phases

**Call sizes** (prompts built from the real library in `data/blocks.jsonl`):
- **tiny:** ≈ 40 prompt tokens, `max_tokens` 16 (1 in the RPM burst).
- **router:** ≈ 250 prompt tokens, `max_tokens` 200. Classify the question; return JSON.
- **draft:** ≈ 1,500 prompt tokens (question + 4 passages), `max_tokens` 500. Cited answer.

Reasoning effort is set to the lowest value each model accepts (P0 finds out which).

**Order matters.** Phases that hit walls run last, so cooldowns don't contaminate latency.
Before each step the script waits until the last seen `remaining-tokens` header shows enough
headroom. Waiting costs nothing.

### P0: smoke and parameter check (all models)
- **Calls:** one tiny call with the planned reasoning settings. For qwen, also one call testing
  `reasoning_effort: "none"`; a 400 body is itself an answer.
- **Records:** headers present on a stream (U10); where usage appears (`usage` vs
  `x_groq.usage`, `queue_time`); reasoning tokens burned by a minimal call (U11).
- **Stop rule:** any 401/403/404, or a 400 on the main settings, stops the whole probe.

### P2: admission vs usage accounting (U2, U3, U10). All models, one pair each.
- **Call A:** tiny prompt, `max_tokens` = 2,000, told to reply "OK". Its headers arrive before
  generation, so `remaining-tokens` shows the **admission** charge.
- **Call B:** tiny, immediately after. Its headers show whether A was **refunded** down to actual
  usage.
- **Reading:** an admission drop ≈ 2,000 means `max_tokens` is reserved; a drop ≈ prompt size
  means only the prompt is.
- **Noise:** refill noise is about 133 tokens/s if U1 turns out to be a bucket, against a signal
  of about 1,950.
- **P2-overflow (gpt-oss-20b only, and only if A showed admission charging):** one tiny call with
  `max_tokens` = 8,500, which is more than the TPM. This should be rejected instantly with a TPM
  429 body (U6) at almost zero cost. The ledger must still hold the full 8.5K reservation
  headroom, so it runs early. If P2 showed usage-based charging, it is **skipped**: the call
  could then really generate 8.5K tokens.

### P1: fixed window or refilling bucket (U1)
- **Tokens (20b, 120b):** continuing from P2's call A, send tiny polls at +1, 3, 6, 10, 15 and
  30 s, reading `remaining-tokens` / `reset-tokens`.
- **Reading:** a linear rise of about TPM/60 per second means a bucket. Flat, then a jump to full
  at a boundary, means a window.
- **Requests:** inferred in P3 from recovery after the RPM wall. A bucket readmits one request
  after about 2 s; a window refuses until the minute boundary.

### P4: concurrency (U5)
- **Steps:** router calls with k requests in flight, k = 1 (×3), 2, 4, 8. That is 20b: 1, 2, 4, 8
  (15 calls); 120b: 1, 4, 8 (13 calls); qwen: 1, 4 (5 calls).
- **Spacing:** steps are at least 20 s apart and gated on TPM headroom.
- **Records:** TTFT and total per call versus k; `headers_ms` (queueing before admission)
  separately from first-token time; any 429 whose body mentions concurrency.
- **Signs of a hidden concurrency limit:** a step jump in `headers_ms` or a concurrency 429 at
  some k.

### P6: latency per model (U7)
- **Samples:** router calls, sequential and spaced at least 3 s apart, round-robin across models,
  plus a few draft calls. P4's k = 1 calls count as router samples.
- **What we split out:** TTFT is about prompt size and queueing; decode rate is tokens/s after
  the first token. The stand-in is calibrated from those two parameters, not just end-to-end
  times.

### P3: walls on several models at once (U4, U6, U1-requests, U3). **Runs last.**
- **Burst:** 34 tiny requests each to **20b and 120b at the same time** (k ≤ 4 per model), plus
  10 tiny requests to **qwen in the same minute** as a bystander.
- **Reading:**
  - Per-model limits → 20b and 120b each hit 429 near request 31, and qwen never does.
  - Pooled limits → 429s arrive well before 30 per model, and qwen is refused too.
- **Recovery:** after each model's first 429, one tiny request every 2 s for up to 70 s (or 3
  successes in a row). We compare `retry-after` and the body's "try again in" against when it
  actually recovered.
- **Request counting:** the `remaining-requests` headers on 429s versus successes show whether
  refused requests count toward RPD (U3).

### Planned allocation (worst-case ledger charge, i.e. the reservation)

| Phase | gpt-oss-20b req / tok | gpt-oss-120b req / tok | qwen3.8-27b req / tok |
|---|---|---|---|
| P0 smoke | 2 / 0.6K | 2 / 0.6K | 3 / 0.9K |
| P2 accounting | 2 / 2.1K | 2 / 2.1K | 2 / 2.1K |
| P2-overflow (conditional) | 1 / ~0.1K charged (8.6K headroom) | — | — |
| P1 token polls | 6 / 0.5K | 3 / 0.3K | — |
| P4 concurrency | 15 / 6.3K | 13 / 5.5K | 5 / 2.1K |
| P6 router (extra beyond P4 k=1) | 7 / 2.9K | 5 / 2.1K | 12 / 5.0K |
| P6 draft | 1 / 2.5K | 2 / 5.0K | 3 / 7.5K |
| P3 burst + recovery | ≤ 50 / 4.3K | ≤ 44 / 3.8K | 12 / 1.0K |
| **Total (cap)** | **≤ 84 / ≤ 19.3K (100 / 20K)** | **≤ 71 / ≤ 19.4K (100 / 20K)** | **≤ 37 / ≤ 18.6K (100 / 20K)** |

**Overall: ≤ 192 requests (cap 220).** The remaining 28 are a margin, never retries. Tiny calls
are budgeted at about 85 tokens each: chat templates add prompt tokens our estimate can't see,
and the ledger charges actual usage when it is larger.

**What this budget can't buy: a real p95.** With ≤ 20K tokens per model we get roughly 8–15
router samples and 1–3 draft samples per model. We will report p50 and the maximum, with *n*. p95
will be "not measurable at n < 20" until P6 is **repeated on later days** under that day's 10%,
which also samples different times of day. That spread matters more for the slow tail than a
bigger single session would.

**Where results go.**
- `data/probe/probe_raw.jsonl` (append-only, real runs): one line per call with phase, exact
  payload (prompt text, prompt chars, settings), ledger entry and full `CallRecord` (every chunk
  time and header); plus `event` lines for waits, skips, refusals and decisions.
- `scripts/analyze_probe.py` → `data/probe/limits_table.md` (Table 1), `data/probe/calibration.json`,
  `data/probe/findings.md`.
- Dry-runs: `data/probe/dryrun/<truth>/` with their own throwaway ledger.

---

## 3. What we will NOT probe, and why

- **The daily walls (RPD 1,000 and TPD 200,000 per model).**
  - **Cost:** reaching them takes 10× this probe's budget.
  - **Shared key:** it would spend the whole day's allowance on a shared key, for everyone else
    and for our own later runs; the retrieval sweep alone needs most of a day.
  - **Time:** seeing the recovery would mean waiting up to 24 h.
  - **What we use instead:**
    - RPD comes from the headers on every call. `limit-requests`, `remaining-requests` and
      `reset-requests` are free and are cross-checked against our ledger count. Medium confidence.
    - The daily reset semantics (U9) come passively from how `reset-requests` moves over the day.
    - TPD rests on the documentation alone, since no header shows it. Low confidence.
    - The "daily budget exhausted" scenario runs on the stand-in.
    - Whether to exhaust a real daily limit once, at the very end of a project day (PLAN.md G15),
      is a separate decision for later.
- **A TPM wall bought with real tokens.** It costs at least 8K tokens, about 40% of a model's probe
  budget, for one data point. We use the free P2-overflow call instead, if P2 makes it safe.
  Otherwise the TPM 429 body is reported as "not measured".
- **Long contexts, tool calling, other models, Batch API, audio limits (ASH/ASD).** None of these
  are in our pipeline.
- **Time-of-day effects inside one session.** P6 is repeated on later days instead.

---

## 4. As implemented: changes from the proposal, and why

Every change below came from running the whole probe against `relay/fake_groq.py`, a local fake
whose behaviour we choose (bucket or window, admission or usage charging, per-model or pooled), and
checking that `scripts/analyze_probe.py` recovers that behaviour (`--dry-run --truth ...`).

- **No per-line fsync.** An fsync on the data disk takes about 30–60 ms, measured on this machine.
  With two per call (ledger and raw log) it blocked the event loop and delayed the timestamps of
  every call in flight: tiny calls measured about 2 s in the dry-run. The ledger and the raw log
  now flush every line, which survives a crash of the process, and fsync on close.
- **P1 polls follow a P6 draft call, not P2's call A.** When unused `max_tokens` are refunded at
  the end of a call, A's 2,000-token hold is gone within a second and the polls see nothing to
  refill. A draft's ~1.5K tokens are real usage, so that deficit can't be refunded away.
  - 120b now polls at +1, 4 and 8 s, so that at least two intervals carry information.
  - The analysis corrects each interval for the poll's own admission charge and for the refund or
    usage charge of the call before it, using P2's measured rule.
- **P3 burst: up to 40 requests per model, stopping after three 429s; recovery polls at +2, 4, 7,
  11, 16, 22, 30, 40, 52 and 65 s, stopping after three successes in a row.**
  - With a refilling bucket, a burst that takes several seconds refills a few requests on the way,
    so 34 requests could miss the wall.
  - The burst starts at second :15 of a minute. A wall-clock-minute window, a sliding 60 s window
    and a bucket then recover at clearly different offsets.
  - Allocation: ≤ 50 per gpt-oss model.
- **The template overhead is measured, not assumed.** The ledger's prompt estimate adds a hidden
  chat-template overhead, set to 128 tokens in `config/budget.toml` until it is measured.
  - P0 measures each model's real overhead from `usage.prompt_tokens` and sets it, plus 8 tokens of
    margin, for the rest of the run (logged as an event).
  - A call whose real prompt exceeds the estimate is logged as `estimate_violation`.
- **P3 is protected.** Before any earlier call, the runner checks that the caps still cover P3's
  worst case. If they don't, that call is skipped and logged as `skip_protect_p3`. P6 routers run
  last, so they absorb any shortfall.
- **Every 4xx is charged its prompt estimate only.** That covers 429, 400 and 413; 413 "request
  too large" is what an over-TPM request may get instead of a 429. The proposal only did this for
  429, so a rejected P2-overflow call would have been charged its full 8.6K reservation.
- **Pre-flight is printed twice: at the configured 128-token overhead and at 64.** The real run
  proceeds if the plan fits at either. The ledger caps hold whatever the overhead turns out to be.
  Only the flexible P6 router samples shrink.
- **Concrete settings.** gpt-oss uses `reasoning_effort: low`. P0 also tries
  `include_reasoning: false`. For qwen, P0 tries `none`, then `default` and `low` (both with
  `reasoning_format: parsed`); the lowest accepted wins. This follows Groq's reasoning docs, read
  2026-11-10.
- **Realistic prompts.** Router prompts are a fixed ~420-character routing instruction plus one of
  12 questions about the three papers. Draft prompts are an instruction plus a question plus 4
  passages of about 1.1–1.3K characters, taken from `data/blocks.jsonl` by keyword overlap. The
  nonce leads the first message, so no prefix cache can hit.
- **Production-relevant latency split.** Every call opens a new TCP+TLS connection. The analysis
  reports `connect_ms` separately, plus "after connect" times: what a kept-alive connection would
  see.
