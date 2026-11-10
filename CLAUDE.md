# The Relay — project rules

Assignment: `The_Relay_Assignment_Brief.pdf`. Plan, deliverables and build order: `PLAN.md`.

## Ground rules (apply to everything in this repo)

1. **Measure first, then design.** Groq's documented limits are hypotheses. Record what the docs
   claim (with URL and date) as a hypothesis, then confirm or refute it with a probe whose raw output
   is saved in the repo. No design decision may rest on an unmeasured number without saying so.
2. **Never make a person wait for budget that isn't coming.** If the remaining per-minute or per-day
   budget cannot serve a request in time, give a smaller answer (fewer steps, smaller model,
   extractive passages, honest "budget gone" message) and say so in the reply. Never queue someone
   behind a reset that is minutes or hours away.
3. **Every number we report comes from a file in the repo that a script produced.** Nothing typed by
   hand. Tables are generated from `results/results.csv` by a script; the write-up pulls numbers from
   generated files. If a number can't be traced to script output, it doesn't go in.
4. **Python + asyncio, as few dependencies as possible, must run on Windows.** Every dependency is
   listed in `requirements.txt` with a one-line justification in `PLAN.md`. Prefer the stdlib.
5. **Commit and push at every milestone** (milestones are listed in `PLAN.md`).

## Working conventions that follow from the rules

- **One shared daily Groq budget.** Every real Groq call goes through the single client, which
  appends to `results/ledger/groq_calls.jsonl` and refuses to call once the day's self-imposed cap is
  reached. Development and tests run against the stand-in by default; real calls require an explicit
  opt-in flag.
- **Secrets.** The Groq key lives only in `.env` (gitignored). Never print it, log it, or commit it.
  Scan for it before every push.
- **Raw data is append-only.** Never edit or delete files under `results/raw/` or the ledger by hand.
  Fix a bad run by re-running it and noting why.
- **Windows.** Open files with `encoding="utf-8"` (and `newline=""` for CSV); use `pathlib`; don't
  rely on `loop.add_signal_handler` or POSIX-only APIs; scripts must run from PowerShell.
