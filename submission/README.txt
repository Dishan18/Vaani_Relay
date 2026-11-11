The Relay: submission
=====================

This folder holds everything I am submitting. Every number in it was produced by the scripts in
code.zip; the only text written by hand is the prose of the write-up.

writeup.pdf
  The two-page write-up, with the brief's headings in order and the four tables inside.
  writeup.html is the page it was printed from. The PDF is built by scripts/build_writeup.py from
  docs/WRITEUP.md, which scripts/render_writeup.py fills in from docs/WRITEUP.tmpl.md,
  docs/writeup/facts.json and results/tables.md. Each number in facts.json names the file it comes
  from, and scripts/check_numbers.py fails if any number in the write-up can't be traced that way.

results.csv
  The raw file the brief asks for: one row per request, the brief's columns first and mine after
  them (each is explained at the top of scripts/build_results.py). Every cell of Tables 2a, 2b and 3
  can be recomputed from this file alone, including peak queue length (from the wait_spans_ms
  column). Table 1 lists the service's limits, which are not per request, so it comes from the
  probe log and the call ledger instead. Rows from excluded runs stay in the file, marked
  superseded = 1 with the reason in the "excluded" column.

code.zip
  The repository at commit 85612c3 (git archive; the zip's top folder carries the same name). It holds the
  code, tests and config; my questions and their gold facts; every raw result (results/raw, real and
  stand-in, with a record of each model call); the ledger of every real Groq call; the probe log;
  the experiment results; my findings log (docs/findings_and_decisions.md); and the write-up's audit
  (docs/writeup/audit.md). It leaves out this submission folder itself, the API key and the local
  key-fingerprint file.

prompts.md
  My prompts to the assistant in this session, copied verbatim and in order from the Claude Code
  session transcript by scripts/extract_prompts.py.

How to reproduce (Windows, PowerShell, inside the unzipped code)
-----------------------------------------------------------------
  python -m venv .venv
  .venv\Scripts\python -m pip install -r requirements.txt
  .venv\Scripts\python -m unittest discover -s tests          all tests run on the stand-in, no key needed

  Tables and write-up from the saved results, with no model calls:
  .venv\Scripts\python scripts\build_results.py               results\raw -> results\results.csv
  .venv\Scripts\python scripts\make_tables.py                 results.csv -> results\tables.md
  .venv\Scripts\python scripts\replay_gate.py                 old vs fixed citation check on every saved draft
  .venv\Scripts\python scripts\writeup_facts.py               -> docs\writeup\facts.json
  .venv\Scripts\python scripts\render_writeup.py              -> docs\WRITEUP.md
  .venv\Scripts\python scripts\check_numbers.py               every number traced, nothing stale
  .venv\Scripts\python scripts\build_writeup.py               -> submission\writeup.pdf (needs Edge or Chrome)

  Load runs on the stand-in, with no key and no budget:
  .venv\Scripts\python scripts\run_load.py

  Real calls spend the account's daily budget, and every one goes through the ledger and its caps.
  Put GROQ_API_KEY=... in .env, then add --real to scripts\eval_retrieval.py answers or
  scripts\run_load.py.

  Before sharing anything:
  .venv\Scripts\python scripts\scan_secrets.py                fails on any key, fingerprint or .env
