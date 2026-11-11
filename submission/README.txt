The Relay - submission
======================

Everything here was produced by scripts in the code at commit 3be364c (the code.zip below), except the
author's own sections of the write-up, which are added by hand to docs/WRITEUP.tmpl.md and rendered the
same way.

writeup.pdf     The write-up: the brief's headings in order, at most 2 pages, the four tables inside.
                Built by scripts/build_writeup.py from docs/WRITEUP.md, which scripts/render_writeup.py
                renders from docs/WRITEUP.tmpl.md, docs/writeup/facts.json (every number, each with the file
                it comes from) and results/tables.md. scripts/check_numbers.py fails if any number in the
                write-up cannot be traced to a generated file. writeup.html is the page the PDF was printed from.

results.csv     The raw file the brief asks for: one row per request, the brief's columns first, then ours
                (explained at the top of scripts/build_results.py). Every table recomputes from it with
                scripts/make_tables.py. Rows of excluded runs stay in it, with superseded = 1 and the reason
                in the "excluded" column.

code.zip        The repository at commit 3be364c (git archive): code, tests, config, the questions and gold
                facts, every raw result (results/raw: real and stand-in runs, per-call records), the call
                ledger (results/ledger/groq_calls.jsonl), the probe log, experiment results, findings
                (docs/findings_and_decisions.md) and the write-up's audit (docs/writeup/audit.md).
                Not included: the API key (.env) and the local key-fingerprint file.

prompts.md      My prompts to the assistant in this session, verbatim and in order, extracted from the
                Claude Code session transcript by scripts/extract_prompts.py (not retyped).

How to reproduce (Windows, PowerShell, from the unzipped code)
---------------------------------------------------------------
  python -m venv .venv
  .venv\Scripts\python -m pip install -r requirements.txt
  .venv\Scripts\python -m unittest discover -s tests          # all against the stand-in, no key needed

  Tables and write-up from the saved results (no model calls):
  .venv\Scripts\python scripts\build_results.py               # results\raw -> results\results.csv
  .venv\Scripts\python scripts\make_tables.py                 # results.csv -> results\tables.md
  .venv\Scripts\python scripts\replay_gate.py                 # old vs fixed citation gate on every saved draft
  .venv\Scripts\python scripts\writeup_facts.py               # -> docs\writeup\facts.json
  .venv\Scripts\python scripts\render_writeup.py              # -> docs\WRITEUP.md
  .venv\Scripts\python scripts\check_numbers.py               # every number traced, nothing stale
  .venv\Scripts\python scripts\build_writeup.py               # -> submission\writeup.pdf (needs Edge or Chrome)

  Load runs on the stand-in (no key, no budget):
  .venv\Scripts\python scripts\run_load.py                    # steady, burst, overload, rpm_halved, daily_exhausted, ...

  Real calls (spend the account's daily budget; every call goes through the ledger and its caps):
  put GROQ_API_KEY=... in .env, then add --real to scripts\eval_retrieval.py answers or scripts\run_load.py.

  Before sharing anything:
  .venv\Scripts\python scripts\scan_secrets.py                # fails on any key, fingerprint or .env
