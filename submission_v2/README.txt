The Relay v2: resubmission
==========================

This folder is the resubmission after the interview. The original submission is unchanged in
../submission/ and in the git tag submission-2026-11-12.

In short: embeddings in retrieval, citations carried as metadata, answers that stream, a Redis-backed
service, and the constraints written down. Answer accuracy on the 18 graded questions is on par with
the first submission, not higher; writeup_v2.pdf says what improved and what didn't.

writeup_v2.pdf (writeup_v2.html, WRITEUP_v2.md)
  The write-up: what the review found, what changed, the inferred constraints, retrieval, metadata
  citations, streaming, the service, honest answer-quality results, what's next, how to reproduce.
  Built by scripts/build_writeup_v2.py from WRITEUP_v2.md.

results_v2.csv
  One row per request of the v2 answer-quality runs on real Groq (4 runs x 18 questions). The brief's
  columns come first. Three of them weren't recorded by these runs and are left empty: start_time,
  retries, and the in/out split of tokens (tokens_total is recorded). Extra columns: tokens_total,
  budget_wait_ms, claims_withheld, grade_sure, level, pipeline, drafter, code_state (which runs predate
  the final fixes). Built by scripts/build_results_v2.py from raw/*.jsonl.

tables_v2.md
  Answer-quality and retrieval tables, computed from results_v2.csv and the retrieval results.

retrieval_v2.md
  The full retrieval comparison: 11 variants on 46 answerable questions (14 original + 32 in plain
  wording), no model calls. Built by scripts/eval_retrieval_v2.py.

raw/
  The four real runs as recorded: every reply, its grade and reasons, its claims with quotes, every
  withheld claim with its reason, and (20b runs) the raw model drafts. raw/README.md says which code
  state each run used. dev_questions.jsonl holds the 32 plain-wording retrieval questions.

code_v2.zip
  The repository as of this resubmission (also at https://github.com/Dishan18/Vaani_Relay, main).
  Leaves out this folder, the API key (.env) and local files.

How to reproduce (inside the unzipped code)
-------------------------------------------
  python -m pip install -r requirements-service.txt
  python -m unittest discover -s tests            197 tests: no key, no network, no Redis
  python scripts/eval_retrieval_v2.py             retrieval tables (no model calls)
  python scripts/build_results_v2.py              results_v2.csv and tables_v2.md from the saved runs
  python scripts/build_writeup_v2.py              writeup_v2.html / .pdf (needs Edge or Chrome)
  docker compose up --build                       the service on http://localhost:8080 (stand-in model
                                                  without a key; set GROQ_API_KEY and
                                                  RELAY_LLM_BASE_URL in .env for Groq)
  python scripts/eval_v2_quality.py --real        answer quality on real Groq (spends the daily budget)
