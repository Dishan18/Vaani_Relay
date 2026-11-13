# Production pipeline quality runs (real Groq, 2026-11-13)

All on the 18 graded questions, LLM verifier off, waiting for budget instead of degrading (as Table 3),
auto-grades only. One question = 5.6 points, so differences of one or two questions are noise.

| file | drafter | code state | correct + cited |
|---|---|---|---|
| `v2-20261113T110454Z.jsonl` | gpt-oss-120b | first claims version, k=3 hybrid | 16.7% |
| `v2-20261113T111518Z.jsonl` (+ `result-...-run2-120b.md`) | gpt-oss-120b | BM25 top-3 kept, k=5, "answer every part" | 33.3% |
| `v2@gpt-oss-20b-20261113T112718Z.jsonl` | gpt-oss-20b | + quote matching for glued PDF words, ellipses, table cells; one-line JSON objects | 44.4% |
| `v2_prose@gpt-oss-20b-20261113T112718Z.jsonl` | gpt-oss-20b | same retrieval, the submitted prose drafter + regex gate | 50.0% |

Baselines from the submission, same questions: gpt-oss-120b 72.2% (auto 66.7%), gpt-oss-20b 38.9% (auto 44.4%).
The two 120b runs predate the quote-matching fixes; 120b was not re-run after them (daily budget).
A partial 20b run stopped after 6 questions, before the last parsing fixes, was discarded.
Each row of the 20b runs keeps the raw drafts and every withheld claim with its reason.
