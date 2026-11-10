"""The rest of the real quality runs (M6), in the author's order, each under the ledger's stop rule.

    .venv\\Scripts\\python scripts\\run_m6_sequence.py

  1. reruns of the 5 requests whose drafts used 【S#】 citations, after the gate fix (the old rows stay
     in results/raw/answers; the reruns supersede them, and their run header says why)
  2. finish small (Q5-Q18; part 1 stopped at Q4 on the old ledger charge)
  3. rebuild results + tables; read the chosen setting (complete settings only)
  4. drafter step at that setting: gpt-oss-20b and qwen (120b's answers at that setting are reused)
  5. sweep part 2: medium+rerank, then small_lean
  6. rebuild results, tables and the trap report
A step that the stop rule halts leaves its remaining questions unrun; the next steps still try.
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
EVAL = [PY, str(ROOT / "scripts" / "eval_retrieval.py")]
GATE_NOTE = ("rerun after the citation-gate fix: this request's drafts used 【S#】 citations, which the old gate "
             "did not read, so the reply fell back to passages or was redrafted; the old row stays in raw")


def run(cmd: list[str]) -> int:
    print(f"\n=== {datetime.now(timezone.utc).isoformat(timespec='seconds')}  {' '.join(cmd[1:])}", flush=True)
    return subprocess.run(cmd, cwd=ROOT).returncode


def rebuild(traps: bool = False) -> None:
    run([PY, str(ROOT / "scripts" / "build_results.py")])
    run([PY, str(ROOT / "scripts" / "make_tables.py")])
    if traps:
        run([PY, str(ROOT / "scripts" / "trap_report.py")])


def main() -> int:
    run(EVAL + ["answers", "--real", "--settings", "large", "--qids", "Q8,Q15",
                "--run-id", "rerun-gatefix-large-20261110", "--note", GATE_NOTE])
    run(EVAL + ["answers", "--real", "--settings", "medium", "--qids", "Q7,Q8,Q11",
                "--run-id", "rerun-gatefix-medium-20261110", "--note", GATE_NOTE])
    run(EVAL + ["answers", "--real", "--settings", "small", "--qids", ",".join(f"Q{i}" for i in range(5, 19)),
                "--run-id", "retrieval-part1b-small-20261110",
                "--note", "finish small: part 1 stopped before Q5 when the old ledger charge (max(estimate + "
                          "max_tokens, usage)) left too little headroom"])
    rebuild()
    winner = json.loads((ROOT / "results" / "quality" / "winner.json").read_text(encoding="utf-8"))
    setting = (winner.get("setting") or {}).get("chosen")
    if not setting:
        print("no complete setting to run the drafter step on; skipping it", flush=True)
    else:
        print(f"drafter step at the chosen setting: {setting}", flush=True)
        run(EVAL + ["drafters", "--real", "--setting", setting, "--drafters", "openai/gpt-oss-20b,qwen/qwen3.8-27b",
                    "--run-id", f"drafter-{setting}-20261110",
                    "--note", f"drafter step at {setting}, the chosen setting after the reruns and small"])
    run(EVAL + ["answers", "--real", "--settings", "medium+rerank", "--run-id", "retrieval-part2-rerank-20261110",
                "--note", "sweep part 2"])
    run(EVAL + ["answers", "--real", "--settings", "small_lean", "--run-id", "retrieval-part2-lean-20261110",
                "--note", "sweep part 2"])
    rebuild(traps=True)
    print("\nsequence done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
