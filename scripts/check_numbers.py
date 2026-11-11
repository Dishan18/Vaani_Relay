"""Fail if any number in the write-up can't be traced to a file a script produced (CLAUDE.md rule 3).

    .venv\\Scripts\\python scripts\\check_numbers.py

Checks, in order (all must pass):
  1. facts are fresh: docs/writeup/facts.json equals a fresh run of scripts/writeup_facts.py;
  2. tables are fresh: results/tables.md equals a fresh run of scripts/make_tables.py;
  3. the template's own text has no digits except names in docs/writeup/allowed_literals.json
     (every number must come in through a {{fact}}, {{quote:fact}} or {{table:...}} placeholder);
  4. every placeholder resolves, and docs/WRITEUP.md is exactly the render of the current inputs
     (nobody edited it by hand).
Reports the prose word count (tables and quotes excluded) and the author placeholders still open.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import render_writeup as rw  # noqa: E402

ALLOWED = ROOT / "docs" / "writeup" / "allowed_literals.json"
DIGIT_TOKEN = re.compile(r"[\w.\-/]*\d[\w.\-/]*")


def untraced_digits(template: str, allowed: list[dict]) -> list[tuple[int, str, str]]:
    """(line number, token, line) for each digit in the template's own text that no allowlist pattern names."""
    pats = [re.compile(p["regex"]) for p in allowed]
    bad = []
    text = rw.COMMENT.sub(lambda m: "\n" * m.group(0).count("\n"), template)
    for n, line in enumerate(text.splitlines(), 1):
        lit = rw.PLACEHOLDER.sub(" ", line)
        for p in pats:
            lit = p.sub(" ", lit)
        for tok in DIGIT_TOKEN.findall(lit):
            bad.append((n, tok, line.strip()))
    return bad


def prose_words(rendered: str) -> int:
    keep = [l for l in rendered.splitlines() if not l.startswith(("|", ">", "<!--", "#"))]
    return len(re.findall(r"\S+", "\n".join(keep)))


def main() -> int:
    py = sys.executable
    failures = []
    r = subprocess.run([py, str(ROOT / "scripts" / "writeup_facts.py"), "--check"], capture_output=True, text=True,
                       encoding="utf-8", cwd=ROOT)
    if r.returncode != 0:
        failures.append("facts are stale: " + (r.stderr or r.stdout).strip())
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "tables.md"
        r = subprocess.run([py, str(ROOT / "scripts" / "make_tables.py"), "--out", str(out)], capture_output=True,
                           text=True, encoding="utf-8", cwd=ROOT)
        if r.returncode != 0:
            failures.append("make_tables.py failed: " + r.stderr.strip()[-300:])
        elif out.read_text(encoding="utf-8") != rw.TABLES.read_text(encoding="utf-8"):
            failures.append("results/tables.md is stale: run scripts/make_tables.py")
    template, facts, tables = rw.load_inputs()
    allowed = json.loads(ALLOWED.read_text(encoding="utf-8"))["patterns"]
    for n, tok, line in untraced_digits(template, allowed):
        failures.append(f"untraced number {tok!r} in the template, line {n}: {line[:120]}")
    try:
        rendered = rw.render(template, facts, tables)
        if not rw.OUT.exists() or rw.OUT.read_text(encoding="utf-8") != rendered:
            failures.append("docs/WRITEUP.md is not the render of the current inputs: run scripts/render_writeup.py")
    except rw.RenderError as e:
        failures.append(f"render failed: {e}")
        rendered = ""
    open_author = re.findall(r"\[AUTHOR:[^\]]*\]", template)
    used = set(m.group(2) for m in rw.PLACEHOLDER.finditer(template) if m.group(1) != "table")
    print(f"placeholders: {len(used)} facts, {sum(1 for m in rw.PLACEHOLDER.finditer(template) if m.group(1) == 'table')} "
          f"tables; prose words (tables, quotes and headings excluded): {prose_words(rendered)}; "
          f"author sections still open: {len(open_author)}")
    if failures:
        print("FAILED:", *failures, sep="\n  ")
        return 1
    print("OK: every number in docs/WRITEUP.md traces to a generated file")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
