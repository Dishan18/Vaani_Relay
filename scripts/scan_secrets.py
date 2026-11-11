"""Fail if a Groq key, the key-fingerprint file, a fingerprint, or .env could leave this machine.

    .venv\\Scripts\\python scripts\\scan_secrets.py

Looks in: every tracked file; every file under submission/ (including each member of a .zip and the text
of each .pdf); and the full git history of every branch (git log -p --all). Fails on:
  * any Groq API key (pattern gsk_ followed by 20+ letters or digits: catches every key ever used,
    including ones no longer on this machine), and the exact key in .env;
  * any key fingerprint listed in results/ledger/accounts.local.json;
  * a file named .env (or .env.*, except .env.example) or accounts.local.json anywhere in the scope.
Prints where something was found and which rule matched, never the match itself. Exit 1 on any finding.
"""
from __future__ import annotations

import io
import json
import re
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KEY_RE = re.compile(rb"gsk_[A-Za-z0-9]{20,}")
FORBIDDEN_NAMES = re.compile(r"(^|[/\\])(\.env(\.(?!example$)[^/\\]+)?|accounts\.local\.json)$")


def secrets() -> list[tuple[str, bytes]]:
    out = []
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if line.startswith("GROQ_API_KEY="):
                k = line.split("=", 1)[1].strip().strip("'\"")
                if k:
                    out.append(("the key in .env", k.encode()))
    acc = ROOT / "results" / "ledger" / "accounts.local.json"
    if acc.exists():
        for fp, label in json.loads(acc.read_text(encoding="utf-8")).items():
            out.append((f"fingerprint of {label}", fp.encode()))
    return out


def check_blob(where: str, data: bytes, known: list[tuple[str, bytes]], findings: list[str]) -> None:
    if KEY_RE.search(data):
        findings.append(f"{where}: a Groq API key (gsk_...)")
    for what, s in known:
        if s and s in data:
            findings.append(f"{where}: {what}")


def pdf_text(data: bytes) -> bytes:
    from pypdf import PdfReader
    return "\n".join(p.extract_text() or "" for p in PdfReader(io.BytesIO(data)).pages).encode("utf-8")


def scan_file(where: str, data: bytes, known, findings) -> None:
    if FORBIDDEN_NAMES.search(where):
        findings.append(f"{where}: forbidden file")
    check_blob(where, data, known, findings)
    if where.lower().endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            for name in z.namelist():
                scan_file(f"{where}!{name}", z.read(name), known, findings)
    elif where.lower().endswith(".pdf"):
        check_blob(f"{where} (text)", pdf_text(data), known, findings)


def main() -> int:
    known = secrets()
    findings: list[str] = []
    tracked = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=True).stdout.split(b"\0")
    n = 0
    for rel in filter(None, tracked):
        p = ROOT / rel.decode("utf-8")
        if p.is_file():
            scan_file(rel.decode("utf-8"), p.read_bytes(), known, findings)
            n += 1
    sub = ROOT / "submission"
    m = 0
    if sub.exists():
        for p in sorted(sub.rglob("*")):
            if p.is_file():
                scan_file(p.relative_to(ROOT).as_posix(), p.read_bytes(), known, findings)
                m += 1
    hist = subprocess.run(["git", "log", "-p", "--all", "--binary", "--no-color"], cwd=ROOT, capture_output=True, check=True).stdout
    check_blob("git history (all branches)", hist, known, findings)
    names = subprocess.run(["git", "log", "--all", "--name-only", "--format="], cwd=ROOT, capture_output=True,
                           check=True).stdout.decode("utf-8", "replace").splitlines()
    for name in sorted(set(filter(None, names))):
        if FORBIDDEN_NAMES.search(name):
            findings.append(f"git history: forbidden file {name} was committed")
    print(f"scanned {n} tracked files, {m} files under submission/, and the git history "
          f"({len(known)} known secrets plus the gsk_ key pattern)")
    if findings:
        print("FAILED:", *sorted(set(findings)), sep="\n  ")
        return 1
    print("OK: no key, fingerprint, .env or fingerprint file found")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
