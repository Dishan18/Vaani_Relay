"""submission_v2/WRITEUP_v2.md -> submission_v2/writeup_v2.html and writeup_v2.pdf (headless Edge or Chrome).

    python scripts/build_writeup_v2.py
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT / "submission_v2"
CSS = """
@page { size: A4; margin: 14mm 14mm 14mm 14mm; }
:root { --ink:#1b1f24; --muted:#5b6470; --accent:#1f5fae; --line:#d6dbe1; --soft:#eef3fa; }
body { font-family: "Segoe UI", system-ui, Arial, sans-serif; color: var(--ink); font-size: 9.6pt; line-height: 1.45;
       margin: 0; background: #fff; }
h1 { font-size: 18pt; color: var(--accent); margin: 0 0 6px; }
h2 { font-size: 12.5pt; color: var(--accent); border-bottom: 1.5px solid var(--accent); padding-bottom: 2px;
     margin: 16px 0 7px; page-break-after: avoid; }
p { margin: 4px 0 7px; }
ul, ol { margin: 3px 0 7px; padding-left: 18px; }
table { border-collapse: collapse; width: 100%; margin: 5px 0 10px; font-size: 8.6pt; page-break-inside: avoid; }
th, td { border: 1px solid var(--line); padding: 3px 5px; text-align: left; vertical-align: top; }
th { background: var(--soft); }
code { font-family: Consolas, monospace; font-size: 8.4pt; background: #f2f4f7; padding: 0 2px; border-radius: 2px; }
pre { background: #f4f6f9; border: 1px solid var(--line); border-radius: 5px; padding: 6px 8px; white-space: pre-wrap;
      font-size: 8.2pt; page-break-inside: avoid; }
pre code { background: none; padding: 0; }
"""


def browser() -> str | None:
    for p in (r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
              r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
              r"C:\Program Files\Google\Chrome\Application\chrome.exe"):
        if Path(p).exists():
            return p
    return shutil.which("chrome") or shutil.which("chromium") or shutil.which("msedge")


def main() -> int:
    md = (DIR / "WRITEUP_v2.md").read_text(encoding="utf-8")
    body = markdown.markdown(md, extensions=["tables", "fenced_code"])
    html = (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><title>The Relay v2</title>'
            f"<style>{CSS}</style></head><body>{body}</body></html>")
    out_html = DIR / "writeup_v2.html"
    out_html.write_text(html, encoding="utf-8")
    exe = browser()
    if not exe:
        print("no Edge/Chrome found: wrote HTML only")
        return 1
    pdf = DIR / "writeup_v2.pdf"
    subprocess.run([exe, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={pdf}", out_html.as_uri()], check=False, capture_output=True, timeout=120)
    print(f"wrote {out_html.name} and {pdf.name} ({pdf.stat().st_size // 1024} KB)" if pdf.exists() else "PDF failed")
    return 0 if pdf.exists() else 1


if __name__ == "__main__":
    sys.exit(main())
