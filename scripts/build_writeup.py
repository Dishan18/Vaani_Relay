"""docs/WRITEUP.md -> submission/writeup.pdf (and writeup.html), and check it fits the brief's 2 pages.

    .venv\\Scripts\\python scripts\\build_writeup.py [--out submission] [--simulate-author-words N]

The markdown is converted with the stdlib (only what the write-up uses: headings, paragraphs, bold, lists,
block quotes, tables) and printed to PDF by the Edge or Chrome already on the machine, headless, so no new
dependency. Pages are counted with pypdf (already a dependency, PLAN.md section 5). Exit 1 if the PDF
is longer than 2 pages. --simulate-author-words N fills each open [AUTHOR: ...] placeholder with N words of
filler, in a separate file, to check the page budget before the author's sections are in.
"""
from __future__ import annotations

import argparse
import html
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "docs" / "WRITEUP.md"
MAX_PAGES = 2
BROWSERS = [Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
            Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
            Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe")]
CSS = """
@page { size: A4; margin: 9mm 10mm; }
body { font-family: "Segoe UI", Arial, sans-serif; font-size: 8.6pt; line-height: 1.17; color: #111; }
h1 { font-size: 12pt; margin: 0 0 3pt; }
h2 { font-size: 9.6pt; margin: 5pt 0 2pt; border-bottom: 0.5pt solid #999; }
p { margin: 0 0 3pt; }
ul { margin: 0 0 3pt 14pt; padding: 0; } li { margin: 0; }
blockquote { margin: 2pt 0 3pt 8pt; padding-left: 5pt; border-left: 1.5pt solid #999; font-size: 6.6pt; line-height: 1.15; }
blockquote p { margin: 0; }
table { border-collapse: collapse; margin: 2pt 0 3pt; font-size: 6.6pt; line-height: 1.1; width: 100%; }
th, td { border: 0.4pt solid #aaa; padding: 0.3pt 2pt; vertical-align: top; text-align: left; }
table.compact td { white-space: nowrap; }
blockquote { page-break-inside: avoid; } .cap { page-break-after: avoid; } tr { page-break-inside: avoid; }
th { background: #eee; }
.cap { font-weight: 600; margin: 3pt 0 1pt; }
"""


def inline(s: str) -> str:
    s = html.escape(s, quote=False)
    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
    return re.sub(r"`([^`]+)`", r"<code>\1</code>", s)


def to_html(md: str) -> str:
    md = re.sub(r"<!--.*?-->", "", md, flags=re.S)
    out, para, i = [], [], 0
    lines = md.splitlines()

    def flush():
        if para:
            out.append(f"<p>{inline(' '.join(para))}</p>")
            para.clear()
    while i < len(lines):
        l = lines[i]
        if not l.strip():
            flush(); i += 1; continue
        if l.startswith("#"):
            flush()
            n = len(l) - len(l.lstrip("#"))
            out.append(f"<h{n}>{inline(l[n:].strip())}</h{n}>"); i += 1; continue
        if l.startswith("|"):
            flush()
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")]); i += 1
            head, body = rows[0], [r for r in rows[1:] if not all(set(c) <= set("-: ") for c in r)]
            compact = all(len(c) <= 40 for r in body for c in r)      # short cells stay on one line
            out.append(f"<table{' class=compact' if compact else ''}><tr>" + "".join(f"<th>{inline(c)}</th>" for c in head) + "</tr>"
                       + "".join("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in r) + "</tr>" for r in body) + "</table>")
            continue
        if l.startswith(">"):
            flush()
            q = []
            while i < len(lines) and lines[i].startswith(">"):
                q.append(lines[i][1:].strip()); i += 1
            out.append("<blockquote>" + "".join(f"<p>{inline(x)}</p>" for x in q if x) + "</blockquote>")
            continue
        if re.match(r"^\s*- ", l):
            flush()
            items = []
            while i < len(lines) and re.match(r"^\s*- ", lines[i]):
                items.append(re.sub(r"^\s*- ", "", lines[i])); i += 1
            out.append("<ul>" + "".join(f"<li>{inline(x)}</li>" for x in items) + "</ul>")
            continue
        if re.fullmatch(r"\*\*Table [^*]+\*\*", l.strip()):
            flush(); out.append(f'<div class="cap">{inline(l.strip()[2:-2])}</div>'); i += 1; continue
        para.append(l.strip()); i += 1
    flush()
    return ("<!doctype html><html><head><meta charset='utf-8'><title>The Relay</title>"
            f"<style>{CSS}</style></head><body>" + "\n".join(out) + "</body></html>")


def browser() -> Path:
    for b in BROWSERS:
        if b.exists():
            return b
    sys.exit("no Edge or Chrome found to print the PDF")


def print_pdf(html_path: Path, pdf_path: Path) -> None:
    with tempfile.TemporaryDirectory() as prof:
        subprocess.run([str(browser()), "--headless=new", "--disable-gpu", f"--user-data-dir={prof}",
                        "--no-pdf-header-footer", f"--print-to-pdf={pdf_path}", html_path.resolve().as_uri()],
                       check=True, capture_output=True, timeout=120)
    if not pdf_path.exists():
        sys.exit(f"the browser wrote no PDF at {pdf_path}")


def pages(pdf: Path) -> int:
    from pypdf import PdfReader
    return len(PdfReader(str(pdf)).pages)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "submission")
    ap.add_argument("--simulate-author-words", type=int, default=0)
    a = ap.parse_args(argv)
    md = SRC.read_text(encoding="utf-8")
    if a.simulate_author_words:
        filler = " ".join(["word"] * a.simulate_author_words)
        md = re.sub(r"\[AUTHOR:[^\]]*\]", filler, md)
        with tempfile.TemporaryDirectory() as td:
            h, p = Path(td) / "sim.html", Path(td) / "sim.pdf"
            h.write_text(to_html(md), encoding="utf-8")
            print_pdf(h, p)
            n = pages(p)
        print(f"simulated: each open author section filled with {a.simulate_author_words} words -> {n} pages "
              f"(limit {MAX_PAGES})")
        return 0 if n <= MAX_PAGES else 1
    open_sections = len(re.findall(r"\[AUTHOR:[^\]]*\]", md))
    a.out.mkdir(parents=True, exist_ok=True)
    h, p = a.out / "writeup.html", a.out / "writeup.pdf"
    h.write_text(to_html(md), encoding="utf-8", newline="\n")
    tmp = p.with_suffix(".tmp.pdf")
    print_pdf(h, tmp)
    shutil.move(str(tmp), str(p))
    n = pages(p)
    print(f"wrote {p.relative_to(ROOT).as_posix()}: {n} pages (limit {MAX_PAGES}); author sections still open: {open_sections}")
    return 0 if n <= MAX_PAGES else 1


if __name__ == "__main__":
    raise SystemExit(main())
