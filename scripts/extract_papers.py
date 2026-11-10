"""Extract the three library PDFs into retrieval blocks: data/blocks.jsonl.

One JSON object per block:
  block_id, paper, arxiv_id, page, section, text      (required schema)
  kind, in_appendix                                   (extra: lets the retriever and the audit
                                                       tell body text from captions, tables,
                                                       footnotes, references and figure labels)

How it works (decisions taken from inspecting the PDFs, see PLAN.md / blocks_audit.json):
  * pypdf's visitor gives every text run with its font, size and transformation matrix.
    Plain extract_text() throws that away, which is what flattens 10^19 into "1019" and
    d_model into "dmodel".
  * Runs are grouped into lines by baseline. A smaller run sitting above/below the baseline
    is a superscript/subscript and is written as ^x / _x.
  * Rotated runs are dropped from the text (figure axis labels), except the arXiv stamp in
    the page-1 margin, which must match the manifest's id and version or we abort.
  * Running headers/footers = lines in the top/bottom page bands that repeat across pages
    (digits normalised), plus bare page numbers. They are removed.
  * Headings = a line set entirely in a bold font that looks like a section heading
    (numbered "3.2 ...", appendix "A.1 ...", or a known unnumbered heading, or larger
    than body text). Everything after the References heading is appendix.
  * Text fixes that are pure extractor bugs: hyphenation at line breaks, spacing accents,
    ligatures, kerning splits ("T ext"), digit-word merges ("6identical"), bracket spacing.
    Every fix that depends on guessing is checked against the papers' own vocabulary.
  * Counters for every fix are written to data/blocks_extract_stats.json for the audit.

Usage (PowerShell):  .venv\\Scripts\\python scripts\\extract_papers.py
"""
from __future__ import annotations

import collections
import json
import math
import re
import sys
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "papers" / "MANIFEST.json"
OUT = ROOT / "data" / "blocks.jsonl"
STATS = ROOT / "data" / "blocks_extract_stats.json"

BOLD_FONT = re.compile(r"Bold|Medi|BX|TB$|Semibold|SemiBold|Black|Heavy|-B$", re.I)
FIGURE_FONT = re.compile(r"Arial|DejaVu|Helvetica|AppleSDGothic", re.I)
STAMP = re.compile(r"arXiv:(\d{4}\.\d{4,5})v(\d+)\s+\[[^\]]+\]\s+\d{1,2} \w{3} \d{4}")
NUMBERED_HEADING = re.compile(
    r"^(?:\d{1,2}(?:\.\d{1,2}){0,3}|[A-H](?:\.\d{1,2}){0,3}|Appendix [A-Z])\.?\s+[A-Z(]")
UNNUMBERED_HEADINGS = {"abstract", "references", "bibliography", "acknowledgements",
                       "acknowledgments", "acknowledgement", "acknowledgment", "appendix",
                       "appendices", "supplementary material", "contents"}
NUM_TOKEN = re.compile(r"[-+]?[\d.,]+(?:[%x×]|\^\S+)?|[\d.]+±[\d.]+|–|-|[\d.]+/[\d.]+|-/[\d.-]+")
CAPTION = re.compile(r"^(?:Figure|Fig\.|Table)\s*\d+\s*[:.|]")

# spacing accent -> combining mark. Most precede the letter they modify; cedilla/ogonek follow it.
ACCENT_BEFORE = {"´": "́", "`": "̀", "¨": "̈", "ˆ": "̂",
                 "˜": "̃", "ˇ": "̌", "˚": "̊", "˝": "̋",
                 "¯": "̄", "˙": "̇"}
BIG_OPERATOR_FONTS = [
    (re.compile(r"^CMEX", re.I), str.maketrans({"P": "∑", "X": "∑", "Q": "∏",
                                                "Y": "∏", "R": "∫", "Z": "∫",
                                                "p": "√", "q": "√", "r": "√",
                                                "s": "√"})),
    (re.compile(r"^txex", re.I), str.maketrans({"˝": "∑"})),
]
_BIGOP_HITS: list[str] = []  # filled by the visitor, drained into Stats per page
MATH_ACCENT = {"ˆ": "̂", "˜": "̃", "¯": "̄", "˙": "̇"}
ACCENT_AFTER = {"¸": "̧", "˛": "̨"}
LIGATURES = "ﬀﬁﬂﬃﬄﬅﬆ"


@dataclass
class Run:
    x: float
    y: float
    size: float
    font: str
    text: str
    bold: bool
    rotated: bool


@dataclass
class Line:
    runs: list[Run]
    y: float = 0.0
    size: float = 0.0
    x: float = 0.0
    text: str = ""
    bold: bool = False
    figure_font: bool = False
    page: int = 0


@dataclass
class Stats:
    c: collections.Counter = field(default_factory=collections.Counter)
    examples: dict = field(default_factory=lambda: collections.defaultdict(list))

    def hit(self, key: str, example: str | None = None, n: int = 1) -> None:
        self.c[key] += n
        if example is not None and len(self.examples[key]) < 5:
            self.examples[key].append(example)


# --------------------------------------------------------------------------- runs

def _form_xobjects(page) -> list:
    res = page.get("/Resources")
    res = res.get_object() if res is not None else {}
    xobjs = res.get("/XObject")
    if xobjs is None:
        return []
    out = []
    for ref in xobjs.get_object().values():
        obj = ref.get_object()
        if obj.get("/Subtype") == "/Form":
            out.append(obj)
    return out


def page_runs(page, include_forms: bool = False) -> list[Run]:
    """Text runs of one page.

    Embedded figures are Form XObjects. Their text comes out with figure-local coordinates
    (y beyond the page, x in the margin) and often a private glyph encoding ("4UBHF" for
    "Stage"), so by default we skip them: pypdf does not descend into an XObject whose
    /Subtype is /Image, so we relabel forms for the duration of the call and restore them.
    """
    forms = [] if include_forms else _form_xobjects(page)
    from pypdf.generic import NameObject
    for obj in forms:
        obj[NameObject("/Subtype")] = NameObject("/Image")
    try:
        return _page_runs(page)
    finally:
        for obj in forms:
            obj[NameObject("/Subtype")] = NameObject("/Form")


def _page_runs(page) -> list[Run]:
    runs: list[Run] = []

    def visit(text, cm, tm, font, size):
        if not text or not text.strip():
            return
        a = tm[0] * cm[0] + tm[1] * cm[2]
        b = tm[0] * cm[1] + tm[1] * cm[3]
        c = tm[2] * cm[0] + tm[3] * cm[2]
        d = tm[2] * cm[1] + tm[3] * cm[3]
        e = tm[4] * cm[0] + tm[5] * cm[2] + cm[4]
        f = tm[4] * cm[1] + tm[5] * cm[3] + cm[5]
        rotated = abs(math.atan2(b, a)) > 0.1
        eff = size * math.hypot(c, d)
        name = str((font or {}).get("/BaseFont", "?")).split("+")[-1]
        # pypdf marks line ends with "\n"; positions are what we trust, so drop it.
        clean = text.replace("\n", " ").rstrip()
        # TeX big-operator fonts have no Unicode map: CMEX stores the summation sign under the
        # code for "P" (display size "X"), so "∑ q_i k_i" extracts as "P q_i k_i". txexs (TX fonts)
        # comes out as U+02DD for the same glyph (observed in dynamic_chunking p8).
        for pattern, table in BIG_OPERATOR_FONTS:
            if pattern.search(name):
                mapped = clean.translate(table)
                if mapped != clean:
                    _BIGOP_HITS.append(clean)
                    clean = mapped
                break
        runs.append(Run(e, f, round(eff, 2), name, clean, bool(BOLD_FONT.search(name)), rotated))

    page.extract_text(visitor_text=visit)
    return runs


def build_lines(runs: list[Run], body: float, stats: Stats) -> list[Line]:
    """Group runs into lines.

    Runs are taken in content-stream order, not sorted by position: LaTeX writes a
    side-by-side minipage (two captions, a figure next to text) one column at a time, so
    stream order keeps the columns apart, while sorting by y interleaves them line by line.
    A new line starts when the baseline moves by more than ~0.6 em or x jumps backwards.
    Smaller runs above/below the baseline are written ^x / _x; a run smaller than and
    below a superscript is a subscript of that superscript (R^d_model).
    """
    rows: list[list[Run]] = []
    cur: list[Run] = []
    host: Run | None = None
    for r in runs:
        if cur and host is not None:
            tol = 0.6 * max(host.size, r.size)
            if abs(r.y - host.y) <= tol and r.x >= cur[-1].x - 1.5 * max(host.size, r.size):
                cur.append(r)
                if r.size > host.size + 0.3 or (abs(r.size - host.size) <= 0.3
                                                and len(r.text.strip()) > len(host.text.strip())):
                    host = r
                continue
            rows.append(cur)
        cur, host = [r], r
    if cur:
        rows.append(cur)

    def row_size(row):
        # Body size wins whenever the line contains body-size text: on short math lines the
        # subscript can carry more characters than its base ("P" + "drop" + "= 0.1").
        if any(abs(r.size - body) <= 0.6 for r in row):
            return body
        chars = collections.Counter()
        for r in row:
            chars[r.size] += len(r.text.strip())
        return chars.most_common(1)[0][0]

    lines = []
    for row in rows:
        host_size = row_size(row)
        base = [r for r in row if r.size >= 0.88 * host_size]
        base_y = collections.Counter(round(r.y, 1) for r in base).most_common(1)[0][0]
        parts: list[str] = []
        mode = ""
        prev_run: Run | None = None
        prev_script: Run | None = None
        for r in sorted(row, key=lambda r: r.x):
            rel = r.y - base_y
            small = r.size < 0.88 * host_size
            lvl = ""
            if small and rel > 0.12 * host_size:
                lvl = "^"
            elif small and rel < -0.08 * host_size:
                lvl = "_"
            if lvl and mode.startswith("^") and prev_script is not None                     and r.size < 0.88 * prev_script.size and r.y < prev_script.y - 0.08 * prev_script.size:
                lvl = "^_"
            # pypdf only sometimes marks a gap between runs with a leading space. Side-by-side
            # cells (author grids, table columns) then glue together ("Google BrainGoogle").
            # Estimate where the previous run ended (0.5 em per char) and add a space if the
            # next run starts clearly after that, or if the font changes between two words
            # ("at the" + bold "Large" -> "theLarge").
            if prev_run is not None and not lvl and parts and not parts[-1].endswith(" ")                     and not r.text.startswith(" "):
                est = 0.5 * prev_run.size * len(prev_run.text)
                if r.x - (prev_run.x + est) > 0.4 * r.size + 0.08 * est:
                    parts.append(" ")
                    stats.hit("gap_space_inserted", f"{prev_run.text[-15:]}|{r.text[:15]}")
                elif not mode and prev_run.font != r.font and re.search(r"[a-z]$", prev_run.text)                         and re.match(r"[A-Za-z]{2}", r.text):
                    parts.append(" ")
                    stats.hit("font_change_space_inserted", f"{prev_run.text[-15:]}|{r.text[:15]}")
            if lvl:
                t = r.text.strip()
                if lvl != mode:
                    if lvl == "^_":
                        parts.append("_")
                    elif not (mode == "^_" and lvl == "^"):
                        parts.append(lvl)
                    stats.hit("supsub_marked", f"{''.join(parts)[-25:]}{t}")
                parts.append(t)
                prev_script = r
            else:
                parts.append(r.text)
                prev_script = None
            mode = lvl
            prev_run = r
        text = re.sub(r"\s+", " ", "".join(parts)).strip()
        chars = sum(len(r.text) for r in row)
        fig_chars = sum(len(r.text) for r in row if FIGURE_FONT.search(r.font))
        lines.append(Line(
            runs=row, y=base_y, size=host_size, x=min(r.x for r in row), text=text,
            bold=all(r.bold for r in base if r.text.strip()),
            figure_font=fig_chars > chars / 2,
        ))
    return lines


# --------------------------------------------------------------------------- text fixes

def fix_accents(t: str, stats: Stats) -> str:
    def before(m):
        stats.hit("accent_fixed", m.group(0))
        return unicodedata.normalize("NFC", m.group(2) + ACCENT_BEFORE[m.group(1)])

    def after(m):
        stats.hit("accent_fixed", m.group(0))
        return unicodedata.normalize("NFC", m.group(1) + ACCENT_AFTER[m.group(2)])

    def math_after(m):
        stats.hit("math_accent_fixed", m.group(0))
        return m.group(1) + MATH_ACCENT[m.group(2)]

    # Math accents (hat, tilde, bar, dot) placed after their variable and not followed by a
    # letter: "xˆ^s" -> "x̂^s", "z ˆ^S" -> "ẑ^S". Text accents precede their letter.
    t = re.sub(r"([A-Za-z]) ?([" + "".join(MATH_ACCENT) + r"])(?! ?[A-Za-z])", math_after, t)
    t = re.sub("([" + "".join(ACCENT_BEFORE) + r"]) ?([A-Za-zı])", before, t)
    t = re.sub(r"([A-Za-z]) ?([" + "".join(ACCENT_AFTER) + "]) ?", after, t)
    return t.replace("ı́", "í")  # dotless i + acute (LaTeX \'{\i})


def fix_ligatures(t: str, stats: Stats) -> str:
    if any(ch in t for ch in LIGATURES):
        stats.hit("ligature_fixed", n=sum(t.count(ch) for ch in LIGATURES))
        t = "".join(unicodedata.normalize("NFKC", ch) if ch in LIGATURES else ch for ch in t)
    return t


def fix_math_alnum(t: str, stats: Stats) -> str:
    """Mathematical italic/bold letters (U+1D400-1D7FF, e.g. '𝑆') -> plain letters, so a
    query for "S = 2" can match. NFKC maps them to their ASCII/Greek base letters."""
    if any(0x1D400 <= ord(ch) <= 0x1D7FF for ch in t):
        n = sum(1 for ch in t if 0x1D400 <= ord(ch) <= 0x1D7FF)
        stats.hit("math_alnum_normalized", t[:40], n=n)
        t = "".join(unicodedata.normalize("NFKC", ch) if 0x1D400 <= ord(ch) <= 0x1D7FF else ch
                    for ch in t)
    return t


def basic_fixes(t: str, stats: Stats) -> str:
    t = fix_math_alnum(fix_ligatures(fix_accents(t, stats), stats), stats)
    new = re.sub(r"([\[(]) +", r"\1", t)
    new = re.sub(r" +([\])])", r"\1", new)
    if new != t:
        stats.hit("bracket_space_fixed")
    return new


def vocab_fixes(t: str, vocab: collections.Counter, stats: Stats) -> str:
    def kern(m):
        joined = m.group(1) + m.group(2)
        if vocab[joined.lower()] >= 2:
            stats.hit("kerning_split_fixed", m.group(0))
            return joined
        return m.group(0)

    def digit_word(m):
        if vocab[m.group(2).lower()] >= 2 and vocab[m.group(0).lower()] == 0:
            stats.hit("digit_word_merge_fixed", m.group(0))
            return f"{m.group(1)} {m.group(2)}"
        return m.group(0)

    t = re.sub(r"\b([A-Z]) ([a-z]{2,})\b", kern, t)
    t = re.sub(r"\b(\d+)([a-z]{4,})\b", digit_word, t)
    return t


def join_lines(lines: list[str], vocab: collections.Counter, stats: Stats, keep_newlines: bool) -> str:
    out = ""
    for ln in lines:
        if not out:
            out = ln
            continue
        m = re.search(r"([A-Za-z]+)-$", out)
        n = re.match(r"([a-z]+)", ln)
        if m and n and not keep_newlines:
            joined = (m.group(1) + n.group(1)).lower()
            hyph = f"{m.group(1)}-{n.group(1)}".lower()
            if vocab[joined] > vocab[hyph]:
                stats.hit("hyphenation_joined", f"{m.group(1)}-|{n.group(1)} -> {joined}")
                out = out[:-1] + ln
                continue
            stats.hit("hyphen_kept", f"{m.group(1)}-{n.group(1)}")
            out = out + ln
            continue
        out = out + ("\n" if keep_newlines else " ") + ln
    return out


# --------------------------------------------------------------------------- headers / footers

def norm_hf(t: str) -> str:
    return re.sub(r"\d+", "#", re.sub(r"\s+", " ", t)).strip().lower()


def find_header_footer(pages_lines, heights) -> set[str]:
    counts: collections.Counter = collections.Counter()
    for lines, (bottom, top) in zip(pages_lines, heights):
        h = top - bottom
        seen = set()
        for ln in lines:
            rel = (ln.y - bottom) / h
            if rel > 0.92 or rel < 0.08:
                seen.add(norm_hf(ln.text))
        counts.update(seen)
    min_pages = max(3, int(0.2 * len(pages_lines)))
    return {t for t, n in counts.items() if n >= min_pages}


# --------------------------------------------------------------------------- headings

def heading_text(ln: Line, body: float) -> bool:
    if not ln.bold or ln.figure_font:
        return False
    t = ln.text.strip()
    words = t.split()
    if not words or len(words) > 16 or t.endswith((".", ",", ":")) and not NUMBERED_HEADING.match(t):
        return False
    if NUMBERED_HEADING.match(t):
        if t.endswith("."):
            return False
        num = words[0].rstrip(".")
        top = num.split(".")[0]
        return not top.isdigit() or int(top) <= 30
    if t.lower() in UNNUMBERED_HEADINGS:
        return True
    return ln.size >= 1.15 * body and len(words) <= 10


# --------------------------------------------------------------------------- per paper

def extract_paper(meta: dict, stats: Stats) -> list[dict]:
    path = ROOT / meta["file"]
    reader = PdfReader(str(path))
    raw_pages = [page_runs(p) for p in reader.pages]
    for hit in _BIGOP_HITS:
        stats.hit("big_operator_glyph_mapped", hit)
    _BIGOP_HITS.clear()
    for p, runs in zip(reader.pages, raw_pages):
        if _form_xobjects(p):
            with_forms = sum(len(r.text) for r in page_runs(p, include_forms=True))
            dropped = with_forms - sum(len(r.text) for r in runs)
            if dropped > 0:
                stats.hit(f"figure_form_chars_dropped:{meta['key']}", n=dropped)
    _BIGOP_HITS.clear()  # the with-forms pass is only for counting; don't count its hits
    heights = [(float(p.mediabox.bottom), float(p.mediabox.top)) for p in reader.pages]

    # arXiv stamp on page 1 (rotated, left margin) must match the manifest.
    stamp_runs = [r for r in raw_pages[0] if r.rotated and STAMP.search(r.text)]
    if not stamp_runs:
        raise SystemExit(f"{meta['arxiv_id']}: no arXiv version stamp found on page 1")
    m = STAMP.search(stamp_runs[0].text)
    if (m.group(1), int(m.group(2))) != (meta["arxiv_id"], meta["version"]):
        raise SystemExit(f"{meta['arxiv_id']}: page-1 stamp says {m.group(0)!r}, "
                         f"manifest says v{meta['version']}")
    stats.hit(f"stamp_ok:{meta['key']}", m.group(0))

    sizes: collections.Counter = collections.Counter()
    for runs in raw_pages:
        for r in runs:
            if not r.rotated:
                sizes[r.size] += len(r.text)
    body = sizes.most_common(1)[0][0]

    pages_lines = []
    for runs in raw_pages:
        kept = []
        for r in runs:
            if r.rotated:
                if not STAMP.search(r.text):
                    stats.hit("rotated_run_dropped", r.text[:40])
            elif r.size > 2.6 * body:
                stats.hit("oversized_figure_run_dropped", r.text[:40])
            else:
                kept.append(r)
        pages_lines.append(build_lines(kept, body, stats))

    hf = find_header_footer(pages_lines, heights)

    # Pass 1: tag lines, drop headers/footers, segment into raw blocks.
    raw_blocks: list[dict] = []
    section, in_appendix, seen_refs, seen_abstract = "Front matter", False, False, False
    for pno, (lines, (bottom, top)) in enumerate(zip(pages_lines, heights), start=1):
        h = top - bottom
        body_x = [round(ln.x) for ln in lines if abs(ln.size - body) < 0.6]
        margin = collections.Counter(body_x).most_common(1)[0][0] if body_x else 72
        # Reference lists use hanging indents: most lines are indented continuations, so the
        # most common x is the indent, not the margin. Use the left edge instead.
        left_edge = sorted(body_x)[len(body_x) // 20] if body_x else 72
        cur = None
        prev = None
        for ln in lines:
            ln.page = pno
            rel = (ln.y - bottom) / h
            if (rel > 0.92 or rel < 0.08) and (norm_hf(ln.text) in hf or re.fullmatch(r"\d{1,3}", ln.text)):
                stats.hit("header_footer_removed", ln.text[:60])
                continue
            is_heading = heading_text(ln, body) and (seen_abstract or ln.text.lower() == "abstract"
                                                     or NUMBERED_HEADING.match(ln.text))
            if pno == 1 and not seen_abstract and not ln.text.lower() == "abstract":
                is_heading = False
            if is_heading:
                title = ln.text.strip()
                low = title.lower()
                if low == "abstract":
                    seen_abstract = True
                if low in ("references", "bibliography"):
                    seen_refs = True
                elif seen_refs:
                    in_appendix = True
                section = title
                raw_blocks.append({"page": pno, "section": section, "in_appendix": in_appendix,
                                   "kind": "heading", "lines": [ln]})
                cur, prev = None, ln
                continue

            in_refs = seen_refs and not in_appendix
            new_block = cur is None or prev is None
            if not new_block:
                gap = prev.y - ln.y
                if abs(ln.size - prev.size) > 0.6 or ln.figure_font != prev.figure_font:
                    new_block = True
                elif gap > 1.55 * max(ln.size, prev.size) or gap < -2:
                    new_block = True
                elif in_refs:
                    new_block = bool(re.match(r"\[\d+\]", ln.text)) or (
                        abs(ln.x - left_edge) <= 2 and prev.x - left_edge > 4)
                elif CAPTION.match(ln.text):
                    new_block = True
                elif abs(ln.size - body) < 0.6 and ln.x - margin > 6 and abs(prev.x - margin) <= 6:
                    new_block = True  # paragraph indent
            if new_block:
                cur = {"page": pno, "section": section, "in_appendix": in_appendix,
                       "kind": None, "lines": [], "in_refs": in_refs, "margin": margin,
                       "rel_y": rel}
                raw_blocks.append(cur)
            cur["lines"].append(ln)
            prev = ln

    # Pass 2: classify blocks.
    def numeric_profile(lines):
        n_num = n_tok = tab_lines = 0
        for ln in lines:
            toks = ln.text.split()
            nums = [t for t in toks if NUM_TOKEN.fullmatch(t)]
            n_num += len(nums)
            n_tok += len(toks)
            if len(nums) >= 2 and len(nums) >= 0.3 * len(toks):
                tab_lines += 1
        return n_num, n_tok, tab_lines

    for b in raw_blocks:
        if b["kind"] == "heading":
            continue
        first = b["lines"][0]
        n_num, n_tok, tab_lines = numeric_profile(b["lines"])
        if first.figure_font:
            b["kind"] = "figure_text"
        elif CAPTION.match(first.text):
            b["kind"] = "caption"
        elif b.get("in_refs"):
            b["kind"] = "reference"
        elif b["section"] == "Front matter":
            b["kind"] = "front_matter"  # title, authors, affiliations, contribution notes
        elif first.size < 0.93 * body and b["rel_y"] < 0.4 and re.match(
                r"^(\^?[\d∗†‡*]+|\d+\.?\s)", first.text):
            b["kind"] = "footnote"
        elif tab_lines >= max(2, 0.5 * len(b["lines"]))                 or (n_num >= 6 and n_num >= 0.3 * n_tok)                 or (len(b["lines"]) == 1 and n_num >= 4 and n_num >= 0.5 * n_tok):
            b["kind"] = "table"
        else:
            b["kind"] = "body"

    # Pass 3: table rules and row spacing split one table into several blocks. Pull adjacent
    # short, sentence-less blocks on the same page (header rows, single data rows) into the
    # table next to them.
    def table_fragment(b):
        if b["kind"] not in ("body", "table"):
            return False
        n_num, n_tok, _ = numeric_profile(b["lines"])
        text = " ".join(ln.text for ln in b["lines"])
        return b["kind"] == "table" or (len(b["lines"]) <= 3 and n_tok <= 40
                                         and not re.search(r"[a-z]{3,}[.;:]\s*$|[a-z]{3,}\. [A-Z]", text))
    changed = True
    while changed:
        changed = False
        for i in range(len(raw_blocks) - 1):
            a, c = raw_blocks[i], raw_blocks[i + 1]
            if a["page"] != c["page"] or a["section"] != c["section"]:
                continue
            if (a["kind"] == "table" and table_fragment(c)) or (c["kind"] == "table" and table_fragment(a)):
                if a["kind"] != c["kind"]:
                    stats.hit("table_fragment_merged", " ".join(ln.text for ln in (c if a["kind"] == "table" else a)["lines"])[:60])
                a["lines"].extend(c["lines"])
                a["kind"] = "table"
                del raw_blocks[i + 1]
                changed = True
                break

    return raw_blocks


def finalize(all_blocks: list[tuple[dict, dict]], stats: Stats) -> list[dict]:
    # Vocabulary from every line in every paper, after the context-free fixes.
    for meta, b in all_blocks:
        for ln in b["lines"]:
            ln.text = basic_fixes(ln.text, stats)
    vocab: collections.Counter = collections.Counter()
    for meta, b in all_blocks:
        for ln in b["lines"]:
            vocab.update(w.lower() for w in re.findall(r"[A-Za-z]+(?:-[A-Za-z]+)*", ln.text))
            vocab.update(w.lower() for w in re.findall(r"[A-Za-z]+", ln.text))
    # A word cut by a line-end hyphen shows up as two fragments; don't let them vouch for kerning.
    out = []
    counters: collections.Counter = collections.Counter()
    for meta, b in all_blocks:
        keep_nl = b["kind"] in ("table",)
        text = join_lines([ln.text for ln in b["lines"]], vocab, stats, keep_nl)
        # Accents again on the joined text: a math hat drawn above its variable can sit on its
        # own baseline and only meets the variable once lines are joined ("z" + "ˆ^S").
        text = vocab_fixes(fix_accents(text, stats), vocab, stats)
        text = text.strip()
        if not text:
            continue
        key = meta["key"]
        counters[(key, b["page"])] += 1
        out.append({
            "block_id": f"{key}-p{b['page']:02d}-b{counters[(key, b['page'])]:03d}",
            "paper": key,
            "arxiv_id": f"{meta['arxiv_id']}v{meta['version']}",
            "page": b["page"],
            "section": vocab_fixes(basic_fixes(b["section"], stats), vocab, stats),
            "text": text,
            "kind": b["kind"],
            "in_appendix": b["in_appendix"],
        })
    return out


def main() -> int:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    stats = Stats()
    all_blocks = []
    for meta in manifest["papers"]:
        for b in extract_paper(meta, stats):
            all_blocks.append((meta, b))
    blocks = finalize(all_blocks, stats)
    with OUT.open("w", encoding="utf-8", newline="\n") as f:
        for b in blocks:
            f.write(json.dumps(b, ensure_ascii=False) + "\n")
    STATS.write_text(json.dumps({
        "generated_by": "scripts/extract_papers.py",
        "counts": dict(sorted(stats.c.items())),
        "examples": {k: v for k, v in sorted(stats.examples.items())},
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    by = collections.Counter((b["paper"], b["kind"]) for b in blocks)
    print(f"wrote {len(blocks)} blocks to {OUT.relative_to(ROOT).as_posix()}")
    for (p, k), n in sorted(by.items()):
        print(f"  {p:17s} {k:12s} {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
