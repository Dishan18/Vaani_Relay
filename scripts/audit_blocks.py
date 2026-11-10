"""Audit data/blocks.jsonl for PDF-extraction failure classes -> data/blocks_audit.json.

For each class it reports:
  fixed_in_extractor  counters the extractor recorded while fixing it
                      (from data/blocks_extract_stats.json)
  remaining           occurrences still detectable in blocks.jsonl, by paper, with examples
  status              "fixed"                  nothing remaining
                      "open_extractor_bug"     remaining, and it is a bug the extractor
                                               should fix (the script exits 1)
                      "future_retrieval_bug"   remaining, and it cannot be fixed from the
                                               PDF text layer; the retriever must cope
  note                why, and what retrieval should do about it

The detectors are deliberately simple regexes so a reviewer can rerun and check them.

Usage (PowerShell):  .venv\\Scripts\\python scripts\\audit_blocks.py
"""
from __future__ import annotations

import collections
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BLOCKS = ROOT / "data" / "blocks.jsonl"
STATS = ROOT / "data" / "blocks_extract_stats.json"
OUT = ROOT / "data" / "blocks_audit.json"
MAX_EXAMPLES = 6


def load():
    blocks = [json.loads(l) for l in BLOCKS.read_text(encoding="utf-8").splitlines() if l.strip()]
    stats = json.loads(STATS.read_text(encoding="utf-8"))["counts"]
    return blocks, stats


class Finding:
    def __init__(self):
        self.by_paper = collections.Counter()
        self.examples = []

    def add(self, b, snippet, n=1):
        self.by_paper[b["paper"]] += n
        if len(self.examples) < MAX_EXAMPLES:
            self.examples.append({"block_id": b["block_id"], "snippet": snippet[:160]})

    @property
    def total(self):
        return sum(self.by_paper.values())


def snippet_around(text, m, pad=50):
    return text[max(0, m.start() - pad): m.end() + pad].replace("\n", " / ")


def scan(blocks, kinds, pattern, flags=0, filt=None):
    f = Finding()
    rx = re.compile(pattern, flags)
    for b in blocks:
        if kinds and b["kind"] not in kinds:
            continue
        for m in rx.finditer(b["text"]):
            if filt is None or filt(m, b):
                f.add(b, snippet_around(b["text"], m))
    return f


def main() -> int:
    blocks, xstats = load()
    vocab = collections.Counter(w.lower() for b in blocks for w in re.findall(r"[A-Za-z]+", b["text"]))
    prose = {"body", "caption", "footnote", "heading", "front_matter"}
    classes = {}

    def record(name, fixed_keys, findings: dict[str, tuple[Finding, str]], note):
        """findings: label -> (Finding, kind) where kind is 'extractor' or 'retrieval'."""
        fixed = {k: xstats.get(k, 0) for k in fixed_keys}
        remaining = {}
        open_bug = False
        future = False
        for label, (f, kind) in findings.items():
            remaining[label] = {"count": f.total, "by_paper": dict(f.by_paper),
                                "type": "extractor_bug" if kind == "extractor" else "retrieval_bug",
                                "examples": f.examples}
            if f.total and kind == "extractor":
                open_bug = True
            if f.total and kind == "retrieval":
                future = True
        status = "open_extractor_bug" if open_bug else ("future_retrieval_bug" if future else "fixed")
        classes[name] = {"status": status, "fixed_in_extractor": fixed, "remaining": remaining,
                         "note": note}

    # ---------------------------------------------------------------- tables
    ragged = Finding()
    leaked = Finding()
    num_tok = re.compile(r"^[-+]?\d[\d.,]*(?:%|x|×|\^\S+)?$")
    for b in blocks:
        if b["kind"] == "table":
            rows = [r for r in b["text"].split("\n") if r.strip()]
            counts = [sum(1 for t in r.split() if num_tok.match(t)) for r in rows]
            data = [c for c in counts if c > 0]
            if len(data) >= 3:
                mode = collections.Counter(data).most_common(1)[0][0]
                for r, c in zip(rows, counts):
                    if 0 < c != mode:
                        ragged.add(b, r)
        elif b["kind"] in ("body",):
            toks = b["text"].split()
            nums = [t for t in toks if num_tok.match(t.strip("(),;"))]
            if len(nums) >= 8 and len(nums) >= 0.35 * len(toks):
                leaked.add(b, b["text"][:160])
    record("tables",
           ["table_fragment_merged"],
           {"rows_with_missing_cells_column_alignment_lost": (ragged, "retrieval"),
            "table_rows_classified_as_body": (leaked, "extractor")},
           "Rows keep their line structure and exponents (10^19), but the PDF text layer joins "
           "cells with single spaces, so a row with an empty cell ('ByteNet [18] 23.75') no "
           "longer says which column the number belongs to. Not recoverable without glyph-level "
           "column analysis. Retrieval: keep each table in one chunk together with its caption "
           "and header rows, and have the answer step refuse to read a value off a ragged row.")

    # ---------------------------------------------------------------- sub/superscripts
    flat_exp = scan(blocks, None, r"\d(?:\.\d+)?\s?[·×]\s?10(\d{1,2})(?![\d^])")
    glued = Finding()
    pairs = set(re.findall(r"\b([A-Za-z]{1,3})_([A-Za-z]{2,})\b", " ".join(b["text"] for b in blocks)))
    for b in blocks:
        if b["kind"] in ("reference",):
            continue
        for base, sub in pairs:
            for m in re.finditer(rf"\b{re.escape(base + sub)}\b", b["text"]):
                if vocab[(base + sub).lower()] and (base + sub).lower() not in ("dk",) or True:
                    glued.add(b, snippet_around(b["text"], m))
    math_alnum = scan(blocks, None, "[\U0001D400-\U0001D7FF]")
    record("sub_superscripts",
           ["supsub_marked", "math_alnum_normalized"],
           {"flattened_exponents": (flat_exp, "extractor"),
            "subscript_flattened_elsewhere": (glued, "extractor"),
            "math_alphanumeric_unicode": (math_alnum, "extractor")},
           "Smaller runs above/below the baseline are written ^x / _x (10^19, d_model, P_drop). "
           "Display equations still come out as linear text with ^/_ markers; that is readable "
           "for an LLM but BM25 will not match 'dmodel' against 'd_model' unless the tokenizer "
           "splits on '_' and '^' (it should).")

    # ---------------------------------------------------------------- captions
    inline_caption = scan(blocks, {"body"}, r"(?:^|(?<=\. ))(?:Figure|Table)\s+\d+:\s")
    orphan_tables = Finding()
    by_page = collections.defaultdict(list)
    for b in blocks:
        by_page[(b["paper"], b["page"])].append(b)
    for (paper, page), bs in by_page.items():
        has_table_caption = any(b["kind"] == "caption" and b["text"].startswith("Table") for b in bs)
        for b in bs:
            if b["kind"] == "table" and not has_table_caption:
                orphan_tables.add(b, b["text"][:120])
    fig_drop = {k: v for k, v in xstats.items() if k.startswith("figure_form_chars_dropped")}
    record("captions",
           list(fig_drop),
           {"caption_text_inside_body_block": (inline_caption, "extractor"),
            "table_without_caption_on_same_page": (orphan_tables, "retrieval")},
           "Captions are their own blocks (kind=caption). Text drawn inside embedded figures is "
           "dropped on purpose (counted in fixed_in_extractor): it had figure-local coordinates "
           "and private glyph encodings ('4UBHF' for 'Stage'). Consequence: anything only stated "
           "inside a figure is not retrievable, so questions about figure contents must be "
           "refused unless the caption or body says it. Chunking must attach captions to the "
           "table/figure they describe.")

    # ---------------------------------------------------------------- footnotes
    fn_in_body = scan(blocks, {"body"}, r"(?:^|\s)\d{1,2}\.?\s?https?://|∗Equal contribution|Work performed while")
    markers = scan(blocks, {"body"}, r"[A-Za-z\)]\^(?:\d{1,2}|∗|†|‡)(?=[\s.,;:]|$)")
    record("footnotes",
           [],
           {"footnote_text_inside_body_block": (fn_in_body, "extractor"),
            "footnote_marker_in_body_text": (markers, "retrieval")},
           "Footnotes are separate blocks (kind=footnote) with the page's markers. The marker in "
           "the sentence ('Alexa^1') stays, so the sentence and its footnote (often a URL) end up "
           "in different chunks. Retrieval: low priority; markers are harmless tokens.")

    # ---------------------------------------------------------------- references
    # Venue/locator phrases that occur in bibliography entries but not in running text.
    # (Author-year citations like "(Zeghidour et al., 2022)" are normal prose, so "et al." and
    # years alone are not evidence.)
    bib_rx = (r"(?:arXiv preprint arXiv:\d|In Proceedings of|URL https?://|abs/\d{4}\.\d{4,5}"
              r"|Advances in Neural Information Processing Systems|pages \d+[-–]\d+|doi: 10\.)")
    bib_in_body = Finding()
    for b in blocks:
        if b["kind"] in ("body", "footnote") and len(re.findall(bib_rx, b["text"])) >= 2:
            bib_in_body.add(b, b["text"][:160])
    ref_blocks = Finding()
    for b in blocks:
        if b["kind"] == "reference":
            ref_blocks.add(b, b["text"][:100])
    cite_markers = scan(blocks, {"body"}, r"\[\d+(?:,\s?\d+)*\]")
    record("references",
           [],
           {"bibliography_text_classified_as_body": (bib_in_body, "extractor"),
            "reference_blocks_that_can_match_queries": (ref_blocks, "retrieval"),
            "numeric_citation_markers_in_body": (cite_markers, "retrieval")},
           "Every bibliography entry is its own block (kind=reference). Their titles share words "
           "with real questions ('attention', 'chunking', 'speech'), so a naive retriever returns "
           "them as evidence. Retrieval: exclude kind=reference by default. Numeric markers like "
           "[13] carry no meaning without the bibliography.")

    # ---------------------------------------------------------------- kerning / spacing
    kern = scan(blocks, prose, r"\b([TVWYPFAL]) ([a-z]{2,})\b",
                filt=lambda m, b: vocab[(m.group(1) + m.group(2)).lower()] >= 2)
    camel_ok = re.compile(r"^(?:[A-Z][a-z]+){2,}$")

    def glued_word(m, b):
        w = m.group(0)
        left, right = m.group(1), m.group(2)
        return vocab[w.lower()] <= 1 and vocab[left.lower()] >= 3 and vocab[right.lower()] >= 3 \
            and not camel_ok.match(w)
    glue = scan(blocks, prose, r"\b([a-z]{3,})([A-Z][a-z]{2,})\b", filt=glued_word)
    letterspaced = scan(blocks, None, r"(?:\b[A-Za-z] ){5,}[A-Za-z]\b")
    period = scan(blocks, prose, r"\b[a-z]{3,}\.[A-Z][a-z]{2,}\b",
                  filt=lambda m, b: not re.search(r"https?://|www\.|\.(?:com|org|io|ai)", b["text"][max(0, m.start()-40):m.end()+5]))
    record("kerning_splits",
           ["kerning_split_fixed", "digit_word_merge_fixed", "gap_space_inserted",
            "font_change_space_inserted", "bracket_space_fixed"],
           {"split_capital_kerning": (kern, "extractor"),
            "glued_words": (glue, "extractor"),
            "missing_space_after_period": (period, "extractor"),
            "letter_spaced_text": (letterspaced, "extractor")},
           "'T ext' -> 'Text' and '6identical' -> '6 identical' are joined/split only when the "
           "papers' own vocabulary confirms the result; side-by-side runs get a space when the "
           "estimated gap is large.")

    # ---------------------------------------------------------------- other classes found while auditing
    accent_chars = "[´¨ˆ˜ˇ˚¸˝]"
    accents_left = scan(blocks, None, rf"[A-Za-z] ?{accent_chars}|{accent_chars} ?[A-Za-z]")
    # An accent character touching no letter is not an accent: it is a math glyph the PDF font
    # maps to the wrong code point (the summation sign in dynamic_chunking comes out as U+02DD).
    misencoded = scan(blocks, None, rf"(?<![A-Za-z ]) ?{accent_chars}(?! ?[A-Za-z])")
    record("accents_and_ligatures", ["accent_fixed", "math_accent_fixed", "ligature_fixed"],
           {"spacing_accent_left": (accents_left, "extractor")},
           "TeX accents arrive as a spacing accent next to the letter ('D´ efossez'); "
           "they are recomposed to 'Défossez'.")

    unmapped = scan(blocks, None, "[�-]")
    record("unmapped_glyphs", ["big_operator_glyph_mapped"],
           {"replacement_or_private_use_chars": (unmapped, "retrieval"),
            "math_glyph_mapped_to_accent_char": (misencoded, "retrieval")},
           "Glyphs with no Unicode mapping in the PDF font (e.g. arrows in 'tokenization�LM�"
           "detokenization'). The character is unknowable from the text layer. Harmless for "
           "retrieval of surrounding words; an answer quoting the span must not invent the symbol.")

    # A broken line-end hyphen leaves "transduc- tion". Suspended hyphens in the source
    # ("content- and context- dependent") are correct, so only flag pairs whose joined form is
    # a word the papers use.
    hyph = scan(blocks, prose, r"\b([a-z]{2,})- ([a-z]{2,})\b",
                filt=lambda m, b: vocab[(m.group(1) + m.group(2)).lower()] >= 2)
    record("hyphenation", ["hyphenation_joined", "hyphen_kept"],
           {"broken_hyphenation_left": (hyph, "extractor")},
           "A line-end hyphen is removed only when the joined word occurs in the papers more often "
           "than the hyphenated form ('transduc-|tion' -> 'transduction', 'sequence-|aligned' kept).")

    hf_left = Finding()
    titles = {"attention": "Attention Is All You Need",
              "moshi": "Moshi: a speech-text foundation model for real-time dialogue",
              "dynamic_chunking": "Dynamic Chunking for End-to-End Hierarchical Sequence Modeling"}
    for b in blocks:
        t = titles[b["paper"]]
        if b["page"] > 1 and t in b["text"] and b["kind"] != "reference":
            hf_left.add(b, t)
    page_num_tail = scan(blocks, {"body"}, r"\n\d{1,2}$")
    record("headers_footers", ["header_footer_removed"],
           {"running_title_left_in_text": (hf_left, "extractor"),
            "page_number_left_in_text": (page_num_tail, "extractor")},
           "Lines in the top/bottom 8% of the page that repeat on at least 20% of pages (digits "
           "normalised), and bare page numbers there, are removed.")

    garble = Finding()
    shift = str.maketrans({chr(c): chr(c - 1) for c in list(range(66, 91)) + list(range(98, 123))})
    for b in blocks:
        for w in re.findall(r"\b[A-Za-z]{5,}\b", b["text"]):
            if vocab[w.lower()] <= 1 and vocab[w.translate(shift).lower()] >= 3:
                garble.add(b, f"{w} -> {w.translate(shift)}")
    record("figure_text", [],
           {"caesar_shifted_figure_glyphs": (garble, "extractor")},
           "Figure fonts with a private encoding produce text shifted by one letter ('4UBHF' = "
           "'Stage'). Skipping embedded figures removes it; this detector checks none leaked.")

    split_para = Finding()
    for a, c in zip(blocks, blocks[1:]):
        if a["paper"] == c["paper"] and c["page"] == a["page"] + 1 and a["kind"] == c["kind"] == "body" \
                and a["section"] == c["section"] and not re.search(r"[.:!?)\]]\s*$", a["text"]) \
                and re.match(r"[a-z(]", c["text"]):
            split_para.add(c, f"...{a['text'][-60:]} || {c['text'][:60]}...")
    record("page_breaks", [],
           {"paragraph_split_across_pages": (split_para, "retrieval")},
           "Blocks never span pages, so that every block has one page to cite. A paragraph that "
           "runs onto the next page becomes two blocks, each half a sentence at the seam. The "
           "chunker should join consecutive body blocks of the same section across a page break "
           "and cite both pages.")

    # ---------------------------------------------------------------- summary
    kinds = collections.Counter((b["paper"], b["kind"]) for b in blocks)
    by_paper = collections.defaultdict(dict)
    for (p, k), n in sorted(kinds.items()):
        by_paper[p][k] = n
    stamps = {k.split(":", 1)[1]: v for k, v in xstats.items() if k.startswith("stamp_ok:")}
    out = {
        "generated_by": "scripts/audit_blocks.py",
        "inputs": {"blocks.jsonl_sha256": hashlib.sha256(BLOCKS.read_bytes()).hexdigest(),
                   "extract_stats": STATS.relative_to(ROOT).as_posix()},
        "blocks_total": len(blocks),
        "blocks_by_paper_and_kind": by_paper,
        "version_stamp_checked": stamps,
        "status_summary": {k: v["status"] for k, v in classes.items()},
        "future_retrieval_bugs": [
            {"class": name, "issue": label, "count": r["count"], "note": c["note"]}
            for name, c in classes.items() for label, r in c["remaining"].items()
            if r["type"] == "retrieval_bug" and r["count"]
        ],
        "classes": classes,
    }
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"{len(blocks)} blocks audited -> {OUT.relative_to(ROOT).as_posix()}")
    for name, c in classes.items():
        rem = ", ".join(f"{k}={v['count']}" for k, v in c["remaining"].items())
        fx = ", ".join(f"{k}={v}" for k, v in c["fixed_in_extractor"].items())
        print(f"  {name:22s} {c['status']:22s} fixed[{fx}] remaining[{rem}]")
    return 1 if any(c["status"] == "open_extractor_bug" for c in classes.values()) else 0


if __name__ == "__main__":
    sys.exit(main())
