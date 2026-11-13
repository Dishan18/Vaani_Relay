"""Metadata-carrying answers: the drafter returns claims, the system renders the citations.

The v1 drafter wrote prose with "[S1]" markers, and a regex gate parsed them back out. Every real
failure of that gate was a parsing failure, not a grounding one: 【S3】 brackets, a zero-width space
inside "[S1]", display formulas split into uncited lines, "{0,1}" read as one number. The question
from the interview was the right one: why parse citations out of text at all?

Now the drafter answers in JSON Lines, one claim per line, and never writes citation markup:

    {"text": "The base model uses a dropout rate of 0.1.", "sources": [1], "quote": "For the base model, we use a rate of P_drop = 0.1"}
    {"not_in_papers": "the papers report no p95 latency"}            (the whole answer, when unanswerable)

The citation is metadata (source ids + the exact supporting words), carried from retrieval to the
reply. The system checks each claim with plain string operations and renders the citation itself:
  * every source id is one that was given;
  * the quote occurs in a cited source (compared as word sequences: case, spacing, punctuation and
    hyphenation are ignored), so the claim is pinned to real text, not to a model's say-so;
  * every number in the claim occurs in a cited source.
A claim that fails is not released (it is dropped and reported). What carrying metadata does NOT
remove: the model can still attach the wrong source, or a quote that is real but doesn't support
the sentence. The quote makes the first case checkable for free; the second is what the optional
LLM verifier (relay.pipeline._verify) is for.

Because each line is a complete claim, claims are validated and released one by one as they stream
in (``ClaimStream.feed``): time to first token becomes time to the first verified claim, not to
the whole answer.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from relay.retrieve import Hit

CLAIMS_SYSTEM = (
    "Answer the question using only the numbered sources below.\n"
    "Output JSON Lines: one JSON object per line and nothing else (no prose, no code fences).\n"
    "Each line is one sentence of the answer:\n"
    '{"text": "<one sentence>", "sources": [<numbers of the sources that support it>], '
    '"quote": "<5 to 30 words copied exactly from one of those sources that support the sentence>"}\n'
    "Rules:\n"
    "1. Copy numbers exactly as the sources write them. Never state a number that is not in a cited source.\n"
    "2. Each source names its paper. Only use a source for claims about that paper.\n"
    "3. Do not write citation markers like [S1] in the text: the sources field carries them.\n"
    "4. At most 6 lines. Put the most direct answer first.\n"
    "5. If the sources do not answer the question, output exactly one line: "
    '{"not_in_papers": "<one short sentence saying what is missing>"}. Do not guess.')

_WORD = re.compile(r"[a-z0-9]+")
_NUM = re.compile(r"(?<![A-Za-z_])\d+(?:\.\d+)*")
MIN_QUOTE_WORDS = 4
QUOTE_MATCH_WORDS = 8     # a quote counts if it shares this many consecutive words with a cited source


def words(s: str) -> list[str]:
    """Lower-cased alphanumeric runs, with line-break hyphenation and thousands separators removed."""
    s = re.sub(r"(?<=\w)-\s+(?=\w)", "", s.lower())
    s = re.sub(r"(?<=\d),(?=\d{3}\b)", "", s)
    return _WORD.findall(s)


def numbers(s: str) -> list[str]:
    """Whole numbers as written ("0.1", "512", "3.5"), thousands separators removed ("36,000" -> "36000")."""
    return _NUM.findall(re.sub(r"(?<=\d),(?=\d{3}\b)", "", s))


def longest_run(hay: list[str], needle: list[str]) -> int:
    """Longest run of consecutive words of ``needle`` that also occurs consecutively in ``hay``."""
    if not hay or not needle:
        return 0
    best, prev = 0, [0] * (len(hay) + 1)
    for w in needle:
        cur = [0] * (len(hay) + 1)
        for j, h in enumerate(hay, 1):
            if h == w:
                cur[j] = prev[j - 1] + 1
                if cur[j] > best:
                    best = cur[j]
        prev = cur
    return best


def squash(s: str) -> str:
    """Letters and digits only, no spaces: "where F represents" and the PDF's "whereF represents" agree."""
    return "".join(words(s))


def quote_found(quote: str, source: str) -> bool:
    """Is the quote real text of this source? Checked three ways, all string operations:
      1. as letters and digits with all spacing removed: the PDF extraction glues words ("whereF",
         "andN") or splits them, which made correct quotes fail a word-by-word match (Q15, real run 2);
      2. a quote that joins passages with "..." counts if every part of 3+ words is in the source;
      3. a trimmed or slightly extended quote counts if 8+ consecutive words, covering most of it, are."""
    src = squash(source)
    parts = [x for x in re.split(r"\.\.\.|…", quote) if len(words(x)) >= 3] or [quote]
    if all(len(squash(x)) >= 12 and squash(x) in src for x in parts):
        return True
    q, sw = words(quote), words(source)
    run = longest_run(sw, q)
    if run >= QUOTE_MATCH_WORDS and run >= 0.6 * len(q):
        return True
    return in_order_nearby(sw, q)


def in_order_nearby(hay: list[str], needle: list[str], spread: float = 2.5) -> bool:
    """4. Table cells: a quote like "Self-Attention O(1) ... Recurrent O(n)" skips the cells between, so
    no run matches. It counts if all its words (5+) occur in order within a window of at most
    ``spread`` x its length (Q4 on gpt-oss-20b). Still the cited source's own words, in its own order."""
    n = len(needle)
    if n < 5:
        return False
    limit = int(spread * n)
    for i, w in enumerate(hay):
        if w != needle[0]:
            continue
        j, k = i + 1, 1
        while k < n and j < len(hay) and j - i < limit:
            if hay[j] == needle[k]:
                k += 1
            j += 1
        if k == n:
            return True
    return False


def contains_seq(hay: list[str], needle: list[str]) -> bool:
    if not needle or len(needle) > len(hay):
        return False
    first, n = needle[0], len(needle)
    for i, w in enumerate(hay):
        if w == first and hay[i:i + n] == needle:
            return True
    return False


@dataclass
class Claim:
    text: str
    sources: list[int]
    quote: str
    ok: bool = True
    reason: str = ""

    def to_dict(self, hits: list[Hit] | None = None) -> dict:
        d = {"text": self.text, "sources": [f"S{i}" for i in self.sources], "quote": self.quote}
        if hits:
            d["cites"] = [hits[i - 1].piece.cite() for i in self.sources if 1 <= i <= len(hits)]
        if not self.ok:
            d["dropped"] = self.reason
        return d


def check_claim(obj: dict, hits: list[Hit]) -> Claim:
    text = str(obj.get("text") or "").strip()
    raw = obj.get("sources") or []
    if isinstance(raw, (int, str)):
        raw = [raw]
    ids = []
    for x in raw:
        m = re.search(r"\d+", str(x))
        if m:
            ids.append(int(m.group(0)))
    quote = str(obj.get("quote") or "").strip().strip('"“”')
    c = Claim(re.sub(r"\s*\[\s*S?\d+\s*\]", "", text).strip(), sorted(set(ids)), quote)
    if len(_WORD.findall(c.text.lower())) < 2:
        c.ok, c.reason = False, "empty claim"
    elif not ids:
        c.ok, c.reason = False, "no source"
    elif any(not 1 <= i <= len(hits) for i in ids):
        c.ok, c.reason = False, f"cites a source that wasn't given: {ids}"
    else:
        cited = [hits[i - 1].piece for i in c.sources]
        if len(words(quote)) < MIN_QUOTE_WORDS:
            c.ok, c.reason = False, "supporting quote missing or too short"
        elif not any(quote_found(quote, f"{p.section} {p.text}") for p in cited):
            c.ok, c.reason = False, "supporting quote not found in the cited sources"
        else:
            pool = set(numbers(" ".join(f"{p.section} {p.cite()} {p.text}" for p in cited)))
            for n in numbers(c.text):
                if n not in pool:
                    c.ok, c.reason = False, f"number {n} is not in the cited sources"
                    break
    return c


def _object_end(buf: str, start: int) -> int | None:
    """Index of the brace closing the JSON object that opens at ``start`` (strings and escapes respected)."""
    depth, in_str, esc = 0, False, False
    for i in range(start, len(buf)):
        ch = buf[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i
    return None


@dataclass
class ClaimStream:
    """Feed streamed content; get back each line as soon as it is complete and checked."""
    hits: list[Hit]
    buf: str = ""
    claims: list[Claim] = field(default_factory=list)
    refusal: str | None = None
    junk: list[str] = field(default_factory=list)

    def feed(self, text: str, final: bool = False) -> list[Claim]:
        """Each complete JSON object is checked as soon as its closing brace arrives, whether or not the
        model put a newline after it (gpt-oss-20b writes "}{" on one line)."""
        self.buf += text
        out = []
        while True:
            start = self.buf.find("{")
            if start < 0:
                if final or "\n" in self.buf:
                    self._prose(self.buf if final else self.buf.rsplit("\n", 1)[0])
                    self.buf = "" if final else self.buf.rsplit("\n", 1)[1]
                break
            end = _object_end(self.buf, start)
            if end is None:                               # the object is still streaming in
                if final:
                    self._prose(self.buf)
                    self.buf = ""
                break
            self._prose(self.buf[:start])
            obj_text, self.buf = self.buf[start:end + 1], self.buf[end + 1:]
            c = self._object(obj_text)
            if c is not None:
                out.append(c)
        return out

    def _prose(self, text: str) -> None:
        for ln in text.splitlines():
            s = ln.strip().strip(",")
            if not s or s.startswith("```") or self.refusal is not None:
                continue
            if s.upper().startswith("NOT_IN_PAPERS"):
                if not self.claims:
                    self.refusal = s.split(":", 1)[1].strip() if ":" in s else ""
            else:
                self.junk.append(s[:200])

    def _object(self, obj_text: str) -> Claim | None:
        if self.refusal is not None:
            return None
        try:
            obj = json.loads(obj_text)
        except json.JSONDecodeError:
            self.junk.append(obj_text[:200])
            return None
        if not isinstance(obj, dict):
            return None
        if "not_in_papers" in obj:
            if not self.claims:
                self.refusal = str(obj.get("not_in_papers") or "").strip()
            return None
        c = check_claim(obj, self.hits)
        self.claims.append(c)
        return c

    @property
    def good(self) -> list[Claim]:
        return [c for c in self.claims if c.ok]

    @property
    def dropped(self) -> list[Claim]:
        return [c for c in self.claims if not c.ok]


def render(claims: list[Claim]) -> str:
    """The reply text. Citation markers are produced here, from metadata, never by the model."""
    return " ".join(f"{c.text.rstrip()} " + "".join(f"[S{i}]" for i in c.sources) for c in claims).strip()


def claims_messages(question: str, hits: list[Hit], render_sources, feedback: str | None = None) -> list[dict]:
    user = f"Question: {question}\n\nSOURCES:\n{render_sources(hits)}"
    if feedback:
        user += f"\n\n{feedback}\nWrite the answer again, following the rules."
    return [{"role": "system", "content": CLAIMS_SYSTEM}, {"role": "user", "content": user}]
