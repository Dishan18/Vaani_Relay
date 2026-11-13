"""The cited-answer pipeline: understand -> retrieve -> decide -> draft -> check -> send.

    pipe = Pipeline(chat, Limiter.from_probe(), Retriever.from_blocks(2400), load_config())
    reply = await pipe.answer("Why divide by sqrt(d_k)?")
    reply.text, reply.sources, reply.level, reply.spans

Steps
  understand  a small, fast model (config [models].understand) picks the papers and rewrites the
              question into a search query in the papers' own vocabulary, plus one alternative
              phrasing. Skipped (heuristic: all papers, raw question) when the router isn't free
              within router_free_within_s, at the lean level, or when its reply doesn't parse.
  retrieve    BM25 as a tool, ``search_papers(query, k, papers)``. One more search (alternative
              phrasing or the raw question, all papers) when the best piece covers too little of
              the question; the two rankings are merged by reciprocal rank.
  decide      refuse with NO model call when retrieval found almost nothing, or when the router
              said the question is about none of the papers (and it names none of them).
  draft       every sentence must cite sources as [S#]; the model replies NOT_IN_PAPERS: ... when
              the sources don't answer the question.
  check       a deterministic citation gate, always: each sentence cites valid sources, every
              number in it appears in a cited source, and every quotation is in a cited source.
              Then an LLM verifier, only at the full level and only if its model has headroom.
              A failed check gets one redraft; a second failure falls back to cited passages.
  send        the reply is buffered and released only after the checks. Nothing unchecked is
              streamed to the person, so time to first token = time to release (PLAN.md G2).

Models. Every limit is per model (data/probe/findings.md Q3), so steps run on different models and
don't compete for the same buckets. Each step has a fallback order; a 429, timeout or error on one
model moves to the next.

Degrade ladder (rule 2: never make a person wait for budget that isn't coming), decided from the
limiter's predicted wait for the draft call before any model is called:
  0 full          preferred drafter free within preferred_free_within_s
  1 full          another drafter free within that time
  2 lean          drafters backed up: fewer pieces, no router, no verifier; waits up to max_wait_s
  3 passages      nothing free within max_wait_s: the cited passages, no model call, and an ETA
                  from the limiter's refill-rate prediction
  4 busy          only on an internal failure: "busy, retry in N s"
The reply says which level it got and why (``notice``).

Every step and every budget wait is a timing span (``Reply.spans``). Tokens per answer are summed
over every model call in the request (``Reply.tokens_in`` / ``tokens_out``).
"""
from __future__ import annotations

import asyncio
import inspect
import json
import re
import time
import tomllib
import uuid
from contextlib import contextmanager
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable

from relay.claims import Claim, ClaimStream, claims_messages, render
from relay.limiter import TOKENS, Grant, Limiter, Ticket, Unavailable
from relay.llm import CallRecord, Timeouts
from relay.retrieve import PAPER_TITLES, PAPERS, Hit, Retriever, tokenize

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "pipeline.toml"

FULL, FULL_OTHER, LEAN, PASSAGES, BUSY = 0, 1, 2, 3, 4
LEVEL_NAMES = {FULL: "full", FULL_OTHER: "full (other model)", LEAN: "lean", PASSAGES: "passages", BUSY: "busy"}

# chat(payload, phase=..., timeouts=...) -> CallRecord. relay.ledger.BudgetedClient.stream_chat for
# real calls (ledger + caps), or ``plain_chat(LLMClient)`` for the stand-in.
ChatFn = Callable[..., Awaitable[CallRecord]]


def plain_chat(client) -> ChatFn:
    async def chat(payload, *, phase, timeouts=None, on_delta=None):
        return await client.stream_chat(payload, timeouts=timeouts, on_delta=on_delta)
    return chat


PRODUCTION_CONFIG = ROOT / "config" / "production.toml"


def load_production_config(path: Path = PRODUCTION_CONFIG) -> dict:
    """config/pipeline.toml (as submitted and measured) with config/production.toml's tables merged
    over it, key by key."""
    base = load_config()
    with path.open("rb") as f:
        over = tomllib.load(f)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k] = {**base[k], **v}
        else:
            base[k] = v
    return base


def load_config(path: Path = CONFIG) -> dict:
    with path.open("rb") as f:
        return tomllib.load(f)


# ----------------------------------------------------------------------------- trace
@dataclass
class Span:
    name: str
    start_ms: float
    end_ms: float | None = None
    model: str | None = None
    outcome: str = ""
    detail: dict = field(default_factory=dict)


class Trace:
    def __init__(self) -> None:
        self.t0 = time.perf_counter()
        self.spans: list[Span] = []
        self.meta: dict = {}          # what the router and retrieval chose, copied into Reply.checks

    def ms(self) -> float:
        return (time.perf_counter() - self.t0) * 1000

    @contextmanager
    def span(self, name: str, model: str | None = None, **detail):
        s = Span(name, round(self.ms(), 1), model=model, detail=dict(detail))
        self.spans.append(s)
        try:
            yield s
        except BaseException as e:
            s.outcome = s.outcome or f"error:{type(e).__name__}"
            raise
        finally:
            s.end_ms = round(self.ms(), 1)


# ----------------------------------------------------------------------------- reply
@dataclass
class ModelCall:
    step: str
    model: str
    status: int | None
    outcome: str
    prompt_tokens: int
    completion_tokens: int
    tokens_estimated: bool
    ms: float | None
    waited_s: float


@dataclass
class Reply:
    request_id: str
    question: str
    kind: str                      # answer | refusal | passages | busy
    level: int
    text: str
    sources: list[dict]
    notice: str = ""
    eta_s: float | None = None
    model_calls: list[ModelCall] = field(default_factory=list)
    spans: list[Span] = field(default_factory=list)
    checks: dict = field(default_factory=dict)
    released_ms: float | None = None   # arrival -> release (= first token: the reply is buffered)
    budget_wait_ms: float = 0.0
    claims: list[dict] = field(default_factory=list)   # claims mode: each sentence with its sources and quote
    first_token_ms: float | None = None  # claims mode, streaming: arrival -> first verified claim released

    @property
    def tokens_in(self) -> int:
        return sum(c.prompt_tokens for c in self.model_calls)

    @property
    def tokens_out(self) -> int:
        return sum(c.completion_tokens for c in self.model_calls)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["tokens_in"], d["tokens_out"] = self.tokens_in, self.tokens_out
        d["level_name"] = LEVEL_NAMES[self.level]
        return d


# ----------------------------------------------------------------------------- deterministic gate
_CITE_ANY = re.compile(r"\[\s*S?\s*\d+(?:\s*[,;]\s*S?\s*\d+)*\s*\]")
# A number continues only through a decimal point. Thousands separators are removed first (_digits_norm),
# so a comma left between digits separates two numbers: "{0,1}" is 0 and 1, not "0,1" (found by the gate
# replay: Q14's b_t in {0,1} failed the check). LEGACY_NUM is the parse as measured at e7aa65f (replay only).
_NUM = re.compile(r"(?<![A-Za-z_])\d+(?:\.\d+)*")
LEGACY_NUM = re.compile(r"(?<![A-Za-z_])\d+(?:[.,]\d+)*")
_QUOTE = re.compile(r"[\"“]([^\"”]{12,})[\"”]")
_LIST_MARK = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")
_WS = re.compile(r"\s+")


def _norm(s: str) -> str:
    return _WS.sub(" ", s).strip().lower()


def _digits_norm(s: str) -> str:
    return re.sub(r"(?<=\d),(?=\d{3}\b)", "", _norm(s))


def _grounding(h: Hit) -> str:
    """What a sentence citing this source may take numbers from: its text, section and citation."""
    return _digits_norm(f"{h.piece.section} {h.piece.cite()} {h.piece.text}")


def numbers_grounded(text: str, sources: list[Hit]) -> bool:
    """Every number in ``text`` appears in at least one source (for refusal reasons)."""
    pool = " ".join(_grounding(h) for h in sources)
    return all(n in pool for n in _NUM.findall(_digits_norm(_CITE_ANY.sub("", text))))


_FULLWIDTH_CITE = re.compile(r"【\s*((?:S\s*)?\d+(?:\s*[,;]\s*(?:S\s*)?\d+)*)[^】]*】")


_INVISIBLE = re.compile("[\u200b\u200c\u200d\u2060\ufeff]")   # zero-width spaces and joiners, BOM


def normalize_citations(text: str, strip_invisible: bool = True) -> str:
    """Citation styles the drafter uses -> [S#]. Found in the first real run: gpt-oss writes
    【S3】 (sometimes 【S3†...】) instead of the [S3] the prompt asks for, and "[ S2 ]" with spaces.
    Lists become [S1][S3]. Found by the gate replay (2026-11-11): gpt-oss also writes a zero-width
    space inside the brackets ("[<zero-width space>S1]"), which the gate didn't read as a citation; invisible
    characters are removed (strip_invisible=False reproduces the gate as measured at e7aa65f)."""
    def fw(m):
        ids = re.findall(r"\d+", m.group(1))
        return "".join(f"[S{i}]" for i in ids)
    if strip_invisible:
        text = _INVISIBLE.sub("", text)
    text = _FULLWIDTH_CITE.sub(fw, text)
    return re.sub(r"\[\s*S\s*(\d+)\s*\]", r"[S\1]", text)


def split_sentences(text: str) -> list[str]:
    out = []
    for line in text.splitlines():
        line = _LIST_MARK.sub("", line).strip()
        if not line:
            continue
        # a sentence ends at . ! ? followed by space, possibly after its citations
        parts = re.split(r"(?<=[.!?])\s+(?=[A-Z(\"“])|(?<=\])\s+(?=[A-Z(\"“])", line)
        out += [p.strip() for p in parts if p.strip()]
    return out


def cited_ids(sentence: str) -> list[int]:
    ids = []
    for m in _CITE_ANY.finditer(sentence):
        ids += [int(x) for x in re.findall(r"\d+", m.group(0))]
    return ids


_TERMINAL = re.compile(r"[.!?][\"”'’)\]*_]*$")
_LATEX_CMD = re.compile(r"\\[A-Za-z]+")
_MATH = re.compile(r"\\[()\[\]]|\$|\\[A-Za-z]+")       # any LaTeX: inline or display math, or a command
_NUMBER_WORDS = re.compile(
    r"\b(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|"
    r"seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|"
    r"million|billion|dozen|half|twice|double|triple|single|first|second|third)\b", re.I)
# Words that open or link sentences: capitalised, they are not names.
_PLAIN_WORDS = frozenset(
    "a an and as at both but by each either for from here how however in instead it its moreover neither "
    "nor of on or overall so such that the their then there therefore these they this those thus to "
    "together what when where which while why with yet also finally".split())
RARE_DF = 0.05          # a word in fewer than 5% of the corpus pieces is a technical term
_rare_cache: dict[int, frozenset] = {}


def rare_terms(retriever) -> frozenset:
    """Content words that occur in fewer than RARE_DF of the corpus pieces (stemmed as in BM25)."""
    key = id(retriever)
    if key not in _rare_cache:
        df: dict[str, int] = {}
        for p in retriever.pieces:
            for t in set(tokenize(p.text)):
                df[t] = df.get(t, 0) + 1
        n = max(1, len(retriever.pieces))
        _rare_cache[key] = frozenset(t for t, c in df.items() if c / n < RARE_DF)
    return _rare_cache[key]


def technical_terms(sentence: str, rare: frozenset = frozenset(), vocab: set | None = None) -> list[str]:
    """What makes an uncited sentence a claim rather than filler: numbers (digits or words), a
    quotation, LaTeX, a name or acronym, a hyphenated or underscored compound, or a word that is rare
    in the corpus and occurs in the given sources (``vocab``; everyday words are rare in the papers
    too, so rarity alone is not enough). A found term fails the draft; a sentence without one is
    dropped, never released."""
    bare = _CITE_ANY.sub("", sentence)
    found = []
    found += _NUM.findall(_digits_norm(bare))
    found += [m.group(0) for m in _NUMBER_WORDS.finditer(bare)]
    found += ["quotation"] if _QUOTE.search(bare) else []
    found += ["LaTeX"] if _MATH.search(bare) else []
    for w in re.findall(r"[A-Za-z][A-Za-z0-9’'_‑-]*", bare):
        core = re.sub(r"[’']s$", "", w)
        if re.search(r"[-‑_]", core.strip("-‑")):
            found.append(core)
        elif sum(c.isupper() for c in core) >= 2 or (core[0].isupper() and core.lower() not in _PLAIN_WORDS):
            found.append(core)
    found += [t for t in tokenize(bare) if t in rare and (vocab is None or t in vocab)]
    return found


def _is_formula(s: str) -> bool:
    """A display-formula line: mostly markup once LaTeX command names are removed (a prose sentence
    with a little inline math, like "\\(F\\) is the fraction of ...", is not one)."""
    bare = _LATEX_CMD.sub("", _CITE_ANY.sub("", s))
    chars = [c for c in bare if not c.isspace()]
    return bool(chars) and sum(c.isalpha() for c in chars) / len(chars) < 0.5


def join_formula_lines(sents: list[str]) -> list[str]:
    """Join uncited lines that are a formula, or lead into one without ending a sentence ("... is
    computed as"), to the next cited line, so a display formula is checked with the cited sentence it
    belongs to. Only when there is math in the joined lines: an uncited list item never borrows a
    later citation. Found in the real load runs: Q3's positional-encoding equations were split into
    uncited lines."""
    out, buf = [], []
    for s in sents:
        if not cited_ids(s) and (_is_formula(s) or not _TERMINAL.search(s.strip())):
            buf.append(s)
            continue
        if buf and (any(_is_formula(b) for b in buf) or _MATH.search(s)):
            out.append(" ".join(buf + [s]))
        else:
            out += buf + [s]
        buf = []
    if buf:
        out.append(" ".join(buf))
    return out


def _drop_sentences(text: str, drop: list[str]) -> str:
    """``text`` without the dropped sentences (and without lines or list items left empty)."""
    for s in drop:
        text = text.replace(s, "", 1)
    lines = []
    for line in text.splitlines():
        if not line.strip():
            lines.append("")
        elif re.search(r"\w", _LIST_MARK.sub("", line)):
            lines.append(re.sub(r"(?<=\S)[ \t]{2,}(?=\S)", " ", line).rstrip())
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def check_citations(text: str, sources: list[Hit], drop_plain: bool = False, join_formulas: bool = False,
                    rare: frozenset = frozenset(), legacy_numbers: bool = False) -> dict:
    """The deterministic gate. Returns {ok, sentences, failures: [{sentence, reason}], dropped, text}.

    ``text`` is what may be released: every sentence in it is cited and checked.
    drop_plain: an uncited sentence with no number, quotation or technical term (technical_terms) is
      dropped from the reply instead of failing the draft; an uncited sentence with one still fails
      it. Found in the real load runs: correct drafts failed twice on filler like "Together, these
      components give ...". An uncited sentence is never released.
    join_formulas: see join_formula_lines."""
    sents = split_sentences(text)
    units = join_formula_lines(sents) if join_formulas else sents
    failures, dropped = [], []
    n = len(sources)
    texts = [_grounding(h) for h in sources]
    content = 0
    vocab = {t for h in sources for t in tokenize(h.piece.text)} if drop_plain else set()
    for s in units:
        bare = _CITE_ANY.sub("", s).strip()
        if len(re.findall(r"[A-Za-z]{2,}", bare)) < 3:
            continue                      # a fragment, not a claim
        ids = cited_ids(s)
        if not ids:
            terms = technical_terms(s, rare, vocab) if drop_plain else []
            if drop_plain and not terms:
                dropped.append(s)
                continue
            content += 1
            failures.append({"sentence": s, "reason": "no citation", "terms": terms})
            continue
        content += 1
        bad = [i for i in ids if not 1 <= i <= n]
        if bad:
            failures.append({"sentence": s, "reason": f"cites a source that wasn't given: {bad}"})
            continue
        cited = " ".join(texts[i - 1] for i in ids)
        for num in (LEGACY_NUM if legacy_numbers else _NUM).findall(_digits_norm(bare)):
            if num not in cited:
                failures.append({"sentence": s, "reason": f"number {num} is not in the cited sources"})
                break
        else:
            for q in _QUOTE.findall(bare):
                if _norm(q) not in _norm(" ".join(h.piece.text for h in (sources[i - 1] for i in ids))):
                    failures.append({"sentence": s, "reason": "quotation not found in the cited sources"})
                    break
    if content == 0:
        failures.append({"sentence": text[:120], "reason": "no answer sentences"})
    out = _drop_sentences(text, dropped) if dropped and not failures else text
    return {"ok": not failures, "sentences": len(sents), "failures": failures, "dropped": dropped, "text": out}


# ----------------------------------------------------------------------------- prompts
ROUTER_SYSTEM = (
    "You route questions about exactly three research papers:\n"
    "  attention: 'Attention Is All You Need' (the Transformer, 2017)\n"
    "  moshi: 'Moshi', a speech-text foundation model for real-time full-duplex spoken dialogue (2024)\n"
    "  dynamic_chunking: 'H-Net', dynamic chunking for end-to-end hierarchical sequence modeling (2025)\n"
    "Reply with JSON only, no prose: {\"papers\": [...], \"query\": \"...\", \"alt_query\": \"...\"}.\n"
    "papers: the ids the question is about (empty list if it is about none of them).\n"
    "query: a keyword search query for the answer, using the words the paper itself would most likely "
    "use (technical terms, not the question's everyday words).\n"
    "alt_query: a second, differently worded keyword query with synonyms.")

# Production router (config/production.toml [router] hyde = true): the same JSON plus one hypothetical
# answer sentence, written the way the paper would phrase it. It is used only as the embedding query
# (HyDE, Gao et al. 2022): a guessed answer sits closer to the real passage in embedding space than
# the question does. It is never shown to anyone, so a wrong guess costs only a worse search.
ROUTER_SYSTEM_HYDE = ROUTER_SYSTEM.replace(
    'Reply with JSON only, no prose: {"papers": [...], "query": "...", "alt_query": "..."}.',
    'Reply with JSON only, no prose: {"papers": [...], "query": "...", "alt_query": "...", "hyde": "..."}.'
) + ("\nhyde: one or two sentences that answer the question the way the paper itself would, using its "
     "technical terms (a plausible guess is fine; it is used only to search).")

DRAFT_SYSTEM = (
    "Answer the question using only the numbered sources below.\n"
    "Rules:\n"
    "1. End every sentence with the citation(s) of the source(s) that support it, like [S1] or [S1][S3].\n"
    "2. Copy numbers exactly as the sources write them. Never use a number that is not in a cited source.\n"
    "3. Each source names its paper. Only use a source for claims about that paper.\n"
    "4. If the sources do not answer the question, reply with one line only: "
    "NOT_IN_PAPERS: <one short sentence saying what is missing>. Do not guess.\n"
    "5. At most 6 sentences, plain prose.")

VERIFY_SYSTEM = (
    "You check an answer against its sources. For each numbered sentence, decide whether the sources it "
    "cites state what it says. Reply with JSON only: {\"unsupported\": [sentence numbers]} "
    "(an empty list if every sentence is supported).")

NOT_IN_PAPERS = "NOT_IN_PAPERS"


def render_sources(hits: list[Hit]) -> str:
    out = []
    for i, h in enumerate(hits, 1):
        p = h.piece
        out.append(f"[S{i}] ({PAPER_TITLES[p.paper]}; {p.cite()})\n{p.text}")
    return "\n\n".join(out)


def draft_messages(question: str, hits: list[Hit], feedback: str | None = None) -> list[dict]:
    user = f"Question: {question}\n\nSOURCES:\n{render_sources(hits)}"
    if feedback:
        user += f"\n\n{feedback}\nWrite the answer again, following the rules."
    return [{"role": "system", "content": DRAFT_SYSTEM}, {"role": "user", "content": user}]


def source_list(hits: list[Hit]) -> list[dict]:
    return [{"sid": f"S{i}", "cite": h.piece.cite(), "pid": h.piece.pid, "paper": h.piece.paper,
             "block_ids": list(h.piece.block_ids)} for i, h in enumerate(hits, 1)]


def best_excerpt(text: str, query: str, max_chars: int = 600) -> str:
    """The run of sentences that best covers the query, for passages-only replies."""
    sents = re.split(r"(?<=[.!?])\s+", text.replace("\n", " "))
    q = set(tokenize(query))
    best, best_score = text[:max_chars], -1.0
    for i in range(len(sents)):
        chunk, j = "", i
        while j < len(sents) and len(chunk) + len(sents[j]) <= max_chars:
            chunk = f"{chunk} {sents[j]}".strip()
            j += 1
        if not chunk:
            chunk = sents[i][:max_chars]
        score = len(q & set(tokenize(chunk)))
        if score > best_score:
            best, best_score = chunk, score
    return best


def _json_obj(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return None
    try:
        v = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return v if isinstance(v, dict) else None


_PAPER_WORDS = {"attention": ("transformer", "attention", "vaswani", "wmt"),
                "moshi": ("moshi", "mimi", "helium", "inner monologue", "full-duplex", "kyutai"),
                "dynamic_chunking": ("h-net", "hnet", "dynamic chunking", "chunking", "byte-level")}


def names_a_paper(question: str) -> bool:
    q = question.lower()
    return any(w in q for ws in _PAPER_WORDS.values() for w in ws)


# ----------------------------------------------------------------------------- pipeline
class _Step(Exception):
    """Internal: a model step could not run (budget, deadline, every model failed). ``rec`` is the
    CallRecord when a request was actually sent and failed (a 429, 5xx, timeout), else None."""

    def __init__(self, msg: str, rec: CallRecord | None = None) -> None:
        super().__init__(msg)
        self.rec = rec


def is_daily_limit(rec: CallRecord | None) -> bool:
    """A 429 naming a per-day limit (RPD, TPD, ...): retrying another minute won't help."""
    if rec is None or rec.status != 429:
        return False
    from relay.limiter import parse_limit_body
    info = parse_limit_body(rec.error_body)
    return bool(info and (info.get("period_s") or 0) >= 3600)


class RetryBudget:
    """Global retry budget: retries may be at most ``fraction`` of the sent attempts in the last
    ``window_s`` seconds (at least ``min_retries``), so retries can't multiply load in an overload."""

    def __init__(self, fraction: float, window_s: float = 60.0, min_retries: int = 1) -> None:
        self.fraction, self.window_s, self.min_retries = fraction, window_s, min_retries
        self.attempts: deque = deque()
        self.retries: deque = deque()

    def _trim(self, now: float) -> None:
        for d in (self.attempts, self.retries):
            while d and d[0] < now - self.window_s:
                d.popleft()

    def note_attempt(self, now: float) -> None:
        self.attempts.append(now)

    def allow(self, now: float) -> bool:
        self._trim(now)
        return len(self.retries) < max(self.min_retries, self.fraction * len(self.attempts))

    def note_retry(self, now: float) -> None:
        self.retries.append(now)


class Pipeline:
    def __init__(self, chat: ChatFn, limiter: Limiter, retriever: Retriever, config: dict | None = None) -> None:
        self.chat, self.lim, self.retriever = chat, limiter, retriever
        try:      # does the chat function take on_delta (streaming)? Without it, claims are parsed after the call
            params = inspect.signature(chat).parameters.values()
            self._chat_streams = any(x.name == "on_delta" or x.kind == x.VAR_KEYWORD for x in params)
        except (TypeError, ValueError):
            self._chat_streams = False
        self.cfg = config or load_config()
        self.models = self.cfg["models"]
        self.max_tokens = self.cfg["max_tokens"]
        self.timeouts = {k: Timeouts(**v) for k, v in self.cfg.get("timeouts", {}).items()}
        rp = self.cfg.get("retry") or {}
        self.retry_policy = rp if rp.get("enabled") else None
        self.retry_budget = RetryBudget(float(rp.get("global_budget_fraction", 0.10)),
                                        float(rp.get("window_s", 60.0))) if self.retry_policy else None
        lc = self.cfg.get("limiter") or {}
        if "token_margin" in lc and hasattr(self.lim, "token_margin"):
            self.lim.token_margin = float(lc["token_margin"])

    # ------------------------------------------------------------ public
    async def answer(self, question: str, request_id: str | None = None, patience_s: float | None = None,
                     trace: Trace | None = None, on_claim: Callable[[dict], None] | None = None) -> Reply:
        """``trace``: pass your own to keep the spans of a request that is cancelled (a person who left).
        ``on_claim``: claims mode only. Called with each verified claim as soon as it is checked, so a
        caller can stream the answer sentence by sentence (the LLM verifier is skipped when streaming:
        it needs the whole answer)."""
        rid = request_id or uuid.uuid4().hex[:12]
        tr = trace if trace is not None else Trace()
        arrival = self.lim.now()
        deadline = arrival + min(self.cfg["ladder"]["deadline_s"], patience_s or 1e9)
        calls: list[ModelCall] = []
        tr.meta["_on_claim"] = on_claim
        try:
            reply = await self._answer(rid, question, tr, calls, deadline)
        except Exception as e:                       # internal failure only -> busy
            try:
                wait = self._eta(self.cfg["retrieval"]["k_lean"])
            except Exception:
                wait = None
            n = int(round(wait)) if wait is not None and wait > 0 else int(self.cfg["ladder"]["busy_retry_s"])
            reply = Reply(rid, question, "busy", BUSY, f"Sorry, the service is busy. Please retry in {n} s.", [],
                          notice=f"internal failure: {type(e).__name__}", eta_s=float(n))
        with tr.span("send") as s:
            s.outcome = reply.kind
        tr.meta.pop("_on_claim", None)
        for k, v in tr.meta.items():
            reply.checks.setdefault(k, v)
        reply.model_calls = calls
        reply.spans = tr.spans
        reply.released_ms = round(tr.ms(), 1)
        if reply.first_token_ms is None:
            reply.first_token_ms = reply.released_ms
        reply.budget_wait_ms = round(sum((x.end_ms - x.start_ms) for x in tr.spans
                                         if x.name.startswith("wait:") and x.end_ms is not None), 1)
        return reply

    # ------------------------------------------------------------ the steps
    async def _answer(self, rid, question, tr: Trace, calls, deadline) -> Reply:
        rc, lad = self.cfg["retrieval"], self.cfg["ladder"]
        with tr.span("plan") as s:
            # plan from the real prompt: one BM25 search on the question (no budget) sizes the draft
            plan_hits = await self._offload(self.search_papers, question, rc["k_full"])
            level, drafter, waits = self._plan(deadline, question, plan_hits)
            s.outcome = LEVEL_NAMES[level]
            s.detail = {"predicted_wait_s": waits, "drafter": drafter}
        if level == PASSAGES:
            hits = await self._offload(self._retrieve, tr, question, question, None, rc["k_lean"])
            eta = min((w for w in waits["lean"].values() if w is not None), default=None)
            # name the daily bucket when every drafter lacks daily room, so the reply gives no time estimate
            mx = self.cfg["max_tokens"]["draft"]
            daily = [self.lim.daily_short(m, self._prompt_for(m, rc["k_lean"], question, hits), mx)
                     for m in self.models["draft"]]
            why = f"no drafter free within {lad['max_wait_s']:g} s"
            if daily and all(daily):
                why += f" (daily allowance: {', '.join(sorted(set(daily)))})"
            return self._passages(rid, question, hits, None if eta is None else round(eta, 1), why)

        # understand
        papers, query, alt = None, question, None
        router_note = ""
        steps = self.cfg.get("steps", {})
        hyde = None
        if level in (FULL, FULL_OTHER) and steps.get("router", True):
            with tr.span("understand") as s:
                try:
                    papers, query, alt, hyde = await self._understand(question, tr, calls, deadline)
                    s.outcome = "router"
                    tr.meta["router"] = {"papers": papers, "query": query, "alt_query": alt, "hyde": hyde}
                except _Step as e:
                    s.outcome = f"heuristic ({e})"
                    router_note = "no router"
                    tr.meta["router"] = {"failed": str(e)}
        else:
            with tr.span("understand") as s:
                s.outcome = "skipped (lean)" if level not in (FULL, FULL_OTHER) else "off (config)"

        # decide (router): about none of the papers
        if papers == [] and not names_a_paper(question):
            with tr.span("decide") as s:
                s.outcome = "refuse: router says no paper"
            return Reply(rid, question, "refusal", level,
                         "The three papers (the Transformer, Moshi and H-Net) don't cover this question.", [],
                         notice="refused without a draft: the question is about none of the papers")

        k = rc["k_full"] if level in (FULL, FULL_OTHER) else rc["k_lean"]
        # dense query: the router's hypothetical answer (HyDE) plus the question, when there is one
        hits = await self._offload(self._retrieve, tr, query, question, papers or None, k, alt,
                              dense_query=f"{hyde} {question}" if hyde else question)
        tr.meta["retrieval"] = {"query": query, "papers": papers or None, "k": k,
                                "pids": [h.piece.pid for h in hits]}

        with tr.span("decide") as s:
            # Evidence is scored against the question AND against the router's rewrite, and the gate
            # refuses only when both are weak. Scoring the question alone turned a vocabulary mismatch
            # into a refusal (Q14: the routing passage shares none of the question's words;
            # docs/findings_and_decisions.md, candidate 1). Changed 2026-11-11, after the retrieval sweep.
            dc = self.cfg["decide"]
            ev_q = self.retriever.evidence(question, hits)
            ev_r = self.retriever.evidence(query, hits) if query != question else ev_q

            def weak(ev):
                return ev["best"] < dc["refuse_if_best_below"] or ev["top_bm25"] < dc["refuse_if_top_bm25_below"]
            ev = ev_q if not weak(ev_q) or weak(ev_r) else ev_r
            s.detail = {**ev, "question": ev_q, "rewrite": ev_r if query != question else None}
            # semantic evidence (production, hybrid retrieval): a question that shares few words with
            # the passages but means the same thing is not refused. Measured: answerable questions'
            # best child cosine >= 0.56, off-topic 0.44-0.63 (data/retrieval/v2.md); the router
            # refuses off-topic questions first, so this only rescues lexically weak answerable ones.
            dense_ok = False
            if dc.get("refuse_if_dense_below") is not None and hits and hasattr(self.retriever, "dense_best") \
                    and getattr(self.retriever, "dense", None) is not None:
                db = await self._offload(self.retriever.dense_best, question, hits)
                s.detail["dense_best"] = round(db, 3)
                dense_ok = db >= dc["refuse_if_dense_below"]
            if not hits or (weak(ev_q) and weak(ev_r) and not dense_ok):
                s.outcome = "refuse: evidence too weak"
                return Reply(rid, question, "refusal", level,
                             "The papers don't appear to cover this: no passage matches the question.",
                             [], notice="refused without a model call: retrieval found almost nothing",
                             checks={"evidence": ev})
            s.outcome = "draft"

        if (self.cfg.get("draft") or {}).get("format") == "claims":
            return await self._answer_claims(rid, question, hits, level, drafter, tr, calls, deadline, router_note)

        # draft + check, stepping down the ladder if the drafter isn't available
        attempt_models = self._draft_order(drafter)
        feedback = None
        for attempt in (1, 2):
            try:
                text, model, level = await self._draft(question, hits, level, attempt_models, tr, calls, deadline,
                                                       feedback)
            except _Step as e:
                lean_hits = hits[: rc["k_lean"]]
                return self._passages(rid, question, lean_hits, self._eta(rc["k_lean"], question, lean_hits),
                                      f"no drafter available ({e})")
            text = normalize_citations(text)     # gpt-oss writes 【S3】; the gate and the person see [S3]
            if level == LEAN and len(hits) > rc["k_lean"]:
                hits = hits[: rc["k_lean"]]
            if text.strip().upper().startswith(NOT_IN_PAPERS):
                with tr.span("check") as s:
                    s.outcome = "refusal"
                reason = text.split(":", 1)[1].strip() if ":" in text else ""
                if reason and not numbers_grounded(reason, hits):
                    reason = ""                     # never pass on a guessed number
                body = "The papers don't say." + (f" {reason}" if reason else "")
                return Reply(rid, question, "refusal", level, body, [], notice=self._notice(level, router_note),
                             checks={"draft_model": model})
            with tr.span("check") as s:
                g = self.cfg.get("gate", {})
                gate = check_citations(text, hits, drop_plain=bool(g.get("drop_plain_sentences", False)),
                                       join_formulas=bool(g.get("join_formulas", False)),
                                       rare=rare_terms(self.retriever) if g.get("drop_plain_sentences") else frozenset())
                s.detail = {"gate_ok": gate["ok"], "failures": len(gate["failures"]),
                            "failed": [f["reason"] + ": " + f["sentence"][:200] for f in gate["failures"]],
                            "dropped": gate["dropped"]}
                if gate["ok"]:
                    text = gate["text"]           # without dropped sentences: every released sentence is cited
                verdict = None
                if gate["ok"] and level in (FULL, FULL_OTHER):
                    verdict = await self._verify(text, hits, model, tr, calls, deadline)
                    s.detail["verifier"] = verdict
                ok = gate["ok"] and (verdict is None or verdict.get("ok", True))
                s.outcome = "pass" if ok else "fail"
            if ok:
                notice = self._notice(level, router_note, verified=verdict is not None)
                return Reply(rid, question, "answer", level, text.strip(), source_list(hits), notice=notice,
                             checks={"gate": gate, "verifier": verdict, "draft_model": model, "attempts": attempt})
            reasons = [f["reason"] + ": " + f["sentence"][:160] for f in gate["failures"]]
            if verdict and not verdict.get("ok", True):
                reasons += [f"sentence {i} is not supported by its sources" for i in verdict.get("unsupported", [])]
            feedback = "Your previous answer failed the check:\n- " + "\n- ".join(reasons[:6])
            attempt_models = [model] + [m for m in attempt_models if m != model]
        return self._passages(rid, question, hits, None, "both drafted answers failed the citation check",
                              eta_note=False)

    async def _offload(self, fn, *args, **kw):
        """Run retrieval off the event loop when it does model inference (embeddings, rerank): a 1.7 s CPU
        rerank inline would stall every other request in the process. Plain BM25 stays inline (< 1 ms)."""
        if hasattr(self.retriever, "dense"):
            return await asyncio.to_thread(fn, *args, **kw)
        return fn(*args, **kw)

    # ------------------------------------------------------------ claims mode (production)
    async def _answer_claims(self, rid, question, hits, level, drafter, tr: Trace, calls, deadline,
                             router_note) -> Reply:
        """Draft as JSON-Lines claims carrying their own citation metadata (relay.claims). Each claim is
        checked as its line arrives; with ``on_claim`` it is released immediately (streaming). A claim
        that fails its check is dropped, never released. No verified claim -> one redraft with the
        reasons -> passages. The LLM verifier runs only when not streaming (it needs the whole answer)
        and, since claims are separate, drops just the claims it finds unsupported."""
        rc = self.cfg["retrieval"]
        on_claim = tr.meta.get("_on_claim")
        released: list[Claim] = []
        holder: dict = {}
        first_ms: list[float] = []

        def factory(use: list[Hit]):
            stream = ClaimStream(use)
            holder["stream"], holder["hits"] = stream, use

            def on_delta(kind, text):
                if kind != "content":
                    return
                for c in stream.feed(text):
                    if c.ok and on_claim is not None:
                        released.append(c)
                        if not first_ms:
                            first_ms.append(round(tr.ms(), 1))
                        on_claim({**c.to_dict(use), "sid_cites": source_list(use)})
            return on_delta

        attempt_models = self._draft_order(drafter)
        feedback, model, last_dropped, verdict_any = None, None, [], False
        for attempt in (1, 2):
            try:
                rec, model, level = await self._draft_claims(question, hits, level, attempt_models, tr, calls,
                                                             deadline, feedback, factory, released)
            except _Step as e:
                if released:                       # partial stream: keep what was verified and sent
                    break
                lean_hits = hits[: rc["k_lean"]]
                return self._passages(rid, question, lean_hits, self._eta(rc["k_lean"], question, lean_hits),
                                      f"no drafter available ({e})")
            stream: ClaimStream = holder["stream"]
            use: list[Hit] = holder["hits"]
            # a chat client that doesn't stream: parse the finished text now (same checks, no early release)
            tail = (rec.text or "") if not self._chat_streams else ""
            for c in stream.feed(tail, final=True):  # (and a last line without a newline)
                if c.ok and on_claim is not None:
                    released.append(c)
                    if not first_ms:
                        first_ms.append(round(tr.ms(), 1))
                    on_claim({**c.to_dict(use), "sid_cites": source_list(use)})
            good, dropped = stream.good, stream.dropped
            last_dropped = dropped
            with tr.span("check") as s:
                s.detail = {"claims": len(stream.claims), "verified": len(good),
                            "dropped": [f"{c.reason}: {c.text[:120]}" for c in dropped], "junk_lines": len(stream.junk)}
                if stream.refusal is not None and not good:
                    s.outcome = "refusal"
                    reason = stream.refusal if numbers_grounded(stream.refusal, use) else ""
                    return Reply(rid, question, "refusal", level,
                                 "The papers don't say." + (f" {reason}" if reason else ""), [],
                                 notice=self._notice(level, router_note), checks={"draft_model": model})
                if good and on_claim is None and level in (FULL, FULL_OTHER):
                    verdict = await self._verify("\n".join(render([c]) for c in good), use, model, tr, calls, deadline)
                    s.detail["verifier"] = verdict
                    verdict_any = verdict is not None
                    if verdict and verdict.get("unsupported"):
                        bad = set(verdict["unsupported"])
                        for i, c in enumerate(list(good), 1):
                            if i in bad:
                                c.ok, c.reason = False, "the verifier found it unsupported by its sources"
                        good, dropped = stream.good, stream.dropped
                        last_dropped = dropped
                # a redraft only when nothing usable came back, or (not streaming) most claims failed
                need_redraft = not good or (on_claim is None and len(dropped) > len(good))
                s.outcome = "pass" if not need_redraft else "fail"
            if not need_redraft or released:
                break
            reasons = [f"{c.reason}: {c.text[:160]}" for c in dropped] or stream.junk[:3] or ["no claims"]
            feedback = ("Your previous answer failed the check (each line must be one JSON object whose quote is "
                        "copied exactly from a cited source):\n- " + "\n- ".join(reasons[:6]))
            attempt_models = [model] + [m for m in attempt_models if m != model]
        stream = holder.get("stream")
        use = holder.get("hits") or hits
        final = released if on_claim is not None else (stream.good if stream else [])
        if not final:
            return self._passages(rid, question, hits, None, "both drafted answers failed the citation check",
                                  eta_note=False)
        verified = on_claim is not None or verdict_any
        notice = self._notice(level, router_note, verified=verified)
        if last_dropped:
            notice = "; ".join(x for x in (notice, f"{len(last_dropped)} statement(s) removed because their "
                                                   "supporting quote or number wasn't in the sources") if x)
        if on_claim is not None:
            notice = "; ".join(x for x in (notice, "streamed sentence by sentence: each checked against its quoted "
                                                   "source, no second-model check") if x)
        reply = Reply(rid, question, "answer", level, render(final), source_list(use), notice=notice,
                      checks={"draft_model": model, "dropped": [c.to_dict(use) for c in last_dropped]},
                      claims=[c.to_dict(use) for c in final])
        if first_ms:
            reply.first_token_ms = first_ms[0]
        return reply

    async def _draft_claims(self, question, hits, level, models, tr, calls, deadline, feedback, factory, released):
        rc, lad = self.cfg["retrieval"], self.cfg["ladder"]
        with tr.span("draft", level=LEVEL_NAMES[level], format="claims") as s:
            if level in (FULL, FULL_OTHER):
                msgs = claims_messages(question, hits, render_sources, feedback)
                try:
                    rec, m = await self._try_models("draft", models, msgs, tr, calls, lad["preferred_free_within_s"],
                                                    deadline, on_delta_factory=lambda: factory(hits),
                                                    stop_if=lambda: bool(released))
                    s.outcome = m
                    return rec, m, (FULL if m == self.models["draft"][0] else FULL_OTHER)
                except _Step:
                    if released:
                        raise
                    level = LEAN
            lean = hits[: rc["k_lean"]]
            msgs = claims_messages(question, lean, render_sources, feedback)
            rec, m = await self._try_models("draft", models, msgs, tr, calls, lad["max_wait_s"], deadline,
                                            on_delta_factory=lambda: factory(lean), stop_if=lambda: bool(released))
            s.outcome = f"{m} (lean)"
            return rec, m, LEAN

    # ------------------------------------------------------------ planning
    def _plan(self, deadline: float, question: str, hits: list[Hit]) -> tuple[int, str | None, dict]:
        lad, rc = self.cfg["ladder"], self.cfg["retrieval"]
        drafters = self.models["draft"]
        mx = self.max_tokens["draft"]
        full = {m: self.lim.predict_wait(m, self._prompt_for(m, rc["k_full"], question, hits), mx) for m in drafters}
        free = lad["preferred_free_within_s"]
        if full[drafters[0]] is not None and full[drafters[0]] <= free:
            return FULL, drafters[0], {"full": full}
        for m in drafters[1:]:
            if full[m] is not None and full[m] <= free:
                return FULL_OTHER, m, {"full": full}
        lean = {m: self.lim.predict_wait(m, self._prompt_for(m, rc["k_lean"], question, hits), mx) for m in drafters}
        room = min(lad["max_wait_s"], deadline - self.lim.now())
        ok = [(w, i, m) for i, m in enumerate(drafters) if (w := lean[m]) is not None and w <= room]
        if ok:
            w, _, m = min(ok)
            return LEAN, m, {"full": full, "lean": lean}
        return PASSAGES, None, {"full": full, "lean": lean}

    def _est_for(self, model: str, k: int) -> int:
        """Worst-case prompt tokens of a draft call with k pieces, when no retrieval is at hand
        (the busy path): system prompt, a question allowance, k full-size pieces with headers."""
        chars = len(DRAFT_SYSTEM) + 400 + k * (self.cfg["retrieval"]["piece_chars"] + 80)
        return self.lim.estimator.estimate(model, chars, 2)

    def _prompt_for(self, model: str, k: int, question: str | None, hits: list[Hit] | None) -> int:
        """Prompt tokens of the draft call: from the real retrieved pieces when we have them."""
        if question and hits:
            return self.lim.estimate(model, draft_messages(question, hits[:k])).tokens
        return self._est_for(model, k)

    def _eta(self, k: int, question: str | None = None, hits: list[Hit] | None = None) -> float | None:
        """Seconds until some drafter could take a lean draft (the limiter's refill-rate prediction)."""
        ws = [self.lim.predict_wait(m, self._prompt_for(m, k, question, hits), self.max_tokens["draft"])
              for m in self.models["draft"]]
        ws = [w for w in ws if w is not None]
        return round(min(ws), 1) if ws else None

    def _draft_order(self, first: str | None) -> list[str]:
        order = list(self.models["draft"])
        if first in order:
            order.remove(first)
            order.insert(0, first)
        return order

    @staticmethod
    def _notice(level: int, router_note: str = "", verified: bool = False) -> str:
        parts = []
        if level == FULL_OTHER:
            parts.append("answered by a fallback model because the preferred one was busy")
        if level == LEAN:
            parts.append("shorter answer from fewer passages, without the extra checks, because the service is busy")
        if router_note:
            parts.append("the question was searched as written (query rewriting was skipped)")
        if level in (FULL, FULL_OTHER) and not verified:
            parts.append("checked for citations only (no second-model check)")
        return "; ".join(parts)

    # ------------------------------------------------------------ retrieval (the tool)
    def search_papers(self, query: str, k: int, papers: list[str] | None = None,
                      dense_query: str | None = None) -> list[Hit]:
        kw = {"dense_query": dense_query} if dense_query and hasattr(self.retriever, "dense") else {}
        return self.retriever.search(query, k, papers, use_rerank=self.cfg["retrieval"].get("rerank", False), **kw)

    def _retrieve(self, tr: Trace, query: str, question: str, papers, k: int, alt: str | None = None,
                  dense_query: str | None = None) -> list[Hit]:
        rc = self.cfg["retrieval"]
        with tr.span("retrieve", query=query, papers=papers, k=k) as s:
            hits = self.search_papers(query, k, papers, dense_query)
            ev = self.retriever.evidence(question, hits)
            s.detail["evidence"] = ev
            s.outcome = "1 search"
            if ev["best"] < rc["second_search_below"]:
                q2 = alt or (question if query != question else None)
                if q2:
                    with tr.span("retrieve:again", query=q2) as s2:
                        more = self.search_papers(q2, k, None)
                        hits = self._fuse([hits, more], k)
                        s2.detail["evidence"] = self.retriever.evidence(question, hits)
                    s.outcome = "2 searches"
        return hits

    @staticmethod
    def _fuse(rankings: list[list[Hit]], k: int, c: int = 60) -> list[Hit]:
        """Reciprocal rank fusion: BM25 scores of different queries aren't comparable, ranks are."""
        score: dict[str, float] = {}
        first: dict[str, Hit] = {}
        for hits in rankings:
            for h in hits:
                score[h.piece.pid] = score.get(h.piece.pid, 0.0) + 1.0 / (c + h.rank)
                if h.piece.pid not in first or h.bm25 > first[h.piece.pid].bm25:
                    first[h.piece.pid] = h
        order = sorted(score, key=lambda p: -score[p])[:k]
        out = []
        for r, pid in enumerate(order, 1):
            h = first[pid]
            out.append(Hit(h.piece, score[pid], r, bm25=h.bm25, features=h.features))
        return out

    # ------------------------------------------------------------ model steps
    async def _call(self, step: str, model: str, messages: list[dict], tr: Trace, calls: list[ModelCall],
                    wait_limit_s: float, deadline: float, on_delta=None) -> CallRecord:
        """Reserve (waiting at most wait_limit_s, never past the deadline), call, settle. Raises
        _Step if this model can't serve the step."""
        mx = self.max_tokens[step]
        prompt = self.lim.estimate(model, messages)
        until = min(self.lim.now() + wait_limit_s, deadline)
        with tr.span(f"wait:{step}", model=model) as s:
            try:
                r = self.lim.reserve(model, prompt, mx, until)
                g: Grant = await r.acquire() if isinstance(r, Ticket) else r
            except Unavailable as e:
                s.outcome = f"unavailable: {e.reason}"
                raise _Step(f"{model}: {e.reason}") from None
            s.outcome = "granted"
            s.detail["waited_s"] = round(g.waited_s, 3)
        payload = {"model": model, "messages": messages, "max_tokens": mx,
                   **self.lim.settings.get(model, {})}
        with tr.span(f"call:{step}", model=model) as s:
            try:
                kw = {"on_delta": on_delta} if on_delta is not None and self._chat_streams else {}
                rec = await self.chat(payload, phase=step, timeouts=self.timeouts.get(step), **kw)
            except asyncio.CancelledError:
                g.limiter.abandon(g)        # sent, then the person left: keep what admission charged
                raise
            except Exception as e:          # e.g. relay.ledger.BudgetExceeded: never sent
                g.release()
                s.outcome = f"not sent: {type(e).__name__}"
                raise _Step(f"{model}: {type(e).__name__}") from None
            self.lim.settle(g, rec)
            s.outcome = f"{rec.status}:{rec.outcome}"
        u = rec.usage or {}
        est = not isinstance(u.get("prompt_tokens"), int)
        calls.append(ModelCall(step, model, rec.status, rec.outcome,
                               u.get("prompt_tokens") if not est else (prompt.tokens if rec.status == 200 else 0),
                               u.get("completion_tokens") if isinstance(u.get("completion_tokens"), int)
                               else -(-len(rec.text or "") // 3),
                               est, rec.total_ms, round(g.waited_s, 3)))
        if rec.status != 200 or rec.outcome != "ok":
            raise _Step(f"{model}: {rec.status} {rec.outcome}", rec)
        return rec

    async def _try_models(self, step: str, models: list[str], messages, tr, calls, wait_limit_s, deadline,
                          exclude: tuple[str, ...] = (), on_delta_factory=None, stop_if=None) -> tuple[CallRecord, str]:
        """Each model in fallback order. With [retry] enabled (config/pipeline.toml): after a request
        that was sent and failed, at most max_per_step retries for the step (counted across the
        request, lean step-down included), each on a model not yet failed (a different bucket), never
        after a daily-limit 429, none for best-effort steps, and only within the global retry budget.
        Moving on because the limiter predicts no room is not a retry: nothing was sent."""
        errs = []
        pol = self.retry_policy
        state = tr.meta.setdefault("retry", {}).setdefault(step, {"sent_failures": 0, "retries": 0, "failed": []})
        for m in models:
            if m in exclude or (pol and m in state["failed"]):
                continue
            if self.lim.now() >= deadline:
                break
            if stop_if is not None and stop_if():
                break                      # claims already streamed to the person: never restart elsewhere
            if pol and state["sent_failures"] > 0:
                why = None
                if step in pol.get("best_effort_steps", []):
                    why = "best-effort step: no retry"
                elif state["retries"] >= int(pol.get("max_per_step", 1)):
                    why = f"retry limit ({pol.get('max_per_step', 1)} per step) used"
                elif state.get("daily"):
                    why = "never retry after a daily-limit 429"
                elif not self.retry_budget.allow(self.lim.now()):
                    why = "global retry budget used"
                if why:
                    errs.append(why)
                    state.setdefault("refused", []).append(why)
                    break
                state["retries"] += 1
                self.retry_budget.note_retry(self.lim.now())
            n_before = len(calls)
            try:
                rec = await self._call(step, m, messages, tr, calls, wait_limit_s, deadline,
                                       on_delta_factory() if on_delta_factory is not None else None)
                if pol and len(calls) > n_before:
                    self.retry_budget.note_attempt(self.lim.now())
                return rec, m
            except _Step as e:
                errs.append(str(e))
                if pol and len(calls) > n_before:          # it was sent, and failed
                    self.retry_budget.note_attempt(self.lim.now())
                    state["sent_failures"] += 1
                    state["failed"].append(m)
                    if is_daily_limit(e.rec):
                        state["daily"] = True
        raise _Step("; ".join(errs) or "deadline")

    async def _understand(self, question, tr, calls, deadline):
        hyde_on = bool((self.cfg.get("router") or {}).get("hyde"))
        msgs = [{"role": "system", "content": ROUTER_SYSTEM_HYDE if hyde_on else ROUTER_SYSTEM},
                {"role": "user", "content": question}]
        rec, _ = await self._try_models("understand", self.models["understand"], msgs, tr, calls,
                                        self.cfg["ladder"]["router_free_within_s"], deadline)
        obj = _json_obj(rec.text)
        if obj is None:
            raise _Step("router reply was not JSON")
        papers = [p for p in obj.get("papers") or [] if p in PAPERS]
        if obj.get("papers") and not papers:
            papers = None                    # named only unknown ids: don't trust it
        query = str(obj.get("query") or "").strip() or question
        alt = str(obj.get("alt_query") or "").strip() or None
        hyde = (str(obj.get("hyde") or "").strip() or None) if hyde_on else None
        # keep the question's own words too: the rewrite adds vocabulary, it shouldn't lose any
        return papers, f"{query} {question}", (f"{alt} {question}" if alt else None), hyde

    async def _draft(self, question, hits, level, models, tr, calls, deadline, feedback=None):
        rc, lad = self.cfg["retrieval"], self.cfg["ladder"]
        use = hits if level in (FULL, FULL_OTHER) else hits[: rc["k_lean"]]
        msgs = draft_messages(question, use, feedback)
        with tr.span("draft", level=LEVEL_NAMES[level]) as s:
            if level in (FULL, FULL_OTHER):
                try:
                    rec, m = await self._try_models("draft", models, msgs, tr, calls,
                                                    lad["preferred_free_within_s"], deadline)
                    s.outcome = m
                    return rec.text, m, (FULL if m == self.models["draft"][0] else FULL_OTHER)
                except _Step:
                    level = LEAN               # backed up after all: step down to lean
                    msgs = draft_messages(question, hits[: rc["k_lean"]], feedback)
            rec, m = await self._try_models("draft", models, msgs, tr, calls, lad["max_wait_s"], deadline)
            s.outcome = f"{m} (lean)"
            return rec.text, m, LEAN

    async def _verify(self, text, hits, drafter, tr, calls, deadline) -> dict | None:
        """LLM verifier, only with headroom: its model is free now and keeps enough TPM after it."""
        sents = split_sentences(text)
        numbered = "\n".join(f"{i}. {s}" for i, s in enumerate(sents, 1))
        msgs = [{"role": "system", "content": VERIFY_SYSTEM},
                {"role": "user", "content": f"SOURCES:\n{render_sources(hits)}\n\nANSWER SENTENCES:\n{numbered}"}]
        frac = self.cfg["ladder"]["verify_min_tpm_left_fraction"]
        with tr.span("verify") as s:
            if not self.cfg.get("steps", {}).get("verifier", True):
                s.outcome = "off (config)"
                return None
            for m in self.models["verify"]:
                if m == drafter:
                    continue
                p = self.lim.estimate(m, msgs)
                w = self.lim.predict_wait(m, p, self.max_tokens["verify"])
                cap = self.lim.capacity(m, "TPM")
                left = self.lim.levels(m).get("TPM") if cap else None
                need = p.tokens + self.max_tokens["verify"]
                if w != 0.0 or (cap and left is not None and left - need < frac * cap):
                    continue
                try:
                    rec, _ = await self._try_models("verify", [m], msgs, tr, calls, 0.0, deadline)
                except _Step as e:
                    s.outcome = f"skipped ({e})"
                    return None
                obj = _json_obj(rec.text)
                if obj is None or not isinstance(obj.get("unsupported", []), list):
                    s.outcome = "unparseable: ignored"
                    return None
                bad = [int(x) for x in obj.get("unsupported", []) if str(x).isdigit()]
                s.outcome = "pass" if not bad else f"unsupported {bad}"
                return {"ok": not bad, "unsupported": bad, "model": m}
            s.outcome = "skipped (no headroom)"
            return None

    # ------------------------------------------------------------ fallbacks
    def _passages(self, rid, question, hits, eta, why, eta_note=True) -> Reply:
        if not hits:
            return Reply(rid, question, "refusal", PASSAGES,
                         "The papers don't appear to cover this: no passage matches the question.", [],
                         notice=f"no model call ({why}) and no matching passage", eta_s=eta)
        lines = []
        for i, h in enumerate(hits, 1):
            lines.append(f"[S{i}] {h.piece.cite()}: \"{best_excerpt(h.piece.text, question)}\"")
        # A daily limit (TPD/RPD) gets no time estimate: how Groq's daily allowance refills is the least
        # certain thing we measured, so a number of seconds would be a guess. Per-minute limits refill on a
        # measured schedule, so they keep theirs. Changed 2026-11-11 after review: the old reply gave a
        # daily-limit estimate and did not say the passages were unchecked.
        daily = bool(re.search(r"\b(?:TPD|RPD)\b", why))
        if daily:
            eta = None
            when = " Today's model allowance is used up, so please try again later."
        elif eta_note and eta is not None:
            when = f" A written answer should be possible in about {max(1, round(eta))} s."
        elif eta_note:
            when = " A written answer isn't possible right now."
        else:
            when = ""
        head = (f"No written answer right now.{when} These are the closest passages from the papers, quoted as "
                "they are; they have not been checked against your question.")
        return Reply(rid, question, "passages", PASSAGES, head + "\n" + "\n".join(lines), source_list(hits),
                     notice=f"passages only, no written answer: {why}", eta_s=eta)
