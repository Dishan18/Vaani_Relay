"""Ask the Relay one question, live: from the three papers, or from a new PDF someone hands you.

    python ask.py "Why divide by sqrt(d_k)?"                        retrieval only: free, no model call
    python ask.py "Why divide by sqrt(d_k)?" --standin              whole pipeline on the stand-in (shows the
                                                                    steps; text is a real replay or a placeholder)
    python ask.py "Why divide by sqrt(d_k)?" --real                 whole pipeline on Groq (needs GROQ_API_KEY in
                                                                    Vaani_Relay\\.env; about 2-4 calls of budget)
    python ask.py "What does the paper propose?" --pdf new.pdf --title "Some Paper"   add a new PDF first
        options: --stream (print each verified sentence as it is released), --legacy (the submitted
        pipeline: BM25 + prose + regex gate; default is config/production.toml), --paper KEY, --k 3

Nothing in the repo is modified. A --pdf is extracted with plain pypdf (one block per paragraph,
sections from numbered headings), kept in memory, and added to the library for this run only.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")    # Windows consoles default to cp1252
    except Exception:
        pass
REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

import relay.pipeline as P                     # noqa: E402
import relay.retrieve as R                     # noqa: E402
from relay.limiter import Limiter              # noqa: E402

HEADING = re.compile(r"^(?:\d{1,2}(?:\.\d{1,2}){0,3}|[A-H](?:\.\d{1,2}){0,2})\.?\s+[A-Z][^\n]{2,80}$")


def pdf_blocks(path: Path, key: str) -> list[dict]:
    """Quick extraction for a PDF the repo has never seen (the real extractor insists on an arXiv
    stamp matching the manifest). Paragraphs become blocks; a numbered heading line starts a section."""
    from pypdf import PdfReader
    reader = PdfReader(str(path))
    m = re.search(r"(\d{4}\.\d{4,5}v\d+)", path.name)
    doc_id = m.group(1) if m else path.stem
    blocks, section, n = [], "text", 0
    for pno, page in enumerate(reader.pages, 1):
        text = page.extract_text() or ""
        text = re.sub(r"-\n(?=[a-z])", "", text)                  # hyphenation at line ends
        para: list[str] = []

        def flush():
            nonlocal n
            t = " ".join(para).strip()
            if len(t) > 30:
                n += 1
                kind = "reference" if section.lower().startswith(("reference", "bibliograph")) else "body"
                blocks.append({"block_id": f"{key}-p{pno:02d}-b{n:03d}", "paper": key, "arxiv_id": doc_id,
                               "page": pno, "section": section, "text": t, "kind": kind, "in_appendix": False})
            para.clear()
        for line in text.splitlines():
            s = line.strip()
            if not s:
                flush()
            elif HEADING.match(s) and len(s) < 90:
                flush()
                section = s
            elif s.lower() in ("references", "bibliography"):
                flush()
                section = "References"
            else:
                para.append(s)
                if s.endswith((".", ":")) and len(" ".join(para)) > 400:
                    flush()
        flush()
    return blocks


def add_paper(key: str, title: str, blocks: list[dict], size: int, cfg: dict | None = None):
    """Register a new paper everywhere the code names the library, for this process only."""
    R.PAPERS = P.PAPERS = tuple(R.PAPERS) + (key,)
    R.PAPER_TITLES[key] = title                         # same dict object the pipeline renders sources with
    words = tuple(w for w in re.findall(r"[a-z][a-z0-9-]{3,}", title.lower())
                  if w not in R.STOPWORDS)[:6]
    P._PAPER_WORDS[key] = words
    P.ROUTER_SYSTEM = P.ROUTER_SYSTEM.replace(
        "Reply with JSON only",
        f"  {key}: '{title}'\nReply with JSON only").replace("exactly three", "exactly these")
    with (R.BLOCKS).open(encoding="utf-8") as f:
        base = [json.loads(x) for x in f if x.strip()]
    return R.Retriever(R.build_pieces(base + blocks, size), size)


def show_retrieval(retr: R.Retriever, question: str, k: int, papers):
    hits = retr.search(question, k, papers)
    ev = retr.evidence(question, hits)
    print(f"\nRETRIEVAL PREVIEW (plain BM25 on the question as typed, {retr.size}-char pieces, top {k}"
          + (f", papers={papers}" if papers else "") + ")")
    print("(--real / --standin then run the full pipeline: the router may pick a paper and rewrite the query,"
          " so its sources can differ)")
    dense = f", best cosine {ev['dense_best']:.2f}" if "dense_best" in ev else ""
    print(f"evidence: best piece covers {ev['best']:.0%} of the question's terms (idf-weighted), "
          f"top BM25 {ev['top_bm25']:.1f}{dense}")
    for i, h in enumerate(hits, 1):
        print(f"\n[S{i}] {h.piece.cite()}  ({R.PAPER_TITLES.get(h.piece.paper, h.piece.paper)})  bm25={h.bm25:.1f}")
        print("     " + P.best_excerpt(h.piece.text, question, 500).replace("\n", " "))


def show_reply(reply):
    print("\n" + "=" * 78)
    print(f"REPLY  kind={reply.kind}  level={P.LEVEL_NAMES.get(reply.level, reply.level)}  "
          f"first token {reply.first_token_ms:.0f} ms, done {reply.released_ms:.0f} ms  "
          f"tokens {reply.tokens_in}+{reply.tokens_out}")
    if reply.notice:
        print(f"notice: {reply.notice}")
    router = reply.checks.get("router") or {}
    if router.get("query"):
        print(f"router: papers={router.get('papers')}  search query: {router['query']}")
    print("-" * 78)
    print("ANSWER:\n")
    print(reply.text)
    if reply.claims:
        print("\nCLAIMS (the citation metadata each sentence carries):")
        for c in reply.claims:
            print(f"  - {c['text']}\n      sources {c['sources']}  quote: \"{c['quote']}\"")
    for c in reply.checks.get("dropped") or []:
        print(f"  DROPPED: {c.get('text')}  ({c.get('dropped')})")
    print("\nSOURCES (what each [S#] in the answer points to):")
    for i, s in enumerate(reply.sources, 1):
        print(f"  [S{i}] {s.get('cite', s)}")
    print("-" * 78 + "\nSTEPS (ms from arrival)")
    for sp in reply.spans:
        end = f"{sp.end_ms:8.0f}" if sp.end_ms is not None else "       ?"
        print(f"  {sp.start_ms:8.0f} -{end}  {sp.name:<18} {sp.model or '':<24} {sp.outcome or ''}")
    for c in reply.model_calls:
        print(f"  call: {c.step:<10} {c.model:<24} {c.status} {c.outcome}  in={c.prompt_tokens} out={c.completion_tokens}")


async def run_pipeline(question, retr, mode, cfg, stream=False):
    from relay.llm import Endpoint, LLMClient
    on_claim = None
    if stream:
        def on_claim(c):
            print(f"  >> streamed: {c['text']}  [{', '.join(c['sources'])}]", flush=True)
    lim = Limiter.from_probe()
    if mode == "standin":
        from standin.server import Replay, StandIn, load_calibration
        async with StandIn(load_calibration(), replay=Replay(REPO / "results" / "raw" / "answers"), seed=1) as s:
            pipe = P.Pipeline(P.plain_chat(LLMClient(Endpoint(s.base_url, source="stand-in"))), lim, retr, cfg)
            reply = await pipe.answer(question, on_claim=on_claim)
            print(f"\n(stand-in: replay hits {s.stats['replay_hits']}, misses {s.stats['replay_misses']}; "
                  "a miss means placeholder text, not a real answer)")
            return reply
    from relay.real import open_real
    with open_real(scope="ask", allow_real=True) as bc:
        async def chat(payload, *, phase, timeouts=None, on_delta=None):
            return await bc.stream_chat(payload, phase=phase, timeouts=timeouts, on_delta=on_delta)
        if stream:
            print("\nSTREAMING (each sentence is released once its quote is found in its source):")
        return await P.Pipeline(chat, lim, retr, cfg).answer(question, on_claim=on_claim)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--pdf", type=Path)
    ap.add_argument("--title", default=None)
    ap.add_argument("--key", default="newpaper")
    ap.add_argument("--paper", default=None, help="restrict retrieval to one paper key")
    ap.add_argument("--size", type=int, default=2400)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--legacy", action="store_true", help="the submitted pipeline (BM25, [S#] prose + regex gate)")
    ap.add_argument("--stream", action="store_true", help="release claims one by one as they are verified")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--standin", action="store_true")
    g.add_argument("--real", action="store_true")
    a = ap.parse_args()

    cfg = P.load_config() if a.legacy else P.load_production_config()
    cfg["retrieval"]["piece_chars"] = a.size if a.legacy else cfg["retrieval"]["piece_chars"]
    if a.pdf:
        blocks = pdf_blocks(a.pdf, a.key)
        retr = add_paper(a.key, a.title or a.pdf.stem, blocks, cfg["retrieval"]["piece_chars"], cfg)
        print(f"added {a.pdf.name} as '{a.key}': {len(blocks)} blocks, "
              f"{sum(p.paper == a.key for p in retr.pieces)} pieces of {a.size} chars")
    else:
        from relay.dense import build_retriever
        retr = build_retriever(cfg)
    show_retrieval(retr, a.question, a.k, [a.paper] if a.paper else None)
    if a.standin or a.real:
        reply = asyncio.run(run_pipeline(a.question, retr, "standin" if a.standin else "real", cfg, a.stream))
        show_reply(reply)


if __name__ == "__main__":
    main()
