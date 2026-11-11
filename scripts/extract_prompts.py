"""Extract the author's prompts, verbatim and in order, from Claude Code session transcripts.

    .venv\\Scripts\\python scripts\\extract_prompts.py --transcript <session>.jsonl [--transcript ...] --out submission\\prompts.md

Claude Code keeps each session as a JSONL file under ~/.claude/projects/<project>/. A prompt is:
  * a "user" record whose message content is a string (what the author typed or pasted), except the
    automatic compaction summaries (isCompactSummary) and the harness's background-task notices
    (content starting "<task-notification>");
  * a "queued_command" attachment whose origin is "human": a message the author sent while the assistant
    was working, delivered mid-turn;
  * the author's choice in a multiple-choice question the assistant asked (toolUseResult "answers").
Text is copied as recorded, nothing retyped or edited. Text the author pasted is wrapped by the harness in
<pasted_content id=...> tags; they are kept as recorded.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def prompts(path: Path) -> list[dict]:
    out = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            t = o.get("type")
            content = (o.get("message") or {}).get("content")
            if t == "user" and isinstance(content, str) and not o.get("isCompactSummary") and not o.get("isMeta"):
                if content.lstrip().startswith("<task-notification>"):
                    continue
                out.append({"ts": o.get("timestamp"), "kind": "prompt", "text": content})
            a = o.get("attachment")
            if t == "attachment" and isinstance(a, dict) and a.get("type") == "queued_command" \
                    and (a.get("origin") or {}).get("kind") == "human":
                out.append({"ts": a.get("timestamp") or o.get("timestamp"), "kind": "sent while the assistant was working",
                            "text": a.get("prompt", "")})
            r = o.get("toolUseResult")
            if t == "user" and isinstance(r, dict) and isinstance(r.get("answers"), dict) and r["answers"]:
                for q, ans in r["answers"].items():
                    out.append({"ts": o.get("timestamp"), "kind": "answer to the assistant's question",
                                "text": ans, "question": q})
    return out


def render(items: list[tuple[str, dict]]) -> str:
    L = ["# My prompts to the assistant", "",
         "Extracted by `scripts/extract_prompts.py` from the Claude Code session transcript(s) named below, verbatim and "
         "in time order. Not retyped. `<pasted_content>` tags mark text I pasted (added by the tool, kept as recorded). "
         "Excluded: the tool's automatic context summaries and background-task notices, which I did not write.", ""]
    for src in dict.fromkeys(s for s, _ in items):
        L.append(f"- transcript: `{src}`")
    L.append("")
    for n, (src, p) in enumerate(items, 1):
        L.append(f"## {n}. {p['ts']} ({p['kind']})")
        L.append("")
        if p.get("question"):
            L += ["Question:", "", "~~~~text", p["question"], "~~~~", "", "My answer:", ""]
        L += ["~~~~text", p["text"].strip("\n"), "~~~~", ""]
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--transcript", type=Path, action="append", required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    items = [(t.name, p) for t in a.transcript for p in prompts(t)]
    items.sort(key=lambda x: x[1]["ts"] or "")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(render(items), encoding="utf-8", newline="\n")
    print(f"{len(items)} prompts from {len(a.transcript)} transcript(s) -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
