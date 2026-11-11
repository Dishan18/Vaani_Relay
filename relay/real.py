"""Factory for the real Groq client. The only place the API key is read.

    async with/with open_real(scope="probe", allow_real=args.real) as groq:
        rec = await groq.stream_chat(payload, phase="P0")

Refuses unless ``allow_real`` is True, so a script makes real calls only when run with an
explicit flag. Everything else (tests, development) uses the stand-in.
"""
from __future__ import annotations

import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator

from relay.ledger import DEFAULT_LEDGER, BudgetedClient, Ledger, load_budget
from relay.llm import Endpoint, LLMClient, Timeouts

ROOT = Path(__file__).resolve().parents[1]
GROQ_BASE_URL = "https://api.groq.com/openai/v1"


def read_env_file(path: Path) -> dict[str, str]:
    """Minimal KEY=VALUE reader for .env (no dependency). Values are never logged."""
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip().strip('"').strip("'")
    return out


ACCOUNTS = ROOT / "results" / "ledger" / "accounts.local.json"   # gitignored: fingerprint -> label


def account_label(key: str | None = None, path: Path = ACCOUNTS) -> str:
    """A stable label (key2, key3, ...) for the API key in use, for the ledger's per-account caps.
    The key is never written anywhere: only a short SHA-256 fingerprint, kept in a gitignored local
    file, maps to the label. The ledger's entries from before labels existed belong to "key1" (the
    key in use until 2026-11-11), so the first key seen here is labelled key2."""
    import hashlib
    key = key or groq_api_key()
    fp = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
    known = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    if fp not in known:
        known[fp] = f"key{len(known) + 2}"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(known, indent=1) + "\n", encoding="utf-8")
    return known[fp]


def groq_api_key() -> str:
    key = os.environ.get("GROQ_API_KEY") or read_env_file(ROOT / ".env").get("GROQ_API_KEY")
    if not key:
        raise RuntimeError("GROQ_API_KEY not set (environment or .env)")
    return key


@contextmanager
def open_real(scope: str, allow_real: bool, timeouts: Timeouts = Timeouts(),
              raw_sink: Callable[[dict[str, Any]], None] | None = None,
              ledger_path: Path = DEFAULT_LEDGER) -> Iterator[BudgetedClient]:
    if not allow_real:
        raise RuntimeError("real Groq calls need an explicit opt-in (run the script with --real)")
    with Ledger(load_budget(), ledger_path, account=account_label()) as ledger:
        client = LLMClient(Endpoint(GROQ_BASE_URL, api_key=groq_api_key(), source="real"), timeouts)
        yield BudgetedClient(client, ledger, scope, raw_sink)
