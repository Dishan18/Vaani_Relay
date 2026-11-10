"""Factory for the real Groq client. The only place the API key is read.

    async with/with open_real(scope="probe", allow_real=args.real) as groq:
        rec = await groq.stream_chat(payload, phase="P0")

Refuses unless ``allow_real`` is True, so a script makes real calls only when run with an
explicit flag. Everything else (tests, development) uses the stand-in.
"""
from __future__ import annotations

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
    with Ledger(load_budget(), ledger_path) as ledger:
        client = LLMClient(Endpoint(GROQ_BASE_URL, api_key=groq_api_key(), source="real"), timeouts)
        yield BudgetedClient(client, ledger, scope, raw_sink)
