"""Is the API key in .env fresh on Groq's side? One tiny call per model, through the ledger.

    .venv\\Scripts\\python scripts\\check_key.py

Groq's limits belong to the key's organization, so a new key in the same organization would still
carry its daily usage. The header x-ratelimit-remaining-requests (daily requests) shows it: near the
limit means unused in the last 24 h. Prints the ledger's account label, never the key.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from relay.real import account_label, open_real  # noqa: E402

MODELS = {"openai/gpt-oss-120b": {"reasoning_effort": "low"}, "openai/gpt-oss-20b": {"reasoning_effort": "low"},
          "qwen/qwen3.8-27b": {"reasoning_effort": "none"}}


async def main_async() -> int:
    print(f"ledger account: {account_label()}")
    fresh = True
    with open_real(scope="preflight", allow_real=True) as bc:
        for m, settings in MODELS.items():
            rec = await bc.stream_chat({"model": m, "messages": [{"role": "user", "content": "Reply with OK."}],
                                        "max_tokens": 32, **settings}, phase="key-check")
            lim, rem = rec.header("x-ratelimit-limit-requests"), rec.header("x-ratelimit-remaining-requests")
            used = (int(lim) - int(rem)) if lim and rem else None
            ok = used is not None and used <= 10
            fresh &= ok
            print(f"{m:<22} status {rec.status}  daily requests remaining {rem}/{lim}  "
                  f"(used ~{used})  tokens/min remaining {rec.header('x-ratelimit-remaining-tokens')}  "
                  f"-> {'fresh' if ok else 'NOT fresh'}")
    print("key is fresh on all three models" if fresh else "key is NOT fresh: its organization has daily usage")
    return 0 if fresh else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main_async()))
