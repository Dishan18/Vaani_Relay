"""Download the three library papers at exact arXiv versions and write a sha256 manifest.

Checks, in order, for each paper:
  1. The arXiv API (export.arxiv.org) returns an entry whose id ends in the exact version.
  2. The API title matches the title the brief names (whitespace/case-insensitive).
  3. The downloaded bytes are a PDF (magic number) from the versioned URL.
Then writes data/papers/MANIFEST.json.

If the manifest already pins a sha256 and the local file matches, nothing is downloaded.
If a fresh download does not match a pinned sha256, the script fails loudly instead of
overwriting the pin.

Stdlib only (urllib). This is a one-off batch of 4 HTTP requests, so asyncio buys nothing;
arXiv asks API clients to pause ~3 s between calls, which we do.

Usage (PowerShell):  .venv\\Scripts\\python scripts\\fetch_papers.py
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER_DIR = ROOT / "data" / "papers"
MANIFEST = PAPER_DIR / "MANIFEST.json"

# Exactly what the brief specifies.
PAPERS = [
    {"key": "attention", "arxiv_id": "1706.03762", "version": 7,
     "title": "Attention Is All You Need"},
    {"key": "moshi", "arxiv_id": "2410.00037", "version": 1,
     "title": "Moshi: a speech-text foundation model for real-time dialogue"},
    {"key": "dynamic_chunking", "arxiv_id": "2507.07955", "version": 2,
     "title": "Dynamic Chunking for End-to-End Hierarchical Sequence Modeling"},
]

API_URL = "https://export.arxiv.org/api/query?id_list={ids}&max_results=10"
PDF_URL = "https://arxiv.org/pdf/{arxiv_id}v{version}"
USER_AGENT = "relay-assignment-fetcher/1.0"  # generic; no personal details
ATOM = {"a": "http://www.w3.org/2005/Atom"}
POLITE_DELAY_S = 3.0


def _get(url: str) -> tuple[bytes, str]:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read(), resp.headers.get("Content-Type", "")


def _norm(title: str) -> str:
    return re.sub(r"\s+", " ", title).strip().casefold()


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def query_api() -> dict[str, dict]:
    ids = ",".join(f"{p['arxiv_id']}v{p['version']}" for p in PAPERS)
    body, _ = _get(API_URL.format(ids=ids))
    root = ET.fromstring(body)
    found: dict[str, dict] = {}
    for entry in root.findall("a:entry", ATOM):
        entry_id = entry.findtext("a:id", default="", namespaces=ATOM)
        m = re.search(r"abs/(\d{4}\.\d{4,5})v(\d+)$", entry_id)
        if not m:
            continue
        found[m.group(1)] = {
            "api_id": entry_id,
            "api_version": int(m.group(2)),
            "api_title": re.sub(r"\s+", " ", entry.findtext("a:title", "", ATOM)).strip(),
            "api_updated": entry.findtext("a:updated", "", ATOM),
        }
    return found


def main() -> int:
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    pinned = {}
    if MANIFEST.exists():
        pinned = {p["arxiv_id"]: p for p in json.loads(MANIFEST.read_text(encoding="utf-8"))["papers"]}

    api = query_api()
    errors: list[str] = []
    out = []
    for p in PAPERS:
        aid, ver = p["arxiv_id"], p["version"]
        meta = api.get(aid)
        if meta is None:
            errors.append(f"{aid}: not returned by arXiv API")
            continue
        if meta["api_version"] != ver:
            errors.append(f"{aid}: API returned v{meta['api_version']}, expected v{ver}")
        if _norm(meta["api_title"]) != _norm(p["title"]):
            errors.append(f"{aid}: API title {meta['api_title']!r} != brief title {p['title']!r}")

        path = PAPER_DIR / f"{aid}v{ver}.pdf"
        url = PDF_URL.format(arxiv_id=aid, version=ver)
        pin = pinned.get(aid, {}).get("sha256")
        if path.exists() and pin and sha256(path.read_bytes()) == pin:
            data = path.read_bytes()
            print(f"{aid}v{ver}: local file matches pinned sha256, not downloading")
        else:
            time.sleep(POLITE_DELAY_S)
            data, ctype = _get(url)
            if not data.startswith(b"%PDF"):
                errors.append(f"{aid}: download from {url} is not a PDF (Content-Type {ctype!r})")
                continue
            if pin and sha256(data) != pin:
                errors.append(f"{aid}: downloaded sha256 {sha256(data)} != pinned {pin}; not overwriting")
                continue
            path.write_bytes(data)
            print(f"{aid}v{ver}: downloaded {len(data)} bytes")

        out.append({
            "key": p["key"],
            "arxiv_id": aid,
            "version": ver,
            "title": meta["api_title"],
            "api_id": meta["api_id"],
            "api_updated": meta["api_updated"],
            "url": url,
            "file": path.relative_to(ROOT).as_posix(),
            "bytes": len(data),
            "sha256": sha256(data),
        })

    if errors:
        for e in errors:
            print("ERROR:", e, file=sys.stderr)
        return 1

    manifest = {
        "generated_by": "scripts/fetch_papers.py",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "papers": out,
    }
    if pinned and all(pinned.get(p["arxiv_id"], {}).get("sha256") == p["sha256"] for p in out):
        print("manifest unchanged (all sha256 match pins)")
        return 0
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {MANIFEST.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
