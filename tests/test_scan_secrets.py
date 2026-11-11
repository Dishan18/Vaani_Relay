"""The secret scan finds what it must (a planted fake key, forbidden files, inside zips) and nothing else."""
from __future__ import annotations

import io
import sys
import unittest
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import scan_secrets as ss  # noqa: E402

FAKE = ("gsk" + "_" + "Zx" * 20).encode()          # built at runtime so this file holds no key-shaped text


class ScanTests(unittest.TestCase):
    def test_a_key_anywhere_is_found_without_printing_it(self):
        f = []
        ss.scan_file("notes.md", b"before " + FAKE + b" after", [], f)
        self.assertEqual(f, ["notes.md: a Groq API key (gsk_...)"])
        self.assertNotIn(FAKE.decode(), " ".join(f))

    def test_known_secret_and_fingerprint(self):
        f = []
        ss.scan_file("a.txt", b"x 0123456789abcdef y", [("fingerprint of key1", b"0123456789abcdef")], f)
        self.assertEqual(f, ["a.txt: fingerprint of key1"])

    def test_forbidden_files(self):
        for name in (".env", "sub/.env.local", "results/ledger/accounts.local.json"):
            f = []
            ss.scan_file(name, b"", [], f)
            self.assertEqual(f, [f"{name}: forbidden file"], name)
        f = []
        ss.scan_file(".env.example", b"GROQ_API_KEY=", [], f)
        self.assertEqual(f, [])

    def test_inside_a_zip(self):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z:
            z.writestr("code/.env", "GROQ_API_KEY=" + FAKE.decode())
        f = []
        ss.scan_file("submission/code.zip", buf.getvalue(), [], f)
        self.assertIn("submission/code.zip!code/.env: forbidden file", f)
        self.assertIn("submission/code.zip!code/.env: a Groq API key (gsk_...)", f)


if __name__ == "__main__":
    unittest.main()
