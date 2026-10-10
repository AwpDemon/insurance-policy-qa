"""Download the five FEMA source PDFs into data/raw/ and record SHA-256 hashes.

    python scripts/download_corpus.py

The PDFs are U.S. federal government works (public domain in the U.S., 17 U.S.C. 105). They are
not committed to git; this script is the reproducible way to get the exact files used.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ipqa.corpus import DOCS  # noqa: E402

UA = "AwpDemon portfolio aliaskari.georgia@gmail.com"


def main() -> None:
    raw = ROOT / "data" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    manifest = {}
    with httpx.Client(headers={"User-Agent": UA}, follow_redirects=True, timeout=120) as client:
        for d in DOCS:
            path = raw / d.filename
            if not path.exists():
                print(f"downloading {d.doc_id} {d.url}")
                r = client.get(d.url)
                r.raise_for_status()
                if not r.content.startswith(b"%PDF"):
                    raise RuntimeError(f"{d.url} did not return a PDF")
                path.write_bytes(r.content)
                time.sleep(1.0)  # be polite to fema.gov
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            manifest[d.filename] = {"doc_id": d.doc_id, "url": d.url, "sha256": digest,
                                    "bytes": path.stat().st_size}
            print(f"{d.doc_id}: {path.name} {digest[:12]}")
    (ROOT / "data" / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
