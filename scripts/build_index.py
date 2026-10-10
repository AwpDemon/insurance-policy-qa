"""Parse the PDFs into clause-level segments (data/processed/segments.jsonl).

    python scripts/build_index.py

Embeddings are computed lazily (and cached in data/cache/) the first time a dense or hybrid
retriever is built, so this step needs no network.
"""
from __future__ import annotations

import collections
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ipqa.chunking import STRATEGIES, build_chunks  # noqa: E402
from ipqa.config import RAW_DIR, SEGMENTS_PATH  # noqa: E402
from ipqa.ingest import ingest  # noqa: E402


def main() -> None:
    segs = ingest(RAW_DIR, SEGMENTS_PATH)
    counts = collections.Counter(s.doc_id for s in segs)
    print(f"{len(segs)} segments -> {SEGMENTS_PATH.relative_to(ROOT)}  {dict(counts)}")
    for st in STRATEGIES:
        print(f"  {st:8s} {len(build_chunks(segs, st))} chunks")


if __name__ == "__main__":
    main()
