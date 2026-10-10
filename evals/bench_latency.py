"""Uncached retrieval latency per configuration (the ablation runner caches reranker scores,
so its rerank timings are not representative).

    python -m evals.bench_latency --n 30

Writes evals/results/latency_bench.csv. Measured on whatever machine runs it; the committed
numbers came from a shared Windows workstation (i9-12900KF) with other jobs running, with the
cross-encoder on CPU.
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

from ipqa.config import Settings
from ipqa.gold import load_gold
from ipqa.metrics import pct
from ipqa.retrieval import CrossEncoderReranker
from ipqa.system import get_reranker_encoder, make_retriever

HERE = Path(__file__).resolve().parent


def main(n: int) -> None:
    s = Settings()
    qs = [g.question for g in load_gold(HERE / "gold_set.yaml")][:n]
    rows = []
    for kind in ["bm25", "dense", "hybrid"]:
        for rr in (False, True):
            base = make_retriever(s, "clause", kind, rerank=False)
            r = CrossEncoderReranker(base, encoder=get_reranker_encoder(), cache_path=None) if rr else base
            r.search("warm up query", 5)
            lat = []
            for q in qs:
                if hasattr(base, "retrievers"):          # clear dense query-vector cache
                    for sub in base.retrievers:
                        getattr(sub, "_qcache", {}).clear()
                getattr(base, "_qcache", {}).clear()
                t0 = time.perf_counter()
                r.search(q, 5)
                lat.append(time.perf_counter() - t0)
            row = {"config": f"clause/{kind}{'+rerank' if rr else ''}", "n": len(qs),
                   "p50_ms": round(pct(lat, 50) * 1000, 1), "p95_ms": round(pct(lat, 95) * 1000, 1)}
            rows.append(row)
            print(row, flush=True)
    with (HERE / "results" / "latency_bench.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    main(ap.parse_args().n)
