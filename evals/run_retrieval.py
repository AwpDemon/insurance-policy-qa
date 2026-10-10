"""Retrieval ablation: chunking x retriever x reranking, scored at several k.

    python -m evals.run_retrieval            # full grid (needs Ollama for dense/hybrid)
    python -m evals.run_retrieval --bm25-only  # no network; what CI runs

Writes evals/results/retrieval_ablation.csv and retrieval_per_question.csv.
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

from ipqa.chunking import STRATEGIES
from ipqa.config import Settings
from ipqa.gold import load_gold
from ipqa.metrics import complete_at_k, mean, pct, recall_at_k, reciprocal_rank
from ipqa.system import get_chunks, make_retriever

HERE = Path(__file__).resolve().parent
KS = (1, 3, 5, 10)


def run(bm25_only: bool = False, out_dir: Path = HERE / "results") -> list[dict]:
    s = Settings()
    gold = [g for g in load_gold(HERE / "gold_set.yaml") if g.answerable]
    retrievers = ["bm25"] if bm25_only else ["bm25", "dense", "hybrid"]
    reranks = [False] if bm25_only else [False, True]
    rows, per_q = [], []
    for chunking in STRATEGIES:
        n_chunks = len(get_chunks(chunking))
        for kind in retrievers:
            for rr in reranks:
                # Cross-encoder reranking ran on CPU on a shared machine (~10-30 s/query under load),
                # so it is ablated on the clause chunker only, for all three first-stage retrievers.
                if rr and chunking != "clause":
                    continue
                r = make_retriever(s, chunking=chunking, retriever=kind, rerank=rr)
                name = f"{chunking}/{kind}{'+rerank' if rr else ''}"
                lat, stats = [], {k: [] for k in KS}
                comp5, rrs = [], []
                for gi, g in enumerate(gold, 1):
                    if rr and gi % 20 == 0:
                        print(f"    {name}: {gi}/{len(gold)}", flush=True)
                    t0 = time.perf_counter()
                    hits = r.search(g.question, max(KS))
                    lat.append(time.perf_counter() - t0)
                    ranked = [h.chunk.clause_ids for h in hits]
                    for k in KS:
                        stats[k].append(recall_at_k(ranked, g.supporting, k))
                    comp5.append(complete_at_k(ranked, g.supporting, 5))
                    rr_ = reciprocal_rank(ranked, g.supporting)
                    rrs.append(rr_)
                    per_q.append({"config": name, "id": g.id, "category": g.category,
                                  "recall@5": stats[5][-1], "rr@10": rr_,
                                  "top1": hits[0].chunk.label if hits else ""})
                row = {"config": name, "chunking": chunking, "retriever": kind, "rerank": rr,
                       "n_chunks": n_chunks, "n_questions": len(gold)}
                for k in KS:
                    row[f"recall@{k}"] = round(mean(stats[k]), 4)
                row["complete@5"] = round(mean(comp5), 4)
                row["mrr@10"] = round(mean(rrs), 4)
                row["p50_ms"] = round(pct(lat, 50) * 1000, 1)
                row["p95_ms"] = round(pct(lat, 95) * 1000, 1)
                rows.append(row)
                if hasattr(r, "_flush"):
                    r._flush()
                print(f"{name:28s} R@1={row['recall@1']:.3f} R@3={row['recall@3']:.3f} "
                      f"R@5={row['recall@5']:.3f} R@10={row['recall@10']:.3f} MRR={row['mrr@10']:.3f} "
                      f"p50={row['p50_ms']}ms", flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_bm25" if bm25_only else ""
    with (out_dir / f"retrieval_ablation{suffix}.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with (out_dir / f"retrieval_per_question{suffix}.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(per_q[0]))
        w.writeheader()
        w.writerows(per_q)
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bm25-only", action="store_true")
    ap.add_argument("--out-dir", type=Path, default=HERE / "results")
    a = ap.parse_args()
    run(a.bm25_only, a.out_dir)
