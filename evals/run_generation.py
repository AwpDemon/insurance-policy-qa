"""End-to-end evaluation: retrieval + generation on the full gold set for one configuration.

    python -m evals.run_generation --chunking clause --retriever hybrid --rerank --k 5
    python -m evals.run_generation --mock --limit 10        # offline smoke test (CI)

Writes evals/results/gen_<config>.jsonl (every answer, citation, latency, token count) and
appends a summary row to evals/results/generation_summary.csv.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

from ipqa.config import Settings
from ipqa.gold import load_gold
from ipqa.metrics import mean, pct, score_answer
from ipqa.qa import QAPipeline
from ipqa.system import make_llm, make_retriever

HERE = Path(__file__).resolve().parent


def summarize(name: str, scored: list[dict], results: list[dict], model: str) -> dict:
    ans = [r for r in scored if r["gold_answerable"]]
    una = [r for r in scored if not r["gold_answerable"]]
    answered = [r for r in ans if not r["refused"]]
    lat = [r["retrieval_s"] + r["generation_s"] for r in results]
    return {
        "config": name, "model": model, "n": len(scored), "n_answerable": len(ans),
        "n_unanswerable": len(una),
        "answer_accuracy": round(mean(r["correct"] for r in ans), 4),
        "fact_recall": round(mean(r["fact_recall"] for r in ans), 4),
        "citation_precision": round(mean(r.get("citation_precision") for r in answered), 4),
        "citation_recall": round(mean(r.get("citation_recall") for r in answered), 4),
        "answers_with_unsupported_number": round(mean(float(r["unsupported_numbers"] > 0) for r in answered), 4),
        "refusal_accuracy": round(mean(r["refusal_correct"] for r in una), 4),
        "false_refusal_rate": round(mean(r["false_refusal"] for r in ans), 4),
        "overall_accuracy": round(mean(r["correct"] for r in scored), 4),
        "p50_latency_s": round(pct(lat, 50), 3), "p95_latency_s": round(pct(lat, 95), 3),
        "mean_prompt_tokens": round(mean(r["prompt_tokens"] for r in results), 1),
        "mean_completion_tokens": round(mean(r["completion_tokens"] for r in results), 1),
        "parse_errors": sum(r["parse_error"] for r in results),
    }


def run(chunking: str, retriever: str, rerank: bool, k: int, mock: bool = False,
        limit: int | None = None, out_dir: Path = HERE / "results", tag: str = "",
        prompt: str = "v1", gold_file: str = "gold_set.yaml") -> dict:
    s = Settings()
    gold = load_gold(HERE / gold_file)
    if limit:
        gold = gold[:limit]
    pipe = QAPipeline(make_retriever(s, chunking, retriever, rerank), make_llm(s, mock), k=k, prompt=prompt)
    name = f"{chunking}/{retriever}{'+rerank' if rerank else ''}/k{k}{tag}"
    if prompt != "v1":
        name += f"/prompt-{prompt}"
    if gold_file != "gold_set.yaml":
        name += f"/{Path(gold_file).stem}"
    if mock:
        name += "/mock"
    results, scored = [], []
    t0 = time.time()
    for i, g in enumerate(gold, 1):
        res = pipe.answer(g.question).to_dict()
        res["id"] = g.id
        sc = score_answer(g, res)
        results.append(res)
        scored.append(sc)
        if i % 10 == 0:
            print(f"  {name}: {i}/{len(gold)} ({time.time() - t0:.0f}s)", flush=True)
    summary = summarize(name, scored, results, pipe.llm.model)
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = name.replace("/", "_").replace("+", "-")
    with (out_dir / f"gen_{slug}.jsonl").open("w", encoding="utf-8") as f:
        for res, sc in zip(results, scored):
            f.write(json.dumps({**res, "score": sc}, ensure_ascii=False) + "\n")
    summ_path = out_dir / ("generation_summary_mock.csv" if mock else "generation_summary.csv")
    existing = []
    if summ_path.exists():
        with summ_path.open(encoding="utf-8") as f:
            existing = [r for r in csv.DictReader(f) if r["config"] != name]
    with summ_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary))
        w.writeheader()
        for r in existing:
            w.writerow({k: r.get(k, "") for k in summary})
        w.writerow(summary)
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunking", default="clause")
    ap.add_argument("--retriever", default="hybrid")
    ap.add_argument("--rerank", action="store_true")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--tag", default="")
    ap.add_argument("--prompt", default="v1", choices=["v1", "v2"])
    ap.add_argument("--gold", default="gold_set.yaml", help="gold file inside evals/")
    ap.add_argument("--out-dir", type=Path, default=HERE / "results")
    a = ap.parse_args()
    run(a.chunking, a.retriever, a.rerank, a.k, a.mock, a.limit, out_dir=a.out_dir, tag=a.tag, prompt=a.prompt, gold_file=a.gold)
