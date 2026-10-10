"""Turn evals/results/*.csv into PNG charts in docs/ and a markdown results file.

    python -m evals.make_report
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
DOCS = HERE.parent / "docs"

# Validated categorical slots (blue, orange, aqua) + neutral inks.
C = {"bm25": "#2a78d6", "dense": "#eb6834", "hybrid": "#1baf7a"}
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": GRID,
    "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
})


def read(name: str) -> list[dict]:
    with (RES / name).open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


CHART_CONFIGS = ["fixed/bm25/k5", "clause/hybrid/k5", "clause/bm25+rerank/k5",
                 "clause/bm25+rerank/k5/prompt-v2"]


def fig_retrieval(rows: list[dict]) -> None:
    chunkings = ["fixed", "section", "clause"]
    kinds = ["bm25", "dense", "hybrid"]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    w = 0.26
    for j, kind in enumerate(kinds):
        vals = []
        for ch in chunkings:
            r = next(x for x in rows if x["chunking"] == ch and x["retriever"] == kind and x["rerank"] == "False")
            vals.append(float(r["recall@5"]))
        xs = [i + (j - 1) * w for i in range(len(chunkings))]
        bars = ax.bar(xs, vals, width=w - 0.03, color=C[kind], label=kind.upper() if kind == "bm25" else kind)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", va="bottom",
                    fontsize=8, color=MUTED)
    # Cross-encoder reranking was ablated on clause chunks only: diamonds above each clause bar.
    ci = chunkings.index("clause")
    for j, kind in enumerate(kinds):
        r = next((x for x in rows if x["chunking"] == "clause" and x["retriever"] == kind
                  and x["rerank"] == "True"), None)
        if r is None:
            continue
        v = float(r["recall@5"])
        ax.scatter([ci + (j - 1) * w], [v], marker="D", s=42, color=INK, zorder=3,
                   label="+ cross-encoder rerank" if j == 0 else None)
        ax.text(ci + (j - 1) * w + 0.05, v + 0.015, f"{v:.2f}", fontsize=8, color=INK)
    ax.set_xticks(range(len(chunkings)))
    ax.set_xticklabels(["Fixed 200-word windows", "Section (lettered paragraph)", "Clause (structure-aware)"])
    ax.set_ylabel("Recall@5 (share of gold clauses retrieved)")
    ax.set_ylim(0, 1.05)
    ax.set_title("Retrieval: chunking strategy x retriever (105 answerable questions)", loc="left", fontsize=11)
    ax.legend(frameon=False, ncol=4, loc="upper left", bbox_to_anchor=(0, -0.1), fontsize=9)
    fig.tight_layout()
    fig.savefig(DOCS / "retrieval_ablation.png", dpi=160)
    plt.close(fig)


def fig_recall_k(rows: list[dict]) -> None:
    ks = [1, 3, 5, 10]
    show = [("fixed/bm25", "Fixed + BM25 (baseline)", MUTED, "--"),
            ("clause/dense", "Clause + dense", C["dense"], "-"),
            ("clause/bm25", "Clause + BM25", C["bm25"], "-"),
            ("clause/bm25+rerank", "Clause + BM25 + rerank (selected)", INK, "-")]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for cfg, label, color, ls in show:
        r = next((x for x in rows if x["config"] == cfg), None)
        if r is None:
            continue
        ys = [float(r[f"recall@{k}"]) for k in ks]
        ax.plot(ks, ys, ls, color=color, lw=2, marker="o", ms=6, label=label)
    ax.set_xticks(ks)
    ax.set_xlabel("k (chunks passed to the generator)")
    ax.set_ylabel("Recall@k")
    ax.set_ylim(0, 1.0)
    ax.set_title("Recall@k", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(DOCS / "recall_at_k.png", dpi=160)
    plt.close(fig)


def fig_generation(rows: list[dict]) -> None:
    metrics = [("answer_accuracy", "Answer accuracy\n(answerable)"),
               ("citation_precision", "Citation precision"),
               ("refusal_accuracy", "Refusal accuracy\n(unanswerable)"),
               ("false_refusal_rate", "False-refusal rate\n(lower is better)")]
    palette = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    n = len(rows)
    w = 0.8 / max(n, 1)
    for j, r in enumerate(rows):
        xs = [i + (j - (n - 1) / 2) * w for i in range(len(metrics))]
        vals = [float(r[m]) for m, _ in metrics]
        bars = ax.bar(xs, vals, width=w - 0.02, color=palette[j % len(palette)], label=r["config"])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", va="bottom",
                    fontsize=7, color=MUTED)
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels([lbl for _, lbl in metrics])
    ax.set_ylim(0, 1.08)
    ax.set_title("End-to-end (qwen2.5:7b-instruct, 120 questions)", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper left", bbox_to_anchor=(0, -0.16))
    fig.tight_layout()
    fig.savefig(DOCS / "generation_metrics.png", dpi=160)
    plt.close(fig)


def category_table(gen_file: Path) -> list[tuple[str, int, float]]:
    by = defaultdict(list)
    for line in gen_file.read_text(encoding="utf-8").splitlines():
        sc = json.loads(line)["score"]
        by[sc["category"]].append(sc["correct"])
    return sorted(((k, len(v), sum(v) / len(v)) for k, v in by.items()), key=lambda t: t[2])


def main() -> None:
    DOCS.mkdir(exist_ok=True)
    out = []
    if (RES / "retrieval_ablation.csv").exists():
        rows = read("retrieval_ablation.csv")
        fig_retrieval(rows)
        fig_recall_k(rows)
        out.append("## Retrieval ablation\n")
        out.append("| config | chunks | R@1 | R@3 | R@5 | R@10 | complete@5 | MRR@10 | p50 ms |")
        out.append("|---|---|---|---|---|---|---|---|---|")
        for r in sorted(rows, key=lambda r: -float(r["recall@5"])):
            out.append(f"| {r['config']} | {r['n_chunks']} | {float(r['recall@1']):.3f} | {float(r['recall@3']):.3f} | "
                       f"{float(r['recall@5']):.3f} | {float(r['recall@10']):.3f} | {float(r['complete@5']):.3f} | "
                       f"{float(r['mrr@10']):.3f} | {r['p50_ms']} |")
    if (RES / "generation_summary.csv").exists():
        rows = read("generation_summary.csv")
        chart = [r for r in rows if r["config"] in CHART_CONFIGS]
        chart.sort(key=lambda r: CHART_CONFIGS.index(r["config"]))
        fig_generation(chart)
        out.append("\n## End-to-end generation\n")
        cols = ["answer_accuracy", "fact_recall", "citation_precision", "citation_recall",
                "answers_with_unsupported_number", "refusal_accuracy", "false_refusal_rate",
                "overall_accuracy", "p50_latency_s", "p95_latency_s", "mean_prompt_tokens",
                "mean_completion_tokens", "parse_errors"]
        out.append("| config | " + " | ".join(cols) + " |")
        out.append("|---|" + "---|" * len(cols))
        for r in rows:
            out.append(f"| {r['config']} | " + " | ".join(r[c] for c in cols) + " |")
        for r in chart:
            f = RES / f"gen_{r['config'].replace('/', '_').replace('+', '-')}.jsonl"
            if f.exists():
                out.append(f"\n### Accuracy by category: {r['config']}\n")
                out.append("| category | n | accuracy |\n|---|---|---|")
                for cat, n, acc in category_table(f):
                    out.append(f"| {cat} | {n} | {acc:.2f} |")
    if (RES / "latency_bench.csv").exists():
        out.append("\n## Uncached retrieval latency (clause chunks)\n")
        out.append("| config | n | p50 ms | p95 ms |\n|---|---|---|---|")
        for r in read("latency_bench.csv"):
            out.append(f"| {r['config']} | {r['n']} | {r['p50_ms']} | {r['p95_ms']} |")
    (RES / "RESULTS.md").write_text("# Results (generated by evals/make_report.py)\n\n" + "\n".join(out) + "\n",
                                    encoding="utf-8")
    print((RES / "RESULTS.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
