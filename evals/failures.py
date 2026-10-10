"""Print every failed item of a generation run with what is needed to diagnose it:
was the gold evidence retrieved (retrieval failure) or not (generation failure)?

    python -m evals.failures evals/results/gen_clause_hybrid_k5.jsonl
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

from ipqa.gold import load_gold
from ipqa.metrics import satisfies

HERE = Path(__file__).resolve().parent


def main(path: str) -> None:
    gold = {g.id: g for g in load_gold(HERE / "gold_set.yaml")}
    kinds = Counter()
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        sc = r["score"]
        if sc["correct"] == 1.0:
            continue
        g = gold[r["id"]]
        if not g.answerable:
            kind = "answered_unanswerable"
        else:
            retrieved = all(any(satisfies(x["clause_ids"], grp) for x in r["retrieved"]) for grp in g.supporting)
            if sc["refused"]:
                kind = "false_refusal(evidence retrieved)" if retrieved else "false_refusal(evidence missing)"
            else:
                kind = "wrong_answer(evidence retrieved)" if retrieved else "wrong_answer(evidence missing)"
        kinds[kind] += 1
        print(f"--- {r['id']} [{g.category}] {kind}")
        print(f"Q: {g.question}")
        print(f"A: {r['answer']}")
        print(f"REF: {g.reference}")
        print(f"cited: {[c['label'] for c in r['citations']]}")
        print(f"retrieved: {[x['label'] for x in r['retrieved']]}")
    print("\nFailure types:", dict(kinds))


if __name__ == "__main__":
    main(sys.argv[1])
