"""Evaluation metrics for retrieval and generation."""
from __future__ import annotations

import re
import statistics

from .gold import GoldItem


def satisfies(clause_ids: list[str], group: list[str]) -> bool:
    """A chunk satisfies an evidence group if it covers any acceptable clause id *or a
    descendant of one* (covering DF:VII.N.6 satisfies a gold label of DF:VII.N)."""
    for g in group:
        for c in clause_ids:
            if c == g or c.startswith(g + "."):
                return True
    return False


# --------------------------------------------------------------------------- retrieval
def recall_at_k(ranked_clause_ids: list[list[str]], groups: list[list[str]], k: int) -> float:
    if not groups:
        return float("nan")
    top = ranked_clause_ids[:k]
    hit = sum(1 for g in groups if any(satisfies(c, g) for c in top))
    return hit / len(groups)


def complete_at_k(ranked_clause_ids: list[list[str]], groups: list[list[str]], k: int) -> float:
    """1.0 only if every evidence group is found in the top k (matters for multi-clause items)."""
    return float(recall_at_k(ranked_clause_ids, groups, k) == 1.0)


def reciprocal_rank(ranked_clause_ids: list[list[str]], groups: list[list[str]]) -> float:
    for i, c in enumerate(ranked_clause_ids, 1):
        if any(satisfies(c, g) for g in groups):
            return 1.0 / i
    return 0.0


# --------------------------------------------------------------------------- generation
_NUM_RE = re.compile(r"\$?\d[\d,]*(?:\.\d+)?")


def _norm_num(s: str) -> str:
    return s.replace("$", "").replace(",", "").rstrip(".")


def unsupported_numbers(answer: str, evidence: str, question: str = "") -> list[str]:
    """Numbers in the answer that appear neither in the cited evidence nor in the question.
    A cheap, deterministic hallucination detector for the facts that matter most in insurance
    answers (limits, days, percentages)."""
    have = {_norm_num(x) for x in _NUM_RE.findall(evidence + " " + question)}
    words = {"one": "1", "two": "2", "three": "3", "five": "5", "six": "6", "ten": "10"}
    for w, d in words.items():
        if re.search(rf"\b{w}\b", evidence, re.I):
            have.add(d)
    out = []
    for x in _NUM_RE.findall(answer):
        n = _norm_num(x)
        if n and n not in have:
            out.append(x)
    return out


def score_answer(item: GoldItem, result: dict) -> dict:
    refused = not result["answerable"]
    cites = result["citations"]
    row = {"id": item.id, "category": item.category, "gold_answerable": item.answerable,
           "refused": refused}
    if not item.answerable:
        row.update(correct=float(refused), refusal_correct=float(refused))
        row["unsupported_numbers"] = 0 if refused else len(_NUM_RE.findall(result["answer"]))
        return row
    facts = item.facts_matched(result["answer"]) if not refused else [False] * len(item.key_facts)
    row["fact_recall"] = sum(facts) / len(facts) if facts else 0.0
    row["correct"] = float(all(facts) and not refused)
    row["false_refusal"] = float(refused)
    if cites:
        good = [any(satisfies(c["clause_ids"], g) for g in item.supporting) for c in cites]
        row["citation_precision"] = sum(good) / len(good)
        row["citation_recall"] = sum(
            1 for g in item.supporting if any(satisfies(c["clause_ids"], g) for c in cites)) / len(item.supporting)
    elif not refused:
        row["citation_precision"] = 0.0
        row["citation_recall"] = 0.0
    if not refused:
        cited_text = " ".join(r["text"] for r in result["retrieved"]
                              if any(r["rank"] == c["n"] for c in cites))
        bad = unsupported_numbers(result["answer"], cited_text, result["question"])
        row["unsupported_numbers"] = len(bad)
        row["has_citation"] = float(bool(cites))
    return row


def mean(xs) -> float:
    xs = [x for x in xs if x is not None and x == x]
    return statistics.fmean(xs) if xs else float("nan")


def pct(xs, p: float) -> float:
    xs = sorted(xs)
    if not xs:
        return float("nan")
    i = min(len(xs) - 1, max(0, round(p / 100 * (len(xs) - 1))))
    return xs[i]
