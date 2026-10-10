"""Gold set loading and validation."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

CATEGORIES = {"definition", "coverage", "limit", "exclusion", "condition", "loss_settlement",
              "claims_process", "cross_form", "multi_clause", "trick", "unanswerable"}


@dataclass
class GoldItem:
    id: str
    category: str
    question: str
    answerable: bool
    reference: str
    key_facts: list[str] = field(default_factory=list)
    supporting: list[list[str]] = field(default_factory=list)

    def facts_matched(self, answer: str) -> list[bool]:
        return [re.search(p, answer, flags=re.IGNORECASE) is not None for p in self.key_facts]


def load_gold(path: Path) -> list[GoldItem]:
    with path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return [GoldItem(**r) for r in raw]


def validate_gold(items: list[GoldItem], known_clause_ids: set[str]) -> list[str]:
    """Return a list of problems (empty = valid)."""
    problems = []
    seen = set()
    for it in items:
        if it.id in seen:
            problems.append(f"{it.id}: duplicate id")
        seen.add(it.id)
        if it.category not in CATEGORIES:
            problems.append(f"{it.id}: unknown category {it.category}")
        if it.answerable:
            if not it.supporting:
                problems.append(f"{it.id}: answerable but no supporting clauses")
            if not it.key_facts:
                problems.append(f"{it.id}: answerable but no key facts")
        else:
            if it.supporting or it.key_facts:
                problems.append(f"{it.id}: unanswerable items must not have evidence/key facts")
            if it.category != "unanswerable":
                problems.append(f"{it.id}: unanswerable item must use category 'unanswerable'")
        for group in it.supporting:
            for cid in group:
                if cid not in known_clause_ids:
                    problems.append(f"{it.id}: unknown clause id {cid}")
        for pat, ok in zip(it.key_facts, it.facts_matched(it.reference)):
            if not ok:
                problems.append(f"{it.id}: key fact /{pat}/ does not match its own reference answer")
    return problems
