import math

from ipqa.chunking import STRATEGIES, ancestors, build_chunks, parent_id
from ipqa.gold import GoldItem
from ipqa.metrics import (complete_at_k, recall_at_k, reciprocal_rank, satisfies, score_answer,
                          unsupported_numbers)


def test_parent_and_ancestors():
    assert parent_id("DF:V.D.5") == "DF:V.D"
    assert parent_id("DF:X") is None
    assert ancestors("DF:V.D.5") == ["DF:V.D", "DF:V"]


def test_every_strategy_covers_every_segment(segments):
    all_ids = {s.clause_id for s in segments}
    for st in STRATEGIES:
        covered = set()
        for c in build_chunks(segments, st):
            covered.update(c.clause_ids)
        missing = all_ids - covered
        # tiny header-only segments ("N. Mortgage Clause") may be absorbed; real text may not
        long_missing = [m for m in missing
                        if len(next(s for s in segments if s.clause_id == m).text.split()) > 12]
        assert not long_missing, (st, long_missing[:5])


def test_clause_chunks_carry_lead_in_and_breadcrumb(segments):
    chunks = build_chunks(segments, "clause")
    sinkhole = next(c for c in chunks if "DF:V.C.4" in c.clause_ids)
    assert "earth movement" in sinkhole.text          # parent lead-in sentence is prepended
    assert sinkhole.text.startswith("[Dwelling Form] V. EXCLUSIONS")
    assert max(len(c.text.split()) for c in chunks) < 350


def test_fixed_chunks_have_no_headers(segments):
    c = build_chunks(segments, "fixed")[0]
    assert not c.text.startswith("[")


def test_satisfies_accepts_descendants_not_ancestors():
    assert satisfies(["DF:VII.N.6"], ["DF:VII.N"])
    assert not satisfies(["DF:VII.N"], ["DF:VII.N.6"])
    assert not satisfies(["DF:VII.N1"], ["DF:VII.N"])  # prefix must end at a dot


def test_retrieval_metrics():
    ranked = [["DF:I.A"], ["DF:V.D.5"], ["GP:III.C.3"]]
    groups = [["DF:V.D.5"], ["GP:III.C.3", "XX:1"]]
    assert recall_at_k(ranked, groups, 1) == 0
    assert recall_at_k(ranked, groups, 2) == 0.5
    assert recall_at_k(ranked, groups, 3) == 1
    assert complete_at_k(ranked, groups, 2) == 0 and complete_at_k(ranked, groups, 3) == 1
    assert reciprocal_rank(ranked, groups) == 0.5
    assert math.isnan(recall_at_k(ranked, [], 3))


def test_unsupported_numbers():
    ev = "We will pay no more than $2,500 for any one loss within 60 days."
    assert unsupported_numbers("The limit is $2,500.", ev) == []
    assert unsupported_numbers("The limit is $5,000 within 60 days.", ev) == ["$5,000"]
    assert unsupported_numbers("Within two years.", "not to exceed two years") == []


def _item(answerable=True):
    return GoldItem("q", "limit" if answerable else "unanswerable", "Jewelry limit?", answerable,
                    "ref", [r"2,?500"] if answerable else [], [["DF:III.B.8"]] if answerable else [])


def _result(answer, cites, answerable=True):
    return {"question": "Jewelry limit?", "answer": answer, "answerable": answerable,
            "citations": [{"n": n, "clause_ids": ids} for n, ids in cites],
            "retrieved": [{"rank": 1, "text": "no more than $2,500"}, {"rank": 2, "text": "other"}]}


def test_score_answer_correct_with_good_citation():
    row = score_answer(_item(), _result("$2,500 per loss.", [(1, ["DF:III.B.8"])]))
    assert row["correct"] == 1 and row["citation_precision"] == 1 and row["unsupported_numbers"] == 0


def test_score_answer_wrong_number_and_bad_citation():
    row = score_answer(_item(), _result("$5,000 per loss.", [(2, ["DF:IV.1"])]))
    assert row["correct"] == 0 and row["citation_precision"] == 0 and row["unsupported_numbers"] == 1


def test_score_answer_refusals():
    assert score_answer(_item(False), _result("Not covered in the documents.", [], False))["correct"] == 1
    assert score_answer(_item(False), _result("$100,000.", []))["correct"] == 0
    row = score_answer(_item(), _result("Not covered in the documents.", [], False))
    assert row["false_refusal"] == 1 and row["correct"] == 0
