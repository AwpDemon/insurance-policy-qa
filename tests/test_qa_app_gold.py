import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from ipqa.chunking import build_chunks
from ipqa.gold import load_gold, validate_gold
from ipqa.llm import LLMResponse, MockLLM
from ipqa.qa import REFUSAL, QAPipeline, parse_response
from ipqa.retrieval import BM25Retriever, HybridRetriever, tokenize

GOLD = Path(__file__).resolve().parent.parent / "evals" / "gold_set.yaml"


def test_gold_set_is_valid(segments):
    items = load_gold(GOLD)
    assert len(items) >= 80
    assert sum(not i.answerable for i in items) >= 10
    assert validate_gold(items, {s.clause_id for s in segments}) == []


def test_tokenize_keeps_money_and_stems():
    toks = tokenize("We will pay $2,500 for flooded basements")
    assert "$2,500" in toks and "basement" in toks and "we" not in toks


@pytest.fixture(scope="module")
def bm25(segments):
    return BM25Retriever(build_chunks(segments, "clause"))


def test_bm25_finds_obvious_clause(bm25):
    hits = bm25.search("detached garage 10 percent of the limit of liability", 3)
    assert any("DF:III.A.3" in h.chunk.clause_ids for h in hits)


def test_hybrid_rrf_merges_rankings(bm25):
    class Fixed:
        name = "fixed"

        def __init__(self, hits):
            self.hits = hits

        def search(self, q, k):
            return self.hits[:k]

    a = bm25.search("sewer backup", 5)
    b = list(reversed(a))
    for i, h in enumerate(b, 1):
        h.rank = i
    fused = HybridRetriever([Fixed(a), Fixed(b)]).search("x", 5)
    assert len(fused) == 5 and len({h.chunk.chunk_id for h in fused}) == 5


def test_parse_response_tolerates_fences():
    d, err = parse_response('```json\n{"answer": "x", "citations": [1], "answerable": true}\n```')
    assert not err and d["citations"] == [1]
    d, err = parse_response("plain text")
    assert err and d["answer"] == "plain text"


def test_pipeline_with_mock_llm_cites_retrieved_clause(bm25):
    res = QAPipeline(bm25, MockLLM(), k=5).answer("How much coverage applies to a detached garage?")
    assert res.answerable and res.citations
    assert res.citations[0].label


class _BadCiteLLM:
    model = "bad"

    def complete(self, system, user, json_mode=True):
        return LLMResponse('{"answer": "Not covered in the documents.", "citations": [99], '
                           '"answerable": false}', 10, 5, 0.01, "bad")


def test_refusal_drops_citations_and_out_of_range_numbers(bm25):
    res = QAPipeline(bm25, _BadCiteLLM(), k=3).answer("What is the premium?")
    assert res.answer == REFUSAL and not res.answerable and res.citations == []


def test_api_end_to_end(bm25):
    os.environ["IPQA_RATE_LIMIT_PER_MIN"] = "1000"
    from app import main
    main.set_pipeline(QAPipeline(bm25, MockLLM(), k=3))
    client = TestClient(main.app)
    assert client.get("/api/health").json()["status"] == "ok"
    assert len(client.get("/api/documents").json()) == 5
    r = client.post("/api/ask", json={"question": "Is sewer backup through drains covered?"})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] and len(body["excerpts"]) == 3
    assert client.post("/api/ask", json={"question": "x"}).status_code == 422
    assert client.get("/").status_code == 200
    main.set_pipeline(None)


def test_prompt_versions_and_flag_only_refusal(bm25):
    class FlagOnly:
        model = "flag"

        def complete(self, system, user, json_mode=True):
            assert "An exclusion, limitation, or condition IS an answer" in system
            return LLMResponse('{"answer": "", "citations": [], "answerable": false}', 1, 1, 0.0, "flag")

    res = QAPipeline(bm25, FlagOnly(), k=2, prompt="v2").answer("What is the premium?")
    assert res.answer == REFUSAL and not res.answerable
    with pytest.raises(KeyError):
        QAPipeline(bm25, MockLLM(), prompt="v9")
