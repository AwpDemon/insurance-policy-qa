"""Factory that wires corpus -> chunks -> retriever -> QA pipeline from Settings."""
from __future__ import annotations

from functools import lru_cache

from .chunking import Chunk, build_chunks
from .config import CACHE_DIR, SEGMENTS_PATH, Settings
from .ingest import load_segments
from .llm import LLM, MockLLM, OpenAICompatibleLLM
from .qa import QAPipeline
from .retrieval import OllamaEmbedder, Retriever, build_retriever


@lru_cache(maxsize=8)
def get_chunks(strategy: str) -> tuple[Chunk, ...]:
    return tuple(build_chunks(load_segments(SEGMENTS_PATH), strategy))


_ENCODER = None


def get_reranker_encoder():
    global _ENCODER
    if _ENCODER is None:
        from fastembed.rerank.cross_encoder import TextCrossEncoder
        _ENCODER = TextCrossEncoder(model_name="Xenova/ms-marco-MiniLM-L-12-v2", threads=8)
    return _ENCODER


def make_retriever(s: Settings, chunking: str | None = None, retriever: str | None = None,
                   rerank: bool | None = None) -> Retriever:
    chunks = list(get_chunks(chunking or s.chunking))
    kind = retriever or s.retriever
    embedder = OllamaEmbedder(s.embed_base_url, s.embed_model, CACHE_DIR) if kind != "bm25" else None
    rr = s.rerank if rerank is None else rerank
    return build_retriever(kind, chunks, embedder, rerank=rr,
                           encoder=get_reranker_encoder() if rr else None,
                           rerank_cache=CACHE_DIR / "rerank_scores.json")


def make_llm(s: Settings, mock: bool = False) -> LLM:
    if mock:
        return MockLLM()
    return OpenAICompatibleLLM(s.llm_base_url, s.llm_model, s.llm_api_key,
                               temperature=s.llm_temperature, max_tokens=s.llm_max_tokens)


def make_pipeline(s: Settings | None = None, mock: bool = False) -> QAPipeline:
    s = s or Settings()
    return QAPipeline(make_retriever(s), make_llm(s, mock), k=s.top_k, prompt=s.prompt)
