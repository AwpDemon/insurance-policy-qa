"""Retrievers: BM25 (lexical), dense (nomic-embed-text via Ollama), hybrid (reciprocal-rank
fusion of the two), and an optional cross-encoder reranker on top of any of them."""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import httpx
import numpy as np
from rank_bm25 import BM25Okapi

from .chunking import Chunk

# --------------------------------------------------------------------------- tokenization
_STOP = set("""a an and are as at be by for from has have if in into is it its of on or that the
their them then there these this to was were will with you your we our us not do does""".split())
_TOKEN_RE = re.compile(r"[a-z0-9$]+(?:[.,][0-9]+)*")

try:  # Snowball stemming via the small Rust wheel that ships with fastembed.
    from py_rust_stemmers import SnowballStemmer
    _stemmer = SnowballStemmer("english")

    def _stem(w: str) -> str:
        return _stemmer.stem_word(w)
except Exception:  # pragma: no cover - fallback keeps BM25 usable without the wheel
    def _stem(w: str) -> str:
        return w


def tokenize(text: str) -> list[str]:
    toks = _TOKEN_RE.findall(text.lower().replace("’", "'"))
    return [_stem(t) for t in toks if t not in _STOP]


@dataclass
class Hit:
    chunk: Chunk
    score: float
    rank: int


class Retriever(Protocol):
    name: str

    def search(self, query: str, k: int) -> list[Hit]: ...


def _to_hits(chunks: list[Chunk], scores: np.ndarray, k: int) -> list[Hit]:
    order = np.argsort(-scores, kind="stable")[:k]
    return [Hit(chunks[i], float(scores[i]), r + 1) for r, i in enumerate(order)]


class BM25Retriever:
    name = "bm25"

    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.bm25 = BM25Okapi([tokenize(c.text) for c in chunks])

    def search(self, query: str, k: int) -> list[Hit]:
        return _to_hits(self.chunks, np.asarray(self.bm25.get_scores(tokenize(query))), k)


# --------------------------------------------------------------------------- embeddings
class OllamaEmbedder:
    """Calls Ollama's /api/embed. nomic-embed-text expects task prefixes on its inputs."""

    def __init__(self, base_url: str, model: str, cache_dir: Path | None = None, batch: int = 32):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.cache_dir = cache_dir
        self.batch = batch
        self._client = httpx.Client(timeout=120)

    def _embed(self, texts: list[str]) -> np.ndarray:
        out = []
        for i in range(0, len(texts), self.batch):
            r = self._client.post(f"{self.base_url}/api/embed",
                                  json={"model": self.model, "input": texts[i:i + self.batch]})
            r.raise_for_status()
            out.extend(r.json()["embeddings"])
        arr = np.asarray(out, dtype=np.float32)
        return arr / np.linalg.norm(arr, axis=1, keepdims=True)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        prefixed = [f"search_document: {t}" for t in texts]
        if self.cache_dir is None:
            return self._embed(prefixed)
        key = hashlib.sha256(json.dumps([self.model, prefixed]).encode()).hexdigest()[:16]
        path = self.cache_dir / f"emb_{key}.npy"
        if path.exists():
            return np.load(path)
        vecs = self._embed(prefixed)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, vecs)
        return vecs

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed([f"search_query: {text}"])[0]


class DenseRetriever:
    name = "dense"

    def __init__(self, chunks: list[Chunk], embedder):
        self.chunks = chunks
        self.embedder = embedder
        self.matrix = embedder.embed_documents([c.text for c in chunks])
        self._qcache: dict[str, np.ndarray] = {}

    def query_vec(self, query: str) -> np.ndarray:
        if query not in self._qcache:
            self._qcache[query] = self.embedder.embed_query(query)
        return self._qcache[query]

    def search(self, query: str, k: int) -> list[Hit]:
        return _to_hits(self.chunks, self.matrix @ self.query_vec(query), k)


class HybridRetriever:
    """Reciprocal Rank Fusion: score(d) = sum_r 1 / (rrf_k + rank_r(d)).

    RRF only uses ranks, so it needs no score normalisation between BM25 (unbounded) and cosine
    similarity (-1..1). rrf_k=60 is the value from Cormack et al. (2009)."""
    name = "hybrid"

    def __init__(self, retrievers: list, depth: int = 50, rrf_k: int = 60):
        self.retrievers = retrievers
        self.depth = depth
        self.rrf_k = rrf_k

    def search(self, query: str, k: int) -> list[Hit]:
        fused: dict[str, float] = {}
        by_id: dict[str, Chunk] = {}
        for r in self.retrievers:
            for h in r.search(query, self.depth):
                fused[h.chunk.chunk_id] = fused.get(h.chunk.chunk_id, 0.0) + 1.0 / (self.rrf_k + h.rank)
                by_id[h.chunk.chunk_id] = h.chunk
        order = sorted(fused, key=lambda cid: -fused[cid])[:k]
        return [Hit(by_id[cid], fused[cid], i + 1) for i, cid in enumerate(order)]


class CrossEncoderReranker:
    """Re-scores the first-stage candidates with a cross-encoder (query and passage read jointly).

    Scores are cached on disk keyed by (model, query, passage) so that ablations and repeated
    eval runs never pay for the same pair twice."""

    def __init__(self, base: Retriever, model: str = "Xenova/ms-marco-MiniLM-L-12-v2",
                 candidates: int = 20, encoder=None, cache_path: Path | None = None):
        self.base = base
        self.candidates = candidates
        self.model = model
        self.name = f"{base.name}+rerank"
        if encoder is None:
            from fastembed.rerank.cross_encoder import TextCrossEncoder
            encoder = TextCrossEncoder(model_name=model)
        self.encoder = encoder
        self.cache_path = cache_path
        self._cache: dict[str, float] = {}
        if cache_path is not None and cache_path.exists():
            self._cache = json.loads(cache_path.read_text(encoding="utf-8"))
        self._dirty = 0

    def _key(self, query: str, text: str) -> str:
        return hashlib.sha1(f"{self.model}|{query}|{text}".encode()).hexdigest()

    def _flush(self) -> None:
        if self.cache_path is not None and self._dirty:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(json.dumps(self._cache), encoding="utf-8")
            self._dirty = 0

    def search(self, query: str, k: int) -> list[Hit]:
        cands = self.base.search(query, self.candidates)
        keys = [self._key(query, h.chunk.text) for h in cands]
        todo = [i for i, key in enumerate(keys) if key not in self._cache]
        if todo:
            new = list(self.encoder.rerank(query, [cands[i].chunk.text for i in todo]))
            for i, sc in zip(todo, new):
                self._cache[keys[i]] = float(sc)
            self._dirty += 1
            if self._dirty >= 10:
                self._flush()
        scores = [self._cache[key] for key in keys]
        order = sorted(range(len(cands)), key=lambda i: -scores[i])[:k]
        return [Hit(cands[i].chunk, float(scores[i]), r + 1) for r, i in enumerate(order)]

    def __del__(self):  # best effort
        try:
            self._flush()
        except Exception:
            pass


def build_retriever(kind: str, chunks: list[Chunk], embedder=None, rerank: bool = False,
                    encoder=None, rerank_cache: Path | None = None) -> Retriever:
    if kind == "bm25":
        r: Retriever = BM25Retriever(chunks)
    elif kind == "dense":
        r = DenseRetriever(chunks, embedder)
    elif kind == "hybrid":
        r = HybridRetriever([BM25Retriever(chunks), DenseRetriever(chunks, embedder)])
    else:
        raise ValueError(f"unknown retriever {kind!r}")
    return CrossEncoderReranker(r, encoder=encoder, cache_path=rerank_cache) if rerank else r
