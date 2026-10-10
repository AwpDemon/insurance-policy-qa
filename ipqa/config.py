"""Runtime configuration. Everything is overridable with environment variables so the same code
runs against local Ollama, a hosted OpenAI-compatible API, or the mock LLM used in CI."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
CACHE_DIR = ROOT / "data" / "cache"
SEGMENTS_PATH = PROCESSED_DIR / "segments.jsonl"


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass
class Settings:
    # Generation: any OpenAI-compatible /v1 endpoint.
    llm_base_url: str = field(default_factory=lambda: _env("IPQA_LLM_BASE_URL", "http://127.0.0.1:11434/v1"))
    llm_model: str = field(default_factory=lambda: _env("IPQA_LLM_MODEL", "qwen2.5:7b-instruct"))
    llm_api_key: str = field(default_factory=lambda: _env("IPQA_LLM_API_KEY", "ollama"))
    llm_temperature: float = 0.0
    llm_max_tokens: int = 400
    # Embeddings: Ollama native /api/embed endpoint.
    embed_base_url: str = field(default_factory=lambda: _env("IPQA_EMBED_BASE_URL", "http://127.0.0.1:11434"))
    embed_model: str = field(default_factory=lambda: _env("IPQA_EMBED_MODEL", "nomic-embed-text"))
    # Retrieval defaults (the configuration the eval selected; see README results).
    chunking: str = field(default_factory=lambda: _env("IPQA_CHUNKING", "clause"))
    retriever: str = field(default_factory=lambda: _env("IPQA_RETRIEVER", "bm25"))
    rerank: bool = field(default_factory=lambda: _env("IPQA_RERANK", "1") == "1")
    top_k: int = field(default_factory=lambda: int(_env("IPQA_TOP_K", "5")))
    prompt: str = field(default_factory=lambda: _env("IPQA_PROMPT", "v2"))
