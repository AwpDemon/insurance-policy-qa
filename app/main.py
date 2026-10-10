"""FastAPI service + single-page UI.

    uvicorn app.main:app --port 8000

Env: IPQA_MOCK=1 uses the deterministic mock LLM (no model server needed); the IPQA_* settings
in ipqa/config.py choose the endpoint, model, chunking, retriever and k.
"""
from __future__ import annotations

import os
import threading
import time
from collections import defaultdict, deque
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from ipqa.config import Settings
from ipqa.corpus import DOCS
from ipqa.qa import QAPipeline
from ipqa.system import make_pipeline

STATIC = Path(__file__).resolve().parent / "static"
RATE_LIMIT = int(os.environ.get("IPQA_RATE_LIMIT_PER_MIN", "20"))

app = FastAPI(title="Insurance Policy QA", version="1.0.0")
_pipe: QAPipeline | None = None
_lock = threading.Lock()
_hits: dict[str, deque] = defaultdict(deque)


def get_pipeline() -> QAPipeline:
    global _pipe
    with _lock:
        if _pipe is None:
            _pipe = make_pipeline(Settings(), mock=os.environ.get("IPQA_MOCK") == "1")
        return _pipe


def set_pipeline(p: QAPipeline | None) -> None:
    """Test hook."""
    global _pipe
    _pipe = p


def _rate_limited(ip: str) -> bool:
    now = time.monotonic()
    q = _hits[ip]
    while q and now - q[0] > 60:
        q.popleft()
    if len(q) >= RATE_LIMIT:
        return True
    q.append(now)
    return False


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/api/health")
def health() -> dict:
    s = Settings()
    return {"status": "ok", "model": s.llm_model, "chunking": s.chunking, "retriever": s.retriever,
            "rerank": s.rerank, "k": s.top_k, "mock": os.environ.get("IPQA_MOCK") == "1"}


@app.get("/api/documents")
def documents() -> list[dict]:
    return [{"id": d.doc_id, "title": d.title, "edition": d.edition, "url": d.url} for d in DOCS]


@app.post("/api/ask")
def ask(req: AskRequest, request: Request) -> dict:
    ip = request.client.host if request.client else "unknown"
    if _rate_limited(ip):
        raise HTTPException(status_code=429, detail="Rate limit exceeded; try again in a minute.")
    res = get_pipeline().answer(req.question.strip())
    return {
        "answer": res.answer,
        "answerable": res.answerable,
        "citations": [{"n": c.n, "label": c.label, "clause_ids": c.clause_ids} for c in res.citations],
        "excerpts": [{"n": r["rank"], "label": r["label"], "text": r["text"]} for r in res.retrieved],
        "timing": {"retrieval_ms": round(res.retrieval_s * 1000), "generation_ms": round(res.generation_s * 1000)},
        "tokens": {"prompt": res.prompt_tokens, "completion": res.completion_tokens},
    }
