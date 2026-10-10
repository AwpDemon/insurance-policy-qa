"""LLM clients. The real client speaks the OpenAI chat-completions protocol, so it works with
Ollama (/v1), vLLM, llama.cpp server, or a hosted API by changing base URL + model + key.
The mock client is deterministic and is what CI uses."""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Protocol


@dataclass
class LLMResponse:
    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_s: float
    model: str


class LLM(Protocol):
    model: str

    def complete(self, system: str, user: str, json_mode: bool = True) -> LLMResponse: ...


class OpenAICompatibleLLM:
    def __init__(self, base_url: str, model: str, api_key: str = "none", temperature: float = 0.0,
                 max_tokens: int = 400, timeout: float = 180.0, seed: int | None = 7):
        from openai import OpenAI
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.seed = seed

    def complete(self, system: str, user: str, json_mode: bool = True) -> LLMResponse:
        kwargs = dict(model=self.model, temperature=self.temperature, max_tokens=self.max_tokens,
                      messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        if self.seed is not None:
            kwargs["seed"] = self.seed
        t0 = time.perf_counter()
        r = self.client.chat.completions.create(**kwargs)
        dt = time.perf_counter() - t0
        u = r.usage
        return LLMResponse(r.choices[0].message.content or "", getattr(u, "prompt_tokens", 0) or 0,
                           getattr(u, "completion_tokens", 0) or 0, dt, self.model)


class MockLLM:
    """Deterministic stand-in used by tests/CI: answers with the first sentence of the
    best-overlapping excerpt and cites it; refuses when no excerpt shares a content word."""
    model = "mock"

    def complete(self, system: str, user: str, json_mode: bool = True) -> LLMResponse:
        q = re.search(r"QUESTION:\s*(.*)", user)
        qwords = set(re.findall(r"[a-z]{5,}", (q.group(1) if q else "").lower()))
        best, best_n, best_text = None, 0, ""
        for m in re.finditer(r"\[(\d+)\][^\n]*\n(.*?)(?=\n\[\d+\]|\Z)", user, flags=re.S):
            n = len(qwords & set(re.findall(r"[a-z]{5,}", m.group(2).lower())))
            if n > best_n:
                best, best_n, best_text = int(m.group(1)), n, m.group(2)
        if best is None:
            out = {"answer": "Not covered in the documents.", "citations": [], "answerable": False}
        else:
            sent = re.split(r"(?<=[.;:])\s", best_text.strip().split("\n")[-1])[0][:300]
            out = {"answer": sent, "citations": [best], "answerable": True}
        return LLMResponse(json.dumps(out), len(user) // 4, 40, 0.0, self.model)
