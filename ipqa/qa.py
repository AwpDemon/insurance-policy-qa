"""Question answering: retrieve -> build a numbered-excerpt prompt -> generate JSON -> map the
model's excerpt numbers back to clause-level citations."""
from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field

from .chunking import Chunk
from .llm import LLM
from .retrieval import Hit, Retriever

REFUSAL = "Not covered in the documents."

SYSTEM_PROMPT = """You answer questions about NFIP flood insurance using ONLY the numbered policy excerpts provided.

Rules:
1. Use only facts stated in the excerpts. Never use outside knowledge, even if you are confident.
2. If the excerpts do not contain the answer, set "answerable" to false and set "answer" to exactly: "Not covered in the documents."
3. Excerpts may come from different policy forms (Dwelling Form, General Property Form, RCBAP) or consumer guides. If the question names a form, answer from that form.
4. If the question contains a false assumption, say so and give the correct fact from the excerpts.
5. Be concise: at most 4 sentences. Include specific numbers (dollar amounts, days, percentages) exactly as written.
6. Cite the excerpt numbers that support your answer.

Respond with a JSON object: {"answer": string, "citations": [excerpt numbers], "answerable": true|false}"""

# v2 was written after the v1 failure analysis: v1's refusal sentence ("Not covered in the
# documents") collides with insurance language ("not covered" = excluded), and the model refused
# many questions whose answer was an exclusion. v2 keeps refusal in the JSON flag only and says
# explicitly that an exclusion IS an answer. (Evaluated on the same gold set - see README.)
SYSTEM_PROMPT_V2 = """You answer questions about NFIP flood insurance using ONLY the numbered policy excerpts provided.

Rules:
1. Use only facts stated in the excerpts. Never use outside knowledge, even if you are confident.
2. An exclusion, limitation, or condition IS an answer. If an excerpt says the policy does not insure something, excludes it, limits it, or pays only under a condition, answer directly (for example "No - ..." or "Only if ...") and cite that excerpt.
3. Set "answerable" to false ONLY when no excerpt addresses the question at all (for example premiums, rates, or topics the excerpts never mention). In that case set "answer" to an empty string and "citations" to [].
4. Excerpts may come from different policy forms (Dwelling Form, General Property Form, RCBAP) or consumer guides. If the question names a form, answer from that form.
5. If the question contains a false assumption, say so and give the correct fact from the excerpts.
6. For yes/no questions, start with "Yes" or "No". Include the key limit, deadline, or exception (dollar amounts, days, percentages exactly as written). At most 4 sentences.
7. Cite the excerpt numbers that support your answer.

Respond with a JSON object: {"answer": string, "citations": [excerpt numbers], "answerable": true|false}"""

PROMPTS = {"v1": SYSTEM_PROMPT, "v2": SYSTEM_PROMPT_V2}


@dataclass
class Citation:
    n: int
    label: str
    clause_ids: list[str]
    doc_id: str


@dataclass
class QAResult:
    question: str
    answer: str
    answerable: bool
    citations: list[Citation]
    retrieved: list[dict]
    prompt_tokens: int = 0
    completion_tokens: int = 0
    retrieval_s: float = 0.0
    generation_s: float = 0.0
    raw: str = ""
    parse_error: bool = False
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def format_context(hits: list[Hit]) -> str:
    parts = []
    for i, h in enumerate(hits, 1):
        parts.append(f"[{i}] {h.chunk.label}\n{h.chunk.text}")
    return "\n\n".join(parts)


def build_user_prompt(question: str, hits: list[Hit]) -> str:
    return f"EXCERPTS:\n{format_context(hits)}\n\nQUESTION: {question}"


def parse_response(text: str) -> tuple[dict, bool]:
    """Parse the model's JSON; tolerate code fences or stray prose around the object."""
    try:
        return json.loads(text), False
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, flags=re.S)
        if m:
            try:
                return json.loads(m.group(0)), False
            except json.JSONDecodeError:
                pass
    return {"answer": text.strip(), "citations": [], "answerable": True}, True


def is_refusal(answer: str, answerable_flag: bool | None = True) -> bool:
    return answerable_flag is False or bool(re.search(r"not covered in the documents", answer, re.I))


class QAPipeline:
    def __init__(self, retriever: Retriever, llm: LLM, k: int = 5, prompt: str = "v1"):
        self.retriever = retriever
        self.llm = llm
        self.k = k
        self.prompt = prompt
        self.system_prompt = PROMPTS[prompt]

    def answer(self, question: str) -> QAResult:
        t0 = time.perf_counter()
        hits = self.retriever.search(question, self.k)
        t_ret = time.perf_counter() - t0
        resp = self.llm.complete(self.system_prompt, build_user_prompt(question, hits))
        data, perr = parse_response(resp.text)
        answer = str(data.get("answer", "")).strip()
        flag = data.get("answerable", True)
        flag = flag if isinstance(flag, bool) else str(flag).lower() != "false"
        refused = is_refusal(answer, flag)
        cites: list[Citation] = []
        raw_cites = data.get("citations", []) or []
        if not isinstance(raw_cites, list):
            raw_cites = [raw_cites]
        for c in raw_cites:
            try:
                n = int(str(c).strip("[] "))
            except ValueError:
                continue
            if 1 <= n <= len(hits) and all(x.n != n for x in cites):
                ch: Chunk = hits[n - 1].chunk
                cites.append(Citation(n, ch.label, ch.clause_ids, ch.doc_id))
        if refused:
            answer, cites = REFUSAL, []
        return QAResult(
            question=question, answer=answer, answerable=not refused, citations=cites,
            retrieved=[{"rank": h.rank, "label": h.chunk.label, "score": h.score,
                        "clause_ids": h.chunk.clause_ids, "text": h.chunk.text} for h in hits],
            prompt_tokens=resp.prompt_tokens, completion_tokens=resp.completion_tokens,
            retrieval_s=t_ret, generation_s=resp.latency_s, raw=resp.text, parse_error=perr,
        )
