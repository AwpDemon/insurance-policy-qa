"""Three chunking strategies, compared in the ablation.

* ``clause``  - structure-aware. One chunk per numbered clause (small siblings merged), with the
  document title, the clause path and the parent paragraph's lead-in sentence prepended. The
  lead-in matters: "4. Sinkholes;" means nothing until you know it sits under "We do not
  insure for loss to property caused directly by earth movement".
* ``section`` - one chunk per lettered paragraph (e.g. all of "V.D"), split only if very long.
* ``fixed``   - the naive baseline: 200-word sliding windows over the raw text, no headers.

Every chunk records which clause ids it *covers* (>= half of a segment's words, or >= 40 of
them), independent of strategy, so retrieval can be scored against clause-level gold labels
fairly across strategies.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .corpus import DOCS_BY_ID
from .ingest import Segment

STRATEGIES = ("clause", "section", "fixed")
SHORT_TITLES = {"DF": "Dwelling Form", "GP": "General Property Form", "RC": "RCBAP Form",
                "SC": "Summary of Coverage", "CH": "Claims Handbook"}


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str                      # what gets embedded / indexed / shown to the LLM
    label: str                     # short citation label, e.g. "Dwelling Form V.D.5 (p. 16)"
    clause_ids: list[str]          # segments meaningfully covered by this chunk
    page_start: int
    page_end: int
    primary: str = ""              # most specific clause id, used for display
    meta: dict = field(default_factory=dict)


def parent_id(cid: str) -> str | None:
    doc, path = cid.split(":", 1)
    if "." not in path:
        return None
    return f"{doc}:{path.rsplit('.', 1)[0]}"


def ancestors(cid: str) -> list[str]:
    out = []
    p = parent_id(cid)
    while p:
        out.append(p)
        p = parent_id(p)
    return out


def _covered(word_counts: dict[str, int], totals: dict[str, int]) -> list[str]:
    return [cid for cid, n in word_counts.items() if n >= 40 or n >= 0.5 * max(totals[cid], 1)]


def _windows(words: list[str], size: int, overlap: int) -> list[tuple[int, int]]:
    if len(words) <= size:
        return [(0, len(words))]
    out, start = [], 0
    while start < len(words):
        end = min(start + size, len(words))
        out.append((start, end))
        if end == len(words):
            break
        start = end - overlap
    return out


def _label(doc_id: str, cid: str, p0: int, p1: int) -> str:
    path = cid.split(":", 1)[1]
    pages = f"p. {p0}" if p0 == p1 else f"pp. {p0}-{p1}"
    if DOCS_BY_ID[doc_id].kind == "guide":
        return f"{SHORT_TITLES[doc_id]} § {path} ({pages})"
    return f"{SHORT_TITLES[doc_id]} {path} ({pages})"


def _crumb(seg: Segment) -> str:
    return " > ".join(seg.breadcrumb)


# --------------------------------------------------------------------------- structure-aware
def chunk_clause(segments: list[Segment], max_words: int = 260, merge_under: int = 120,
                 overlap: int = 40) -> list[Chunk]:
    by_id = {s.clause_id: s for s in segments}
    totals = {s.clause_id: len(s.text.split()) for s in segments}
    chunks: list[Chunk] = []

    def emit(group: list[Segment]) -> None:
        first = group[0]
        doc_id = first.doc_id
        # Lead-in: the parent paragraph's own text, if it is short (e.g. "We do not insure ...:").
        lead = ""
        par = parent_id(first.clause_id)
        par_seg = by_id.get(par) if par else None
        if par_seg is not None and par_seg not in group and len(par_seg.text.split()) <= 60:
            lead = par_seg.text
        header = f"[{SHORT_TITLES[doc_id]}] {_crumb(first)}"
        body_words = " ".join(s.text for s in group).split()
        seg_spans = []  # (clause_id, start, end) word offsets in body
        pos = 0
        for s in group:
            n = len(s.text.split())
            seg_spans.append((s.clause_id, pos, pos + n))
            pos += n
        for w0, w1 in _windows(body_words, max_words, overlap):
            counts: dict[str, int] = {}
            for cid, a, b in seg_spans:
                ov = max(0, min(b, w1) - max(a, w0))
                if ov:
                    counts[cid] = ov
            covered = _covered(counts, totals)
            if lead and par_seg is not None:
                covered.append(par_seg.clause_id)
            text = header + "\n" + (lead + "\n" if lead else "") + " ".join(body_words[w0:w1])
            cid0 = group[0].clause_id
            label = _label(doc_id, cid0, min(s.page_start for s in group), max(s.page_end for s in group))
            if len(group) > 1:  # "Dwelling Form II.C.1-4 (p. 4)"
                last = group[-1].clause_id.rsplit(".", 1)[-1]
                label = label.replace(cid0.split(":", 1)[1], f"{cid0.split(':', 1)[1]}-{last}", 1)
            chunks.append(Chunk(
                chunk_id=f"clause:{cid0}:{w0}", doc_id=doc_id, text=text, label=label,
                clause_ids=sorted(set(covered)), primary=cid0,
                page_start=min(s.page_start for s in group), page_end=max(s.page_end for s in group),
            ))

    # Group consecutive small siblings (same parent) so definitions like "Act." don't become
    # 8-word chunks, but never merge across a parent boundary.
    group: list[Segment] = []
    group_words = 0
    for s in segments:
        n = len(s.text.split())
        par = parent_id(s.clause_id)
        same_parent = bool(group) and par is not None and parent_id(group[0].clause_id) == par
        if group and same_parent and group_words + n <= merge_under and n < merge_under:
            group.append(s)
            group_words += n
            continue
        if group:
            emit(group)
        group, group_words = [s], n
    if group:
        emit(group)
    return chunks


# --------------------------------------------------------------------------- section-level
def _section_key(cid: str) -> str:
    doc, path = cid.split(":", 1)
    parts = path.split(".")
    return f"{doc}:{'.'.join(parts[:2])}"


def chunk_section(segments: list[Segment], max_words: int = 450, overlap: int = 60) -> list[Chunk]:
    totals = {s.clause_id: len(s.text.split()) for s in segments}
    groups: dict[str, list[Segment]] = {}
    for s in segments:
        groups.setdefault(_section_key(s.clause_id), []).append(s)
    chunks = []
    for key, segs in groups.items():
        doc_id = segs[0].doc_id
        header = f"[{SHORT_TITLES[doc_id]}] {' > '.join(segs[0].breadcrumb[:2])}"
        words, spans, pos = [], [], 0
        for s in segs:
            w = s.text.split()
            spans.append((s.clause_id, pos, pos + len(w)))
            words.extend(w)
            pos += len(w)
        for w0, w1 in _windows(words, max_words, overlap):
            counts = {cid: max(0, min(b, w1) - max(a, w0)) for cid, a, b in spans}
            counts = {k: v for k, v in counts.items() if v}
            p0 = min(s.page_start for s in segs if counts.get(s.clause_id))
            p1 = max(s.page_end for s in segs if counts.get(s.clause_id))
            chunks.append(Chunk(
                chunk_id=f"section:{key}:{w0}", doc_id=doc_id,
                text=header + "\n" + " ".join(words[w0:w1]), label=_label(doc_id, key, p0, p1),
                clause_ids=sorted(_covered(counts, totals)), primary=key, page_start=p0, page_end=p1,
            ))
    return chunks


# --------------------------------------------------------------------------- naive fixed windows
def chunk_fixed(segments: list[Segment], size: int = 200, overlap: int = 40) -> list[Chunk]:
    totals = {s.clause_id: len(s.text.split()) for s in segments}
    chunks = []
    for doc_id in dict.fromkeys(s.doc_id for s in segments):
        segs = [s for s in segments if s.doc_id == doc_id]
        words: list[str] = []
        owner: list[Segment] = []
        for s in segs:
            w = s.text.split()
            words.extend(w)
            owner.extend([s] * len(w))
        for w0, w1 in _windows(words, size, overlap):
            counts: dict[str, int] = {}
            for s in owner[w0:w1]:
                counts[s.clause_id] = counts.get(s.clause_id, 0) + 1
            covered = sorted(_covered(counts, totals))
            primary = max(counts, key=lambda c: counts[c])
            p0, p1 = owner[w0].page_start, owner[w1 - 1].page_end
            chunks.append(Chunk(
                chunk_id=f"fixed:{doc_id}:{w0}", doc_id=doc_id, text=" ".join(words[w0:w1]),
                label=_label(doc_id, primary, p0, p1), clause_ids=covered, primary=primary,
                page_start=p0, page_end=p1,
            ))
    return chunks


def build_chunks(segments: list[Segment], strategy: str) -> list[Chunk]:
    if strategy == "clause":
        return chunk_clause(segments)
    if strategy == "section":
        return chunk_section(segments)
    if strategy == "fixed":
        return chunk_fixed(segments)
    raise ValueError(f"unknown chunking strategy {strategy!r}; choose from {STRATEGIES}")
