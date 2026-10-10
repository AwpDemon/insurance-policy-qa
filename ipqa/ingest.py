"""PDF -> structured segments.

A *segment* is the smallest citable unit we extract:

* For SFIP policy forms, the numbering hierarchy is ``Article (I., II., ...) > Letter (A., B.)
  > Number (1., 2.)``. Each line of text is assigned to the deepest clause path open at that
  point, so ``DF:V.D.5`` is "Dwelling Form, Article V (Exclusions), paragraph D, item 5".
  Lower-case items ``a.``/``(1)`` roll up into their numbered parent.
* For consumer guides (Summary of Coverage, Claims Handbook), sections are delimited by
  large-font headings detected from the PDF's span sizes; ids are heading slugs.

Markers are only accepted when they continue the expected sequence (``B.`` must follow ``A.``,
``4.`` must follow ``3.``). That one rule removes almost every false positive caused by text
that merely starts with "U.S." or a wrapped number.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pymupdf

from .corpus import DOCS, SourceDoc

ROMANS = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII", "XIII"]

ARTICLE_RE = re.compile(r"^([IVX]+)\.\s+([A-Z][A-Z ,'&\-–—]+)$")
LETTER_RE = re.compile(r"^([A-Z])\.\s+(.*)$")
NUMBER_RE = re.compile(r"^(\d{1,2})\.(?:\s+(.*))?$")
RUNNING_HEADER_RE = re.compile(r"^(NFIP .* SFIP|PAGE \d+ OF \d+)$")

# Headings in the guides that are running headers/footers rather than section starts.
GUIDE_RUNNING = {"CLAIMS HANDBOOK", "BEFORE A FLOOD", "AFTER A FLOOD"}


@dataclass
class Segment:
    doc_id: str
    clause_id: str                 # e.g. "DF:V.D.5" or "CH:filing-an-increased-cost-of-compliance-icc-claim"
    breadcrumb: list[str]          # human-readable titles, e.g. ["V. EXCLUSIONS", "D."]
    page_start: int                # 1-based PDF page
    page_end: int
    text: str
    lines: list[str] = field(default_factory=list, repr=False)


def clean_line(s: str) -> str:
    s = s.replace(" ", " ").replace("\t", " ")
    return re.sub(r"\s+", " ", s).strip()


def join_lines(lines: list[str]) -> str:
    """Join wrapped PDF lines, healing soft-hyphen breaks ("ele­" + "vation")."""
    out = ""
    for ln in lines:
        if out.endswith("­"):
            out = out[:-1] + ln
        elif out:
            out += " " + ln
        else:
            out = ln
    return out.replace("­", "").strip()


def _page_lines(page: pymupdf.Page) -> list[str]:
    return [clean_line(x) for x in page.get_text().split("\n")]


_SMALL = {"and", "or", "of", "the", "to", "in", "by", "for", "a", "an", "on", "than", "other"}
_SENTENCE_START = ("The ", "We ", "This ", "A ", "If ", "In ", "When ", "Any ", "You ", "Our ")


def _heading_guess(rest: str) -> str:
    """Return a paragraph heading such as "Pair and Set Clause" or "Flood in Progress", else ''.

    SFIP lettered paragraphs either start with a Title-Case heading or go straight into a
    sentence. We keep the text before the first ". " and accept it if it is Title Case.
    """
    cand = rest.split(". ")[0].rstrip(".:")
    words = [w for w in re.split(r"[\s—–/-]+", cand) if w]
    if not words or len(cand) > 60 or cand.startswith(_SENTENCE_START):
        return ""
    content = [w for w in words if w.lower() not in _SMALL]
    if not content:
        return ""
    caps = sum(1 for w in content if w[0].isupper() or w[0].isdigit())
    return cand if caps / len(content) >= 0.85 else ""


# --------------------------------------------------------------------------- policy forms
def parse_policy(doc: SourceDoc, pdf_path: Path) -> list[Segment]:
    pdf = pymupdf.open(pdf_path)
    segs: list[Segment] = []
    art_i = -1          # index into ROMANS of the open article
    art_title = ""
    letter: str | None = None
    letter_title = ""
    number: int | None = None
    started = False
    cur: Segment | None = None

    def open_segment(page_no: int) -> Segment:
        parts = [ROMANS[art_i]] + ([letter] if letter else []) + ([str(number)] if number else [])
        crumb = [f"{ROMANS[art_i]}. {art_title}"]
        if letter:
            crumb.append(f"{letter}. {letter_title}".strip())
        if number:
            crumb.append(f"{number}.")
        return Segment(doc.doc_id, f"{doc.doc_id}:{'.'.join(parts)}", crumb, page_no, page_no, "")

    for page_idx, page in enumerate(pdf):
        page_no = page_idx + 1
        for raw in page.get_text().split("\n"):
            # Soft hyphen must survive until join_lines, so clean everything except it.
            line = clean_line(raw)
            if not line or RUNNING_HEADER_RE.match(line):
                continue
            m_art = ARTICLE_RE.match(line)
            nxt_roman = ROMANS[art_i + 1] if art_i + 1 < len(ROMANS) else None
            if m_art and m_art.group(1) == nxt_roman:
                started = True
                art_i += 1
                art_title = m_art.group(2).strip()
                letter, letter_title, number = None, "", None
                cur = open_segment(page_no)
                segs.append(cur)
                continue
            if not started:
                continue  # cover page / preamble before Article I
            m_let = LETTER_RE.match(line)
            exp_letter = "A" if letter is None else chr(ord(letter) + 1)
            if m_let and m_let.group(1) == exp_letter:
                letter = exp_letter
                rest = m_let.group(2).strip()
                letter_title = _heading_guess(rest)
                number = None
                cur = open_segment(page_no)
                segs.append(cur)
                cur.lines.append(line)
                continue
            m_num = NUMBER_RE.match(line)
            exp_num = 1 if number is None else number + 1
            if m_num and int(m_num.group(1)) == exp_num:
                number = exp_num
                cur = open_segment(page_no)
                segs.append(cur)
                cur.lines.append(line)
                continue
            assert cur is not None
            cur.lines.append(line)
            cur.page_end = page_no
        # end page
    if art_i < 0:
        raise ValueError(f"No articles found in {pdf_path}")
    for s in segs:
        s.text = join_lines(s.lines)
    # The signature block after the last article is not policy language.
    last = segs[-1]
    last.text = re.split(r"In Witness Whereof", last.text)[0].strip()
    return [s for s in segs if s.text]


# --------------------------------------------------------------------------- consumer guides
def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:60]


def _join_spans(spans: list[dict]) -> str:
    """Concatenate spans of one PDF line, inserting a space where there is a visual gap."""
    out = ""
    prev_x1 = None
    for sp in spans:
        t = sp["text"]
        gap = prev_x1 is not None and sp["bbox"][0] - prev_x1 > 1.0
        if gap and not out.endswith(" ") and not t.startswith(" "):
            out += " "
        out += t
        prev_x1 = sp["bbox"][2]
    return out


# Pages that are covers or tables of contents (1-based), per guide.
GUIDE_SKIP_PAGES = {"SC": {1}, "CH": {1, 2}}


def parse_guide(doc: SourceDoc, pdf_path: Path, min_heading_size: float = 11.0) -> list[Segment]:
    pdf = pymupdf.open(pdf_path)
    skip_pages = GUIDE_SKIP_PAGES.get(doc.doc_id, {1})
    segs: list[Segment] = []
    used: dict[str, int] = {}
    cur: Segment | None = None
    for page_idx, page in enumerate(pdf):
        page_no = page_idx + 1
        if page_no in skip_pages:
            continue  # cover / table of contents
        for block in page.get_text("dict")["blocks"]:
            lines: list[tuple[str, float]] = []
            for ln in block.get("lines", []):
                spans = [sp for sp in ln["spans"] if sp["text"].strip()]
                if not spans:
                    continue
                text = clean_line(_join_spans(spans))
                size = max(sp["size"] for sp in spans)
                if not text or re.fullmatch(r"\d{1,2}", text):
                    continue
                if text.upper() in GUIDE_RUNNING and size < min_heading_size:
                    continue  # running header/footer
                lines.append((text, size))
            if not lines:
                continue
            # A heading is a short block set entirely in large type. Large-type *paragraphs*
            # (e.g. the brochure's intro) are long, so they stay body text.
            block_text = " ".join(t for t, _ in lines)
            is_heading = (all(min_heading_size <= sz < 40 for _, sz in lines)
                          and len(block_text) < 80 and re.search(r"[A-Za-z]{3}", block_text))
            # Run-in sub-headings ("Limitations On Appeals") are body-size but set in a demi-bold
            # face, alone in their own one-line block.
            raw_spans = [sp for ln in block.get("lines", []) for sp in ln["spans"] if sp["text"].strip()]
            if (not is_heading and len(lines) == 1 and 4 <= len(block_text) <= 50
                    and lines[0][1] >= 9.4 and all("Dem" in sp["font"] for sp in raw_spans)
                    and re.match(r"[A-Z]", block_text) and not block_text.rstrip().endswith((".", ":"))):
                is_heading = True
            if is_heading:
                if cur is not None and not cur.lines and len(cur.breadcrumb[0]) + len(block_text) < 80:
                    cur.breadcrumb[0] += " " + block_text  # e.g. "REPAIRING & REBUILDING" + subtitle
                    continue
                cur = Segment(doc.doc_id, "", [block_text], page_no, page_no, "")
                segs.append(cur)
                continue
            if cur is None:
                cur = Segment(doc.doc_id, "", ["Introduction"], page_no, page_no, "")
                segs.append(cur)
            cur.lines.extend(t for t, _ in lines)
            cur.page_end = page_no
    out = []
    for s in segs:
        s.text = join_lines(s.lines)
        if len(s.text.split()) < 3:
            continue  # stray label such as a repeated section title
        slug = _slug(s.breadcrumb[0]) or "section"
        used[slug] = used.get(slug, 0) + 1
        if used[slug] > 1:
            slug = f"{slug}-{used[slug]}"
        s.clause_id = f"{doc.doc_id}:{slug}"
        out.append(s)
    return out


def parse_doc(doc: SourceDoc, raw_dir: Path) -> list[Segment]:
    path = raw_dir / doc.filename
    if doc.kind == "policy":
        return parse_policy(doc, path)
    return parse_guide(doc, path)


def ingest(raw_dir: Path, out_path: Path) -> list[Segment]:
    segs: list[Segment] = []
    for doc in DOCS:
        segs.extend(parse_doc(doc, raw_dir))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for s in segs:
            d = asdict(s)
            d.pop("lines")
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    return segs


def load_segments(path: Path) -> list[Segment]:
    with path.open(encoding="utf-8") as f:
        return [Segment(**json.loads(line)) for line in f if line.strip()]
