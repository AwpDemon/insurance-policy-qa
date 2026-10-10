from __future__ import annotations

import sys
from pathlib import Path

import pymupdf
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ipqa.config import SEGMENTS_PATH  # noqa: E402
from ipqa.ingest import load_segments  # noqa: E402


@pytest.fixture(scope="session")
def segments():
    return load_segments(SEGMENTS_PATH)


@pytest.fixture()
def policy_pdf(tmp_path: Path) -> Path:
    """A tiny synthetic two-page PDF laid out like an SFIP form (tabs after markers,
    running headers, a soft-hyphen line break, and a decoy line starting with 'U.S.')."""
    lines_p1 = [
        "NFIP TEST FORM SFIP\t", "PAGE 1 OF 2", "Cover text before the agreement.",
        "I. AGREEMENT", "A.\tWe will pay for direct physical loss by or from flood.",
        "B.\tThis policy insures only one building.",
        "II. DEFINITIONS", "A.\tFlood means a general and temporary condition of inun­",
        "dation of two or more acres.", "U.S. Army Corps of Engineers is mentioned here.",
        "B.\tThe following definitions apply:", "1.\t Act. The National Flood Insurance Act.",
        "2.\t Basement. Floor below ground level on all sides.",
    ]
    lines_p2 = [
        "NFIP TEST FORM SFIP\t", "PAGE 2 OF 2", "III. PROPERTY NOT INSURED",
        "We do not insure any of the following:", "1.\tCurrency and money.", "2.\tSwimming pools.",
        "7.\tThis out-of-sequence number must not open a clause.",
        "IV. WHAT LAW GOVERNS", "Federal common law governs.",
        "In Witness Whereof, we have signed this policy.",
    ]
    doc = pymupdf.open()
    for lines in (lines_p1, lines_p2):
        page = doc.new_page()
        y = 72
        for ln in lines:
            page.insert_text((72, y), ln.replace("\t", "    "), fontsize=10)
            y += 14
    path = tmp_path / "test_form.pdf"
    doc.save(path)
    return path
