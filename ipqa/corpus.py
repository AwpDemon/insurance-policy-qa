"""Corpus registry: which public documents we index, where they come from, and their terms.

All five documents are works of the U.S. federal government (FEMA / NFIP) and are therefore
not subject to copyright in the United States (17 U.S.C. 105). They are still not committed to
the repo; `scripts/download_corpus.py` fetches them from FEMA so the source of truth stays FEMA.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SourceDoc:
    doc_id: str          # short stable id used in clause ids, e.g. "DF"
    title: str
    kind: str            # "policy" (numbered SFIP clauses) or "guide" (heading-based consumer guide)
    filename: str
    url: str
    edition: str


DOCS: tuple[SourceDoc, ...] = (
    SourceDoc(
        "DF", "SFIP Dwelling Form (FEMA F-122)", "policy", "sfip_dwelling_f122_2021.pdf",
        "https://www.fema.gov/sites/default/files/documents/fema_F-122-Dwelling-SFIP_2021.pdf",
        "October 2021",
    ),
    SourceDoc(
        "GP", "SFIP General Property Form (FEMA F-123)", "policy", "sfip_general_property_f123_2021.pdf",
        "https://www.fema.gov/sites/default/files/documents/fema_F-123-general-property-SFIP_2021.pdf",
        "October 2021",
    ),
    SourceDoc(
        "RC", "SFIP Residential Condominium Building Association Policy (FEMA F-144)", "policy",
        "sfip_rcbap_f144_2021.pdf",
        "https://www.fema.gov/sites/default/files/documents/fema_F-144-RCBAP-SFIP_2021.pdf",
        "October 2021",
    ),
    SourceDoc(
        "SC", "NFIP Summary of Coverage (FEMA P-2144)", "guide", "nfip_summary_of_coverage_2023.pdf",
        "https://agents.floodsmart.gov/sites/default/files/media/document/2025-07/"
        "fema-nfip-summary-coverage-brochure-12-2023.pdf",
        "December 2023",
    ),
    SourceDoc(
        "CH", "NFIP Claims Handbook (FEMA F-687)", "guide", "nfip_claims_handbook_2024.pdf",
        "https://agents.floodsmart.gov/sites/default/files/media/document/2025-07/"
        "fema-nfip-claims-handbook-08-2024.pdf",
        "August 2024",
    ),
)

DOCS_BY_ID = {d.doc_id: d for d in DOCS}
