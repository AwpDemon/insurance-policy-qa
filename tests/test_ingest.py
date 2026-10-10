from ipqa.corpus import SourceDoc
from ipqa.ingest import _heading_guess, join_lines, parse_policy

DOC = SourceDoc("DF", "Test form", "policy", "x.pdf", "http://example", "2021")


def test_parse_policy_builds_clause_hierarchy(policy_pdf):
    segs = parse_policy(DOC, policy_pdf)
    ids = [s.clause_id for s in segs]
    assert ids == ["DF:I", "DF:I.A", "DF:I.B", "DF:II", "DF:II.A", "DF:II.B", "DF:II.B.1",
                   "DF:II.B.2", "DF:III", "DF:III.1", "DF:III.2", "DF:IV"] or \
        ids == ["DF:I.A", "DF:I.B", "DF:II.A", "DF:II.B", "DF:II.B.1", "DF:II.B.2", "DF:III",
                "DF:III.1", "DF:III.2", "DF:IV"]


def test_cover_text_and_running_headers_dropped(policy_pdf):
    text = " ".join(s.text for s in parse_policy(DOC, policy_pdf))
    assert "Cover text" not in text
    assert "PAGE 1 OF 2" not in text and "NFIP TEST FORM" not in text


def test_decoy_and_out_of_sequence_markers_stay_in_parent(policy_pdf):
    segs = {s.clause_id: s for s in parse_policy(DOC, policy_pdf)}
    assert "U.S. Army" in segs["DF:II.A"].text            # 'U.' is not letter B
    assert "out-of-sequence" in segs["DF:III.2"].text     # '7.' after '2.' is not a clause
    assert "DF:III.7" not in segs


def test_signature_block_removed_and_pages_tracked(policy_pdf):
    segs = {s.clause_id: s for s in parse_policy(DOC, policy_pdf)}
    assert "In Witness" not in segs["DF:IV"].text
    assert segs["DF:I.A"].page_start == 1
    assert segs["DF:III.1"].page_start == 2


def test_join_lines_heals_soft_hyphen():
    assert join_lines(["inun­", "dation of land"]) == "inundation of land"
    assert join_lines(["two", "acres"]) == "two acres"


def test_heading_guess():
    assert _heading_guess("Pair and Set Clause") == "Pair and Set Clause"
    assert _heading_guess("Flood in Progress. If this policy became effective") == "Flood in Progress"
    assert _heading_guess("We will pay you for direct physical loss by or from flood") == ""
    assert _heading_guess("Coverage A—Building Property") == "Coverage A—Building Property"


def test_real_corpus_has_expected_structure(segments):
    ids = {s.clause_id for s in segments}
    # Spot checks against the FEMA forms (Dwelling III.B.8 = $2,500 special limits, etc.)
    for cid in ["DF:III.B.8", "DF:V.D.5", "DF:VII.R.4", "DF:X", "GP:III.C.3", "RC:VII.C.3",
                "CH:submitting-your-appeal", "SC:how-flood-damage-is-paid"]:
        assert cid in ids, cid
    seg = {s.clause_id: s for s in segments}
    assert "$2,500" in seg["DF:III.B.8"].text
    assert "$10,000" in seg["GP:III.C.3"].text
    assert "134,500" in seg["RC:VII.C.3"].text
    assert {s.doc_id for s in segments} == {"DF", "GP", "RC", "SC", "CH"}
