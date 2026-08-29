"""Unit tests for cross-reference extraction and normalization."""

import pytest
from backend.app.ingestion.cross_ref import extract_cross_references


def test_extract_simple_section_reference():
    text = "as defined under section 2 or section 103 of this Sanhita"
    refs = extract_cross_references(text, default_act_short="BNSS")
    assert "BNSS s.2" in refs
    assert "BNSS s.103" in refs


def test_extract_subsection_reference():
    text = "subject to the provisions of section 35(1) and section 2(11)"
    refs = extract_cross_references(text, default_act_short="BNSS")
    assert "BNSS s.35(1)" in refs
    assert "BNSS s.2(11)" in refs


def test_extract_bns_cross_references():
    text = (
        "punishable under section 70 of the Bharatiya Nyaya Sanhita, 2023 "
        "or sub-section (1) of section 353 of the Bharatiya Nyaya Sanhita, 2023"
    )
    refs = extract_cross_references(text, default_act_short="BNSS")
    assert "BNS s.70" in refs
    assert "BNS s.353(1)" in refs


def test_extract_chapter_and_schedule_references():
    text = "provisions relating to Chapter IX and the First Schedule shall apply"
    refs = extract_cross_references(text, default_act_short="BNSS")
    assert "Chapter IX" in refs
    assert "First Schedule" in refs


def test_multiple_sections_list():
    text = "an order passed under sections 125, 126 and 127"
    refs = extract_cross_references(text, default_act_short="BNSS")
    assert "BNSS s.125" in refs
    assert "BNSS s.126" in refs
    assert "BNSS s.127" in refs
