"""Ground-truth verification tests comparing parsed outputs directly with known PDF ground truths."""

import pytest
from backend.app.ingestion.parser import StatutoryParser
from backend.app.ingestion.models import ChunkType


@pytest.fixture(scope="module")
def parsed_result():
    parser = StatutoryParser("BNS bare act 2023.pdf")
    return parser.parse()


def test_ground_truth_section_1(parsed_result):
    """Verify Section 1 (Page 1) matches enacted text."""
    s1 = next(s for s in parsed_result.document.sections if s.section_number == "1")
    assert s1.section_title == "Short title, extent and commencement"
    assert s1.chapter_number == "I"
    assert s1.page_start == 1
    assert "Bharatiya Nagarik Suraksha Sanhita, 2023" in s1.raw_text
    assert len(s1.explanations) == 1
    assert "tribal areas" in s1.explanations[0].text


def test_ground_truth_section_2(parsed_result):
    """Verify Section 2 (Definitions, Pages 2-3) matches enacted text."""
    s2 = next(s for s in parsed_result.document.sections if s.section_number == "2")
    assert s2.section_title == "Definitions"
    assert s2.chapter_number == "I"
    assert "audio-video electronic means" in s2.raw_text
    assert "bailable offence" in s2.raw_text
    assert "First Schedule" in s2.references


def test_ground_truth_section_35(parsed_result):
    """Verify Section 35 (Arrest without warrant, Pages 13-14) matches enacted text."""
    s35 = next(s for s in parsed_result.document.sections if s.section_number == "35")
    assert "When police may arrest without warrant" in s35.section_title
    assert s35.chapter_number == "V"
    assert s35.page_start == 13
    assert len(s35.provisos) >= 1
    assert "cognizable offence" in s35.raw_text


def test_ground_truth_section_187(parsed_result):
    """Verify Section 187 (Police Custody / 15-day detention period, Pages 59-61)."""
    s187 = next(s for s in parsed_result.document.sections if s.section_number == "187")
    assert s187.chapter_number == "XIII"
    assert s187.page_start in [59, 60]
    assert "fifteen days" in s187.raw_text
    assert "sixty days" in s187.raw_text


def test_ground_truth_section_531(parsed_result):
    """Verify Section 531 (Repeal and Savings, Pages 156-157)."""
    s531 = next(s for s in parsed_result.document.sections if s.section_number == "531")
    assert "Repeal and savings" in s531.section_title or "Repeal" in s531.section_title
    assert s531.chapter_number == "XXXIX"
    assert "Code of Criminal Procedure, 1973 is hereby repealed" in s531.raw_text


def test_ground_truth_schedule_entries(parsed_result):
    """Verify First Schedule entries for BNS offences."""
    entry_map = {e.section_number: e for e in parsed_result.schedule_entries}
    
    # BNS s.105
    assert "105" in entry_map
    e105 = entry_map["105"]
    assert "Non-bailable" in e105.bailable_status
    assert "Cognizable" in e105.cognizable_status
    assert "Court of Session" in e105.triable_court
    
    # BNS s.281
    assert "281" in entry_map
    e281 = entry_map["281"]
    assert "Bailable" in e281.bailable_status
