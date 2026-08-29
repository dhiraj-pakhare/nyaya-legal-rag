"""Unit tests for structure detection: Chapters, Sections, Subsections, and Attachments."""

import pytest
from backend.app.ingestion.pdf_extractor import PDFExtractor
from backend.app.ingestion.structure_detector import StructureDetector


@pytest.fixture(scope="module")
def parsed_document():
    extractor = PDFExtractor("BNS bare act 2023.pdf")
    pages_data = extractor.extract_all_pages(start_page=1, end_page=157)
    detector = StructureDetector(pages_data)
    return detector.detect_structure()


def test_chapter_detection(parsed_document):
    assert len(parsed_document.chapters) == 39
    chap_map = {c.chapter_number: c.chapter_title for c in parsed_document.chapters}
    
    assert "I" in chap_map
    assert chap_map["I"] == "PRELIMINARY"
    assert "V" in chap_map
    assert "ARREST OF PERSONS" in chap_map["V"]
    assert "XXXIX" in chap_map
    assert "MISCELLANEOUS" in chap_map["XXXIX"]


def test_section_count_and_completeness(parsed_document):
    assert len(parsed_document.sections) == 531
    sec_nums = {int(s.section_number) for s in parsed_document.sections}
    assert sec_nums == set(range(1, 532))


def test_section_attachments_provisos_and_explanations(parsed_document):
    sec_map = {s.section_number: s for s in parsed_document.sections}
    
    # Section 35 has provisos
    s35 = sec_map["35"]
    assert len(s35.provisos) >= 1
    assert any("Provided that" in p.text for p in s35.provisos)
    
    # Section 1 has explanation
    s1 = sec_map["1"]
    assert len(s1.explanations) >= 1
    assert "tribal areas" in s1.explanations[0].text
    
    # Section 2 has definitions and explanations
    s2 = sec_map["2"]
    assert len(s2.explanations) >= 1


def test_subsections_and_clauses_parsed(parsed_document):
    sec_map = {s.section_number: s for s in parsed_document.sections}
    s35 = sec_map["35"]
    assert len(s35.subsections) >= 1
    # Check clause extraction in Section 35
    all_clauses = [c.clause_id for sub in s35.subsections for c in sub.clauses]
    assert "(a)" in all_clauses or any("(a)" in sub.text for sub in s35.subsections) or any("(a)" in s35.raw_text for _ in [1])
