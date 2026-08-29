"""Unit tests for PDF coordinate layout extraction and marginal note association."""

import pytest
from backend.app.ingestion.pdf_extractor import PDFExtractor
from backend.app.ingestion.marginal_notes import (
    associate_marginal_notes_with_sections,
    cluster_marginal_notes,
)


def test_pdf_extractor_page_count():
    extractor = PDFExtractor("BNS bare act 2023.pdf")
    assert extractor.total_pages == 249


def test_pdf_extractor_page_layout():
    extractor = PDFExtractor("BNS bare act 2023.pdf")
    p13 = extractor.get_page_layout(13)
    assert p13.page_number == 13
    assert len(p13.main_column_elements) > 0
    assert len(p13.marginal_note_elements) > 0
    assert "ARREST OF PERSONS" in p13.cleaned_main_text
    assert "35." in p13.cleaned_main_text


def test_marginal_note_clustering():
    extractor = PDFExtractor("BNS bare act 2023.pdf")
    p10 = extractor.get_page_layout(10)
    clusters = cluster_marginal_notes(p10.marginal_note_elements)
    assert len(clusters) >= 3
    titles = [c.title for c in clusters]
    assert any("Sentences which High Courts" in t for t in titles)
    assert any("Sentences which Magistrates" in t for t in titles)


def test_marginal_note_association():
    extractor = PDFExtractor("BNS bare act 2023.pdf")
    p13 = extractor.get_page_layout(13)
    
    # Section 35 starts on page 13 around y=716
    assignments = associate_marginal_notes_with_sections(p13, [("35", 716.4)])
    assert "35" in assignments
    assert "When police may arrest without warrant" in assignments["35"]
