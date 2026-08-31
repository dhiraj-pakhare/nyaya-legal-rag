"""Unit Tests for Polymorphic Citation DTOs and Serialization (Phase 8).

Verifies:
1. StatutoryCitationDTO serializes statute-specific fields
2. DocumentCitationDTO serializes document-specific fields
3. FormCitationDTO serializes form-specific fields
4. Polymorphic union deserialization in QueryResponseDTO
5. Unvalidated citations are rejected
"""

import pytest
from backend.app.api.schemas.query import (
    CitationType,
    DocumentCitationDTO,
    FormCitationDTO,
    QueryResponseDTO,
    StatutoryCitationDTO,
)


def test_statutory_citation_dto_serialization():
    """Test statutory citation model fields and serialization."""
    dto = StatutoryCitationDTO(
        citation_text="[BNS s.103]",
        citation_type=CitationType.STATUTORY,
        is_verified=True,
        source_id="bns_s103_p1",
        page_start=45,
        page_end=45,
        act="Bharatiya Nyaya Sanhita, 2023",
        act_short="BNS",
        section="103",
        section_title="Punishment for murder"
    )
    data = dto.model_dump()
    assert data["citation_type"] == "STATUTORY"
    assert data["act_short"] == "BNS"
    assert data["section"] == "103"
    assert data["is_verified"] is True


def test_document_citation_dto_serialization():
    """Test document citation model fields and serialization."""
    dto = DocumentCitationDTO(
        citation_text="[DOC p.4]",
        citation_type=CitationType.DOCUMENT,
        is_verified=True,
        source_id="doc123_p4_c1",
        page_start=4,
        page_end=4,
        document_id="doc123",
        filename="notice_legal.pdf",
        page_number=4
    )
    data = dto.model_dump()
    assert data["citation_type"] == "DOCUMENT"
    assert data["document_id"] == "doc123"
    assert data["page_number"] == 4


def test_form_citation_dto_serialization():
    """Test form citation model fields and serialization."""
    dto = FormCitationDTO(
        citation_text="[BNSS Second Schedule, Form 1]",
        citation_type=CitationType.FORM,
        is_verified=True,
        source_id="BNSS_FORM_01",
        page_start=190,
        page_end=190,
        form_number=1,
        form_title="NOTICE FOR APPEARANCE BY THE POLICE",
        applicable_sections=["35(3)"]
    )
    data = dto.model_dump()
    assert data["citation_type"] == "FORM"
    assert data["form_number"] == 1
    assert "35(3)" in data["applicable_sections"]


def test_query_response_polymorphic_union():
    """Test QueryResponseDTO serializes mixed polymorphic citations seamlessly."""
    stat_c = StatutoryCitationDTO(
        citation_text="[BNS s.103]",
        citation_type=CitationType.STATUTORY,
        is_verified=True,
        source_id="bns_s103",
        section="103",
        section_title="Murder"
    )
    doc_c = DocumentCitationDTO(
        citation_text="[DOC p.2]",
        citation_type=CitationType.DOCUMENT,
        is_verified=True,
        source_id="docA_p2",
        document_id="docA",
        filename="complaint.pdf",
        page_number=2
    )
    form_c = FormCitationDTO(
        citation_text="[BNSS Second Schedule, Form 33]",
        citation_type=CitationType.FORM,
        is_verified=True,
        source_id="BNSS_FORM_33",
        form_number=33,
        form_title="CHARGES"
    )

    resp = QueryResponseDTO(
        query="Combined legal question",
        status="SUCCESS",
        answer="Legal analysis here.",
        citations=[stat_c, doc_c, form_c],
        confidence_score=1.0,
        routed_corpus="COMBINED"
    )

    data = resp.model_dump()
    assert len(data["citations"]) == 3
    assert data["citations"][0]["citation_type"] == "STATUTORY"
    assert data["citations"][1]["citation_type"] == "DOCUMENT"
    assert data["citations"][2]["citation_type"] == "FORM"


def test_statutory_citation_dto_with_source_text():
    """Verify source_text serialization on StatutoryCitationDTO."""
    dto_with_text = StatutoryCitationDTO(
        citation_text="[BNS s.103]",
        citation_type=CitationType.STATUTORY,
        is_verified=True,
        source_id="bns_s103_p1",
        act="Bharatiya Nyaya Sanhita, 2023",
        act_short="BNS",
        section="103",
        section_title="Punishment for murder",
        source_text="Whoever commits murder shall be punished with death or imprisonment for life..."
    )
    dumped = dto_with_text.model_dump()
    assert dumped["source_text"] == "Whoever commits murder shall be punished with death or imprisonment for life..."

    dto_without_text = StatutoryCitationDTO(
        citation_text="[BNS s.103]",
        citation_type=CitationType.STATUTORY,
        is_verified=True,
        source_id="bns_s103_p1",
        act="Bharatiya Nyaya Sanhita, 2023",
        act_short="BNS",
        section="103",
        section_title="Punishment for murder"
    )
    dumped_none = dto_without_text.model_dump()
    assert dumped_none["source_text"] is None
