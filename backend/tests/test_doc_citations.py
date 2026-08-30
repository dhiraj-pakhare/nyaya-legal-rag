"""Tests for Dual-Domain Citation Parser and AST Validator."""

from backend.app.document_rag.citation_validator import DualCitationParser, DualCitationValidator
from backend.app.document_rag.models import UserDocumentChunk
from backend.app.ingestion.models import StatutoryChunk


def test_dual_citation_parser_extracts_both_types():
    """Test extracting statutory and document citations."""
    text = (
        "According to the notice [DOC p.2], the party committed extortion. "
        "Under [BNS s.308(2)], extortion is punishable with imprisonment."
    )
    parser = DualCitationParser()
    citations = parser.parse_all(text)

    assert len(citations) == 2
    doc_cite = next(c for c in citations if c.citation_type == "USER_DOCUMENT")
    stat_cite = next(c for c in citations if c.citation_type == "STATUTE")

    assert doc_cite.page_number == 2
    assert doc_cite.canonical_tag == "[DOC p.2]"

    assert stat_cite.act_short == "BNS"
    assert stat_cite.section_number == "308"
    assert stat_cite.subsection == "(2)"


def test_dual_citation_validator_valid_document_and_statute():
    """Test validating both document and statutory citations against evidence pools."""
    validator = DualCitationValidator()

    stat_chunks = [
        StatutoryChunk(
            chunk_id="BNS_s103_p158",
            act="Bharatiya Nyaya Sanhita, 2023",
            act_short="BNS",
            chapter="Chapter VI",
            chapter_title="Of Offences Affecting the Human Body",
            section_number="103",
            section_title="Punishment for murder",
            text="Whoever commits murder shall be punished with death or imprisonment for life.",
            pages="158",
            page_start=158,
            page_end=158
        )
    ]
    doc_chunks = [
        UserDocumentChunk(
            chunk_id="doc_1_p2_c1",
            document_id="doc_1",
            user_id="user_1",
            filename="FIR.pdf",
            page_start=2,
            page_end=2,
            chunk_index=1,
            text="The suspect attacked the victim with intent to cause fatal injury.",
            token_count=11
        )
    ]

    answer = (
        "The FIR states that the suspect attacked with fatal intent [DOC p.2]. "
        "Under [BNS s.103], whoever commits murder shall be punished with death or imprisonment for life."
    )

    status = validator.validate(answer, stat_chunks, doc_chunks)
    assert status.is_valid is True
    assert len(status.verified_citations) == 2
    assert len(status.invalid_citations) == 0


def test_dual_citation_validator_invalid_document_page_fails():
    """Test that citing a non-existent document page fails validation."""
    validator = DualCitationValidator()

    doc_chunks = [
        UserDocumentChunk(
            chunk_id="doc_1_p2_c1",
            document_id="doc_1",
            user_id="user_1",
            filename="FIR.pdf",
            page_start=2,
            page_end=2,
            chunk_index=1,
            text="Page 2 text.",
            token_count=3
        )
    ]

    # LLM hallucinates page 99
    answer = "The notice demands payment of arrears [DOC p.99]."
    status = validator.validate(answer, statutory_evidence=[], document_evidence=doc_chunks)

    assert status.is_valid is False
    assert len(status.invalid_citations) == 1
    assert "DOC p.99" in status.invalid_citations[0]["citation"]
    assert "does not exist in retrieved document evidence" in status.failure_reasons[0]


def test_dual_citation_validator_uncited_claims_detected():
    """Test that legal assertions without citations trigger validation failure."""
    validator = DualCitationValidator()
    doc_chunks = [
        UserDocumentChunk(
            chunk_id="doc_1_p1_c1",
            document_id="doc_1",
            user_id="user_1",
            filename="FIR.pdf",
            page_start=1,
            page_end=1,
            chunk_index=1,
            text="Allegation of fraud.",
            token_count=3
        )
    ]

    # No citation tags at all
    answer = "The accused is liable for cheating and shall be punished with rigorous imprisonment up to seven years."
    status = validator.validate(answer, statutory_evidence=[], document_evidence=doc_chunks)

    assert status.is_valid is False
    assert len(status.uncited_claims_detected) >= 1
