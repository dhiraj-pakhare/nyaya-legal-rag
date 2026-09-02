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


def test_combined_answer_with_doc_citation_passes_validation():
    """a) A COMBINED answer containing a document fact with [DOC p.1] passes citation validation."""
    validator = DualCitationValidator()
    doc_chunk = UserDocumentChunk(
        chunk_id="resume_p1_c1",
        document_id="doc_resume_123",
        user_id="user_1",
        filename="Dhiraj_Resume.pdf",
        page_start=1,
        page_end=1,
        chunk_index=0,
        text="Candidate Name: Dhiraj Pakhare. Full-stack AI Engineer.",
        token_count=8
    )

    answer = "Based on your resume [DOC p.1], your name is Dhiraj Pakhare."
    status = validator.validate(answer, statutory_evidence=[], document_evidence=[doc_chunk])

    assert status.is_valid is True
    assert len(status.verified_citations) == 1
    assert status.verified_citations[0].citation_text == "[DOC p.1]"
    assert status.verified_citations[0].page_start == 1


def test_combined_answer_without_doc_citation_is_rejected():
    """b) A COMBINED answer containing a document fact with no [DOC p.1] is rejected."""
    validator = DualCitationValidator()
    doc_chunk = UserDocumentChunk(
        chunk_id="resume_p1_c1",
        document_id="doc_resume_123",
        user_id="user_1",
        filename="Dhiraj_Resume.pdf",
        page_start=1,
        page_end=1,
        chunk_index=0,
        text="Candidate Name: Dhiraj Pakhare. Full-stack AI Engineer.",
        token_count=8
    )

    answer = "Based on your resume, your name is Dhiraj Pakhare."
    status = validator.validate(answer, statutory_evidence=[], document_evidence=[doc_chunk])

    assert status.is_valid is False
    assert status.checked_citations_count == 0
    assert any("contains 0 citations" in reason for reason in status.failure_reasons)


def test_statutory_citations_still_pass_validation():
    """c) Pure statutory BNS/BNSS citations continue to pass validation."""
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
            text="103. (1) Whoever commits murder shall be punished with death or imprisonment for life.",
            pages="158",
            page_start=158,
            page_end=158
        ),
        StatutoryChunk(
            chunk_id="BNSS_s35_p22",
            act="Bharatiya Nagarik Suraksha Sanhita, 2023",
            act_short="BNSS",
            chapter="Chapter V",
            chapter_title="Arrest of Persons",
            section_number="35",
            subsection="(1)(c)",
            section_title="When police may arrest without warrant",
            text="35. (1)(c) Any police officer may without an order from a Magistrate and without a warrant arrest any person.",
            pages="22",
            page_start=22,
            page_end=22
        )
    ]

    answer = (
        "Under [BNS s.103], whoever commits murder shall be punished with death or imprisonment for life. "
        "Under [BNSS s.35(1)(c)], a police officer may arrest without an order or warrant."
    )
    status = validator.validate(answer, statutory_evidence=stat_chunks, document_evidence=[])

    assert status.is_valid is True
    assert len(status.verified_citations) == 2
    assert status.verified_citations[0].act_short in ("BNS", "BNSS")
    assert status.verified_citations[1].act_short in ("BNS", "BNSS")


def test_mixed_statutory_and_document_citations_pass_validation():
    """d) Mixed statutory + document citations pass validation together."""
    validator = DualCitationValidator()
    stat_chunk = StatutoryChunk(
        chunk_id="BNS_s303_p210",
        act="Bharatiya Nyaya Sanhita, 2023",
        act_short="BNS",
        chapter="Chapter XVII",
        chapter_title="Of Offences Against Property",
        section_number="303",
        section_title="Theft",
        text="303. (2) Whoever commits theft shall be punished with imprisonment.",
        pages="210",
        page_start=210,
        page_end=210
    )
    doc_chunk = UserDocumentChunk(
        chunk_id="complaint_p1_c1",
        document_id="doc_complaint_456",
        user_id="user_1",
        filename="Complaint.pdf",
        page_start=1,
        page_end=1,
        chunk_index=0,
        text="The complainant reported the theft of jewelry on June 1st.",
        token_count=10
    )

    answer = (
        "The complainant reported theft of jewelry on June 1st [DOC p.1]. "
        "Under [BNS s.303(2)], whoever commits theft shall be punished with imprisonment."
    )
    status = validator.validate(answer, statutory_evidence=[stat_chunk], document_evidence=[doc_chunk])

    assert status.is_valid is True
    assert len(status.verified_citations) == 2
    tags = {c.citation_text for c in status.verified_citations}
    assert "[DOC p.1]" in tags
    assert "[BNS s.303(2)]" in tags


def test_document_only_answers_require_doc_citation():
    """e) DOCUMENT_ONLY answers require valid [DOC p.X] citations and reject uncited answers."""
    validator = DualCitationValidator()
    doc_chunk = UserDocumentChunk(
        chunk_id="project_p3_c1",
        document_id="doc_portfolio_789",
        user_id="user_1",
        filename="Portfolio.pdf",
        page_start=3,
        page_end=3,
        chunk_index=0,
        text="Worked on Nyaya Legal RAG Platform and Distributed Crawlers.",
        token_count=9
    )

    # Valid answer citing page 3
    valid_answer = "You worked on Nyaya Legal RAG Platform [DOC p.3]."
    status_valid = validator.validate(valid_answer, statutory_evidence=[], document_evidence=[doc_chunk])
    assert status_valid.is_valid is True
    assert status_valid.verified_citations[0].citation_text == "[DOC p.3]"

    # Uncited answer
    uncited_answer = "You worked on Nyaya Legal RAG Platform and Distributed Crawlers."
    status_uncited = validator.validate(uncited_answer, statutory_evidence=[], document_evidence=[doc_chunk])
    assert status_uncited.is_valid is False
    assert any("contains 0 citations" in r for r in status_uncited.failure_reasons)


def test_parser_does_not_confuse_doc_with_statutory_citations():
    """f) The parser correctly discriminates [DOC p.X] from statutory citations [BNS s.X] / [BNSS s.X]."""
    parser = DualCitationParser()
    text = (
        "Document pages: [DOC p.1], [DOC p.2], [DOC p.12], [DOC p. 12], [DOC page 4]. "
        "Statutes: [BNS s.103], [BNSS s.103], [BNSS s.35(1)(c)]."
    )
    citations = parser.parse_all(text)

    doc_cites = [c for c in citations if c.citation_type == "USER_DOCUMENT"]
    stat_cites = [c for c in citations if c.citation_type == "STATUTE"]

    assert len(doc_cites) == 5
    assert len(stat_cites) == 3

    # Check document page numbers
    doc_pages = [c.page_number for c in doc_cites]
    assert doc_pages == [1, 2, 12, 12, 4]

    # Check canonical tags
    for dc in doc_cites:
        assert dc.canonical_tag.startswith("[DOC p.")

    # Check statutory citations
    stat_sections = [(c.act_short, c.section_number, c.subsection, c.clause) for c in stat_cites]
    assert ("BNS", "103", None, None) in stat_sections
    assert ("BNSS", "103", None, None) in stat_sections
    assert ("BNSS", "35", "(1)(c)", "(c)") in stat_sections
