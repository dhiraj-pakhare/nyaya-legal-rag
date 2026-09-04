"""Unit tests for programmatic AST citation and claim validator."""

from backend.app.generation.citation_validator import CitationValidator
from backend.app.retrieval.models import RetrievedDocument


def create_mock_doc(act_short: str, section: str, title: str, text: str, sub: str = None) -> RetrievedDocument:
    act_full = "Bharatiya Nyaya Sanhita, 2023" if act_short == "BNS" else "Bharatiya Nagarik Suraksha Sanhita, 2023"
    return RetrievedDocument(
        chunk_id=f"{act_short}_s{section}_p1",
        act=act_full,
        act_short=act_short,
        chapter="VI",
        chapter_title="OFFENCES",
        section_number=section,
        section_title=title,
        subsection=sub,
        text=text,
        page_start=45,
        page_end=46,
        score=0.95,
        final_rank=1
    )


def test_validator_valid_section_and_claim():
    """Test validation passes for correctly cited statutory claims present in evidence."""
    validator = CitationValidator()
    docs = [
        create_mock_doc(
            "BNS",
            "103",
            "Punishment for murder",
            "103. (1) Whoever commits murder shall be punished with death or imprisonment for life, and shall also be liable to fine.\n(2) When a group of five or more persons...",
            sub="(1)"
        )
    ]
    answer = "Whoever commits murder shall be punished with death or life imprisonment [BNS s.103(1)]."
    
    status = validator.validate(answer, docs)
    assert status.is_valid is True
    assert status.valid_citations_count == 1
    assert status.invalid_citations_count == 0
    assert len(status.failure_reasons) == 0


def test_validator_unsupported_section_fails():
    """Test validation fails when LLM cites a section number not present in evidence."""
    validator = CitationValidator()
    docs = [
        create_mock_doc("BNS", "103", "Punishment for murder", "Whoever commits murder...")
    ]
    # Hallucinated section 999
    answer = "The punishment for this offence is provided in [BNS s.999]."
    
    status = validator.validate(answer, docs)
    assert status.is_valid is False
    assert status.invalid_citations_count == 1
    assert any("[BNS s.999]" in r for r in status.failure_reasons)


def test_validator_wrong_act_fails():
    """Test validation fails when LLM cites an Act not present in retrieved context."""
    validator = CitationValidator()
    docs = [
        create_mock_doc("BNS", "103", "Punishment for murder", "Whoever commits murder...")
    ]
    # Act cited as BNSS instead of BNS
    answer = "The procedure is described in [BNSS s.103]."
    
    status = validator.validate(answer, docs)
    assert status.is_valid is False
    assert status.invalid_citations_count == 1
    assert any("BNSS" in r for r in status.failure_reasons)


def test_validator_unsupported_subsection_fails():
    """Test validation fails when cited subsection does not exist in chunk text or metadata."""
    validator = CitationValidator()
    docs = [
        create_mock_doc(
            "BNS",
            "103",
            "Punishment for murder",
            "103. (1) Whoever commits murder... (2) Group of five or more...",
            sub="(1)"
        )
    ]
    # Subsection (99) does not exist in text or metadata
    answer = "The special penalty is under [BNS s.103(99)]."
    
    status = validator.validate(answer, docs)
    assert status.is_valid is False
    assert status.invalid_citations_count == 1
    assert any("(99)" in r for r in status.failure_reasons)


def test_validator_uncited_legal_claim_fails():
    """Test validation detects substantive legal claims without inline citations."""
    validator = CitationValidator()
    docs = [
        create_mock_doc("BNS", "103", "Punishment for murder", "Whoever commits murder...")
    ]
    # Substantive penal claim with zero citation
    answer = "Whoever commits murder shall be punished with death or imprisonment for life and fine."
    
    status = validator.validate(answer, docs)
    assert status.is_valid is False
    assert len(status.uncited_claims_detected) > 0


def test_validator_refusal_text_passes():
    """Test that explicit refusal / insufficient evidence statements pass validation without requiring citations."""
    validator = CitationValidator()
    docs = []
    answer = "Insufficient statutory evidence in the retrieved provisions to answer the question."
    
    status = validator.validate(answer, docs)
    assert status.is_valid is True
    assert status.checked_citations_count == 0


def test_validator_quoted_statutory_text_with_internal_periods():
    """Verify that statutory quotations containing internal periods are not split into false uncited claims."""
    validator = CitationValidator()
    docs = [
        create_mock_doc(
            "BNS",
            "303(2)",
            "Theft. Cognizable. Non-bailable. Any Magistrate",
            "Bharatiya Nyaya Sanhita, 2023 (BNS) Section 303(2)\nOffence: Theft. Cognizable. Non-bailable. Any Magistrate\nPunishment: Rigorous imprisonment for not be less than"
        )
    ]
    # Single sentence with quoted schedule text containing multiple periods and one citation
    answer = 'Section 303(2) of the BNS states: "Offence: Theft. Cognizable. Non-bailable. Any Magistrate" [BNS s.303(2)].'

    status = validator.validate(answer, docs)
    assert status.is_valid is True
    assert len(status.uncited_claims_detected) == 0
    assert status.valid_citations_count == 1


def test_validator_normal_sentences_still_split_normally():
    """Verify that unquoted sentences separated by periods are still split normally."""
    validator = CitationValidator()
    # Test sentence splitting directly
    sentences = validator._split_into_sentences("Theft is an offence. It is cognizable.")
    assert len(sentences) == 2
    assert sentences[0] == "Theft is an offence."
    assert sentences[1] == "It is cognizable."

    # In validation, each separate legal claim sentence must still have a citation
    docs = [create_mock_doc("BNS", "303", "Theft", "Text...")]
    # Sentence 1 has citation, Sentence 2 does not
    answer = "Theft is an offence [BNS s.303]. It is punishable with imprisonment."
    status = validator.validate(answer, docs)
    assert status.is_valid is False
    assert len(status.uncited_claims_detected) == 1
    assert "punishable with imprisonment" in status.uncited_claims_detected[0]


def test_validator_bracket_with_title_valid():
    """Verify that bracketed citations containing section title like [BNS s.187: Mint] are valid."""
    validator = CitationValidator()
    docs = [create_mock_doc("BNS", "187", "Person employed in Mint", "Text...", sub=None)]
    answer = "[BNS s.187: Person employed in Mint] Offence is cognizable and punishable with imprisonment."
    status = validator.validate(answer, docs)
    assert status.is_valid is True
    assert status.valid_citations_count == 1
    assert len(status.uncited_claims_detected) == 0


def test_validator_bns_bnss_act_mismatch_rejected():
    """Verify that citing BNSS when evidence is BNS is strictly rejected."""
    validator = CitationValidator()
    docs = [create_mock_doc("BNS", "103", "Murder", "Text...", sub=None)]
    answer = "[BNSS s.103(2)] Offence of murder by group of five is punishable with death."
    status = validator.validate(answer, docs)
    assert status.is_valid is False
    assert status.invalid_citations_count == 1
    assert any("BNSS" in r for r in status.failure_reasons)


def test_validator_doc_citations_valid():
    """Verify that user document citations [DOC p.X] are accepted for non-statutory claims."""
    validator = CitationValidator()
    docs = [create_mock_doc("BNS", "103", "Murder", "Text...", sub=None)]
    answer = "According to clause 4 of the contract [DOC p.3], arbitration is mandatory."
    status = validator.validate(answer, docs)
    assert status.is_valid is True
    assert len(status.uncited_claims_detected) == 0


def test_validator_multisentence_all_cited_valid():
    """Verify that multi-sentence legal answers where each sentence is cited pass cleanly."""
    validator = CitationValidator()
    docs = [
        create_mock_doc("BNS", "103", "Murder", "Text...", sub=None),
        create_mock_doc("BNSS", "187", "Investigation", "Text...", sub=None)
    ]
    answer = "[BNS s.103] Murder is punishable with death or life imprisonment. [BNSS s.187] Investigation must be completed within 24 hours."
    status = validator.validate(answer, docs)
    assert status.is_valid is True
    assert status.valid_citations_count == 2
    assert len(status.uncited_claims_detected) == 0
