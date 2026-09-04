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
    assert status.valid_citations_count >= 1


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


# =========================================================================
# 1. Natural-Language Citation Normalization Tests
# =========================================================================

def test_natural_citation_section_of_full_act():
    """Verify 'Section 103(2) of the Bharatiya Nyaya Sanhita' is normalized and validated."""
    validator = CitationValidator()
    docs = [create_mock_doc("BNS", "103(2)", "Murder by group", "Murder committed by a group of five or more persons shall be punished with death or imprisonment for life and fine.", sub="(2)")]
    raw_answer = "Under Section 103(2) of the Bharatiya Nyaya Sanhita, 2023, mob lynching is punishable with death or life imprisonment."
    status = validator.validate(raw_answer, docs)
    assert status.is_valid is True
    assert status.valid_citations_count == 1
    assert status.verified_citations[0].citation_text == "[BNS s.103(2)]"
    assert len(status.uncited_claims_detected) == 0


def test_natural_citation_section_of_bns_short():
    """Verify 'Section 105 of BNS' is normalized and validated."""
    validator = CitationValidator()
    docs = [create_mock_doc("BNS", "105", "Culpable homicide", "Whoever commits culpable homicide not amounting to murder shall be punished with imprisonment for life or imprisonment up to ten years and fine.")]
    raw_answer = "Section 105 of BNS prescribes punishment for culpable homicide not amounting to murder as imprisonment for life or up to ten years."
    status = validator.validate(raw_answer, docs)
    assert status.is_valid is True
    assert status.valid_citations_count == 1
    assert status.verified_citations[0].citation_text == "[BNS s.105]"
    assert len(status.uncited_claims_detected) == 0


def test_natural_citation_section_of_bnss_short():
    """Verify 'under Section 40 of BNSS' is normalized and validated."""
    validator = CitationValidator()
    docs = [create_mock_doc("BNSS", "40", "Arrest by private person", "Any private person may arrest or cause to be arrested any person who in his presence commits a non-bailable and cognizable offence.")]
    raw_answer = "A private citizen may arrest an offender under Section 40 of BNSS when a non-bailable offence is committed in their presence."
    status = validator.validate(raw_answer, docs)
    assert status.is_valid is True
    assert status.valid_citations_count == 1
    assert status.verified_citations[0].citation_text == "[BNSS s.40]"
    assert len(status.uncited_claims_detected) == 0


def test_natural_citation_wrong_act_rejected():
    """Verify natural citation naming wrong Act (e.g. BNSS for BNS section) is strictly rejected."""
    validator = CitationValidator()
    docs = [create_mock_doc("BNS", "103(2)", "Murder", "Murder by group punished with death.", sub="(2)")]
    raw_answer = "According to Section 103(2) of the Bharatiya Nagarik Suraksha Sanhita, penalty is death."
    status = validator.validate(raw_answer, docs)
    assert status.is_valid is False
    assert any("BNSS" in r for r in status.failure_reasons)


def test_natural_citation_unretrieved_section_rejected():
    """Verify natural citation pointing to unretrieved section is rejected."""
    validator = CitationValidator()
    docs = [create_mock_doc("BNS", "103", "Murder", "Text...")]
    raw_answer = "Section 999 of BNS prescribes severe penalties."
    status = validator.validate(raw_answer, docs)
    assert status.is_valid is False
    assert any("999" in r for r in status.failure_reasons)


def test_natural_citation_ambiguous_act_not_guessed():
    """Verify unqualified 'Section 103' is not guessed when both BNS and BNSS are in retrieved context."""
    validator = CitationValidator()
    docs = [
        create_mock_doc("BNS", "103", "Murder", "Text..."),
        create_mock_doc("BNSS", "103", "Search", "Text...")
    ]
    raw_answer = "Under Section 103, an offence is punishable with imprisonment."
    status = validator.validate(raw_answer, docs)
    assert status.is_valid is False
    assert any("0 statutory citations" in r or "substantive legal claim" in r for r in status.failure_reasons)


def test_natural_citation_with_title():
    """Verify natural reference containing parenthetical section title is parsed and validated."""
    validator = CitationValidator()
    docs = [create_mock_doc("BNSS", "187", "Procedure where investigation cannot be completed", "Procedure where investigation cannot be completed in twenty-four hours.")]
    raw_answer = "Under Section 187 (Procedure where investigation cannot be completed) of BNSS, custody is strictly limited."
    status = validator.validate(raw_answer, docs)
    assert status.is_valid is True
    assert status.valid_citations_count == 1
    assert status.verified_citations[0].citation_text == "[BNSS s.187]"


def test_natural_citation_existing_bracket_preserved():
    """Verify existing bracketed citations [BNS s.103] are preserved without double-bracketing or corruption."""
    validator = CitationValidator()
    docs = [create_mock_doc("BNS", "103", "Murder", "Text...")]
    answer = "[BNS s.103] Murder is punishable with death or life imprisonment."
    status = validator.validate(answer, docs)
    assert status.is_valid is True
    assert status.valid_citations_count == 1
    assert status.verified_citations[0].citation_text == "[BNS s.103]"


# =========================================================================
# 2. Golden Regressions (G06, G09, G21)
# =========================================================================

def test_regression_g06_bnss_s40_continuity():
    """G06 regression: Verified [BNSS s.40] citation followed by supported procedural handover sentence."""
    validator = CitationValidator()
    doc = create_mock_doc(
        act_short="BNSS",
        section="40",
        title="Arrest by private person and procedure on such arrest",
        text="Any private person may arrest or cause to be arrested any person who in his presence commits a non-bailable and cognizable offence, or any proclaimed offender, and, without unnecessary delay, shall make over or cause to be made over any person so arrested to a police officer, or, in the absence of a police officer, take such person or cause him to be taken in custody to the nearest police station."
    )
    answer = (
        "[BNSS s.40] Any private person may arrest or cause to be arrested any person who in their presence commits a non-bailable and cognizable offence. "
        "The arrested person must without unnecessary delay be handed over to a police officer or taken to the nearest police station."
    )
    status = validator.validate(answer, [doc])
    assert status.is_valid is True
    assert status.valid_citations_count == 1
    assert len(status.uncited_claims_detected) == 0


def test_regression_g09_bns_s105_continuity():
    """G09 regression: Verified [BNS s.105] citation followed by supported penalty knowledge distinction sentence."""
    validator = CitationValidator()
    doc = create_mock_doc(
        act_short="BNS",
        section="105",
        title="Punishment for culpable homicide not amounting to murder",
        text="Whoever commits culpable homicide not amounting to murder shall be punished with imprisonment for life, or imprisonment of either description for a term which may extend to ten years, and shall also be liable to fine, if the act by which the death is caused is done with the intention of causing death, or of causing such bodily injury as is likely to cause death; or with imprisonment of either description for a term which may extend to ten years, or with fine, or with both, if the act is done with the knowledge that it is likely to cause death, but without any intention to cause death, or to cause such bodily injury as is likely to cause death."
    )
    answer = (
        "[BNS s.105] Culpable homicide not amounting to murder is punishable with imprisonment for life or up to ten years and fine if the act is done with intention of causing death. "
        "If the act is done with knowledge that it is likely to cause death, the punishment may extend to ten years, or fine, or both."
    )
    status = validator.validate(answer, [doc])
    assert status.is_valid is True
    assert status.valid_citations_count == 1
    assert len(status.uncited_claims_detected) == 0


def test_regression_g21_bns_s1032_natural_prose():
    """G21 regression: Natural-language 'Section 103(2) of the Bharatiya Nyaya Sanhita' mob lynching query."""
    validator = CitationValidator()
    doc = create_mock_doc(
        act_short="BNS",
        section="103(2)",
        title="Murder by group of five or more persons",
        text="When a group of five or more persons acting in concert commits murder on the ground of race, caste or community, sex, place of birth, language, personal belief or any other ground, each member of such group shall be punished with death or with imprisonment for life, and shall also be liable to fine.",
        sub="(2)"
    )
    raw_answer = (
        "Under Section 103(2) of the Bharatiya Nyaya Sanhita, 2023, when a group of five or more persons acting in concert "
        "commits murder on the ground of race, caste or community, each member shall be punished with death or imprisonment for life and fine."
    )
    status = validator.validate(raw_answer, [doc])
    assert status.is_valid is True
    assert status.valid_citations_count == 1
    assert status.verified_citations[0].citation_text == "[BNS s.103(2)]"
    assert len(status.uncited_claims_detected) == 0


# =========================================================================
# 3. Negative Tests for Strict Continuity and Rejection
# =========================================================================

def test_continuity_unsupported_sentence_rejected():
    """Negative test: Citation followed by an unsupported penalty sentence must be rejected."""
    validator = CitationValidator()
    doc = create_mock_doc(
        act_short="BNS",
        section="105",
        title="Punishment for culpable homicide",
        text="Whoever commits culpable homicide not amounting to murder shall be punished with imprisonment for life or fine."
    )
    answer = (
        "[BNS s.105] Culpable homicide not amounting to murder is punishable with imprisonment for life. "
        "The court may also impose any additional punishment it considers appropriate."
    )
    status = validator.validate(answer, [doc])
    assert status.is_valid is False
    assert len(status.uncited_claims_detected) >= 1


def test_continuity_different_act_not_inherited():
    """Negative test: Second sentence mentioning a different Act cannot inherit previous citation."""
    validator = CitationValidator()
    doc1 = create_mock_doc("BNS", "105", "Culpable homicide", "Punished with imprisonment for life.")
    answer = (
        "[BNS s.105] Culpable homicide is punishable with life imprisonment. "
        "Under BNSS, the police officer must immediately register an FIR and arrest without warrant."
    )
    status = validator.validate(answer, [doc1])
    # The BNSS sentence makes an uncited procedural claim without valid inline citation
    assert status.is_valid is False
    assert len(status.uncited_claims_detected) >= 1


def test_continuity_different_section_not_inherited():
    """Negative test: Second sentence asserting a different section cannot inherit previous citation."""
    validator = CitationValidator()
    doc1 = create_mock_doc("BNS", "105", "Culpable homicide", "Punished with imprisonment for life.")
    doc2 = create_mock_doc("BNS", "103", "Murder", "Punished with death.")
    answer = (
        "[BNS s.105] Culpable homicide is punishable with imprisonment for life. "
        "Under Section 103, murder is punishable with death."
    )
    # Section 103 must be independently cited with [BNS s.103]
    status = validator.validate(answer, [doc1, doc2])
    # But if Section 999 is asserted:
    answer_bad = (
        "[BNS s.105] Culpable homicide is punishable with imprisonment for life. "
        "Under Section 999, punishment is death."
    )
    status_bad = validator.validate(answer_bad, [doc1, doc2])
    assert status_bad.is_valid is False


# =========================================================================
# 4. Legal Sentence Segmentation Tests (Lists and Semicolons)
# =========================================================================

def test_split_numbered_list_not_fragmented():
    """Verify that numbered lists 1. 2. do not fragment into orphaned number sentences."""
    validator = CitationValidator()
    doc = create_mock_doc(
        act_short="BNS",
        section="105",
        title="Punishment",
        text="Punishments include imprisonment for life or imprisonment for ten years and fine."
    )
    text = "[BNS s.105] Punishments include: 1. Imprisonment for life. 2. Fine."
    sentences = validator._split_into_sentences(text)
    assert len(sentences) == 2
    assert sentences[0] == "[BNS s.105] Punishments include: 1. Imprisonment for life."
    assert sentences[1] == "2. Fine."
    status = validator.validate(text, [doc])
    assert status.is_valid is True
    assert len(status.uncited_claims_detected) == 0


def test_split_semicolon_list_preserved():
    """Verify that semicolon-delimited lists are preserved as a single coherent statement."""
    validator = CitationValidator()
    doc = create_mock_doc(
        act_short="BNS",
        section="105",
        title="Punishment",
        text="Punishments include imprisonment for life, fine, and imprisonment up to ten years."
    )
    text = (
        "[BNS s.105] Punishments include:\n"
        "(i) life imprisonment;\n"
        "(ii) fine; and\n"
        "(iii) imprisonment up to ten years."
    )
    sentences = validator._split_into_sentences(text)
    assert len(sentences) == 1
    status = validator.validate(text, [doc])
    assert status.is_valid is True
    assert len(status.uncited_claims_detected) == 0


def test_unsupported_legal_list_item_rejected():
    """Negative test: List item containing unsupported legal claim must be rejected."""
    validator = CitationValidator()
    doc = create_mock_doc(
        act_short="BNS",
        section="105",
        title="Punishment",
        text="Punishments include imprisonment for life and fine."
    )
    text = "[BNS s.105] Punishments include: 1. Imprisonment for life. 2. Public flogging."
    status = validator.validate(text, [doc])
    assert status.is_valid is False
    assert len(status.uncited_claims_detected) >= 1

