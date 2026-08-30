"""End-to-End Pipeline Tests for Statutory Forms (Phase 7).

Tests full pipeline execution:
1. Deterministic Form Lookups (Form 1, Form 33, Form 58, Section 35(3))
2. Markdown rendering verification
3. Ambiguity and Refusal handling
4. Conversational QA with AST Citation Validation and 1-time Regeneration Guard
"""

import pytest
from backend.app.forms.pipeline import StatutoryFormPipeline
from backend.app.forms.repository import get_form_registry
from backend.app.generation.providers import MockLLMProvider


@pytest.fixture(scope="function")
def forms_pipeline():
    registry = get_form_registry("BNS bare act 2023.pdf")
    mock_llm = MockLLMProvider()
    pipeline = StatutoryFormPipeline(registry=registry, llm_provider=mock_llm)
    return pipeline, mock_llm, registry


def test_e2e_deterministic_form_1_lookup_and_rendering(forms_pipeline):
    """Scenario 1: Direct Form 1 lookup returns structured form and markdown without LLM."""
    pipeline, mock_llm, registry = forms_pipeline
    resp = pipeline.lookup("Form 1")

    assert resp.status == "SUCCESS"
    assert resp.form is not None
    assert resp.form.form_number == 1
    assert resp.provenance == "[BNSS Second Schedule, Form 1]"
    assert resp.rendered_markdown is not None
    assert "# FORM No. 1" in resp.rendered_markdown
    assert "NOTICE FOR APPEARANCE BY THE POLICE" in resp.rendered_markdown
    assert len(mock_llm.call_history) == 0  # Zero LLM token cost


def test_e2e_deterministic_form_33_multi_page(forms_pipeline):
    """Scenario 2: Form 33 (Charges) lookup spanning pages 222-224."""
    pipeline, mock_llm, registry = forms_pipeline
    resp = pipeline.lookup("Form No. 33")

    assert resp.status == "SUCCESS"
    assert resp.form.form_number == 33
    assert resp.form.page_start == 222
    assert resp.form.page_end == 224
    assert len(resp.form.tables) >= 1
    assert "Head I: CHARGES WITH ONE-HEAD" in resp.rendered_markdown or "Charge Heads" in resp.rendered_markdown


def test_e2e_deterministic_form_58_final_form(forms_pipeline):
    """Scenario 3: Form 58 on Page 249."""
    pipeline, mock_llm, registry = forms_pipeline
    resp = pipeline.lookup("BNSS Form 58")

    assert resp.status == "SUCCESS"
    assert resp.form.form_number == 58
    assert resp.form.page_start == 249
    assert resp.form.page_end == 249


def test_e2e_section_reference_lookup(forms_pipeline):
    """Scenario 4: Section 35(3) lookup resolves to Form 1."""
    pipeline, mock_llm, registry = forms_pipeline
    resp = pipeline.lookup("What is the statutory form for Section 35(3)?")

    assert resp.status == "SUCCESS"
    assert resp.form.form_number == 1
    assert resp.provenance == "[BNSS Second Schedule, Form 1]"


def test_e2e_nonexistent_form_99_refusal(forms_pipeline):
    """Scenario 5: Form 99 query cleanly refuses without LLM."""
    pipeline, mock_llm, registry = forms_pipeline
    resp = pipeline.lookup("Form 99")

    assert resp.status == "NOT_FOUND"
    assert resp.is_refused is True
    assert resp.form is None
    assert len(mock_llm.call_history) == 0


def test_e2e_conversational_qa_valid_citation(forms_pipeline):
    """Scenario 6: Grounded conversational QA with valid citation passes AST validation."""
    pipeline, mock_llm, registry = forms_pipeline
    mock_llm.set_canned_response(
        "According to [BNSS Second Schedule, Form 1], the police notice requires the accused to appear at the specified date and time."
    )

    resp = pipeline.query("What does a police notice of appearance require under Form 1?")

    assert resp.status == "SUCCESS"
    assert resp.answer is not None
    assert len(resp.citations) == 1
    assert resp.citations[0].citation_text == "[BNSS Second Schedule, Form 1]"
    assert resp.citations[0].is_verified is True
    assert len(mock_llm.call_history) == 1


def test_e2e_conversational_qa_hallucination_regeneration(forms_pipeline):
    """Scenario 7: Hallucinated citation triggers 1 regeneration pass and succeeds."""
    pipeline, mock_llm, registry = forms_pipeline
    mock_llm.set_response_queue([
        "Details are in [BNSS Second Schedule, Form 99].",  # Hallucinated
        "Details are in [BNSS Second Schedule, Form 1]."   # Correct
    ])

    resp = pipeline.query("Explain the requirements of Form 1.")

    assert resp.status == "SUCCESS"
    assert resp.regeneration_attempted is True
    assert len(resp.citations) == 1
    assert resp.citations[0].citation_text == "[BNSS Second Schedule, Form 1]"
    assert len(mock_llm.call_history) == 2


def test_e2e_conversational_qa_double_hallucination_refusal(forms_pipeline):
    """Scenario 8: Double invalid citation results in clean refusal with answer=None."""
    pipeline, mock_llm, registry = forms_pipeline
    mock_llm.set_response_queue([
        "Details are in [BNSS Second Schedule, Form 88].",  # Hallucinated 1
        "Details are in [BNSS Second Schedule, Form 99]."   # Hallucinated 2
    ])

    resp = pipeline.query("Explain the requirements of Form 1.")

    assert resp.status == "VALIDATION_FAILED"
    assert resp.is_refused is True
    assert resp.answer is None
    assert resp.regeneration_attempted is True
    assert len(mock_llm.call_history) == 2
