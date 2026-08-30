"""AST Citation Validation Tests for Statutory Forms (Phase 7).

Tests:
1. Form citation parsing for [BNSS Second Schedule, Form X]
2. Valid citation validation against retrieved form context
3. Rejection of hallucinated form citations ([BNSS Second Schedule, Form 99])
4. Rejection of valid form citations not present in retrieved context
"""

import pytest
from backend.app.forms.citation_validator import (
    FormCitationParser,
    FormCitationValidator,
)
from backend.app.forms.repository import get_form_registry


@pytest.fixture(scope="module")
def validator():
    registry = get_form_registry("BNS bare act 2023.pdf")
    return FormCitationValidator(registry=registry)


@pytest.fixture(scope="module")
def registry():
    return get_form_registry("BNS bare act 2023.pdf")


def test_form_citation_parser_variations():
    """Test parser extracts various standard statutory form citation formats."""
    text = (
        "As stated in [BNSS Second Schedule, Form 1], a police notice requires attendance. "
        "Furthermore, see [BNSS Form 33] and [Second Schedule, Form 4]."
    )
    citations = FormCitationParser.parse_citations(text)
    assert len(citations) == 3
    assert citations[0].form_number == 1
    assert citations[1].form_number == 33
    assert citations[2].form_number == 4


def test_validator_valid_grounded_citation(validator, registry):
    """Test valid citation matching retrieved form context passes validation."""
    form_1 = registry.get_by_number(1)
    answer = "According to [BNSS Second Schedule, Form 1], the notice directs the accused to appear."
    val_status = validator.validate(answer, [form_1])

    assert val_status.is_valid is True
    assert val_status.valid_citations_count == 1
    assert len(val_status.verified_citations) == 1
    assert val_status.verified_citations[0].citation_text == "[BNSS Second Schedule, Form 1]"


def test_validator_hallucinated_form_99_rejected(validator, registry):
    """Test non-existent form citation [BNSS Second Schedule, Form 99] is rejected."""
    form_1 = registry.get_by_number(1)
    answer = "The procedure is described in [BNSS Second Schedule, Form 99]."
    val_status = validator.validate(answer, [form_1])

    assert val_status.is_valid is False
    assert val_status.invalid_citations_count == 1
    assert any("does not exist" in r for r in val_status.failure_reasons)


def test_validator_unretrieved_valid_form_rejected(validator, registry):
    """Test valid form citation (e.g. Form 4) rejected when only Form 1 was in context."""
    form_1 = registry.get_by_number(1)
    answer = "The bail bond rules are in [BNSS Second Schedule, Form 4]."
    val_status = validator.validate(answer, [form_1])  # Form 4 not retrieved

    assert val_status.is_valid is False
    assert val_status.invalid_citations_count == 1
    assert any("NOT present in retrieved evidence" in r for r in val_status.failure_reasons)
