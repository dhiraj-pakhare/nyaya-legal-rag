"""Disambiguation and Refusal Tests for Statutory Form Lookup (Phase 7).

Tests:
1. Ambiguous multi-match queries return status="AMBIGUOUS" with candidate lists
2. Non-existent form numbers return status="NOT_FOUND" and is_refused=True
3. Unmapped section queries return status="NOT_FOUND"
"""

import pytest
from backend.app.forms.lookup import DeterministicFormIdentifier
from backend.app.forms.repository import get_form_registry


@pytest.fixture(scope="module")
def identifier():
    registry = get_form_registry("BNS bare act 2023.pdf")
    return DeterministicFormIdentifier(registry=registry)


def test_ambiguous_attachment_warrant_query(identifier):
    """Test query matching multiple forms (e.g. 'attachment warrant') returns AMBIGUOUS with candidates."""
    resp = identifier.identify("attachment warrant")
    assert resp.status == "AMBIGUOUS"
    assert resp.form is None
    assert len(resp.candidate_forms) > 1
    
    # Check that candidate forms include attachment forms (e.g. Form 7, Form 8, Form 27, Form 49)
    candidate_nums = [c["form_number"] for c in resp.candidate_forms]
    assert any(n in candidate_nums for n in [7, 8, 27, 49, 52, 55, 57])


def test_ambiguous_multi_form_section_reference(identifier):
    """Test section reference applying to multiple forms (e.g. Section 85 -> Form 7 & Form 8)."""
    resp = identifier.identify("Form for section 85")
    assert resp.status == "AMBIGUOUS"
    assert resp.form is None
    assert len(resp.candidate_forms) >= 2
    candidate_nums = [c["form_number"] for c in resp.candidate_forms]
    assert 7 in candidate_nums
    assert 8 in candidate_nums


def test_nonexistent_form_99_refusal(identifier):
    """Test non-existent Form 99 returns NOT_FOUND and is_refused=True."""
    resp = identifier.identify("Form 99")
    assert resp.status == "NOT_FOUND"
    assert resp.is_refused is True
    assert resp.form is None
    assert "Form No. 99 does not exist" in resp.refusal_reason


def test_nonexistent_form_0_refusal(identifier):
    """Test Form 0 returns NOT_FOUND and is_refused=True."""
    resp = identifier.identify("Form 0")
    assert resp.status == "NOT_FOUND"
    assert resp.is_refused is True
    assert resp.form is None


def test_unmapped_section_refusal(identifier):
    """Test section with no statutory forms (e.g. Section 9999) returns NOT_FOUND."""
    resp = identifier.identify("Form for section 9999")
    assert resp.status == "NOT_FOUND"
    assert resp.is_refused is True
    assert resp.form is None
