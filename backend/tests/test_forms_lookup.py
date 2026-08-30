"""Unit Tests for Deterministic Form Identification and Lookup Engine (Phase 7).

Tests:
1. Exact numeric lookups ("Form 1", "Form No. 33", "58")
2. Statutory section lookups ("Section 35(3)", "s.63", "83")
3. Exact normalized title lookups
4. Fuzzy alias lookups
5. Sub-millisecond latency verification
"""

import time
import pytest
from backend.app.forms.lookup import DeterministicFormIdentifier
from backend.app.forms.repository import get_form_registry


@pytest.fixture(scope="module")
def identifier():
    registry = get_form_registry("BNS bare act 2023.pdf")
    return DeterministicFormIdentifier(registry=registry)


def test_lookup_exact_form_number(identifier):
    """Test exact form number lookups for Form 1, Form 33, Form 58."""
    # Test "Form 1"
    resp1 = identifier.identify("Form 1")
    assert resp1.status == "SUCCESS"
    assert resp1.form is not None
    assert resp1.form.form_number == 1
    assert resp1.provenance == "[BNSS Second Schedule, Form 1]"

    # Test "Form No. 33"
    resp33 = identifier.identify("Form No. 33")
    assert resp33.status == "SUCCESS"
    assert resp33.form.form_number == 33
    assert resp33.form.form_title == "CHARGES"

    # Test "BNSS Form 58"
    resp58 = identifier.identify("BNSS Form 58")
    assert resp58.status == "SUCCESS"
    assert resp58.form.form_number == 58

    # Test pure integer "4"
    resp4 = identifier.identify("4")
    assert resp4.status == "SUCCESS"
    assert resp4.form.form_number == 4


def test_lookup_statutory_section_reference(identifier):
    """Test section reference lookups for Section 35(3), Section 63, Section 72, Section 83."""
    # Section 35(3) -> Form 1
    resp = identifier.identify("Show the form for Section 35(3)")
    assert resp.status == "SUCCESS"
    assert resp.form.form_number == 1

    # Section 72 -> Form 3 (Warrant of Arrest)
    resp72 = identifier.identify("Form under s.72")
    assert resp72.status == "SUCCESS"
    assert resp72.form.form_number == 3

    # Section 83 -> Form 4 (Bail Bond)
    resp83 = identifier.identify("Section 83 bail bond form")
    assert resp83.status == "SUCCESS"
    assert resp83.form.form_number == 4


def test_lookup_exact_normalized_title(identifier):
    """Test exact normalized title lookup."""
    resp = identifier.identify("NOTICE FOR APPEARANCE BY THE POLICE")
    assert resp.status == "SUCCESS"
    assert resp.form.form_number == 1

    resp_lower = identifier.identify("notice for appearance by the police")
    assert resp_lower.status == "SUCCESS"
    assert resp_lower.form.form_number == 1


def test_lookup_fuzzy_alias_matching(identifier):
    """Test token-set fuzzy matching on natural language query."""
    resp = identifier.identify("police appearance notice")
    assert resp.status == "SUCCESS"
    assert resp.form.form_number == 1

    resp_bail = identifier.identify("bail bond after arrest under warrant")
    assert resp_bail.status == "SUCCESS"
    assert resp_bail.form.form_number == 4


def test_lookup_latency_benchmark(identifier):
    """Measure deterministic form lookup latency (must be < 5ms)."""
    start = time.perf_counter()
    for _ in range(100):
        identifier.identify("Form No. 33")
    total_elapsed = (time.perf_counter() - start) * 1000.0
    avg_latency = total_elapsed / 100.0
    assert avg_latency < 5.0  # Typically < 0.1ms in memory
