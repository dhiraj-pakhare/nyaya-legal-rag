"""Unit Tests for Statutory Form Domain Models and Pydantic Validation (Phase 7)."""

import pytest
from backend.app.forms.models import (
    FormField,
    FormFieldType,
    FormLookupResponse,
    FormSignature,
    FormTableHead,
    StatutoryForm,
)


def test_statutory_form_model_instantiation_and_serialization():
    """Test StatutoryForm model properties and dict roundtrip."""
    form = StatutoryForm(
        form_id="BNSS_FORM_01",
        form_number=1,
        form_title="NOTICE FOR APPEARANCE BY THE POLICE",
        applicable_sections=["35(3)"],
        page_start=190,
        page_end=190,
        raw_text="Sample text for Form 1",
        fields=[
            FormField(
                field_id="f1",
                label="Name of Noticee",
                field_type=FormFieldType.TEXT_PLACEHOLDER,
                raw_text="To, ........... [Name of Noticee]",
                placeholder="..........."
            )
        ],
        signatures=[
            FormSignature(
                signatory_title="Police Officer",
                seal_required=False
            )
        ],
        provenance_citation="[BNSS Second Schedule, Form 1]"
    )

    data = form.model_dump()
    assert data["form_id"] == "BNSS_FORM_01"
    assert data["form_number"] == 1
    assert data["provenance_citation"] == "[BNSS Second Schedule, Form 1]"
    assert len(data["fields"]) == 1
    assert data["fields"][0]["field_type"] == "text_placeholder"

    # Reconstruct from dict
    reconstructed = StatutoryForm.model_validate(data)
    assert reconstructed.form_id == form.form_id
    assert reconstructed.form_number == form.form_number


def test_form_lookup_response_models():
    """Test FormLookupResponse model for success and refusal scenarios."""
    success_resp = FormLookupResponse(
        status="SUCCESS",
        query="Form 1",
        provenance="[BNSS Second Schedule, Form 1]",
        latency_ms=0.45
    )
    assert success_resp.status == "SUCCESS"
    assert success_resp.is_refused is False

    refused_resp = FormLookupResponse(
        status="NOT_FOUND",
        query="Form 99",
        is_refused=True,
        refusal_reason="Form 99 does not exist.",
        latency_ms=0.25
    )
    assert refused_resp.status == "NOT_FOUND"
    assert refused_resp.is_refused is True
