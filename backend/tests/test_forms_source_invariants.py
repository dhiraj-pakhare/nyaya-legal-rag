"""Source Integrity Invariant Tests for Statutory Forms (Phase 7).

Verifies that the parser enforces strict mathematical invariants over the actual
source PDF (BNS bare act 2023.pdf) without hardcoded assumptions or dropped content.
"""

import pytest
from backend.app.forms.models import StatutoryForm
from backend.app.forms.parser import InvariantValidationError, SecondScheduleParser


@pytest.fixture(scope="module")
def parsed_forms():
    parser = SecondScheduleParser("BNS bare act 2023.pdf")
    return parser.parse_forms()


def test_invariants_exact_58_forms(parsed_forms):
    """Invariant 1: Second Schedule must contain exactly 58 forms."""
    assert len(parsed_forms) == 58


def test_invariants_contiguous_numbering_1_to_58(parsed_forms):
    """Invariant 2: Form numbers must be strictly contiguous from 1 to 58."""
    numbers = [f.form_number for f in parsed_forms]
    assert numbers == list(range(1, 59))


def test_invariants_no_missing_or_duplicate_form_ids(parsed_forms):
    """Invariant 3: No duplicate or missing form IDs."""
    form_ids = [f.form_id for f in parsed_forms]
    assert len(form_ids) == 58
    assert len(set(form_ids)) == 58
    assert form_ids[0] == "BNSS_FORM_01"
    assert form_ids[-1] == "BNSS_FORM_58"


def test_invariants_page_boundaries_and_provenance(parsed_forms):
    """Invariant 4: Form 1 starts on page 190 and Form 58 is on page 249."""
    form_1 = parsed_forms[0]
    form_58 = parsed_forms[-1]

    assert form_1.form_number == 1
    assert form_1.page_start == 190
    assert form_1.provenance_citation == "[BNSS Second Schedule, Form 1]"

    assert form_58.form_number == 58
    assert form_58.page_start == 249
    assert form_58.provenance_citation == "[BNSS Second Schedule, Form 58]"


def test_invariants_multi_page_form_33_continuity(parsed_forms):
    """Invariant 5: Form 33 (Charges) spans pages 222 to 224 without interruption."""
    form_33 = next((f for f in parsed_forms if f.form_number == 33), None)
    assert form_33 is not None
    assert form_33.form_title == "CHARGES"
    assert form_33.page_start == 222
    assert form_33.page_end == 224
    assert len(form_33.raw_text) > 1000  # Multi-page form content
    assert len(form_33.tables) >= 1      # Multi-head charge tables parsed


def test_invariants_preamble_and_boilerplate_exclusion(parsed_forms):
    """Invariant 6: Schedule preamble and Gazette publication boilerplate must not leak into form text."""
    form_1 = parsed_forms[0]
    assert "THE SECOND SCHEDULE" not in form_1.raw_text
    assert "(See section 522)" not in form_1.raw_text

    form_58 = parsed_forms[-1]
    assert "DIWAKAR SINGH" not in form_58.raw_text
    assert "UPLOADED BY THE MANAGER" not in form_58.raw_text


def test_invariants_all_forms_non_empty_and_valid_pages(parsed_forms):
    """Invariant 7: Every form must have valid titles, non-empty text, and page_start <= page_end."""
    for f in parsed_forms:
        assert len(f.form_title.strip()) >= 3
        assert len(f.raw_text.strip()) >= 50
        assert 190 <= f.page_start <= 249
        assert 190 <= f.page_end <= 249
        assert f.page_start <= f.page_end


def test_invariants_validation_failure_loudly_raises():
    """Invariant 8: Programmatic validation must loudly fail if corrupted forms list is supplied."""
    fake_forms = [
        StatutoryForm(
            form_id="BNSS_FORM_01",
            form_number=1,
            form_title="Fake Form",
            page_start=190,
            page_end=190,
            raw_text="Short",
            provenance_citation="[BNSS Second Schedule, Form 1]"
        )
    ]
    with pytest.raises(InvariantValidationError):
        SecondScheduleParser.validate_invariants(fake_forms)
