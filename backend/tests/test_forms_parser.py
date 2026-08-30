"""Parser Unit Tests for Second Schedule Statutory Forms (Phase 7).

Tests dynamic boundary detection, placeholder extraction, section references,
and signature/seal extraction across statutory forms.
"""

import pytest
from backend.app.forms.models import FormFieldType
from backend.app.forms.parser import SecondScheduleParser


@pytest.fixture(scope="module")
def parser():
    return SecondScheduleParser("BNS bare act 2023.pdf")


@pytest.fixture(scope="module")
def parsed_forms(parser):
    return parser.parse_forms()


def test_parser_form_1_fields_and_sections(parsed_forms):
    """Test Form 1 (Notice of Appearance) metadata and field parsing."""
    form_1 = parsed_forms[0]
    assert form_1.form_number == 1
    assert form_1.form_title == "NOTICE FOR APPEARANCE BY THE POLICE"
    assert "35(3)" in form_1.applicable_sections
    assert form_1.page_start == 190
    assert form_1.page_end == 190

    # Verify placeholder field detection
    labels = [f.label for f in form_1.fields]
    assert any("Name of the Accused" in l or "Noticee" in l for l in labels)


def test_parser_form_2_summons(parsed_forms):
    """Test Form 2 (Summons to Accused Person) section and fields."""
    form_2 = parsed_forms[1]
    assert form_2.form_number == 2
    assert form_2.form_title == "SUMMONS TO AN ACCUSED PERSON"
    assert "63" in form_2.applicable_sections
    assert form_2.page_start == 191


def test_parser_form_4_bail_bond(parsed_forms):
    """Test Form 4 (Bond and Bail-Bond After Arrest Under Warrant)."""
    form_4 = parsed_forms[3]
    assert form_4.form_number == 4
    assert form_4.form_title == "BOND AND BAIL-BOND AFTER ARREST UNDER A WARRANT"
    assert "83" in form_4.applicable_sections
    assert form_4.page_start == 193


def test_parser_form_33_multi_head_charges(parsed_forms):
    """Test Form 33 (Charges) multi-head structure and section references."""
    form_33 = parsed_forms[32]
    assert form_33.form_number == 33
    assert form_33.form_title == "CHARGES"
    assert set(form_33.applicable_sections).issuperset({"234", "235", "236"})
    assert form_33.page_start == 222
    assert form_33.page_end == 224

    # Verify charge heads extracted
    assert len(form_33.tables) >= 1
    head_titles = [t.head_title for t in form_33.tables]
    assert any("CHARGES WITH ONE-HEAD" in h or "ONE-HEAD" in h for h in head_titles)


def test_parser_form_58_final_form(parsed_forms):
    """Test Form 58 (Imprisonment on Forfeiture of Bond for Good Behaviour)."""
    form_58 = parsed_forms[57]
    assert form_58.form_number == 58
    assert "WARRANT OF IMPRISONMENT ON FORFEITURE OF BOND FOR GOOD BEHAVIOUR" in form_58.form_title
    assert "491" in form_58.applicable_sections
    assert form_58.page_start == 249
    assert form_58.page_end == 249
