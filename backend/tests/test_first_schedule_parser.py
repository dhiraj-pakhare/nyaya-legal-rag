"""Unit tests for First Schedule BNS offences table parser."""

import pytest
from backend.app.ingestion.first_schedule_parser import FirstScheduleParser


@pytest.fixture(scope="module")
def schedule_entries():
    parser = FirstScheduleParser("BNS bare act 2023.pdf")
    return parser.parse_schedule(start_page=158, end_page=189)


def test_schedule_entry_count(schedule_entries):
    assert len(schedule_entries) >= 400


def test_schedule_specific_offences(schedule_entries):
    entry_map = {e.section_number: e for e in schedule_entries}
    
    # Section 105: Culpable homicide not amounting to murder
    assert "105" in entry_map
    e105 = entry_map["105"]
    assert "Culpable homicide" in e105.offence_name or "culpable homicide" in e105.raw_text.lower()
    assert "Cognizable" in e105.cognizable_status
    assert "Non-bailable" in e105.bailable_status
    assert "Court of Session" in e105.triable_court
    
    # Section 281: Rash driving
    assert "281" in entry_map
    e281 = entry_map["281"]
    assert "Rash driving" in e281.offence_name or "rash driving" in e281.raw_text.lower()
    assert "Bailable" in e281.bailable_status
