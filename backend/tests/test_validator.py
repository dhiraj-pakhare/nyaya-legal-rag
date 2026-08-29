"""Unit tests for IngestionValidator and consistency assertions."""

import pytest
from backend.app.ingestion.parser import StatutoryParser
from backend.app.ingestion.validator import IngestionValidator


@pytest.fixture(scope="module")
def parsed_result():
    parser = StatutoryParser("BNS bare act 2023.pdf")
    return parser.parse()


def test_validator_passes_on_full_pdf(parsed_result):
    validator = IngestionValidator()
    report = validator.validate(
        sections=parsed_result.document.sections,
        chunks=parsed_result.chunks,
        total_pages=249
    )
    assert report.is_valid
    assert len(report.missing_sections) == 0
    assert len(report.duplicate_sections) == 0
    assert report.total_chapters == 39
    assert report.total_sections == 531
    assert report.total_chunks == len(parsed_result.chunks)
