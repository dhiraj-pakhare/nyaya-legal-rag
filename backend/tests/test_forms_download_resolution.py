"""Regression & Integration Tests for Statutory Form PDF Download & Source Resolution.

Verifies:
1. Source PDF resolution via configured path, environment variable, and container paths
2. Form 2 ("Summons to an Accused Person") successful download and valid PDF bytes
3. BNSS-backed form (Form 1, Form 2) and BNS-backed form (Form 33 "Charges") downloads
4. Dynamic extraction fallback when disk cache is empty
5. Missing source PDF cleanly returns HTTP 404 (NotFoundError) and never HTTP 500
6. Manifest compatibility with exported form PDFs
"""

import io
import os
import tempfile
import pypdf
import pytest

from backend.app.api.errors import NotFoundError
from backend.app.core.config import settings
from backend.app.main import create_app
from backend.app.services.forms_service import StatutoryFormsService, get_forms_service
from backend.tests.api_client import TestAPIClient, create_in_memory_test_services


@pytest.fixture
def api_client(monkeypatch):
    monkeypatch.setattr(settings, "auth_mode", "dev")
    create_in_memory_test_services()
    app = create_app()
    return TestAPIClient(app)


def test_source_pdf_resolution_configured_path():
    """Verify StatutoryFormsService.resolve_source_pdf respects explicit and configured paths."""
    resolved = StatutoryFormsService.resolve_source_pdf("BNS bare act 2023.pdf")
    assert os.path.isabs(resolved)
    assert os.path.exists(resolved)
    assert resolved.endswith("BNS bare act 2023.pdf")


def test_source_pdf_resolution_fallback_to_default():
    """Verify that an unresolvable path falls back to the provided string without raising."""
    nonexistent = "nonexistent_statutory_source.pdf"
    fallback = StatutoryFormsService.resolve_source_pdf(nonexistent)
    assert fallback == nonexistent


def test_form_2_download_success(api_client):
    """Verify GET /api/v1/forms/2/download returns HTTP 200 with valid binary PDF."""
    resp = api_client.get("/api/v1/forms/2/download")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/pdf"
    assert "attachment; filename=" in resp.headers["content-disposition"]
    assert "FORM-2_Summons-to-an-Accused-Person.pdf" in resp.headers["content-disposition"]

    # Verify PDF validity with pypdf
    pdf_reader = pypdf.PdfReader(io.BytesIO(resp.content))
    assert len(pdf_reader.pages) == 1
    page_text = pdf_reader.pages[0].extract_text()
    assert "SUMMONS" in page_text.upper()
    assert "ACCUSED" in page_text.upper()


def test_bnss_and_bns_backed_forms_download(api_client):
    """Verify download for procedural BNSS form (Form 1) and substantive BNS charge form (Form 33)."""
    # 1. BNSS procedural form (Form 1: Notice for Appearance by Police, Sec 35(3))
    resp1 = api_client.get("/api/v1/forms/1/download")
    assert resp1.status_code == 200
    assert resp1.headers["content-type"] == "application/pdf"
    assert "FORM-1_Notice-for-Appearance-by-the-Police.pdf" in resp1.headers["content-disposition"]
    reader1 = pypdf.PdfReader(io.BytesIO(resp1.content))
    assert len(reader1.pages) == 1

    # 2. BNS charge form (Form 33: Charges under BNS penal offences, 3 pages)
    resp33 = api_client.get("/api/v1/forms/33/download")
    assert resp33.status_code == 200
    assert resp33.headers["content-type"] == "application/pdf"
    assert "FORM-33_Charges.pdf" in resp33.headers["content-disposition"]
    reader33 = pypdf.PdfReader(io.BytesIO(resp33.content))
    assert len(reader33.pages) == 3


def test_dynamic_extraction_when_cache_empty():
    """Verify on-demand extraction extracts from source PDF and caches when disk cache is empty."""
    with tempfile.TemporaryDirectory() as empty_dir:
        service = StatutoryFormsService(
            forms_dir=empty_dir,
            source_pdf_path="BNS bare act 2023.pdf"
        )
        # Disk cache is currently empty
        assert len(os.listdir(empty_dir)) == 0

        # Request Form 2
        filename, pdf_bytes = service.get_form_pdf_bytes("2")
        assert filename == "FORM-2_Summons-to-an-Accused-Person.pdf"
        assert len(pdf_bytes) > 1000

        # Verify PDF validity
        reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
        assert len(reader.pages) == 1

        # Verify opportunistically cached to empty_dir
        cached_file = os.path.join(empty_dir, filename)
        assert os.path.exists(cached_file)
        with open(cached_file, "rb") as f:
            assert f.read() == pdf_bytes


def test_missing_source_pdf_returns_404_not_500():
    """Verify that when source PDF is genuinely unavailable, NotFoundError (404) is raised, never a 500."""
    with tempfile.TemporaryDirectory() as empty_dir:
        service = StatutoryFormsService(
            forms_dir=empty_dir,
            source_pdf_path="/path/to/missing_source.pdf"
        )

        with pytest.raises(NotFoundError) as exc_info:
            service.get_form_pdf_bytes("2")

        err = exc_info.value
        assert err.status_code == 404
        assert "Source statutory PDF unavailable for export" in err.message
        assert err.details.get("source_pdf") == "/path/to/missing_source.pdf"


def test_forms_manifest_compatibility_with_generated_pdfs():
    """Verify that forms_manifest.json contains 58 forms compatible with StatutoryFormsService."""
    service = get_forms_service()
    list_dto = service.list_forms()
    assert list_dto.total_forms == 58
    assert len(list_dto.forms) == 58

    # Form 2 check
    f2 = [f for f in list_dto.forms if f.form_number == 2][0]
    assert f2.form_number == 2
    assert "SUMMONS" in f2.title.upper()
    assert f2.page_count == 1
    assert "/api/v1/forms/2/download" in f2.download_url

    # Form 33 check (multi-page)
    f33 = [f for f in list_dto.forms if f.form_number == 33][0]
    assert f33.form_number == 33
    assert f33.page_count == 3
    assert f33.page_start == 222
    assert f33.page_end == 224
    assert "/api/v1/forms/33/download" in f33.download_url
