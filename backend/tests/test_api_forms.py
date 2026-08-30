"""Statutory Forms API Integration Tests (Part B & Phase 8).

Verifies:
1. Exact form number lookups (Form 1, Form 33, Form 58)
2. Section reference lookups (Section 35(3))
3. Ambiguous form queries return AMBIGUOUS with candidate list
4. Non-existent Form 99 returns NOT_FOUND refusal
5. Direct form retrieval by ID/number and 404 for invalid numbers
6. GET /api/v1/forms listing all 58 forms
7. GET /api/v1/forms/search?q= query parameter search
8. GET /api/v1/forms/{id}/download individual PDF download
9. GET /api/v1/forms/download-all bulk ZIP download
"""

import io
import zipfile
import pypdf
import pytest

from backend.app.core.config import settings
from backend.app.main import create_app
from backend.tests.api_client import TestAPIClient, create_in_memory_test_services


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(settings, "auth_mode", "dev")
    create_in_memory_test_services()
    app = create_app()
    return TestAPIClient(app)


def test_api_forms_lookup_form_1(client):
    """Test POST /api/v1/forms/lookup for Form 1."""
    resp = client.post("/api/v1/forms/lookup", json={"query": "Form 1"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "SUCCESS"
    assert data["form"]["form_number"] == 1
    assert data["form"]["form_title"] == "NOTICE FOR APPEARANCE BY THE POLICE"
    assert data["provenance"] == "[BNSS Second Schedule, Form 1]"
    assert "# FORM No. 1" in data["rendered_markdown"]


def test_api_forms_lookup_form_33_multi_page(client):
    """Test POST /api/v1/forms/lookup for Form 33 spanning pages 222-224."""
    resp = client.post("/api/v1/forms/lookup", json={"query": "Form No. 33"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "SUCCESS"
    assert data["form"]["form_number"] == 33
    assert data["form"]["page_start"] == 222
    assert data["form"]["page_end"] == 224
    assert len(data["form"]["tables"]) >= 1


def test_api_forms_lookup_section_35(client):
    """Test POST /api/v1/forms/lookup for statutory section reference."""
    resp = client.post("/api/v1/forms/lookup", json={"query": "Section 35(3)"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "SUCCESS"
    assert data["form"]["form_number"] == 1


def test_api_forms_lookup_ambiguous_attachment(client):
    """Test POST /api/v1/forms/lookup for ambiguous query returns candidate list."""
    resp = client.post("/api/v1/forms/lookup", json={"query": "Attachment warrant"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "AMBIGUOUS"
    assert data["form"] is None
    assert len(data["candidate_forms"]) > 1


def test_api_forms_lookup_nonexistent_form_99_refusal(client):
    """Test POST /api/v1/forms/lookup for Form 99 returns NOT_FOUND."""
    resp = client.post("/api/v1/forms/lookup", json={"query": "Form 99"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "NOT_FOUND"
    assert data["is_refused"] is True
    assert "Form No. 99 does not exist" in data["refusal_reason"]


def test_api_forms_direct_get_by_id_and_number(client):
    """Test GET /api/v1/forms/{id_or_number}."""
    resp_num = client.get("/api/v1/forms/1")
    assert resp_num.status_code == 200
    assert resp_num.json()["form_number"] == 1

    resp_id = client.get("/api/v1/forms/BNSS_FORM_58")
    assert resp_id.status_code == 200
    assert resp_id.json()["form_number"] == 58

    # Non-existent form number returns 404
    resp_invalid = client.get("/api/v1/forms/99")
    assert resp_invalid.status_code == 404
    assert resp_invalid.json()["error"]["code"] == "NOT_FOUND"


def test_api_list_all_forms(client):
    """Test GET /api/v1/forms returns list of all 58 statutory forms."""
    resp = client.get("/api/v1/forms")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_forms"] == 58
    assert len(data["forms"]) == 58

    f1 = data["forms"][0]
    assert f1["form_number"] == 1
    assert f1["filename"] == "FORM-1_Notice-for-Appearance-by-the-Police.pdf"
    assert "/api/v1/forms/1/download" in f1["download_url"]
    assert f1["extraction_confidence"] > 0.8
    assert f1["needs_review"] is False

    f33 = [f for f in data["forms"] if f["form_number"] == 33][0]
    assert f33["form_number"] == 33
    assert f33["page_start"] == 222
    assert f33["page_end"] == 224
    assert f33["page_count"] == 3


def test_api_search_forms_query_param(client):
    """Test GET /api/v1/forms/search?q= query parameter search."""
    resp = client.get("/api/v1/forms/search?q=Section 35(3)")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "SUCCESS"
    assert data["form"]["form_number"] == 1

    resp_title = client.get("/api/v1/forms/search?q=Charges")
    assert resp_title.status_code == 200
    assert resp_title.json()["form"]["form_number"] == 33


def test_api_download_individual_form_pdf(client):
    """Test GET /api/v1/forms/{id}/download for Form 1, Form 33, Form 58, and Form 99."""
    # 1. Form 1
    resp1 = client.get("/api/v1/forms/1/download")
    assert resp1.status_code == 200
    assert resp1.headers["content-type"] == "application/pdf"
    assert "attachment; filename=" in resp1.headers["content-disposition"]
    reader1 = pypdf.PdfReader(io.BytesIO(resp1.content))
    assert len(reader1.pages) == 1

    # 2. Form 33 (multi-page)
    resp33 = client.get("/api/v1/forms/33/download")
    assert resp33.status_code == 200
    assert "FORM-33_Charges.pdf" in resp33.headers["content-disposition"]
    reader33 = pypdf.PdfReader(io.BytesIO(resp33.content))
    assert len(reader33.pages) == 3

    # 3. Form 58
    resp58 = client.get("/api/v1/forms/58/download")
    assert resp58.status_code == 200
    reader58 = pypdf.PdfReader(io.BytesIO(resp58.content))
    assert len(reader58.pages) == 1

    # 4. Form 99 (Not Found)
    resp99 = client.get("/api/v1/forms/99/download")
    assert resp99.status_code == 404
    assert resp99.json()["error"]["code"] == "NOT_FOUND"


def test_api_download_all_forms_zip(client):
    """Test GET /api/v1/forms/download-all returns valid ZIP with 58 PDF files."""
    resp = client.get("/api/v1/forms/download-all")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/zip"
    assert 'attachment; filename="bnss_second_schedule_forms_1_to_58.zip"' in resp.headers["content-disposition"]

    # Inspect ZIP contents
    zip_bytes = io.BytesIO(resp.content)
    with zipfile.ZipFile(zip_bytes, "r") as zf:
        file_list = zf.namelist()
        assert len(file_list) == 58
        assert all(f.endswith(".pdf") for f in file_list)
        assert all(f.startswith("FORM-") for f in file_list)
        # Form 1 and Form 33 exist in ZIP
        assert any(f.startswith("FORM-1_") for f in file_list)
        assert "FORM-33_Charges.pdf" in file_list
        assert any(f.startswith("FORM-58_") for f in file_list)
