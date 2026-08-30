"""Statutory Forms API Integration Tests (Phase 8).

Verifies:
1. Exact form number lookups (Form 1, Form 33, Form 58)
2. Section reference lookups (Section 35(3))
3. Ambiguous form queries return AMBIGUOUS with candidate list
4. Non-existent Form 99 returns NOT_FOUND refusal
5. Direct form retrieval by ID/number and 404 for invalid numbers
"""

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
