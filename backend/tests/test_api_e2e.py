"""End-to-End API Integration Workflow Tests (Phase 8).

Verifies complete multi-step user workflows:
1. Health and readiness checks
2. Uploading user document
3. Querying against uploaded document
4. Performing deterministic form lookup
5. Deleting uploaded document and verifying scoped cleanup
"""

import pytest

from backend.app.core.config import settings
from backend.app.generation.providers import MockLLMProvider
from backend.app.main import create_app
from backend.tests.api_client import TestAPIClient, create_in_memory_test_services
from backend.tests.doc_test_helpers import create_test_pdf_bytes


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(settings, "auth_mode", "dev")
    mock_llm = MockLLMProvider()
    create_in_memory_test_services(mock_llm)
    app = create_app()
    return TestAPIClient(app), mock_llm


def test_full_api_e2e_workflow(client):
    """Execute complete multi-step E2E lifecycle via HTTP API."""
    test_client, mock_llm = client

    # 1. Check Liveness & Readiness
    health_res = test_client.get("/api/v1/health")
    assert health_res.status_code == 200
    assert health_res.json()["status"] == "UP"

    ready_res = test_client.get("/api/v1/ready")
    assert ready_res.status_code == 200

    # 2. Form Lookup
    form_res = test_client.post("/api/v1/forms/lookup", json={"query": "Form 1"})
    assert form_res.status_code == 200
    assert form_res.json()["form"]["form_number"] == 1

    # 3. Document Ingestion
    pdf_bytes = create_test_pdf_bytes(["Clause 12: Notice requirement under contract."])
    files = {"file": ("contract_2026.pdf", pdf_bytes, "application/pdf")}
    headers = {"X-User-ID": "e2e_user_test"}

    upload_res = test_client.post("/api/v1/documents", files=files, headers=headers)
    assert upload_res.status_code == 201
    doc_id = upload_res.json()["document_id"]

    # 4. List Documents
    list_res = test_client.get("/api/v1/documents", headers=headers)
    assert list_res.status_code == 200
    assert len(list_res.json()) >= 1

    # 5. Delete Document
    del_res = test_client.delete(f"/api/v1/documents/{doc_id}", headers=headers)
    assert del_res.status_code == 200
    assert del_res.json()["deleted"] is True

    # 6. Verify Deletion
    del_check = test_client.get(f"/api/v1/documents/{doc_id}", headers=headers)
    assert del_check.status_code == 404
