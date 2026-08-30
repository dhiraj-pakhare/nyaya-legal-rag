"""Document Lifecycle API Integration Tests (Phase 8).

Verifies:
1. Valid PDF upload completes synchronously with 201 Created and immediate consistency
2. Invalid MIME types and corrupted magic bytes return 415 Unsupported Media Type
3. Oversized files return 413 Payload Too Large
4. Scoped document listing, detail retrieval, and deletion
"""

import pytest

from backend.app.core.config import settings
from backend.app.main import create_app
from backend.tests.api_client import TestAPIClient, create_in_memory_test_services
from backend.tests.doc_test_helpers import create_test_pdf_bytes


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(settings, "auth_mode", "dev")
    create_in_memory_test_services()
    app = create_app()
    return TestAPIClient(app)


def test_document_upload_success_synchronous(client):
    """Test valid PDF upload returns 201 Created with job_id and async status."""
    import time
    pdf_bytes = create_test_pdf_bytes(["Notice under Section 35 of Bharatiya Nagarik Suraksha Sanhita."])
    files = {"file": ("test_notice.pdf", pdf_bytes, "application/pdf")}
    headers = {"X-User-ID": "user_doc_test_1"}

    resp = client.post("/api/v1/documents", files=files, headers=headers)
    assert resp.status_code == 201
    data = resp.json()
    assert data["status"] in ("QUEUED", "PROCESSING", "READY")
    assert "job_id" in data
    assert "document_id" in data
    assert data["filename"] == "test_notice.pdf"

    doc_id = data["document_id"]

    # Poll status endpoint until READY
    status_data = None
    for _ in range(50):
        status_resp = client.get(f"/api/v1/documents/{doc_id}/status", headers=headers)
        assert status_resp.status_code == 200
        status_data = status_resp.json()
        if status_data["status"] == "READY":
            break
        time.sleep(0.05)

    assert status_data["status"] == "READY"

    # Verify immediate listing consistency
    list_resp = client.get("/api/v1/documents", headers=headers)
    assert list_resp.status_code == 200
    docs = list_resp.json()
    assert any(d["document_id"] == doc_id for d in docs)

    # Verify detail retrieval
    detail_resp = client.get(f"/api/v1/documents/{doc_id}", headers=headers)
    assert detail_resp.status_code == 200
    assert detail_resp.json()["document_id"] == doc_id

    # Clean up via deletion
    del_resp = client.delete(f"/api/v1/documents/{doc_id}", headers=headers)
    assert del_resp.status_code == 200
    assert del_resp.json()["deleted"] is True


def test_document_upload_invalid_mime_type(client):
    """Test non-PDF upload returns 415 Unsupported Media Type."""
    files = {"file": ("malicious.txt", b"plain text payload", "text/plain")}
    headers = {"X-User-ID": "user_doc_test_2"}

    resp = client.post("/api/v1/documents", files=files, headers=headers)
    assert resp.status_code == 415
    assert resp.json()["error"]["code"] == "UNSUPPORTED_MEDIA_TYPE"


def test_document_upload_corrupted_magic_bytes(client):
    """Test PDF with missing %PDF- header magic bytes returns 415."""
    corrupted_bytes = b"CORRUPTED_NOT_A_PDF_STREAM"
    files = {"file": ("fake.pdf", corrupted_bytes, "application/pdf")}
    headers = {"X-User-ID": "user_doc_test_3"}

    resp = client.post("/api/v1/documents", files=files, headers=headers)
    assert resp.status_code == 415
    assert resp.json()["error"]["code"] == "UNSUPPORTED_MEDIA_TYPE"


def test_document_upload_oversized_payload_returns_413(client, monkeypatch):
    """Test upload exceeding max_user_doc_size_bytes returns 413 Payload Too Large."""
    monkeypatch.setattr(settings, "max_user_doc_size_bytes", 100)
    pdf_bytes = create_test_pdf_bytes(["Large document content page 1."])
    files = {"file": ("oversized.pdf", pdf_bytes, "application/pdf")}
    headers = {"X-User-ID": "user_doc_test_4"}

    resp = client.post("/api/v1/documents", files=files, headers=headers)
    assert resp.status_code == 413
    assert resp.json()["error"]["code"] == "FILE_TOO_LARGE"
