"""Privacy, Anti-Enumeration, and Security API Tests (Phase 8).

Verifies:
1. Anti-enumeration: Uniform 404 on non-existent vs unowned cross-user documents
2. Cross-user document deletion protection
3. Filename path traversal sanitization
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


def test_anti_enumeration_cross_user_and_nonexistent_produce_same_404(client):
    """Privacy test: Unowned document and non-existent document return exact same 404 response."""
    # 1. User A uploads a private document
    pdf_bytes = create_test_pdf_bytes(["Private financial contract for User Alpha."])
    files = {"file": ("userA_private.pdf", pdf_bytes, "application/pdf")}
    headers_A = {"X-User-ID": "tenant_user_alpha"}
    headers_B = {"X-User-ID": "tenant_user_beta"}

    resp_A = client.post("/api/v1/documents", files=files, headers=headers_A)
    assert resp_A.status_code == 201
    userA_doc_id = resp_A.json()["document_id"]

    # 2. User B tries to retrieve User A's document (IDOR attempt)
    resp_B_idor = client.get(f"/api/v1/documents/{userA_doc_id}", headers=headers_B)
    assert resp_B_idor.status_code == 404
    idor_data = resp_B_idor.json()
    assert idor_data["error"]["code"] == "DOCUMENT_NOT_FOUND"
    assert idor_data["error"]["message"] == "Document not found or inaccessible."

    # 3. User B tries to retrieve a completely non-existent document
    resp_B_nonexistent = client.get("/api/v1/documents/doc_nonexistent_9999", headers=headers_B)
    assert resp_B_nonexistent.status_code == 404
    nonexistent_data = resp_B_nonexistent.json()
    assert nonexistent_data["error"]["code"] == "DOCUMENT_NOT_FOUND"
    assert nonexistent_data["error"]["message"] == "Document not found or inaccessible."

    # 4. Assert responses are identical (preventing existence enumeration)
    assert idor_data == nonexistent_data

    # 5. User B tries to delete User A's document
    del_idor = client.delete(f"/api/v1/documents/{userA_doc_id}", headers=headers_B)
    assert del_idor.status_code == 404
    assert del_idor.json()["error"]["code"] == "DOCUMENT_NOT_FOUND"


def test_filename_path_traversal_sanitization(client):
    """Security test: Upload with path traversal filename '../../etc/passwd.pdf' is sanitized."""
    pdf_bytes = create_test_pdf_bytes(["System config contents."])
    files = {"file": ("../../etc/passwd.pdf", pdf_bytes, "application/pdf")}
    headers = {"X-User-ID": "user_traversal_test"}

    resp = client.post("/api/v1/documents", files=files, headers=headers)
    assert resp.status_code == 201
    sanitized_name = resp.json()["filename"]
    assert ".." not in sanitized_name
    assert "/" not in sanitized_name
    assert "\\" not in sanitized_name
