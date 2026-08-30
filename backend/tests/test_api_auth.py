"""Authentication and Security Boundary API Tests (Phase 8).

Verifies:
1. Production mode strictly rejects unauthenticated requests (401)
2. Production mode ignores client X-User-ID / X-Session-ID headers
3. Production mode rejects client JSON body user_id spoofing
4. Development mode allows explicit test principal headers only under AUTH_MODE=dev
5. Dev fallback default principal cannot activate when AUTH_MODE=prod
"""

import pytest

from backend.app.core.config import settings
from backend.app.main import create_app
from backend.tests.api_client import TestAPIClient, create_in_memory_test_services


@pytest.fixture
def client():
    create_in_memory_test_services()
    app = create_app()
    return TestAPIClient(app)


def test_prod_auth_missing_credentials_returns_401(client, monkeypatch):
    """Production guard 1: Missing Bearer token returns 401 in prod mode."""
    monkeypatch.setattr(settings, "auth_mode", "prod")
    resp = client.get("/api/v1/documents")
    assert resp.status_code == 401
    data = resp.json()
    assert data["error"]["code"] == "UNAUTHORIZED"
    assert "Bearer token" in data["error"]["message"]


def test_prod_auth_x_user_id_spoofing_rejected_with_401(client, monkeypatch):
    """Production guard 2: X-User-ID header cannot bypass authentication in prod mode."""
    monkeypatch.setattr(settings, "auth_mode", "prod")
    headers = {"X-User-ID": "admin_user", "X-Session-ID": "admin_session"}
    resp = client.get("/api/v1/documents", headers=headers)
    assert resp.status_code == 401
    assert resp.json()["error"]["code"] == "UNAUTHORIZED"


def test_prod_auth_valid_bearer_token_accepted(client, monkeypatch):
    """Production guard 3: Valid Bearer token successfully resolves principal."""
    monkeypatch.setattr(settings, "auth_mode", "prod")
    headers = {"Authorization": "Bearer token_testuser123"}
    resp = client.get("/api/v1/documents", headers=headers)
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


def test_dev_auth_mode_accepts_test_headers(client, monkeypatch):
    """Development guard 1: Under AUTH_MODE=dev, X-User-ID is accepted for test harnesses."""
    monkeypatch.setattr(settings, "auth_mode", "dev")
    headers = {"X-User-ID": "dev_harness_user", "X-Session-ID": "sess_1"}
    resp = client.get("/api/v1/documents", headers=headers)
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


def test_dev_auth_mode_fallback_to_dev_default_without_headers(client, monkeypatch):
    """Development guard 2: Under AUTH_MODE=dev, missing headers falls back to dev_user_default."""
    monkeypatch.setattr(settings, "auth_mode", "dev")
    resp = client.get("/api/v1/documents")
    assert resp.status_code == 200


def test_query_endpoint_ignores_body_user_id_spoofing(client, monkeypatch):
    """Production guard 4: JSON body containing spoofed user_id is ignored."""
    monkeypatch.setattr(settings, "auth_mode", "prod")
    payload = {"query": "What is Section 103?", "user_id": "spoofed_admin"}
    resp = client.post("/api/v1/query", json=payload)
    assert resp.status_code == 401
