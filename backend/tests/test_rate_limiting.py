"""Part D API Rate Limiting Tests.

Verifies:
1. Requests under limit succeed (200 / 201).
2. Requests over limit return HTTP 429 Too Many Requests.
3. Retry-After header is present in 429 response.
4. Limits are strictly scoped to authenticated principal identity.
5. User A quota usage does NOT affect User B's quota.
6. Body/header spoofing cannot bypass rate limits.
7. Configuration changes (requests_per_minute) control limits deterministically.
"""

import pytest
from fastapi.testclient import TestClient

from backend.app.core.config import settings
from backend.app.core.rate_limiter import APIRateLimiter, get_rate_limiter
from backend.app.main import create_app
from backend.tests.api_client import TestAPIClient, create_in_memory_test_services
from backend.tests.doc_test_helpers import create_test_pdf_bytes


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(settings, "auth_mode", "dev")
    monkeypatch.setattr(settings, "rate_limit_enabled", True)
    create_in_memory_test_services()
    limiter = get_rate_limiter()
    limiter.reset()
    app = create_app()
    return TestAPIClient(app)


def test_rate_limiting_under_quota_succeeds(client):
    """Test requests within rate limit quota succeed with 200 OK."""
    limiter = get_rate_limiter()
    limiter.requests_per_minute = 5
    limiter.reset()

    headers = {"X-User-ID": "user_under_limit"}

    for _ in range(3):
        resp = client.post("/api/v1/query", json={"query": "What is section 103 BNS?"}, headers=headers)
        assert resp.status_code == 200


def test_rate_limiting_exceeded_returns_429_and_retry_after(client):
    """Test requests exceeding quota return HTTP 429 with Retry-After header."""
    limiter = get_rate_limiter()
    limiter.requests_per_minute = 3
    limiter.reset()

    headers = {"X-User-ID": "user_over_limit"}

    # 3 allowed requests
    for _ in range(3):
        resp = client.post("/api/v1/query", json={"query": "What is section 35 BNSS?"}, headers=headers)
        assert resp.status_code == 200

    # 4th request MUST return 429 Too Many Requests
    exceeded_resp = client.post("/api/v1/query", json={"query": "What is section 35 BNSS?"}, headers=headers)
    assert exceeded_resp.status_code == 429
    data = exceeded_resp.json()
    assert data["error"]["code"] == "RATE_LIMIT_EXCEEDED"
    assert "Retry-After" in exceeded_resp.headers
    assert int(exceeded_resp.headers["Retry-After"]) >= 1


def test_rate_limiting_user_quota_isolation(client):
    """Test User A exceeding quota does NOT block User B."""
    limiter = get_rate_limiter()
    limiter.requests_per_minute = 2
    limiter.reset()

    headers_a = {"X-User-ID": "user_quota_a"}
    headers_b = {"X-User-ID": "user_quota_b"}

    # User A exhausts quota
    assert client.post("/api/v1/query", json={"query": "BNS 103"}, headers=headers_a).status_code == 200
    assert client.post("/api/v1/query", json={"query": "BNS 103"}, headers=headers_a).status_code == 200
    assert client.post("/api/v1/query", json={"query": "BNS 103"}, headers=headers_a).status_code == 429

    # User B should still succeed
    resp_b = client.post("/api/v1/query", json={"query": "BNS 103"}, headers=headers_b)
    assert resp_b.status_code == 200


def test_rate_limiting_cannot_be_bypassed_by_body_spoofing(client):
    """Test changing body payload user_id does NOT bypass principal identity rate limit."""
    limiter = get_rate_limiter()
    limiter.requests_per_minute = 2
    limiter.reset()

    headers = {"X-User-ID": "real_authenticated_user"}

    # Attacker tries changing fake body payload on each request
    assert client.post("/api/v1/query", json={"query": "BNS 103", "user_id": "fake_user_1"}, headers=headers).status_code == 200
    assert client.post("/api/v1/query", json={"query": "BNS 103", "user_id": "fake_user_2"}, headers=headers).status_code == 200

    # 3rd request should fail under real principal key
    resp = client.post("/api/v1/query", json={"query": "BNS 103", "user_id": "fake_user_3"}, headers=headers)
    assert resp.status_code == 429


def test_chat_alias_route_is_rate_limited(client):
    """Test POST /api/v1/chat route alias is protected by rate limiting."""
    limiter = get_rate_limiter()
    limiter.requests_per_minute = 1
    limiter.reset()

    headers = {"X-User-ID": "chat_user_limit"}

    resp1 = client.post("/api/v1/chat", json={"query": "What is section 103 BNS?"}, headers=headers)
    assert resp1.status_code == 200

    resp2 = client.post("/api/v1/chat", json={"query": "What is section 103 BNS?"}, headers=headers)
    assert resp2.status_code == 429


def test_document_upload_route_is_rate_limited(client):
    """Test POST /api/v1/documents/upload route is protected by rate limiting."""
    limiter = get_rate_limiter()
    limiter.requests_per_minute = 1
    limiter.reset()

    pdf_bytes = create_test_pdf_bytes(["Test document payload."])
    headers = {"X-User-ID": "upload_user_limit"}

    resp1 = client.post("/api/v1/documents/upload", files={"file": ("d1.pdf", pdf_bytes, "application/pdf")}, headers=headers)
    assert resp1.status_code == 201

    resp2 = client.post("/api/v1/documents/upload", files={"file": ("d2.pdf", pdf_bytes, "application/pdf")}, headers=headers)
    assert resp2.status_code == 429
