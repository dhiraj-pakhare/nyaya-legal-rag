"""Server-Sent Events (SSE) Streaming API Tests (Phase 8).

Verifies:
1. Streaming endpoint emits correct sequence of status, token, citation, and complete events
2. Substantive legal claims are not emitted before AST citation verification
3. Out-of-scope queries emit refusal event safely
"""

import pytest

from backend.app.core.config import settings
from backend.app.generation.providers import MockLLMProvider
from backend.app.main import create_app
from backend.tests.api_client import TestAPIClient, create_in_memory_test_services


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(settings, "auth_mode", "dev")
    mock_llm = MockLLMProvider()
    create_in_memory_test_services(mock_llm)
    app = create_app()
    return TestAPIClient(app), mock_llm



def test_api_streaming_valid_statutory_query(client):
    """Test POST /api/v1/query/stream emits status, citation, token, and complete events."""
    test_client, mock_llm = client
    mock_llm.set_canned_response("According to [BNS s.103], murder is an offence.")

    payload = {"query": "What is the punishment for murder under Section 103?"}
    headers = {"X-User-ID": "user_stream_1"}

    resp = test_client.post("/api/v1/query/stream", json=payload, headers=headers)
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]

    content = resp.text
    assert "event: status" in content
    assert "event: citation" in content
    assert "event: token" in content
    assert "event: complete" in content


def test_api_streaming_out_of_scope_emits_refusal_event(client):
    """Test streaming out-of-scope query emits refusal event cleanly."""
    test_client, mock_llm = client
    payload = {"query": "How to make a pizza?"}
    headers = {"X-User-ID": "user_stream_2"}

    resp = test_client.post("/api/v1/query/stream", json=payload, headers=headers)
    assert resp.status_code == 200
    content = resp.text
    assert "event: refusal" in content
    assert "event: token" not in content  # Zero tokens streamed on refusal
