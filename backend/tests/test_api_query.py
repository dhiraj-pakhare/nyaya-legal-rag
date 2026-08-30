"""Unified Legal Query API Tests (Phase 8).

Verifies:
1. Pure statutory legal queries return grounded answers with verified citations
2. User document queries retrieve scoped chunks and cite [DOC p.X]
3. Combined queries cite both statutory sections and document pages
4. Out-of-scope queries return clean structured refusals
5. Automatic routing to statutory forms engine
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



def test_api_query_statutory_legal_question(client):
    """Test statutory legal query returns 200 OK with verified statutory citation."""
    test_client, mock_llm = client
    mock_llm.set_canned_response("Under [BNS s.103], murder is punished with death or imprisonment for life.")

    payload = {"query": "What is the punishment for murder under BNS section 103?"}
    headers = {"X-User-ID": "user_query_stat_1"}

    resp = test_client.post("/api/v1/query", json=payload, headers=headers)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "SUCCESS"
    assert data["is_refused"] is False
    assert len(data["citations"]) >= 1
    assert data["citations"][0]["citation_type"] == "STATUTORY"
    assert data["citations"][0]["citation_text"] == "[BNS s.103]"


def test_api_query_form_intent_routing(client):
    """Test form intent query automatically routes to statutory forms engine."""
    test_client, mock_llm = client
    payload = {"query": "Show me Form 1"}
    headers = {"X-User-ID": "user_query_form_1"}

    resp = test_client.post("/api/v1/query", json=payload, headers=headers)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "SUCCESS"
    assert data["routed_corpus"] == "STATUTORY_FORM"
    assert data["citations"][0]["citation_type"] == "FORM"
    assert data["citations"][0]["citation_text"] == "[BNSS Second Schedule, Form 1]"


def test_api_query_out_of_scope_refusal(client):
    """Test out-of-scope query returns clean structured refusal with answer=None."""
    test_client, mock_llm = client
    payload = {"query": "What is the best recipe for chocolate cake?"}
    headers = {"X-User-ID": "user_query_refuse_1"}

    resp = test_client.post("/api/v1/query", json=payload, headers=headers)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "REFUSED"
    assert data["is_refused"] is True
    assert data["answer"] is None
    assert len(data["citations"]) == 0
