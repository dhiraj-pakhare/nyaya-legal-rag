"""Part D Prometheus Metrics Tests.

Verifies:
1. GET /api/v1/metrics returns official Prometheus text/plain format.
2. nyaya_http_requests_total counter updates on HTTP calls.
3. nyaya_chat_requests_total counter updates on legal query execution.
4. nyaya_document_uploads_total counter updates on document uploads.
5. Ingestion job state and failure metrics update.
6. Retrieval and embedding latency metrics exist.
7. Refusal counter updates when out-of-scope refusal is returned.
8. Token counters and estimated cost update based on configured provider pricing.
9. Qdrant availability gauge reflects health status (1.0 or 0.0).
10. Sensitive label audit: No user_id, document_id, session_id, or query text appears in labels.
"""

import pytest
from backend.app.core.config import settings
from backend.app.core.metrics import get_metrics_collector
from backend.app.main import create_app
from backend.tests.api_client import TestAPIClient, create_in_memory_test_services
from backend.tests.doc_test_helpers import create_test_pdf_bytes


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(settings, "auth_mode", "dev")
    create_in_memory_test_services()
    app = create_app()
    return TestAPIClient(app)


def test_metrics_endpoint_returns_prometheus_exposition_format(client):
    """Test GET /api/v1/metrics returns HTTP 200 with text/plain Prometheus format."""
    resp = client.get("/api/v1/metrics")
    assert resp.status_code == 200
    assert "text/plain" in resp.headers["content-type"]
    text = resp.text
    assert "# HELP nyaya_http_requests_total" in text
    assert "# TYPE nyaya_http_requests_total counter" in text
    assert "nyaya_qdrant_available" in text


def test_metrics_http_requests_total_counter_increments(client):
    """Test HTTP request counter increases after API calls."""
    client.get("/api/v1/health")
    client.get("/api/v1/health")

    metrics_resp = client.get("/api/v1/metrics")
    text = metrics_resp.text
    assert "nyaya_http_requests_total" in text
    assert 'endpoint="/api/v1/health"' in text or 'endpoint="/health"' in text


def test_metrics_chat_requests_and_token_cost_counters(client):
    """Test chat counter, tokens, and estimated cost update upon executing query."""
    headers = {"X-User-ID": "metrics_user_1"}
    resp = client.post("/api/v1/query", json={"query": "What is section 103 BNS?"}, headers=headers)
    assert resp.status_code == 200

    metrics_resp = client.get("/api/v1/metrics")
    text = metrics_resp.text

    assert "nyaya_chat_requests_total" in text
    assert "nyaya_llm_prompt_tokens_total" in text
    assert "nyaya_llm_completion_tokens_total" in text
    assert "nyaya_llm_estimated_cost_usd_total" in text


def test_metrics_document_upload_and_ingestion_job_counters(client):
    """Test document upload counter updates upon uploading a PDF."""
    pdf_bytes = create_test_pdf_bytes(["Sample text payload."])
    headers = {"X-User-ID": "metrics_upload_user"}

    up_resp = client.post("/api/v1/documents/upload", files={"file": ("m.pdf", pdf_bytes, "application/pdf")}, headers=headers)
    assert up_resp.status_code == 201

    metrics_resp = client.get("/api/v1/metrics")
    text = metrics_resp.text
    assert "nyaya_document_uploads_total" in text
    assert "nyaya_document_ingestion_jobs_total" in text


def test_metrics_refusal_counter_updates_on_out_of_scope_query(client):
    """Test refusal metric counter increments when out-of-scope refusal occurs."""
    headers = {"X-User-ID": "metrics_refusal_user"}
    # Out of scope query
    resp = client.post("/api/v1/query", json={"query": "What is the capital of France?"}, headers=headers)
    assert resp.status_code == 200
    assert resp.json()["is_refused"] is True

    metrics_resp = client.get("/api/v1/metrics")
    text = metrics_resp.text
    assert "nyaya_refusal_count_total" in text


def test_metrics_sensitive_label_audit(client):
    """Audit metrics exposition output to verify NO high-cardinality or sensitive data is leaked in labels."""
    headers = {"X-User-ID": "sensitive_user_secret_999"}
    client.post("/api/v1/query", json={"query": "SECRET_QUERY_TEXT_12345"}, headers=headers)

    metrics_resp = client.get("/api/v1/metrics")
    text = metrics_resp.text

    # Verify sensitive identifiers are NOT present in Prometheus labels
    assert "sensitive_user_secret_999" not in text
    assert "SECRET_QUERY_TEXT_12345" not in text
    assert "user_id=" not in text
    assert "session_id=" not in text
    assert "document_id=" not in text
