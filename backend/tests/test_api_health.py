"""Health and Readiness Diagnostic API Tests (Phase 8).

Verifies:
1. /health returns 200 OK with lightweight liveness state
2. /ready returns 200 OK with dependency checks (Qdrant, forms, embeddings, LLM)
3. Zero secrets or filesystem paths exposed in response
"""

import pytest

from backend.app.main import create_app
from backend.tests.api_client import TestAPIClient, create_in_memory_test_services


@pytest.fixture
def client():
    create_in_memory_test_services()
    app = create_app()
    return TestAPIClient(app)


def test_api_health_liveness(client):
    """Test GET /api/v1/health returns 200 UP."""
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "UP"
    assert "timestamp" in data
    assert data["version"] == "1.0.0"


def test_api_readiness_probe(client):
    """Test GET /api/v1/ready inspects core components."""
    resp = client.get("/api/v1/ready")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "READY"
    assert len(data["dependencies"]) >= 3

    dep_names = [d["name"] for d in data["dependencies"]]
    assert "statutory_forms_registry" in dep_names
    assert "embedding_engine" in dep_names

    # Verify no secrets or sensitive internal paths leaked
    raw_str = str(data)
    assert "password" not in raw_str.lower()
    assert "secret" not in raw_str.lower()
    assert "/Users/" not in raw_str
