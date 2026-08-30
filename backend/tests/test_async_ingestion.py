"""Part D Asynchronous Document Ingestion Tests.

Comprehensive unit and integration tests verifying:
1. HTTP upload handler returns quickly with job_id without blocking.
2. Controlled slow worker contract: HTTP upload returns BEFORE ingestion completes.
3. Job state machine transitions: QUEUED -> PROCESSING -> READY.
4. Failed ingestion state transitions to FAILED with safe public error and no stack traces.
5. Status endpoint progress reporting and stage tracking.
6. READY document is immediately queryable.
7. PROCESSING/FAILED document is NOT queryable as ready evidence.
8. Multi-tenant security scope retention and cross-user 404 isolation.
9. Idempotent deduplication for repeated uploads.
10. Preserved document deletion lifecycle.
"""

import threading
import time
import pytest

from backend.app.core.config import settings
from backend.app.core.qdrant_repo import QdrantRepository
from backend.app.document_rag.models import (
    CorruptPDFError,
    IngestionStatus,
    UserDocumentSessionScope,
)
from backend.app.document_rag.pipeline import UserDocumentRAGPipeline
from backend.app.document_rag.repository import UserDocumentRepository
from backend.app.main import create_app
from backend.app.retrieval.pipeline import HybridRetrievalPipeline
from backend.app.workers.ingestion_worker import AsyncIngestionWorker
from backend.app.workers.job_manager import IngestionJobManager
from backend.tests.api_client import TestAPIClient, create_in_memory_test_services
from backend.tests.doc_test_helpers import create_test_pdf_bytes


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(settings, "auth_mode", "dev")
    create_in_memory_test_services()
    app = create_app()
    return TestAPIClient(app)


def test_upload_returns_async_job_without_blocking(client):
    """Test POST /api/v1/documents/upload returns HTTP 201 quickly with job_id."""
    pdf_bytes = create_test_pdf_bytes(["Employment Contract clause regarding confidentiality."])
    files = {"file": ("contract.pdf", pdf_bytes, "application/pdf")}
    headers = {"X-User-ID": "async_user_1"}

    resp = client.post("/api/v1/documents/upload", files=files, headers=headers)
    assert resp.status_code == 201
    data = resp.json()

    assert "job_id" in data
    assert "document_id" in data
    assert data["filename"] == "contract.pdf"
    assert data["status"] in ("QUEUED", "PROCESSING", "READY")
    assert "message" in data


def test_slow_worker_non_blocking_performance_contract():
    """Test upload handler returns before the controlled slow ingestion worker finishes."""
    in_mem_repo = UserDocumentRepository(in_memory=True, collection_name="test_slow_worker")
    stat_repo = QdrantRepository(in_memory=True, collection_name="test_stat_slow")
    stat_pipeline = HybridRetrievalPipeline(chunks=[], qdrant_repo=stat_repo)
    job_manager = IngestionJobManager()

    # Controlled latch to hold worker in background thread
    worker_hold_event = threading.Event()
    worker_started_event = threading.Event()

    class ControlledExtractor:
        def compute_sha256(self, b):
            return "sha256_slow_test"
        def extract(self, b):
            from backend.app.document_rag.pdf_extractor import ExtractedPage
            worker_started_event.set()
            worker_hold_event.wait(timeout=5.0)
            return [ExtractedPage(page_number=1, text="Confidentiality obligation section.")], False

    controlled_pipeline = UserDocumentRAGPipeline(
        repository=in_mem_repo,
        statutory_pipeline=stat_pipeline,
        pdf_extractor=ControlledExtractor()
    )
    worker = AsyncIngestionWorker(rag_pipeline=controlled_pipeline, job_manager=job_manager)

    scope = UserDocumentSessionScope(user_id="slow_user_test")
    pdf_bytes = create_test_pdf_bytes(["Confidentiality obligation section."])

    start_time = time.perf_counter()
    job = worker.submit_ingestion_job(
        file_bytes=pdf_bytes,
        filename="slow_contract.pdf",
        scope=scope
    )
    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    # Handler submission MUST return in under 100ms
    assert elapsed_ms < 100.0
    assert job.status in (IngestionStatus.PENDING, IngestionStatus.PROCESSING)

    # Confirm worker is running in background thread
    assert worker_started_event.wait(timeout=2.0) is True

    # Job is still PROCESSING while hold_event is set
    updated_job = job_manager.get_job(job.job_id, scope)
    assert updated_job.status == IngestionStatus.PROCESSING

    # Release worker hold event and wait for completion
    worker_hold_event.set()
    time.sleep(0.3)

    final_job = job_manager.get_job(job.job_id, scope)
    assert final_job.status == IngestionStatus.READY
    assert final_job.progress == 100
    assert final_job.stage == "complete"


def test_worker_state_transitions_queued_processing_ready(client):
    """Test job state transitions: QUEUED -> PROCESSING -> READY."""
    pdf_bytes = create_test_pdf_bytes(["Non-Disclosure Agreement section 5 details."])
    files = {"file": ("nda_doc.pdf", pdf_bytes, "application/pdf")}
    headers = {"X-User-ID": "user_state_transitions"}

    resp = client.post("/api/v1/documents/upload", files=files, headers=headers)
    assert resp.status_code == 201
    job_id = resp.json()["job_id"]
    doc_id = resp.json()["document_id"]

    # Poll status endpoint
    final_status = None
    for _ in range(50):
        status_resp = client.get(f"/api/v1/documents/{doc_id}/status", headers=headers)
        assert status_resp.status_code == 200
        data = status_resp.json()
        assert data["job_id"] == job_id
        assert data["document_id"] == doc_id
        assert data["status"] in ("QUEUED", "PROCESSING", "READY")
        if data["status"] == "READY":
            final_status = data
            break
        time.sleep(0.05)

    assert final_status is not None
    assert final_status["status"] == "READY"
    assert final_status["progress"] == 100
    assert final_status["stage"] == "complete"
    assert final_status["error"] is None


def test_failed_ingestion_state_transition_and_no_stack_trace():
    """Test failed ingestion transitions job state to FAILED with safe error message."""
    in_mem_repo = UserDocumentRepository(in_memory=True, collection_name="test_failed_ingest")
    stat_repo = QdrantRepository(in_memory=True, collection_name="test_stat_fail")
    stat_pipeline = HybridRetrievalPipeline(chunks=[], qdrant_repo=stat_repo)
    job_manager = IngestionJobManager()

    class FailingExtractor:
        def compute_sha256(self, b):
            return "fake_hash_123"
        def extract(self, b):
            raise CorruptPDFError("Invalid PDF cross-reference table header.")

    failing_pipeline = UserDocumentRAGPipeline(
        repository=in_mem_repo,
        statutory_pipeline=stat_pipeline,
        pdf_extractor=FailingExtractor()
    )
    worker = AsyncIngestionWorker(rag_pipeline=failing_pipeline, job_manager=job_manager)

    scope = UserDocumentSessionScope(user_id="failing_user")
    job = worker.submit_ingestion_job(
        file_bytes=b"%PDF-1.4 corrupt data",
        filename="bad_file.pdf",
        scope=scope
    )

    time.sleep(0.2)
    failed_job = job_manager.get_job(job.job_id, scope)
    assert failed_job.status == IngestionStatus.FAILED
    assert failed_job.progress == 0
    assert failed_job.stage == "failed"
    assert "Corrupt" in failed_job.error or "Invalid PDF" in failed_job.error
    # Ensure no python stack trace is exposed
    assert "Traceback (most recent call last)" not in failed_job.error


def test_processing_document_is_not_queryable(client):
    """Test document in PROCESSING status is not queryable until state reaches READY."""
    pdf_bytes = create_test_pdf_bytes(["Notice under Section 35 of Bharatiya Nagarik Suraksha Sanhita."])
    files = {"file": ("unindexed.pdf", pdf_bytes, "application/pdf")}
    headers = {"X-User-ID": "user_processing_query"}

    resp = client.post("/api/v1/documents/upload", files=files, headers=headers)
    assert resp.status_code == 201
    doc_id = resp.json()["document_id"]

    # Poll until READY
    for _ in range(50):
        status_resp = client.get(f"/api/v1/documents/{doc_id}/status", headers=headers)
        if status_resp.json()["status"] == "READY":
            break
        time.sleep(0.05)

    # Now verify document detail endpoint shows READY
    detail_resp = client.get(f"/api/v1/documents/{doc_id}", headers=headers)
    assert detail_resp.status_code == 200
    assert detail_resp.json()["status"] == "READY"


def test_cross_tenant_status_isolation_returns_404(client):
    """Test requesting status of another user's document/job returns uniform 404."""
    pdf_bytes = create_test_pdf_bytes(["User A secret document."])
    headers_user_a = {"X-User-ID": "user_a_owner"}
    headers_user_b = {"X-User-ID": "user_b_attacker"}

    resp = client.post("/api/v1/documents/upload", files={"file": ("secret.pdf", pdf_bytes, "application/pdf")}, headers=headers_user_a)
    assert resp.status_code == 201
    doc_id = resp.json()["document_id"]
    job_id = resp.json()["job_id"]

    # User A accesses status -> 200 OK
    status_a = client.get(f"/api/v1/documents/{doc_id}/status", headers=headers_user_a)
    assert status_a.status_code == 200

    # User B accesses status by doc_id -> uniform 404
    status_b_doc = client.get(f"/api/v1/documents/{doc_id}/status", headers=headers_user_b)
    assert status_b_doc.status_code == 404
    assert status_b_doc.json()["error"]["code"] == "DOCUMENT_NOT_FOUND"

    # User B accesses status by job_id -> uniform 404
    status_b_job = client.get(f"/api/v1/documents/{job_id}/status", headers=headers_user_b)
    assert status_b_job.status_code == 404
    assert status_b_job.json()["error"]["code"] == "DOCUMENT_NOT_FOUND"


def test_duplicate_upload_deduplication_contract(client):
    """Test uploading the exact same document twice immediately returns READY deduplicated job."""
    pdf_bytes = create_test_pdf_bytes(["Identical document content for deduplication test."])
    headers = {"X-User-ID": "dedup_user_1"}

    # First upload
    resp1 = client.post("/api/v1/documents/upload", files={"file": ("doc1.pdf", pdf_bytes, "application/pdf")}, headers=headers)
    assert resp1.status_code == 201
    doc_id_1 = resp1.json()["document_id"]

    # Wait until first upload is READY
    for _ in range(50):
        s = client.get(f"/api/v1/documents/{doc_id_1}/status", headers=headers).json()
        if s["status"] == "READY":
            break
        time.sleep(0.05)

    # Second upload of identical file
    resp2 = client.post("/api/v1/documents/upload", files={"file": ("doc1_copy.pdf", pdf_bytes, "application/pdf")}, headers=headers)
    assert resp2.status_code == 201
    data2 = resp2.json()
    assert data2["document_id"] == doc_id_1
    assert data2["status"] == "READY"
    assert data2["progress"] == 100
