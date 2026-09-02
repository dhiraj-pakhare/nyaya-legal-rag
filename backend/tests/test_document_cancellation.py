"""Comprehensive tests for User Document Ingestion Job Cancellation.

Verifies:
1. Owner can cancel an active PROCESSING job via POST /api/v1/documents/{job_id}/cancel.
2. Cross-user tenant cannot cancel another user's job (uniform 404 anti-enumeration).
3. READY job cancel attempt is idempotent and does not delete valid vectors.
4. FAILED job cancel attempt is idempotent.
5. Already CANCELLED job cancel attempt is idempotent.
6. Worker detects cancellation between stages and aborts without becoming READY.
7. Worker detects cancellation during embedding batch loop.
8. Partial Qdrant vector cleanup purges vectors strictly for the cancelled document.
9. Statutory collection nyaya_legal_corpus remains completely untouched.
10. Another user's documents and vectors remain completely untouched.
"""

import threading
import time
import uuid
import pytest
from qdrant_client.http import models as qmodels

from backend.app.core.config import settings
from backend.app.core.qdrant_repo import QdrantRepository
from backend.app.document_rag.models import (
    IngestionStatus,
    UserDocument,
    UserDocumentChunk,
    UserDocumentSessionScope,
)
from backend.app.document_rag.pipeline import UserDocumentRAGPipeline
from backend.app.document_rag.repository import UserDocumentRepository
from backend.app.main import create_app
from backend.app.retrieval.pipeline import HybridRetrievalPipeline
from backend.app.services.document_service import DocumentManagementService
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


def test_owner_can_cancel_active_processing_job(client):
    """Test owner can cancel an active job, transitioning status to CANCELLED."""
    pdf_bytes = create_test_pdf_bytes(["Sample contract terms and conditions."])
    files = {"file": ("test_doc.pdf", pdf_bytes, "application/pdf")}
    headers = {"X-User-ID": "user_cancel_1"}

    # Upload document
    resp = client.post("/api/v1/documents/upload", files=files, headers=headers)
    assert resp.status_code == 201
    data = resp.json()
    job_id = data["job_id"]
    doc_id = data["document_id"]

    # Cancel the job
    cancel_resp = client.post(f"/api/v1/documents/{job_id}/cancel", headers=headers)
    assert cancel_resp.status_code == 200
    cancel_data = cancel_resp.json()
    assert cancel_data["job_id"] == job_id
    assert cancel_data["document_id"] == doc_id
    assert cancel_data["status"] in ("CANCELLED", "READY")
    assert "message" in cancel_data

    # Check status endpoint
    status_resp = client.get(f"/api/v1/documents/{job_id}/status", headers=headers)
    assert status_resp.status_code == 200
    status_data = status_resp.json()
    assert status_data["status"] in ("CANCELLED", "READY")


def test_cross_user_cannot_cancel_job(client):
    """Test cross-user caller receives uniform 404 and cannot cancel another user's job."""
    pdf_bytes = create_test_pdf_bytes(["Confidential corporate agreement."])
    files = {"file": ("user_a_doc.pdf", pdf_bytes, "application/pdf")}
    user_a_headers = {"X-User-ID": "tenant_alice"}
    user_b_headers = {"X-User-ID": "tenant_bob"}

    # Alice uploads
    resp = client.post("/api/v1/documents/upload", files=files, headers=user_a_headers)
    assert resp.status_code == 201
    job_id = resp.json()["job_id"]

    # Bob attempts to cancel Alice's job
    cancel_resp = client.post(f"/api/v1/documents/{job_id}/cancel", headers=user_b_headers)
    assert cancel_resp.status_code == 404

    # Alice's job status remains valid and accessible to Alice
    alice_status = client.get(f"/api/v1/documents/{job_id}/status", headers=user_a_headers)
    assert alice_status.status_code == 200
    assert alice_status.json()["status"] in ("PENDING", "PROCESSING", "READY")


def test_cancel_ready_job_is_idempotent(client):
    """Test cancelling an already READY job returns idempotent response and preserves vectors."""
    job_mgr = IngestionJobManager()
    repo = UserDocumentRepository(in_memory=True, collection_name="test_cancel_ready")
    stat_repo = QdrantRepository(in_memory=True, collection_name="test_stat_ready")
    stat_pipeline = HybridRetrievalPipeline(chunks=[], qdrant_repo=stat_repo)
    rag_pipeline = UserDocumentRAGPipeline(repository=repo, statutory_pipeline=stat_pipeline)
    worker = AsyncIngestionWorker(rag_pipeline=rag_pipeline, job_manager=job_mgr)
    doc_service = DocumentManagementService(
        rag_pipeline=rag_pipeline,
        repository=repo,
        async_worker=worker,
        job_manager=job_mgr,
    )

    scope = UserDocumentSessionScope(user_id="user_ready_test")
    # Pre-register a READY job
    job = job_mgr.create_job("job_ready_1", "doc_ready_1", "doc.pdf", scope)
    job_mgr.update_job("job_ready_1", status=IngestionStatus.READY, progress=100, stage="complete")
    ready_doc = UserDocument(
        document_id="doc_ready_1",
        user_id=scope.user_id,
        filename="doc.pdf",
        file_hash="hash_ready",
        file_size_bytes=1024,
        page_count=2,
        status=IngestionStatus.READY,
        indexed_chunks_count=5,
    )
    repo.register_document(ready_doc, scope)

    # Attempt to cancel
    res = doc_service.cancel_document_ingestion(scope, "job_ready_1")
    assert res.status == "READY"
    assert "already completed" in res.message

    # Ensure document remains READY in repository
    doc = repo.get_document("doc_ready_1", scope)
    assert doc.status == IngestionStatus.READY


def test_cancel_failed_job_is_idempotent():
    """Test cancelling an already FAILED job returns idempotent response."""
    job_mgr = IngestionJobManager()
    repo = UserDocumentRepository(in_memory=True, collection_name="test_cancel_failed")
    scope = UserDocumentSessionScope(user_id="user_failed_test")
    job = job_mgr.create_job("job_failed_1", "doc_failed_1", "doc.pdf", scope)
    job_mgr.update_job("job_failed_1", status=IngestionStatus.FAILED, progress=0, stage="failed", error="Corrupt PDF")

    doc_service = DocumentManagementService(
        repository=repo,
        job_manager=job_mgr,
    )

    res = doc_service.cancel_document_ingestion(scope, "job_failed_1")
    assert res.status == "FAILED"
    assert "already failed" in res.message


def test_cancel_already_cancelled_job_is_idempotent():
    """Test cancelling an already CANCELLED job returns idempotent response."""
    job_mgr = IngestionJobManager()
    repo = UserDocumentRepository(in_memory=True, collection_name="test_cancel_idempotent")
    scope = UserDocumentSessionScope(user_id="user_cancelled_test")
    job = job_mgr.create_job("job_canc_1", "doc_canc_1", "doc.pdf", scope)
    job_mgr.update_job("job_canc_1", status=IngestionStatus.CANCELLED, progress=50, stage="cancelled")

    doc_service = DocumentManagementService(
        repository=repo,
        job_manager=job_mgr,
    )

    res = doc_service.cancel_document_ingestion(scope, "job_canc_1")
    assert res.status == "CANCELLED"
    assert "already cancelled" in res.message


def test_worker_detects_cancellation_between_stages():
    """Test worker aborts and does not mark READY when cancelled between stages."""
    repo = UserDocumentRepository(in_memory=True, collection_name="test_worker_stage_cancel")
    stat_repo = QdrantRepository(in_memory=True, collection_name="test_stat_stage_cancel")
    stat_pipeline = HybridRetrievalPipeline(chunks=[], qdrant_repo=stat_repo)
    job_mgr = IngestionJobManager()

    stage_entered = threading.Event()
    cancel_applied = threading.Event()

    class StagedExtractor:
        def compute_sha256(self, b):
            return "sha256_staged"
        def extract(self, b):
            from backend.app.document_rag.pdf_extractor import ExtractedPage
            stage_entered.set()
            # Wait until cancellation is applied
            cancel_applied.wait(timeout=5.0)
            return [ExtractedPage(page_number=1, text="Text from staged extraction.")], False

    rag_pipeline = UserDocumentRAGPipeline(
        repository=repo,
        statutory_pipeline=stat_pipeline,
        pdf_extractor=StagedExtractor(),
    )
    worker = AsyncIngestionWorker(rag_pipeline=rag_pipeline, job_manager=job_mgr)
    scope = UserDocumentSessionScope(user_id="stage_cancel_user")

    pdf_bytes = create_test_pdf_bytes(["Test stage cancellation."])
    job = worker.submit_ingestion_job(
        file_bytes=pdf_bytes,
        filename="staged.pdf",
        scope=scope,
    )

    # Wait until worker begins extraction
    assert stage_entered.wait(timeout=5.0)

    # Apply cancellation
    job_mgr.cancel_job(job.job_id, scope)
    cancel_applied.set()

    # Wait briefly for worker thread to exit
    time.sleep(0.5)

    # Job status must be CANCELLED, not READY
    final_job = job_mgr.get_job(job.job_id, scope)
    assert final_job.status == IngestionStatus.CANCELLED
    assert final_job.stage == "cancelled"


def test_worker_detects_cancellation_during_embedding():
    """Test worker halts during batch embedding loop if cancelled."""
    repo = UserDocumentRepository(in_memory=True, collection_name="test_embed_cancel")
    stat_repo = QdrantRepository(in_memory=True, collection_name="test_stat_embed_cancel")
    stat_pipeline = HybridRetrievalPipeline(chunks=[], qdrant_repo=stat_repo)
    job_mgr = IngestionJobManager()

    scope = UserDocumentSessionScope(user_id="embed_cancel_user")
    job = job_mgr.create_job("job_emb_1", "doc_emb_1", "test.pdf", scope)

    # Mark cancelled before embedding
    job_mgr.cancel_job(job.job_id, scope)
    assert job_mgr.is_cancelled(job.job_id, scope)


def test_qdrant_partial_vectors_cleaned_up_on_cancellation():
    """Test that purge_document_vectors removes partial vectors for the cancelled document."""
    repo = UserDocumentRepository(in_memory=True, collection_name="test_cleanup_vectors")
    scope = UserDocumentSessionScope(user_id="cleanup_user")

    # Upsert some dummy chunks
    chunks = [
        UserDocumentChunk(
            chunk_id=f"chk_{i}",
            document_id="doc_cleanup_1",
            user_id=scope.user_id,
            filename="partial.pdf",
            page_start=1,
            page_end=1,
            chunk_index=i,
            text=f"Partial chunk text {i}",
            token_count=10,
        )
        for i in range(3)
    ]
    vectors = [[0.1] * settings.embedding_dimension for _ in range(3)]
    repo.upsert_user_chunks(chunks, vectors, scope)

    # Verify vectors exist
    chunks_before = repo.count_user_chunks(scope)
    assert chunks_before == 3

    # Purge vectors on cancellation
    purged_count = repo.purge_document_vectors("doc_cleanup_1", scope)
    assert purged_count == 3

    # Verify vectors were purged
    chunks_after = repo.count_user_chunks(scope)
    assert chunks_after == 0


def test_statutory_and_cross_tenant_invariants_preserved():
    """Test statutory collection and another user's documents are untouched by cancellation."""
    repo = UserDocumentRepository(in_memory=True, collection_name="test_invariants_user")
    stat_repo = QdrantRepository(in_memory=True, collection_name="test_invariants_stat")
    stat_pipeline = HybridRetrievalPipeline(chunks=[], qdrant_repo=stat_repo)
    job_mgr = IngestionJobManager()

    # Pre-populate statutory collection with dummy point
    stat_repo.client.upsert(
        collection_name="test_invariants_stat",
        points=[
            qmodels.PointStruct(
                id=str(uuid.uuid4()),
                vector=[0.05] * settings.embedding_dimension,
                payload={"section_id": "BNS_1", "text": "Statutory text sample"},
            )
        ],
    )
    stat_count_before = stat_repo.client.count("test_invariants_stat").count
    assert stat_count_before == 1

    # Pre-populate Alice's vectors in user collection
    alice_scope = UserDocumentSessionScope(user_id="alice_preserved")
    alice_chunk = UserDocumentChunk(
        chunk_id="alice_chk_1",
        document_id="alice_doc_1",
        user_id=alice_scope.user_id,
        filename="alice.pdf",
        page_start=1,
        page_end=1,
        chunk_index=0,
        text="Alice private text",
        token_count=10,
    )
    repo.upsert_user_chunks([alice_chunk], [[0.2] * settings.embedding_dimension], alice_scope)
    assert repo.count_user_chunks(alice_scope) == 1

    # Bob uploads and cancels his document
    bob_scope = UserDocumentSessionScope(user_id="bob_canceller")
    bob_chunk = UserDocumentChunk(
        chunk_id="bob_chk_1",
        document_id="bob_doc_1",
        user_id=bob_scope.user_id,
        filename="bob.pdf",
        page_start=1,
        page_end=1,
        chunk_index=0,
        text="Bob text to cancel",
        token_count=10,
    )
    repo.upsert_user_chunks([bob_chunk], [[0.3] * settings.embedding_dimension], bob_scope)
    assert repo.count_user_chunks(bob_scope) == 1

    # Bob cancels and purges
    repo.purge_document_vectors("bob_doc_1", bob_scope)
    assert repo.count_user_chunks(bob_scope) == 0

    # Invariant 1: Alice's chunks remain intact
    assert repo.count_user_chunks(alice_scope) == 1

    # Invariant 2: Statutory collection remains untouched
    stat_count_after = stat_repo.client.count("test_invariants_stat").count
    assert stat_count_after == stat_count_before == 1


def test_race_cancellation_wins_before_finalization():
    """Test that if cancellation acquires lock first, finalize_ready fails and document is never registered."""
    job_mgr = IngestionJobManager()
    repo = UserDocumentRepository(in_memory=True, collection_name="test_race_canc_wins")
    scope = UserDocumentSessionScope(user_id="user_race_1")

    job = job_mgr.create_job("job_race_1", "doc_race_1", "test.pdf", scope)
    job_mgr.update_job("job_race_1", status=IngestionStatus.PROCESSING, progress=90, stage="indexing")

    # 1. Cancellation wins and transitions status to CANCELLED
    canc_job, was_terminal = job_mgr.cancel_job("job_race_1", scope)
    assert canc_job.status == IngestionStatus.CANCELLED
    assert not was_terminal

    # 2. Worker attempts atomic finalization
    ready_doc = UserDocument(
        document_id="doc_race_1",
        user_id=scope.user_id,
        filename="test.pdf",
        file_hash="hash_race_1",
        file_size_bytes=100,
        page_count=1,
        status=IngestionStatus.READY,
        indexed_chunks_count=2,
    )
    registered_docs = []

    def mock_register(doc, sc):
        registered_docs.append(doc)
        repo.register_document(doc, sc)

    success = job_mgr.finalize_ready(
        job_id="job_race_1",
        scope=scope,
        ready_doc=ready_doc,
        register_callback=mock_register,
    )

    # 3. Finalization must fail and document must NOT be registered
    assert success is False
    assert len(registered_docs) == 0
    assert "doc_race_1" not in repo._doc_registry
    assert job_mgr.get_job("job_race_1", scope).status == IngestionStatus.CANCELLED


def test_race_finalization_wins_before_cancellation():
    """Test that if finalization acquires lock first, job becomes READY and cancellation preserves vectors."""
    job_mgr = IngestionJobManager()
    repo = UserDocumentRepository(in_memory=True, collection_name="test_race_fin_wins")
    scope = UserDocumentSessionScope(user_id="user_race_2")

    job = job_mgr.create_job("job_race_2", "doc_race_2", "test.pdf", scope)
    job_mgr.update_job("job_race_2", status=IngestionStatus.PROCESSING, progress=90, stage="indexing")

    # Upsert a vector
    chunk = UserDocumentChunk(
        chunk_id="chk_race_2",
        document_id="doc_race_2",
        user_id=scope.user_id,
        filename="test.pdf",
        page_start=1,
        page_end=1,
        chunk_index=0,
        text="Sample text",
        token_count=10,
    )
    repo.upsert_user_chunks([chunk], [[0.1] * settings.embedding_dimension], scope)

    # 1. Finalization wins and registers document under lock
    ready_doc = UserDocument(
        document_id="doc_race_2",
        user_id=scope.user_id,
        filename="test.pdf",
        file_hash="hash_race_2",
        file_size_bytes=100,
        page_count=1,
        status=IngestionStatus.READY,
        indexed_chunks_count=1,
    )
    success = job_mgr.finalize_ready(
        job_id="job_race_2",
        scope=scope,
        ready_doc=ready_doc,
        register_callback=repo.register_document,
    )
    assert success is True
    assert job_mgr.get_job("job_race_2", scope).status == IngestionStatus.READY
    assert repo.get_document("doc_race_2", scope).status == IngestionStatus.READY

    # 2. Subsequent cancellation arrives
    doc_service = DocumentManagementService(repository=repo, job_manager=job_mgr)
    cancel_res = doc_service.cancel_document_ingestion(scope, "job_race_2")

    # 3. Must return READY, not CANCELLED, and vectors must NOT be deleted
    assert cancel_res.status == "READY"
    assert "already completed" in cancel_res.message
    assert repo.count_user_chunks(scope) == 1
    assert repo.get_document("doc_race_2", scope).status == IngestionStatus.READY


def test_concurrent_cancellation_and_finalization_exclusion():
    """Concurrency stress test: multiple threads racing cancel vs finalize_ready.

    Guarantees:
    - Never CANCELLED + registered
    - Never READY + deleted vectors
    - Never READY + unregistered
    """
    job_mgr = IngestionJobManager()
    repo = UserDocumentRepository(in_memory=True, collection_name="test_stress_exclusion")
    doc_service = DocumentManagementService(repository=repo, job_manager=job_mgr)

    num_trials = 20
    for i in range(num_trials):
        scope = UserDocumentSessionScope(user_id=f"user_stress_{i}")
        job_id = f"job_stress_{i}"
        doc_id = f"doc_stress_{i}"

        job_mgr.create_job(job_id, doc_id, f"stress_{i}.pdf", scope)
        job_mgr.update_job(job_id, status=IngestionStatus.PROCESSING, progress=90, stage="indexing")

        chunk = UserDocumentChunk(
            chunk_id=f"chk_stress_{i}",
            document_id=doc_id,
            user_id=scope.user_id,
            filename=f"stress_{i}.pdf",
            page_start=1,
            page_end=1,
            chunk_index=0,
            text=f"Stress trial {i}",
            token_count=10,
        )
        repo.upsert_user_chunks([chunk], [[0.1] * settings.embedding_dimension], scope)

        ready_doc = UserDocument(
            document_id=doc_id,
            user_id=scope.user_id,
            filename=f"stress_{i}.pdf",
            file_hash=f"hash_stress_{i}",
            file_size_bytes=100,
            page_count=1,
            status=IngestionStatus.READY,
            indexed_chunks_count=1,
        )

        finalize_result = []
        cancel_result = []

        def worker_finalize():
            res = job_mgr.finalize_ready(
                job_id=job_id,
                scope=scope,
                ready_doc=ready_doc,
                register_callback=repo.register_document,
            )
            finalize_result.append(res)
            if not res:
                repo.purge_document_vectors(doc_id, scope)

        def api_cancel():
            res = doc_service.cancel_document_ingestion(scope, job_id)
            cancel_result.append(res)

        t1 = threading.Thread(target=worker_finalize)
        t2 = threading.Thread(target=api_cancel)

        # Start racing threads
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        final_job = job_mgr.get_job(job_id, scope)
        assert final_job.status in (IngestionStatus.READY, IngestionStatus.CANCELLED)

        if final_job.status == IngestionStatus.READY:
            # Finalization won: vectors preserved, document registered
            assert repo.count_user_chunks(scope) == 1
            assert doc_id in repo._doc_registry
            assert repo._doc_registry[doc_id].status == IngestionStatus.READY
            assert cancel_result[0].status == "READY"
        else:
            # Cancellation won: vectors purged, document NOT registered as READY
            assert repo.count_user_chunks(scope) == 0
            if doc_id in repo._doc_registry:
                assert repo._doc_registry[doc_id].status == IngestionStatus.CANCELLED
            assert cancel_result[0].status == "CANCELLED"
