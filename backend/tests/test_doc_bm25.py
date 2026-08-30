"""Tests for Multi-Worker BM25 Index Builder and Cache Management."""

from backend.app.document_rag.bm25 import UserDocumentBM25Manager, UserDocumentBM25Index
from backend.app.document_rag.models import UserDocumentChunk, UserDocumentSessionScope


def test_bm25_deterministic_reconstruction():
    """Test that two separate worker instances construct identical BM25 rankings from the same chunks."""
    chunks = [
        UserDocumentChunk(
            chunk_id="doc_1_p1_c1",
            document_id="doc_1",
            user_id="user_1",
            filename="fir.pdf",
            page_start=1,
            page_end=1,
            chunk_index=1,
            text="The complainant lodged a complaint regarding extortion and blackmail.",
            token_count=10
        ),
        UserDocumentChunk(
            chunk_id="doc_1_p2_c2",
            document_id="doc_1",
            user_id="user_1",
            filename="fir.pdf",
            page_start=2,
            page_end=2,
            chunk_index=2,
            text="The bank statements show unexplained fund transfers to the accused.",
            token_count=10
        )
    ]

    # Worker 1 constructs index
    index_worker_1 = UserDocumentBM25Index(chunks)
    res_1 = index_worker_1.search("extortion and blackmail", top_k=2)

    # Worker 2 constructs index from same chunks
    index_worker_2 = UserDocumentBM25Index(chunks)
    res_2 = index_worker_2.search("extortion and blackmail", top_k=2)

    assert len(res_1) == len(res_2) == 1
    assert res_1[0].chunk_id == res_2[0].chunk_id == "doc_1_p1_c1"
    assert abs(res_1[0].score - res_2[0].score) < 1e-6


def test_bm25_manager_caching_and_invalidation():
    """Test manager caching and scope invalidation."""
    manager = UserDocumentBM25Manager()
    scope = UserDocumentSessionScope(user_id="user_alice", active_document_ids=["doc_1"])

    chunks = [
        UserDocumentChunk(
            chunk_id="doc_1_p1_c1",
            document_id="doc_1",
            user_id="user_alice",
            filename="a.pdf",
            page_start=1,
            page_end=1,
            chunk_index=1,
            text="Arbitration notice regarding breach of contract.",
            token_count=6
        )
    ]

    # First access builds and caches
    idx1 = manager.get_or_build_index(chunks, scope)
    # Second access returns cached object
    idx2 = manager.get_or_build_index(chunks, scope)
    assert idx1 is idx2

    # Invalidate
    manager.invalidate(scope)
    # Third access builds new instance
    idx3 = manager.get_or_build_index(chunks, scope)
    assert idx3 is not idx1
