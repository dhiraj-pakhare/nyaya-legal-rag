"""Tests for Multi-Tenant Isolation, Cross-User Isolation, and Scope Enforcement."""

import pytest
import numpy as np
from backend.app.document_rag.models import (
    DocumentNotFoundError,
    SecurityScopeError,
    UserDocumentChunk,
    UserDocumentSessionScope,
)
from backend.app.document_rag.repository import UserDocumentRepository


def test_cross_user_search_isolation():
    """Test that User A cannot retrieve User B's chunks even with an identical query vector."""
    repo = UserDocumentRepository(in_memory=True, collection_name="test_isolation_cross_user")

    scope_user_a = UserDocumentSessionScope(user_id="user_alice", active_document_ids=["doc_alice"])
    scope_user_b = UserDocumentSessionScope(user_id="user_bob", active_document_ids=["doc_bob"])

    vec = np.ones((1, 768), dtype=np.float32)

    # Index User A chunk
    chunk_a = UserDocumentChunk(
        chunk_id="doc_alice_p1_c1",
        document_id="doc_alice",
        user_id="user_alice",
        filename="alice_secret.pdf",
        page_start=1,
        page_end=1,
        chunk_index=1,
        text="Alice's confidential financial audit.",
        token_count=5
    )
    repo.upsert_user_chunks([chunk_a], vec, scope=scope_user_a)

    # Index User B chunk with exact same vector
    chunk_b = UserDocumentChunk(
        chunk_id="doc_bob_p1_c1",
        document_id="doc_bob",
        user_id="user_bob",
        filename="bob_secret.pdf",
        page_start=1,
        page_end=1,
        chunk_index=1,
        text="Bob's private employment dispute.",
        token_count=5
    )
    repo.upsert_user_chunks([chunk_b], vec, scope=scope_user_b)

    # User A searches with identical vector -> must ONLY return Alice's chunk
    results_a = repo.search_dense(vec[0].tolist(), scope=scope_user_a, limit=10)
    assert len(results_a) == 1
    assert results_a[0].chunk_id == "doc_alice_p1_c1"
    assert results_a[0].user_id == "user_alice"

    # User B searches with identical vector -> must ONLY return Bob's chunk
    results_b = repo.search_dense(vec[0].tolist(), scope=scope_user_b, limit=10)
    assert len(results_b) == 1
    assert results_b[0].chunk_id == "doc_bob_p1_c1"
    assert results_b[0].user_id == "user_bob"


def test_cross_session_isolation():
    """Test that Session 2 scoped to doc 2 cannot retrieve doc 1 belonging to Session 1."""
    repo = UserDocumentRepository(in_memory=True, collection_name="test_isolation_cross_session")

    scope_sess1 = UserDocumentSessionScope(user_id="user_alice", session_id="sess_1", active_document_ids=["doc_1"])
    scope_sess2 = UserDocumentSessionScope(user_id="user_alice", session_id="sess_2", active_document_ids=["doc_2"])

    vec1 = np.ones((1, 768), dtype=np.float32)
    vec2 = np.ones((1, 768), dtype=np.float32)

    chunk1 = UserDocumentChunk(
        chunk_id="doc_1_p1_c1",
        document_id="doc_1",
        user_id="user_alice",
        session_id="sess_1",
        filename="case1.pdf",
        page_start=1,
        page_end=1,
        chunk_index=1,
        text="Case 1 documents.",
        token_count=4
    )
    chunk2 = UserDocumentChunk(
        chunk_id="doc_2_p1_c1",
        document_id="doc_2",
        user_id="user_alice",
        session_id="sess_2",
        filename="case2.pdf",
        page_start=1,
        page_end=1,
        chunk_index=1,
        text="Case 2 documents.",
        token_count=4
    )

    repo.upsert_user_chunks([chunk1], vec1, scope=scope_sess1)
    repo.upsert_user_chunks([chunk2], vec2, scope=scope_sess2)

    # Session 2 search -> must only return doc_2
    results_sess2 = repo.search_dense(vec1[0].tolist(), scope=scope_sess2, limit=10)
    assert len(results_sess2) == 1
    assert results_sess2[0].document_id == "doc_2"


def test_manipulated_chunk_user_id_rejected():
    """Test that attempting to index a chunk with user_id mismatched from caller scope raises SecurityScopeError."""
    repo = UserDocumentRepository(in_memory=True, collection_name="test_isolation_spoof")
    scope = UserDocumentSessionScope(user_id="user_attacker")

    # Chunk attempts to spoof victim user_id
    spoofed_chunk = UserDocumentChunk(
        chunk_id="doc_spoof_p1_c1",
        document_id="doc_spoof",
        user_id="user_victim",
        filename="spoofed.pdf",
        page_start=1,
        page_end=1,
        chunk_index=1,
        text="Spoofed text.",
        token_count=2
    )

    with pytest.raises(SecurityScopeError):
        repo.upsert_user_chunks([spoofed_chunk], np.zeros((1, 768)), scope=scope)


def test_manipulated_document_id_lookup_rejected():
    """Test that User A querying User B's document_id gets DocumentNotFoundError."""
    repo = UserDocumentRepository(in_memory=True, collection_name="test_isolation_lookup")
    scope_b = UserDocumentSessionScope(user_id="user_bob", active_document_ids=["doc_bob_private"])
    scope_a = UserDocumentSessionScope(user_id="user_alice", active_document_ids=["doc_bob_private"])

    chunk = UserDocumentChunk(
        chunk_id="doc_bob_private_p1_c1",
        document_id="doc_bob_private",
        user_id="user_bob",
        filename="bob_secret.pdf",
        page_start=1,
        page_end=1,
        chunk_index=1,
        text="Secret contents.",
        token_count=2
    )
    repo.upsert_user_chunks([chunk], np.ones((1, 768)), scope=scope_b)

    # User A tries to get Bob's document chunks
    with pytest.raises(DocumentNotFoundError):
        repo.get_document_chunks("doc_bob_private", scope=scope_a)
