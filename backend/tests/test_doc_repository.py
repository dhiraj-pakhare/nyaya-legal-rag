"""Tests for UserDocumentRepository multi-tenant storage, scoping, and deletion."""

import pytest
import numpy as np
from backend.app.document_rag.models import (
    DocumentNotFoundError,
    SecurityScopeError,
    UserDocument,
    UserDocumentChunk,
    UserDocumentSessionScope,
)
from backend.app.document_rag.repository import UserDocumentRepository


def test_repository_upsert_and_scoped_search():
    """Test indexing chunks and executing scoped dense search."""
    repo = UserDocumentRepository(in_memory=True, collection_name="test_repo_scoped")
    scope = UserDocumentSessionScope(user_id="user_123", active_document_ids=["doc_abc"])

    chunks = [
        UserDocumentChunk(
            chunk_id="doc_abc_p1_c1",
            document_id="doc_abc",
            user_id="user_123",
            filename="notice.pdf",
            page_start=1,
            page_end=1,
            chunk_index=1,
            text="Notice of arbitration under clause 14.",
            token_count=10
        )
    ]
    vectors = np.random.randn(1, 768).astype(np.float32)

    upserted = repo.upsert_user_chunks(chunks, vectors, scope=scope)
    assert upserted == 1

    # Search with matching scope
    query_vec = vectors[0].tolist()
    results = repo.search_dense(query_vec, scope=scope, limit=5)
    assert len(results) == 1
    assert results[0].chunk_id == "doc_abc_p1_c1"
    assert results[0].user_id == "user_123"


def test_repository_empty_scope_raises_security_error():
    """Test that calling repository operations with empty user_id raises SecurityScopeError."""
    repo = UserDocumentRepository(in_memory=True, collection_name="test_repo_empty_scope")
    invalid_scope = UserDocumentSessionScope(user_id="")

    with pytest.raises(SecurityScopeError):
        repo.search_dense([0.1] * 768, scope=invalid_scope)


def test_repository_scoped_deletion():
    """Test that deleting a document purges points and makes subsequent lookups fail."""
    repo = UserDocumentRepository(in_memory=True, collection_name="test_repo_delete")
    scope = UserDocumentSessionScope(user_id="user_alice", active_document_ids=["doc_to_delete"])

    chunks = [
        UserDocumentChunk(
            chunk_id="doc_to_delete_p1_c1",
            document_id="doc_to_delete",
            user_id="user_alice",
            filename="temp.pdf",
            page_start=1,
            page_end=1,
            chunk_index=1,
            text="Temporary document content.",
            token_count=5
        )
    ]
    vectors = np.random.randn(1, 768).astype(np.float32)
    repo.upsert_user_chunks(chunks, vectors, scope=scope)

    # Register in metadata registry
    doc = UserDocument(
        document_id="doc_to_delete",
        user_id="user_alice",
        filename="temp.pdf",
        file_hash="hash123",
        file_size_bytes=100,
        page_count=1,
        indexed_chunks_count=1
    )
    repo.register_document(doc, scope)

    # Verify retrieval works
    retrieved = repo.get_document_chunks("doc_to_delete", scope)
    assert len(retrieved) == 1

    # Delete
    deleted_count = repo.delete_document("doc_to_delete", scope)
    assert deleted_count == 1

    # Verify subsequent lookup raises DocumentNotFoundError
    with pytest.raises(DocumentNotFoundError):
        repo.get_document_chunks("doc_to_delete", scope)
