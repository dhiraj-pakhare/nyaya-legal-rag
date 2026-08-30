"""Tests for Document Existence Privacy, Anti-Enumeration, and Log Sanitization."""

import pytest
import numpy as np
from backend.app.document_rag.models import (
    DocumentNotFoundError,
    UserDocument,
    UserDocumentChunk,
    UserDocumentSessionScope,
)
from backend.app.document_rag.repository import UserDocumentRepository
from backend.app.document_rag.security import sanitize_filename, sanitize_for_logs


def test_anti_enumeration_uniform_404_on_get():
    """Test that unauthorized document lookups produce identical error to non-existent documents."""
    repo = UserDocumentRepository(in_memory=True, collection_name="test_privacy_anti_enum")
    scope_alice = UserDocumentSessionScope(user_id="user_alice")
    scope_bob = UserDocumentSessionScope(user_id="user_bob")

    # Bob registers a document
    doc_bob = UserDocument(
        document_id="doc_bob_secret",
        user_id="user_bob",
        filename="secret.pdf",
        file_hash="hash_b",
        file_size_bytes=100,
        page_count=1
    )
    repo.register_document(doc_bob, scope=scope_bob)

    # 1. Alice queries non-existent document
    with pytest.raises(DocumentNotFoundError) as exc_nonexistent:
        repo.get_document("doc_totally_nonexistent", scope=scope_alice)

    # 2. Alice queries Bob's existing document
    with pytest.raises(DocumentNotFoundError) as exc_unauthorized:
        repo.get_document("doc_bob_secret", scope=scope_alice)

    # Verify both error messages are identical (Zero information leakage)
    assert str(exc_nonexistent.value) == str(exc_unauthorized.value)
    assert str(exc_nonexistent.value) == "Document not found or inaccessible"


def test_anti_enumeration_uniform_404_on_delete():
    """Test that unauthorized deletion attempts produce identical 404 behavior."""
    repo = UserDocumentRepository(in_memory=True, collection_name="test_privacy_delete_enum")
    scope_alice = UserDocumentSessionScope(user_id="user_alice")
    scope_bob = UserDocumentSessionScope(user_id="user_bob")

    # Bob uploads and registers a document
    doc_bob = UserDocument(
        document_id="doc_bob_delete_target",
        user_id="user_bob",
        filename="bob.pdf",
        file_hash="hash_bob",
        file_size_bytes=50,
        page_count=1
    )
    repo.register_document(doc_bob, scope=scope_bob)

    # Alice tries to delete non-existent document
    with pytest.raises(DocumentNotFoundError) as exc_1:
        repo.delete_document("doc_fake", scope=scope_alice)

    # Alice tries to delete Bob's document
    with pytest.raises(DocumentNotFoundError) as exc_2:
        repo.delete_document("doc_bob_delete_target", scope=scope_alice)

    assert str(exc_1.value) == str(exc_2.value) == "Document not found or inaccessible"


def test_list_and_count_isolation():
    """Test that list and count operations never reflect another user's documents."""
    repo = UserDocumentRepository(in_memory=True, collection_name="test_privacy_list_count")
    scope_alice = UserDocumentSessionScope(user_id="user_alice")
    scope_bob = UserDocumentSessionScope(user_id="user_bob")

    chunk_bob = UserDocumentChunk(
        chunk_id="doc_bob_p1_c1",
        document_id="doc_bob",
        user_id="user_bob",
        filename="bob.pdf",
        page_start=1,
        page_end=1,
        chunk_index=1,
        text="Bob chunk.",
        token_count=2
    )
    repo.upsert_user_chunks([chunk_bob], np.zeros((1, 768)), scope=scope_bob)

    # Alice's chunk count must be 0
    assert repo.count_user_chunks(scope=scope_alice) == 0
    # Alice's list must be empty
    assert len(repo.list_documents(scope=scope_alice)) == 0


def test_log_sanitizer_removes_sensitive_data():
    """Test that log sanitizer redacts raw document text and hashes user_id."""
    raw_log = {
        "user_id": "confidential_client_789",
        "document_id": "doc_123",
        "text": "Highly confidential trade secret text.",
        "page_count": 5
    }
    sanitized = sanitize_for_logs(raw_log)

    assert sanitized["text"] == "<REDACTED_CONTENT>"
    assert "confidential_client_789" not in sanitized.values()
    assert "user_id_hash" in sanitized
    assert sanitized["document_id"] == "doc_123"
    assert sanitized["page_count"] == 5


def test_filename_sanitizer():
    """Test sanitizing filename against directory traversal attacks."""
    malicious_filename = "../../../etc/passwd"
    clean = sanitize_filename(malicious_filename)
    assert ".." not in clean
    assert clean == "passwd"
