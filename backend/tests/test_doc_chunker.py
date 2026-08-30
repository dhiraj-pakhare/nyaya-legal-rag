"""Tests for User Document Chunker."""

from backend.app.document_rag.chunker import UserDocumentChunker
from backend.app.document_rag.pdf_extractor import ExtractedPage


def test_chunker_deterministic_chunk_ids_and_metadata():
    """Test that chunker creates deterministic chunk IDs and attaches correct metadata."""
    pages = [
        ExtractedPage(page_number=1, text="Paragraph one.\n\nParagraph two with legal claims."),
        ExtractedPage(page_number=2, text="Paragraph three on page two.\n\nParagraph four.")
    ]
    chunker = UserDocumentChunker(target_chunk_chars=1000)
    chunks = chunker.chunk_document(
        pages=pages,
        document_id="doc_test123",
        user_id="user_alice",
        filename="notice.pdf",
        session_id="sess_abc"
    )

    assert len(chunks) >= 2
    assert chunks[0].chunk_id == "doc_test123_p1_c1"
    assert chunks[0].document_id == "doc_test123"
    assert chunks[0].user_id == "user_alice"
    assert chunks[0].session_id == "sess_abc"
    assert chunks[0].filename == "notice.pdf"
    assert chunks[0].page_start == 1
    assert chunks[0].page_end == 1
    assert "Paragraph one." in chunks[0].text


def test_chunker_long_paragraph_splitting():
    """Test splitting long paragraphs while preserving page numbers."""
    long_text = "Word " * 600  # ~3000 chars
    pages = [ExtractedPage(page_number=5, text=long_text)]
    chunker = UserDocumentChunker(target_chunk_chars=500, overlap_chars=50)
    chunks = chunker.chunk_document(
        pages=pages,
        document_id="doc_long",
        user_id="user_bob",
        filename="contract.pdf"
    )

    assert len(chunks) > 1
    for c in chunks:
        assert c.page_start == 5
        assert c.page_end == 5
        assert c.document_id == "doc_long"
        assert c.user_id == "user_bob"
