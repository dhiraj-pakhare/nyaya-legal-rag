"""Tests for User PDF Extractor and Bounded OCR handling."""

import pytest
from backend.app.document_rag.models import (
    CorruptPDFError,
    OCRUnavailableError,
    OversizedDocumentError,
)
from backend.app.document_rag.pdf_extractor import UserPDFExtractor
from backend.tests.doc_test_helpers import create_test_pdf_bytes


def test_pdf_extractor_valid_digital_text():
    """Test extracting clean digital text across multiple pages."""
    pages = [
        "First Information Report\nPolice Station: Cyber Crime Division\nComplainant: Rajesh Kumar\nAccused: Unknown",
        "Page 2 Details:\nThe accused allegedly initiated unauthorized access into the server on 14th August 2024.",
        "Page 3 Conclusions:\nThe investigation officer requested forensic seizure of the server hard drives."
    ]
    pdf_bytes = create_test_pdf_bytes(pages)
    extractor = UserPDFExtractor()
    extracted_pages, has_ocr = extractor.extract(pdf_bytes)

    assert len(extracted_pages) == 3
    assert "First Information Report" in extracted_pages[0].text
    assert extracted_pages[0].page_number == 1
    assert "unauthorized access" in extracted_pages[1].text
    assert extracted_pages[1].page_number == 2
    assert "forensic seizure" in extracted_pages[2].text
    assert extracted_pages[2].page_number == 3
    assert has_ocr is False


def test_pdf_extractor_corrupt_pdf():
    """Test that malformed PDF bytes raise CorruptPDFError."""
    corrupt_bytes = b"This is not a valid PDF file at all."
    extractor = UserPDFExtractor()
    with pytest.raises(CorruptPDFError):
        extractor.extract(corrupt_bytes)


def test_pdf_extractor_empty_bytes():
    """Test that empty byte stream raises CorruptPDFError."""
    extractor = UserPDFExtractor()
    with pytest.raises(CorruptPDFError):
        extractor.extract(b"")


def test_pdf_extractor_oversized_file():
    """Test that files exceeding size limit raise OversizedDocumentError."""
    pages = ["A small page of text."]
    pdf_bytes = create_test_pdf_bytes(pages)
    # Set max size to 10 bytes to trigger oversize error
    extractor = UserPDFExtractor(max_size_bytes=10)
    with pytest.raises(OversizedDocumentError):
        extractor.extract(pdf_bytes)


def test_pdf_extractor_password_protected():
    """Test that encrypted/password-protected PDFs raise CorruptPDFError."""
    pages = ["Secret legal notice confidential."]
    encrypted_bytes = create_test_pdf_bytes(pages, password="SecretPassword123")
    extractor = UserPDFExtractor()
    with pytest.raises(CorruptPDFError) as exc_info:
        extractor.extract(encrypted_bytes)
    assert "password-protected" in str(exc_info.value).lower()


def test_pdf_extractor_scanned_without_text_raises_ocr_unavailable():
    """Test that a blank/scanned PDF without text layer raises OCRUnavailableError when OCR is disabled."""
    blank_pages = [" "]  # Whitespace only
    pdf_bytes = create_test_pdf_bytes(blank_pages)
    extractor = UserPDFExtractor(ocr_enabled=False)
    with pytest.raises(OCRUnavailableError) as exc_info:
        extractor.extract(pdf_bytes)
    assert "UNSUPPORTED_SCANNED_PDF" in str(exc_info.value)
