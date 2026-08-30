"""User PDF Extractor with Bounded OCR Fallback and Validation."""

import hashlib
import io
import logging
from typing import List, Optional, Tuple
import pypdf

from backend.app.core.config import settings
from backend.app.document_rag.models import (
    CorruptPDFError,
    OCRUnavailableError,
    OversizedDocumentError
)

logger = logging.getLogger("nyaya.document_rag.extractor")


class ExtractedPage:
    """Extracted text payload for a single PDF page."""

    def __init__(self, page_number: int, text: str, is_ocr: bool = False):
        self.page_number = page_number
        self.text = text.strip()
        self.is_ocr = is_ocr


class UserPDFExtractor:
    """Extracts digital text from uploaded user PDFs with bounded OCR fallback."""

    def __init__(
        self,
        max_size_bytes: int = settings.max_user_doc_size_bytes,
        ocr_enabled: bool = settings.ocr_enabled,
        ocr_max_pages: int = settings.ocr_max_pages,
        ocr_timeout: float = settings.ocr_timeout_seconds
    ):
        self.max_size_bytes = max_size_bytes
        self.ocr_enabled = ocr_enabled
        self.ocr_max_pages = ocr_max_pages
        self.ocr_timeout = ocr_timeout

    def compute_sha256(self, file_bytes: bytes) -> str:
        """Compute SHA-256 hash of file bytes for deterministic deduplication."""
        return hashlib.sha256(file_bytes).hexdigest()

    def validate_file_bytes(self, file_bytes: bytes) -> None:
        """Validate file size and PDF magic bytes."""
        if not file_bytes:
            raise CorruptPDFError("Uploaded file is empty (0 bytes).")

        if len(file_bytes) > self.max_size_bytes:
            raise OversizedDocumentError(
                f"File size ({len(file_bytes)} bytes) exceeds maximum limit of {self.max_size_bytes} bytes."
            )

        if not file_bytes.startswith(b"%PDF-"):
            raise CorruptPDFError("Uploaded file does not start with standard PDF magic bytes (%PDF-).")

    def _try_ocr_page(self, page_obj, page_num: int) -> Optional[str]:
        """Attempt bounded OCR if local OCR tools are available."""
        if not self.ocr_enabled:
            return None
        try:
            import pytesseract
            # Check for images in pypdf page
            for count, image_file_object in enumerate(page_obj.images):
                from PIL import Image
                img = Image.open(io.BytesIO(image_file_object.data))
                ocr_text = pytesseract.image_to_string(img, timeout=self.ocr_timeout)
                if ocr_text and ocr_text.strip():
                    return ocr_text.strip()
        except Exception as e:
            logger.debug(f"OCR execution skipped or unavailable on page {page_num}: {e}")
        return None

    def extract(self, file_bytes: bytes) -> Tuple[List[ExtractedPage], bool]:
        """Extract pages from PDF bytes using pypdf with bounded OCR fallback.
        
        Returns:
            (List[ExtractedPage], has_ocr_applied: bool)
        """
        self.validate_file_bytes(file_bytes)

        try:
            reader = pypdf.PdfReader(io.BytesIO(file_bytes))
            if reader.is_encrypted:
                try:
                    decrypted = reader.decrypt("")
                    if decrypted == 0:
                        raise CorruptPDFError("Uploaded PDF is password-protected and cannot be parsed.")
                except Exception:
                    raise CorruptPDFError("Uploaded PDF is password-protected and cannot be parsed.")

            page_count = len(reader.pages)
            if page_count == 0:
                raise CorruptPDFError("Uploaded PDF contains 0 pages.")
        except CorruptPDFError:
            raise
        except Exception as e:
            raise CorruptPDFError(f"Malformed or unreadable PDF: {str(e)}") from e

        extracted_pages: List[ExtractedPage] = []
        has_ocr = False
        total_chars = 0

        for idx, page in enumerate(reader.pages, start=1):
            try:
                raw_text = page.extract_text() or ""
            except Exception as e:
                logger.warning(f"Error extracting text on page {idx}: {e}")
                raw_text = ""

            clean_text = raw_text.strip()
            is_page_ocr = False

            # If page has very little text and OCR is enabled, attempt bounded OCR
            if len(clean_text) < 50 and self.ocr_enabled and idx <= self.ocr_max_pages:
                ocr_result = self._try_ocr_page(page, idx)
                if ocr_result:
                    clean_text = ocr_result
                    is_page_ocr = True
                    has_ocr = True

            total_chars += len(clean_text)
            extracted_pages.append(ExtractedPage(page_number=idx, text=clean_text, is_ocr=is_page_ocr))

        # If document has zero extractable text across all pages
        if total_chars == 0:
            raise OCRUnavailableError(
                "UNSUPPORTED_SCANNED_PDF: Document contains scanned images without a digital text layer "
                "and OCR engine is unavailable or produced no text."
            )

        return extracted_pages, has_ocr
