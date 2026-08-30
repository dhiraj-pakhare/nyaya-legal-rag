"""User Document Application Service (Phase 8).

Coordinates synchronous PDF upload, file security validation, temporary file lifecycle,
scoped multi-tenant listing, retrieval, and deletion with anti-enumeration protection.
"""

import logging
from typing import Any, Dict, List, Optional
from fastapi import UploadFile

from backend.app.api.errors import (
    APIError,
    DocumentNotFoundError as APIDocumentNotFoundError,
    PayloadTooLargeError,
    UnsupportedMediaTypeError,
)
from backend.app.api.schemas.documents import (
    DocumentDetailDTO,
    DocumentIngestResponseDTO,
    DocumentListItemDTO,
)
from backend.app.core.config import settings
from backend.app.document_rag.models import (
    CorruptPDFError,
    DocumentNotFoundError as DomainDocumentNotFoundError,
    OversizedDocumentError,
    UserDocumentSessionScope,
)
from backend.app.document_rag.pipeline import (
    UserDocumentRAGPipeline,
    get_user_doc_rag_pipeline,
)
from backend.app.document_rag.repository import (
    UserDocumentRepository,
    get_user_doc_repository,
)
from backend.app.document_rag.security import sanitize_filename, sanitize_for_logs

logger = logging.getLogger("nyaya.services.documents")


class DocumentManagementService:
    """Application service for multi-tenant user document lifecycle management."""

    def __init__(
        self,
        rag_pipeline: Optional[UserDocumentRAGPipeline] = None,
        repository: Optional[UserDocumentRepository] = None
    ):
        self.pipeline = rag_pipeline or get_user_doc_rag_pipeline()
        self.repository = repository or get_user_doc_repository()

    async def upload_and_ingest(
        self,
        scope: UserDocumentSessionScope,
        file: UploadFile
    ) -> DocumentIngestResponseDTO:
        """Validate, stage, and synchronously ingest user PDF into isolated vector/BM25 storage."""
        # 1. Filename sanitization
        original_filename = file.filename or "uploaded_document.pdf"
        safe_filename = sanitize_filename(original_filename)

        # 2. Content-Type and extension check
        if not safe_filename.lower().endswith(".pdf") and file.content_type != "application/pdf":
            raise UnsupportedMediaTypeError("Only PDF documents (.pdf) are supported.")

        # 3. Read content and validate size
        content = await file.read()
        file_size = len(content)

        if file_size > settings.max_user_doc_size_bytes:
            raise PayloadTooLargeError(
                f"File size ({file_size} bytes) exceeds maximum limit of {settings.max_user_doc_size_bytes} bytes (25MB)."
            )

        if file_size < 10:
            raise UnsupportedMediaTypeError("Uploaded file is empty or corrupted.")

        # 4. Magic-byte verification (%PDF-)
        if not content.startswith(b"%PDF-"):
            raise UnsupportedMediaTypeError("Uploaded file does not contain valid PDF header magic bytes (%PDF-).")

        try:
            # 5. Synchronous Ingestion Execution
            ingest_res = self.pipeline.ingest_pdf(
                file_bytes=content,
                filename=safe_filename,
                scope=scope
            )

            doc = ingest_res.document
            user_log_hash = sanitize_for_logs({"user_id": scope.user_id}).get("user_id_hash", "anon")
            logger.info(
                f"Document successfully ingested for user={user_log_hash}: "
                f"doc_id={doc.document_id}, pages={doc.page_count}, chunks={ingest_res.chunks_count}"
            )

            return DocumentIngestResponseDTO(
                document_id=doc.document_id,
                filename=doc.filename,
                status=doc.status.value,
                page_count=doc.page_count,
                chunk_count=ingest_res.chunks_count,
                file_size_bytes=doc.file_size_bytes,
                created_at=doc.uploaded_at.isoformat(),
                message="Document successfully ingested and indexed."
            )

        except CorruptPDFError as e:
            logger.warning(f"Corrupt PDF upload: {str(e)}")
            raise UnsupportedMediaTypeError(f"Corrupted or unreadable PDF: {str(e)}")
        except OversizedDocumentError as e:
            logger.warning(f"Oversized document: {str(e)}")
            raise PayloadTooLargeError(str(e))
        except Exception as e:
            logger.warning(f"Document ingestion failed: {str(e)}")
            raise APIError(message=f"Document ingestion failed: {str(e)}", code="INGESTION_FAILED", status_code=422)

    def list_documents(self, scope: UserDocumentSessionScope) -> List[DocumentListItemDTO]:
        """List all documents owned by the authenticated principal."""
        items = self.repository.list_documents(scope)
        return [
            DocumentListItemDTO(
                document_id=doc.document_id,
                filename=doc.filename,
                file_size_bytes=doc.file_size_bytes,
                page_count=doc.page_count,
                chunk_count=doc.indexed_chunks_count,
                created_at=doc.uploaded_at.isoformat(),
                status=doc.status.value
            )
            for doc in items
        ]

    def get_document(
        self,
        scope: UserDocumentSessionScope,
        document_id: str
    ) -> DocumentDetailDTO:
        """Retrieve scoped document metadata. Returns uniform 404 for unowned or missing IDs."""
        try:
            doc = self.repository.get_document(document_id, scope)
        except DomainDocumentNotFoundError:
            raise APIDocumentNotFoundError()

        return DocumentDetailDTO(
            document_id=doc.document_id,
            filename=doc.filename,
            file_size_bytes=doc.file_size_bytes,
            page_count=doc.page_count,
            chunk_count=doc.indexed_chunks_count,
            created_at=doc.uploaded_at.isoformat(),
            status=doc.status.value,
            sha256_hash=doc.file_hash
        )

    def delete_document(
        self,
        scope: UserDocumentSessionScope,
        document_id: str
    ) -> Dict[str, Any]:
        """Delete scoped document. Returns uniform 404 for unowned or missing IDs."""
        try:
            self.repository.delete_document(document_id, scope)
        except DomainDocumentNotFoundError:
            raise APIDocumentNotFoundError()

        user_log_hash = sanitize_for_logs({"user_id": scope.user_id}).get("user_id_hash", "anon")
        logger.info(f"Document {document_id} deleted for user={user_log_hash}")
        return {"deleted": True, "document_id": document_id}


_document_service_instance: Optional[DocumentManagementService] = None


def get_document_service() -> DocumentManagementService:
    """Singleton provider for DocumentManagementService."""
    global _document_service_instance
    if _document_service_instance is None:
        _document_service_instance = DocumentManagementService()
    return _document_service_instance
