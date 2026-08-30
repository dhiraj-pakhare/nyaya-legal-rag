"""User Document Application Service (Part D & Phase 8).

Coordinates asynchronous PDF upload, background worker dispatch, job status tracking,
scoped multi-tenant document lifecycle, retrieval, and deletion with anti-enumeration protection.
"""

from datetime import datetime, timezone
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
    DocumentStatusDTO,
    DocumentUploadResponseDTO,
)
from backend.app.core.config import settings
from backend.app.document_rag.models import (
    CorruptPDFError,
    DocumentNotFoundError as DomainDocumentNotFoundError,
    IngestionStatus,
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
from backend.app.workers.ingestion_worker import AsyncIngestionWorker, get_async_worker
from backend.app.workers.job_manager import IngestionJobManager, get_job_manager

logger = logging.getLogger("nyaya.services.documents")


class DocumentManagementService:
    """Application service for multi-tenant user document lifecycle management."""

    def __init__(
        self,
        rag_pipeline: Optional[UserDocumentRAGPipeline] = None,
        repository: Optional[UserDocumentRepository] = None,
        async_worker: Optional[AsyncIngestionWorker] = None,
        job_manager: Optional[IngestionJobManager] = None
    ):
        self.pipeline = rag_pipeline or get_user_doc_rag_pipeline()
        self.repository = repository or get_user_doc_repository()
        self.job_manager = job_manager or get_job_manager()
        self.async_worker = async_worker or AsyncIngestionWorker(
            rag_pipeline=self.pipeline,
            job_manager=self.job_manager
        )

    async def upload_and_ingest(
        self,
        scope: UserDocumentSessionScope,
        file: UploadFile
    ) -> DocumentUploadResponseDTO:
        """Validate upload and submit document for asynchronous background ingestion."""
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
            # 5. Submit job to background ingestion worker without blocking
            job = self.async_worker.submit_ingestion_job(
                file_bytes=content,
                filename=safe_filename,
                scope=scope
            )

            user_log_hash = sanitize_for_logs({"user_id": scope.user_id}).get("user_id_hash", "anon")
            logger.info(
                f"Async document upload job '{job.job_id}' created for user={user_log_hash}: "
                f"doc_id={job.document_id}, status={job.status.value}"
            )

            return DocumentUploadResponseDTO(
                job_id=job.job_id,
                document_id=job.document_id,
                filename=job.filename,
                status=job.status.value,
                progress=job.progress,
                stage=job.stage,
                created_at=job.created_at.isoformat(),
                message="Document upload accepted for asynchronous processing."
            )

        except CorruptPDFError as e:
            logger.warning(f"Corrupt PDF upload: {str(e)}")
            raise UnsupportedMediaTypeError(f"Corrupted or unreadable PDF: {str(e)}")
        except OversizedDocumentError as e:
            logger.warning(f"Oversized document: {str(e)}")
            raise PayloadTooLargeError(str(e))
        except Exception as e:
            logger.warning(f"Document upload dispatch failed: {str(e)}")
            raise APIError(message=f"Document ingestion failed: {str(e)}", code="INGESTION_FAILED", status_code=422)

    def get_document_status(
        self,
        scope: UserDocumentSessionScope,
        document_id_or_job_id: str
    ) -> DocumentStatusDTO:
        """Get ingestion status and progress for a document/job. Enforces uniform 404 anti-enumeration."""
        scope.validate_scope()
        target = document_id_or_job_id.strip()

        # 1. Try finding job by job_id
        job = self.job_manager.get_job(target, scope)
        if not job:
            # 2. Try finding job by document_id
            job = self.job_manager.get_job_by_document(target, scope)

        if job:
            return DocumentStatusDTO(
                job_id=job.job_id,
                document_id=job.document_id,
                status=job.status.value,
                progress=job.progress,
                stage=job.stage,
                error=job.error,
                page_count=job.page_count,
                chunk_count=job.chunk_count,
                updated_at=job.updated_at.isoformat()
            )

        # 3. Fallback: check document registry in repository
        try:
            doc = self.repository.get_document(target, scope)
            progress_val = 100 if doc.status == IngestionStatus.READY else (0 if doc.status == IngestionStatus.FAILED else 50)
            stage_val = "complete" if doc.status == IngestionStatus.READY else ("failed" if doc.status == IngestionStatus.FAILED else "processing")
            return DocumentStatusDTO(
                job_id=f"job_{doc.document_id[:12]}",
                document_id=doc.document_id,
                status=doc.status.value,
                progress=progress_val,
                stage=stage_val,
                error=doc.error_message,
                page_count=doc.page_count,
                chunk_count=doc.indexed_chunks_count,
                updated_at=doc.uploaded_at.isoformat()
            )
        except DomainDocumentNotFoundError:
            raise APIDocumentNotFoundError()

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
