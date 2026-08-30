"""Asynchronous Background Worker Engine for User Document Ingestion."""

from concurrent.futures import ThreadPoolExecutor
import logging
import uuid
from typing import Optional

from backend.app.core.qdrant_repo import NYAYA_NAMESPACE
from backend.app.document_rag.models import (
    CorruptPDFError,
    IngestionStatus,
    UserDocument,
    UserDocumentSessionScope,
)
from backend.app.document_rag.pipeline import UserDocumentRAGPipeline
from backend.app.workers.job_manager import IngestionJob, IngestionJobManager, get_job_manager

logger = logging.getLogger("nyaya.workers.ingestion_worker")


class AsyncIngestionWorker:
    """Background worker engine executing document extraction, chunking, embedding, and vector indexing."""

    def __init__(
        self,
        rag_pipeline: Optional[UserDocumentRAGPipeline] = None,
        job_manager: Optional[IngestionJobManager] = None,
        max_workers: int = 4
    ):
        self.pipeline = rag_pipeline or UserDocumentRAGPipeline()
        self.job_manager = job_manager or get_job_manager()
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="ingest-worker")

    def submit_ingestion_job(
        self,
        file_bytes: bytes,
        filename: str,
        scope: UserDocumentSessionScope,
        job_id: Optional[str] = None,
        document_id: Optional[str] = None
    ) -> IngestionJob:
        """Queue and submit document ingestion job asynchronously without blocking caller."""
        scope.validate_scope()
        file_hash = self.pipeline.pdf_extractor.compute_sha256(file_bytes)
        doc_id = document_id or str(uuid.uuid5(NYAYA_NAMESPACE, f"nyaya://doc/{scope.user_id}/{file_hash}"))
        j_id = job_id or f"job_{uuid.uuid4().hex[:12]}"

        # Deduplication check: return READY status immediately if doc is already indexed
        try:
            existing_doc = self.pipeline.repository.get_document(doc_id, scope)
            if existing_doc.status == IngestionStatus.READY:
                job = self.job_manager.create_job(
                    job_id=j_id,
                    document_id=doc_id,
                    filename=filename,
                    scope=scope
                )
                self.job_manager.update_job(
                    job_id=j_id,
                    status=IngestionStatus.READY,
                    progress=100,
                    stage="complete",
                    page_count=existing_doc.page_count,
                    chunk_count=existing_doc.indexed_chunks_count
                )
                logger.info(f"Deduplicated upload for doc '{doc_id}' for user '{scope.user_id}'.")
                return job
        except Exception:
            pass

        # Create new job tracking record
        job = self.job_manager.create_job(
            job_id=j_id,
            document_id=doc_id,
            filename=filename,
            scope=scope
        )

        # Register preliminary document in PROCESSING status (not queryable)
        preliminary_doc = UserDocument(
            document_id=doc_id,
            user_id=scope.user_id,
            session_id=scope.session_id,
            filename=filename,
            file_hash=file_hash,
            file_size_bytes=len(file_bytes),
            page_count=0,
            status=IngestionStatus.PROCESSING,
            indexed_chunks_count=0
        )
        self.pipeline.repository.register_document(preliminary_doc, scope)

        # Submit task to background thread pool executor
        self._executor.submit(
            self._execute_ingestion_task,
            j_id,
            doc_id,
            file_bytes,
            filename,
            scope
        )
        return job

    def _execute_ingestion_task(
        self,
        job_id: str,
        document_id: str,
        file_bytes: bytes,
        filename: str,
        scope: UserDocumentSessionScope
    ) -> None:
        """Task worker routine executed asynchronously in background thread."""
        try:
            # Stage 1: Parsing
            self.job_manager.update_job(
                job_id=job_id,
                status=IngestionStatus.PROCESSING,
                progress=25,
                stage="parsing"
            )
            extracted_pages, has_ocr = self.pipeline.pdf_extractor.extract(file_bytes)
            page_count = len(extracted_pages)

            # Stage 2: Chunking
            self.job_manager.update_job(
                job_id=job_id,
                progress=50,
                stage="chunking",
                page_count=page_count
            )
            chunks = self.pipeline.chunker.chunk_document(
                pages=extracted_pages,
                document_id=document_id,
                user_id=scope.user_id,
                filename=filename,
                session_id=scope.session_id
            )
            if not chunks:
                raise CorruptPDFError(f"No text content could be extracted from '{filename}'.")

            # Stage 3: Embedding
            self.job_manager.update_job(
                job_id=job_id,
                progress=75,
                stage="embedding"
            )
            chunk_texts = [c.text for c in chunks]
            vectors = self.pipeline.embedding_model.embed_documents(chunk_texts)

            # Stage 4: Indexing
            self.job_manager.update_job(
                job_id=job_id,
                progress=90,
                stage="indexing"
            )
            indexed_count = self.pipeline.repository.upsert_user_chunks(
                chunks=chunks,
                vectors=vectors,
                scope=scope
            )

            # Stage 5: Completion & Ready Registration
            file_hash = self.pipeline.pdf_extractor.compute_sha256(file_bytes)
            ready_doc = UserDocument(
                document_id=document_id,
                user_id=scope.user_id,
                session_id=scope.session_id,
                filename=filename,
                file_hash=file_hash,
                file_size_bytes=len(file_bytes),
                page_count=page_count,
                status=IngestionStatus.READY,
                has_ocr_applied=has_ocr,
                indexed_chunks_count=indexed_count
            )
            self.pipeline.repository.register_document(ready_doc, scope)
            self.pipeline.document_retriever.bm25_manager.invalidate(scope)

            self.job_manager.update_job(
                job_id=job_id,
                status=IngestionStatus.READY,
                progress=100,
                stage="complete",
                chunk_count=indexed_count
            )
            logger.info(f"Background ingestion job '{job_id}' completed successfully for doc '{document_id}'.")

        except Exception as e:
            safe_error = str(e)
            logger.warning(f"Background ingestion job '{job_id}' failed for doc '{document_id}': {safe_error}")
            self.job_manager.update_job(
                job_id=job_id,
                status=IngestionStatus.FAILED,
                progress=0,
                stage="failed",
                error=safe_error
            )
            # Update repository document status to FAILED
            try:
                file_hash = self.pipeline.pdf_extractor.compute_sha256(file_bytes)
                failed_doc = UserDocument(
                    document_id=document_id,
                    user_id=scope.user_id,
                    session_id=scope.session_id,
                    filename=filename,
                    file_hash=file_hash,
                    file_size_bytes=len(file_bytes),
                    page_count=0,
                    status=IngestionStatus.FAILED,
                    error_message=safe_error,
                    indexed_chunks_count=0
                )
                self.pipeline.repository.register_document(failed_doc, scope)
            except Exception:
                pass


_async_worker_instance: Optional[AsyncIngestionWorker] = None


def get_async_worker() -> AsyncIngestionWorker:
    """Singleton provider for AsyncIngestionWorker."""
    global _async_worker_instance
    if _async_worker_instance is None:
        _async_worker_instance = AsyncIngestionWorker()
    return _async_worker_instance
