"""Asynchronous Background Worker Engine for User Document Ingestion."""

from concurrent.futures import ThreadPoolExecutor
import logging
import uuid
from typing import Optional

from backend.app.core.qdrant_repo import NYAYA_NAMESPACE
from backend.app.document_rag.models import (
    CorruptPDFError,
    IngestionCancelledException,
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

    def _handle_cancellation(
        self,
        job_id: str,
        document_id: str,
        scope: UserDocumentSessionScope
    ) -> None:
        """Clean up partial Qdrant vectors and mark state as CANCELLED."""
        logger.info(f"Ingestion worker halting task '{job_id}' (doc='{document_id}') due to cancellation.")
        try:
            self.pipeline.repository.purge_document_vectors(document_id, scope)
        except Exception as e:
            logger.warning(f"Error purging vectors on cancellation for doc '{document_id}': {e}")

        self.job_manager.update_job(
            job_id=job_id,
            status=IngestionStatus.CANCELLED,
            stage="cancelled",
            error="Job was cancelled by user."
        )
        try:
            self.pipeline.document_retriever.bm25_manager.invalidate(scope)
        except Exception:
            pass

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
            if self.job_manager.is_cancelled(job_id, scope):
                self._handle_cancellation(job_id, document_id, scope)
                return

            # Stage 1: Parsing
            self.job_manager.update_job(
                job_id=job_id,
                status=IngestionStatus.PROCESSING,
                progress=25,
                stage="parsing"
            )
            extracted_pages, has_ocr = self.pipeline.pdf_extractor.extract(file_bytes)
            page_count = len(extracted_pages)

            if self.job_manager.is_cancelled(job_id, scope):
                self._handle_cancellation(job_id, document_id, scope)
                return

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

            if self.job_manager.is_cancelled(job_id, scope):
                self._handle_cancellation(job_id, document_id, scope)
                return

            # Stage 3: Embedding (batch by batch checking cancellation)
            self.job_manager.update_job(
                job_id=job_id,
                progress=75,
                stage="embedding"
            )
            chunk_texts = [c.text for c in chunks]
            batch_size = 16
            all_vectors = []
            for i in range(0, len(chunk_texts), batch_size):
                if self.job_manager.is_cancelled(job_id, scope):
                    self._handle_cancellation(job_id, document_id, scope)
                    return
                batch_texts = chunk_texts[i:i + batch_size]
                batch_vecs = self.pipeline.embedding_model.embed_documents(batch_texts)
                all_vectors.append(batch_vecs)

            if not all_vectors:
                vectors = []
            elif isinstance(all_vectors[0], list):
                vectors = [v for b in all_vectors for v in b]
            else:
                import numpy as np
                vectors = np.vstack(all_vectors)

            if self.job_manager.is_cancelled(job_id, scope):
                self._handle_cancellation(job_id, document_id, scope)
                return

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

            if self.job_manager.is_cancelled(job_id, scope):
                self._handle_cancellation(job_id, document_id, scope)
                return

            # Stage 5: Completion & Atomic Ready Finalization
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

            # Atomic finalization under IngestionJobManager lock
            finalized = self.job_manager.finalize_ready(
                job_id=job_id,
                scope=scope,
                ready_doc=ready_doc,
                register_callback=self.pipeline.repository.register_document
            )

            if not finalized:
                self._handle_cancellation(job_id, document_id, scope)
                return

            self.pipeline.document_retriever.bm25_manager.invalidate(scope)
            logger.info(f"Background ingestion job '{job_id}' completed successfully for doc '{document_id}'.")

        except IngestionCancelledException:
            self._handle_cancellation(job_id, document_id, scope)
        except Exception as e:
            if self.job_manager.is_cancelled(job_id, scope):
                self._handle_cancellation(job_id, document_id, scope)
                return
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
