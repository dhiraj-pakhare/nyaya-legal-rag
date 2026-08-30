"""Background Ingestion Job Store and Thread-Safe State Machine."""

from datetime import datetime, timezone
import logging
import threading
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field

from backend.app.document_rag.models import IngestionStatus, UserDocumentSessionScope

logger = logging.getLogger("nyaya.workers.job_manager")


class IngestionJob(BaseModel):
    """Data model representing an asynchronous document ingestion job."""
    job_id: str
    document_id: str
    user_id: str
    session_id: Optional[str] = None
    filename: str
    status: IngestionStatus = IngestionStatus.PENDING
    progress: int = 0
    stage: str = "queued"
    error: Optional[str] = None
    page_count: Optional[int] = None
    chunk_count: Optional[int] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def update(
        self,
        status: Optional[IngestionStatus] = None,
        progress: Optional[int] = None,
        stage: Optional[str] = None,
        error: Optional[str] = None,
        page_count: Optional[int] = None,
        chunk_count: Optional[int] = None
    ) -> None:
        if status is not None:
            self.status = status
        if progress is not None:
            self.progress = progress
        if stage is not None:
            self.stage = stage
        if error is not None:
            self.error = error
        if page_count is not None:
            self.page_count = page_count
        if chunk_count is not None:
            self.chunk_count = chunk_count
        self.updated_at = datetime.now(timezone.utc)


class IngestionJobManager:
    """Thread-safe in-memory manager for background ingestion job tracking."""

    def __init__(self):
        self._jobs: Dict[str, IngestionJob] = {}
        self._doc_to_job: Dict[str, str] = {}
        self._lock = threading.Lock()

    def create_job(
        self,
        job_id: str,
        document_id: str,
        filename: str,
        scope: UserDocumentSessionScope
    ) -> IngestionJob:
        """Create a new job tracking record under security scope."""
        with self._lock:
            job = IngestionJob(
                job_id=job_id,
                document_id=document_id,
                user_id=scope.user_id,
                session_id=scope.session_id,
                filename=filename,
                status=IngestionStatus.PENDING,
                progress=0,
                stage="queued"
            )
            self._jobs[job_id] = job
            self._doc_to_job[document_id] = job_id
            return job

    def get_job(self, job_id: str, scope: UserDocumentSessionScope) -> Optional[IngestionJob]:
        """Retrieve job by job_id. Enforces tenant scope matching."""
        scope.validate_scope()
        with self._lock:
            job = self._jobs.get(job_id)
            if not job or job.user_id != scope.user_id:
                return None
            return job

    def get_job_by_document(self, document_id: str, scope: UserDocumentSessionScope) -> Optional[IngestionJob]:
        """Retrieve job associated with a document_id. Enforces tenant scope matching."""
        scope.validate_scope()
        with self._lock:
            job_id = self._doc_to_job.get(document_id)
            if not job_id:
                return None
            job = self._jobs.get(job_id)
            if not job or job.user_id != scope.user_id:
                return None
            return job

    def update_job(
        self,
        job_id: str,
        status: Optional[IngestionStatus] = None,
        progress: Optional[int] = None,
        stage: Optional[str] = None,
        error: Optional[str] = None,
        page_count: Optional[int] = None,
        chunk_count: Optional[int] = None
    ) -> Optional[IngestionJob]:
        """Update progress and stage of an active job."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                job.update(
                    status=status,
                    progress=progress,
                    stage=stage,
                    error=error,
                    page_count=page_count,
                    chunk_count=chunk_count
                )
            return job


_job_manager_instance: Optional[IngestionJobManager] = None


def get_job_manager() -> IngestionJobManager:
    """Singleton provider for IngestionJobManager."""
    global _job_manager_instance
    if _job_manager_instance is None:
        _job_manager_instance = IngestionJobManager()
    return _job_manager_instance
