"""User Document Request and Response DTO schemas (Part D)."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class DocumentUploadResponseDTO(BaseModel):
    """Response returned upon submitting an async document upload job."""
    job_id: str
    document_id: str
    filename: str
    status: str = "QUEUED"
    progress: int = 0
    stage: str = "queued"
    created_at: str
    message: str = "Document upload accepted for asynchronous processing."


class DocumentStatusDTO(BaseModel):
    """Status progress response for polling ingestion jobs."""
    job_id: str
    document_id: str
    status: str                         # "QUEUED", "PROCESSING", "READY", "FAILED"
    progress: int = 0                   # 0 to 100
    stage: str = "queued"               # "queued", "parsing", "chunking", "embedding", "indexing", "complete", "failed"
    error: Optional[str] = None
    page_count: Optional[int] = None
    chunk_count: Optional[int] = None
    updated_at: str


class DocumentIngestResponseDTO(BaseModel):
    """Response returned upon document ingestion."""
    job_id: str
    document_id: str
    filename: str
    status: str = "QUEUED"
    progress: int = 0
    stage: str = "queued"
    page_count: int = 0
    chunk_count: int = 0
    file_size_bytes: int = 0
    created_at: str
    message: str = "Document upload accepted for asynchronous processing."


class DocumentListItemDTO(BaseModel):
    """Item schema for document listing endpoint."""
    document_id: str
    filename: str
    file_size_bytes: int
    page_count: int
    chunk_count: int
    created_at: str
    status: str = "READY"


class DocumentDetailDTO(BaseModel):
    """Detailed metadata schema for single document retrieval."""
    document_id: str
    filename: str
    file_size_bytes: int
    page_count: int
    chunk_count: int
    created_at: str
    status: str = "READY"
    sha256_hash: str
