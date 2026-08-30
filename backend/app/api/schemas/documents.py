"""User Document Request and Response DTO schemas (Phase 8)."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class DocumentIngestResponseDTO(BaseModel):
    """Response returned upon successful synchronous PDF ingestion."""
    document_id: str
    filename: str
    status: str = "READY"
    page_count: int
    chunk_count: int
    file_size_bytes: int
    created_at: str
    message: str = "Document successfully ingested and indexed."


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
