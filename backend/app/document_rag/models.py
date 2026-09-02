"""Data models, scopes, and schemas for User-Document RAG."""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class IngestionStatus(str, Enum):
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    READY = "READY"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class QueryIntent(str, Enum):
    STATUTORY_ONLY = "STATUTORY_ONLY"
    DOCUMENT_ONLY = "DOCUMENT_ONLY"
    COMBINED = "COMBINED"


class SecurityScopeError(Exception):
    """Raised when an operation violates multi-tenant security boundary."""
    pass


class DocumentNotFoundError(Exception):
    """Raised when a document is not found or inaccessible (uniform 404 anti-enumeration protocol)."""
    pass


class CorruptPDFError(Exception):
    """Raised when an uploaded PDF cannot be read or parsed."""
    pass


class OversizedDocumentError(Exception):
    """Raised when an uploaded document exceeds size limit."""
    pass


class OCRUnavailableError(Exception):
    """Raised when scanned document requires OCR but OCR engine is unavailable."""
    pass


class IngestionCancelledException(Exception):
    """Raised when background document ingestion is cancelled by the caller."""
    pass


class UserDocument(BaseModel):
    """Canonical metadata for an uploaded user document."""
    document_id: str
    user_id: str
    session_id: Optional[str] = None
    filename: str
    file_hash: str
    file_size_bytes: int
    page_count: int
    content_type: str = "application/pdf"
    status: IngestionStatus = IngestionStatus.PENDING
    has_ocr_applied: bool = False
    error_message: Optional[str] = None
    uploaded_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    indexed_chunks_count: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class UserDocumentChunk(BaseModel):
    """Atomic text chunk extracted from a user document."""
    chunk_id: str
    document_id: str
    user_id: str
    session_id: Optional[str] = None
    filename: str
    page_start: int
    page_end: int
    chunk_index: int
    text: str
    token_count: int
    score: float = 0.0
    final_rank: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class UserDocumentSessionScope(BaseModel):
    """Immutable security scope bounding all user-document retrieval and storage operations."""
    user_id: str
    session_id: Optional[str] = None
    active_document_ids: List[str] = Field(default_factory=list)

    def validate_scope(self) -> None:
        """Validate that user_id is non-empty."""
        if not self.user_id or not self.user_id.strip():
            raise SecurityScopeError("Security scope violation: trusted user_id cannot be empty")


class RoutingDecision(BaseModel):
    """Classification verdict from query intent router."""
    intent: QueryIntent
    confidence: float
    detected_statutory_sections: List[str] = Field(default_factory=list)
    target_document_ids: List[str] = Field(default_factory=list)
    reason: str


class DocumentIngestionResult(BaseModel):
    """Result returned upon completing PDF ingestion."""
    document: UserDocument
    chunks_count: int
    is_deduplicated: bool = False
    latency_ms: float = 0.0
