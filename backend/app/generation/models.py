"""Data contracts and schemas for LLM Generation and Citation Validation."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from backend.app.retrieval.models import RetrievedDocument


class LLMMessage(BaseModel):
    """Single message in a chat completion prompt."""
    role: str = Field(..., description="'system' | 'user' | 'assistant'")
    content: str


class LLMResponse(BaseModel):
    """Raw response contract from an LLM provider."""
    content: str
    model: str
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    latency_ms: float = 0.0
    raw_response: Optional[Dict[str, Any]] = None


class ParsedCitation(BaseModel):
    """AST structure extracted from generated text."""
    raw_text: str
    act_short: str  # "BNS" or "BNSS"
    section_number: str  # e.g. "103", "35"
    subsection: Optional[str] = None  # e.g. "(1)", "(2)"
    clause: Optional[str] = None  # e.g. "(a)", "(b)"
    canonical_tag: str  # e.g. "[BNS s.103(1)]"
    start_char: int
    end_char: int


class CitationVerification(BaseModel):
    """Verified citation item enriched with source chunk metadata for source drawer display."""
    citation_text: str
    act: str
    act_short: str
    section: str
    subsection: Optional[str] = None
    clause: Optional[str] = None
    section_title: str
    page_start: int
    page_end: int
    chunk_id: str
    source_text: str
    is_verified: bool = True
    failure_reason: Optional[str] = None


class ValidationStatus(BaseModel):
    """Detailed result of programmatic citation and claim validation."""
    is_valid: bool
    checked_citations_count: int = 0
    valid_citations_count: int = 0
    invalid_citations_count: int = 0
    invalid_citations: List[Dict[str, Any]] = Field(default_factory=list)
    uncited_claims_detected: List[str] = Field(default_factory=list)
    regeneration_attempted: bool = False
    failure_reasons: List[str] = Field(default_factory=list)
    error_details: Optional[str] = None


class GenerationTelemetry(BaseModel):
    """Complete performance, latency, and token telemetry for a generation request."""
    retrieval_latency_ms: float = 0.0
    rerank_latency_ms: float = 0.0
    generation_latency_ms: float = 0.0
    validation_latency_ms: float = 0.0
    total_latency_ms: float = 0.0
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    estimated_cost_usd: Optional[float] = None
    model: str = ""
    provider: str = ""


class LegalAnswerResponse(BaseModel):
    """Complete top-level answer contract returned to clients and callers."""
    query: str
    answer: Optional[str] = None
    status: str = Field(..., description="'SUCCESS' | 'REFUSED' | 'VALIDATION_FAILED' | 'ERROR'")
    is_refused: bool = False
    refusal_reason: Optional[str] = None
    confidence: Optional[Dict[str, Any]] = None
    citations: List[CitationVerification] = Field(default_factory=list)
    sources: List[RetrievedDocument] = Field(default_factory=list)
    retrieval_metadata: Dict[str, Any] = Field(default_factory=dict)
    validation_status: Optional[ValidationStatus] = None
    telemetry: Optional[GenerationTelemetry] = None
