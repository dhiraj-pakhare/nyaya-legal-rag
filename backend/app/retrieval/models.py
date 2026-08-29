"""Data contracts and schemas for the retrieval subsystem."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class RetrievalFilter(BaseModel):
    """Filters that can be applied to statutory retrieval."""
    act: Optional[str] = None
    act_short: Optional[str] = None
    chapter: Optional[str] = None
    section_number: Optional[str] = None
    chunk_type: Optional[str] = None


class RetrievedDocument(BaseModel):
    """Stable, typed contract for a single retrieved statutory document."""
    chunk_id: str
    act: str
    act_short: str
    chapter: str
    chapter_title: str
    section_number: str
    section_title: str
    subsection: Optional[str] = None
    clause: Optional[str] = None
    text: str
    page_start: int
    page_end: int
    chunk_type: str = "substantive_section"
    
    # Ranking & Scoring telemetry
    score: float = Field(..., description="Primary ranking score (RRF, dense cosine, or BM25)")
    final_rank: int
    dense_rank: Optional[int] = None
    dense_score: Optional[float] = None
    bm25_rank: Optional[int] = None
    bm25_score: Optional[float] = None
    rrf_score: Optional[float] = None
    is_exact_match: bool = False
    
    # Statutory references & extra metadata
    references: List[Any] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RetrievalResult(BaseModel):
    """Complete response contract from the retrieval pipeline."""
    query: str
    mode: str = Field(..., description="'exact_lookup' | 'hybrid_rrf' | 'dense_only' | 'bm25_only'")
    documents: List[RetrievedDocument] = Field(default_factory=list)
    total_retrieved: int = 0
    latency_ms: float = 0.0
    applied_filters: Optional[Dict[str, Any]] = None
    is_empty: bool = False
    detected_intent: Optional[Dict[str, Any]] = None
    
    # Confidence and Refusal contract
    confidence: Optional[Dict[str, Any]] = None
    is_refused: bool = False
    refusal_reason: Optional[str] = None
