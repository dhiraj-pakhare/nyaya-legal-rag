"""Query Request and Response DTO schemas with polymorphic citations (Phase 8)."""

from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field


class CitationType(str, Enum):
    STATUTORY = "STATUTORY"          # [BNS s.103], [BNSS s.35]
    DOCUMENT = "DOCUMENT"            # [DOC p.4]
    FORM = "FORM"                    # [BNSS Second Schedule, Form 1]


class BaseCitationDTO(BaseModel):
    """Common baseline metadata shared across all citation classes."""
    citation_text: str               # e.g. "[BNS s.103]", "[DOC p.4]", "[BNSS Second Schedule, Form 1]"
    citation_type: CitationType
    is_verified: bool = True
    source_id: str                   # chunk_id, document_id, or form_id
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    source_text: Optional[str] = None


class StatutoryCitationDTO(BaseCitationDTO):
    """Citation metadata for penal/procedural statutory sections."""
    citation_type: Literal[CitationType.STATUTORY] = CitationType.STATUTORY
    act: str = "Bharatiya Nyaya Sanhita, 2023"
    act_short: str = "BNS"
    section: str
    section_title: str


class DocumentCitationDTO(BaseCitationDTO):
    """Citation metadata for user-uploaded document pages."""
    citation_type: Literal[CitationType.DOCUMENT] = CitationType.DOCUMENT
    document_id: str
    filename: str
    page_number: int


class FormCitationDTO(BaseCitationDTO):
    """Citation metadata for Second Schedule statutory forms."""
    citation_type: Literal[CitationType.FORM] = CitationType.FORM
    form_number: int                 # 1 .. 58
    form_title: str
    applicable_sections: List[str] = Field(default_factory=list)


# Discriminated Union for polymorphic citation serialization
CitationDTO = Union[StatutoryCitationDTO, DocumentCitationDTO, FormCitationDTO]


class QueryRequestDTO(BaseModel):
    """Unified client query request schema."""
    query: str = Field(..., min_length=2, max_length=4000, description="Legal question or query string")
    document_ids: Optional[List[str]] = Field(default=None, description="Optional active document IDs for scoping")
    enable_forms: bool = Field(default=True, description="Allow automatic routing to statutory forms engine")


class QueryResponseDTO(BaseModel):
    """Unified client query response schema."""
    query: str
    status: str                         # "SUCCESS", "REFUSED", "VALIDATION_FAILED", "AMBIGUOUS", "NOT_FOUND"
    answer: Optional[str] = None
    is_refused: bool = False
    refusal_reason: Optional[str] = None
    citations: List[CitationDTO] = Field(default_factory=list)
    confidence_score: float = 1.0
    routed_corpus: str                  # "STATUTORY", "USER_DOCUMENT", "COMBINED", "STATUTORY_FORM"
    candidate_forms: Optional[List[Dict[str, Any]]] = None
    telemetry: Optional[Dict[str, Any]] = None


class StreamEventDTO(BaseModel):
    """Server-Sent Event payload representation."""
    event: str                          # "status", "token", "citation", "complete", "refusal"
    data: Dict[str, Any]
