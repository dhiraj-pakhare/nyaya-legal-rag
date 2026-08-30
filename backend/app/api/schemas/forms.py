"""Statutory Form Request and Response DTO schemas (Part B)."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class FormFieldDTO(BaseModel):
    """Field/placeholder schema within a statutory form."""
    field_id: str
    label: str
    field_type: str
    placeholder: Optional[str] = None
    is_required: bool = True


class FormSignatureDTO(BaseModel):
    """Signature/seal block schema."""
    signatory_title: str
    seal_required: bool = True


class FormTableHeadDTO(BaseModel):
    """Charge head table component schema."""
    head_number: str
    head_title: str
    charge_text: str


class StatutoryFormDTO(BaseModel):
    """Canonical statutory form schema."""
    form_id: str
    form_number: int
    form_title: str
    act: str = "Bharatiya Nagarik Suraksha Sanhita, 2023"
    act_short: str = "BNSS"
    schedule: str = "The Second Schedule"
    applicable_sections: List[str] = Field(default_factory=list)
    page_start: int
    page_end: int
    raw_text: str
    fields: List[FormFieldDTO] = Field(default_factory=list)
    signatures: List[FormSignatureDTO] = Field(default_factory=list)
    tables: List[FormTableHeadDTO] = Field(default_factory=list)
    provenance_citation: str


class StatutoryFormListItemDTO(BaseModel):
    """Summary item for statutory forms library listing."""
    form_number: int
    form_id: str
    title: str
    slug: str
    filename: str
    applicable_sections: List[str] = Field(default_factory=list)
    page_start: int
    page_end: int
    page_count: int
    byte_size: Optional[int] = None
    sha256: Optional[str] = None
    extraction_confidence: float = 1.0
    needs_review: bool = False
    download_url: str
    provenance: str


class StatutoryFormListResponseDTO(BaseModel):
    """Response envelope for forms library listing."""
    total_forms: int = 58
    schedule: str = "The Second Schedule"
    forms: List[StatutoryFormListItemDTO] = Field(default_factory=list)


class FormLookupRequestDTO(BaseModel):
    """Request schema for deterministic form lookup."""
    query: str = Field(..., min_length=1, max_length=500, description="Form number, section, or title query")


class FormLookupResponseDTO(BaseModel):
    """Response schema for form lookup."""
    status: str                         # "SUCCESS", "AMBIGUOUS", "NOT_FOUND"
    query: str
    form: Optional[StatutoryFormDTO] = None
    candidate_forms: List[Dict[str, Any]] = Field(default_factory=list)
    provenance: Optional[str] = None
    rendered_markdown: Optional[str] = None
    is_refused: bool = False
    refusal_reason: Optional[str] = None
    latency_ms: float = 0.0
