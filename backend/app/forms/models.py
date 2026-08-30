"""Statutory Forms Models and Data Schemas for Nyaya Legal RAG (Phase 7).

Defines typed Pydantic models for statutory forms extracted from The Second
Schedule of the Bharatiya Nagarik Suraksha Sanhita, 2023 (BNSS Forms 1–58).
"""

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class FormFieldType(str, Enum):
    """Type of field, placeholder, or structural element within a statutory form."""
    TEXT_PLACEHOLDER = "text_placeholder"       # e.g., "................................ [Name of Accused]"
    DATE_PLACEHOLDER = "date_placeholder"       # e.g., "dated ............."
    TIME_PLACEHOLDER = "time_placeholder"       # e.g., "at ............... AM/PM"
    LOCATION_PLACEHOLDER = "location"           # e.g., "Police Station ..............."
    SECTION_REFERENCE = "section_reference"     # e.g., "u/s ..............."
    RECITALS = "recitals"                       # Statutory narrative body e.g. "WHEREAS ..."
    DIRECTIVE = "directive"                     # Mandatory order/instruction e.g. "Hence you are directed..."
    SIGNATURE_BLOCK = "signature_block"         # Seal/signature requirements


class FormField(BaseModel):
    """Atomic field, placeholder, or structural directive within a statutory form."""
    field_id: str                               # e.g., "form1_f1_noticee_name"
    label: str                                  # e.g., "Name of the Accused/Noticee"
    field_type: FormFieldType = FormFieldType.TEXT_PLACEHOLDER
    raw_text: str                               # Original statutory snippet
    placeholder: Optional[str] = None           # Dotted sequence e.g., "................................."
    default_value: Optional[str] = None
    is_required: bool = True
    context_prefix: Optional[str] = None        # e.g., "To," or "In pursuance of sub-section (3)..."


class FormSignature(BaseModel):
    """Signature, seal, or attestation metadata block."""
    signatory_title: str                        # e.g., "Signature and seal of the Magistrate", "Police Officer"
    seal_required: bool = True
    location_line: Optional[str] = None         # e.g., "Dated, this ........ day of ........"


class FormTableHead(BaseModel):
    """Structured charge head or tabular component (e.g. in Form 33 Charges)."""
    head_number: str                            # e.g., "I", "II", "(1)(a)"
    head_title: str                             # e.g., "CHARGES WITH ONE-HEAD", "On section 147"
    charge_text: str
    statutory_reference: Optional[str] = None


class StatutoryForm(BaseModel):
    """Canonical typed model representing a complete Statutory Form from the Second Schedule."""
    form_id: str                                # Canonical ID e.g., "BNSS_FORM_01"
    form_number: int                            # 1 .. 58
    form_title: str                             # e.g., "NOTICE FOR APPEARANCE BY THE POLICE"
    act: str = "Bharatiya Nagarik Suraksha Sanhita, 2023"
    act_short: str = "BNSS"
    schedule: str = "The Second Schedule"
    parent_section: str = "522"                 # Statutory provision empowering forms
    applicable_sections: List[str] = Field(default_factory=list)  # e.g., ["35(3)"] or ["234", "235", "236"]
    page_start: int                             # Source Gazette PDF Start Page e.g. 190
    page_end: int                               # Source Gazette PDF End Page e.g. 190 (or 224 for Form 33)
    raw_text: str                               # Verbatim extracted form text
    fields: List[FormField] = Field(default_factory=list)
    signatures: List[FormSignature] = Field(default_factory=list)
    tables: List[FormTableHead] = Field(default_factory=list)
    instructions: List[str] = Field(default_factory=list)
    provenance_citation: str                    # Canonical citation e.g., "[BNSS Second Schedule, Form 1]"


class FormLookupIntent(BaseModel):
    """Intent classification result for form queries."""
    is_form_query: bool
    target_form_number: Optional[int] = None
    target_form_title: Optional[str] = None
    target_section: Optional[str] = None
    normalized_query: str
    confidence: float = 1.0


class FormLookupResponse(BaseModel):
    """Structured API/Pipeline response for statutory form queries."""
    status: str                                 # "SUCCESS", "AMBIGUOUS", "NOT_FOUND"
    query: str
    form: Optional[StatutoryForm] = None
    candidate_forms: List[Dict[str, Any]] = Field(default_factory=list)
    provenance: Optional[str] = None
    rendered_markdown: Optional[str] = None
    is_refused: bool = False
    refusal_reason: Optional[str] = None
    latency_ms: float = 0.0
