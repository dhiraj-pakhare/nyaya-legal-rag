"""Statutory Document Models and Metadata Schemas for Nyaya Legal RAG.

Defines Pydantic models for statutory hierarchy: Chapters, Sections, Subsections,
Clauses, Provisos, Exceptions, Explanations, Illustrations, Schedule Entries,
and the final StatutoryChunk conforming to the DhronAI technical assignment.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ActIdentity(str, Enum):
    BNSS = "Bharatiya Nagarik Suraksha Sanhita, 2023"
    BNS = "Bharatiya Nyaya Sanhita, 2023"


class ActShortName(str, Enum):
    BNSS = "BNSS"
    BNS = "BNS"


class ChunkType(str, Enum):
    SUBSTANTIVE_SECTION = "substantive_section"
    SCHEDULE_ENTRY = "schedule_entry"
    STATUTORY_FORM = "statutory_form"


class Proviso(BaseModel):
    """Statutory Proviso attached to a section or clause."""
    prefix: str = "Provided that"
    text: str
    page: int


class ExceptionModel(BaseModel):
    """Statutory Exception attached to a section."""
    prefix: str = "Exception"
    text: str
    page: int


class Explanation(BaseModel):
    """Statutory Explanation attached to a section."""
    prefix: str = "Explanation"
    text: str
    page: int


class Illustration(BaseModel):
    """Statutory Illustration attached to a section."""
    label: Optional[str] = None
    text: str
    page: int


class Clause(BaseModel):
    """Statutory Clause under a subsection or section e.g. (a), (b)."""
    clause_id: str  # e.g. "(a)", "(b)"
    text: str
    page_start: int
    page_end: int
    sub_clauses: List[str] = Field(default_factory=list)  # e.g. "(i)", "(ii)"
    provisos: List[Proviso] = Field(default_factory=list)
    explanations: List[Explanation] = Field(default_factory=list)


class Subsection(BaseModel):
    """Statutory Subsection under a section e.g. (1), (2)."""
    subsection_id: str  # e.g. "(1)", "(2)"
    text: str
    page_start: int
    page_end: int
    clauses: List[Clause] = Field(default_factory=list)
    provisos: List[Proviso] = Field(default_factory=list)
    exceptions: List[ExceptionModel] = Field(default_factory=list)
    explanations: List[Explanation] = Field(default_factory=list)
    illustrations: List[Illustration] = Field(default_factory=list)


class Section(BaseModel):
    """Atomic Statutory Section unit."""
    section_number: str  # e.g. "1", "35", "103"
    section_title: str  # Extracted dynamically from marginal note
    act: str = ActIdentity.BNSS.value
    act_short: str = ActShortName.BNSS.value
    chapter_number: str  # e.g. "V", "XXXIX"
    chapter_title: str
    page_start: int
    page_end: int
    raw_text: str
    subsections: List[Subsection] = Field(default_factory=list)
    provisos: List[Proviso] = Field(default_factory=list)
    exceptions: List[ExceptionModel] = Field(default_factory=list)
    explanations: List[Explanation] = Field(default_factory=list)
    illustrations: List[Illustration] = Field(default_factory=list)
    references: List[str] = Field(default_factory=list)


class Chapter(BaseModel):
    """Statutory Chapter grouping."""
    chapter_number: str  # Roman numeral e.g. "I", "V", "XXXIX"
    chapter_title: str
    page_start: int
    page_end: int
    sections: List[Section] = Field(default_factory=list)


class FirstScheduleEntry(BaseModel):
    """Row entry from The First Schedule: Classification of Offences under BNS."""
    section_number: str  # BNS section number e.g. "105", "64(2)"
    offence_name: str
    punishment: str
    cognizable_status: str  # Cognizable / Non-cognizable
    bailable_status: str  # Bailable / Non-bailable
    triable_court: str  # Court of Session / Any Magistrate
    page: int
    raw_text: str


class StatutoryChunk(BaseModel):
    """The canonical chunk schema required by the DhronAI specification."""
    act: str
    act_short: str
    chapter: Optional[str] = None
    chapter_title: Optional[str] = None
    section_number: str
    section_title: str
    subsection: Optional[str] = None
    clause: Optional[str] = None
    chunk_type: str = ChunkType.SUBSTANTIVE_SECTION.value
    text: str
    has_illustration: bool = False
    has_proviso: bool = False
    has_exception: bool = False
    has_explanation: bool = False
    page_start: int
    page_end: int
    chunk_id: str
    source_uri: str = "BNS bare act 2023.pdf"
    ingested_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    references: List[str] = Field(default_factory=list)
    
    # Optional schedule-specific fields
    offence_name: Optional[str] = None
    punishment: Optional[str] = None
    cognizable_status: Optional[str] = None
    bailable_status: Optional[str] = None
    triable_court: Optional[str] = None


class ValidationIssue(BaseModel):
    severity: str  # "ERROR", "WARNING", "INFO"
    code: str
    message: str
    location: str
    details: Optional[Dict[str, Any]] = None


class ValidationReport(BaseModel):
    total_pages: int = 0
    total_chapters: int = 0
    total_sections: int = 0
    total_chunks: int = 0
    total_schedule_entries: int = 0
    missing_sections: List[str] = Field(default_factory=list)
    duplicate_sections: List[str] = Field(default_factory=list)
    orphan_provisos: int = 0
    orphan_exceptions: int = 0
    orphan_explanations: int = 0
    orphan_illustrations: int = 0
    issues: List[ValidationIssue] = Field(default_factory=list)
    is_valid: bool = True
