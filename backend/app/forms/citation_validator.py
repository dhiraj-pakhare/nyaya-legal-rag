"""Statutory Form Citation Parser and AST Validator (Phase 7).

Parses and validates canonical statutory form citations:
    [BNSS Second Schedule, Form X]
    [BNSS Form X]

Verifies that cited form numbers exist in The Second Schedule (1..58)
and are present in the retrieved statutory form evidence context.
"""

import logging
import re
from typing import List, Optional, Set
from pydantic import BaseModel, Field

from backend.app.forms.models import StatutoryForm
from backend.app.forms.repository import StatutoryFormRegistry, get_form_registry
from backend.app.generation.models import CitationVerification, ValidationStatus

logger = logging.getLogger("nyaya.forms.citation_validator")

# Regex to match canonical statutory form citations
FORM_CITATION_RE = re.compile(
    r'\[\s*(?:BNSS\s+)?(?:Second\s+Schedule\s*,\s*)?Form\s+(?:No\.?\s*)?(\d+)\s*\]',
    re.IGNORECASE
)


class ParsedFormCitation(BaseModel):
    """Raw parsed statutory form citation."""
    raw_text: str                               # e.g. "[BNSS Second Schedule, Form 1]"
    form_number: int                            # e.g. 1
    start_pos: int
    end_pos: int


class FormCitationParser:
    """Extracts statutory form citations from generated text."""

    @staticmethod
    def parse_citations(text: str) -> List[ParsedFormCitation]:
        """Extract all statutory form citation tags from text."""
        citations: List[ParsedFormCitation] = []
        for m in FORM_CITATION_RE.finditer(text):
            f_num = int(m.group(1))
            citations.append(
                ParsedFormCitation(
                    raw_text=m.group(0),
                    form_number=f_num,
                    start_pos=m.start(),
                    end_pos=m.end()
                )
            )
        return citations


class FormCitationValidator:
    """Validates extracted form citations against the StatutoryFormRegistry and retrieved context."""

    def __init__(self, registry: Optional[StatutoryFormRegistry] = None):
        self.registry = registry or get_form_registry()

    def validate(
        self,
        answer_text: str,
        retrieved_forms: List[StatutoryForm]
    ) -> ValidationStatus:
        """Validate all statutory form citations in the answer text."""
        citations = FormCitationParser.parse_citations(answer_text)
        
        if not citations:
            return ValidationStatus(
                is_valid=True,
                checked_citations_count=0,
                valid_citations_count=0,
                invalid_citations_count=0,
                verified_citations=[],
                invalid_citations=[],
                failure_reasons=[]
            )

        retrieved_form_numbers: Set[int] = {f.form_number for f in retrieved_forms}
        verified_citations: List[CitationVerification] = []
        invalid_citations: List[CitationVerification] = []
        failure_reasons: List[str] = []

        for cit in citations:
            f_num = cit.form_number
            canonical_tag = f"[BNSS Second Schedule, Form {f_num}]"

            # 1. Existence check in authoritative registry (1..58)
            registry_form = self.registry.get_by_number(f_num)
            if not registry_form:
                reason = f"Statutory Form {f_num} does not exist in The Second Schedule of BNSS (valid range: Forms 1–58)."
                cv = CitationVerification(
                    citation_text=cit.raw_text,
                    act="Bharatiya Nagarik Suraksha Sanhita, 2023",
                    act_short="BNSS",
                    section=f"Form {f_num}",
                    section_title=f"Form {f_num}",
                    page_start=0,
                    page_end=0,
                    chunk_id=f"BNSS_FORM_{f_num}",
                    source_text="",
                    is_verified=False,
                    failure_reason=reason
                )
                invalid_citations.append(cv)
                failure_reasons.append(reason)
                continue

            # 2. Context Retrieval Grounding Check
            if f_num not in retrieved_form_numbers:
                reason = f"Form {f_num} is a valid statutory form but was NOT present in retrieved evidence context."
                cv = CitationVerification(
                    citation_text=cit.raw_text,
                    act="Bharatiya Nagarik Suraksha Sanhita, 2023",
                    act_short="BNSS",
                    section=f"Form {f_num}",
                    section_title=registry_form.form_title,
                    page_start=registry_form.page_start,
                    page_end=registry_form.page_end,
                    chunk_id=registry_form.form_id,
                    source_text=registry_form.raw_text[:200],
                    is_verified=False,
                    failure_reason=reason
                )
                invalid_citations.append(cv)
                failure_reasons.append(reason)
                continue

            # Citation is completely valid and grounded
            cv = CitationVerification(
                citation_text=canonical_tag,
                act="Bharatiya Nagarik Suraksha Sanhita, 2023",
                act_short="BNSS",
                section=f"Form {f_num}",
                section_title=registry_form.form_title,
                page_start=registry_form.page_start,
                page_end=registry_form.page_end,
                chunk_id=registry_form.form_id,
                source_text=registry_form.raw_text[:200],
                is_verified=True
            )
            verified_citations.append(cv)

        is_valid = len(invalid_citations) == 0
        return ValidationStatus(
            is_valid=is_valid,
            checked_citations_count=len(citations),
            valid_citations_count=len(verified_citations),
            invalid_citations_count=len(invalid_citations),
            verified_citations=verified_citations,
            invalid_citations=[c.model_dump() for c in invalid_citations],
            failure_reasons=failure_reasons
        )
