"""Dual-Domain Citation Parser and AST Validator for Statutory and User-Document Citations."""

import logging
import re
from typing import Dict, List, Optional, Tuple
from pydantic import BaseModel, Field

from backend.app.document_rag.models import UserDocumentChunk
from backend.app.generation.citation_parser import CitationParser, ParsedCitation
from backend.app.generation.citation_validator import CitationValidator
from backend.app.generation.models import CitationVerification, ValidationStatus
from backend.app.ingestion.models import StatutoryChunk

logger = logging.getLogger("nyaya.document_rag.citation_validator")


class DualParsedCitation(BaseModel):
    """Structured representation of either a statutory citation or a user-document citation."""
    raw_tag: str
    citation_type: str = Field(..., description="'STATUTE' | 'USER_DOCUMENT'")
    canonical_tag: str

    # Statutory fields
    act_short: Optional[str] = None
    section_number: Optional[str] = None
    subsection: Optional[str] = None
    clause: Optional[str] = None

    # Document fields
    document_id: Optional[str] = None
    filename: Optional[str] = None
    page_number: Optional[int] = None
    page_end: Optional[int] = None


class DualCitationParser:
    """Extracts and normalizes statutory ([BNS s.103]) and user document ([DOC p.4]) citations."""

    # Regex for user document citations: [DOC p.4], [DOC p. 4], [DOC page 4], [DOC:notice.pdf p.4], [DOC p.4-5]
    DOC_CITATION_PATTERN = re.compile(
        r'\[\s*DOC(?::\s*([^\]\s]+))?\s+(?:p\.\s*|page\s*)(\d+)(?:\s*-\s*(\d+))?\s*\]',
        re.IGNORECASE
    )

    def __init__(self):
        self.statutory_parser = CitationParser()

    def parse_all(self, text: str) -> List[DualParsedCitation]:
        """Extract all statutory and user document citations from generation text."""
        citations: List[DualParsedCitation] = []

        # 1. Extract statutory citations using Phase 5 parser
        stat_parsed = self.statutory_parser.parse(text)
        for sp in stat_parsed:
            citations.append(
                DualParsedCitation(
                    raw_tag=sp.raw_text,
                    citation_type="STATUTE",
                    canonical_tag=sp.canonical_tag,
                    act_short=sp.act_short,
                    section_number=sp.section_number,
                    subsection=sp.subsection,
                    clause=sp.clause
                )
            )

        # 2. Extract document citations
        for match in self.DOC_CITATION_PATTERN.finditer(text):
            raw_tag = match.group(0)
            doc_label = match.group(1)
            page_start = int(match.group(2))
            page_end = int(match.group(3)) if match.group(3) else page_start

            canonical_tag = f"[DOC p.{page_start}]" if page_start == page_end else f"[DOC p.{page_start}-{page_end}]"
            citations.append(
                DualParsedCitation(
                    raw_tag=raw_tag,
                    citation_type="USER_DOCUMENT",
                    canonical_tag=canonical_tag,
                    filename=doc_label,
                    page_number=page_start,
                    page_end=page_end
                )
            )

        return citations


class DualCitationValidator:
    """Programmatic AST validator verifying both statutory provisions and user-document pages."""

    UNCITED_CLAIM_PATTERNS = [
        re.compile(r'\b(?:punish(?:ed|able|ment)?|imprison(?:ment)?|fine|death sentence)\b', re.IGNORECASE),
        re.compile(r'\b(?:cogniz(?:able|ance)|non-cognizable|bailable|non-bailable)\b', re.IGNORECASE),
        re.compile(r'\b(?:arrest(?:ed)?|custody|warrant|summons|magistrate|police officer)\b', re.IGNORECASE),
        re.compile(r'\b(?:alleg(?:ed|ation|es)?|complainant|accused|stated in the notice|according to the fir)\b', re.IGNORECASE)
    ]

    def __init__(self, statutory_validator: Optional[CitationValidator] = None):
        self.statutory_validator = statutory_validator or CitationValidator()
        self.parser = DualCitationParser()

    def validate(
        self,
        answer_text: str,
        statutory_evidence: List[StatutoryChunk],
        document_evidence: List[UserDocumentChunk]
    ) -> ValidationStatus:
        """Validate all citations in answer text against both statutory and document evidence pools."""
        if not answer_text or not answer_text.strip():
            return ValidationStatus(
                is_valid=False,
                failure_reasons=["Empty generation text."]
            )

        # Check for direct refusal phrasing
        refusal_phrases = [
            "i cannot answer", "does not contain sufficient",
            "no information", "cannot be determined", "not mentioned in the provided"
        ]
        if any(rp in answer_text.lower() for rp in refusal_phrases):
            return ValidationStatus(
                is_valid=True,
                checked_citations_count=0,
                valid_citations_count=0,
                invalid_citations_count=0,
                verified_citations=[],
                invalid_citations=[],
                uncited_claims_detected=[],
                failure_reasons=[]
            )

        parsed_citations = self.parser.parse_all(answer_text)
        if not parsed_citations:
            # Detect uncited claims
            uncited_claims = self._detect_uncited_claims(answer_text)
            reasons = ["Answer contains 0 citations. All legal statements and document facts must cite retrieved evidence."]
            if uncited_claims:
                reasons.append(f"Detected {len(uncited_claims)} substantive claim(s) without required citations.")
            return ValidationStatus(
                is_valid=False,
                checked_citations_count=0,
                valid_citations_count=0,
                invalid_citations_count=0,
                verified_citations=[],
                invalid_citations=[],
                uncited_claims_detected=uncited_claims,
                failure_reasons=reasons
            )

        verified_citations: List[CitationVerification] = []
        invalid_citations: List[Dict[str, str]] = []
        failure_reasons: List[str] = []

        # Validate statutory citations
        statutory_citations = [c for c in parsed_citations if c.citation_type == "STATUTE"]
        if statutory_citations:
            stat_status = self.statutory_validator.validate(
                answer=answer_text,
                retrieved_documents=statutory_evidence
            )
            # Find matching statutory chunks for verified citations
            for cit in stat_status.checked_citations if hasattr(stat_status, "checked_citations") else []:
                pass
            if hasattr(stat_status, "verified_citations"):
                verified_citations.extend(stat_status.verified_citations)
            else:
                # Build verified citations from valid statutory references
                for sc in statutory_citations:
                    for doc in statutory_evidence:
                        doc_act = doc.act_short.upper()
                        doc_sec = str(doc.section_number).strip()
                        if doc_act == sc.act_short and doc_sec.startswith(sc.section_number):
                            verified_citations.append(
                                CitationVerification(
                                    citation_text=sc.canonical_tag,
                                    act=doc.act,
                                    act_short=doc.act_short,
                                    section=doc.section_number,
                                    subsection=sc.subsection or getattr(doc, "subsection", None),
                                    clause=sc.clause or getattr(doc, "clause", None),
                                    section_title=doc.section_title,
                                    page_start=doc.page_start,
                                    page_end=doc.page_end,
                                    chunk_id=doc.chunk_id,
                                    source_text=doc.text,
                                    is_verified=True
                                )
                            )
                            break
            invalid_citations.extend(stat_status.invalid_citations)
            stat_failure_reasons = [
                r for r in stat_status.failure_reasons
                if not r.startswith("Detected ") and not r.startswith("Answer contains 0 statutory citations")
            ]
            failure_reasons.extend(stat_failure_reasons)

        # Validate user document citations
        doc_citations = [c for c in parsed_citations if c.citation_type == "USER_DOCUMENT"]
        for dc in doc_citations:
            match_chunk = None
            for chunk in document_evidence:
                if chunk.page_start <= dc.page_number <= chunk.page_end:
                    match_chunk = chunk
                    break

            if match_chunk:
                verification = CitationVerification(
                    citation_text=dc.canonical_tag,
                    act=f"DOC:{match_chunk.filename}",
                    act_short="DOC",
                    section=f"p.{dc.page_number}",
                    section_title=f"Page {dc.page_number} ({match_chunk.filename})",
                    page_start=match_chunk.page_start,
                    page_end=match_chunk.page_end,
                    chunk_id=match_chunk.chunk_id,
                    source_text=match_chunk.text,
                    is_verified=True
                )
                verified_citations.append(verification)
            else:
                reason = f"Cited user-document page {dc.canonical_tag} does not exist in retrieved document evidence."
                invalid_citations.append({"citation": dc.raw_tag, "reason": reason})
                failure_reasons.append(reason)

        uncited_claims = self._detect_uncited_claims(answer_text)
        if uncited_claims:
            failure_reasons.append(f"Detected {len(uncited_claims)} substantive claim(s) without required citations.")
        is_valid = len(invalid_citations) == 0 and len(failure_reasons) == 0

        return ValidationStatus(
            is_valid=is_valid,
            checked_citations_count=len(parsed_citations),
            valid_citations_count=len(verified_citations),
            invalid_citations_count=len(invalid_citations),
            verified_citations=verified_citations,
            invalid_citations=invalid_citations,
            uncited_claims_detected=uncited_claims,
            failure_reasons=failure_reasons
        )

    def _detect_uncited_claims(self, text: str) -> List[str]:
        """Identify sentences making factual or legal assertions without an inline citation tag."""
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
        uncited: List[str] = []

        for sentence in sentences:
            has_citation = bool(re.search(r'\[(?:BNS|BNSS|DOC)(?::|\s)[^\]]+\]', sentence, re.IGNORECASE))
            if not has_citation:
                has_legal_keyword = any(pat.search(sentence) for pat in self.UNCITED_CLAIM_PATTERNS)
                if has_legal_keyword and len(sentence) > 30:
                    uncited.append(sentence)

        return uncited
