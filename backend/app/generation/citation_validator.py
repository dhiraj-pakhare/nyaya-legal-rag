"""Programmatic AST Citation and Legal Claim Validator for Nyaya Legal RAG."""

import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from backend.app.generation.citation_parser import CitationParser
from backend.app.generation.models import (
    CitationVerification,
    ParsedCitation,
    ValidationStatus
)
from backend.app.retrieval.models import RetrievedDocument

logger = logging.getLogger("nyaya.generation.citation_validator")


class CitationValidator:
    """Deterministic, programmatic safety layer validating all citations and claims against retrieved evidence."""

    # Keywords indicating a sentence makes a substantive statutory or penal assertion
    LEGAL_CLAIM_KEYWORDS = [
        "punish", "imprison", "fine", "cogniz", "bailable", "non-bailable",
        "warrant", "arrest", "magistrate", "custody", "police officer",
        "court of session", "death penalty", "offence", "offenses", "triable",
        "prescribed", "schedule", "proviso", "statute", "culpable homicide",
        "murder", "assault", "theft", "seizure", "bail", "investigation"
    ]

    # Conversational or meta sentences that do not make substantive legal claims
    META_SENTENCE_PATTERNS = [
        r"^(based on|according to|as per|in summary|to summarize|the provided|referring to)",
        r"^(insufficient statutory evidence|the statutory evidence does not|no provision)",
        r"^(here is the|the following are|in conclusion)"
    ]

    def __init__(self, parser: Optional[CitationParser] = None):
        self.parser = parser or CitationParser()

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences while respecting abbreviations like s. or sec."""
        # Protect s. and sec. from sentence split
        protected = re.sub(r'\b(s|sec|no|mr|mrs|dr)\.\s*', r'\1<DOT> ', text, flags=re.IGNORECASE)
        # Split on sentence boundaries
        raw_sentences = re.split(r'(?<=[.!?])\s+', protected)
        sentences = [s.replace('<DOT>', '.').strip() for s in raw_sentences if s.strip()]
        return sentences

    def _is_meta_sentence(self, sentence: str) -> bool:
        """Check if a sentence is purely introductory, conclusion, or refusal notice."""
        s_lower = sentence.lower().strip()
        for pat in self.META_SENTENCE_PATTERNS:
            if re.search(pat, s_lower):
                return True
        return False

    def _contains_legal_claim(self, sentence: str) -> bool:
        """Check if sentence asserts substantive legal rules, procedures, or penalties."""
        s_lower = sentence.lower()
        return any(kw in s_lower for kw in self.LEGAL_CLAIM_KEYWORDS)

    def validate(
        self,
        answer: str,
        retrieved_documents: List[RetrievedDocument]
    ) -> ValidationStatus:
        """Execute complete programmatic validation:
        
        1. Extract all citations via CitationParser.
        2. Verify each citation exists in retrieved context (Act, Section, Subsection).
        3. Verify source chunk relevance and attach source drawer metadata.
        4. Detect uncited substantive legal claims.
        5. Formulate final ValidationStatus.
        """
        if not answer or not answer.strip():
            return ValidationStatus(
                is_valid=False,
                checked_citations_count=0,
                valid_citations_count=0,
                invalid_citations_count=0,
                failure_reasons=["Empty generation received."]
            )

        parsed_citations = self.parser.parse(answer)
        
        # Build catalog of available retrieved evidence
        # Map: (act_short.upper(), section_number) -> List[RetrievedDocument]
        evidence_catalog: Dict[Tuple[str, str], List[RetrievedDocument]] = {}
        available_acts: Set[str] = set()
        available_sections: Set[str] = set()

        for doc in retrieved_documents:
            act_short = doc.act_short.upper()
            sec_raw = str(doc.section_number).strip()
            available_acts.add(act_short)
            
            # Extract base section number e.g. "103" from "103(1)" or "35" from "35"
            base_match = re.match(r'^(\d+[A-Za-z]?)', sec_raw)
            base_sec = base_match.group(1) if base_match else sec_raw
            available_sections.add(f"{act_short} s.{base_sec}")
            
            for k in [(act_short, sec_raw), (act_short, base_sec)]:
                if k not in evidence_catalog:
                    evidence_catalog[k] = []
                if doc not in evidence_catalog[k]:
                    evidence_catalog[k].append(doc)

        verified_citations: List[CitationVerification] = []
        invalid_citations: List[Dict[str, Any]] = []
        failure_reasons: List[str] = []

        # 1. Validate every cited statutory reference
        for cit in parsed_citations:
            act = cit.act_short.upper()
            sec = cit.section_number.strip()
            key = (act, sec)

            # Check Act existence
            if act not in available_acts:
                reason = f"Cited Act '{act}' is not present in retrieved context (available: {list(available_acts)})."
                invalid_citations.append({"citation": cit.raw_text, "reason": reason})
                failure_reasons.append(reason)
                continue

            # Check Section existence
            if key not in evidence_catalog:
                reason = f"Cited section [{act} s.{sec}] does not exist in retrieved evidence."
                invalid_citations.append({"citation": cit.raw_text, "reason": reason})
                failure_reasons.append(reason)
                continue

            matching_docs = evidence_catalog[key]
            
            # Check Subsection existence if cited
            sub_verified = True
            if cit.subsection:
                sub_clean = cit.subsection.replace(" ", "")
                found_in_docs = False
                for d in matching_docs:
                    d_sub = (d.subsection or "").replace(" ", "")
                    d_sec = str(d.section_number).replace(" ", "")
                    if d_sub and (sub_clean in d_sub or d_sub in sub_clean):
                        found_in_docs = True
                        break
                    if sub_clean in d_sec:
                        found_in_docs = True
                        break
                    if sub_clean in d.text or f"sub-section {sub_clean}" in d.text:
                        found_in_docs = True
                        break
                
                if not found_in_docs:
                    sub_verified = False
                    reason = f"Cited subsection {cit.subsection} for [{act} s.{sec}] is not supported by retrieved evidence text."
                    invalid_citations.append({"citation": cit.raw_text, "reason": reason})
                    failure_reasons.append(reason)
                    continue

            # Verified! Pick top matching document to enrich metadata
            best_doc = matching_docs[0]
            verified_citations.append(
                CitationVerification(
                    citation_text=cit.canonical_tag,
                    act=best_doc.act,
                    act_short=best_doc.act_short,
                    section=best_doc.section_number,
                    subsection=cit.subsection or best_doc.subsection,
                    clause=cit.clause or best_doc.clause,
                    section_title=best_doc.section_title,
                    page_start=best_doc.page_start,
                    page_end=best_doc.page_end,
                    chunk_id=best_doc.chunk_id,
                    source_text=best_doc.text,
                    is_verified=True
                )
            )

        # 2. Check for missing citations and uncited legal claims
        sentences = self._split_into_sentences(answer)
        uncited_claims: List[str] = []

        # If answer is a refusal/insufficient evidence statement, skip uncited claim check
        is_refusal_answer = any(
            phrase in answer.lower()
            for phrase in [
                "insufficient statutory evidence",
                "cannot be answered from the provided",
                "no statutory evidence",
                "not provided in the retrieved"
            ]
        )

        if not is_refusal_answer:
            if not parsed_citations and sentences:
                # Answer provided without any citations at all
                failure_reasons.append("Answer contains 0 statutory citations. All legal statements must cite retrieved evidence.")

            for s in sentences:
                if self._is_meta_sentence(s):
                    continue
                # If sentence makes strong legal claim, verify it contains an inline citation
                if self._contains_legal_claim(s):
                    s_cits = self.parser.parse(s)
                    if not s_cits:
                        uncited_claims.append(s)

            if uncited_claims:
                reason = f"Detected {len(uncited_claims)} substantive legal claim(s) without required inline citations."
                failure_reasons.append(reason)

        is_valid = (len(failure_reasons) == 0) and (len(invalid_citations) == 0)

        return ValidationStatus(
            is_valid=is_valid,
            checked_citations_count=len(parsed_citations),
            valid_citations_count=len(verified_citations),
            invalid_citations_count=len(invalid_citations),
            invalid_citations=invalid_citations,
            uncited_claims_detected=uncited_claims,
            regeneration_attempted=False,
            failure_reasons=failure_reasons,
            error_details="; ".join(failure_reasons) if failure_reasons else None
        )
