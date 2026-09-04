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

# Stop words ignored during substantive legal continuity overlap check
LEGAL_CONTINUITY_STOP_WORDS = {
    "that", "this", "these", "those", "which", "where", "there", "their",
    "shall", "would", "could", "should", "about", "under", "being", "having",
    "person", "other", "state", "first", "second", "third", "after",
    "before", "while", "since", "during", "within", "without", "between",
    "either", "neither", "also", "both", "such", "than", "then", "into",
    "from", "with", "upon", "when", "what", "must", "done", "does",
    "make", "made", "said", "same", "only", "well", "more", "less"
}


class CitationValidator:
    """Deterministic, programmatic safety layer validating all citations and claims against retrieved evidence."""

    # Keywords indicating a sentence makes a substantive statutory or penal assertion
    LEGAL_CLAIM_KEYWORDS = [
        "punish", "imprison", "fine", "cogniz", "bailable", "non-bailable",
        "warrant", "arrest", "magistrate", "custody", "police officer",
        "court of session", "death penalty", "offence", "offenses", "triable",
        "prescribed", "schedule", "proviso", "statute", "culpable homicide",
        "murder", "assault", "theft", "seizure", "bail", "investigation",
        "penalty", "penal", "sentence", "flog", "labour", "detention"
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
        """Split text into sentences while respecting abbreviations, decimals, list markers, and trailing citation tags."""
        if not text:
            return []

        # 0. Normalize trailing bracketed citations placed right after sentence punctuation
        # e.g., "punishable with fine. [BNS s.105]" -> "punishable with fine [BNS s.105]."
        # Disambiguate trailing citations from leading citations of subsequent sentences
        normalized_text = re.sub(
            r'([.!?])\s*(\[\s*(?:BNS|BNSS|DOC)\s+[^\]]+\])(?=\s*(?:$|\n|[.!?]))',
            r' \2\1',
            text,
            flags=re.IGNORECASE
        )

        # 1. Protect punctuation inside quotation marks (both standard and smart quotes)
        def _mask_quoted_punct(match):
            quoted = match.group(0)
            return (
                quoted.replace('.', '<QDOT>')
                      .replace('!', '<QEXCL>')
                      .replace('?', '<QQMARK>')
            )

        protected = re.sub(r'("[^"]*?"|“[^”]*?”)', _mask_quoted_punct, normalized_text)
        protected = re.sub(r"('[^'\n]*?')", _mask_quoted_punct, protected)

        # 2. Protect abbreviations like s., sec., no., etc. from sentence split
        protected = re.sub(r'\b(s|sec|no|mr|mrs|dr)\.', r'\1<DOT>', protected, flags=re.IGNORECASE)

        # 3. Protect decimal numbers (e.g. 3.5)
        protected = re.sub(r'(?<=\d)\.(?=\d)', '<DECDOT>', protected)

        # 4. Protect list markers and numbering (e.g. "1. ", "2. ", "(i). ", "a. ")
        protected = re.sub(r'(?:^|(?<=\s))(\d{1,2})\.(?=\s+)', r'\1<LISTDOT>', protected)
        protected = re.sub(r'(?:^|(?<=\s))(\([0-9a-zA-Z]+\))\.(?=\s+)', r'\1<LISTDOT>', protected)
        protected = re.sub(r'(?:^|(?<=\s))([a-zA-Z])\.(?=\s+)', r'\1<LISTDOT>', protected)
        protected = re.sub(r'(?:^|(?<=\s))((?:i|ii|iii|iv|v|vi|vii|viii|ix|x))\.(?=\s+)', r'\1<LISTDOT>', protected, flags=re.IGNORECASE)

        # 5. Split on sentence boundaries
        raw_sentences = re.split(r'(?<=[.!?])\s+', protected)

        # 6. Restore placeholders and clean
        sentences = []
        for s in raw_sentences:
            cleaned = (
                s.replace('<DOT>', '.')
                 .replace('<DECDOT>', '.')
                 .replace('<LISTDOT>', '.')
                 .replace('<QDOT>', '.')
                 .replace('<QEXCL>', '!')
                 .replace('<QQMARK>', '?')
                 .strip()
            )
            if cleaned:
                sentences.append(cleaned)
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

    def normalize_natural_citations(self, text: str, retrieved_documents: List[RetrievedDocument]) -> str:
        """Normalize explicit natural-language statutory references into canonical bracketed tags.
        
        Rules:
        - Never guess the Act: resolve explicit references or uniquely matched retrieved sections.
        - Verify against retrieved evidence catalog: ambiguous or unretrieved references are not guessed.
        - Preserves existing bracketed citations without corruption.
        """
        if not text:
            return text

        # 0. Normalize placeholder Act references emitted by LLMs (e.g. "[Act s.105]" or "[Statute s.40]")
        def _rep_placeholder_act(m):
            sec = m.group("sec")
            sub1 = m.group("sub1")
            sub2 = m.group("sub2")
            sub3 = m.group("sub3")
            subs = f"({sub1})" if sub1 else ""
            if sub2:
                subs += f"({sub2})"
            if sub3:
                subs += f"({sub3})"
            matching_acts: Set[str] = set()
            for d in retrieved_documents:
                d_sec = str(d.section_number).strip()
                base_m = re.match(r'^(\d+[A-Za-z]?)', d_sec)
                d_base = base_m.group(1) if base_m else d_sec
                if sec == d_base or sec == d_sec:
                    matching_acts.add(d.act_short.upper())
            if len(matching_acts) == 1:
                act = list(matching_acts)[0]
                return f"[{act} s.{sec}{subs}]"
            return m.group(0)

        pat_placeholder_act = re.compile(
            r'\[\s*(?:Act|Statute)\s+(?:s\.|sec\.|section\s*)\s*(?P<sec>\d+[A-Za-z]?)(?:\s*:[^,\.\n\)]+)?(?:\s*\((?P<sub1>[0-9a-zA-Z]+)\))?(?:\s*\((?P<sub2>[0-9a-zA-Z]+)\))?(?:\s*\((?P<sub3>[0-9a-zA-Z]+)\))?\s*\]',
            re.IGNORECASE
        )
        t = pat_placeholder_act.sub(_rep_placeholder_act, text)

        placeholders: Dict[str, str] = {}

        def _mask_brackets(t_in: str) -> str:
            def _sub(m):
                k = f"__CIT_MASK_{len(placeholders)}__"
                placeholders[k] = m.group(0)
                return k
            return re.sub(r'\[\s*(?:BNS|BNSS|DOC)\s+[^\]]+\]', _sub, t_in, flags=re.IGNORECASE)

        t = _mask_brackets(t)

        # 1. References with explicit Act following Section:
        # e.g. "Section 103(2) of the Bharatiya Nyaya Sanhita, 2023", "Section 105 of BNS", "under Section 40 of BNSS"
        pat_sec_of_act = re.compile(
            r'(?:under\s+|in\s+|as\s+per\s+)?\b(?:Section|Sec\.|s\.)\s*(?P<sec>\d+[A-Za-z]?)(?:\s*:[^,\.\n\)]+)?(?:\s*\([A-Za-z\s,-]+\))?(?:\s*\((?P<sub1>[0-9a-zA-Z]+)\))?(?:\s*\((?P<sub2>[0-9a-zA-Z]+)\))?(?:\s*\((?P<sub3>[0-9a-zA-Z]+)\))?\s+(?:of\s+(?:the\s+)?)?(?P<act>Bharatiya\s+Nyaya\s+Sanhita(?:,\s*2023)?|Bharatiya\s+Nagarik\s+Suraksha\s+Sanhita(?:,\s*2023)?|BNS|BNSS)\b',
            re.IGNORECASE
        )

        def _rep_sec_of_act(m):
            sec = m.group("sec")
            sub1 = m.group("sub1")
            sub2 = m.group("sub2")
            sub3 = m.group("sub3")
            subs = f"({sub1})" if sub1 else ""
            if sub2:
                subs += f"({sub2})"
            if sub3:
                subs += f"({sub3})"
            act_raw = m.group("act").upper()
            act = "BNSS" if ("BNSS" in act_raw or "SURAKSHA" in act_raw or "NAGARIK" in act_raw) else "BNS"
            return f"[{act} s.{sec}{subs}]"

        t = pat_sec_of_act.sub(_rep_sec_of_act, t)
        t = _mask_brackets(t)

        # 2. References with Act preceding Section:
        # e.g. "BNS Section 103(2)", "BNSS Section 40", "Bharatiya Nyaya Sanhita Section 105"
        pat_act_sec = re.compile(
            r'\b(?P<act>Bharatiya\s+Nyaya\s+Sanhita(?:,\s*2023)?|Bharatiya\s+Nagarik\s+Suraksha\s+Sanhita(?:,\s*2023)?|BNS|BNSS)\s+(?:Section|Sec\.|s\.)\s*(?P<sec>\d+[A-Za-z]?)(?:\s*:[^,\.\n\)]+)?(?:\s*\([A-Za-z\s,-]+\))?(?:\s*\((?P<sub1>[0-9a-zA-Z]+)\))?(?:\s*\((?P<sub2>[0-9a-zA-Z]+)\))?(?:\s*\((?P<sub3>[0-9a-zA-Z]+)\))?',
            re.IGNORECASE
        )
        t = pat_act_sec.sub(_rep_sec_of_act, t)
        t = _mask_brackets(t)

        # 3. Unqualified Section references: "Section 40", "Section 103(2)"
        # Only resolve if the section exists in retrieved evidence AND belongs to exactly ONE Act in retrieved context!
        def _rep_unqualified(m):
            sec = m.group("sec")
            sub1 = m.group("sub1")
            sub2 = m.group("sub2")
            sub3 = m.group("sub3")
            subs = f"({sub1})" if sub1 else ""
            if sub2:
                subs += f"({sub2})"
            if sub3:
                subs += f"({sub3})"

            matching_acts: Set[str] = set()
            for d in retrieved_documents:
                d_sec = str(d.section_number).strip()
                base_m = re.match(r'^(\d+[A-Za-z]?)', d_sec)
                d_base = base_m.group(1) if base_m else d_sec
                if sec == d_base or sec == d_sec:
                    matching_acts.add(d.act_short.upper())

            # Only normalize when uniquely grounded in a single Act in retrieved context
            if len(matching_acts) == 1:
                act = list(matching_acts)[0]
                return f"[{act} s.{sec}{subs}]"
            # If 0 or multiple Acts in retrieved context match, do not guess! Return original
            return m.group(0)

        pat_unqualified = re.compile(
            r'(?<!\[)\b(?:Section|Sec\.|s\.)\s*(?P<sec>\d+[A-Za-z]?)(?:\s*:[^,\.\n\)]+)?(?:\s*\((?P<sub1>[0-9a-zA-Z]+)\))?(?:\s*\((?P<sub2>[0-9a-zA-Z]+)\))?(?:\s*\((?P<sub3>[0-9a-zA-Z]+)\))?(?!\])',
            re.IGNORECASE
        )
        t = pat_unqualified.sub(_rep_unqualified, t)
        t = _mask_brackets(t)

        # Unmask all placeholders in reverse order
        for k in sorted(placeholders.keys(), key=lambda x: int(re.search(r'\d+', x).group()), reverse=True):
            t = t.replace(k, placeholders[k])

        return t

    def validate(
        self,
        answer: str,
        retrieved_documents: List[RetrievedDocument]
    ) -> ValidationStatus:
        """Execute complete programmatic validation:
        
        1. Normalize natural-language statutory references into canonical [Act s.X] citations.
        2. Extract all citations via CitationParser.
        3. Verify each citation exists in retrieved context (Act, Section, Subsection).
        4. Verify source chunk relevance and attach source drawer metadata.
        5. Detect uncited substantive legal claims, evaluating conservative same-provision continuity.
        6. Formulate final ValidationStatus.
        """
        if not answer or not answer.strip():
            return ValidationStatus(
                is_valid=False,
                checked_citations_count=0,
                valid_citations_count=0,
                invalid_citations_count=0,
                failure_reasons=["Empty generation received."]
            )

        # 1. Normalize natural-language statutory references
        normalized_answer = self.normalize_natural_citations(answer, retrieved_documents)
        parsed_citations = self.parser.parse(normalized_answer)
        
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

        # 2. Validate every cited statutory reference against retrieved evidence
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

        # 3. Check for missing citations and uncited legal claims with conservative continuity
        sentences = self._split_into_sentences(normalized_answer)
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
            has_doc_citations = bool(re.search(r'\[\s*DOC\s+p\.\d+\s*\]', normalized_answer, re.IGNORECASE))
            if not parsed_citations and not has_doc_citations and sentences:
                # Answer provided without any citations at all
                failure_reasons.append("Answer contains 0 statutory citations. All legal statements must cite retrieved evidence.")

            active_scope: Optional[Tuple[str, str, str]] = None  # (act_upper, sec_num, prov_text)

            for s in sentences:
                if self._is_meta_sentence(s):
                    continue

                s_cits = self.parser.parse(s)
                s_has_doc = bool(re.search(r'\[\s*DOC\s+p\.\d+\s*\]', s, re.IGNORECASE))

                if s_cits:
                    # Check if any citation in this sentence is verified
                    matched_vc = [
                        vc for vc in verified_citations
                        if any(
                            cit.act_short.upper() == vc.act_short.upper() and
                            cit.section_number.strip() == str(vc.section).strip()
                            for cit in s_cits
                        )
                    ]
                    if matched_vc:
                        top_vc = matched_vc[0]
                        act_upper = top_vc.act_short.upper()
                        sec_num = str(top_vc.section).strip()
                        matching_chunks = [
                            d.text for d in retrieved_documents
                            if d.act_short.upper() == act_upper and (
                                str(d.section_number).strip() == sec_num or
                                (re.match(r'^(\d+[A-Za-z]?)', str(d.section_number).strip()) and
                                 re.match(r'^(\d+[A-Za-z]?)', str(d.section_number).strip()).group(1) == sec_num)
                            )
                        ]
                        prov_text = " ".join(matching_chunks).lower()
                        active_scope = (act_upper, sec_num, prov_text)
                    else:
                        active_scope = None
                    continue

                if s_has_doc:
                    # Document citations apply to factual claims, not statutory continuity
                    active_scope = None
                    continue

                if not self._contains_legal_claim(s):
                    continue

                # Sentence contains a substantive legal claim without an inline citation
                is_valid_continuation = False
                if active_scope is not None:
                    act_upper, sec_num, prov_text = active_scope

                    # Invariant 1: No competing statutory reference in this sentence,
                    # unless it is an explicit internal cross-reference present in prov_text
                    competing = False
                    for m in re.finditer(r'\b(?:(BNS|BNSS|IPC|CrPC)|(?:Section|Sec\.|s\.)\s*(\d+[A-Za-z]?))\b', s, re.IGNORECASE):
                        act_ref = m.group(1)
                        sec_ref = m.group(2)
                        if act_ref and act_ref.lower() not in prov_text:
                            competing = True
                            break
                        if sec_ref and (f"section {sec_ref.lower()}" not in prov_text and f"s.{sec_ref.lower()}" not in prov_text and f" {sec_ref.lower()} " not in f" {prov_text} "):
                            competing = True
                            break

                    has_competing_ref = competing

                    if not has_competing_ref:
                        # Invariant 2: Substantive content words must have strong grounding in prov_text
                        content_words = [
                            w for w in re.findall(r'[a-zA-Z]{4,}', s.lower())
                            if w not in LEGAL_CONTINUITY_STOP_WORDS
                        ]
                        if content_words:
                            overlap = sum(
                                1 for w in content_words
                                if w in prov_text or (len(w) > 5 and w[:5] in prov_text)
                            )
                            ratio = overlap / len(content_words)

                            if ratio >= 0.60:
                                # Invariant 3: Any penalty stems asserted must be present in prov_text
                                penalty_stems = ["punish", "imprison", "death", "fine", "cogniz", "bail", "years", "months"]
                                s_penalties = [ps for ps in penalty_stems if ps in s.lower()]
                                penalties_supported = all(ps in prov_text for ps in s_penalties)

                                # Invariant 4: Guard against arbitrary/unsupported power claims
                                arbitrary_phrases = ["additional punishment", "considers appropriate", "any other punishment", "court discretion"]
                                has_arbitrary = any(p in s.lower() and p not in prov_text for p in arbitrary_phrases)

                                if penalties_supported and not has_arbitrary:
                                    is_valid_continuation = True

                if is_valid_continuation:
                    continue
                else:
                    active_scope = None
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
            verified_citations=verified_citations,
            invalid_citations=invalid_citations,
            uncited_claims_detected=uncited_claims,
            regeneration_attempted=False,
            normalized_answer=normalized_answer,
            failure_reasons=failure_reasons,
            error_details="; ".join(failure_reasons) if failure_reasons else None
        )
