"""Deterministic Statutory Form Lookup and Intent Classification Engine (Phase 7).

Implements strict deterministic lookup hierarchy:
1. Exact Form Number (e.g. "Form 1", "Form No. 33", "33")
2. Statutory Section Reference (e.g. "Section 35(3)", "s.63", "83")
3. Exact Normalized Title (e.g. "Notice for appearance by the police")
4. Token-Set Fuzzy Matching over canonical form titles and aliases
5. Ambiguity Resolution (returns candidate forms when query is underspecified)
6. Non-Existent Form Refusal (e.g. "Form 99" returns NOT_FOUND cleanly without LLM)
"""

import logging
import re
import time
from typing import Any, Dict, List, Optional, Set, Tuple

from backend.app.forms.models import (
    FormLookupIntent,
    FormLookupResponse,
    StatutoryForm,
)
from backend.app.forms.repository import StatutoryFormRegistry, get_form_registry

logger = logging.getLogger("nyaya.forms.lookup")

# Regex to detect direct Form Number queries e.g. "Form 1", "Form No. 33", "BNSS Form 4", "#1"
FORM_NUM_QUERY_RE = re.compile(
    r'(?:^|\b)(?:bnss\s+)?form\s+(?:no\.?|#)?\s*(\d+)(?:\b|$)',
    re.IGNORECASE
)

# Regex to detect statutory section references e.g. "section 35(3)", "s.63", "u/s 83"
SECTION_QUERY_RE = re.compile(
    r'(?:^|\b)(?:under\s+)?(?:section|sec\.?|s\.|u/s)\s*(\d+(?:\s*\(\s*[0-9a-zA-Z]+\s*\))*)',
    re.IGNORECASE
)


class DeterministicFormIdentifier:
    """High-performance deterministic statutory form identification engine."""

    def __init__(self, registry: Optional[StatutoryFormRegistry] = None):
        self.registry = registry or get_form_registry()

    def identify(self, query: str) -> FormLookupResponse:
        """Execute deterministic form identification and ambiguity check."""
        start_time = time.perf_counter()
        clean_query = query.strip()

        if not clean_query:
            return FormLookupResponse(
                status="NOT_FOUND",
                query=query,
                is_refused=True,
                refusal_reason="Empty query provided.",
                latency_ms=0.0
            )

        # 1. Check for Direct Form Number (e.g. "Form 1", "Form No. 33", "BNSS Form 4")
        form_num_m = FORM_NUM_QUERY_RE.search(clean_query)
        if form_num_m:
            target_num = int(form_num_m.group(1))
            form = self.registry.get_by_number(target_num)
            latency_ms = (time.perf_counter() - start_time) * 1000.0

            if form:
                return FormLookupResponse(
                    status="SUCCESS",
                    query=query,
                    form=form,
                    provenance=form.provenance_citation,
                    latency_ms=round(latency_ms, 3)
                )
            else:
                # Unsupported / non-existent form number e.g. Form 99
                return FormLookupResponse(
                    status="NOT_FOUND",
                    query=query,
                    is_refused=True,
                    refusal_reason=f"Statutory Form No. {target_num} does not exist in The Second Schedule of BNSS (available: Forms 1–58).",
                    latency_ms=round(latency_ms, 3)
                )

        # 2. Check for pure integer query e.g. "1", "33", "58"
        if clean_query.isdigit():
            target_num = int(clean_query)
            form = self.registry.get_by_number(target_num)
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            if form:
                return FormLookupResponse(
                    status="SUCCESS",
                    query=query,
                    form=form,
                    provenance=form.provenance_citation,
                    latency_ms=round(latency_ms, 3)
                )
            else:
                return FormLookupResponse(
                    status="NOT_FOUND",
                    query=query,
                    is_refused=True,
                    refusal_reason=f"Form number {target_num} is out of range (Second Schedule contains Forms 1–58).",
                    latency_ms=round(latency_ms, 3)
                )

        # 3. Check for Statutory Provision Reference e.g. "Section 35(3)", "s.63"
        sec_m = SECTION_QUERY_RE.search(clean_query)
        if sec_m:
            target_sec = sec_m.group(1).strip()
            matching_forms = self.registry.get_by_section(target_sec)
            latency_ms = (time.perf_counter() - start_time) * 1000.0

            if len(matching_forms) == 1:
                form = matching_forms[0]
                return FormLookupResponse(
                    status="SUCCESS",
                    query=query,
                    form=form,
                    provenance=form.provenance_citation,
                    latency_ms=round(latency_ms, 3)
                )
            elif len(matching_forms) > 1:
                candidates = [
                    {
                        "form_number": f.form_number,
                        "form_id": f.form_id,
                        "form_title": f.form_title,
                        "applicable_sections": f.applicable_sections,
                        "provenance": f.provenance_citation
                    }
                    for f in matching_forms
                ]
                return FormLookupResponse(
                    status="AMBIGUOUS",
                    query=query,
                    candidate_forms=candidates,
                    is_refused=False,
                    refusal_reason=f"Multiple statutory forms apply to Section {target_sec}.",
                    latency_ms=round(latency_ms, 3)
                )

        # 4. Check for Exact Normalized Title Match
        exact_form = self.registry.get_by_exact_title(clean_query)
        if exact_form:
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            return FormLookupResponse(
                status="SUCCESS",
                query=query,
                form=exact_form,
                provenance=exact_form.provenance_citation,
                latency_ms=round(latency_ms, 3)
            )

        # 5. Token-Set Fuzzy Matching over all 58 statutory form titles
        scored_candidates = self._fuzzy_score_forms(clean_query)
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        if not scored_candidates:
            return FormLookupResponse(
                status="NOT_FOUND",
                query=query,
                is_refused=True,
                refusal_reason="No matching statutory form found in The Second Schedule.",
                latency_ms=round(latency_ms, 3)
            )

        top_score, top_form = scored_candidates[0]

        # High confidence unambiguous match
        if top_score >= 0.70 and (len(scored_candidates) == 1 or top_score - scored_candidates[1][0] >= 0.15):
            return FormLookupResponse(
                status="SUCCESS",
                query=query,
                form=top_form,
                provenance=top_form.provenance_citation,
                latency_ms=round(latency_ms, 3)
            )

        # Ambiguous match (multiple forms share high similarity e.g. "Attachment warrant")
        if top_score >= 0.50:
            candidates = [
                {
                    "form_number": f.form_number,
                    "form_id": f.form_id,
                    "form_title": f.form_title,
                    "applicable_sections": f.applicable_sections,
                    "match_score": round(score, 2),
                    "provenance": f.provenance_citation
                }
                for score, f in scored_candidates[:5] if score >= 0.45
            ]
            return FormLookupResponse(
                status="AMBIGUOUS",
                query=query,
                candidate_forms=candidates,
                is_refused=False,
                refusal_reason="Query matches multiple statutory forms. Please select the desired form.",
                latency_ms=round(latency_ms, 3)
            )

        # Low confidence -> clean refusal
        return FormLookupResponse(
            status="NOT_FOUND",
            query=query,
            is_refused=True,
            refusal_reason="Insufficient relevance to statutory forms in The Second Schedule.",
            latency_ms=round(latency_ms, 3)
        )

    def _fuzzy_score_forms(self, query: str) -> List[Tuple[float, StatutoryForm]]:
        """Compute token-set overlap scores between query and all 58 form titles."""
        query_tokens = set(StatutoryFormRegistry.normalize_text(query).split())
        # Strip generic stopwords
        query_tokens = {w for w in query_tokens if w not in {"the", "of", "to", "an", "a", "for", "in", "by", "under", "show", "give", "what", "is", "form"}}

        if not query_tokens:
            return []

        scored: List[Tuple[float, StatutoryForm]] = []
        for form in self.registry.list_all_forms():
            title_tokens = set(StatutoryFormRegistry.normalize_text(form.form_title).split())
            title_tokens = {w for w in title_tokens if w not in {"the", "of", "to", "an", "a", "for", "in", "by", "under"}}

            if not title_tokens:
                continue

            intersection = query_tokens.intersection(title_tokens)
            if not intersection:
                continue

            # Jaccard overlap combined with precision over query terms
            precision = len(intersection) / len(query_tokens)
            recall = len(intersection) / len(title_tokens)
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

            # Weight by precision so specific queries score higher
            score = 0.6 * precision + 0.4 * f1
            if score > 0.3:
                scored.append((score, form))

        scored.sort(key=lambda x: x[0], reverse=True)
        return scored
