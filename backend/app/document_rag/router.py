"""Query Intent Router for Nyaya Legal RAG (Statutory, Document, Combined)."""

import re
from typing import List
from backend.app.document_rag.models import QueryIntent, RoutingDecision, UserDocumentSessionScope
from backend.app.retrieval.intent import SectionIntentDetector


class QueryIntentRouter:
    """Classifies user queries into Statutory, Document, or Combined retrieval paths."""

    DOCUMENT_SIGNALS = [
        "document", "notice", "fir", "petition", "attached file", "uploaded",
        "allegation", "complainant", "accused", "according to the document",
        "agreement", "contract", "clause", "complaint", "in my case",
        "bail application", "charge sheet", "affidavit", "plaint", "written statement"
    ]

    STATUTORY_SIGNALS = [
        "bns", "bnss", "section", "s.", "sec.", "punishment for",
        "bailable", "cognizable", "magistrate power", "arrest procedure",
        "schedule", "offence under", "ipc", "crpc", "bare act",
        "culpable homicide", "murder", "theft", "extortion", "robbery",
        "dacoity", "cheating", "mischief", "criminal breach of trust"
    ]

    def __init__(self):
        self.exact_detector = SectionIntentDetector()

    def route(self, query: str, scope: UserDocumentSessionScope) -> RoutingDecision:
        """Determine whether query targets statutory law, user document, or both."""
        scope.validate_scope()

        # If user has no active documents uploaded, route strictly to Statutory
        if not scope.active_document_ids:
            return RoutingDecision(
                intent=QueryIntent.STATUTORY_ONLY,
                confidence=1.0,
                target_document_ids=[],
                reason="No active user documents in session scope; routing to statutory corpus."
            )

        q_lower = query.lower()
        exact_intent = self.exact_detector.detect(query)
        is_exact = exact_intent is not None and exact_intent.is_exact_lookup

        # Check signals
        has_doc_signal = any(sig in q_lower for sig in self.DOCUMENT_SIGNALS)
        has_stat_signal = (
            is_exact or
            any(sig in q_lower for sig in self.STATUTORY_SIGNALS)
        )

        detected_sections = [exact_intent.section_number] if is_exact and exact_intent.section_number else []

        if has_doc_signal and has_stat_signal:
            return RoutingDecision(
                intent=QueryIntent.COMBINED,
                confidence=0.95,
                detected_statutory_sections=detected_sections,
                target_document_ids=scope.active_document_ids,
                reason="Query explicitly references both user document facts and statutory provisions."
            )
        elif has_doc_signal:
            return RoutingDecision(
                intent=QueryIntent.DOCUMENT_ONLY,
                confidence=0.90,
                detected_statutory_sections=[],
                target_document_ids=scope.active_document_ids,
                reason="Query explicitly references uploaded document context."
            )
        elif has_stat_signal:
            return RoutingDecision(
                intent=QueryIntent.STATUTORY_ONLY,
                confidence=0.90,
                detected_statutory_sections=detected_sections,
                target_document_ids=[],
                reason="Query explicitly references statutory legal provisions."
            )
        else:
            # Ambiguous query with active documents in session -> Safe Combined Route
            return RoutingDecision(
                intent=QueryIntent.COMBINED,
                confidence=0.70,
                detected_statutory_sections=[],
                target_document_ids=scope.active_document_ids,
                reason="Ambiguous query with active document session; executing combined dual retrieval."
            )
