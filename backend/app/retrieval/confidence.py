"""Multi-factor confidence scoring and refusal decision engine."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from backend.app.core.config import settings
from backend.app.retrieval.models import RetrievedDocument


class ConfidenceResult(BaseModel):
    """Structured confidence evaluation and refusal decision."""
    confidence_score: float = Field(..., description="Calibrated confidence score in [0.0, 1.0]")
    decision: str = Field(..., description="'ACCEPT' or 'REFUSE'")
    threshold: float = Field(..., description="Active threshold evaluated against")
    reason: str = Field(..., description="Detailed categorical reason for decision")
    top_result_score: float = Field(default=0.0)
    score_margin: float = Field(default=0.0)
    retrieval_evidence: Dict[str, Any] = Field(default_factory=dict)


class ConfidenceScorer:
    """Evaluates multi-factor retrieval signals to determine confidence and refusal."""

    def __init__(self, threshold: float = settings.confidence_threshold):
        self.threshold = threshold

    def evaluate(
        self,
        query: str,
        documents: List[RetrievedDocument],
        mode: str = "hybrid_rrf",
        detected_intent: Optional[Dict[str, Any]] = None,
        override_threshold: Optional[float] = None
    ) -> ConfidenceResult:
        """Compute multi-factor confidence and produce an ACCEPT/REFUSE decision.
        
        Factors evaluated:
        1. Exact deterministic section match (score = 1.0)
        2. Top candidate relevance score
        3. Score margin between rank #1 and rank #2
        4. Cross-retriever agreement (Dense + BM25 alignment)
        5. Total candidate volume
        """
        active_threshold = override_threshold if override_threshold is not None else self.threshold

        # Case 1: Empty retrieval
        if not documents:
            if detected_intent and detected_intent.get("is_exact_lookup"):
                sec = detected_intent.get("section_number")
                act = detected_intent.get("act_short") or "Statute"
                return ConfidenceResult(
                    confidence_score=0.0,
                    decision="REFUSE",
                    threshold=active_threshold,
                    reason="exact_section_not_found",
                    top_result_score=0.0,
                    score_margin=0.0,
                    retrieval_evidence={
                        "total_documents": 0,
                        "query": query,
                        "missing_section": f"{act} Section {sec}"
                    }
                )
            return ConfidenceResult(
                confidence_score=0.0,
                decision="REFUSE",
                threshold=active_threshold,
                reason="no_retrieval_results",
                top_result_score=0.0,
                score_margin=0.0,
                retrieval_evidence={"total_documents": 0, "query": query}
            )

        top_doc = documents[0]

        # Case 2: Exact deterministic section lookup
        if top_doc.is_exact_match:
            return ConfidenceResult(
                confidence_score=1.0,
                decision="ACCEPT",
                threshold=active_threshold,
                reason="exact_section_match",
                top_result_score=1.0,
                score_margin=1.0,
                retrieval_evidence={
                    "mode": "exact_lookup",
                    "matched_section": f"{top_doc.act_short} s.{top_doc.section_number}",
                    "chunk_id": top_doc.chunk_id
                }
            )

        # Case 3: Statistical multi-factor confidence for normal search
        top_score = float(top_doc.score)
        second_score = float(documents[1].score) if len(documents) > 1 else 0.0
        margin = max(0.0, top_score - second_score)

        # Factor A: Top relevance score [0.0, 1.0]
        s_top = max(0.0, min(1.0, top_score))

        # Factor B: Discriminative margin (normalized to [0, 1], saturates at 0.20 margin)
        s_margin = min(1.0, margin / 0.20)

        # Factor C: Dual-retriever agreement (Did dense and BM25 both rank top candidate in top-10?)
        dense_rank = top_doc.dense_rank
        bm25_rank = top_doc.bm25_rank
        if dense_rank is not None and bm25_rank is not None:
            if dense_rank <= 5 and bm25_rank <= 5:
                s_agree = 1.0
            elif dense_rank <= 10 and bm25_rank <= 10:
                s_agree = 0.75
            else:
                s_agree = 0.50
        elif dense_rank is not None or bm25_rank is not None:
            s_agree = 0.40
        else:
            s_agree = 0.20

        # Weighted combination: 60% Top Score + 20% Margin + 20% Agreement
        raw_confidence = (0.60 * s_top) + (0.20 * s_margin) + (0.20 * s_agree)
        confidence_score = round(max(0.0, min(1.0, raw_confidence)), 4)

        decision = "ACCEPT" if confidence_score >= active_threshold else "REFUSE"
        reason = "high_retrieval_confidence" if decision == "ACCEPT" else "low_retrieval_confidence"

        return ConfidenceResult(
            confidence_score=confidence_score,
            decision=decision,
            threshold=active_threshold,
            reason=reason,
            top_result_score=round(top_score, 4),
            score_margin=round(margin, 4),
            retrieval_evidence={
                "top_chunk_id": top_doc.chunk_id,
                "top_section": f"{top_doc.act_short} s.{top_doc.section_number}",
                "dense_rank": dense_rank,
                "bm25_rank": bm25_rank,
                "agreement_score": round(s_agree, 2),
                "margin_score": round(s_margin, 2)
            }
        )
