"""Cross-encoder reranker module for post-retrieval candidate refinement."""

import logging
import math
import time
from typing import List, Optional

import numpy as np
from sentence_transformers import CrossEncoder

from backend.app.core.config import settings
from backend.app.retrieval.models import RetrievedDocument

logger = logging.getLogger("nyaya.retrieval.reranker")


class CrossEncoderReranker:
    """Cross-Encoder reranking engine using MS-MARCO MiniLM-L-6-v2."""

    def __init__(
        self,
        model_name: str = settings.reranker_model_name,
        device: str = settings.embedding_device
    ):
        self.model_name = model_name
        self.device = device
        
        logger.info(f"Loading CrossEncoder model '{model_name}' on device '{device}'...")
        start_t = time.perf_counter()
        self.model = CrossEncoder(model_name, device=device)
        self.load_duration = time.perf_counter() - start_t
        logger.info(f"CrossEncoder loaded in {self.load_duration:.2f}s.")

    def _sigmoid(self, x: float) -> float:
        """Convert uncalibrated logit to [0, 1] probability."""
        try:
            return 1.0 / (1.0 + math.exp(-float(x)))
        except OverflowError:
            return 0.0 if x < 0 else 1.0

    def rerank(
        self,
        query: str,
        documents: List[RetrievedDocument],
        top_k: int = settings.reranker_top_k
    ) -> List[RetrievedDocument]:
        """Rerank a list of candidate documents against the query.
        
        Args:
            query: User's search question.
            documents: Candidate documents from Hybrid RRF or Exact Lookup.
            top_k: Number of highest-relevance items to retain.
            
        Returns:
            List[RetrievedDocument] ordered descending by cross-encoder score.
        """
        if not documents or not query.strip():
            return []

        # If only 1 document (e.g. unique exact section hit), return as-is
        if len(documents) == 1 and documents[0].is_exact_match:
            return documents[:top_k]

        # Construct context-enriched query-passage pairs
        pairs = []
        for doc in documents:
            context_header = (
                f"[{doc.act_short}] Chapter {doc.chapter}: {doc.chapter_title} | "
                f"Section {doc.section_number}: {doc.section_title}\n{doc.text}"
            )
            pairs.append((query, context_header))

        start_t = time.perf_counter()
        raw_scores = self.model.predict(pairs)
        duration_ms = (time.perf_counter() - start_t) * 1000

        # Pair scores with original document objects
        scored_pairs = []
        for doc, raw_score in zip(documents, raw_scores):
            norm_score = self._sigmoid(float(raw_score))
            scored_pairs.append((norm_score, float(raw_score), doc))

        # Sort descending by normalized reranker score
        scored_pairs.sort(key=lambda x: x[0], reverse=True)
        top_candidates = scored_pairs[:top_k]

        reranked_results: List[RetrievedDocument] = []
        for new_rank, (norm_score, raw_score, base_doc) in enumerate(top_candidates, 1):
            doc_copy = base_doc.model_copy(deep=True)
            doc_copy.score = round(norm_score, 4)
            doc_copy.final_rank = new_rank
            doc_copy.metadata["reranker_raw_score"] = round(raw_score, 4)
            doc_copy.metadata["reranker_normalized_score"] = round(norm_score, 4)
            doc_copy.metadata["reranking_latency_ms"] = round(duration_ms, 2)
            reranked_results.append(doc_copy)

        logger.debug(f"Reranked {len(documents)} candidates to top-{len(reranked_results)} in {duration_ms:.2f}ms.")
        return reranked_results


_GLOBAL_RERANKER: Optional[CrossEncoderReranker] = None


def get_reranker() -> CrossEncoderReranker:
    """Get or initialize singleton CrossEncoderReranker instance."""
    global _GLOBAL_RERANKER
    if _GLOBAL_RERANKER is None:
        _GLOBAL_RERANKER = CrossEncoderReranker()
    return _GLOBAL_RERANKER
