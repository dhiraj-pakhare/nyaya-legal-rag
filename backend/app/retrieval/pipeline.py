"""Master Hybrid Retrieval Pipeline for Nyaya Legal RAG."""

import logging
import time
from typing import Any, Dict, List, Optional

from backend.app.core.config import settings
from backend.app.core.embeddings import EmbeddingModel, get_embedding_model
from backend.app.core.qdrant_repo import QdrantRepository
from backend.app.ingestion.models import StatutoryChunk
from backend.app.retrieval.bm25 import BM25Retriever
from backend.app.retrieval.confidence import ConfidenceScorer, ConfidenceResult
from backend.app.retrieval.dense import DenseRetriever
from backend.app.retrieval.exact_lookup import ExactSectionLookup
from backend.app.retrieval.intent import SectionIntentDetector
from backend.app.retrieval.models import RetrievalFilter, RetrievalResult, RetrievedDocument
from backend.app.retrieval.reranker import CrossEncoderReranker, get_reranker
from backend.app.retrieval.rrf import reciprocal_rank_fusion

logger = logging.getLogger("nyaya.retrieval.pipeline")


class HybridRetrievalPipeline:
    """Coordinated statutory retrieval pipeline supporting:
    
    1. Deterministic Section-Number Intent Routing & Exact Metadata Lookup
    2. Dense Vector Semantic Retrieval (BGE-base-en-v1.5 + Qdrant)
    3. BM25 Sparse Keyword Retrieval (BM25Okapi)
    4. Reciprocal Rank Fusion (RRF)
    5. Cross-Encoder Reranking (ms-marco-MiniLM-L-6-v2)
    6. Multi-Factor Confidence Scoring & Refusal Decision
    7. Statutory Metadata Filtering (act, act_short, chapter, section_number)
    """

    def __init__(
        self,
        chunks: List[StatutoryChunk],
        qdrant_repo: Optional[QdrantRepository] = None,
        embedding_model: Optional[EmbeddingModel] = None,
        bm25_retriever: Optional[BM25Retriever] = None,
        reranker: Optional[CrossEncoderReranker] = None,
        confidence_scorer: Optional[ConfidenceScorer] = None,
        rrf_k: int = 60
    ):
        self.chunks = chunks
        self.rrf_k = rrf_k
        self.intent_detector = SectionIntentDetector()
        self.exact_lookup = ExactSectionLookup(chunks)
        
        # Dense Retriever
        if qdrant_repo is None:
            qdrant_repo = QdrantRepository(
                path="./qdrant_storage",
                collection_name=settings.qdrant_collection,
                vector_dim=settings.embedding_dimension
            )
        self.dense_retriever = DenseRetriever(
            repo=qdrant_repo,
            embedding_model=embedding_model or get_embedding_model()
        )
        
        # BM25 Retriever
        self.bm25_retriever = bm25_retriever or BM25Retriever(chunks=chunks)
        
        # Cross-Encoder Reranker
        self.reranker = reranker or get_reranker()
        
        # Confidence Scorer
        self.confidence_scorer = confidence_scorer or ConfidenceScorer(threshold=settings.confidence_threshold)

    def retrieve(
        self,
        query: str,
        mode: str = "auto",
        top_k: int = settings.reranker_top_k,
        candidate_k: int = settings.reranker_candidate_k,
        filters: Optional[RetrievalFilter] = None,
        k_dense: int = 25,
        k_sparse: int = 25,
        enable_reranking: bool = True,
        override_threshold: Optional[float] = None
    ) -> RetrievalResult:
        """Execute statutory retrieval according to requested mode with confidence scoring.
        
        Modes:
            - 'auto': Intent detection -> Exact lookup -> Fallback to Hybrid RRF + Reranking
            - 'hybrid': Force Dense + BM25 + RRF (+ Reranker if enabled)
            - 'dense': Dense semantic search only
            - 'bm25': BM25 keyword search only
            - 'exact': Exact section lookup only
        """
        start_time = time.perf_counter()
        clean_query = query.strip()
        
        if not clean_query:
            conf = self.confidence_scorer.evaluate(
                query=query,
                documents=[],
                override_threshold=override_threshold
            )
            return RetrievalResult(
                query=query,
                mode=mode,
                documents=[],
                total_retrieved=0,
                latency_ms=0.0,
                is_empty=True,
                confidence=conf.model_dump(),
                is_refused=True,
                refusal_reason=conf.reason
            )

        # 1. Exact Section Lookup Path (Auto or Exact mode)
        intent = self.intent_detector.detect(clean_query) if mode in ("auto", "exact") else None
        if intent:
            exact_docs = self.exact_lookup.lookup(intent, top_k=top_k)
            if exact_docs:
                # If multiple exact matches exist and reranking enabled, refine ordering
                if len(exact_docs) > 1 and enable_reranking:
                    reranked = self.reranker.rerank(clean_query, exact_docs, top_k=top_k)
                    for d in reranked:
                        d.is_exact_match = True
                        d.score = 1.0
                    exact_docs = reranked
                
                conf = self.confidence_scorer.evaluate(
                    query=clean_query,
                    documents=exact_docs,
                    mode="exact_lookup",
                    detected_intent=intent.model_dump(),
                    override_threshold=override_threshold
                )
                latency_ms = (time.perf_counter() - start_time) * 1000
                return RetrievalResult(
                    query=query,
                    mode="exact_lookup",
                    documents=exact_docs,
                    total_retrieved=len(exact_docs),
                    latency_ms=round(latency_ms, 2),
                    applied_filters=filters.model_dump(exclude_none=True) if filters else None,
                    is_empty=False,
                    detected_intent=intent.model_dump(),
                    confidence=conf.model_dump(),
                    is_refused=(conf.decision == "REFUSE"),
                    refusal_reason=conf.reason
                )
            else:
                conf = self.confidence_scorer.evaluate(
                    query=clean_query,
                    documents=[],
                    mode="exact_lookup",
                    detected_intent=intent.model_dump(),
                    override_threshold=override_threshold
                )
                latency_ms = (time.perf_counter() - start_time) * 1000
                return RetrievalResult(
                    query=query,
                    mode="exact_lookup",
                    documents=[],
                    total_retrieved=0,
                    latency_ms=round(latency_ms, 2),
                    is_empty=True,
                    detected_intent=intent.model_dump(),
                    confidence=conf.model_dump(),
                    is_refused=True,
                    refusal_reason=conf.reason
                )

        # 2. Dense-Only Path
        if mode == "dense":
            dense_docs = self.dense_retriever.search(clean_query, top_k=top_k, filters=filters)
            conf = self.confidence_scorer.evaluate(
                query=clean_query,
                documents=dense_docs,
                mode="dense_only",
                override_threshold=override_threshold
            )
            latency_ms = (time.perf_counter() - start_time) * 1000
            return RetrievalResult(
                query=query,
                mode="dense_only",
                documents=dense_docs,
                total_retrieved=len(dense_docs),
                latency_ms=round(latency_ms, 2),
                applied_filters=filters.model_dump(exclude_none=True) if filters else None,
                is_empty=len(dense_docs) == 0,
                confidence=conf.model_dump(),
                is_refused=(conf.decision == "REFUSE"),
                refusal_reason=conf.reason
            )

        # 3. BM25-Only Path
        if mode == "bm25":
            bm25_docs = self.bm25_retriever.search(clean_query, top_k=top_k, filters=filters)
            conf = self.confidence_scorer.evaluate(
                query=clean_query,
                documents=bm25_docs,
                mode="bm25_only",
                override_threshold=override_threshold
            )
            latency_ms = (time.perf_counter() - start_time) * 1000
            return RetrievalResult(
                query=query,
                mode="bm25_only",
                documents=bm25_docs,
                total_retrieved=len(bm25_docs),
                latency_ms=round(latency_ms, 2),
                applied_filters=filters.model_dump(exclude_none=True) if filters else None,
                is_empty=len(bm25_docs) == 0,
                confidence=conf.model_dump(),
                is_refused=(conf.decision == "REFUSE"),
                refusal_reason=conf.reason
            )

        # 4. Hybrid Path (Dense + BM25 + RRF + Cross-Encoder Reranker)
        dense_docs = self.dense_retriever.search(clean_query, top_k=k_dense, filters=filters)
        bm25_docs = self.bm25_retriever.search(clean_query, top_k=k_sparse, filters=filters)
        
        # Merge candidate pool using RRF
        fused_pool = reciprocal_rank_fusion(
            dense_results=dense_docs,
            bm25_results=bm25_docs,
            k=self.rrf_k,
            top_k=candidate_k
        )
        
        # Apply Cross-Encoder reranking if enabled
        if enable_reranking and fused_pool:
            final_docs = self.reranker.rerank(clean_query, fused_pool, top_k=top_k)
        else:
            final_docs = fused_pool[:top_k]

        conf = self.confidence_scorer.evaluate(
            query=clean_query,
            documents=final_docs,
            mode="hybrid_rrf_reranked" if enable_reranking else "hybrid_rrf",
            override_threshold=override_threshold
        )

        latency_ms = (time.perf_counter() - start_time) * 1000
        return RetrievalResult(
            query=query,
            mode="hybrid_rrf",
            documents=final_docs,
            total_retrieved=len(final_docs),
            latency_ms=round(latency_ms, 2),
            applied_filters=filters.model_dump(exclude_none=True) if filters else None,
            is_empty=len(final_docs) == 0,
            confidence=conf.model_dump(),
            is_refused=(conf.decision == "REFUSE"),
            refusal_reason=conf.reason
        )
