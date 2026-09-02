"""Scoped Hybrid Retriever for User-Uploaded Documents."""

import logging
from typing import Dict, List, Optional
import numpy as np

from backend.app.core.embeddings import EmbeddingModel, get_embedding_model
from backend.app.document_rag.bm25 import UserDocumentBM25Manager
from backend.app.document_rag.models import (
    UserDocumentChunk,
    UserDocumentSessionScope,
)
from backend.app.document_rag.repository import UserDocumentRepository
from backend.app.retrieval.reranker import CrossEncoderReranker

logger = logging.getLogger("nyaya.document_rag.retriever")


class UserDocumentRetriever:
    """Retrieves and reranks relevant passages from user-uploaded documents under strict security scope."""

    def __init__(
        self,
        repository: UserDocumentRepository,
        embedding_model: Optional[EmbeddingModel] = None,
        reranker: Optional[CrossEncoderReranker] = None,
        bm25_manager: Optional[UserDocumentBM25Manager] = None,
        rrf_k: int = 60
    ):
        self.repository = repository
        self.embedding_model = embedding_model or get_embedding_model()
        if reranker is not None:
            self.reranker = reranker
        else:
            from backend.app.retrieval.reranker import get_reranker
            self.reranker = get_reranker()
        self.bm25_manager = bm25_manager or UserDocumentBM25Manager()
        self.rrf_k = rrf_k

    def retrieve(
        self,
        query: str,
        scope: UserDocumentSessionScope,
        top_k: int = 5,
        dense_limit: int = 10,
        bm25_limit: int = 10
    ) -> List[UserDocumentChunk]:
        """Execute hybrid dense + BM25 retrieval and cross-encoder reranking within the caller's scope."""
        scope.validate_scope()

        if not query or not query.strip():
            return []

        # 1. Fetch dense candidates strictly filtered by user_id and active documents
        query_vec = self.embedding_model.embed_query(query)
        dense_candidates = self.repository.search_dense(
            query_vector=query_vec,
            scope=scope,
            limit=dense_limit
        )

        # 2. Fetch canonical chunks to construct/retrieve BM25 index
        # If active_document_ids is specified, gather chunks from those docs; else gather dense candidates as pool
        all_scope_chunks: List[UserDocumentChunk] = []
        if scope.active_document_ids:
            for doc_id in scope.active_document_ids:
                try:
                    doc_chunks = self.repository.get_document_chunks(doc_id, scope)
                    all_scope_chunks.extend(doc_chunks)
                except Exception:
                    pass
        else:
            all_scope_chunks = list(dense_candidates)

        bm25_index = self.bm25_manager.get_or_build_index(all_scope_chunks, scope)
        bm25_candidates = bm25_index.search(query, top_k=bm25_limit)

        # 3. Reciprocal Rank Fusion (RRF)
        rrf_scores: Dict[str, float] = {}
        chunk_map: Dict[str, UserDocumentChunk] = {}

        # Dense rank scoring
        for rank, chunk in enumerate(dense_candidates, start=1):
            rrf_scores[chunk.chunk_id] = rrf_scores.get(chunk.chunk_id, 0.0) + 1.0 / (self.rrf_k + rank)
            chunk_map[chunk.chunk_id] = chunk

        # BM25 rank scoring
        for rank, chunk in enumerate(bm25_candidates, start=1):
            rrf_scores[chunk.chunk_id] = rrf_scores.get(chunk.chunk_id, 0.0) + 1.0 / (self.rrf_k + rank)
            if chunk.chunk_id not in chunk_map:
                chunk_map[chunk.chunk_id] = chunk

        if not rrf_scores:
            return []

        fused_sorted_ids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)
        fused_candidates = [chunk_map[cid] for cid in fused_sorted_ids[:max(dense_limit, bm25_limit)]]

        # 4. Cross-Encoder Reranking
        # Prepare pairs: (query, chunk_text)
        pairs = [[query, chunk.text] for chunk in fused_candidates]
        if hasattr(self.reranker, "model") and hasattr(self.reranker.model, "predict"):
            raw_scores = self.reranker.model.predict(pairs)
            for chunk, raw_s in zip(fused_candidates, raw_scores):
                chunk.score = float(self.reranker._sigmoid(raw_s)) if hasattr(self.reranker, "_sigmoid") else float(raw_s)
        else:
            for chunk in fused_candidates:
                chunk.score = rrf_scores.get(chunk.chunk_id, 0.0)

        fused_candidates.sort(key=lambda c: c.score, reverse=True)
        top_results = fused_candidates[:top_k]

        for rank, chunk in enumerate(top_results, start=1):
            chunk.final_rank = rank

        return top_results
