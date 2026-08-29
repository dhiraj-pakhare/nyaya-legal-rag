"""Reciprocal Rank Fusion (RRF) for combining dense and sparse retrieval ranks."""

from typing import Dict, List, Optional
from backend.app.retrieval.models import RetrievedDocument


def reciprocal_rank_fusion(
    dense_results: List[RetrievedDocument],
    bm25_results: List[RetrievedDocument],
    k: int = 60,
    top_k: int = 25
) -> List[RetrievedDocument]:
    """Merge ranked lists from dense and BM25 retrievers using Reciprocal Rank Fusion (RRF).
    
    Formula:
        RRF_score(d) = sum( 1 / (k + rank_i(d)) ) for retriever i in [dense, bm25]
        
    Args:
        dense_results: Ordered list of documents from dense vector search.
        bm25_results: Ordered list of documents from BM25 sparse search.
        k: Smoothing constant (default: 60, configurable).
        top_k: Maximum number of fused items to return.
        
    Returns:
        List[RetrievedDocument] sorted descending by RRF score.
    """
    fused_scores: Dict[str, float] = {}
    doc_registry: Dict[str, RetrievedDocument] = {}
    dense_ranks: Dict[str, int] = {}
    dense_scores: Dict[str, float] = {}
    bm25_ranks: Dict[str, int] = {}
    bm25_scores: Dict[str, float] = {}

    # Process dense results
    for rank, doc in enumerate(dense_results, 1):
        cid = doc.chunk_id
        dense_ranks[cid] = rank
        dense_scores[cid] = doc.score
        doc_registry[cid] = doc
        fused_scores[cid] = fused_scores.get(cid, 0.0) + (1.0 / (k + rank))

    # Process BM25 results
    for rank, doc in enumerate(bm25_results, 1):
        cid = doc.chunk_id
        bm25_ranks[cid] = rank
        bm25_scores[cid] = doc.score
        if cid not in doc_registry:
            doc_registry[cid] = doc
        fused_scores[cid] = fused_scores.get(cid, 0.0) + (1.0 / (k + rank))

    # Sort items deterministically:
    # 1. RRF score descending
    # 2. Minimum rank among individual retrievers ascending
    # 3. chunk_id ascending (lexicographical tie-break)
    sorted_chunk_ids = sorted(
        fused_scores.keys(),
        key=lambda cid: (
            -fused_scores[cid],
            min(dense_ranks.get(cid, 9999), bm25_ranks.get(cid, 9999)),
            cid
        )
    )

    fused_results: List[RetrievedDocument] = []
    for final_rank, cid in enumerate(sorted_chunk_ids[:top_k], 1):
        base_doc = doc_registry[cid]
        fused_doc = base_doc.model_copy(deep=True)
        
        fused_doc.score = round(fused_scores[cid], 6)
        fused_doc.rrf_score = round(fused_scores[cid], 6)
        fused_doc.final_rank = final_rank
        fused_doc.dense_rank = dense_ranks.get(cid)
        fused_doc.dense_score = dense_scores.get(cid)
        fused_doc.bm25_rank = bm25_ranks.get(cid)
        fused_doc.bm25_score = bm25_scores.get(cid)
        
        fused_results.append(fused_doc)

    return fused_results
