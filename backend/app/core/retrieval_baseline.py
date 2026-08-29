"""Dense Retrieval Baseline for Vector Index Verification in Nyaya Legal RAG.

Provides a minimal dense search baseline for querying the Qdrant index and evaluating
vector retrieval behavior before hybrid search and reranking are added in later phases.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel

from backend.app.core.embeddings import EmbeddingModel, get_embedding_model
from backend.app.core.qdrant_repo import QdrantRepository


class DenseSearchResult(BaseModel):
    """Structured search result from dense vector retrieval."""
    chunk_id: str
    act_short: str
    chapter: Optional[str] = None
    section_number: str
    section_title: str
    score: float
    page_start: int
    page_end: int
    text: str
    metadata: Dict[str, Any]


def search_dense(
    query: str,
    repo: QdrantRepository,
    embedding_model: Optional[EmbeddingModel] = None,
    top_k: int = 5,
    filters: Optional[Dict[str, Any]] = None,
    score_threshold: Optional[float] = None
) -> List[DenseSearchResult]:
    """Execute a dense similarity query against the Qdrant repository."""
    model = embedding_model or get_embedding_model()
    query_vector = model.embed_query(query)
    
    scored_points = repo.search_dense(
        query_vector=query_vector,
        limit=top_k,
        filters=filters,
        score_threshold=score_threshold
    )
    
    results: List[DenseSearchResult] = []
    for sp in scored_points:
        payload = sp.payload or {}
        results.append(DenseSearchResult(
            chunk_id=payload.get("chunk_id", str(sp.id)),
            act_short=payload.get("act_short", "BNSS"),
            chapter=payload.get("chapter"),
            section_number=payload.get("section_number", ""),
            section_title=payload.get("section_title", ""),
            score=sp.score,
            page_start=payload.get("page_start", 1),
            page_end=payload.get("page_end", 1),
            text=payload.get("text", ""),
            metadata=payload
        ))
        
    return results
