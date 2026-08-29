"""Dense vector retrieval engine using Qdrant and BGE embeddings."""

import logging
from typing import Any, Dict, List, Optional

from backend.app.core.embeddings import EmbeddingModel, get_embedding_model
from backend.app.core.qdrant_repo import QdrantRepository
from backend.app.retrieval.models import RetrievedDocument, RetrievalFilter

logger = logging.getLogger("nyaya.retrieval.dense")


class DenseRetriever:
    """Dense semantic vector retriever backed by Qdrant and BAAI/bge-base-en-v1.5."""

    def __init__(
        self,
        repo: QdrantRepository,
        embedding_model: Optional[EmbeddingModel] = None
    ):
        self.repo = repo
        self.embedding_model = embedding_model or get_embedding_model()

    def search(
        self,
        query: str,
        top_k: int = 25,
        filters: Optional[RetrievalFilter] = None
    ) -> List[RetrievedDocument]:
        """Execute dense vector similarity search with optional payload filters."""
        if not query.strip():
            return []

        # Generate normalized query vector with asymmetric BGE instruction prefix
        query_vector = self.embedding_model.embed_query(query)
        if hasattr(query_vector, "tolist"):
            query_vector = query_vector.tolist()
        
        # Build filter dictionary for Qdrant payload
        qdrant_filters = None
        if filters:
            qdrant_filters = {}
            if filters.act:
                qdrant_filters["act"] = filters.act
            if filters.act_short:
                qdrant_filters["act_short"] = filters.act_short
            if filters.chapter:
                qdrant_filters["chapter"] = filters.chapter
            if filters.section_number:
                qdrant_filters["section_number"] = filters.section_number
            if filters.chunk_type:
                qdrant_filters["chunk_type"] = filters.chunk_type
            if not qdrant_filters:
                qdrant_filters = None

        scored_points = self.repo.search_dense(
            query_vector=query_vector,
            limit=top_k,
            filters=qdrant_filters
        )

        results = []
        for rank, sp in enumerate(scored_points, 1):
            p = sp.payload or {}
            results.append(
                RetrievedDocument(
                    chunk_id=p.get("chunk_id", str(sp.id)),
                    act=p.get("act", ""),
                    act_short=p.get("act_short", ""),
                    chapter=p.get("chapter", ""),
                    chapter_title=p.get("chapter_title", ""),
                    section_number=p.get("section_number", ""),
                    section_title=p.get("section_title", ""),
                    subsection=p.get("subsection"),
                    clause=p.get("clause"),
                    text=p.get("text", ""),
                    page_start=p.get("page_start", 0),
                    page_end=p.get("page_end", 0),
                    chunk_type=p.get("chunk_type", "substantive_section"),
                    score=float(sp.score),
                    final_rank=rank,
                    dense_rank=rank,
                    dense_score=float(sp.score),
                    references=p.get("references", []),
                    metadata={
                        "has_illustration": p.get("has_illustration", False),
                        "has_proviso": p.get("has_proviso", False),
                        "has_exception": p.get("has_exception", False),
                        "has_explanation": p.get("has_explanation", False),
                        "source_uri": p.get("source_uri", ""),
                        "ingested_at": p.get("ingested_at", ""),
                        "offence_name": p.get("offence_name"),
                        "punishment": p.get("punishment"),
                        "cognizable_status": p.get("cognizable_status"),
                        "bailable_status": p.get("bailable_status"),
                        "triable_court": p.get("triable_court"),
                    }
                )
            )
        return results
