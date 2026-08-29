"""Qdrant Vector Database Repository for Nyaya Legal RAG.

Provides an idempotent abstraction over Qdrant collections, deterministic UUIDv5 point ID
generation, structured metadata indexing, payload filtering, and dense vector search.
"""

import uuid
import logging
from typing import Any, Dict, List, Optional, Union
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from backend.app.core.config import settings
from backend.app.ingestion.models import StatutoryChunk

logger = logging.getLogger("nyaya.qdrant")

NYAYA_NAMESPACE = uuid.UUID("6ba7b811-9dad-11d1-80b4-00c04fd430c8")  # UUID namespace


def chunk_id_to_point_id(chunk_id: str) -> str:
    """Generate a deterministic UUIDv5 string from a statutory chunk_id."""
    return str(uuid.uuid5(NYAYA_NAMESPACE, f"nyaya://chunk/{chunk_id}"))


class QdrantRepository:
    """Repository managing Qdrant vector collection lifecycle, upsert, and search."""

    def __init__(
        self,
        client: Optional[QdrantClient] = None,
        url: Optional[str] = None,
        collection_name: str = settings.qdrant_collection,
        vector_dim: int = settings.embedding_dimension,
        path: Optional[str] = None,
        in_memory: bool = False
    ):
        self.collection_name = collection_name
        self.vector_dim = vector_dim
        
        if client is not None:
            self.client = client
        elif in_memory:
            self.client = QdrantClient(location=":memory:")
        elif path is not None:
            self.client = QdrantClient(path=path)
        else:
            q_url = url or settings.qdrant_url
            self.client = QdrantClient(url=q_url, api_key=settings.qdrant_api_key or None)
            
        self.ensure_collection()

    def ensure_collection(self, recreate: bool = False) -> None:
        """Create or recreate the Qdrant collection with cosine distance and payload indexes."""
        collections = [c.name for c in self.client.get_collections().collections]
        
        if recreate and self.collection_name in collections:
            logger.info(f"Recreating Qdrant collection '{self.collection_name}'...")
            self.client.delete_collection(self.collection_name)
            collections.remove(self.collection_name)
            
        if self.collection_name not in collections:
            logger.info(f"Creating Qdrant collection '{self.collection_name}' (dim={self.vector_dim}, metric=Cosine)...")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=qmodels.VectorParams(
                    size=self.vector_dim,
                    distance=qmodels.Distance.COSINE
                )
            )
            
            # Create payload indexes for fast statutory filtering
            for field in ["act", "act_short", "chapter", "section_number", "chunk_type"]:
                try:
                    self.client.create_payload_index(
                        collection_name=self.collection_name,
                        field_name=field,
                        field_schema=qmodels.PayloadSchemaType.KEYWORD
                    )
                except Exception as e:
                    logger.debug(f"Payload index on '{field}' creation notice: {e}")

    def upsert_chunks(
        self,
        chunks: List[StatutoryChunk],
        vectors: Union[np.ndarray, List[List[float]]],
        batch_size: int = 64
    ) -> int:
        """Idempotently upsert statutory chunks and dense vectors into Qdrant.
        
        Returns total number of points upserted.
        """
        if not chunks:
            return 0
            
        if isinstance(vectors, np.ndarray):
            vectors_list = vectors.tolist()
        else:
            vectors_list = vectors
            
        total_points = len(chunks)
        for i in range(0, total_points, batch_size):
            batch_chunks = chunks[i:i + batch_size]
            batch_vecs = vectors_list[i:i + batch_size]
            
            points: List[qmodels.PointStruct] = []
            for chunk, vec in zip(batch_chunks, batch_vecs):
                point_id = chunk_id_to_point_id(chunk.chunk_id)
                # Store full structured metadata as payload
                payload = chunk.model_dump()
                points.append(qmodels.PointStruct(
                    id=point_id,
                    vector=vec,
                    payload=payload
                ))
                
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            
        logger.info(f"Successfully indexed {total_points} statutory points into '{self.collection_name}'.")
        return total_points

    def search_dense(
        self,
        query_vector: List[float],
        limit: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        score_threshold: Optional[float] = None
    ) -> List[qmodels.ScoredPoint]:
        """Execute a dense vector similarity search with optional metadata filters."""
        query_filter: Optional[qmodels.Filter] = None
        if filters:
            must_conditions = []
            for key, val in filters.items():
                must_conditions.append(
                    qmodels.FieldCondition(
                        key=key,
                        match=qmodels.MatchValue(value=val)
                    )
                )
            query_filter = qmodels.Filter(must=must_conditions)
            
        # Support modern qdrant-client query_points API
        if hasattr(self.client, "query_points"):
            response = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                query_filter=query_filter,
                limit=limit,
                score_threshold=score_threshold
            )
            return response.points
        elif hasattr(self.client, "search"):
            return self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                query_filter=query_filter,
                limit=limit,
                score_threshold=score_threshold
            )
        else:
            raise AttributeError("QdrantClient has neither 'query_points' nor 'search' method.")

    def count(self) -> int:
        """Get total count of points in the collection."""
        try:
            return self.client.count(self.collection_name).count
        except Exception:
            return 0
