"""Multi-tenant Qdrant repository for user-uploaded documents with mandatory security scoping."""

import logging
import uuid
from typing import Any, Dict, List, Optional, Union
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from backend.app.core.config import settings
from backend.app.core.qdrant_repo import NYAYA_NAMESPACE, get_shared_qdrant_client
from backend.app.document_rag.models import (
    DocumentNotFoundError,
    SecurityScopeError,
    UserDocument,
    UserDocumentChunk,
    UserDocumentSessionScope,
)

logger = logging.getLogger("nyaya.document_rag.repository")


def user_chunk_to_point_id(user_id: str, document_id: str, chunk_id: str) -> str:
    """Generate a deterministic UUIDv5 point ID for a user document chunk."""
    return str(uuid.uuid5(NYAYA_NAMESPACE, f"nyaya://user_point/{user_id}/{document_id}/{chunk_id}"))


class UserDocumentRepository:
    """Multi-tenant repository for isolated user document indexing, retrieval, and deletion."""

    def __init__(
        self,
        client: Optional[QdrantClient] = None,
        url: Optional[str] = None,
        collection_name: str = settings.qdrant_user_collection,
        vector_dim: int = settings.embedding_dimension,
        path: Optional[str] = None,
        in_memory: bool = False
    ):
        self.collection_name = collection_name
        self.vector_dim = vector_dim
        self._doc_registry: Dict[str, UserDocument] = {}

        if client is not None:
            self.client = client
        elif in_memory:
            self.client = QdrantClient(location=":memory:")
        elif path is not None:
            self.client = get_shared_qdrant_client(path=path)
        elif settings.qdrant_path:
            self.client = get_shared_qdrant_client(path=settings.qdrant_path)
        else:
            q_url = url or settings.qdrant_url
            self.client = QdrantClient(url=q_url, api_key=settings.qdrant_api_key or None)

        self.ensure_collection()

    def ensure_collection(self, recreate: bool = False) -> None:
        """Ensure the isolated user document collection and payload indexes exist."""
        collections = [c.name for c in self.client.get_collections().collections]

        if recreate and self.collection_name in collections:
            logger.info(f"Recreating user document collection '{self.collection_name}'...")
            self.client.delete_collection(self.collection_name)
            collections.remove(self.collection_name)

        if self.collection_name not in collections:
            logger.info(f"Creating user document collection '{self.collection_name}' (dim={self.vector_dim})...")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=qmodels.VectorParams(
                    size=self.vector_dim,
                    distance=qmodels.Distance.COSINE
                )
            )

            # Mandatory payload indexes for fast and secure multi-tenant filtering
            for field in ["user_id", "document_id", "session_id"]:
                try:
                    self.client.create_payload_index(
                        collection_name=self.collection_name,
                        field_name=field,
                        field_schema=qmodels.PayloadSchemaType.KEYWORD
                    )
                except Exception as e:
                    logger.debug(f"Payload index on '{field}' creation notice: {e}")

    def register_document(self, document: UserDocument, scope: UserDocumentSessionScope) -> None:
        """Register or update a document in the document registry under strict scope."""
        scope.validate_scope()
        if document.user_id != scope.user_id:
            raise SecurityScopeError("Security scope mismatch: document user_id does not match caller scope")
        self._doc_registry[document.document_id] = document

    def get_document(self, document_id: str, scope: UserDocumentSessionScope) -> UserDocument:
        """Retrieve document metadata. Enforces uniform 404 anti-enumeration protocol."""
        scope.validate_scope()
        doc = self._doc_registry.get(document_id)
        if doc is None or doc.user_id != scope.user_id:
            raise DocumentNotFoundError("Document not found or inaccessible")
        return doc

    def list_documents(self, scope: UserDocumentSessionScope) -> List[UserDocument]:
        """List all documents owned by the caller's trusted identity."""
        scope.validate_scope()
        return [doc for doc in self._doc_registry.values() if doc.user_id == scope.user_id]

    def upsert_user_chunks(
        self,
        chunks: List[UserDocumentChunk],
        vectors: Union[np.ndarray, List[List[float]]],
        scope: UserDocumentSessionScope,
        batch_size: int = 64
    ) -> int:
        """Idempotently index user document chunks into Qdrant."""
        scope.validate_scope()
        if not chunks:
            return 0

        # Validate that every chunk belongs strictly to the caller's scope
        for chunk in chunks:
            if chunk.user_id != scope.user_id:
                raise SecurityScopeError(
                    f"Security violation: chunk user_id '{chunk.user_id}' does not match caller scope '{scope.user_id}'"
                )

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
                point_id = user_chunk_to_point_id(chunk.user_id, chunk.document_id, chunk.chunk_id)
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

        logger.info(f"Indexed {total_points} chunks for doc '{chunks[0].document_id}' in '{self.collection_name}'.")
        return total_points

    def search_dense(
        self,
        query_vector: List[float],
        scope: UserDocumentSessionScope,
        limit: int = 5,
        score_threshold: Optional[float] = None
    ) -> List[UserDocumentChunk]:
        """Execute scoped dense vector search strictly filtered by user_id and active_document_ids."""
        scope.validate_scope()

        # Hard-enforce mandatory user_id filter
        must_conditions: List[qmodels.Condition] = [
            qmodels.FieldCondition(
                key="user_id",
                match=qmodels.MatchValue(value=scope.user_id)
            )
        ]

        # If specific active document IDs are scoped, enforce matching
        if scope.active_document_ids:
            if len(scope.active_document_ids) == 1:
                must_conditions.append(
                    qmodels.FieldCondition(
                        key="document_id",
                        match=qmodels.MatchValue(value=scope.active_document_ids[0])
                    )
                )
            else:
                must_conditions.append(
                    qmodels.FieldCondition(
                        key="document_id",
                        match=qmodels.MatchAny(any=scope.active_document_ids)
                    )
                )

        query_filter = qmodels.Filter(must=must_conditions)

        # Support modern qdrant-client search
        if hasattr(self.client, "query_points"):
            response = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                query_filter=query_filter,
                limit=limit,
                score_threshold=score_threshold
            )
            points = response.points
        elif hasattr(self.client, "search"):
            points = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                query_filter=query_filter,
                limit=limit,
                score_threshold=score_threshold
            )
        else:
            raise AttributeError("QdrantClient lacks search/query_points")

        results: List[UserDocumentChunk] = []
        for p in points:
            payload = p.payload or {}
            # Verify payload security boundary defensively
            if payload.get("user_id") != scope.user_id:
                logger.error("CRITICAL: Scoped search retrieved cross-tenant point payload! Suppressing.")
                continue
            chunk = UserDocumentChunk(
                chunk_id=payload.get("chunk_id", ""),
                document_id=payload.get("document_id", ""),
                user_id=payload.get("user_id", ""),
                session_id=payload.get("session_id"),
                filename=payload.get("filename", ""),
                page_start=payload.get("page_start", 1),
                page_end=payload.get("page_end", 1),
                chunk_index=payload.get("chunk_index", 0),
                text=payload.get("text", ""),
                token_count=payload.get("token_count", 0),
                score=float(p.score) if hasattr(p, "score") and p.score is not None else 0.0,
                metadata=payload.get("metadata", {})
            )
            results.append(chunk)

        return results

    def get_document_chunks(self, document_id: str, scope: UserDocumentSessionScope) -> List[UserDocumentChunk]:
        """Fetch all stored chunks for a specific document within the caller's scope."""
        scope.validate_scope()

        filter_condition = qmodels.Filter(
            must=[
                qmodels.FieldCondition(key="user_id", match=qmodels.MatchValue(value=scope.user_id)),
                qmodels.FieldCondition(key="document_id", match=qmodels.MatchValue(value=document_id))
            ]
        )

        scroll_res, _ = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=filter_condition,
            limit=1000,
            with_payload=True,
            with_vectors=False
        )

        if not scroll_res:
            raise DocumentNotFoundError("Document not found or inaccessible")

        chunks: List[UserDocumentChunk] = []
        for record in scroll_res:
            payload = record.payload or {}
            if payload.get("user_id") != scope.user_id:
                continue
            chunks.append(
                UserDocumentChunk(
                    chunk_id=payload.get("chunk_id", ""),
                    document_id=payload.get("document_id", ""),
                    user_id=payload.get("user_id", ""),
                    session_id=payload.get("session_id"),
                    filename=payload.get("filename", ""),
                    page_start=payload.get("page_start", 1),
                    page_end=payload.get("page_end", 1),
                    chunk_index=payload.get("chunk_index", 0),
                    text=payload.get("text", ""),
                    token_count=payload.get("token_count", 0),
                    metadata=payload.get("metadata", {})
                )
            )

        chunks.sort(key=lambda c: (c.page_start, c.chunk_index))
        return chunks

    def delete_document(self, document_id: str, scope: UserDocumentSessionScope) -> int:
        """Purge all vector points and metadata for a user document. Enforces uniform 404 on error."""
        scope.validate_scope()

        # Check ownership and existence under caller scope
        filter_condition = qmodels.Filter(
            must=[
                qmodels.FieldCondition(key="user_id", match=qmodels.MatchValue(value=scope.user_id)),
                qmodels.FieldCondition(key="document_id", match=qmodels.MatchValue(value=document_id))
            ]
        )

        # Verify matching points exist
        count_res = self.client.count(
            collection_name=self.collection_name,
            count_filter=filter_condition
        )

        has_registered_doc = (
            document_id in self._doc_registry and self._doc_registry[document_id].user_id == scope.user_id
        )
        if count_res.count == 0 and not has_registered_doc:
            raise DocumentNotFoundError("Document not found or inaccessible")

        # Delete points matching the scope
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=filter_condition
        )

        # Remove from local registry
        if document_id in self._doc_registry and self._doc_registry[document_id].user_id == scope.user_id:
            del self._doc_registry[document_id]

        logger.info(f"Deleted document '{document_id}' ({count_res.count} points) for user '{scope.user_id}'.")
        return count_res.count

    def count_user_chunks(self, scope: UserDocumentSessionScope) -> int:
        """Count total chunks owned strictly by the caller's trusted identity."""
        scope.validate_scope()
        filter_condition = qmodels.Filter(
            must=[
                qmodels.FieldCondition(key="user_id", match=qmodels.MatchValue(value=scope.user_id))
            ]
        )
        return self.client.count(
            collection_name=self.collection_name,
            count_filter=filter_condition
        ).count


_GLOBAL_USER_DOC_REPO: Optional[UserDocumentRepository] = None


def get_user_doc_repository() -> UserDocumentRepository:
    """Get or initialize the global singleton UserDocumentRepository instance."""
    global _GLOBAL_USER_DOC_REPO
    if _GLOBAL_USER_DOC_REPO is None:
        _GLOBAL_USER_DOC_REPO = UserDocumentRepository()
    return _GLOBAL_USER_DOC_REPO

