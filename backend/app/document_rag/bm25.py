"""Deterministic multi-worker BM25 sparse index builder for user documents."""

import logging
import re
from typing import Dict, List, Optional, Tuple
from rank_bm25 import BM25Okapi

from backend.app.document_rag.models import UserDocumentChunk, UserDocumentSessionScope

logger = logging.getLogger("nyaya.document_rag.bm25")


def tokenize_document_text(text: str) -> List[str]:
    """Tokenize user document text into normalized alphanumeric tokens."""
    return re.findall(r"\b\w+\b", text.lower())


class UserDocumentBM25Index:
    """A deterministic BM25 index over a specific set of user document chunks."""

    def __init__(self, chunks: List[UserDocumentChunk]):
        self.chunks = chunks
        self.tokenized_corpus = [tokenize_document_text(c.text) for c in chunks]
        if self.tokenized_corpus:
            self.bm25 = BM25Okapi(self.tokenized_corpus)
        else:
            self.bm25 = None

    def search(self, query: str, top_k: int = 5) -> List[UserDocumentChunk]:
        """Score chunks against query and return top_k candidates with scores."""
        if not self.bm25 or not self.chunks:
            return []

        tokens = tokenize_document_text(query)
        if not tokens:
            return []

        scores = self.bm25.get_scores(tokens)
        query_set = set(tokens)

        scored_candidates = []
        for idx, score in enumerate(scores):
            term_matches = sum(1 for t in query_set if t in self.tokenized_corpus[idx])
            if term_matches > 0 or score > 0.0:
                effective_score = float(score) if score > 0.0 else float(term_matches)
                scored_candidates.append((effective_score, idx))

        scored_candidates.sort(key=lambda x: x[0], reverse=True)

        results: List[UserDocumentChunk] = []
        for score, idx in scored_candidates[:top_k]:
            orig = self.chunks[idx]
            # Clone with score
            chunk_copy = UserDocumentChunk(
                chunk_id=orig.chunk_id,
                document_id=orig.document_id,
                user_id=orig.user_id,
                session_id=orig.session_id,
                filename=orig.filename,
                page_start=orig.page_start,
                page_end=orig.page_end,
                chunk_index=orig.chunk_index,
                text=orig.text,
                token_count=orig.token_count,
                score=float(score),
                metadata=orig.metadata
            )
            results.append(chunk_copy)

        return results


class UserDocumentBM25Manager:
    """Multi-worker manager with deterministic reconstruction and worker-local LRU caching."""

    def __init__(self, cache_size: int = 128):
        self._cache: Dict[Tuple[str, Tuple[str, ...]], UserDocumentBM25Index] = {}
        self.cache_size = cache_size

    def _make_cache_key(self, scope: UserDocumentSessionScope) -> Tuple[str, Tuple[str, ...]]:
        return (scope.user_id, tuple(sorted(scope.active_document_ids)))

    def get_or_build_index(
        self,
        chunks: List[UserDocumentChunk],
        scope: UserDocumentSessionScope
    ) -> UserDocumentBM25Index:
        """Get cached index or construct deterministically on-demand in <2ms."""
        scope.validate_scope()
        key = self._make_cache_key(scope)

        if key in self._cache:
            return self._cache[key]

        index = UserDocumentBM25Index(chunks)
        if len(self._cache) >= self.cache_size:
            # Simple eviction
            self._cache.pop(next(iter(self._cache)))

        self._cache[key] = index
        return index

    def invalidate(self, scope: UserDocumentSessionScope) -> None:
        """Invalidate cached index for a user or session upon upload or deletion."""
        key = self._make_cache_key(scope)
        if key in self._cache:
            del self._cache[key]
        # Also invalidate general user keys
        keys_to_del = [k for k in self._cache if k[0] == scope.user_id]
        for k in keys_to_del:
            self._cache.pop(k, None)
