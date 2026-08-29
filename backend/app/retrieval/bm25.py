"""BM25 sparse keyword retrieval engine for statutory chunks."""

import logging
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

from rank_bm25 import BM25Okapi

from backend.app.core.embedding_input import format_chunk_for_embedding
from backend.app.ingestion.models import StatutoryChunk
from backend.app.retrieval.models import RetrievedDocument, RetrievalFilter
from backend.app.retrieval.tokenizer import tokenize_statutory_text

logger = logging.getLogger("nyaya.retrieval.bm25")


class BM25Retriever:
    """BM25Okapi sparse lexical retriever operating over canonical statutory chunks."""

    def __init__(self, chunks: Optional[List[StatutoryChunk]] = None):
        self.chunks: List[StatutoryChunk] = chunks or []
        self.bm25: Optional[BM25Okapi] = None
        self.corpus_tokens: List[List[str]] = []
        
        if self.chunks:
            self._build_index()

    def _build_index(self) -> None:
        """Construct the BM25 index from the chunk list."""
        logger.info(f"Building BM25 index over {len(self.chunks)} statutory chunks...")
        self.corpus_tokens = [
            tokenize_statutory_text(format_chunk_for_embedding(chunk))
            for chunk in self.chunks
        ]
        self.bm25 = BM25Okapi(self.corpus_tokens)
        logger.info("BM25 index built successfully.")

    def search(
        self,
        query: str,
        top_k: int = 25,
        filters: Optional[RetrievalFilter] = None
    ) -> List[RetrievedDocument]:
        """Execute a BM25 sparse keyword search with optional statutory filtering."""
        if not self.bm25 or not self.chunks or not query.strip():
            return []

        query_tokens = tokenize_statutory_text(query)
        if not query_tokens:
            return []

        raw_scores = self.bm25.get_scores(query_tokens)
        
        # Collect scored candidates
        scored_candidates = []
        for idx, score in enumerate(raw_scores):
            chunk = self.chunks[idx]
            
            # Apply metadata filters if specified
            if filters:
                if filters.act and chunk.act != filters.act:
                    continue
                if filters.act_short and chunk.act_short != filters.act_short:
                    continue
                if filters.chapter and chunk.chapter != filters.chapter:
                    continue
                if filters.section_number and chunk.section_number != filters.section_number:
                    continue
                if filters.chunk_type and chunk.chunk_type != filters.chunk_type:
                    continue
                    
            if score > 0.0:  # Only consider positive BM25 scores
                scored_candidates.append((score, idx, chunk))

        # Sort descending by BM25 score
        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        top_candidates = scored_candidates[:top_k]

        results = []
        for rank, (score, _, chunk) in enumerate(top_candidates, 1):
            results.append(
                RetrievedDocument(
                    chunk_id=chunk.chunk_id,
                    act=chunk.act,
                    act_short=chunk.act_short,
                    chapter=chunk.chapter,
                    chapter_title=chunk.chapter_title,
                    section_number=chunk.section_number,
                    section_title=chunk.section_title,
                    subsection=chunk.subsection,
                    clause=chunk.clause,
                    text=chunk.text,
                    page_start=chunk.page_start,
                    page_end=chunk.page_end,
                    chunk_type=chunk.chunk_type,
                    score=float(score),
                    final_rank=rank,
                    bm25_rank=rank,
                    bm25_score=float(score),
                    references=chunk.references,
                    metadata={
                        "has_illustration": chunk.has_illustration,
                        "has_proviso": chunk.has_proviso,
                        "has_exception": chunk.has_exception,
                        "has_explanation": chunk.has_explanation,
                        "source_uri": chunk.source_uri,
                        "ingested_at": chunk.ingested_at,
                        "offence_name": getattr(chunk, "offence_name", None),
                        "punishment": getattr(chunk, "punishment", None),
                        "cognizable_status": getattr(chunk, "cognizable_status", None),
                        "bailable_status": getattr(chunk, "bailable_status", None),
                        "triable_court": getattr(chunk, "triable_court", None),
                    }
                )
            )
        return results

    def save(self, filepath: str) -> None:
        """Serialize BM25 index and chunk metadata to disk."""
        target_path = Path(filepath)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with open(target_path, "wb") as f:
            pickle.dump({"chunks": self.chunks, "corpus_tokens": self.corpus_tokens}, f)
        logger.info(f"BM25 index saved to '{filepath}'.")

    @classmethod
    def load(cls, filepath: str) -> "BM25Retriever":
        """Load BM25 index and chunk metadata from disk."""
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        retriever = cls(chunks=data["chunks"])
        retriever.corpus_tokens = data["corpus_tokens"]
        retriever.bm25 = BM25Okapi(retriever.corpus_tokens)
        logger.info(f"BM25 index loaded from '{filepath}' ({len(retriever.chunks)} chunks).")
        return retriever
