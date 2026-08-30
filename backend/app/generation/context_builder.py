"""Deterministic Context Builder for statutory evidence formatting."""

import logging
from typing import List, Optional

from backend.app.core.config import settings
from backend.app.retrieval.models import RetrievedDocument

logger = logging.getLogger("nyaya.generation.context_builder")


class StatutoryContextBuilder:
    """Formats retrieved statutory documents into structured LLM context while preserving complete metadata."""

    def __init__(self, max_context_chars: int = settings.llm_max_context_chars):
        self.max_context_chars = max_context_chars

    def format_chunk(self, doc: RetrievedDocument, rank: int) -> str:
        """Format a single retrieved statutory document with complete hierarchy metadata."""
        sub_info = f"Subsection: {doc.subsection}" if doc.subsection else "Subsection: None"
        clause_info = f"Clause: {doc.clause}" if doc.clause else "Clause: None"
        
        header_lines = [
            f"--- [EVIDENCE ITEM #{rank}] ---",
            f"Chunk ID: {doc.chunk_id}",
            f"Act: {doc.act} ({doc.act_short})",
            f"Chapter: {doc.chapter} - {doc.chapter_title}",
            f"Section: {doc.section_number} - {doc.section_title}",
            f"{sub_info} | {clause_info}",
            f"Pages: {doc.page_start}–{doc.page_end}",
            f"Relevance Score: {doc.score:.4f}",
            "Statutory Text:",
            doc.text.strip(),
            "----------------------------"
        ]
        return "\n".join(header_lines)

    def build_context(
        self,
        documents: List[RetrievedDocument],
        max_docs: Optional[int] = None
    ) -> str:
        """Construct deterministic context string from ranked retrieved documents up to character budget."""
        if not documents:
            return "No statutory evidence retrieved."

        docs_to_process = documents[:max_docs] if max_docs is not None else documents
        formatted_chunks: List[str] = []
        current_length = 0

        for rank, doc in enumerate(docs_to_process, start=1):
            chunk_str = self.format_chunk(doc, rank)
            projected_length = current_length + len(chunk_str) + 2

            if current_length > 0 and projected_length > self.max_context_chars:
                logger.info(
                    f"Context budget reached ({current_length}/{self.max_context_chars} chars). "
                    f"Truncating at item {rank - 1} of {len(docs_to_process)}."
                )
                break

            formatted_chunks.append(chunk_str)
            current_length = projected_length

        return "\n\n".join(formatted_chunks)
