"""Multi-source Context Builder with Strict XML Security Delimiters."""

import logging
from typing import List, Optional
from backend.app.core.config import settings
from backend.app.document_rag.models import UserDocumentChunk
from backend.app.generation.context_builder import StatutoryContextBuilder
from backend.app.ingestion.models import StatutoryChunk

logger = logging.getLogger("nyaya.document_rag.context_builder")


class MultiSourceContextBuilder:
    """Builds unified multi-source LLM context with strict XML security boundaries."""

    def __init__(
        self,
        max_context_chars: int = settings.llm_max_context_chars,
        statutory_builder: Optional[StatutoryContextBuilder] = None
    ):
        self.max_context_chars = max_context_chars
        self.statutory_builder = statutory_builder or StatutoryContextBuilder(max_context_chars=max_context_chars)

    def build_context(
        self,
        statutory_chunks: List[StatutoryChunk],
        document_chunks: List[UserDocumentChunk]
    ) -> str:
        """Construct multi-source context containing statutory evidence and untrusted user document evidence."""
        blocks: List[str] = []
        current_chars = 0

        # 1. Statutory Evidence Block (if present)
        if statutory_chunks:
            stat_content = self.statutory_builder.build_context(statutory_chunks)
            if stat_content:
                blocks.append(stat_content)
                current_chars += len(stat_content)

        # 2. User Document Evidence Block (if present)
        if document_chunks:
            doc_blocks: List[str] = []
            for rank, chunk in enumerate(document_chunks, start=1):
                chunk_entry = (
                    f"--- [USER DOCUMENT EVIDENCE #{rank} (Page {chunk.page_start})] ---\n"
                    f"Document ID: {chunk.document_id}\n"
                    f"Filename: {chunk.filename}\n"
                    f"Page: {chunk.page_start}\n"
                    f"Content:\n{chunk.text.strip()}"
                )
                if current_chars + len(chunk_entry) + 100 > self.max_context_chars:
                    logger.warning("Context budget reached; truncating remaining user document chunks.")
                    break

                doc_blocks.append(chunk_entry)
                current_chars += len(chunk_entry)

            if doc_blocks:
                doc_xml = (
                    "<user_document_evidence>\n"
                    + "\n\n".join(doc_blocks)
                    + "\n</user_document_evidence>"
                )
                blocks.append(doc_xml)

        return "\n\n".join(blocks)
