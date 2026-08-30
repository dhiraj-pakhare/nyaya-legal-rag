"""Structure and token-aware chunker for user documents."""

import re
from typing import List, Optional

from backend.app.document_rag.models import UserDocumentChunk
from backend.app.document_rag.pdf_extractor import ExtractedPage


class UserDocumentChunker:
    """Chunks extracted user PDF pages into atomic semantic passages."""

    def __init__(
        self,
        target_chunk_chars: int = 1800,  # ~450 tokens
        overlap_chars: int = 200,        # ~50 tokens
        min_chunk_chars: int = 1
    ):
        self.target_chunk_chars = target_chunk_chars
        self.overlap_chars = overlap_chars
        self.min_chunk_chars = min_chunk_chars

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count based on whitespace splitting and punctuation."""
        return len(re.findall(r"\w+|[^\w\s]", text))

    def chunk_document(
        self,
        pages: List[ExtractedPage],
        document_id: str,
        user_id: str,
        filename: str,
        session_id: Optional[str] = None
    ) -> List[UserDocumentChunk]:
        """Chunk a list of extracted pages into UserDocumentChunk objects."""
        chunks: List[UserDocumentChunk] = []
        chunk_index = 0

        for page in pages:
            page_text = page.text.strip()
            if not page_text:
                continue

            # Split into paragraphs by double newlines or line breaks
            paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", page_text) if p.strip()]
            if not paragraphs:
                paragraphs = [page_text]

            current_chunk_text = ""
            for para in paragraphs:
                if len(current_chunk_text) + len(para) + 1 <= self.target_chunk_chars:
                    current_chunk_text = (current_chunk_text + "\n\n" + para).strip()
                else:
                    if current_chunk_text:
                        chunk_index += 1
                        chunk_id = f"{document_id}_p{page.page_number}_c{chunk_index}"
                        token_count = self._estimate_tokens(current_chunk_text)
                        chunks.append(
                            UserDocumentChunk(
                                chunk_id=chunk_id,
                                document_id=document_id,
                                user_id=user_id,
                                session_id=session_id,
                                filename=filename,
                                page_start=page.page_number,
                                page_end=page.page_number,
                                chunk_index=chunk_index,
                                text=current_chunk_text,
                                token_count=token_count,
                                metadata={
                                    "is_ocr": page.is_ocr,
                                    "page_number": page.page_number
                                }
                            )
                        )
                        # Sliding window overlap
                        overlap = current_chunk_text[-self.overlap_chars:] if len(current_chunk_text) > self.overlap_chars else ""
                        current_chunk_text = (overlap + "\n" + para).strip()
                    else:
                        # Para itself is longer than target_chunk_chars -> split in slices
                        for i in range(0, len(para), self.target_chunk_chars - self.overlap_chars):
                            sub_text = para[i:i + self.target_chunk_chars].strip()
                            if sub_text:
                                chunk_index += 1
                                chunk_id = f"{document_id}_p{page.page_number}_c{chunk_index}"
                                token_count = self._estimate_tokens(sub_text)
                                chunks.append(
                                    UserDocumentChunk(
                                        chunk_id=chunk_id,
                                        document_id=document_id,
                                        user_id=user_id,
                                        session_id=session_id,
                                        filename=filename,
                                        page_start=page.page_number,
                                        page_end=page.page_number,
                                        chunk_index=chunk_index,
                                        text=sub_text,
                                        token_count=token_count,
                                        metadata={
                                            "is_ocr": page.is_ocr,
                                            "page_number": page.page_number
                                        }
                                    )
                                )
                        current_chunk_text = ""

            # Flush remaining buffer for the page
            if current_chunk_text and len(current_chunk_text) >= self.min_chunk_chars:
                chunk_index += 1
                chunk_id = f"{document_id}_p{page.page_number}_c{chunk_index}"
                token_count = self._estimate_tokens(current_chunk_text)
                chunks.append(
                    UserDocumentChunk(
                        chunk_id=chunk_id,
                        document_id=document_id,
                        user_id=user_id,
                        session_id=session_id,
                        filename=filename,
                        page_start=page.page_number,
                        page_end=page.page_number,
                        chunk_index=chunk_index,
                        text=current_chunk_text,
                        token_count=token_count,
                        metadata={
                            "is_ocr": page.is_ocr,
                            "page_number": page.page_number
                        }
                    )
                )

        return chunks
