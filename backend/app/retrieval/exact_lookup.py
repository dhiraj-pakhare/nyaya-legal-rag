"""Deterministic exact statutory section lookup engine."""

import logging
from typing import Dict, List, Optional

from backend.app.ingestion.models import StatutoryChunk
from backend.app.retrieval.intent import SectionIntent
from backend.app.retrieval.models import RetrievedDocument

logger = logging.getLogger("nyaya.retrieval.exact_lookup")


class ExactSectionLookup:
    """Performs deterministic, non-vector metadata lookups for exact section numbers."""

    def __init__(self, chunks: List[StatutoryChunk]):
        self.chunks = chunks
        self._section_index: Dict[str, List[StatutoryChunk]] = {}
        self._build_lookup_index()

    def _build_lookup_index(self) -> None:
        """Index chunks by section number and act_short for O(1) lookup."""
        import re
        for chunk in self.chunks:
            sec_num = chunk.section_number.strip().lower()
            act_short = chunk.act_short.strip().upper()
            
            # Key 1: "BNSS:35" or "BNS:103(1)"
            key_act_sec = f"{act_short}:{sec_num}"
            self._section_index.setdefault(key_act_sec, []).append(chunk)
            
            # Key 2: "ANY:35" or "ANY:103(1)"
            key_any_sec = f"ANY:{sec_num}"
            self._section_index.setdefault(key_any_sec, []).append(chunk)

            # If section number contains a sub-identifier like '103(1)' or '103a', also index base number '103'
            base_match = re.match(r'^([0-9]+)', sec_num)
            if base_match:
                base_num = base_match.group(1)
                if base_num != sec_num:
                    key_act_base = f"{act_short}:{base_num}"
                    key_any_base = f"ANY:{base_num}"
                    if chunk not in self._section_index.setdefault(key_act_base, []):
                        self._section_index[key_act_base].append(chunk)
                    if chunk not in self._section_index.setdefault(key_any_base, []):
                        self._section_index[key_any_base].append(chunk)

    def lookup(self, intent: SectionIntent, top_k: int = 10) -> List[RetrievedDocument]:
        """Perform exact statutory lookup for detected SectionIntent."""
        sec_num = intent.section_number.strip().lower()
        act_short = intent.act_short.strip().upper() if intent.act_short else "ANY"
        
        lookup_key = f"{act_short}:{sec_num}"
        matched_chunks = self._section_index.get(lookup_key, [])
        
        # If specific act was requested but yielded 0 results, fall back to ANY
        if not matched_chunks and act_short != "ANY":
            matched_chunks = self._section_index.get(f"ANY:{sec_num}", [])

        if not matched_chunks:
            logger.info(f"No exact match found for section {sec_num} ({act_short}).")
            return []

        # If a subsection was requested, prioritize chunks containing that subsection
        if intent.subsection:
            target_subsec = intent.subsection.strip().lower()
            exact_subsec_chunks = [
                c for c in matched_chunks
                if c.subsection and c.subsection.strip().lower() == target_subsec
            ]
            if exact_subsec_chunks:
                # Place exact subsection chunk first, followed by sibling chunks
                other_chunks = [c for c in matched_chunks if c not in exact_subsec_chunks]
                matched_chunks = exact_subsec_chunks + other_chunks

        results: List[RetrievedDocument] = []
        for rank, chunk in enumerate(matched_chunks[:top_k], 1):
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
                    score=1.0,  # Exact deterministic match gets full confidence
                    final_rank=rank,
                    is_exact_match=True,
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
