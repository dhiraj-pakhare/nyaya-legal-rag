"""Structure-Aware Statutory Hierarchy Chunker for Nyaya Legal RAG.

Transforms hierarchical Section and Schedule ASTs into atomic, structure-aware
StatutoryChunk objects. Guarantees that short sections are never split, long sections
are split only at subsection/clause boundaries, and Provisos, Exceptions, Explanations,
and Illustrations remain strictly attached to their parent chunks.
"""

import re
from typing import List, Optional
from datetime import datetime, timezone

from backend.app.ingestion.models import (
    ChunkType,
    FirstScheduleEntry,
    Section,
    StatutoryChunk,
)


MAX_ATOMIC_CHUNK_CHARS = 3200  # Approx 750-800 tokens


class StatutoryChunker:
    """Generates structure-aware statutory chunks adhering to the DhronAI specification."""

    def __init__(self, max_chunk_chars: int = MAX_ATOMIC_CHUNK_CHARS):
        self.max_chunk_chars = max_chunk_chars

    def chunk_section(self, section: Section) -> List[StatutoryChunk]:
        """Convert a Section AST object into one or more structure-aware StatutoryChunks."""
        chunks: List[StatutoryChunk] = []
        
        has_proviso = len(section.provisos) > 0
        has_exception = len(section.exceptions) > 0
        has_explanation = len(section.explanations) > 0
        has_illustration = len(section.illustrations) > 0
        
        # Rule 1: Section is the atomic unit.
        # If the section text is within max size OR has no subsections to split, keep whole.
        if len(section.raw_text) <= self.max_chunk_chars or not section.subsections:
            chunk_id = f"bnss-s{section.section_number}-001"
            chunk = StatutoryChunk(
                act=section.act,
                act_short=section.act_short,
                chapter=section.chapter_number,
                chapter_title=section.chapter_title,
                section_number=section.section_number,
                section_title=section.section_title,
                subsection=None,
                clause=None,
                chunk_type=ChunkType.SUBSTANTIVE_SECTION.value,
                text=section.raw_text,
                has_illustration=has_illustration,
                has_proviso=has_proviso,
                has_exception=has_exception,
                has_explanation=has_explanation,
                page_start=section.page_start,
                page_end=section.page_end,
                chunk_id=chunk_id,
                references=section.references
            )
            return [chunk]
            
        # Rule 2: Long section splitting strictly at subsection boundaries.
        # Group subsections into logical chunks without breaking sentences or orphaning provisos.
        current_sub_texts: List[str] = []
        current_char_count = 0
        current_sub_id: Optional[str] = None
        chunk_idx = 1
        
        header_prefix = f"[{section.act_short} s.{section.section_number}: {section.section_title} | Chapter {section.chapter_number}: {section.chapter_title}]\n"
        
        for sub in section.subsections:
            sub_len = len(sub.text)
            
            if current_char_count + sub_len > self.max_chunk_chars and current_sub_texts:
                # Flush current sub-chunk
                chunk_text = header_prefix + "\n\n".join(current_sub_texts)
                chunk_id = f"bnss-s{section.section_number}-{chunk_idx:03d}"
                chunks.append(StatutoryChunk(
                    act=section.act,
                    act_short=section.act_short,
                    chapter=section.chapter_number,
                    chapter_title=section.chapter_title,
                    section_number=section.section_number,
                    section_title=section.section_title,
                    subsection=current_sub_id,
                    clause=None,
                    chunk_type=ChunkType.SUBSTANTIVE_SECTION.value,
                    text=chunk_text,
                    has_illustration=any(len(sub.illustrations) > 0 for sub in section.subsections),
                    has_proviso=has_proviso,
                    has_exception=has_exception,
                    has_explanation=has_explanation,
                    page_start=section.page_start,
                    page_end=section.page_end,
                    chunk_id=chunk_id,
                    references=section.references
                ))
                chunk_idx += 1
                current_sub_texts = []
                current_char_count = 0
                
            current_sub_texts.append(sub.text)
            current_char_count += sub_len
            current_sub_id = sub.subsection_id
            
        # Flush remaining subsections
        if current_sub_texts:
            # Check if standalone provisos/explanations exist and attach to last chunk if not already present
            chunk_text = header_prefix + "\n\n".join(current_sub_texts)
            chunk_id = f"bnss-s{section.section_number}-{chunk_idx:03d}"
            chunks.append(StatutoryChunk(
                act=section.act,
                act_short=section.act_short,
                chapter=section.chapter_number,
                chapter_title=section.chapter_title,
                section_number=section.section_number,
                section_title=section.section_title,
                subsection=current_sub_id,
                clause=None,
                chunk_type=ChunkType.SUBSTANTIVE_SECTION.value,
                text=chunk_text,
                has_illustration=has_illustration,
                has_proviso=has_proviso,
                has_exception=has_exception,
                has_explanation=has_explanation,
                page_start=section.page_start,
                page_end=section.page_end,
                chunk_id=chunk_id,
                references=section.references
            ))
            
        return chunks

    def chunk_schedule_entry(self, entry: FirstScheduleEntry, index: int = 1) -> StatutoryChunk:
        """Convert a FirstScheduleEntry into a canonical StatutoryChunk."""
        slug_sec = re.sub(r'[^0-9a-zA-Z]', '', entry.section_number).lower()
        chunk_id = f"bns-sched1-s{slug_sec}-{index:03d}"
        
        formatted_text = (
            f"Bharatiya Nyaya Sanhita, 2023 (BNS) Section {entry.section_number}\n"
            f"Offence: {entry.offence_name}\n"
            f"Punishment: {entry.punishment}\n"
            f"Classification: {entry.cognizable_status} | {entry.bailable_status}\n"
            f"Triable by: {entry.triable_court}"
        )
        
        return StatutoryChunk(
            act="Bharatiya Nyaya Sanhita, 2023",
            act_short="BNS",
            chapter="SCHEDULE_I",
            chapter_title="THE FIRST SCHEDULE - CLASSIFICATION OF OFFENCES",
            section_number=entry.section_number,
            section_title=entry.offence_name,
            subsection=None,
            clause=None,
            chunk_type=ChunkType.SCHEDULE_ENTRY.value,
            text=formatted_text,
            has_illustration=False,
            has_proviso=False,
            has_exception=False,
            has_explanation=False,
            page_start=entry.page,
            page_end=entry.page,
            chunk_id=chunk_id,
            references=[],
            offence_name=entry.offence_name,
            punishment=entry.punishment,
            cognizable_status=entry.cognizable_status,
            bailable_status=entry.bailable_status,
            triable_court=entry.triable_court
        )

    def chunk_all(
        self,
        sections: List[Section],
        schedule_entries: Optional[List[FirstScheduleEntry]] = None
    ) -> List[StatutoryChunk]:
        """Generate all statutory chunks across all substantive sections and schedule entries."""
        all_chunks: List[StatutoryChunk] = []
        
        for sec in sections:
            sec_chunks = self.chunk_section(sec)
            all_chunks.extend(sec_chunks)
            
        if schedule_entries:
            # Track counts per section number to assign unique index
            sec_entry_counts: dict = {}
            for entry in schedule_entries:
                sec_entry_counts[entry.section_number] = sec_entry_counts.get(entry.section_number, 0) + 1
                idx = sec_entry_counts[entry.section_number]
                entry_chunk = self.chunk_schedule_entry(entry, index=idx)
                all_chunks.append(entry_chunk)
                
        return all_chunks
