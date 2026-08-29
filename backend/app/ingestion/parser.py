"""Unified Master Statutory Parser Pipeline for Nyaya Legal RAG.

Coordinates coordinate-aware extraction, structure detection, First Schedule table parsing,
statutory hierarchy chunking, cross-reference normalization, and validation.
"""

from typing import List, Optional, Tuple
from pydantic import BaseModel

from backend.app.ingestion.cleaner import clean_statutory_text
from backend.app.ingestion.models import (
    Chapter,
    FirstScheduleEntry,
    Section,
    StatutoryChunk,
    ValidationReport,
)
from backend.app.ingestion.pdf_extractor import PDFExtractor
from backend.app.ingestion.structure_detector import StructureDetector, StatutoryDocument
from backend.app.ingestion.first_schedule_parser import FirstScheduleParser
from backend.app.ingestion.chunker import StatutoryChunker
from backend.app.ingestion.validator import IngestionValidator


class IngestionResult(BaseModel):
    document: StatutoryDocument
    schedule_entries: List[FirstScheduleEntry]
    chunks: List[StatutoryChunk]
    validation_report: ValidationReport


class StatutoryParser:
    """Master pipeline for ingesting the official BNS/BNSS bare act PDF."""

    def __init__(
        self,
        pdf_path: str = "BNS bare act 2023.pdf",
        substantive_pages: Tuple[int, int] = (1, 157),
        schedule_pages: Tuple[int, int] = (158, 189),
        max_chunk_chars: int = 3200
    ):
        self.pdf_path = pdf_path
        self.substantive_start, self.substantive_end = substantive_pages
        self.schedule_start, self.schedule_end = schedule_pages
        self.chunker = StatutoryChunker(max_chunk_chars=max_chunk_chars)
        self.validator = IngestionValidator()

    def parse(self) -> IngestionResult:
        """Run the end-to-end statutory parsing pipeline."""
        # 1. Extract coordinate-aware layout for substantive sections (Pages 1–157)
        extractor = PDFExtractor(self.pdf_path)
        pages_data = extractor.extract_all_pages(
            start_page=self.substantive_start,
            end_page=self.substantive_end
        )
        
        # 2. Detect Chapters, Sections, Subsections, Provisos, Explanations
        detector = StructureDetector(pages_data)
        document = detector.detect_structure()
        
        # 3. Parse The First Schedule (Offences under BNS, Pages 158–189)
        schedule_parser = FirstScheduleParser(self.pdf_path)
        schedule_entries = schedule_parser.parse_schedule(
            start_page=self.schedule_start,
            end_page=self.schedule_end
        )
        
        # 4. Generate Structure-Aware Statutory Chunks
        chunks = self.chunker.chunk_all(
            sections=document.sections,
            schedule_entries=schedule_entries
        )
        
        # 5. Validate the output and generate report
        validation_report = self.validator.validate(
            sections=document.sections,
            chunks=chunks,
            total_pages=extractor.total_pages
        )
        
        return IngestionResult(
            document=document,
            schedule_entries=schedule_entries,
            chunks=chunks,
            validation_report=validation_report
        )
