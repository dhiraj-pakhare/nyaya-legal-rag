"""Unit tests for structure-aware statutory chunking."""

import pytest
from backend.app.ingestion.parser import StatutoryParser
from backend.app.ingestion.models import ChunkType


@pytest.fixture(scope="module")
def parsed_result():
    parser = StatutoryParser("BNS bare act 2023.pdf")
    return parser.parse()


def test_chunk_count_and_types(parsed_result):
    chunks = parsed_result.chunks
    assert len(chunks) >= 950
    
    substantive_chunks = [c for c in chunks if c.chunk_type == ChunkType.SUBSTANTIVE_SECTION.value]
    schedule_chunks = [c for c in chunks if c.chunk_type == ChunkType.SCHEDULE_ENTRY.value]
    
    assert len(substantive_chunks) >= 531
    assert len(schedule_chunks) >= 400


def test_section_atomicity_and_splitting(parsed_result):
    chunks = parsed_result.chunks
    
    # Section 1 is short (<= 3200 chars), should be exactly 1 chunk
    s1_chunks = [c for c in chunks if c.section_number == "1" and c.chunk_type == ChunkType.SUBSTANTIVE_SECTION.value]
    assert len(s1_chunks) == 1
    assert s1_chunks[0].chunk_id == "bnss-s1-001"
    
    # Section 187 is very long (> 6000 chars), should be split into multiple chunks
    s187_chunks = [c for c in chunks if c.section_number == "187" and c.chunk_type == ChunkType.SUBSTANTIVE_SECTION.value]
    assert len(s187_chunks) >= 2
    # Verify sub-chunks have contextual header
    for sc in s187_chunks:
        assert "BNSS s.187:" in sc.text or "Chapter XIII:" in sc.text


def test_provisos_and_explanations_attached(parsed_result):
    chunks = parsed_result.chunks
    
    # Chunks for Section 35 must have has_proviso=True
    s35_chunks = [c for c in chunks if c.section_number == "35" and c.chunk_type == ChunkType.SUBSTANTIVE_SECTION.value]
    assert len(s35_chunks) >= 1
    assert any(c.has_proviso for c in s35_chunks)
    assert any("Provided that" in c.text for c in s35_chunks)
    
    # Chunks for Section 1 must have has_explanation=True
    s1_chunks = [c for c in chunks if c.section_number == "1" and c.chunk_type == ChunkType.SUBSTANTIVE_SECTION.value]
    assert s1_chunks[0].has_explanation
    assert "Bharatiya Nagarik Suraksha Sanhita, 2023" in s1_chunks[0].text


def test_deterministic_chunk_ids(parsed_result):
    chunks = parsed_result.chunks
    chunk_ids = [c.chunk_id for c in chunks]
    # Check no duplicate chunk IDs
    assert len(chunk_ids) == len(set(chunk_ids))
    assert "bnss-s35-001" in chunk_ids
    assert "bnss-s1-001" in chunk_ids
    assert "bns-sched1-s105-001" in chunk_ids
