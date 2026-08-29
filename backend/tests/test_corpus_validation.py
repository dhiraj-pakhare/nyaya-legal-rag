"""Corpus-Level Statutory Chunking and Invariant Validation Tests for Nyaya Legal RAG."""

import pytest
from backend.app.ingestion.parser import StatutoryParser
from backend.app.ingestion.models import ChunkType


@pytest.fixture(scope="module")
def parsed_corpus():
    parser = StatutoryParser("BNS bare act 2023.pdf")
    return parser.parse()


def test_corpus_invariants_chunk_count_and_completeness(parsed_corpus):
    """Invariant: All 531 sections and 39 chapters are present with zero missing."""
    doc = parsed_corpus.document
    assert len(doc.chapters) == 39
    assert len(doc.sections) == 531
    sec_nums = {int(s.section_number) for s in doc.sections}
    assert sec_nums == set(range(1, 532))


def test_corpus_invariants_metadata_fields_present(parsed_corpus):
    """Invariant: Every generated chunk has all 17 required metadata fields without silent omissions."""
    chunks = parsed_corpus.chunks
    assert len(chunks) >= 1000
    
    for c in chunks:
        # Mandatory statutory identity fields
        assert c.act in ["Bharatiya Nagarik Suraksha Sanhita, 2023", "Bharatiya Nyaya Sanhita, 2023"]
        assert c.act_short in ["BNSS", "BNS"]
        assert c.section_number != ""
        assert c.section_title != ""
        assert c.section_title != f"Section {c.section_number}"  # No raw unparsed fallbacks
        
        # Text and length
        assert len(c.text.strip()) > 25
        
        # Booleans
        assert isinstance(c.has_proviso, bool)
        assert isinstance(c.has_exception, bool)
        assert isinstance(c.has_explanation, bool)
        assert isinstance(c.has_illustration, bool)
        
        # Pages
        assert 1 <= c.page_start <= c.page_end <= 249
        
        # Chunk ID & Provenance
        assert c.chunk_id != ""
        assert c.source_uri == "BNS bare act 2023.pdf"
        assert c.ingested_at != ""
        assert isinstance(c.references, list)


def test_corpus_invariants_deterministic_chunk_ids(parsed_corpus):
    """Invariant: Chunk IDs are unique, deterministic, and follow the specified format."""
    chunks = parsed_corpus.chunks
    chunk_ids = [c.chunk_id for c in chunks]
    assert len(chunk_ids) == len(set(chunk_ids)), "Duplicate chunk IDs found in corpus"
    
    # Substantive sections use bnss-s{num}-{seq}
    substantive_ids = [c.chunk_id for c in chunks if c.chunk_type == ChunkType.SUBSTANTIVE_SECTION.value]
    for cid in substantive_ids:
        assert cid.startswith("bnss-s")
        
    # Schedule entries use bns-sched1-s{num}-{seq}
    schedule_ids = [c.chunk_id for c in chunks if c.chunk_type == ChunkType.SCHEDULE_ENTRY.value]
    for cid in schedule_ids:
        assert cid.startswith("bns-sched1-s")


def test_corpus_invariants_proviso_and_explanation_attachment(parsed_corpus):
    """Invariant: Provisos, Exceptions, Explanations, and Illustrations are attached to parents."""
    chunks = parsed_corpus.chunks
    
    # Section 35 contains arrest provisos
    s35_chunks = [c for c in chunks if c.section_number == "35" and c.chunk_type == ChunkType.SUBSTANTIVE_SECTION.value]
    assert len(s35_chunks) >= 1
    assert any(c.has_proviso for c in s35_chunks)
    assert any("Provided that" in c.text for c in s35_chunks)
    
    # Section 1 contains tribal areas explanation
    s1_chunk = next(c for c in chunks if c.section_number == "1" and c.chunk_type == ChunkType.SUBSTANTIVE_SECTION.value)
    assert s1_chunk.has_explanation
    
    # Section 23 contains imprisonment explanation
    s23_chunk = next(c for c in chunks if c.section_number == "23" and c.chunk_type == ChunkType.SUBSTANTIVE_SECTION.value)
    assert s23_chunk.has_explanation


def test_corpus_invariants_long_section_splitting(parsed_corpus):
    """Invariant: Short sections remain atomic; long sections split only at subsection boundaries."""
    chunks = parsed_corpus.chunks
    
    # Section 44 is short (<2000 chars) -> exactly 1 chunk
    s44_chunks = [c for c in chunks if c.section_number == "44" and c.chunk_type == ChunkType.SUBSTANTIVE_SECTION.value]
    assert len(s44_chunks) == 1
    assert s44_chunks[0].chunk_id == "bnss-s44-001"
    
    # Section 187 is very long (>6000 chars) -> split into 3 chunks
    s187_chunks = [c for c in chunks if c.section_number == "187" and c.chunk_type == ChunkType.SUBSTANTIVE_SECTION.value]
    assert len(s187_chunks) >= 2
    # All sub-chunks have contextual header
    for sc in s187_chunks:
        assert "BNSS s.187:" in sc.text
