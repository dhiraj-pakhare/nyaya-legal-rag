"""Unit tests for deterministic StatutoryContextBuilder."""

from backend.app.generation.context_builder import StatutoryContextBuilder
from backend.app.retrieval.models import RetrievedDocument


def test_context_builder_metadata_preservation():
    """Test that all statutory hierarchy and metadata fields are preserved in formatted context."""
    builder = StatutoryContextBuilder(max_context_chars=5000)
    doc = RetrievedDocument(
        chunk_id="BNSS_s35_p1_chunk0",
        act="Bharatiya Nagarik Suraksha Sanhita, 2023",
        act_short="BNSS",
        chapter="V",
        chapter_title="ARREST OF PERSONS",
        section_number="35",
        section_title="When police may arrest without warrant",
        subsection="(1)",
        clause="(c)",
        text="Any police officer may without an order from a Magistrate and without a warrant, arrest any person...",
        page_start=12,
        page_end=13,
        score=0.95,
        final_rank=1
    )

    context = builder.build_context([doc])
    assert "--- [EVIDENCE ITEM #1] ---" in context
    assert "Chunk ID: BNSS_s35_p1_chunk0" in context
    assert "Act: Bharatiya Nagarik Suraksha Sanhita, 2023 (BNSS)" in context
    assert "Chapter: V - ARREST OF PERSONS" in context
    assert "Section: 35 - When police may arrest without warrant" in context
    assert "Subsection: (1) | Clause: (c)" in context
    assert "Pages: 12–13" in context
    assert "Any police officer may without an order" in context


def test_context_builder_ordering_and_truncation():
    """Test deterministic rank ordering and character budget truncation."""
    builder = StatutoryContextBuilder(max_context_chars=350)
    
    docs = [
        RetrievedDocument(
            chunk_id=f"doc_{i}",
            act="Bharatiya Nyaya Sanhita, 2023",
            act_short="BNS",
            chapter="VI",
            chapter_title="OF OFFENCES AGAINST THE HUMAN BODY",
            section_number=f"10{i}",
            section_title=f"Section {i} Title",
            text=f"Statutory text for section 10{i} with substantial length to test truncation.",
            page_start=50,
            page_end=51,
            score=0.9 - (i * 0.1),
            final_rank=i
        )
        for i in range(1, 6)
    ]

    context = builder.build_context(docs)
    # First item must be present
    assert "[EVIDENCE ITEM #1]" in context
    assert "Section: 101" in context
    # Budget of 350 chars should truncate later items
    assert "[EVIDENCE ITEM #5]" not in context


def test_context_builder_empty_documents():
    """Test context builder behavior with empty document list."""
    builder = StatutoryContextBuilder()
    context = builder.build_context([])
    assert context == "No statutory evidence retrieved."
