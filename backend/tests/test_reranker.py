"""Unit tests for Cross-Encoder Reranker."""

import pytest
from backend.app.retrieval.models import RetrievedDocument
from backend.app.retrieval.reranker import CrossEncoderReranker, get_reranker


def _make_doc(chunk_id: str, sec: str, title: str, text: str, score: float = 0.5) -> RetrievedDocument:
    return RetrievedDocument(
        chunk_id=chunk_id,
        act="Bharatiya Nagarik Suraksha Sanhita, 2023",
        act_short="BNSS",
        chapter="V",
        chapter_title="ARREST",
        section_number=sec,
        section_title=title,
        text=text,
        page_start=10,
        page_end=10,
        score=score,
        final_rank=1
    )


@pytest.fixture(scope="module")
def reranker():
    return get_reranker()


def test_reranker_reordering(reranker):
    """Verify that cross-encoder correctly elevates the most semantically relevant passage."""
    doc_irrelevant = _make_doc(
        "c-1", "125", "Endangering life",
        "Whoever does any act so rashly or negligently as to endanger human life or personal safety..."
    )
    doc_relevant = _make_doc(
        "c-2", "35", "When police may arrest without warrant",
        "Any police officer may without an order from a Magistrate and without a warrant, arrest any person..."
    )

    # Initial order: [irrelevant, relevant]
    candidates = [doc_irrelevant, doc_relevant]
    query = "Under what law can a police officer make an arrest without a court warrant?"

    reranked = reranker.rerank(query, candidates, top_k=2)
    assert len(reranked) == 2
    assert reranked[0].chunk_id == "c-2"
    assert reranked[0].section_number == "35"
    assert reranked[0].score > reranked[1].score


def test_reranker_score_normalization(reranker):
    """Verify that scores returned by reranker are in [0.0, 1.0]."""
    doc = _make_doc("c-1", "35", "Arrest", "Police arrest procedure without warrant.")
    reranked = reranker.rerank("arrest without warrant", [doc], top_k=1)
    
    assert len(reranked) == 1
    assert 0.0 <= reranked[0].score <= 1.0
    assert "reranker_raw_score" in reranked[0].metadata
    assert "reranker_normalized_score" in reranked[0].metadata


def test_reranker_empty_candidates(reranker):
    """Verify empty candidate list returns empty result cleanly."""
    assert reranker.rerank("test query", [], top_k=5) == []
    assert reranker.rerank("", [_make_doc("c-1", "1", "T", "Text")], top_k=5) == []


def test_reranker_top_k_truncation(reranker):
    """Verify that reranker respects top_k limit."""
    docs = [
        _make_doc(f"c-{i}", f"{i}", f"Title {i}", f"Legal text for section {i}")
        for i in range(1, 8)
    ]
    reranked = reranker.rerank("legal section query", docs, top_k=3)
    assert len(reranked) == 3
    assert [d.final_rank for d in reranked] == [1, 2, 3]
