"""Unit tests for Confidence Scorer and Refusal Mechanism."""

import pytest
from backend.app.retrieval.confidence import ConfidenceScorer
from backend.app.retrieval.models import RetrievedDocument


def _make_doc(chunk_id: str, sec: str, score: float, dense_rank: int = 1, bm25_rank: int = 1, is_exact: bool = False) -> RetrievedDocument:
    return RetrievedDocument(
        chunk_id=chunk_id,
        act="Bharatiya Nagarik Suraksha Sanhita, 2023",
        act_short="BNSS",
        chapter="V",
        chapter_title="ARREST",
        section_number=sec,
        section_title="Test Section",
        text="Statutory text content",
        page_start=10,
        page_end=10,
        score=score,
        final_rank=1,
        dense_rank=dense_rank,
        bm25_rank=bm25_rank,
        is_exact_match=is_exact
    )


@pytest.fixture
def confidence_scorer():
    return ConfidenceScorer(threshold=0.35)


def test_confidence_exact_match(confidence_scorer):
    """Verify that exact deterministic section matches achieve 1.0 confidence and ACCEPT."""
    doc = _make_doc("c-1", "35", score=1.0, is_exact=True)
    res = confidence_scorer.evaluate("What is section 35 BNSS?", [doc])
    
    assert res.decision == "ACCEPT"
    assert res.confidence_score == 1.0
    assert res.reason == "exact_section_match"


def test_confidence_high_scoring_query(confidence_scorer):
    """Verify high relevance and dual-retriever agreement produces ACCEPT."""
    doc1 = _make_doc("c-1", "187", score=0.85, dense_rank=1, bm25_rank=1)
    doc2 = _make_doc("c-2", "188", score=0.40, dense_rank=5, bm25_rank=4)
    
    res = confidence_scorer.evaluate("maximum police custody period", [doc1, doc2])
    assert res.decision == "ACCEPT"
    assert res.confidence_score >= 0.35
    assert res.reason == "high_retrieval_confidence"


def test_confidence_low_scoring_out_of_scope_refusal(confidence_scorer):
    """Verify low relevance scores and weak agreement produce REFUSE."""
    doc1 = _make_doc("c-1", "50", score=0.08, dense_rank=20, bm25_rank=22)
    doc2 = _make_doc("c-2", "51", score=0.07, dense_rank=24, bm25_rank=25)
    
    res = confidence_scorer.evaluate("jaywalking penalties in Ohio", [doc1, doc2])
    assert res.decision == "REFUSE"
    assert res.confidence_score < 0.35
    assert res.reason == "low_retrieval_confidence"


def test_confidence_empty_retrieval(confidence_scorer):
    """Verify empty retrieval list produces REFUSE."""
    res = confidence_scorer.evaluate("nonexistent query", [])
    assert res.decision == "REFUSE"
    assert res.confidence_score == 0.0
    assert res.reason == "no_retrieval_results"


def test_confidence_nonexistent_section_refusal(confidence_scorer):
    """Verify nonexistent exact section intent produces exact_section_not_found refusal."""
    intent = {"is_exact_lookup": True, "section_number": "9999", "act_short": "BNS"}
    res = confidence_scorer.evaluate("Section 9999 BNS", [], detected_intent=intent)
    
    assert res.decision == "REFUSE"
    assert res.confidence_score == 0.0
    assert res.reason == "exact_section_not_found"
