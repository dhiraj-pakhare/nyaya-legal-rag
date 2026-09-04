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


def test_confidence_tie_aware_subsections_no_false_penalty():
    """Verify BNS s.103(1) and s.103(2) tied with near-identical scores do not trigger a false ambiguity penalty."""
    scorer = ConfidenceScorer(threshold=0.75)
    doc1 = RetrievedDocument(
        chunk_id="bns-103-1",
        act="Bharatiya Nyaya Sanhita, 2023",
        act_short="BNS",
        chapter="VI",
        chapter_title="OF OFFENCES AFFECTING LIFE",
        section_number="103(1)",
        section_title="Murder",
        text="Whoever commits murder shall be punished with death or imprisonment for life...",
        page_start=40,
        page_end=40,
        score=0.9995,
        final_rank=1,
        dense_rank=None,
        bm25_rank=1,
        metadata={"reranker_raw_score": 7.59}
    )
    doc2 = RetrievedDocument(
        chunk_id="bns-103-2",
        act="Bharatiya Nyaya Sanhita, 2023",
        act_short="BNS",
        chapter="VI",
        chapter_title="OF OFFENCES AFFECTING LIFE",
        section_number="103(2)",
        section_title="Murder by group of five or more",
        text="When a group of five or more persons commit murder...",
        page_start=40,
        page_end=40,
        score=0.9994,
        final_rank=2,
        dense_rank=None,
        bm25_rank=2,
        metadata={"reranker_raw_score": 7.58}
    )
    res = scorer.evaluate("punishment for murder under BNS", [doc1, doc2])
    assert res.decision == "ACCEPT"
    assert res.confidence_score >= 0.75
    assert res.reason == "high_retrieval_confidence"


def test_confidence_genuine_ambiguity_penalized():
    """Verify unrelated sections with identical scores trigger ambiguity penalty and lower confidence."""
    scorer = ConfidenceScorer(threshold=0.75)
    doc1 = RetrievedDocument(
        chunk_id="bns-303",
        act="Bharatiya Nyaya Sanhita, 2023",
        act_short="BNS",
        chapter="XVII",
        chapter_title="OF OFFENCES AGAINST PROPERTY",
        section_number="303",
        section_title="Theft",
        text="Whoever intending to take dishonestly any movable property...",
        page_start=110,
        page_end=110,
        score=0.65,
        final_rank=1,
        dense_rank=10,
        bm25_rank=10,
        metadata={"reranker_raw_score": -4.0}
    )
    doc2 = RetrievedDocument(
        chunk_id="bns-318",
        act="Bharatiya Nyaya Sanhita, 2023",
        act_short="BNS",
        chapter="XVII",
        chapter_title="OF OFFENCES AGAINST PROPERTY",
        section_number="318",
        section_title="Cheating",
        text="Whoever by deceiving any person fraudulently or dishonestly...",
        page_start=115,
        page_end=115,
        score=0.65,
        final_rank=2,
        dense_rank=12,
        bm25_rank=12,
        metadata={"reranker_raw_score": -4.0}
    )
    res = scorer.evaluate("property dispute offence", [doc1, doc2])
    assert res.confidence_score < 0.75
    assert res.decision == "REFUSE"
    assert res.reason == "low_retrieval_confidence"


def test_confidence_multilingual_statutory_grounding():
    """Verify non-English query with strong BM25 consensus grounds retrieval even with low CrossEncoder logits."""
    scorer = ConfidenceScorer(threshold=0.75)
    doc1 = RetrievedDocument(
        chunk_id="bns-310-3",
        act="Bharatiya Nyaya Sanhita, 2023",
        act_short="BNS",
        chapter="XI",
        chapter_title="OF OFFENCES AGAINST PROPERTY",
        section_number="310(3)",
        section_title="Murder in dacoity",
        text="If any one of five or more persons committing dacoity commits murder...",
        page_start=112,
        page_end=112,
        score=0.0001,
        final_rank=1,
        dense_rank=None,
        bm25_rank=3,
        metadata={"reranker_raw_score": -10.48}
    )
    doc2 = RetrievedDocument(
        chunk_id="bns-103-1",
        act="Bharatiya Nyaya Sanhita, 2023",
        act_short="BNS",
        chapter="VI",
        chapter_title="OF OFFENCES AFFECTING LIFE",
        section_number="103(1)",
        section_title="Murder",
        text="Whoever commits murder shall be punished with death or imprisonment for life...",
        page_start=40,
        page_end=40,
        score=0.0001,
        final_rank=2,
        dense_rank=None,
        bm25_rank=1,
        metadata={"reranker_raw_score": -10.58}
    )
    doc3 = RetrievedDocument(
        chunk_id="bns-103-2",
        act="Bharatiya Nyaya Sanhita, 2023",
        act_short="BNS",
        chapter="VI",
        chapter_title="OF OFFENCES AFFECTING LIFE",
        section_number="103(2)",
        section_title="Murder by group of five or more",
        text="When a group of five or more persons commit murder...",
        page_start=40,
        page_end=40,
        score=0.0001,
        final_rank=3,
        dense_rank=18,
        bm25_rank=2,
        metadata={"reranker_raw_score": -10.62}
    )
    res = scorer.evaluate("murder kelyavar kay shiksha milte", [doc1, doc2, doc3])
    assert res.decision == "ACCEPT"
    assert res.confidence_score >= 0.75
    assert res.reason == "high_retrieval_confidence"


def test_confidence_determinism():
    """Verify that confidence calculation is strictly deterministic across multiple invocations."""
    scorer = ConfidenceScorer(threshold=0.75)
    doc1 = _make_doc("c-1", "103(1)", score=0.85, dense_rank=1, bm25_rank=1)
    doc2 = _make_doc("c-2", "103(2)", score=0.80, dense_rank=2, bm25_rank=2)
    results = [scorer.evaluate("punishment for murder", [doc1, doc2]) for _ in range(5)]
    first_score = results[0].confidence_score
    first_decision = results[0].decision
    for r in results[1:]:
        assert r.confidence_score == first_score
        assert r.decision == first_decision


def test_confidence_g25_warrant_search_oos_refused():
    """Verify G25-style US Constitution Fourth Amendment search query is refused despite topical keyword cluster agreement."""
    scorer = ConfidenceScorer(threshold=0.75)
    doc1 = RetrievedDocument(
        chunk_id="bnss-49",
        act="Bharatiya Nagarik Suraksha Sanhita, 2023",
        act_short="BNSS",
        chapter="V",
        chapter_title="ARREST OF PERSONS",
        section_number="49",
        section_title="Search of arrested person",
        text="Whenever a person is arrested by a police officer under a warrant...",
        page_start=18,
        page_end=18,
        score=0.0003,
        final_rank=1,
        dense_rank=1,
        bm25_rank=1,
        metadata={"reranker_raw_score": -14.8}
    )
    doc2 = RetrievedDocument(
        chunk_id="bnss-44",
        act="Bharatiya Nagarik Suraksha Sanhita, 2023",
        act_short="BNSS",
        chapter="V",
        chapter_title="ARREST OF PERSONS",
        section_number="44",
        section_title="Search of place entered by person sought to be arrested",
        text="If any person acting under a warrant of arrest...",
        page_start=16,
        page_end=16,
        score=0.0001,
        final_rank=2,
        dense_rank=2,
        bm25_rank=2,
        metadata={"reranker_raw_score": -15.1}
    )
    res = scorer.evaluate("How does the Fourth Amendment of the US Constitution protect against warrantless search and seizure?", [doc1, doc2])
    assert res.decision == "REFUSE"
    assert res.confidence_score < 0.75
    assert res.reason == "low_retrieval_confidence"


def test_confidence_g26_gst_rates_oos_refused():
    """Verify G26-style GST smartphone tax rates query is refused despite 'outside India' cluster overlap."""
    scorer = ConfidenceScorer(threshold=0.75)
    doc1 = RetrievedDocument(
        chunk_id="bnss-209",
        act="Bharatiya Nagarik Suraksha Sanhita, 2023",
        act_short="BNSS",
        chapter="XVI",
        chapter_title="CONDITIONS REQUISITE FOR INITIATION OF PROCEEDINGS",
        section_number="209",
        section_title="Receipt of evidence relating to offences committed outside India",
        text="When any offence alleged to have been committed outside India is being inquired into...",
        page_start=68,
        page_end=68,
        score=0.0,
        final_rank=1,
        dense_rank=1,
        bm25_rank=1,
        metadata={"reranker_raw_score": -15.2}
    )
    doc2 = RetrievedDocument(
        chunk_id="bnss-208",
        act="Bharatiya Nagarik Suraksha Sanhita, 2023",
        act_short="BNSS",
        chapter="XVI",
        chapter_title="CONDITIONS REQUISITE FOR INITIATION OF PROCEEDINGS",
        section_number="208",
        section_title="Offence committed outside India",
        text="When an offence is committed outside India by a citizen of India...",
        page_start=68,
        page_end=68,
        score=0.0,
        final_rank=2,
        dense_rank=2,
        bm25_rank=2,
        metadata={"reranker_raw_score": -15.5}
    )
    res = scorer.evaluate("What are the applicable GST tax rates on imported smartphones in India?", [doc1, doc2])
    assert res.decision == "REFUSE"
    assert res.confidence_score < 0.75
    assert res.reason == "low_retrieval_confidence"


def test_confidence_english_murder_query_accepted():
    """Verify English murder query achieves high confidence ACCEPT."""
    scorer = ConfidenceScorer(threshold=0.75)
    doc1 = RetrievedDocument(
        chunk_id="bns-sched1-s1031-001",
        act="Bharatiya Nyaya Sanhita, 2023",
        act_short="BNS",
        chapter="First Schedule",
        chapter_title="Classification of Offences",
        section_number="103(1)",
        section_title="Murder",
        text="Offence: Murder\nPunishment: Death or imprisonment for life and fine",
        page_start=163,
        page_end=163,
        score=0.9995,
        final_rank=1,
        dense_rank=18,
        bm25_rank=1,
        metadata={"reranker_raw_score": 7.59}
    )
    doc2 = RetrievedDocument(
        chunk_id="bns-sched1-s1032-001",
        act="Bharatiya Nyaya Sanhita, 2023",
        act_short="BNS",
        chapter="First Schedule",
        chapter_title="Classification of Offences",
        section_number="103(2)",
        section_title="Murder by group of five or more",
        text="Offence: Murder by group\nPunishment: Death or imprisonment for life and fine",
        page_start=163,
        page_end=163,
        score=0.9994,
        final_rank=2,
        dense_rank=19,
        bm25_rank=2,
        metadata={"reranker_raw_score": 7.58}
    )
    res = scorer.evaluate("What is the penalty for murder under BNS?", [doc1, doc2])
    assert res.decision == "ACCEPT"
    assert res.confidence_score >= 0.75
    assert res.reason == "high_retrieval_confidence"

