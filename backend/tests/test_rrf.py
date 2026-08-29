"""Unit tests for Reciprocal Rank Fusion (RRF)."""

import pytest
from backend.app.retrieval.models import RetrievedDocument
from backend.app.retrieval.rrf import reciprocal_rank_fusion


def _make_doc(chunk_id: str, sec: str, title: str, score: float = 0.5) -> RetrievedDocument:
    return RetrievedDocument(
        chunk_id=chunk_id,
        act="Bharatiya Nagarik Suraksha Sanhita, 2023",
        act_short="BNSS",
        chapter="V",
        chapter_title="ARREST",
        section_number=sec,
        section_title=title,
        text=f"Text of section {sec}",
        page_start=10,
        page_end=10,
        score=score,
        final_rank=1
    )


def test_rrf_rank_fusion_combined_ranking():
    """Verify that documents appearing near the top of both lists receive highest RRF rank."""
    doc_a = _make_doc("chunk-a", "35", "Arrest without warrant")
    doc_b = _make_doc("chunk-b", "36", "Officer duty")
    doc_c = _make_doc("chunk-c", "37", "Private person")

    # Dense ranking: [A, B, C]
    dense_results = [doc_a, doc_b, doc_c]
    # BM25 ranking:  [A, C, B]
    bm25_results = [doc_a, doc_c, doc_b]

    # With k=60:
    # A score: 1/(60+1) + 1/(60+1) = 2/61 = 0.032787
    # B score: 1/(60+2) + 1/(60+3) = 1/62 + 1/63 = 0.016129 + 0.015873 = 0.032002
    # C score: 1/(60+3) + 1/(60+2) = 1/63 + 1/62 = 0.032002
    fused = reciprocal_rank_fusion(dense_results, bm25_results, k=60)
    
    assert len(fused) == 3
    assert fused[0].chunk_id == "chunk-a"
    assert fused[0].final_rank == 1
    assert fused[0].dense_rank == 1
    assert fused[0].bm25_rank == 1
    assert pytest.approx(fused[0].rrf_score, 0.0001) == 2 / 61


def test_rrf_disjoint_lists():
    """Verify RRF handling when documents appear in only one retriever's output."""
    doc_a = _make_doc("chunk-a", "35", "Dense Only")
    doc_b = _make_doc("chunk-b", "36", "Sparse Only")

    dense_results = [doc_a]
    bm25_results = [doc_b]

    fused = reciprocal_rank_fusion(dense_results, bm25_results, k=60)
    assert len(fused) == 2
    # Both had rank 1 in their respective lists, so RRF score is 1/61 each
    assert pytest.approx(fused[0].rrf_score, 0.0001) == 1 / 61
    assert pytest.approx(fused[1].rrf_score, 0.0001) == 1 / 61


def test_rrf_configurable_k():
    """Verify that different k parameters modulate the score magnitude properly."""
    doc_a = _make_doc("chunk-a", "35", "Arrest")
    fused_k20 = reciprocal_rank_fusion([doc_a], [doc_a], k=20)
    fused_k60 = reciprocal_rank_fusion([doc_a], [doc_a], k=60)

    # Score with k=20: 2/21 = 0.095238
    # Score with k=60: 2/61 = 0.032787
    assert fused_k20[0].rrf_score > fused_k60[0].rrf_score
    assert pytest.approx(fused_k20[0].rrf_score, 0.0001) == 2 / 21
    assert pytest.approx(fused_k60[0].rrf_score, 0.0001) == 2 / 61
