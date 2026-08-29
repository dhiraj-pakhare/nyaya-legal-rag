"""Integration and unit tests for HybridRetrievalPipeline."""

import pytest
from backend.app.core.embeddings import get_embedding_model
from backend.app.core.qdrant_repo import QdrantRepository
from backend.app.ingestion.parser import StatutoryParser
from backend.app.retrieval.models import RetrievalFilter
from backend.app.retrieval.pipeline import HybridRetrievalPipeline


@pytest.fixture(scope="module")
def pipeline():
    parser = StatutoryParser("BNS bare act 2023.pdf")
    res = parser.parse()
    
    qdrant_repo = QdrantRepository(
        path="./qdrant_storage",
        collection_name="nyaya_legal_corpus",
        vector_dim=768
    )
    embed_model = get_embedding_model()
    
    return HybridRetrievalPipeline(
        chunks=res.chunks,
        qdrant_repo=qdrant_repo,
        embedding_model=embed_model,
        rrf_k=60
    )


def test_pipeline_exact_lookup_routing(pipeline):
    """Verify 'auto' mode routes exact section queries to exact lookup."""
    res = pipeline.retrieve("What is section 103 BNS?", mode="auto", top_k=5)
    assert res.mode == "exact_lookup"
    assert not res.is_empty
    assert len(res.documents) > 0
    assert res.documents[0].act_short == "BNS"
    assert res.documents[0].is_exact_match is True
    assert res.documents[0].score == 1.0


def test_pipeline_hybrid_rrf_routing(pipeline):
    """Verify 'auto' mode routes conceptual queries to hybrid RRF."""
    res = pipeline.retrieve("Can a private citizen arrest someone who commits a crime?", mode="auto", top_k=5)
    assert res.mode == "hybrid_rrf"
    assert not res.is_empty
    assert len(res.documents) > 0
    # Section 40 BNSS (Arrest by private person) should be retrieved
    assert any(d.section_number == "40" for d in res.documents)


def test_pipeline_dense_only_mode(pipeline):
    """Verify explicit dense retrieval mode."""
    res = pipeline.retrieve("maximum period of police custody during investigation", mode="dense", top_k=5)
    assert res.mode == "dense_only"
    assert not res.is_empty
    assert len(res.documents) == 5
    assert res.documents[0].dense_score is not None


def test_pipeline_bm25_only_mode(pipeline):
    """Verify explicit BM25 retrieval mode."""
    res = pipeline.retrieve("audio-video electronic means search", mode="bm25", top_k=5)
    assert res.mode == "bm25_only"
    assert not res.is_empty
    assert len(res.documents) > 0
    assert res.documents[0].bm25_score is not None


def test_pipeline_metadata_filtering(pipeline):
    """Verify metadata filtering across hybrid retrieval."""
    # Filter for BNSS Chapter V (Arrest)
    chap_filter = RetrievalFilter(act_short="BNSS", chapter="V")
    res = pipeline.retrieve("arrest procedure", mode="hybrid", top_k=5, filters=chap_filter)
    assert not res.is_empty
    for doc in res.documents:
        assert doc.act_short == "BNSS"
        assert doc.chapter == "V"


def test_pipeline_empty_query(pipeline):
    """Verify empty query handling."""
    res = pipeline.retrieve("", mode="auto")
    assert res.is_empty
    assert res.total_retrieved == 0
    assert res.documents == []
