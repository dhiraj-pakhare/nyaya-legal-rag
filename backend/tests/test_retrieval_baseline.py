"""Unit and baseline retrieval tests for dense similarity search."""

import pytest
import numpy as np

from backend.app.core.embeddings import get_embedding_model
from backend.app.core.embedding_input import format_chunk_for_embedding
from backend.app.core.qdrant_repo import QdrantRepository
from backend.app.core.retrieval_baseline import search_dense
from backend.app.ingestion.parser import StatutoryParser


@pytest.fixture(scope="module")
def populated_test_repo():
    """Populate an in-memory Qdrant repository with a representative subset of chunks."""
    parser = StatutoryParser("BNS bare act 2023.pdf")
    res = parser.parse()
    
    # Pick 20 representative chunks
    sample_secs = ["1", "2", "23", "35", "44", "103", "141", "187", "245", "246", "531"]
    sample_scheds = ["105", "281"]
    
    selected_chunks = [
        c for c in res.chunks
        if (c.chunk_type == "substantive_section" and c.section_number in sample_secs) or
           (c.chunk_type == "schedule_entry" and c.section_number in sample_scheds)
    ]
    
    embed_model = get_embedding_model()
    texts = [format_chunk_for_embedding(c) for c in selected_chunks]
    embeddings = embed_model.embed_documents(texts, batch_size=16)
    
    repo = QdrantRepository(
        collection_name="test_retrieval_corpus",
        vector_dim=768,
        in_memory=True
    )
    repo.upsert_chunks(selected_chunks, embeddings)
    return repo, embed_model


def test_dense_retrieval_exact_section_query(populated_test_repo):
    """Test exact section-oriented query: 'When police may arrest without warrant section 35'."""
    repo, embed_model = populated_test_repo
    results = search_dense(
        query="When police may arrest without warrant section 35",
        repo=repo,
        embedding_model=embed_model,
        top_k=3
    )
    assert len(results) > 0
    top_result = results[0]
    assert top_result.section_number == "35"
    assert top_result.score > 0.60


def test_dense_retrieval_conceptual_legal_query(populated_test_repo):
    """Test conceptual legal query: 'detention and custody period under police investigation'."""
    repo, embed_model = populated_test_repo
    results = search_dense(
        query="detention and custody period under police investigation",
        repo=repo,
        embedding_model=embed_model,
        top_k=3
    )
    assert len(results) > 0
    # Section 187 should be among top results
    retrieved_sections = [r.section_number for r in results]
    assert "187" in retrieved_sections


def test_dense_retrieval_schedule_offence_query(populated_test_repo):
    """Test Schedule I BNS offence query: 'culpable homicide not amounting to murder bailable or non-bailable'."""
    repo, embed_model = populated_test_repo
    results = search_dense(
        query="culpable homicide not amounting to murder bailable or non-bailable",
        repo=repo,
        embedding_model=embed_model,
        top_k=3
    )
    assert len(results) > 0
    retrieved_sections = [r.section_number for r in results]
    assert "105" in retrieved_sections
