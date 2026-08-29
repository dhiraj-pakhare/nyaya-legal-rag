"""Unit and integration tests for BGE-base-en-v1.5 embedding model wrapper."""

import pytest
import numpy as np

from backend.app.core.embeddings import EmbeddingModel, get_embedding_model


@pytest.fixture(scope="module")
def embedding_model():
    return get_embedding_model()


def test_embedding_model_dimension(embedding_model):
    """Verify that the embedding dimension is exactly 768 for BGE-base-en-v1.5."""
    assert embedding_model.dimension == 768
    
    vec = embedding_model.embed_query("test legal query")
    assert isinstance(vec, list)
    assert len(vec) == 768


def test_embedding_normalization(embedding_model):
    """Verify that output embeddings have unit L2 norm."""
    vec = np.array(embedding_model.embed_query("bail provisions under BNSS"))
    norm = np.linalg.norm(vec)
    assert pytest.approx(norm, rel=1e-3) == 1.0


def test_batch_document_embedding(embedding_model):
    """Verify batch document encoding."""
    docs = [
        "BNSS Section 35: When police may arrest without warrant",
        "BNSS Section 187: Procedure when investigation cannot be completed in twenty-four hours",
        "BNS Section 105: Culpable homicide not amounting to murder"
    ]
    embs = embedding_model.embed_documents(docs, batch_size=2)
    assert isinstance(embs, np.ndarray)
    assert embs.shape == (3, 768)
    
    # Norm of all rows should be 1.0
    row_norms = np.linalg.norm(embs, axis=1)
    for rn in row_norms:
        assert pytest.approx(rn, rel=1e-3) == 1.0


def test_query_vs_document_asymmetric_encoding(embedding_model):
    """Verify that query encoding prepends instruction prefix while document encoding does not."""
    text = "powers of arrest without warrant"
    q_vec = np.array(embedding_model.embed_query(text))
    d_vec = embedding_model.embed_documents([text])[0]
    
    # Because query has instruction prefix, q_vec and d_vec should be similar but not identical
    dot_prod = np.dot(q_vec, d_vec)
    assert 0.70 < dot_prod < 0.99
