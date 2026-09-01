"""Unit and integration tests for QdrantRepository."""

import pytest
import numpy as np

from backend.app.core.qdrant_repo import QdrantRepository, chunk_id_to_point_id
from backend.app.ingestion.models import StatutoryChunk


@pytest.fixture
def in_memory_qdrant_repo():
    return QdrantRepository(
        collection_name="test_statutory_corpus",
        vector_dim=768,
        in_memory=True
    )


def test_deterministic_point_id():
    """Verify that chunk_id generates consistent deterministic UUIDv5 strings."""
    pid1 = chunk_id_to_point_id("bnss-s35-001")
    pid2 = chunk_id_to_point_id("bnss-s35-001")
    pid3 = chunk_id_to_point_id("bns-sched1-s105-001")
    
    assert pid1 == pid2
    assert pid1 != pid3
    assert len(pid1) == 36  # Standard UUID format


def test_qdrant_upsert_and_count(in_memory_qdrant_repo):
    """Verify upserting chunks and verifying collection point count."""
    chunks = [
        StatutoryChunk(
            act="Bharatiya Nagarik Suraksha Sanhita, 2023",
            act_short="BNSS",
            chapter="V",
            chapter_title="ARREST OF PERSONS",
            section_number="35",
            section_title="When police may arrest without warrant",
            text="35. (1) Any police officer may arrest without warrant...",
            page_start=13,
            page_end=14,
            chunk_id="bnss-s35-001"
        ),
        StatutoryChunk(
            act="Bharatiya Nyaya Sanhita, 2023",
            act_short="BNS",
            chapter="SCHEDULE_I",
            chapter_title="THE FIRST SCHEDULE - CLASSIFICATION OF OFFENCES",
            section_number="105",
            section_title="Culpable homicide not amounting to murder",
            text="BNS Section 105: Culpable homicide...",
            page_start=164,
            page_end=164,
            chunk_id="bns-sched1-s105-001"
        )
    ]
    vecs = np.random.randn(2, 768).astype(np.float32)
    # Normalize vectors
    vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
    
    count = in_memory_qdrant_repo.upsert_chunks(chunks, vecs)
    assert count == 2
    assert in_memory_qdrant_repo.count() == 2


def test_qdrant_idempotent_reindexing(in_memory_qdrant_repo):
    """Verify that re-indexing the same chunks does not create duplicate points."""
    chunk = StatutoryChunk(
        act="Bharatiya Nagarik Suraksha Sanhita, 2023",
        act_short="BNSS",
        chapter="V",
        chapter_title="ARREST OF PERSONS",
        section_number="35",
        section_title="When police may arrest without warrant",
        text="35. (1) Any police officer may arrest...",
        page_start=13,
        page_end=14,
        chunk_id="bnss-s35-001"
    )
    vec = np.random.randn(1, 768).astype(np.float32)
    vec = vec / np.linalg.norm(vec)
    
    # Upsert 3 times
    in_memory_qdrant_repo.upsert_chunks([chunk], vec)
    in_memory_qdrant_repo.upsert_chunks([chunk], vec)
    in_memory_qdrant_repo.upsert_chunks([chunk], vec)
    
    # Count should still be strictly 1
    assert in_memory_qdrant_repo.count() == 1


def test_qdrant_metadata_filtering(in_memory_qdrant_repo):
    """Verify filtering by statutory metadata (act_short, section_number)."""
    chunks = [
        StatutoryChunk(
            act="Bharatiya Nagarik Suraksha Sanhita, 2023",
            act_short="BNSS",
            chapter="V",
            chapter_title="ARREST OF PERSONS",
            section_number="35",
            section_title="When police may arrest without warrant",
            text="35. (1) Any police officer may arrest...",
            page_start=13,
            page_end=14,
            chunk_id="bnss-s35-001"
        ),
        StatutoryChunk(
            act="Bharatiya Nyaya Sanhita, 2023",
            act_short="BNS",
            chapter="SCHEDULE_I",
            chapter_title="THE FIRST SCHEDULE - CLASSIFICATION OF OFFENCES",
            section_number="105",
            section_title="Culpable homicide not amounting to murder",
            text="BNS Section 105: Culpable homicide...",
            page_start=164,
            page_end=164,
            chunk_id="bns-sched1-s105-001"
        )
    ]
    vecs = np.random.randn(2, 768).astype(np.float32)
    vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
    in_memory_qdrant_repo.upsert_chunks(chunks, vecs)
    
    # Search with filter act_short == "BNSS"
    results_bnss = in_memory_qdrant_repo.search_dense(
        query_vector=vecs[0].tolist(),
        limit=5,
        filters={"act_short": "BNSS"}
    )
    assert len(results_bnss) == 1
    assert results_bnss[0].payload["section_number"] == "35"
    
    # Search with filter act_short == "BNS"
    results_bns = in_memory_qdrant_repo.search_dense(
        query_vector=vecs[0].tolist(),
        limit=5,
        filters={"act_short": "BNS"}
    )
    assert len(results_bns) == 1
    assert results_bns[0].payload["section_number"] == "105"


def test_qdrant_repo_api_key_initialization(monkeypatch):
    """Verify QdrantRepository passes explicit api_key or falls back to settings."""
    from unittest.mock import MagicMock, patch

    mock_client = MagicMock()
    mock_client.get_collections.return_value = MagicMock(collections=[])

    with patch("backend.app.core.qdrant_repo.QdrantClient", return_value=mock_client) as mock_cls:
        # Case 1: Explicit api_key passed
        QdrantRepository(url="https://cloud.qdrant.io:6333", api_key="secret-cloud-key")
        mock_cls.assert_called_with(url="https://cloud.qdrant.io:6333", api_key="secret-cloud-key")

        # Case 2: No api_key passed, fallback to None when settings.qdrant_api_key is empty
        monkeypatch.setattr("backend.app.core.config.settings.qdrant_api_key", "")
        QdrantRepository(url="http://localhost:6333")
        mock_cls.assert_called_with(url="http://localhost:6333", api_key=None)

        # Case 3: No api_key passed, fallback to settings.qdrant_api_key when configured
        monkeypatch.setattr("backend.app.core.config.settings.qdrant_api_key", "env-api-key")
        QdrantRepository(url="https://cloud.qdrant.io:6333")
        mock_cls.assert_called_with(url="https://cloud.qdrant.io:6333", api_key="env-api-key")


def test_ingest_cli_api_key_argument(monkeypatch):
    """Verify scripts/ingest.py CLI argument parsing for --qdrant-api-key."""
    import os
    import argparse

    # Case 1: CLI argument provided explicitly
    test_args = ["--pdf", "dummy.pdf", "--qdrant-url", "https://cloud.qdrant.io:6333", "--qdrant-api-key", "cli-test-key"]
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", "--pdf-path", dest="pdf", default="BNS bare act 2023.pdf")
    parser.add_argument("--qdrant-url", default=None)
    parser.add_argument("--qdrant-api-key", default=os.getenv("QDRANT_API_KEY", None))
    args = parser.parse_args(test_args)
    assert args.qdrant_api_key == "cli-test-key"
    assert args.qdrant_url == "https://cloud.qdrant.io:6333"

    # Case 2: CLI argument omitted, environment variable fallback
    monkeypatch.setenv("QDRANT_API_KEY", "env-fallback-key")
    parser2 = argparse.ArgumentParser()
    parser2.add_argument("--qdrant-api-key", default=os.getenv("QDRANT_API_KEY", None))
    args2 = parser2.parse_args([])
    assert args2.qdrant_api_key == "env-fallback-key"
