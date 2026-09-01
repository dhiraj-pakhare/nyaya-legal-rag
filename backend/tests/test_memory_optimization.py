"""Tests verifying memory footprint optimizations and lazy pipeline initialization."""

import pytest
from unittest.mock import MagicMock, patch

from backend.app.document_rag.models import UserDocumentSessionScope
from backend.app.api.schemas.query import QueryRequestDTO
from backend.app.services.query_service import LegalQueryService


def test_query_service_lazy_user_doc_pipeline():
    """Verify LegalQueryService does NOT instantiate UserDocumentRAGPipeline on startup or statutory query."""
    mock_statutory = MagicMock()
    mock_telemetry = MagicMock()
    mock_telemetry.prompt_tokens = 10
    mock_telemetry.completion_tokens = 10
    mock_telemetry.model_dump.return_value = {
        "retrieval_latency_ms": 10.0,
        "generation_latency_ms": 10.0,
        "validation_latency_ms": 10.0,
        "total_latency_ms": 30.0,
        "prompt_tokens": 10,
        "completion_tokens": 10,
        "total_tokens": 20,
        "model": "test",
        "provider": "test",
    }
    mock_statutory.generate.return_value = MagicMock(
        status="SUCCESS",
        answer="Section 103 defines punishment for murder.",
        retrieved_documents=[],
        citations=[],
        confidence={"confidence_score": 1.0},
        confidence_score=1.0,
        is_refusal=False,
        refusal_reason=None,
        retrieval_metadata={"routed_corpus": "STATUTORY"},
        telemetry=mock_telemetry,
    )
    mock_forms = MagicMock()
    mock_forms.can_handle.return_value = False

    service = LegalQueryService(
        statutory_pipeline=mock_statutory,
        user_doc_pipeline=None,
        forms_pipeline=mock_forms,
    )

    # _user_doc_pipeline should remain uninstantiated initially
    assert service._user_doc_pipeline is None

    scope = UserDocumentSessionScope(
        user_id="test_user",
        session_id="test_session",
        active_document_ids=[],
    )
    req = QueryRequestDTO(query="What is Section 103 of BNSS?")

    resp = service.execute_query(scope=scope, request=req)
    assert resp.status == "SUCCESS"
    assert resp.routed_corpus == "STATUTORY"
    # Even after pure statutory execution, user_doc_pipeline must not be instantiated
    assert service._user_doc_pipeline is None


def test_pytorch_thread_bounding():
    """Verify PyTorch threads are bounded to at most 2."""
    import torch
    if hasattr(torch, "get_num_threads"):
        assert torch.get_num_threads() <= 2


def test_get_statutory_chunks_batch_streaming():
    """Verify get_statutory_chunks streams batches without accumulating duplicate Record lists."""
    from backend.app.retrieval.pipeline import get_statutory_chunks, _GLOBAL_STATUTORY_CHUNKS
    import backend.app.retrieval.pipeline as pipeline_mod

    mock_record = MagicMock()
    mock_record.payload = {
        "chunk_id": "test-chunk-1",
        "act": "Bharatiya Nyaya Sanhita, 2023",
        "act_short": "BNS",
        "chapter": "VI",
        "chapter_title": "Of Offences Affecting the Human Body",
        "section_number": "103",
        "section_title": "Punishment for murder",
        "subsection": None,
        "clause": None,
        "page_start": 40,
        "page_end": 41,
        "text": "Whoever commits murder shall be punished...",
        "chunk_type": "statutory_section",
        "parent_section": "103",
        "subsections_present": [],
        "schedules_present": [],
        "token_count": 50,
        "char_count": 200,
    }

    # Simulate 1,027 records in 5 batches
    mock_batches = [([mock_record] * 250, "off1"), ([mock_record] * 250, "off2"),
                    ([mock_record] * 250, "off3"), ([mock_record] * 250, "off4"),
                    ([mock_record] * 27, None)]

    mock_repo = MagicMock()
    mock_repo.collection_name = "nyaya_legal_corpus"
    mock_repo.client.scroll.side_effect = mock_batches

    with patch("os.path.exists", return_value=False):
        with patch("backend.app.core.qdrant_repo.get_qdrant_repository", return_value=mock_repo):
            pipeline_mod._GLOBAL_STATUTORY_CHUNKS = None
            chunks = get_statutory_chunks()
            assert len(chunks) == 1027
            assert chunks[0].section_number == "103"
            # Verify batch scroll was called 5 times
            assert mock_repo.client.scroll.call_count == 5
            pipeline_mod._GLOBAL_STATUTORY_CHUNKS = None
