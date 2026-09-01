"""Integration tests for controlled single-attempt regeneration flow."""

from backend.app.generation.generator import StatutoryGenerationPipeline
from backend.app.generation.providers import MockLLMProvider
from backend.app.retrieval.models import RetrievalResult, RetrievedDocument


def create_retrieved_doc() -> RetrievedDocument:
    return RetrievedDocument(
        chunk_id="BNS_s103_p1",
        act="Bharatiya Nyaya Sanhita, 2023",
        act_short="BNS",
        chapter="VI",
        chapter_title="OFFENCES AGAINST HUMAN BODY",
        section_number="103",
        section_title="Punishment for murder",
        subsection="(1)",
        text="103. (1) Whoever commits murder shall be punished with death or imprisonment for life, and shall also be liable to fine.",
        page_start=40,
        page_end=41,
        score=0.95,
        final_rank=1
    )


def test_regeneration_first_invalid_second_valid():
    """Test that when the first output fails validation, the second output succeeds and is returned."""
    mock_llm = MockLLMProvider()
    mock_llm.set_responses([
        "The punishment for murder is under [BNS s.999].",  # Invalid (hallucinated section 999)
        "Whoever commits murder shall be punished with death or imprisonment for life [BNS s.103(1)]."  # Valid!
    ])

    pipeline = StatutoryGenerationPipeline(llm_provider=mock_llm)
    
    retrieval_result = RetrievalResult(
        query="What is the punishment for murder?",
        mode="exact_lookup",
        documents=[create_retrieved_doc()],
        total_retrieved=1,
        latency_ms=5.0,
        is_refused=False
    )

    resp = pipeline.generate(
        query="What is the punishment for murder?",
        retrieval_result=retrieval_result
    )

    assert resp.status == "SUCCESS"
    assert resp.answer is not None
    assert "[BNS s.103(1)]" in resp.answer
    assert resp.validation_status is not None
    assert resp.validation_status.is_valid is True
    assert resp.validation_status.regeneration_attempted is True
    assert len(mock_llm.call_history) == 2


def test_regeneration_first_invalid_second_invalid_refuses():
    """Test that when both attempts fail validation, the system cleanly refuses and NEVER returns hallucinated law."""
    mock_llm = MockLLMProvider()
    mock_llm.set_responses([
        "The punishment is defined in [BNS s.999].",  # Invalid attempt 1
        "The offence is governed by [BNS s.888]."   # Invalid attempt 2
    ])

    pipeline = StatutoryGenerationPipeline(llm_provider=mock_llm)
    
    retrieval_result = RetrievalResult(
        query="What is the punishment for murder?",
        mode="exact_lookup",
        documents=[create_retrieved_doc()],
        total_retrieved=1,
        latency_ms=5.0,
        is_refused=False
    )

    resp = pipeline.generate(
        query="What is the punishment for murder?",
        retrieval_result=retrieval_result
    )

    assert resp.status == "VALIDATION_FAILED"
    assert resp.answer is None  # MUST NEVER RETURN INVALID ANSWER
    assert resp.is_refused is True
    assert "Citation validation failed" in (resp.refusal_reason or "")
    assert resp.validation_status is not None
    assert resp.validation_status.is_valid is False
    assert resp.validation_status.regeneration_attempted is True
    assert len(mock_llm.call_history) == 2


def test_generation_debug_logging(caplog):
    """Verify diagnostic logging outputs LLM content without altering pipeline behavior."""
    import logging
    mock_llm = MockLLMProvider()
    mock_llm.set_responses([
        "Invalid attempt [BNS s.999].",
        "Whoever commits murder shall be punished [BNS s.103(1)]."
    ])
    pipeline = StatutoryGenerationPipeline(llm_provider=mock_llm)
    retrieval_result = RetrievalResult(
        query="What is the punishment for murder?",
        mode="exact_lookup",
        documents=[create_retrieved_doc()],
        total_retrieved=1,
        latency_ms=5.0,
        is_refused=False
    )

    with caplog.at_level(logging.WARNING, logger="nyaya.generation.generator"):
        resp = pipeline.generate(
            query="What is the punishment for murder?",
            retrieval_result=retrieval_result
        )

    assert resp.status == "SUCCESS"
    assert "DEBUG LLM INITIAL CONTENT: 'Invalid attempt [BNS s.999].'" in caplog.text
    assert "DEBUG LLM REGEN CONTENT: 'Whoever commits murder shall be punished [BNS s.103(1)].'" in caplog.text
