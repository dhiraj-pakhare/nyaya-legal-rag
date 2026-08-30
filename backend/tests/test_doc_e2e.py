"""End-to-End Tests for User Document RAG, Isolation, Dual Citations, and Safe Regeneration."""

import pytest
from backend.app.document_rag.models import (
    DocumentNotFoundError,
    UserDocumentSessionScope,
)
from backend.app.document_rag.pipeline import UserDocumentRAGPipeline
from backend.app.document_rag.repository import UserDocumentRepository
from backend.app.generation.providers import MockLLMProvider
from backend.tests.doc_test_helpers import create_test_pdf_bytes


from backend.app.core.qdrant_repo import QdrantRepository
from backend.app.ingestion.models import StatutoryChunk
from backend.app.retrieval.pipeline import HybridRetrievalPipeline


@pytest.fixture
def mock_pipeline():
    """Create a fully isolated in-memory UserDocumentRAGPipeline with MockLLMProvider."""
    repo = UserDocumentRepository(in_memory=True, collection_name="test_e2e_user_docs")
    mock_llm = MockLLMProvider()

    stat_chunks = [
        StatutoryChunk(
            chunk_id="BNS_s103_p158",
            act="Bharatiya Nyaya Sanhita, 2023",
            act_short="BNS",
            chapter="Chapter VI",
            chapter_title="Of Offences Affecting the Human Body",
            section_number="103",
            section_title="Punishment for murder",
            text="103. (1) Whoever commits murder shall be punished with death or imprisonment for life, and shall also be liable to fine.",
            pages="158",
            page_start=158,
            page_end=158
        ),
        StatutoryChunk(
            chunk_id="BNS_s309_p165",
            act="Bharatiya Nyaya Sanhita, 2023",
            act_short="BNS",
            chapter="Chapter XVII",
            chapter_title="Of Offences Against Property",
            section_number="309",
            section_title="Robbery",
            text="309. (1) In all robbery there is either theft or extortion. Theft is robbery if the offender causes or attempts to cause fear of instant death.",
            pages="165",
            page_start=165,
            page_end=165
        )
    ]
    stat_repo = QdrantRepository(in_memory=True, collection_name="test_e2e_stat_docs")
    from backend.app.core.embeddings import get_embedding_model
    embed_model = get_embedding_model()
    stat_vecs = embed_model.embed_documents([c.text for c in stat_chunks])
    stat_repo.upsert_chunks(stat_chunks, stat_vecs)

    stat_pipeline = HybridRetrievalPipeline(chunks=stat_chunks, qdrant_repo=stat_repo, embedding_model=embed_model)

    pipeline = UserDocumentRAGPipeline(
        repository=repo,
        statutory_pipeline=stat_pipeline,
        embedding_model=embed_model,
        llm_provider=mock_llm
    )
    return pipeline, mock_llm, repo


def test_scenario_a_document_only_query(mock_pipeline):
    """Scenario A: Ingest PDF and ask document-only question; verify [DOC p.X] citation."""
    pipeline, mock_llm, repo = mock_pipeline
    scope = UserDocumentSessionScope(user_id="user_alice")

    pages = [
        "First Information Report\nComplainant: Sunita Sharma\nAccused: Ramesh Verma\nOffence: Extortion of money.",
        "Page 2:\nThe accused threatened the complainant with physical harm on 5th July."
    ]
    pdf_bytes = create_test_pdf_bytes(pages)
    ingest_res = pipeline.ingest_pdf(pdf_bytes, filename="fir_notice.pdf", scope=scope)

    assert ingest_res.chunks_count >= 2
    doc_id = ingest_res.document.document_id

    # Attach document to active scope
    scope.active_document_ids = [doc_id]

    # Configure mock LLM response with valid [DOC p.1] citation
    mock_llm.set_canned_response(
        "According to the FIR [DOC p.1], the complainant is Sunita Sharma and the accused is Ramesh Verma."
    )

    resp = pipeline.query("Who is the complainant named in my FIR?", scope=scope)

    assert resp.status == "SUCCESS"
    assert resp.answer is not None
    assert len(resp.verified_citations) == 1
    assert resp.verified_citations[0].citation_tag == "[DOC p.1]"
    assert resp.verified_citations[0].is_valid is True


def test_scenario_b_statutory_only_query(mock_pipeline):
    """Scenario B: Statutory query executed through the pipeline; verify [BNS s.X] citation."""
    pipeline, mock_llm, repo = mock_pipeline
    scope = UserDocumentSessionScope(user_id="user_alice", active_document_ids=[])

    mock_llm.set_canned_response(
        "Under [BNS s.103(1)], whoever commits murder shall be punished with death or imprisonment for life."
    )

    resp = pipeline.query("What is the punishment for murder under section 103 BNS?", scope=scope)

    assert resp.status == "SUCCESS"
    assert resp.answer is not None
    assert len(resp.verified_citations) == 1
    assert "BNS" in resp.verified_citations[0].act or resp.verified_citations[0].act_short == "BNS"
    assert resp.verified_citations[0].section.startswith("103")


def test_scenario_c_combined_query(mock_pipeline):
    """Scenario C: Combined query; verify both [DOC p.X] and [BNS s.X] citations."""
    pipeline, mock_llm, repo = mock_pipeline
    scope = UserDocumentSessionScope(user_id="user_alice")

    pages = [
        "Complaint Details:\nThe accused forcibly took the complainant's gold chain by putting him in fear of instant death."
    ]
    pdf_bytes = create_test_pdf_bytes(pages)
    ingest_res = pipeline.ingest_pdf(pdf_bytes, filename="robbery_complaint.pdf", scope=scope)
    scope.active_document_ids = [ingest_res.document.document_id]

    mock_llm.set_canned_response(
        "The complaint alleges that the accused took gold by inducing fear of instant death [DOC p.1]. "
        "Under [BNS s.309(1)], robbery is committed when theft involves causing or attempting to cause fear of instant death."
    )

    resp = pipeline.query("Does the incident in my complaint constitute robbery under BNS?", scope=scope)

    assert resp.status == "SUCCESS"
    assert len(resp.verified_citations) == 2
    tags = [c.citation_tag for c in resp.verified_citations]
    assert "[DOC p.1]" in tags
    assert "[BNS s.309(1)]" in tags


def test_scenario_d_cross_tenant_isolation(mock_pipeline):
    """Scenario D: Verify User A cannot retrieve User B's document."""
    pipeline, mock_llm, repo = mock_pipeline

    scope_bob = UserDocumentSessionScope(user_id="user_bob")
    pages_bob = ["Confidential trade secrets and proprietary formulas of Bob Corp."]
    pdf_bytes = create_test_pdf_bytes(pages_bob)
    ingest_res = pipeline.ingest_pdf(pdf_bytes, filename="bob_secrets.pdf", scope=scope_bob)
    doc_id_bob = ingest_res.document.document_id

    # User Alice tries to query Bob's document ID
    scope_alice = UserDocumentSessionScope(user_id="user_alice", active_document_ids=[doc_id_bob])
    resp = pipeline.query("What are the proprietary formulas?", scope=scope_alice)

    # Retrieval should be refused or return zero document evidence
    assert resp.is_refused is True
    assert mock_llm.call_history == []  # Zero LLM token waste


def test_scenario_e_idempotent_deduplication(mock_pipeline):
    """Scenario E: Ingesting the same file twice returns existing document_id with is_deduplicated=True."""
    pipeline, mock_llm, repo = mock_pipeline
    scope = UserDocumentSessionScope(user_id="user_alice")

    pages = ["Notice of default under loan agreement."]
    pdf_bytes = create_test_pdf_bytes(pages)

    res1 = pipeline.ingest_pdf(pdf_bytes, filename="loan.pdf", scope=scope)
    assert res1.is_deduplicated is False

    res2 = pipeline.ingest_pdf(pdf_bytes, filename="loan.pdf", scope=scope)
    assert res2.is_deduplicated is True
    assert res1.document.document_id == res2.document.document_id


def test_scenario_f_document_deletion(mock_pipeline):
    """Scenario F: Deleting a document removes it from retrieval."""
    pipeline, mock_llm, repo = mock_pipeline
    scope = UserDocumentSessionScope(user_id="user_alice")

    pages = ["Temporary document for deletion."]
    pdf_bytes = create_test_pdf_bytes(pages)
    ingest_res = pipeline.ingest_pdf(pdf_bytes, filename="temp.pdf", scope=scope)
    doc_id = ingest_res.document.document_id

    deleted = pipeline.delete_document(doc_id, scope=scope)
    assert deleted >= 1

    # Attempting to delete again raises DocumentNotFoundError (uniform 404)
    with pytest.raises(DocumentNotFoundError):
        pipeline.delete_document(doc_id, scope=scope)


def test_scenario_g_hallucinated_citation_triggers_regeneration(mock_pipeline):
    """Scenario G: LLM hallucinates document page on Attempt 1, regenerates correctly on Attempt 2."""
    pipeline, mock_llm, repo = mock_pipeline
    scope = UserDocumentSessionScope(user_id="user_alice")

    pages = ["Page 1: Complainant details."]
    pdf_bytes = create_test_pdf_bytes(pages)
    ingest_res = pipeline.ingest_pdf(pdf_bytes, filename="fir.pdf", scope=scope)
    scope.active_document_ids = [ingest_res.document.document_id]

    # Queue Attempt 1 (invalid page 99) and Attempt 2 (valid page 1)
    mock_llm.set_response_queue([
        "The complainant is mentioned in [DOC p.99].",
        "The complainant is mentioned in [DOC p.1]."
    ])

    resp = pipeline.query("Who is the complainant?", scope=scope)

    assert resp.status == "SUCCESS"
    assert resp.regeneration_attempted is True
    assert len(resp.verified_citations) == 1
    assert resp.verified_citations[0].citation_tag == "[DOC p.1]"
    assert len(mock_llm.call_history) == 2


def test_scenario_h_double_invalid_citation_refusal(mock_pipeline):
    """Scenario H: LLM produces invalid citations on both attempts -> returns clean refusal."""
    pipeline, mock_llm, repo = mock_pipeline
    scope = UserDocumentSessionScope(user_id="user_alice")

    pages = [
        "First Information Report Details:\nThe complainant reported that the accused stole a laptop from the office on 10th August."
    ]
    pdf_bytes = create_test_pdf_bytes(pages)
    ingest_res = pipeline.ingest_pdf(pdf_bytes, filename="fir_theft.pdf", scope=scope)
    scope.active_document_ids = [ingest_res.document.document_id]

    # Both attempts hallucinate non-existent pages
    mock_llm.set_response_queue([
        "The stolen laptop details are in [DOC p.88].",
        "The stolen laptop details are in [DOC p.99]."
    ])

    resp = pipeline.query("What was stolen according to the complainant in my FIR?", scope=scope)

    assert resp.status == "VALIDATION_FAILED"
    assert resp.is_refused is True
    assert resp.answer is None  # Zero unvalidated output exposed
    assert resp.regeneration_attempted is True
    assert len(mock_llm.call_history) == 2
