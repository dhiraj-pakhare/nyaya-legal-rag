"""Regression and verification tests for User-Document Query Resolution and Intent Routing.

Covers:
A. READY document exists + document_ids omitted: resolves caller's READY doc IDs and searches chunks
B. Resume query: "What projects have I worked on according to my uploaded resume?" -> DOCUMENT_ONLY -> [DOC p.X]
C. Resume signal: "Summarize my resume" -> DOCUMENT_ONLY
D. CV signal: "What skills are in my CV?" -> DOCUMENT_ONLY
E. Ambiguous query: "What is my name?" -> COMBINED dual-retrieval
F. No READY documents: fallback to statutory corpus
G. Explicit document_ids: respects specified documents, filters unauthorized/unready IDs
H. Cross-user isolation: User A cannot retrieve User B's documents even if requested explicitly
"""

import numpy as np
import pytest

from backend.app.api.schemas.query import QueryRequestDTO
from backend.app.document_rag.models import (
    IngestionStatus,
    QueryIntent,
    UserDocument,
    UserDocumentChunk,
    UserDocumentSessionScope,
)
from backend.app.document_rag.pipeline import UserDocumentRAGPipeline
from backend.app.document_rag.repository import UserDocumentRepository
from backend.app.document_rag.retriever import UserDocumentRetriever
from backend.app.document_rag.router import QueryIntentRouter
from backend.app.generation.generator import StatutoryGenerationPipeline
from backend.app.generation.models import LegalAnswerResponse
from backend.app.generation.providers import MockLLMProvider
from backend.app.retrieval.models import RetrievalResult
from backend.app.services.query_service import LegalQueryService


class DummyStatutoryPipeline:
    """Mock statutory pipeline for testing routing boundaries."""
    def __init__(self, answer: str = "Statutory section answer [BNS s.103]."):
        self.answer = answer

    def retrieve(self, query: str, top_k: int = 5):
        return RetrievalResult(
            query=query,
            mode="hybrid_rrf",
            documents=[],
            confidence={"confidence_score": 0.8, "reason": "Statutory hit"},
            is_refused=False
        )

    def generate(self, query: str) -> LegalAnswerResponse:
        return LegalAnswerResponse(
            query=query,
            status="SUCCESS",
            answer=self.answer,
            citations=[],
            is_refused=False
        )


class MockReranker:
    class Model:
        def predict(self, pairs):
            return [3.0] * len(pairs)
    model = Model()
    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))


def _setup_test_environment():
    """Create isolated in-memory repositories and pipelines."""
    repo = UserDocumentRepository(in_memory=True, collection_name="test_user_doc_query_resolution")
    mock_llm = MockLLMProvider()
    mock_reranker = MockReranker()

    retriever = UserDocumentRetriever(
        repository=repo,
        reranker=mock_reranker
    )

    stat_pipeline = DummyStatutoryPipeline()

    user_doc_pipeline = UserDocumentRAGPipeline(
        repository=repo,
        statutory_pipeline=stat_pipeline,
        document_retriever=retriever,
        llm_provider=mock_llm
    )

    query_service = LegalQueryService(
        statutory_pipeline=stat_pipeline,
        user_doc_pipeline=user_doc_pipeline
    )

    return repo, user_doc_pipeline, query_service, mock_llm


def test_router_signals_resume_and_cv():
    """Verify 'resume' and 'cv' route to DOCUMENT_ONLY, while 'my name' routes to COMBINED."""
    router = QueryIntentRouter()
    scope = UserDocumentSessionScope(user_id="user_alice", active_document_ids=["doc_resume_1"])

    # C. Resume signal
    decision_resume = router.route("Summarize my resume", scope)
    assert decision_resume.intent == QueryIntent.DOCUMENT_ONLY
    assert decision_resume.target_document_ids == ["doc_resume_1"]

    # D. CV signal
    decision_cv = router.route("What skills are in my CV?", scope)
    assert decision_cv.intent == QueryIntent.DOCUMENT_ONLY
    assert decision_cv.target_document_ids == ["doc_resume_1"]

    # B. Resume project query
    decision_proj = router.route("What projects have I worked on according to my uploaded resume?", scope)
    assert decision_proj.intent == QueryIntent.DOCUMENT_ONLY
    assert decision_proj.target_document_ids == ["doc_resume_1"]

    # E. Ambiguous query "What is my name?" -> COMBINED
    decision_name = router.route("What is my name?", scope)
    assert decision_name.intent == QueryIntent.COMBINED
    assert decision_name.target_document_ids == ["doc_resume_1"]

    # Statutory query remains STATUTORY_ONLY
    decision_stat = router.route("What is Section 103 of BNS?", scope)
    assert decision_stat.intent == QueryIntent.STATUTORY_ONLY


def test_query_service_omitted_document_ids_resolves_ready_documents():
    """A & B: Verify query without document_ids automatically resolves READY documents."""
    repo, user_doc_pipeline, query_service, mock_llm = _setup_test_environment()
    scope = UserDocumentSessionScope(user_id="user_alice")

    # Ingest READY resume document for user_alice
    doc_id = "doc_resume_alice"
    user_doc = UserDocument(
        document_id=doc_id,
        user_id="user_alice",
        filename="Dhiraj_Resume.pdf",
        file_hash="hash123",
        file_size_bytes=5000,
        page_count=1,
        status=IngestionStatus.READY,
        indexed_chunks_count=1
    )
    repo.register_document(user_doc, scope)

    chunk = UserDocumentChunk(
        chunk_id=f"{doc_id}_p1_c1",
        document_id=doc_id,
        user_id="user_alice",
        filename="Dhiraj_Resume.pdf",
        page_start=1,
        page_end=1,
        chunk_index=0,
        text="Dhiraj Pakhare. Projects: Built Nyaya Legal RAG, Distributed Ingestion, Qdrant Vector Search.",
        token_count=16
    )
    # 768-dim mock vector
    vec = np.ones((1, 768), dtype=np.float32)
    repo.upsert_user_chunks([chunk], vec, scope=scope)

    # Set mock LLM response citing the document
    mock_llm.set_canned_response("According to [DOC p.1], you worked on Nyaya Legal RAG and Distributed Ingestion.")

    # Execute query with document_ids=None (as sent by frontend ChatView)
    req = QueryRequestDTO(
        query="What projects have I worked on according to my uploaded resume?",
        document_ids=None,
        enable_forms=True
    )

    resp = query_service.execute_query(scope=scope, request=req)

    assert resp.status == "SUCCESS"
    assert resp.is_refused is False
    assert resp.routed_corpus == "USER_DOCUMENT"
    assert resp.answer is not None
    assert "Nyaya Legal RAG" in resp.answer
    assert len(resp.citations) >= 1
    assert resp.citations[0].citation_type.value == "DOCUMENT"
    assert resp.citations[0].citation_text == "[DOC p.1]"


def test_query_service_ambiguous_name_query_resolves_ready_documents():
    """E: Verify ambiguous query 'What is my name?' uses COMBINED route and grounds on document."""
    repo, user_doc_pipeline, query_service, mock_llm = _setup_test_environment()
    scope = UserDocumentSessionScope(user_id="user_alice")

    doc_id = "doc_resume_alice"
    user_doc = UserDocument(
        document_id=doc_id,
        user_id="user_alice",
        filename="Dhiraj_Resume.pdf",
        file_hash="hash123",
        file_size_bytes=5000,
        page_count=1,
        status=IngestionStatus.READY,
        indexed_chunks_count=1
    )
    repo.register_document(user_doc, scope)

    chunk = UserDocumentChunk(
        chunk_id=f"{doc_id}_p1_c1",
        document_id=doc_id,
        user_id="user_alice",
        filename="Dhiraj_Resume.pdf",
        page_start=1,
        page_end=1,
        chunk_index=0,
        text="Name: Dhiraj Pakhare. Full-stack AI Engineer.",
        token_count=8
    )
    vec = np.ones((1, 768), dtype=np.float32)
    repo.upsert_user_chunks([chunk], vec, scope=scope)

    mock_llm.set_canned_response("Based on [DOC p.1], your name is Dhiraj Pakhare.")

    req = QueryRequestDTO(query="What is my name?", document_ids=None)
    resp = query_service.execute_query(scope=scope, request=req)

    assert resp.status == "SUCCESS"
    assert resp.is_refused is False
    assert resp.routed_corpus == "COMBINED"
    assert "Dhiraj Pakhare" in resp.answer
    assert resp.citations[0].citation_text == "[DOC p.1]"


def test_query_service_no_ready_documents_falls_back_to_statutory():
    """F: When user has 0 ready documents, query safely routes to statutory corpus."""
    repo, user_doc_pipeline, query_service, mock_llm = _setup_test_environment()
    scope = UserDocumentSessionScope(user_id="user_empty")

    req = QueryRequestDTO(query="What is Section 103 BNS?", document_ids=None)
    resp = query_service.execute_query(scope=scope, request=req)

    assert resp.status == "SUCCESS"
    assert resp.is_refused is False
    assert resp.routed_corpus == "STATUTORY"
    assert "Statutory section answer" in resp.answer


def test_query_service_explicit_document_ids_restriction():
    """G: When client explicitly specifies document_ids, only matching READY docs are scoped."""
    repo, user_doc_pipeline, query_service, mock_llm = _setup_test_environment()
    scope = UserDocumentSessionScope(user_id="user_alice")

    # Ingest two documents
    doc1 = UserDocument(document_id="doc_1", user_id="user_alice", filename="f1.pdf", file_hash="h1", file_size_bytes=100, page_count=1, status=IngestionStatus.READY)
    doc2 = UserDocument(document_id="doc_2", user_id="user_alice", filename="f2.pdf", file_hash="h2", file_size_bytes=100, page_count=1, status=IngestionStatus.READY)
    repo.register_document(doc1, scope)
    repo.register_document(doc2, scope)

    vec = np.ones((1, 768), dtype=np.float32)
    c1 = UserDocumentChunk(chunk_id="doc_1_p1_c1", document_id="doc_1", user_id="user_alice", filename="f1.pdf", page_start=1, page_end=1, chunk_index=0, text="First doc text.", token_count=4)
    c2 = UserDocumentChunk(chunk_id="doc_2_p1_c1", document_id="doc_2", user_id="user_alice", filename="f2.pdf", page_start=1, page_end=1, chunk_index=0, text="Second doc text.", token_count=4)
    repo.upsert_user_chunks([c1], vec, scope=scope)
    repo.upsert_user_chunks([c2], vec, scope=scope)

    mock_llm.set_canned_response("From doc 2 [DOC p.1], second doc text.")

    # Explicitly request only doc_2
    req = QueryRequestDTO(query="Summarize my resume", document_ids=["doc_2"])
    resp = query_service.execute_query(scope=scope, request=req)

    assert resp.status == "SUCCESS"
    assert resp.citations[0].document_id == "doc_2"


def test_query_service_explicit_empty_document_ids_with_ready_documents():
    """Verify that when document_ids=[] is explicitly supplied, even if the caller has READY
    documents, they MUST NOT be automatically selected, and pure statutory behavior is executed."""
    repo, user_doc_pipeline, query_service, mock_llm = _setup_test_environment()
    scope = UserDocumentSessionScope(user_id="user_alice")

    # Ingest READY document for user_alice
    doc_id = "doc_resume_alice"
    user_doc = UserDocument(
        document_id=doc_id,
        user_id="user_alice",
        filename="Dhiraj_Resume.pdf",
        file_hash="hash123",
        file_size_bytes=5000,
        page_count=1,
        status=IngestionStatus.READY,
        indexed_chunks_count=1
    )
    repo.register_document(user_doc, scope)

    chunk = UserDocumentChunk(
        chunk_id=f"{doc_id}_p1_c1",
        document_id=doc_id,
        user_id="user_alice",
        filename="Dhiraj_Resume.pdf",
        page_start=1,
        page_end=1,
        chunk_index=0,
        text="Dhiraj Pakhare. Projects: Built Nyaya Legal RAG.",
        token_count=8
    )
    vec = np.ones((1, 768), dtype=np.float32)
    repo.upsert_user_chunks([chunk], vec, scope=scope)

    # 1. First verify that when document_ids is None (omitted), the ready doc IS selected:
    req_omitted = QueryRequestDTO(query="Summarize my resume", document_ids=None)
    mock_llm.set_canned_response("According to [DOC p.1], you built Nyaya Legal RAG.")
    resp_omitted = query_service.execute_query(scope=scope, request=req_omitted)
    assert resp_omitted.routed_corpus == "USER_DOCUMENT"
    assert len(resp_omitted.citations) > 0
    assert resp_omitted.citations[0].document_id == doc_id

    # 2. Now verify that when document_ids=[] is explicitly supplied, ready docs are NOT selected:
    req_empty = QueryRequestDTO(query="Summarize my resume", document_ids=[])
    resp_empty = query_service.execute_query(scope=scope, request=req_empty)
    # Pure statutory routing occurs:
    assert resp_empty.routed_corpus == "STATUTORY"
    assert "Statutory section answer" in (resp_empty.answer or "")
    assert not any(c.citation_type.value == "DOCUMENT" for c in resp_empty.citations)


def test_query_service_cross_user_isolation_prevented():
    """H: User B cannot query User A's document even if passing User A's document_id."""
    repo, user_doc_pipeline, query_service, mock_llm = _setup_test_environment()
    scope_a = UserDocumentSessionScope(user_id="user_alice")
    scope_b = UserDocumentSessionScope(user_id="user_bob")

    # User A uploads secret doc
    doc_a = UserDocument(
        document_id="doc_secret_a",
        user_id="user_alice",
        filename="secret.pdf",
        file_hash="h_sec",
        file_size_bytes=100,
        page_count=1,
        status=IngestionStatus.READY
    )
    repo.register_document(doc_a, scope_a)
    c_a = UserDocumentChunk(
        chunk_id="doc_secret_a_p1_c1",
        document_id="doc_secret_a",
        user_id="user_alice",
        filename="secret.pdf",
        page_start=1,
        page_end=1,
        chunk_index=0,
        text="Top secret proprietary plan of Alice.",
        token_count=6
    )
    vec = np.ones((1, 768), dtype=np.float32)
    repo.upsert_user_chunks([c_a], vec, scope=scope_a)

    # User B sends query explicitly requesting doc_secret_a
    req_malicious = QueryRequestDTO(
        query="What is Alice's secret plan?",
        document_ids=["doc_secret_a"]
    )
    # Query executed as User B
    resp = query_service.execute_query(scope=scope_b, request=req_malicious)

    # Ownership check excludes doc_secret_a from active_doc_ids -> has_documents becomes False
    # -> falls back to statutory query, never reveals User A's chunk
    assert resp.routed_corpus == "STATUTORY"
    assert "Alice" not in (resp.answer or "")


def test_end_to_end_resume_ingestion_and_chat_flow():
    """End-to-end integration flow:
    1. Authenticated user uploads resume PDF
    2. Ingestion reaches READY status
    3. Query sent without document_ids
    4. Resume chunks are retrieved
    5. Answer is generated with [DOC p.X] citation
    6. Unrelated statutory query still behaves correctly
    """
    from backend.tests.doc_test_helpers import create_test_pdf_bytes

    repo, user_doc_pipeline, query_service, mock_llm = _setup_test_environment()
    scope = UserDocumentSessionScope(user_id="demo_user")

    # Step 1 & 2: Ingest genuine PDF into repository
    resume_pages = [
        "Curriculum Vitae\nCandidate: Dhiraj Pakhare\nEducation: Computer Engineering\n"
        "Projects:\n- Nyaya Legal RAG Platform\n- High-throughput Vector Indexing Pipeline\n"
        "- Cross-Encoder Reranking Architecture"
    ]
    pdf_bytes = create_test_pdf_bytes(resume_pages)
    ingest_res = user_doc_pipeline.ingest_pdf(
        file_bytes=pdf_bytes,
        filename="DHI_AI_Resume1.pdf",
        scope=scope
    )

    assert ingest_res.document.status == IngestionStatus.READY
    assert ingest_res.chunks_count >= 1
    doc_id = ingest_res.document.document_id

    # Step 3, 4, 5: Query sent WITHOUT document_ids
    mock_llm.set_canned_response(
        "Based on your resume [DOC p.1], you worked on Nyaya Legal RAG Platform, High-throughput Vector Indexing Pipeline, and Cross-Encoder Reranking Architecture."
    )

    req_resume = QueryRequestDTO(
        query="What projects have I worked on according to my uploaded resume?",
        document_ids=None,
        enable_forms=True
    )
    resp_resume = query_service.execute_query(scope=scope, request=req_resume)

    assert resp_resume.status == "SUCCESS"
    assert resp_resume.is_refused is False
    assert resp_resume.routed_corpus == "USER_DOCUMENT"
    assert "Nyaya Legal RAG" in resp_resume.answer
    assert len(resp_resume.citations) >= 1
    assert resp_resume.citations[0].citation_type.value == "DOCUMENT"
    assert resp_resume.citations[0].citation_text == "[DOC p.1]"
    assert resp_resume.citations[0].document_id == doc_id

    # Step 6: Unrelated statutory query executed by same user
    # Set statutory-grounded canned response or clear canned response
    mock_llm.set_canned_response(
        "Section 103 of Bharatiya Nyaya Sanhita, 2023 [BNS s.103(1)] provides punishment for murder."
    )
    # Add Section 103 retrieved document to DummyStatutoryPipeline
    from backend.app.retrieval.models import RetrievedDocument
    stat_pipeline = user_doc_pipeline.statutory_pipeline
    stat_doc = RetrievedDocument(
        chunk_id="bns_s103_p1_c1",
        act="Bharatiya Nyaya Sanhita",
        act_short="BNS",
        chapter="VI",
        chapter_title="Of Offences Affecting Life",
        section_number="103",
        section="103",
        section_title="Punishment for murder",
        page_start=45,
        page_end=45,
        text="103. (1) Whoever commits murder shall be punished with death or imprisonment for life.",
        score=0.95,
        final_rank=1
    )
    stat_pipeline.documents = [stat_doc]
    def mock_retrieve(query, top_k=5):
        return RetrievalResult(
            query=query,
            mode="exact_lookup",
            documents=[stat_doc],
            confidence={"confidence_score": 0.95, "reason": "Exact match"},
            is_refused=False
        )
    stat_pipeline.retrieve = mock_retrieve

    req_statutory = QueryRequestDTO(
        query="What is Section 103 BNS?",
        document_ids=None,
        enable_forms=True
    )
    resp_statutory = query_service.execute_query(scope=scope, request=req_statutory)

    assert resp_statutory.status == "SUCCESS"
    assert resp_statutory.is_refused is False
    assert resp_statutory.routed_corpus == "STATUTORY"
    assert "Section 103" in resp_statutory.answer
    assert any("[BNS s.103" in c.citation_text for c in resp_statutory.citations)

