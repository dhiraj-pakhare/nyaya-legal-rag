"""Synchronous TestClient Adapter using httpx.AsyncClient & ASGITransport for Phase 8 API Tests."""

import asyncio
from typing import Any, Dict, Optional
import httpx
from fastapi import FastAPI

from backend.app.core.embeddings import get_embedding_model
from backend.app.core.qdrant_repo import QdrantRepository
from backend.app.document_rag.pipeline import UserDocumentRAGPipeline
from backend.app.document_rag.repository import UserDocumentRepository
from backend.app.forms.pipeline import StatutoryFormPipeline
from backend.app.forms.repository import get_form_registry
from backend.app.generation.generator import StatutoryGenerationPipeline
from backend.app.generation.providers import MockLLMProvider
from backend.app.ingestion.models import StatutoryChunk
from backend.app.retrieval.pipeline import HybridRetrievalPipeline
from backend.app.services.diagnostics_service import DiagnosticsService
from backend.app.services.document_service import DocumentManagementService
from backend.app.services.forms_service import StatutoryFormsService
from backend.app.services.query_service import LegalQueryService


class TestAPIClient:
    """Synchronous test client wrapping httpx.AsyncClient with ASGITransport."""
    __test__ = False  # Suppress pytest test-class collection warning

    def __init__(self, app: FastAPI):
        self.app = app
        self._transport = httpx.ASGITransport(app=app)
        self._async_client = httpx.AsyncClient(
            transport=self._transport,
            base_url="http://testserver"
        )
        self._loop = asyncio.new_event_loop()

    def get(self, url: str, **kwargs) -> httpx.Response:
        return self._loop.run_until_complete(self._async_client.get(url, **kwargs))

    def post(self, url: str, **kwargs) -> httpx.Response:
        return self._loop.run_until_complete(self._async_client.post(url, **kwargs))

    def delete(self, url: str, **kwargs) -> httpx.Response:
        return self._loop.run_until_complete(self._async_client.delete(url, **kwargs))

    def put(self, url: str, **kwargs) -> httpx.Response:
        return self._loop.run_until_complete(self._async_client.put(url, **kwargs))


def create_in_memory_test_services(mock_llm: Optional[MockLLMProvider] = None):
    """Build isolated in-memory pipelines and services for API testing."""
    llm = mock_llm or MockLLMProvider()
    embed_model = get_embedding_model()

    # 1. Statutory in-memory chunks and retrieval
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
        )
    ]
    stat_repo = QdrantRepository(in_memory=True, collection_name="test_api_stat_docs")
    stat_vecs = embed_model.embed_documents([c.text for c in stat_chunks])
    stat_repo.upsert_chunks(stat_chunks, stat_vecs)
    stat_retrieval = HybridRetrievalPipeline(chunks=stat_chunks, qdrant_repo=stat_repo, embedding_model=embed_model)

    # 2. Statutory Generation Pipeline
    stat_gen = StatutoryGenerationPipeline(retrieval_pipeline=stat_retrieval, llm_provider=llm)

    # 3. User Document Pipeline
    user_doc_repo = UserDocumentRepository(in_memory=True, collection_name="test_api_user_docs")
    user_doc_pipeline = UserDocumentRAGPipeline(
        repository=user_doc_repo,
        statutory_pipeline=stat_retrieval,
        llm_provider=llm,
        embedding_model=embed_model
    )

    # 4. Statutory Forms Pipeline
    forms_registry = get_form_registry()
    forms_pipeline = StatutoryFormPipeline(registry=forms_registry, llm_provider=llm)

    # 5. Application Services
    q_service = LegalQueryService(
        statutory_pipeline=stat_gen,
        user_doc_pipeline=user_doc_pipeline,
        forms_pipeline=forms_pipeline
    )
    doc_service = DocumentManagementService(
        rag_pipeline=user_doc_pipeline,
        repository=user_doc_repo
    )
    forms_service = StatutoryFormsService(registry=forms_registry)
    diag_service = DiagnosticsService(
        qdrant_repo=stat_repo,
        forms_registry=forms_registry,
        embedding_model=embed_model,
        llm_provider=llm
    )

    # Wire singletons
    import backend.app.services.query_service as qs_mod
    import backend.app.services.document_service as ds_mod
    import backend.app.services.forms_service as fs_mod
    import backend.app.services.diagnostics_service as diag_mod

    qs_mod._query_service_instance = q_service
    ds_mod._document_service_instance = doc_service
    fs_mod._forms_service_instance = forms_service
    diag_mod._diagnostics_service_instance = diag_service

    return q_service, doc_service, forms_service, diag_service, llm
