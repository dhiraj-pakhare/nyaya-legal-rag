"""Master Pipeline for User-Document Ingestion, Intent Routing, Scoped Retrieval, and Safe Generation."""

import logging
import time
import uuid
from typing import List, Optional

from backend.app.core.config import settings
from backend.app.core.qdrant_repo import NYAYA_NAMESPACE
from backend.app.document_rag.citation_validator import DualCitationValidator
from backend.app.document_rag.context_builder import MultiSourceContextBuilder
from backend.app.document_rag.models import (
    DocumentIngestionResult,
    IngestionStatus,
    QueryIntent,
    UserDocument,
    UserDocumentChunk,
    UserDocumentSessionScope,
)
from backend.app.document_rag.pdf_extractor import UserPDFExtractor
from backend.app.document_rag.chunker import UserDocumentChunker
from backend.app.document_rag.repository import UserDocumentRepository
from backend.app.document_rag.retriever import UserDocumentRetriever
from backend.app.document_rag.router import QueryIntentRouter
from backend.app.core.embeddings import EmbeddingModel, get_embedding_model
from backend.app.generation.models import GenerationTelemetry, LegalAnswerResponse, LLMMessage
from backend.app.generation.prompt import (
    SYSTEM_PROMPT,
    REGENERATION_SYSTEM_PROMPT,
    build_generation_messages,
    build_regeneration_messages,
)
from backend.app.generation.providers import LLMProvider, get_llm_provider
from backend.app.ingestion.models import StatutoryChunk
from backend.app.retrieval.models import RetrievedDocument
from backend.app.retrieval.pipeline import HybridRetrievalPipeline

logger = logging.getLogger("nyaya.document_rag.pipeline")


class UserDocumentRAGPipeline:
    """Master orchestrator for user document ingestion, scoped retrieval, and citation-grounded generation."""

    def __init__(
        self,
        repository: Optional[UserDocumentRepository] = None,
        statutory_pipeline: Optional[HybridRetrievalPipeline] = None,
        document_retriever: Optional[UserDocumentRetriever] = None,
        llm_provider: Optional[LLMProvider] = None,
        pdf_extractor: Optional[UserPDFExtractor] = None,
        chunker: Optional[UserDocumentChunker] = None,
        embedding_model: Optional[EmbeddingModel] = None,
        router: Optional[QueryIntentRouter] = None,
        context_builder: Optional[MultiSourceContextBuilder] = None,
        citation_validator: Optional[DualCitationValidator] = None
    ):
        from backend.app.document_rag.repository import get_user_doc_repository
        from backend.app.retrieval.pipeline import get_hybrid_retrieval_pipeline
        self.repository = repository or get_user_doc_repository()
        self.statutory_pipeline = statutory_pipeline or get_hybrid_retrieval_pipeline()
        self.embedding_model = embedding_model or get_embedding_model()
        self.document_retriever = document_retriever or UserDocumentRetriever(
            repository=self.repository,
            embedding_model=self.embedding_model
        )
        self.llm = llm_provider or get_llm_provider()
        self.pdf_extractor = pdf_extractor or UserPDFExtractor()
        self.chunker = chunker or UserDocumentChunker()
        self.router = router or QueryIntentRouter()
        self.context_builder = context_builder or MultiSourceContextBuilder()
        self.citation_validator = citation_validator or DualCitationValidator()

    def ingest_pdf(
        self,
        file_bytes: bytes,
        filename: str,
        scope: UserDocumentSessionScope
    ) -> DocumentIngestionResult:
        """Ingest, validate, chunk, embed, and index an uploaded user PDF."""
        start_time = time.perf_counter()
        scope.validate_scope()

        # 1. Validation & Hash computation
        file_hash = self.pdf_extractor.compute_sha256(file_bytes)
        doc_id = str(uuid.uuid5(NYAYA_NAMESPACE, f"nyaya://doc/{scope.user_id}/{file_hash}"))

        # 2. Idempotent Deduplication Check
        try:
            existing_doc = self.repository.get_document(doc_id, scope)
            if existing_doc.status == IngestionStatus.READY:
                latency_ms = (time.perf_counter() - start_time) * 1000.0
                logger.info(f"Document '{doc_id}' already indexed for user '{scope.user_id}'. Returning existing.")
                return DocumentIngestionResult(
                    document=existing_doc,
                    chunks_count=existing_doc.indexed_chunks_count,
                    is_deduplicated=True,
                    latency_ms=latency_ms
                )
        except Exception:
            pass  # Document not yet present in repository

        # 3. Extraction & Bounded OCR
        extracted_pages, has_ocr = self.pdf_extractor.extract(file_bytes)

        # 4. Chunking
        chunks = self.chunker.chunk_document(
            pages=extracted_pages,
            document_id=doc_id,
            user_id=scope.user_id,
            filename=filename,
            session_id=scope.session_id
        )

        if not chunks:
            raise ValueError(f"No textual chunks could be extracted from '{filename}'.")

        # 5. Embedding
        chunk_texts = [c.text for c in chunks]
        vectors = self.embedding_model.embed_documents(chunk_texts)

        # 6. Isolated Indexing
        indexed_count = self.repository.upsert_user_chunks(
            chunks=chunks,
            vectors=vectors,
            scope=scope
        )

        # 7. Document Registration
        user_doc = UserDocument(
            document_id=doc_id,
            user_id=scope.user_id,
            session_id=scope.session_id,
            filename=filename,
            file_hash=file_hash,
            file_size_bytes=len(file_bytes),
            page_count=len(extracted_pages),
            status=IngestionStatus.READY,
            has_ocr_applied=has_ocr,
            indexed_chunks_count=indexed_count
        )
        self.repository.register_document(user_doc, scope)

        # Invalidate BM25 cache for this scope
        self.document_retriever.bm25_manager.invalidate(scope)

        latency_ms = (time.perf_counter() - start_time) * 1000.0
        return DocumentIngestionResult(
            document=user_doc,
            chunks_count=indexed_count,
            is_deduplicated=False,
            latency_ms=latency_ms
        )

    def delete_document(self, document_id: str, scope: UserDocumentSessionScope) -> int:
        """Purge all document chunks and invalidate worker cache."""
        scope.validate_scope()
        deleted_count = self.repository.delete_document(document_id, scope)
        self.document_retriever.bm25_manager.invalidate(scope)
        return deleted_count

    def query(
        self,
        query_text: str,
        scope: UserDocumentSessionScope,
        top_k: int = 5
    ) -> LegalAnswerResponse:
        """Execute full end-to-end question answering over statutory and/or user document evidence."""
        total_start = time.perf_counter()
        scope.validate_scope()

        # 1. Intent Classification
        routing = self.router.route(query_text, scope)
        logger.info(f"Query routed as '{routing.intent.value}' (confidence={routing.confidence:.2f})")

        statutory_evidence: List[RetrievedDocument] = []
        document_evidence: List[UserDocumentChunk] = []
        retrieval_start = time.perf_counter()

        if routing.intent == QueryIntent.STATUTORY_ONLY:
            retrieval_res = self.statutory_pipeline.retrieve(query_text, top_k=top_k)
            statutory_evidence = retrieval_res.documents
            conf_data = retrieval_res.confidence or {}
            confidence_score = conf_data.get("confidence_score", 0.0)
            is_refused = retrieval_res.is_refused
            refusal_reason = conf_data.get("reason") if is_refused else None

        elif routing.intent == QueryIntent.DOCUMENT_ONLY:
            document_evidence = self.document_retriever.retrieve(query_text, scope=scope, top_k=top_k)
            if not document_evidence:
                confidence_score = 0.0
                is_refused = True
                refusal_reason = "No relevant information found in the uploaded document."
            else:
                confidence_score = max((c.score for c in document_evidence), default=1.0)
                is_refused = False
                refusal_reason = None

        else:  # COMBINED
            # Parallel/Dual Retrieval
            stat_res = self.statutory_pipeline.retrieve(query_text, top_k=top_k)
            doc_res = self.document_retriever.retrieve(query_text, scope=scope, top_k=top_k)

            statutory_evidence = [] if stat_res.is_refused else stat_res.documents
            document_evidence = doc_res

            if not statutory_evidence and not document_evidence:
                confidence_score = 0.0
                is_refused = True
                refusal_reason = "Neither statutory law nor user document contained relevant evidence."
            else:
                top_stat_score = (stat_res.confidence or {}).get("confidence_score", 0.0) if statutory_evidence else 0.0
                top_doc_score = max((c.score for c in document_evidence), default=0.0)
                confidence_score = max(top_stat_score, top_doc_score)
                if not document_evidence and top_stat_score < 0.35:
                    is_refused = True
                    refusal_reason = "No relevant document evidence found and statutory confidence is low."
                elif not statutory_evidence and top_doc_score < 0.1:
                    is_refused = True
                    refusal_reason = "Insufficient document evidence relevance."
                else:
                    is_refused = False
                    refusal_reason = None

        retrieval_latency_ms = (time.perf_counter() - retrieval_start) * 1000.0

        # Phase 4/6 Security Gate: Refuse early without calling LLM
        if is_refused:
            total_latency_ms = (time.perf_counter() - total_start) * 1000.0
            return LegalAnswerResponse(
                query=query_text,
                status="REFUSED",
                answer=None,
                citations=[],
                is_refused=True,
                refusal_reason=refusal_reason or "Low confidence retrieval refusal.",
                confidence={"confidence_score": confidence_score, "reason": refusal_reason},
                telemetry=GenerationTelemetry(
                    retrieval_latency_ms=retrieval_latency_ms,
                    generation_latency_ms=0.0,
                    validation_latency_ms=0.0,
                    total_latency_ms=total_latency_ms,
                    prompt_tokens=0,
                    completion_tokens=0,
                    total_tokens=0,
                    model=getattr(self.llm, "model", "mock"),
                    provider=getattr(self.llm, "provider_name", "mock")
                )
            )

        # 2. Build Multi-Source Context
        context_str = self.context_builder.build_context(statutory_evidence, document_evidence)
        messages = build_generation_messages(query_text, context_str)

        # 3. LLM Generation - Attempt 1
        gen_start = time.perf_counter()
        llm_resp = self.llm.generate(messages)
        gen_latency_ms = (time.perf_counter() - gen_start) * 1000.0

        # 4. AST Dual-Citation Validation - Attempt 1
        val_start = time.perf_counter()
        val_status = self.citation_validator.validate(
            llm_resp.content,
            statutory_evidence,
            document_evidence
        )
        val_latency_ms = (time.perf_counter() - val_start) * 1000.0

        regeneration_attempted = False

        # 5. Controlled 1-Time Regeneration Pass if Validation Fails
        if not val_status.is_valid:
            logger.warning(f"Citation validation failed on Attempt 1: {val_status.failure_reasons}. Triggering regeneration.")
            regen_messages = build_regeneration_messages(
                query=query_text,
                context_str=context_str,
                invalid_answer=llm_resp.content,
                failure_reasons=val_status.failure_reasons
            )
            regen_start = time.perf_counter()
            regen_resp = self.llm.generate(regen_messages)
            gen_latency_ms += (time.perf_counter() - regen_start) * 1000.0

            regen_val_start = time.perf_counter()
            val_status = self.citation_validator.validate(
                regen_resp.content,
                statutory_evidence,
                document_evidence
            )
            val_latency_ms += (time.perf_counter() - regen_val_start) * 1000.0
            regeneration_attempted = True
            val_status.regeneration_attempted = True
            llm_resp = regen_resp

        # 6. Final Decision
        total_latency_ms = (time.perf_counter() - total_start) * 1000.0
        prompt_tokens = llm_resp.prompt_tokens
        completion_tokens = llm_resp.completion_tokens

        if not val_status.is_valid:
            return LegalAnswerResponse(
                query=query_text,
                status="VALIDATION_FAILED",
                answer=None,
                citations=[],
                is_refused=True,
                refusal_reason=f"Citation validation failed: {'; '.join(val_status.failure_reasons)}",
                confidence={"confidence_score": confidence_score},
                validation_status=val_status,
                telemetry=GenerationTelemetry(
                    retrieval_latency_ms=retrieval_latency_ms,
                    generation_latency_ms=gen_latency_ms,
                    validation_latency_ms=val_latency_ms,
                    total_latency_ms=total_latency_ms,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=prompt_tokens + completion_tokens if prompt_tokens is not None and completion_tokens is not None else None,
                    model=getattr(self.llm, "model", "mock"),
                    provider=getattr(self.llm, "provider_name", "mock")
                )
            )

        routed_corpus_name = "USER_DOCUMENT" if routing.intent == QueryIntent.DOCUMENT_ONLY else (
            "COMBINED" if routing.intent == QueryIntent.COMBINED else "STATUTORY"
        )

        return LegalAnswerResponse(
            query=query_text,
            status="SUCCESS",
            answer=llm_resp.content,
            citations=val_status.verified_citations,
            is_refused=False,
            refusal_reason=None,
            confidence={"confidence_score": confidence_score},
            validation_status=val_status,
            retrieval_metadata={"routed_corpus": routed_corpus_name},
            telemetry=GenerationTelemetry(
                retrieval_latency_ms=retrieval_latency_ms,
                generation_latency_ms=gen_latency_ms,
                validation_latency_ms=val_latency_ms,
                total_latency_ms=total_latency_ms,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens if prompt_tokens is not None and completion_tokens is not None else None,
                model=getattr(self.llm, "model", "mock"),
                provider=getattr(self.llm, "provider_name", "mock")
            )
        )


_GLOBAL_USER_DOC_RAG_PIPELINE: Optional[UserDocumentRAGPipeline] = None


def get_user_doc_rag_pipeline() -> UserDocumentRAGPipeline:
    """Get or initialize the global singleton UserDocumentRAGPipeline instance."""
    global _GLOBAL_USER_DOC_RAG_PIPELINE
    if _GLOBAL_USER_DOC_RAG_PIPELINE is None:
        _GLOBAL_USER_DOC_RAG_PIPELINE = UserDocumentRAGPipeline()
    return _GLOBAL_USER_DOC_RAG_PIPELINE

