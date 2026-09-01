"""Unified Legal Query and Streaming Application Service (Phase 8).

Coordinates:
1. Form intent, statutory, user document, and combined multi-corpus query execution
2. Polymorphic Citation DTO mapping (Statutory, Document, Form)
3. Safe Server-Sent Events (SSE) streaming with pre-emission AST citation verification
"""

import asyncio
import json
import logging
import time
from typing import Any, AsyncGenerator, Dict, List, Optional

from backend.app.core.metrics import get_metrics_collector
from backend.app.api.schemas.query import (
    CitationDTO,
    CitationType,
    DocumentCitationDTO,
    FormCitationDTO,
    QueryRequestDTO,
    QueryResponseDTO,
    StatutoryCitationDTO,
)
from backend.app.document_rag.models import (
    RoutingDecision,
    UserDocumentSessionScope,
)
from backend.app.document_rag.pipeline import (
    UserDocumentRAGPipeline,
    get_user_doc_rag_pipeline,
)
from backend.app.forms.pipeline import StatutoryFormPipeline
from backend.app.forms.repository import get_form_registry
from backend.app.generation.generator import (
    StatutoryGenerationPipeline,
    get_generation_pipeline,
)
from backend.app.generation.models import LegalAnswerResponse

logger = logging.getLogger("nyaya.services.query")


class LegalQueryService:
    """Application service orchestrating legal reasoning across statutory, document, and form domain pipelines."""

    def __init__(
        self,
        statutory_pipeline: Optional[StatutoryGenerationPipeline] = None,
        user_doc_pipeline: Optional[UserDocumentRAGPipeline] = None,
        forms_pipeline: Optional[StatutoryFormPipeline] = None
    ):
        self.statutory_pipeline = statutory_pipeline or get_generation_pipeline()
        self._user_doc_pipeline = user_doc_pipeline
        self.forms_pipeline = forms_pipeline or StatutoryFormPipeline(registry=get_form_registry())

    @property
    def user_doc_pipeline(self) -> UserDocumentRAGPipeline:
        """Lazily initialize UserDocumentRAGPipeline to conserve RAM when serving statutory queries."""
        if self._user_doc_pipeline is None:
            self._user_doc_pipeline = get_user_doc_rag_pipeline()
        return self._user_doc_pipeline

    @user_doc_pipeline.setter
    def user_doc_pipeline(self, pipeline: Optional[UserDocumentRAGPipeline]) -> None:
        self._user_doc_pipeline = pipeline

    def execute_query(
        self,
        scope: UserDocumentSessionScope,
        request: QueryRequestDTO
    ) -> QueryResponseDTO:
        """Execute unified grounded legal query across statutory, document, and form corpora."""
        start_time = time.perf_counter()
        query_text = request.query.strip()
        collector = get_metrics_collector()

        # 1. Statutory Form Intent Check (e.g. "Form 1", "Form under s.35(3)")
        if request.enable_forms and self._is_form_intent(query_text):
            form_lookup = self.forms_pipeline.lookup(query_text)
            if form_lookup.status == "SUCCESS" and form_lookup.form:
                # Direct deterministic form response
                citations = [
                    FormCitationDTO(
                        citation_text=form_lookup.provenance or f"[BNSS Second Schedule, Form {form_lookup.form.form_number}]",
                        citation_type=CitationType.FORM,
                        is_verified=True,
                        source_id=form_lookup.form.form_id,
                        page_start=form_lookup.form.page_start,
                        page_end=form_lookup.form.page_end,
                        form_number=form_lookup.form.form_number,
                        form_title=form_lookup.form.form_title,
                        applicable_sections=form_lookup.form.applicable_sections
                    )
                ]
                dto = QueryResponseDTO(
                    query=query_text,
                    status="SUCCESS",
                    answer=form_lookup.rendered_markdown or form_lookup.form.raw_text,
                    is_refused=False,
                    citations=citations,
                    confidence_score=1.0,
                    routed_corpus="STATUTORY_FORM",
                    telemetry={"latency_ms": form_lookup.latency_ms}
                )
                collector.record_chat_request("STATUTORY_FORM", time.perf_counter() - start_time)
                return dto
            elif form_lookup.status == "AMBIGUOUS":
                dto = QueryResponseDTO(
                    query=query_text,
                    status="AMBIGUOUS",
                    answer=None,
                    is_refused=False,
                    refusal_reason=form_lookup.refusal_reason or "Query matches multiple statutory forms.",
                    citations=[],
                    confidence_score=0.5,
                    routed_corpus="STATUTORY_FORM",
                    candidate_forms=form_lookup.candidate_forms,
                    telemetry={"latency_ms": form_lookup.latency_ms}
                )
                collector.record_chat_request("STATUTORY_FORM", time.perf_counter() - start_time)
                return dto
            elif form_lookup.status == "NOT_FOUND" and form_lookup.is_refused:
                dto = QueryResponseDTO(
                    query=query_text,
                    status="NOT_FOUND",
                    answer=None,
                    is_refused=True,
                    refusal_reason=form_lookup.refusal_reason,
                    citations=[],
                    confidence_score=0.0,
                    routed_corpus="STATUTORY_FORM",
                    telemetry={"latency_ms": form_lookup.latency_ms}
                )
                collector.record_chat_request("STATUTORY_FORM", time.perf_counter() - start_time)
                collector.record_refusal("FORM_NOT_FOUND")
                return dto

        # 2. Check if active user documents are present or specified
        active_doc_ids = request.document_ids or []
        effective_scope = UserDocumentSessionScope(
            user_id=scope.user_id,
            session_id=scope.session_id,
            active_document_ids=active_doc_ids
        )

        has_documents = False
        if len(active_doc_ids) > 0:
            has_documents = True
        elif len(scope.active_document_ids) > 0:
            has_documents = self.user_doc_pipeline.repository.count_user_chunks(effective_scope) > 0
        elif self._user_doc_pipeline is not None:
            has_documents = self.user_doc_pipeline.repository.count_user_chunks(effective_scope) > 0

        if has_documents:
            # Multi-tenant document RAG pipeline (handles statutory, document, and combined)
            doc_resp = self.user_doc_pipeline.query(
                query_text=query_text,
                scope=effective_scope
            )
            dto = self._map_pipeline_response_to_dto(query_text, doc_resp)
            dur = time.perf_counter() - start_time
            collector.record_chat_request(dto.routed_corpus or "DOCUMENT", dur)
            if dto.is_refused:
                collector.record_refusal(dto.status)
            if doc_resp.telemetry:
                p_tok = doc_resp.telemetry.prompt_tokens or 0
                c_tok = doc_resp.telemetry.completion_tokens or 0
                if p_tok > 0 or c_tok > 0:
                    collector.record_tokens_and_cost(p_tok, c_tok)
            return dto

        # 3. Pure Statutory Legal Query (Phase 5)
        statutory_resp = self.statutory_pipeline.generate(query_text)
        dto = self._map_pipeline_response_to_dto(query_text, statutory_resp)
        dur = time.perf_counter() - start_time
        collector.record_chat_request(dto.routed_corpus or "STATUTORY", dur)
        if dto.is_refused:
            collector.record_refusal(dto.status)
        if statutory_resp.telemetry:
            p_tok = statutory_resp.telemetry.prompt_tokens or 0
            c_tok = statutory_resp.telemetry.completion_tokens or 0
            if p_tok > 0 or c_tok > 0:
                collector.record_tokens_and_cost(p_tok, c_tok)
        return dto

    async def stream_query(
        self,
        scope: UserDocumentSessionScope,
        request: QueryRequestDTO
    ) -> AsyncGenerator[str, None]:
        """Server-Sent Events generator enforcing strict pre-emission AST citation verification."""
        yield self._format_sse("status", {"step": "RETRIEVING", "message": "Retrieving relevant legal evidence..."})
        await asyncio.sleep(0.01)

        yield self._format_sse("status", {"step": "RERANKING", "message": "Reranking candidates and checking confidence..."})
        await asyncio.sleep(0.01)

        yield self._format_sse("status", {"step": "GENERATING", "message": "Synthesizing answer and validating citations..."})

        # Execute grounded query and validation in memory
        result_dto = self.execute_query(scope, request)

        if result_dto.is_refused or result_dto.status in ("REFUSED", "VALIDATION_FAILED", "NOT_FOUND", "AMBIGUOUS"):
            yield self._format_sse("refusal", result_dto.model_dump())
            return

        # Emit verified citation metadata first
        if result_dto.citations:
            yield self._format_sse("citation", {"citations": [c.model_dump() for c in result_dto.citations]})

        # Stream answer tokens safely post-validation
        if result_dto.answer:
            tokens = result_dto.answer.split(" ")
            for idx, token in enumerate(tokens):
                word = token + (" " if idx < len(tokens) - 1 else "")
                yield self._format_sse("token", {"token": word})
                await asyncio.sleep(0.005)

        # Final complete payload
        yield self._format_sse("complete", result_dto.model_dump())

    def _map_pipeline_response_to_dto(
        self,
        query_text: str,
        resp: LegalAnswerResponse
    ) -> QueryResponseDTO:
        """Map domain LegalAnswerResponse to client QueryResponseDTO with polymorphic citations."""
        citations: List[CitationDTO] = []
        for cv in resp.citations:
            if not cv.is_verified:
                continue

            # Discriminate citation type based on prefix and metadata
            if "[DOC" in cv.citation_text:
                doc_id = cv.chunk_id.split("_p")[0] if "_p" in cv.chunk_id else cv.chunk_id
                citations.append(
                    DocumentCitationDTO(
                        citation_text=cv.citation_text,
                        citation_type=CitationType.DOCUMENT,
                        is_verified=True,
                        source_id=cv.chunk_id,
                        page_start=cv.page_start,
                        page_end=cv.page_end,
                        document_id=doc_id,
                        filename=cv.section_title or "user_document.pdf",
                        page_number=cv.page_start,
                        source_text=getattr(cv, "source_text", None)
                    )
                )
            elif "Second Schedule" in cv.citation_text or "Form" in cv.citation_text:
                f_num = 1
                try:
                    f_num = int("".join(filter(str.isdigit, cv.section)))
                except Exception:
                    pass
                citations.append(
                    FormCitationDTO(
                        citation_text=cv.citation_text,
                        citation_type=CitationType.FORM,
                        is_verified=True,
                        source_id=cv.chunk_id,
                        page_start=cv.page_start,
                        page_end=cv.page_end,
                        form_number=f_num,
                        form_title=cv.section_title,
                        source_text=getattr(cv, "source_text", None)
                    )
                )
            else:
                citations.append(
                    StatutoryCitationDTO(
                        citation_text=cv.citation_text,
                        citation_type=CitationType.STATUTORY,
                        is_verified=True,
                        source_id=cv.chunk_id,
                        page_start=cv.page_start,
                        page_end=cv.page_end,
                        act=cv.act,
                        act_short=cv.act_short,
                        section=cv.section,
                        section_title=cv.section_title,
                        source_text=getattr(cv, "source_text", None)
                    )
                )

        routed_corpus = "STATUTORY"
        if resp.retrieval_metadata:
            routed_corpus = resp.retrieval_metadata.get("routed_corpus", "STATUTORY")

        telemetry_dict = resp.telemetry.model_dump() if resp.telemetry else None

        return QueryResponseDTO(
            query=query_text,
            status=resp.status,
            answer=resp.answer,
            is_refused=resp.is_refused,
            refusal_reason=resp.refusal_reason,
            citations=citations,
            confidence_score=resp.confidence_score if hasattr(resp, "confidence_score") else (resp.confidence.get("confidence_score", 1.0) if resp.confidence else 1.0),
            routed_corpus=routed_corpus,
            telemetry=telemetry_dict
        )

    @staticmethod
    def _is_form_intent(query: str) -> bool:
        """Check if query specifically targets statutory forms."""
        q = query.lower()
        if "form" in q and any(w in q for w in ["no", "number", "1", "2", "3", "4", "33", "58", "show", "get", "what is"]):
            return True
        if "notice for appearance" in q or "warrant of arrest" in q or "bail bond" in q:
            return True
        return False

    @staticmethod
    def _format_sse(event: str, data: Dict[str, Any]) -> str:
        """Format data into Server-Sent Event text payload."""
        return f"event: {event}\ndata: {json.dumps(data)}\n\n"


_query_service_instance: Optional[LegalQueryService] = None


def get_query_service() -> LegalQueryService:
    """Singleton provider for LegalQueryService."""
    global _query_service_instance
    if _query_service_instance is None:
        _query_service_instance = LegalQueryService()
    return _query_service_instance
