"""Master Statutory Generation Pipeline with Citation Validation and Refusal Integration."""

import logging
import re
import time
from typing import Any, Dict, List, Optional

from backend.app.core.config import settings
from backend.app.generation.citation_validator import CitationValidator
from backend.app.generation.context_builder import StatutoryContextBuilder
from backend.app.generation.models import (
    CitationVerification,
    GenerationTelemetry,
    LegalAnswerResponse,
    ValidationStatus
)
from backend.app.generation.prompt import (
    build_generation_messages,
    build_regeneration_messages
)
from backend.app.generation.providers import LLMProvider, get_llm_provider
from backend.app.retrieval.models import RetrievalResult
from backend.app.retrieval.pipeline import HybridRetrievalPipeline

logger = logging.getLogger("nyaya.generation.generator")


class StatutoryGenerationPipeline:
    """End-to-end statutory legal generation pipeline integrating:
    
    1. Phase 4 Retrieval & Calibrated Refusal Gating
    2. Deterministic Context Construction
    3. Provider-Independent LLM Generation
    4. Programmatic AST Citation & Claim Validation
    5. Controlled Single-Attempt Regeneration for Hallucinations
    6. Complete Telemetry and Source Drawer Contracts
    """

    def __init__(
        self,
        retrieval_pipeline: Optional[HybridRetrievalPipeline] = None,
        llm_provider: Optional[LLMProvider] = None,
        context_builder: Optional[StatutoryContextBuilder] = None,
        citation_validator: Optional[CitationValidator] = None
    ):
        if retrieval_pipeline is not None:
            self.retrieval_pipeline = retrieval_pipeline
        else:
            try:
                from backend.app.retrieval.pipeline import get_hybrid_retrieval_pipeline
                self.retrieval_pipeline = get_hybrid_retrieval_pipeline()
            except Exception as e:
                logger.warning(f"Could not lazily initialize default hybrid retrieval pipeline: {e}")
                self.retrieval_pipeline = None
        self.llm_provider = llm_provider or get_llm_provider()
        self.context_builder = context_builder or StatutoryContextBuilder()
        self.citation_validator = citation_validator or CitationValidator()

    def generate(
        self,
        query: str,
        retrieval_result: Optional[RetrievalResult] = None,
        retrieval_mode: str = "auto",
        top_k: int = settings.reranker_top_k
    ) -> LegalAnswerResponse:
        """Execute complete generation flow for a user query."""
        overall_start = time.perf_counter()
        clean_query = query.strip()

        # Step 1: Execute retrieval if not pre-supplied
        retrieval_latency = 0.0
        if retrieval_result is None:
            if self.retrieval_pipeline is None:
                raise ValueError("Retrieval pipeline must be provided if retrieval_result is not supplied")
            ret_start = time.perf_counter()
            retrieval_result = self.retrieval_pipeline.retrieve(clean_query, mode=retrieval_mode, top_k=top_k)
            retrieval_latency = (time.perf_counter() - ret_start) * 1000
        else:
            retrieval_latency = retrieval_result.latency_ms

        # Step 2: REFUSAL INTEGRATION (Phase 4 Confidence Gate)
        if retrieval_result.is_refused:
            logger.info(f"Query refused by retrieval confidence gate: {retrieval_result.refusal_reason}. Bypassing LLM.")
            total_latency = (time.perf_counter() - overall_start) * 1000
            
            telemetry = GenerationTelemetry(
                retrieval_latency_ms=round(retrieval_latency, 2),
                rerank_latency_ms=0.0,
                generation_latency_ms=0.0,
                validation_latency_ms=0.0,
                total_latency_ms=round(total_latency, 2),
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                estimated_cost_usd=0.0,
                model=self.llm_provider.model,
                provider=self.llm_provider.__class__.__name__
            )

            return LegalAnswerResponse(
                query=query,
                answer=None,
                status="REFUSED",
                is_refused=True,
                refusal_reason=retrieval_result.refusal_reason,
                confidence=retrieval_result.confidence,
                citations=[],
                sources=retrieval_result.documents,
                retrieval_metadata={
                    "mode": retrieval_result.mode,
                    "total_retrieved": retrieval_result.total_retrieved
                },
                validation_status=None,
                telemetry=telemetry
            )

        # Step 3: Context Construction
        context_str = self.context_builder.build_context(retrieval_result.documents)

        # Step 4: Initial LLM Generation
        gen_start = time.perf_counter()
        initial_messages = build_generation_messages(clean_query, context_str)
        llm_response = self.llm_provider.generate(initial_messages)
        logger.warning(
            "DEBUG LLM INITIAL CONTENT: %r",
            llm_response.content
        )
        gen_latency = (time.perf_counter() - gen_start) * 1000

        total_prompt_tokens = llm_response.prompt_tokens or 0
        total_completion_tokens = llm_response.completion_tokens or 0

        # Step 5: Programmatic Citation & Claim Validation
        val_start = time.perf_counter()
        validation_status = self.citation_validator.validate(
            answer=llm_response.content,
            retrieved_documents=retrieval_result.documents
        )
        val_latency = (time.perf_counter() - val_start) * 1000

        final_answer = llm_response.content
        final_status = "SUCCESS"
        is_refused = False
        refusal_reason = None

        # Step 6: Controlled Regeneration if Validation Failed
        if not validation_status.is_valid:
            logger.warning(
                f"Initial generation failed citation validation: {validation_status.error_details}. "
                f"Attempting ONE controlled regeneration."
            )
            regen_start = time.perf_counter()
            regen_messages = build_regeneration_messages(
                query=clean_query,
                context_str=context_str,
                invalid_answer=llm_response.content,
                failure_reasons=validation_status.failure_reasons
            )
            regen_response = self.llm_provider.generate(regen_messages)
            logger.warning(
                "DEBUG LLM REGEN CONTENT: %r",
                regen_response.content
            )
            gen_latency += (time.perf_counter() - regen_start) * 1000

            if regen_response.prompt_tokens:
                total_prompt_tokens += regen_response.prompt_tokens
            if regen_response.completion_tokens:
                total_completion_tokens += regen_response.completion_tokens

            # Re-validate 2nd generation
            val_start_2 = time.perf_counter()
            validation_status_2 = self.citation_validator.validate(
                answer=regen_response.content,
                retrieved_documents=retrieval_result.documents
            )
            val_latency += (time.perf_counter() - val_start_2) * 1000
            validation_status_2.regeneration_attempted = True

            if validation_status_2.is_valid:
                logger.info("Regenerated answer successfully passed citation validation.")
                final_answer = regen_response.content
                validation_status = validation_status_2
            else:
                logger.error(
                    f"Regenerated answer also failed citation validation: {validation_status_2.error_details}. "
                    f"Returning structured refusal."
                )
                final_answer = None
                final_status = "VALIDATION_FAILED"
                is_refused = True
                refusal_reason = f"Citation validation failed: {validation_status_2.error_details}"
                validation_status = validation_status_2

        # Extract verified citations for source drawer
        verified_citations: List[CitationVerification] = []
        if validation_status.is_valid and final_answer:
            parsed = self.citation_validator.parser.parse(final_answer)
            for cit in parsed:
                for doc in retrieval_result.documents:
                    doc_act = doc.act_short.upper()
                    doc_sec_raw = str(doc.section_number).strip()
                    doc_base_sec = doc_sec_raw
                    base_m = re.match(r'^(\d+[A-Za-z]?)', doc_sec_raw)
                    if base_m:
                        doc_base_sec = base_m.group(1)
                    
                    if doc_act == cit.act_short and (doc_sec_raw == cit.section_number or doc_base_sec == cit.section_number):
                        verified_citations.append(
                            CitationVerification(
                                citation_text=cit.canonical_tag,
                                act=doc.act,
                                act_short=doc.act_short,
                                section=doc.section_number,
                                subsection=cit.subsection or doc.subsection,
                                clause=cit.clause or doc.clause,
                                section_title=doc.section_title,
                                page_start=doc.page_start,
                                page_end=doc.page_end,
                                chunk_id=doc.chunk_id,
                                source_text=doc.text,
                                is_verified=True
                            )
                        )
                        break

        total_latency = (time.perf_counter() - overall_start) * 1000

        # Calculate estimated cost if configured
        estimated_cost = 0.0
        if settings.llm_cost_per_1k_input_tokens > 0 or settings.llm_cost_per_1k_output_tokens > 0:
            estimated_cost = (
                (total_prompt_tokens / 1000.0) * settings.llm_cost_per_1k_input_tokens +
                (total_completion_tokens / 1000.0) * settings.llm_cost_per_1k_output_tokens
            )

        telemetry = GenerationTelemetry(
            retrieval_latency_ms=round(retrieval_latency, 2),
            rerank_latency_ms=0.0,
            generation_latency_ms=round(gen_latency, 2),
            validation_latency_ms=round(val_latency, 2),
            total_latency_ms=round(total_latency, 2),
            prompt_tokens=total_prompt_tokens if total_prompt_tokens > 0 else None,
            completion_tokens=total_completion_tokens if total_completion_tokens > 0 else None,
            total_tokens=(total_prompt_tokens + total_completion_tokens) if (total_prompt_tokens + total_completion_tokens) > 0 else None,
            estimated_cost_usd=round(estimated_cost, 6) if estimated_cost > 0 else 0.0,
            model=self.llm_provider.model,
            provider=self.llm_provider.__class__.__name__
        )

        return LegalAnswerResponse(
            query=query,
            answer=final_answer,
            status=final_status,
            is_refused=is_refused,
            refusal_reason=refusal_reason,
            confidence=retrieval_result.confidence,
            citations=verified_citations,
            sources=retrieval_result.documents,
            retrieval_metadata={
                "mode": retrieval_result.mode,
                "total_retrieved": retrieval_result.total_retrieved
            },
            validation_status=validation_status,
            telemetry=telemetry
        )


_GLOBAL_GENERATION_PIPELINE: Optional[StatutoryGenerationPipeline] = None


def get_generation_pipeline() -> StatutoryGenerationPipeline:
    """Get or initialize the global singleton StatutoryGenerationPipeline instance."""
    global _GLOBAL_GENERATION_PIPELINE
    if _GLOBAL_GENERATION_PIPELINE is None:
        _GLOBAL_GENERATION_PIPELINE = StatutoryGenerationPipeline()
    return _GLOBAL_GENERATION_PIPELINE

