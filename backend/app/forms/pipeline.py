"""Master Statutory Forms Pipeline for Nyaya Legal RAG (Phase 7).

Coordinates:
1. Deterministic Form Lookup (Zero-LLM sub-millisecond execution)
2. Deterministic Form Rendering (Markdown & Plaintext presentation)
3. Safe Grounded Conversational QA (Grounded in retrieved StatutoryForm with AST validation)
"""

import logging
import time
from typing import List, Optional

from backend.app.forms.citation_validator import FormCitationValidator
from backend.app.forms.lookup import DeterministicFormIdentifier
from backend.app.forms.models import FormLookupResponse, StatutoryForm
from backend.app.forms.renderer import DeterministicFormRenderer
from backend.app.forms.repository import StatutoryFormRegistry, get_form_registry
from backend.app.generation.context_builder import StatutoryContextBuilder
from backend.app.generation.models import (
    GenerationTelemetry,
    LegalAnswerResponse,
    ValidationStatus,
)
from backend.app.generation.providers import LLMProvider, get_llm_provider
from backend.app.retrieval.models import RetrievedDocument

logger = logging.getLogger("nyaya.forms.pipeline")


class StatutoryFormPipeline:
    """End-to-end pipeline for statutory forms lookup, presentation, and safe Q&A."""

    def __init__(
        self,
        registry: Optional[StatutoryFormRegistry] = None,
        llm_provider: Optional[LLMProvider] = None
    ):
        self.registry = registry or get_form_registry()
        self.identifier = DeterministicFormIdentifier(registry=self.registry)
        self.renderer = DeterministicFormRenderer()
        self.citation_validator = FormCitationValidator(registry=self.registry)
        self.llm = llm_provider or get_llm_provider()

    def lookup(self, query: str) -> FormLookupResponse:
        """Execute deterministic form retrieval and rendering with ZERO LLM invocation."""
        start_time = time.perf_counter()
        resp = self.identifier.identify(query)

        if resp.status == "SUCCESS" and resp.form:
            resp.rendered_markdown = self.renderer.render_markdown(resp.form)

        resp.latency_ms = round((time.perf_counter() - start_time) * 1000.0, 3)
        return resp

    def query(
        self,
        query_text: str
    ) -> LegalAnswerResponse:
        """Execute conversational grounded QA over a statutory form with citation validation."""
        total_start = time.perf_counter()
        
        # 1. Deterministic Form Lookup First
        lookup_res = self.lookup(query_text)

        # 2. Gatekeeper: If form is ambiguous or not found, refuse without calling LLM
        if lookup_res.status != "SUCCESS" or not lookup_res.form:
            total_latency_ms = (time.perf_counter() - total_start) * 1000.0
            reason = lookup_res.refusal_reason or "Statutory form not found."
            return LegalAnswerResponse(
                query=query_text,
                status=lookup_res.status,
                answer=None,
                citations=[],
                is_refused=True,
                refusal_reason=reason,
                confidence={"confidence_score": 0.0 if lookup_res.status == "NOT_FOUND" else 0.5},
                telemetry=GenerationTelemetry(
                    retrieval_latency_ms=lookup_res.latency_ms,
                    generation_latency_ms=0.0,
                    validation_latency_ms=0.0,
                    total_latency_ms=total_latency_ms,
                    prompt_tokens=0,
                    completion_tokens=0,
                    total_tokens=0,
                    model=self.llm.model,
                    provider=self.llm.provider_name
                )
            )

        form = lookup_res.form
        retrieval_latency_ms = lookup_res.latency_ms

        # 3. Construct Grounded Prompt from Canonical Form Data
        form_context_str = (
            f"=== STATUTORY FORM EVIDENCE ===\n"
            f"FORM No. {form.form_number}: {form.form_title}\n"
            f"Statutory Provision: See Section(s) {', '.join(form.applicable_sections)}\n"
            f"Citation Tag: {form.provenance_citation}\n"
            f"Source: Gazette PDF Pages {form.page_start}-{form.page_end}\n\n"
            f"Form Text:\n{form.raw_text}\n"
            f"=== END FORM EVIDENCE ==="
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are Nyaya Legal Assistant specializing in statutory forms under the Bharatiya Nagarik Suraksha Sanhita, 2023.\n"
                    "Answer the question using ONLY the provided statutory form evidence.\n"
                    f"Every factual claim about the form MUST include the canonical citation tag: {form.provenance_citation}.\n"
                    "Do NOT invent form numbers, fields, or statutory sections."
                )
            },
            {
                "role": "user",
                "content": f"Evidence:\n{form_context_str}\n\nQuestion: {query_text}"
            }
        ]

        # 4. LLM Generation
        gen_start = time.perf_counter()
        llm_resp = self.llm.generate(messages)
        gen_latency_ms = (time.perf_counter() - gen_start) * 1000.0

        # 5. AST Citation Validation
        val_start = time.perf_counter()
        val_status = self.citation_validator.validate(llm_resp.content, [form])
        val_latency_ms = (time.perf_counter() - val_start) * 1000.0

        regeneration_attempted = False

        # 6. Controlled 1-Time Regeneration Pass if Validation Fails
        if not val_status.is_valid:
            logger.warning(f"Citation validation failed on Attempt 1: {val_status.failure_reasons}. Triggering regeneration.")
            regen_messages = [
                {
                    "role": "system",
                    "content": (
                        "You are Nyaya Legal Assistant. Your previous response failed citation validation.\n"
                        f"Validation errors: {'; '.join(val_status.failure_reasons)}.\n"
                        f"You MUST use the exact citation tag {form.provenance_citation}."
                    )
                },
                {
                    "role": "user",
                    "content": f"Evidence:\n{form_context_str}\n\nQuestion: {query_text}\nPlease regenerate a strictly grounded answer."
                }
            ]
            regen_start = time.perf_counter()
            regen_resp = self.llm.generate(regen_messages)
            gen_latency_ms += (time.perf_counter() - regen_start) * 1000.0

            regen_val_start = time.perf_counter()
            val_status = self.citation_validator.validate(regen_resp.content, [form])
            val_latency_ms += (time.perf_counter() - regen_val_start) * 1000.0
            regeneration_attempted = True
            val_status.regeneration_attempted = True
            llm_resp = regen_resp

        total_latency_ms = (time.perf_counter() - total_start) * 1000.0

        # 7. Final Decision
        if not val_status.is_valid:
            return LegalAnswerResponse(
                query=query_text,
                status="VALIDATION_FAILED",
                answer=None,
                citations=[],
                is_refused=True,
                refusal_reason=f"Citation validation failed: {'; '.join(val_status.failure_reasons)}",
                confidence={"confidence_score": 1.0},
                validation_status=val_status,
                telemetry=GenerationTelemetry(
                    retrieval_latency_ms=retrieval_latency_ms,
                    generation_latency_ms=gen_latency_ms,
                    validation_latency_ms=val_latency_ms,
                    total_latency_ms=total_latency_ms,
                    prompt_tokens=llm_resp.prompt_tokens,
                    completion_tokens=llm_resp.completion_tokens,
                    total_tokens=llm_resp.total_tokens,
                    model=llm_resp.model,
                    provider=getattr(self.llm, "provider_name", type(self.llm).__name__)
                )
            )

        return LegalAnswerResponse(
            query=query_text,
            status="SUCCESS",
            answer=llm_resp.content,
            citations=val_status.verified_citations,
            is_refused=False,
            confidence={"confidence_score": 1.0},
            validation_status=val_status,
            telemetry=GenerationTelemetry(
                retrieval_latency_ms=retrieval_latency_ms,
                generation_latency_ms=gen_latency_ms,
                validation_latency_ms=val_latency_ms,
                total_latency_ms=total_latency_ms,
                prompt_tokens=llm_resp.prompt_tokens,
                completion_tokens=llm_resp.completion_tokens,
                total_tokens=llm_resp.total_tokens,
                model=llm_resp.model,
                provider=getattr(self.llm, "provider_name", type(self.llm).__name__)
            )
        )
