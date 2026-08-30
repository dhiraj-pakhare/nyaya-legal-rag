"""Safe Streaming Adapter ensuring zero unvalidated legal claims reach the client."""

import json
import logging
from typing import Any, Dict, Generator, Optional

from backend.app.generation.generator import StatutoryGenerationPipeline
from backend.app.generation.models import LegalAnswerResponse
from backend.app.retrieval.models import RetrievalResult

logger = logging.getLogger("nyaya.generation.streaming")


class StreamEvent:
    """Structured Server-Sent Event for legal answer streaming."""

    def __init__(self, event_type: str, data: Any):
        self.event_type = event_type
        self.data = data

    def to_sse(self) -> str:
        payload = json.dumps(self.data) if isinstance(self.data, dict) else str(self.data)
        return f"event: {self.event_type}\ndata: {payload}\n\n"


class SafeStatutoryStreamer:
    """Streams statutory answers only AFTER complete programmatic citation verification."""

    def __init__(self, pipeline: StatutoryGenerationPipeline):
        self.pipeline = pipeline

    def stream_validated_response(
        self,
        query: str,
        retrieval_result: Optional[RetrievalResult] = None,
        retrieval_mode: str = "auto"
    ) -> Generator[str, None, None]:
        """Execute full generation and validation pipeline before emitting tokens to client."""
        yield StreamEvent("status", {"message": "Retrieving and reranking statutory provisions..."}).to_sse()

        # Step 1: Execute pipeline (buffered generation + validation + regeneration guard)
        response: LegalAnswerResponse = self.pipeline.generate(
            query=query,
            retrieval_result=retrieval_result,
            retrieval_mode=retrieval_mode
        )

        # Step 2: If refused or validation failed, emit refusal event immediately
        if response.is_refused:
            yield StreamEvent("refusal", {
                "status": response.status,
                "reason": response.refusal_reason,
                "confidence": response.confidence
            }).to_sse()
            yield StreamEvent("complete", response.model_dump()).to_sse()
            return

        # Step 3: Stream validated answer words/chunks
        yield StreamEvent("status", {"message": "Citation validation passed. Streaming validated response."}).to_sse()
        
        if response.answer:
            # Safely stream word by word
            words = response.answer.split(" ")
            for idx, word in enumerate(words):
                suffix = " " if idx < len(words) - 1 else ""
                yield StreamEvent("token", {"token": word + suffix}).to_sse()

        # Step 4: Emit complete final payload with sources and telemetry for UI source drawer
        yield StreamEvent("complete", response.model_dump()).to_sse()
