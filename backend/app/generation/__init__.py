"""Generation, Citation Contract, and Validation Subsystem for Nyaya Legal RAG."""

from backend.app.generation.citation_parser import CitationParser
from backend.app.generation.citation_validator import CitationValidator
from backend.app.generation.context_builder import StatutoryContextBuilder
from backend.app.generation.generator import StatutoryGenerationPipeline
from backend.app.generation.models import (
    CitationVerification,
    GenerationTelemetry,
    LegalAnswerResponse,
    LLMMessage,
    LLMResponse,
    ParsedCitation,
    ValidationStatus
)
from backend.app.generation.prompt import (
    REGENERATION_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    build_generation_messages,
    build_regeneration_messages
)
from backend.app.generation.providers import (
    LLMProvider,
    MockLLMProvider,
    OllamaProvider,
    OpenAICompatibleProvider,
    get_llm_provider
)
from backend.app.generation.streaming import SafeStatutoryStreamer

__all__ = [
    "LLMMessage",
    "LLMResponse",
    "ParsedCitation",
    "CitationVerification",
    "ValidationStatus",
    "GenerationTelemetry",
    "LegalAnswerResponse",
    "LLMProvider",
    "OllamaProvider",
    "OpenAICompatibleProvider",
    "MockLLMProvider",
    "get_llm_provider",
    "SYSTEM_PROMPT",
    "REGENERATION_SYSTEM_PROMPT",
    "build_generation_messages",
    "build_regeneration_messages",
    "StatutoryContextBuilder",
    "CitationParser",
    "CitationValidator",
    "StatutoryGenerationPipeline",
    "SafeStatutoryStreamer"
]
