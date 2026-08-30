"""User Document RAG & Multi-Tenant Isolation Package for Nyaya Legal RAG."""

from backend.app.document_rag.models import (
    IngestionStatus,
    QueryIntent,
    UserDocument,
    UserDocumentChunk,
    UserDocumentSessionScope,
    RoutingDecision,
    DocumentIngestionResult,
    SecurityScopeError,
    DocumentNotFoundError,
    CorruptPDFError,
    OversizedDocumentError,
    OCRUnavailableError,
)
from backend.app.document_rag.pdf_extractor import UserPDFExtractor, ExtractedPage
from backend.app.document_rag.chunker import UserDocumentChunker
from backend.app.document_rag.repository import UserDocumentRepository
from backend.app.document_rag.bm25 import UserDocumentBM25Manager, UserDocumentBM25Index
from backend.app.document_rag.retriever import UserDocumentRetriever
from backend.app.document_rag.router import QueryIntentRouter
from backend.app.document_rag.context_builder import MultiSourceContextBuilder
from backend.app.document_rag.citation_validator import DualCitationParser, DualCitationValidator
from backend.app.document_rag.pipeline import UserDocumentRAGPipeline
from backend.app.document_rag.security import resolve_trusted_identity, sanitize_filename, sanitize_for_logs

__all__ = [
    "IngestionStatus",
    "QueryIntent",
    "UserDocument",
    "UserDocumentChunk",
    "UserDocumentSessionScope",
    "RoutingDecision",
    "DocumentIngestionResult",
    "SecurityScopeError",
    "DocumentNotFoundError",
    "CorruptPDFError",
    "OversizedDocumentError",
    "OCRUnavailableError",
    "UserPDFExtractor",
    "ExtractedPage",
    "UserDocumentChunker",
    "UserDocumentRepository",
    "UserDocumentBM25Manager",
    "UserDocumentBM25Index",
    "UserDocumentRetriever",
    "QueryIntentRouter",
    "MultiSourceContextBuilder",
    "DualCitationParser",
    "DualCitationValidator",
    "UserDocumentRAGPipeline",
    "resolve_trusted_identity",
    "sanitize_filename",
    "sanitize_for_logs",
]
