"""Retrieval subsystem package for Nyaya Legal RAG."""

from backend.app.retrieval.models import RetrievedDocument, RetrievalResult, RetrievalFilter
from backend.app.retrieval.bm25 import BM25Retriever
from backend.app.retrieval.dense import DenseRetriever
from backend.app.retrieval.rrf import reciprocal_rank_fusion
from backend.app.retrieval.reranker import CrossEncoderReranker, get_reranker
from backend.app.retrieval.confidence import ConfidenceScorer, ConfidenceResult
from backend.app.retrieval.intent import SectionIntentDetector, SectionIntent
from backend.app.retrieval.exact_lookup import ExactSectionLookup
from backend.app.retrieval.pipeline import HybridRetrievalPipeline

__all__ = [
    "RetrievedDocument",
    "RetrievalResult",
    "RetrievalFilter",
    "BM25Retriever",
    "DenseRetriever",
    "reciprocal_rank_fusion",
    "CrossEncoderReranker",
    "get_reranker",
    "ConfidenceScorer",
    "ConfidenceResult",
    "SectionIntentDetector",
    "SectionIntent",
    "ExactSectionLookup",
    "HybridRetrievalPipeline",
]
