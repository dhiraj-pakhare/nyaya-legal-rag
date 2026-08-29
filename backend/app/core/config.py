"""Configuration settings for Nyaya Legal RAG."""

import os
from pydantic import BaseModel, Field


class Settings(BaseModel):
    # Model configuration
    embedding_model_name: str = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-base-en-v1.5")
    embedding_dimension: int = int(os.getenv("EMBEDDING_DIMENSION", "768"))
    embedding_batch_size: int = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))
    embedding_max_seq_length: int = int(os.getenv("EMBEDDING_MAX_SEQ_LENGTH", "512"))
    embedding_device: str = os.getenv("EMBEDDING_DEVICE", "cpu")
    
    # Qdrant configuration
    qdrant_url: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_collection: str = os.getenv("QDRANT_COLLECTION", "nyaya_legal_corpus")
    qdrant_api_key: str = os.getenv("QDRANT_API_KEY", "")
    
    # Document paths
    pdf_path: str = os.getenv("PDF_PATH", "BNS bare act 2023.pdf")
    
    # Retrieval & Reranking configuration
    reranker_model_name: str = os.getenv("RERANKER_MODEL_NAME", "cross-encoder/ms-marco-TinyBERT-L-2-v2")
    reranker_candidate_k: int = int(os.getenv("RERANKER_CANDIDATE_K", "10"))
    reranker_top_k: int = int(os.getenv("RERANKER_TOP_K", "5"))
    confidence_threshold: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.75"))


settings = Settings()
