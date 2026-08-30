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
    
    # Phase 5: LLM Generation & Citation Configuration
    llm_provider: str = os.getenv("LLM_PROVIDER", "ollama")
    llm_model: str = os.getenv("LLM_MODEL", "llama3.2")
    llm_base_url: str = os.getenv("LLM_BASE_URL", "http://localhost:11434")
    llm_api_key: str = os.getenv("LLM_API_KEY", "")
    llm_temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.0"))
    llm_max_tokens: int = int(os.getenv("LLM_MAX_TOKENS", "1024"))
    llm_timeout: float = float(os.getenv("LLM_TIMEOUT", "30.0"))
    llm_max_retries: int = int(os.getenv("LLM_MAX_RETRIES", "2"))
    llm_max_context_chars: int = int(os.getenv("LLM_MAX_CONTEXT_CHARS", "8000"))
    llm_cost_per_1k_input_tokens: float = float(os.getenv("LLM_COST_PER_1K_INPUT_TOKENS", "0.0"))
    llm_cost_per_1k_output_tokens: float = float(os.getenv("LLM_COST_PER_1K_OUTPUT_TOKENS", "0.0"))
    
    # Phase 6: User-Document RAG & Multi-Tenant Isolation
    qdrant_user_collection: str = os.getenv("QDRANT_USER_COLLECTION", "nyaya_user_documents")
    max_user_doc_size_bytes: int = int(os.getenv("MAX_USER_DOC_SIZE_BYTES", str(25 * 1024 * 1024)))  # 25 MB
    ocr_enabled: bool = os.getenv("OCR_ENABLED", "true").lower() in ("true", "1", "yes")
    ocr_max_pages: int = int(os.getenv("OCR_MAX_PAGES", "10"))
    ocr_timeout_seconds: float = float(os.getenv("OCR_TIMEOUT_SECONDS", "10.0"))

    # Phase 8: API & Application Integration
    api_prefix: str = os.getenv("API_PREFIX", "/api/v1")
    auth_mode: str = os.getenv("AUTH_MODE", "prod").lower()  # "prod" | "dev"
    jwt_secret: str = os.getenv("JWT_SECRET", "dev_secret_key_change_in_production")
    jwt_algorithm: str = os.getenv("JWT_ALGORITHM", "HS256")
    cors_origins: str = os.getenv("CORS_ORIGINS", "*")


settings = Settings()

