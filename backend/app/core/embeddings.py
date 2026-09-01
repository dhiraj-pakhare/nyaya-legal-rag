"""Self-hosted Open-Source Embedding Model Wrapper for Nyaya Legal RAG.

Wraps BAAI/bge-base-en-v1.5 using sentence-transformers, providing batch document embedding,
asymmetric query instruction encoding, and embedding normalization.
"""

import os
# Force PyTorch backend for transformers to avoid Keras 3 conflicts
os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"

import time
import logging
from typing import List, Optional
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from backend.app.core.config import settings

# Restrict intra-op CPU threads to avoid excessive memory arena allocation in containerized runtimes
if hasattr(torch, "set_num_threads"):
    try:
        torch.set_num_threads(min(2, torch.get_num_threads()))
    except Exception:
        pass

logger = logging.getLogger("nyaya.embeddings")


class EmbeddingModel:
    """Encapsulates the BAAI/bge-base-en-v1.5 dense embedding model."""

    def __init__(
        self,
        model_name: str = settings.embedding_model_name,
        device: str = settings.embedding_device,
        normalize_embeddings: bool = True,
        max_seq_length: int = settings.embedding_max_seq_length
    ):
        self.model_name = model_name
        self.device = device
        self.normalize_embeddings = normalize_embeddings
        self.dimension = settings.embedding_dimension
        self.max_seq_length = max_seq_length
        self.query_instruction = "Represent this sentence for searching relevant passages: "
        
        logger.info(f"Loading embedding model '{model_name}' on device '{device}'...")
        start_t = time.perf_counter()
        self.model = SentenceTransformer(model_name, device=device)
        self.model.max_seq_length = max_seq_length
        self.load_duration = time.perf_counter() - start_t
        logger.info(f"Embedding model loaded in {self.load_duration:.2f}s. Dimension: {self.dimension}")

    def embed_documents(
        self,
        texts: List[str],
        batch_size: int = settings.embedding_batch_size,
        show_progress: bool = False
    ) -> np.ndarray:
        """Encode a batch of statutory document/chunk texts.
        
        Passages do NOT receive the query instruction prefix per BGE specification.
        """
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
            
        start_t = time.perf_counter()
        with torch.inference_mode():
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                normalize_embeddings=self.normalize_embeddings,
                convert_to_numpy=True
            )
        duration = time.perf_counter() - start_t
        throughput = len(texts) / max(duration, 0.001)
        
        logger.info(
            f"Encoded {len(texts)} document chunks in {duration:.2f}s "
            f"({throughput:.1f} chunks/sec, batch_size={batch_size})"
        )
        return embeddings

    def embed_query(self, query: str) -> List[float]:
        """Encode a search query with the official BGE query instruction prefix.
        
        BGE models require the prefix 'Represent this sentence for searching relevant passages: '
        on queries to align query space with passage representation space.
        """
        prefixed_query = f"{self.query_instruction}{query.strip()}"
        with torch.inference_mode():
            embedding = self.model.encode(
                prefixed_query,
                normalize_embeddings=self.normalize_embeddings,
                convert_to_numpy=True
            )
        return embedding.tolist()


_GLOBAL_EMBEDDING_MODEL: Optional[EmbeddingModel] = None


def get_embedding_model() -> EmbeddingModel:
    """Get or initialize the global singleton EmbeddingModel instance."""
    global _GLOBAL_EMBEDDING_MODEL
    if _GLOBAL_EMBEDDING_MODEL is None:
        _GLOBAL_EMBEDDING_MODEL = EmbeddingModel()
    return _GLOBAL_EMBEDDING_MODEL
