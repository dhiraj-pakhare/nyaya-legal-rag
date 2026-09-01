#!/usr/bin/env python3
"""One-shot Statutory Ingestion and Indexing CLI for Nyaya Legal RAG.

Extracts official BNS/BNSS Gazette PDF, detects statutory AST, generates structure-aware
canonical chunks, encodes dense vector embeddings using BAAI/bge-base-en-v1.5, and
idempotently indexes structured metadata payloads and vectors into Qdrant.
"""

import os
# Force PyTorch backend for transformers
os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"

import argparse
import hashlib
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.app.core.config import settings
from backend.app.core.embedding_input import format_chunk_for_embedding
from backend.app.core.embeddings import EmbeddingModel
from backend.app.core.qdrant_repo import QdrantRepository
from backend.app.ingestion.parser import StatutoryParser
from backend.app.ingestion.validator import IngestionValidator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("nyaya.ingest")


def compute_file_hash(filepath: str) -> str:
    """Compute SHA-256 hash of source PDF file."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            sha256.update(chunk)
    return sha256.hexdigest()


def run_ingestion(
    pdf_path: str = "BNS bare act 2023.pdf",
    qdrant_url: Optional[str] = None,
    qdrant_path: Optional[str] = None,
    qdrant_api_key: Optional[str] = None,
    collection_name: str = settings.qdrant_collection,
    recreate: bool = False,
    batch_size: int = settings.embedding_batch_size,
    device: str = settings.embedding_device,
    in_memory: bool = False
) -> dict:
    """Execute end-to-end ingestion, validation, embedding, and indexing."""
    total_start_t = time.perf_counter()
    
    # 1. Inspect source PDF
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"Source PDF not found at path: {pdf_path}")
    source_hash = compute_file_hash(pdf_path)
    file_size_mb = os.path.getsize(pdf_path) / (1024 * 1024)
    logger.info(f"Source PDF: '{pdf_path}' ({file_size_mb:.2f} MB, SHA-256: {source_hash[:16]}...)")
    
    # 2. Run Phase 1 Parser & AST detector
    logger.info("Starting Phase 1 PDF extraction & statutory structure detection...")
    parse_start_t = time.perf_counter()
    parser = StatutoryParser(pdf_path=pdf_path)
    result = parser.parse()
    parse_duration = time.perf_counter() - parse_start_t
    
    doc = result.document
    chunks = result.chunks
    sched_entries = result.schedule_entries
    report = result.validation_report
    
    logger.info(
        f"Parsed in {parse_duration:.2f}s: {len(doc.chapters)} chapters, "
        f"{len(doc.sections)} sections, {len(sched_entries)} schedule entries, "
        f"{len(chunks)} statutory chunks. Valid: {report.is_valid}"
    )
    
    if not report.is_valid:
        error_msgs = [f"{i.code}: {i.message}" for i in report.issues if i.severity == "ERROR"]
        raise ValueError(f"Corpus validation failed with errors: {error_msgs}")
        
    # 3. Format chunks for embedding
    logger.info("Formatting chunks for dense embedding representation...")
    embedding_texts = [format_chunk_for_embedding(c) for c in chunks]
    
    # 4. Load Embedding Model & Compute Embeddings
    logger.info(f"Loading embedding model '{settings.embedding_model_name}' on device '{device}'...")
    embed_model = EmbeddingModel(
        model_name=settings.embedding_model_name,
        device=device,
        normalize_embeddings=True
    )
    
    logger.info(f"Encoding {len(embedding_texts)} statutory chunks (batch_size={batch_size})...")
    embed_start_t = time.perf_counter()
    embeddings = embed_model.embed_documents(embedding_texts, batch_size=batch_size, show_progress=True)
    embed_duration = time.perf_counter() - embed_start_t
    throughput = len(chunks) / max(embed_duration, 0.001)
    
    logger.info(
        f"Embedding completed in {embed_duration:.2f}s "
        f"({throughput:.1f} chunks/sec, dim={embeddings.shape[1]})"
    )
    
    # 5. Initialize Qdrant & Index Points
    logger.info(f"Connecting to Qdrant (collection='{collection_name}')...")
    qdrant_repo = QdrantRepository(
        url=qdrant_url,
        path=qdrant_path,
        api_key=qdrant_api_key,
        collection_name=collection_name,
        vector_dim=embed_model.dimension,
        in_memory=in_memory
    )
    
    if recreate:
        qdrant_repo.ensure_collection(recreate=True)
        
    index_start_t = time.perf_counter()
    indexed_count = qdrant_repo.upsert_chunks(chunks=chunks, vectors=embeddings, batch_size=64)
    index_duration = time.perf_counter() - index_start_t
    
    total_duration = time.perf_counter() - total_start_t
    collection_count = qdrant_repo.count()
    
    telemetry = {
        "source_pdf": pdf_path,
        "source_sha256": source_hash,
        "total_pdf_pages": report.total_pages,
        "total_chapters": len(doc.chapters),
        "total_sections": len(doc.sections),
        "total_schedule_entries": len(sched_entries),
        "total_chunks": len(chunks),
        "substantive_chunks": sum(1 for c in chunks if c.chunk_type == "substantive_section"),
        "schedule_chunks": sum(1 for c in chunks if c.chunk_type == "schedule_entry"),
        "validation_passed": report.is_valid,
        "embedding_model": embed_model.model_name,
        "embedding_dimension": embed_model.dimension,
        "embedding_batch_size": batch_size,
        "embedding_duration_seconds": round(embed_duration, 2),
        "embedding_throughput_chunks_per_sec": round(throughput, 1),
        "qdrant_collection": collection_name,
        "qdrant_indexed_points": indexed_count,
        "qdrant_collection_total_points": collection_count,
        "indexing_duration_seconds": round(index_duration, 2),
        "total_duration_seconds": round(total_duration, 2),
        "failures": 0
    }
    
    logger.info("=== INGESTION & INDEXING COMPLETE ===")
    logger.info(json.dumps(telemetry, indent=2))
    return telemetry


def main():
    parser = argparse.ArgumentParser(description="One-shot Ingestion & Indexing CLI for Nyaya Legal RAG")
    parser.add_argument("--pdf", "--pdf-path", dest="pdf", default="BNS bare act 2023.pdf", help="Path to BNS/BNSS bare act PDF")
    parser.add_argument("--qdrant-url", default=None, help="Qdrant server URL (e.g. http://localhost:6333)")
    parser.add_argument("--qdrant-path", default="./qdrant_storage", help="Local directory for embedded Qdrant storage")
    parser.add_argument("--qdrant-api-key", default=os.getenv("QDRANT_API_KEY", None), help="Qdrant API key for authentication (e.g. Qdrant Cloud)")
    parser.add_argument("--collection", default=settings.qdrant_collection, help="Qdrant collection name")
    parser.add_argument("--recreate", action="store_true", help="Force recreate collection")
    parser.add_argument("--batch-size", type=int, default=32, help="Embedding batch size")
    parser.add_argument("--device", default="cpu", help="Device for PyTorch embeddings (cpu, mps, cuda)")
    parser.add_argument("--in-memory", action="store_true", help="Use in-memory Qdrant storage (testing)")
    
    args = parser.parse_args()
    
    try:
        telemetry = run_ingestion(
            pdf_path=args.pdf,
            qdrant_url=args.qdrant_url,
            qdrant_path=args.qdrant_path if not args.qdrant_url and not args.in_memory else None,
            qdrant_api_key=args.qdrant_api_key,
            collection_name=args.collection,
            recreate=args.recreate,
            batch_size=args.batch_size,
            device=args.device,
            in_memory=args.in_memory
        )
        print("\nIngestion Summary:")
        for k, v in telemetry.items():
            print(f"  {k}: {v}")
    except Exception as e:
        logger.error(f"Ingestion failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
