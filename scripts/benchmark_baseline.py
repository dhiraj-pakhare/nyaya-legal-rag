#!/usr/bin/env python3
"""Dense Retrieval Baseline Benchmark for Nyaya Legal RAG.

Evaluates dense-only vector retrieval across 5 canonical statutory query types:
1. Exact section-number query
2. Section-title query
3. Direct factual query
4. Indirectly phrased legal query
5. Identifier challenge query (where dense retrieval struggles without lexical matching)
"""

import os
os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.app.core.config import settings
from backend.app.core.embeddings import get_embedding_model
from backend.app.core.qdrant_repo import QdrantRepository
from backend.app.core.retrieval_baseline import search_dense

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")
logger = logging.getLogger("nyaya.benchmark")

BENCHMARK_QUERIES = [
    {
        "id": "Q1_EXACT_SECTION",
        "category": "Exact section-number query",
        "query": "Section 35 arrest of person without warrant by police officer",
        "target_act": "BNSS",
        "target_section": "35",
        "notes": "Tests exact statutory identifier matching alongside descriptive text"
    },
    {
        "id": "Q2_SECTION_TITLE",
        "category": "Section-title query",
        "query": "Imprisonment in default of security",
        "target_act": "BNSS",
        "target_section": "141",
        "notes": "Tests exact title alignment with marginal note AST title"
    },
    {
        "id": "Q3_DIRECT_FACTUAL",
        "category": "Direct factual legal query",
        "query": "What is the maximum period of police custody authorised by a Magistrate during investigation?",
        "target_act": "BNSS",
        "target_section": "187",
        "notes": "Tests semantic retrieval of the 15-day custody / 60-90 day detention rule"
    },
    {
        "id": "Q4_INDIRECT_CONCEPTUAL",
        "category": "Indirectly phrased legal query",
        "query": "Can a common citizen or private person apprehend someone who commits a crime in front of them?",
        "target_act": "BNSS",
        "target_section": "44",
        "notes": "Tests semantic mapping of 'private person arrest' without verbatim enactment keywords"
    },
    {
        "id": "Q5_IDENTIFIER_CHALLENGE",
        "category": "Identifier challenge query (Dense limitation test)",
        "query": "What is the punishment and bailability for offence under Section 105?",
        "target_act": "BNS",
        "target_section": "105",
        "notes": "Dense retrieval typically struggles on bare number queries without hybrid lexical matching"
    }
]


def run_benchmark(
    qdrant_url: Optional[str] = None,
    qdrant_path: Optional[str] = "./qdrant_storage",
    collection_name: str = settings.qdrant_collection,
    top_k: int = 5
) -> dict:
    """Run dense retrieval benchmark across test queries and record scores and ranks."""
    embed_model = get_embedding_model()
    repo = QdrantRepository(
        url=qdrant_url,
        path=qdrant_path if not qdrant_url else None,
        collection_name=collection_name,
        vector_dim=embed_model.dimension
    )
    
    total_points = repo.count()
    logger.info(f"Connected to Qdrant collection '{collection_name}' with {total_points} points.")
    
    results = []
    for bq in BENCHMARK_QUERIES:
        q_text = bq["query"]
        target_sec = bq["target_section"]
        
        start_t = time.perf_counter()
        search_res = search_dense(
            query=q_text,
            repo=repo,
            embedding_model=embed_model,
            top_k=top_k
        )
        latency_ms = (time.perf_counter() - start_t) * 1000
        
        retrieved_list = []
        target_rank = None
        target_found = False
        target_score = None
        
        for rank, item in enumerate(search_res, 1):
            is_match = (item.section_number == target_sec)
            if is_match and not target_found:
                target_found = True
                target_rank = rank
                target_score = item.score
                
            retrieved_list.append({
                "rank": rank,
                "act_short": item.act_short,
                "section_number": item.section_number,
                "section_title": item.section_title,
                "score": round(item.score, 4),
                "chunk_id": item.chunk_id,
                "is_target": is_match
            })
            
        results.append({
            "id": bq["id"],
            "category": bq["category"],
            "query": q_text,
            "target_act": bq["target_act"],
            "target_section": target_sec,
            "target_found": target_found,
            "target_rank": target_rank,
            "target_score": round(target_score, 4) if target_score else None,
            "latency_ms": round(latency_ms, 2),
            "top_retrieved": retrieved_list
        })
        
    return {
        "collection": collection_name,
        "total_indexed_points": total_points,
        "embedding_model": embed_model.model_name,
        "queries_tested": len(BENCHMARK_QUERIES),
        "results": results
    }


def main():
    parser = argparse.ArgumentParser(description="Dense Retrieval Baseline Benchmark")
    parser.add_argument("--qdrant-url", default=None, help="Qdrant server URL")
    parser.add_argument("--qdrant-path", default="./qdrant_storage", help="Local Qdrant directory")
    parser.add_argument("--collection", default=settings.qdrant_collection, help="Collection name")
    parser.add_argument("--top-k", type=int, default=5, help="Top K results")
    args = parser.parse_args()
    
    report = run_benchmark(
        qdrant_url=args.qdrant_url,
        qdrant_path=args.qdrant_path,
        collection_name=args.collection,
        top_k=args.top_k
    )
    
    print("\n" + "="*80)
    print("DENSE RETRIEVAL BASELINE BENCHMARK REPORT")
    print("="*80)
    print(f"Collection: {report['collection']} ({report['total_indexed_points']} points)")
    print(f"Model: {report['embedding_model']}")
    print("-" * 80)
    
    for r in report["results"]:
        status = f"FOUND at Rank #{r['target_rank']} (Score: {r['target_score']})" if r["target_found"] else "NOT IN TOP-K"
        print(f"[{r['id']}] {r['category']}")
        print(f"Query: \"{r['query']}\"")
        print(f"Target: {r['target_act']} Section {r['target_section']} -> {status} (Latency: {r['latency_ms']} ms)")
        print("Top 3 Retrieved:")
        for item in r["top_retrieved"][:3]:
            marker = " -> [TARGET]" if item["is_target"] else ""
            print(f"   #{item['rank']} [{item['act_short']} s.{item['section_number']}] {item['section_title']} (score: {item['score']}){marker}")
        print("-" * 80)


if __name__ == "__main__":
    main()
