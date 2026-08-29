#!/usr/bin/env python3
"""Comprehensive Hybrid Retrieval Benchmark for Nyaya Legal RAG.

Evaluates and compares:
- CONFIGURATION A: Dense Only (BGE-base-en-v1.5)
- CONFIGURATION B: BM25 Only (BM25Okapi)
- CONFIGURATION C: Hybrid (Dense + BM25 + Reciprocal Rank Fusion)
- CONFIGURATION D: Auto Pipeline (Intent Detection + Exact Lookup + Hybrid RRF)

Computes: Recall@5, Recall@10, Mean Reciprocal Rank (MRR), and Latency (p50/p95).
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
from backend.app.ingestion.parser import StatutoryParser
from backend.app.retrieval.pipeline import HybridRetrievalPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")
logger = logging.getLogger("nyaya.benchmark.hybrid")

BENCHMARK_DATASET = [
    {
        "id": "Q1_EXACT_SECTION_BNS",
        "category": "Exact section lookup",
        "query": "What is section 103 BNS?",
        "target_act": "BNS",
        "target_section": "103",
        "target_section_variants": ["103", "103(1)", "103(2)"],
        "notes": "Tests exact statutory identifier lookup"
    },
    {
        "id": "Q2_SECTION_VARIANT",
        "category": "Section-number variant",
        "query": "BNS s.103",
        "target_act": "BNS",
        "target_section": "103",
        "target_section_variants": ["103", "103(1)", "103(2)"],
        "notes": "Tests abbreviated section syntax"
    },
    {
        "id": "Q3_IDENTIFIER_CHALLENGE",
        "category": "Exact identifier challenge",
        "query": "Section 105 BNS",
        "target_act": "BNS",
        "target_section": "105",
        "target_section_variants": ["105"],
        "notes": "Tests distinguishing BNS Section 105 from BNSS Section 105"
    },
    {
        "id": "Q4_DIRECT_LEGAL_FACT",
        "category": "Direct factual question",
        "query": "What is the punishment for culpable homicide not amounting to murder?",
        "target_act": "BNS",
        "target_section": "105",
        "target_section_variants": ["105"],
        "notes": "Tests substantive penal lookup"
    },
    {
        "id": "Q5_INDIRECT_SEMANTIC",
        "category": "Indirect semantic question",
        "query": "Under what provision is a person punished for causing someone's death without the intention to murder?",
        "target_act": "BNS",
        "target_section": "105",
        "target_section_variants": ["105"],
        "notes": "Tests indirect conceptual retrieval"
    },
    {
        "id": "Q6_SECTION_TITLE",
        "category": "Section title query",
        "query": "Imprisonment in default of security",
        "target_act": "BNSS",
        "target_section": "141",
        "target_section_variants": ["141"],
        "notes": "Tests exact statutory heading alignment"
    },
    {
        "id": "Q7_DIFFICULT_LEXICAL",
        "category": "Difficult lexical query",
        "query": "Recording of search and seizure through audio-video electronic means",
        "target_act": "BNSS",
        "target_section": "105",
        "target_section_variants": ["105"],
        "notes": "Tests compound technical legal phrase matching"
    },
    {
        "id": "Q8_CITIZEN_ARREST",
        "category": "Indirect procedural query",
        "query": "Can a common citizen or private person apprehend someone who commits a crime in front of them?",
        "target_act": "BNSS",
        "target_section": "40",
        "target_section_variants": ["40"],
        "notes": "Tests arrest by private person without enactive jargon"
    },
    {
        "id": "Q9_POLICE_CUSTODY",
        "category": "Direct procedural question",
        "query": "What is the maximum period of police custody authorised by a Magistrate during investigation?",
        "target_act": "BNSS",
        "target_section": "187",
        "target_section_variants": ["187"],
        "notes": "Tests 15-day custody / 60-90 day detention rule"
    },
    {
        "id": "Q10_EXACT_BNSS_SECTION",
        "category": "Exact section lookup (procedural)",
        "query": "What is section 35 BNSS?",
        "target_act": "BNSS",
        "target_section": "35",
        "target_section_variants": ["35"],
        "notes": "Tests exact BNSS arrest section lookup"
    }
]


def evaluate_configuration(pipeline: HybridRetrievalPipeline, mode: str, k: int = 10) -> Dict[str, Any]:
    """Run full benchmark against a specified retrieval configuration."""
    hits_at_5 = 0
    hits_at_10 = 0
    reciprocal_ranks = []
    latencies = []
    detailed_results = []

    for item in BENCHMARK_DATASET:
        query = item["query"]
        target_act = item["target_act"]
        target_sec = item["target_section"]
        variants = item.get("target_section_variants", [target_sec])
        
        start_t = time.perf_counter()
        retrieval_res = pipeline.retrieve(query, mode=mode, top_k=k)
        lat = (time.perf_counter() - start_t) * 1000
        latencies.append(lat)

        found_rank = None
        for r_idx, doc in enumerate(retrieval_res.documents, 1):
            sec_match = doc.section_number in variants or doc.section_number.startswith(target_sec)
            act_match = (doc.act_short == target_act) if target_act else True
            if sec_match and act_match:
                found_rank = r_idx
                break

        if found_rank is not None:
            if found_rank <= 5:
                hits_at_5 += 1
            if found_rank <= 10:
                hits_at_10 += 1
            reciprocal_ranks.append(1.0 / found_rank)
        else:
            reciprocal_ranks.append(0.0)

        detailed_results.append({
            "id": item["id"],
            "query": query,
            "target": f"{target_act} s.{target_sec}",
            "found_rank": found_rank,
            "latency_ms": round(lat, 2),
            "top_results": [
                {
                    "rank": doc.final_rank,
                    "act_short": doc.act_short,
                    "section_number": doc.section_number,
                    "section_title": doc.section_title,
                    "score": doc.score,
                    "page": doc.page_start,
                    "chunk_id": doc.chunk_id
                }
                for doc in retrieval_res.documents[:3]
            ]
        })

    n = len(BENCHMARK_DATASET)
    latencies.sort()
    p50_lat = latencies[len(latencies) // 2]
    p95_lat = latencies[int(len(latencies) * 0.95)]

    return {
        "mode": mode,
        "total_queries": n,
        "recall_at_5": round(hits_at_5 / n, 4),
        "recall_at_10": round(hits_at_10 / n, 4),
        "mrr": round(sum(reciprocal_ranks) / n, 4),
        "p50_latency_ms": round(p50_lat, 2),
        "p95_latency_ms": round(p95_lat, 2),
        "detailed_results": detailed_results
    }


def main():
    parser = argparse.ArgumentParser(description="Hybrid Retrieval Benchmark Runner")
    parser.add_argument("--qdrant-path", default="./qdrant_storage", help="Path to local Qdrant index")
    parser.add_argument("--pdf-path", default="BNS bare act 2023.pdf", help="Path to source PDF")
    args = parser.parse_args()

    logger.info("Initializing Statutory Parser and chunks...")
    pdf_parser = StatutoryParser(args.pdf_path)
    parse_res = pdf_parser.parse()

    logger.info("Initializing QdrantRepository and EmbeddingModel...")
    qdrant_repo = QdrantRepository(
        path=args.qdrant_path,
        collection_name=settings.qdrant_collection,
        vector_dim=settings.embedding_dimension
    )
    embed_model = get_embedding_model()

    pipeline = HybridRetrievalPipeline(
        chunks=parse_res.chunks,
        qdrant_repo=qdrant_repo,
        embedding_model=embed_model,
        rrf_k=60
    )

    configs = ["dense", "bm25", "hybrid", "auto"]
    config_names = {
        "dense": "Config A: Dense Only (BGE-base-en-v1.5)",
        "bm25": "Config B: BM25 Only (BM25Okapi)",
        "hybrid": "Config C: Hybrid (Dense + BM25 + RRF)",
        "auto": "Config D: Auto (Intent Routing + Exact Lookup + Hybrid RRF)"
    }

    summary_rows = []
    full_reports = {}

    for cfg in configs:
        logger.info(f"Evaluating {config_names[cfg]}...")
        report = evaluate_configuration(pipeline, mode=cfg)
        full_reports[cfg] = report
        summary_rows.append({
            "Configuration": config_names[cfg],
            "Recall@5": f"{report['recall_at_5']*100:.1f}%",
            "Recall@10": f"{report['recall_at_10']*100:.1f}%",
            "MRR": f"{report['mrr']:.4f}",
            "p50 Latency (ms)": f"{report['p50_latency_ms']:.1f}",
            "p95 Latency (ms)": f"{report['p95_latency_ms']:.1f}"
        })

    print("\n" + "="*95)
    print("HYBRID RETRIEVAL BENCHMARK SUMMARY TABLE")
    print("="*95)
    header = f"{'Configuration':<45} | {'Recall@5':<10} | {'Recall@10':<10} | {'MRR':<8} | {'p50 (ms)':<9} | {'p95 (ms)':<9}"
    print(header)
    print("-" * 95)
    for row in summary_rows:
        print(f"{row['Configuration']:<45} | {row['Recall@5']:<10} | {row['Recall@10']:<10} | {row['MRR']:<8} | {row['p50 Latency (ms)']:<9} | {row['p95 Latency (ms)']:<9}")
    print("="*95)

    # Save machine readable report
    out_path = Path("eval_results_hybrid.json")
    with open(out_path, "w") as f:
        json.dump(full_reports, f, indent=2)
    logger.info(f"Full benchmark results saved to '{out_path}'.")


if __name__ == "__main__":
    main()
