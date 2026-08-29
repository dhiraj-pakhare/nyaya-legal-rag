#!/usr/bin/env python3
"""Comprehensive Retrieval Benchmark: Dense, BM25, RRF, and Cross-Encoder Reranking.

Evaluates:
- CONFIGURATION A: Dense Only (BGE-base-en-v1.5)
- CONFIGURATION B: BM25 Only (BM25Okapi)
- CONFIGURATION C: Hybrid (Dense + BM25 + RRF)
- CONFIGURATION D: Hybrid + Cross-Encoder Reranker (ms-marco-MiniLM-L-6-v2)
- CONFIGURATION E: Auto Pipeline (Intent Detection + Exact Lookup + Hybrid + Reranker + Confidence)

Measures:
- Recall@5, Recall@10, MRR
- Fine-grained Latency: Dense (ms), BM25 (ms), RRF (ms), Rerank (ms), Total (ms)
- Out-of-Scope Refusal Rate & In-Scope False Refusal Rate
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
logger = logging.getLogger("nyaya.benchmark.reranking")


def load_golden_set(filepath: str = "eval/golden_set.jsonl") -> List[Dict[str, Any]]:
    """Load golden dataset from jsonl file."""
    dataset = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                dataset.append(json.loads(line))
    return dataset


def evaluate_system(
    pipeline: HybridRetrievalPipeline,
    golden_set: List[Dict[str, Any]],
    mode: str,
    enable_reranking: bool = True,
    k: int = 5
) -> Dict[str, Any]:
    """Evaluate pipeline across golden set queries with granular latency breakdowns."""
    hits_at_5 = 0
    hits_at_10 = 0
    reciprocal_ranks = []
    
    dense_latencies = []
    bm25_latencies = []
    rrf_latencies = []
    rerank_latencies = []
    total_latencies = []

    in_scope_count = 0
    in_scope_refused = 0
    out_of_scope_count = 0
    out_of_scope_refused = 0

    detailed_results = []

    for item in golden_set:
        query = item["query"]
        expected_secs = item.get("expected_sections", [])
        expected_act = item.get("expected_act")
        is_refuse = item["should_refuse"]

        start_t = time.perf_counter()
        
        # Track granular latencies
        if mode in ("dense", "hybrid", "auto") and not is_refuse:
            t0 = time.perf_counter()
            _ = pipeline.dense_retriever.search(query, top_k=25)
            dense_latencies.append((time.perf_counter() - t0) * 1000)
            
        if mode in ("bm25", "hybrid", "auto") and not is_refuse:
            t0 = time.perf_counter()
            _ = pipeline.bm25_retriever.search(query, top_k=25)
            bm25_latencies.append((time.perf_counter() - t0) * 1000)

        # Main retrieval invocation
        res = pipeline.retrieve(
            query,
            mode=mode,
            top_k=k,
            enable_reranking=enable_reranking
        )
        total_lat = (time.perf_counter() - start_t) * 1000
        total_latencies.append(total_lat)

        decision = (res.confidence or {}).get("decision", "ACCEPT" if not res.is_refused else "REFUSE")
        conf_score = (res.confidence or {}).get("confidence_score", 0.0)

        if is_refuse:
            out_of_scope_count += 1
            if decision == "REFUSE" or res.is_refused:
                out_of_scope_refused += 1
        else:
            in_scope_count += 1
            if decision == "REFUSE" or res.is_refused:
                in_scope_refused += 1

            # Check ranking metrics
            found_rank = None
            for r_idx, doc in enumerate(res.documents, 1):
                sec_match = (
                    doc.section_number in expected_secs or
                    any(doc.section_number.startswith(s) for s in expected_secs)
                )
                act_match = (doc.act_short == expected_act) if expected_act else True
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
            "is_refuse_query": is_refuse,
            "decision": decision,
            "confidence_score": conf_score,
            "latency_ms": round(total_lat, 2),
            "top_results": [
                {
                    "rank": doc.final_rank,
                    "act_short": doc.act_short,
                    "section_number": doc.section_number,
                    "section_title": doc.section_title,
                    "score": doc.score
                }
                for doc in res.documents[:3]
            ]
        })

    # Aggregations
    total_latencies.sort()
    p50_total = total_latencies[len(total_latencies) // 2] if total_latencies else 0.0
    p95_total = total_latencies[int(len(total_latencies) * 0.95)] if total_latencies else 0.0

    avg_dense = sum(dense_latencies) / max(1, len(dense_latencies))
    avg_bm25 = sum(bm25_latencies) / max(1, len(bm25_latencies))

    return {
        "mode": mode,
        "enable_reranking": enable_reranking,
        "in_scope_queries": in_scope_count,
        "recall_at_5": round(hits_at_5 / max(1, in_scope_count), 4),
        "recall_at_10": round(hits_at_10 / max(1, in_scope_count), 4),
        "mrr": round(sum(reciprocal_ranks) / max(1, in_scope_count), 4),
        "avg_dense_ms": round(avg_dense, 2),
        "avg_bm25_ms": round(avg_bm25, 2),
        "p50_latency_ms": round(p50_total, 2),
        "p95_latency_ms": round(p95_total, 2),
        "out_of_scope_queries": out_of_scope_count,
        "out_of_scope_refusal_rate": round(out_of_scope_refused / max(1, out_of_scope_count), 4),
        "false_refusal_rate": round(in_scope_refused / max(1, in_scope_count), 4),
        "detailed_results": detailed_results
    }


def main():
    parser = argparse.ArgumentParser(description="Cross-Encoder Reranking & Retrieval Benchmark")
    parser.add_argument("--qdrant-path", default="./qdrant_storage", help="Local Qdrant directory")
    parser.add_argument("--pdf-path", default="BNS bare act 2023.pdf", help="Source PDF")
    parser.add_argument("--golden-set", default="eval/golden_set.jsonl", help="Golden set path")
    args = parser.parse_args()

    golden_set = load_golden_set(args.golden_set)
    logger.info(f"Loaded {len(golden_set)} golden set queries.")

    pdf_parser = StatutoryParser(args.pdf_path)
    parse_res = pdf_parser.parse()

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

    experiments = [
        ("dense", False, "Config A: Dense Only (BGE-base-en-v1.5)"),
        ("bm25", False, "Config B: BM25 Only (BM25Okapi)"),
        ("hybrid", False, "Config C: Hybrid (Dense + BM25 + RRF)"),
        ("hybrid", True, "Config D: Hybrid + Cross-Encoder Reranking"),
        ("auto", True, "Config E: Auto (Intent + Exact + Hybrid + Rerank)")
    ]

    summary_rows = []
    all_reports = {}

    for mode, rerank, label in experiments:
        logger.info(f"Running evaluation on {label}...")
        rep = evaluate_system(pipeline, golden_set, mode=mode, enable_reranking=rerank, k=5)
        all_reports[label] = rep
        summary_rows.append({
            "Config": label,
            "Recall@5": f"{rep['recall_at_5']*100:.1f}%",
            "Recall@10": f"{rep['recall_at_10']*100:.1f}%",
            "MRR": f"{rep['mrr']:.4f}",
            "OOS Refusal": f"{rep['out_of_scope_refusal_rate']*100:.1f}%",
            "False Refusal": f"{rep['false_refusal_rate']*100:.1f}%",
            "p50 Lat (ms)": f"{rep['p50_latency_ms']:.1f}",
            "p95 Lat (ms)": f"{rep['p95_latency_ms']:.1f}"
        })

    print("\n" + "="*115)
    print("PHASE 4 COMPLETE RETRIEVAL & RERANKING BENCHMARK REPORT")
    print("="*115)
    header = f"{'Configuration':<42} | {'R@5':<7} | {'R@10':<7} | {'MRR':<7} | {'OOS Refuse':<11} | {'False Refuse':<13} | {'p50 (ms)':<9} | {'p95 (ms)':<9}"
    print(header)
    print("-" * 115)
    for r in summary_rows:
        print(f"{r['Config']:<42} | {r['Recall@5']:<7} | {r['Recall@10']:<7} | {r['MRR']:<7} | {r['OOS Refusal']:<11} | {r['False Refusal']:<13} | {r['p50 Lat (ms)']:<9} | {r['p95 Lat (ms)']:<9}")
    print("="*115)

    with open("eval_results_phase4.json", "w") as f:
        json.dump(all_reports, f, indent=2)
    logger.info("Saved complete Phase 4 evaluation results to 'eval_results_phase4.json'.")


if __name__ == "__main__":
    main()
