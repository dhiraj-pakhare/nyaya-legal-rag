#!/usr/bin/env python3
"""Confidence Threshold Calibration Tool for Nyaya Legal RAG.

Sweeps confidence threshold theta in [0.10, 0.90] with step 0.05 across the
golden evaluation dataset (eval/golden_set.jsonl) to calibrate optimal trade-off
between In-Scope Recall vs. Out-of-Scope Refusal Rate.
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
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.app.core.config import settings
from backend.app.core.embeddings import get_embedding_model
from backend.app.core.qdrant_repo import QdrantRepository
from backend.app.ingestion.parser import StatutoryParser
from backend.app.retrieval.pipeline import HybridRetrievalPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")
logger = logging.getLogger("nyaya.calibrate")


def load_golden_set(filepath: str = "eval/golden_set.jsonl") -> List[Dict[str, Any]]:
    """Load golden dataset from jsonl file."""
    dataset = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                dataset.append(json.loads(line))
    return dataset


def run_calibration_sweep(
    pipeline: HybridRetrievalPipeline,
    golden_set: List[Dict[str, Any]],
    thresholds: List[float]
) -> List[Dict[str, Any]]:
    """Evaluate pipeline across candidate confidence thresholds."""
    results = []
    
    # Pre-execute retrieval for each query once to record scores and docs
    query_cache = []
    for item in golden_set:
        query = item["query"]
        res = pipeline.retrieve(query, mode="auto", top_k=5)
        query_cache.append((item, res))

    for theta in thresholds:
        in_scope_total = 0
        in_scope_accepted = 0
        in_scope_refused = 0
        in_scope_correct = 0

        out_of_scope_total = 0
        out_of_scope_refused = 0
        out_of_scope_accepted = 0

        for item, res in query_cache:
            is_refuse_query = item["should_refuse"]
            conf_dict = res.confidence or {}
            conf_score = conf_dict.get("confidence_score", 0.0)
            
            # Re-evaluate decision for active theta
            decision = "ACCEPT" if conf_score >= theta else "REFUSE"

            if is_refuse_query:
                out_of_scope_total += 1
                if decision == "REFUSE":
                    out_of_scope_refused += 1
                else:
                    out_of_scope_accepted += 1
            else:
                in_scope_total += 1
                if decision == "ACCEPT":
                    in_scope_accepted += 1
                    # Check correctness
                    expected_secs = item.get("expected_sections", [])
                    matched = any(
                        doc.section_number in expected_secs or any(doc.section_number.startswith(s) for s in expected_secs)
                        for doc in res.documents
                    )
                    if matched:
                        in_scope_correct += 1
                else:
                    in_scope_refused += 1

        oos_refusal_rate = out_of_scope_refused / max(1, out_of_scope_total)
        false_refusal_rate = in_scope_refused / max(1, in_scope_total)
        in_scope_accuracy = in_scope_correct / max(1, in_scope_total)

        results.append({
            "threshold": round(theta, 2),
            "in_scope_total": in_scope_total,
            "in_scope_accepted": in_scope_accepted,
            "in_scope_refused": in_scope_refused,
            "in_scope_accuracy": round(in_scope_accuracy, 4),
            "false_refusal_rate": round(false_refusal_rate, 4),
            "out_of_scope_total": out_of_scope_total,
            "out_of_scope_refused": out_of_scope_refused,
            "out_of_scope_accepted": out_of_scope_accepted,
            "out_of_scope_refusal_rate": round(oos_refusal_rate, 4)
        })

    return results


def main():
    parser = argparse.ArgumentParser(description="Confidence Threshold Calibration")
    parser.add_argument("--qdrant-path", default="./qdrant_storage", help="Local Qdrant directory")
    parser.add_argument("--pdf-path", default="BNS bare act 2023.pdf", help="Source PDF")
    parser.add_argument("--golden-set", default="eval/golden_set.jsonl", help="Golden set path")
    args = parser.parse_args()

    golden_set = load_golden_set(args.golden_set)
    logger.info(f"Loaded {len(golden_set)} queries from '{args.golden_set}'.")

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

    thresholds = [round(t * 0.05, 2) for t in range(2, 19)]  # 0.10 to 0.90
    logger.info(f"Sweeping thresholds: {thresholds}")
    
    sweep_results = run_calibration_sweep(pipeline, golden_set, thresholds)

    print("\n" + "="*95)
    print("CONFIDENCE THRESHOLD CALIBRATION REPORT")
    print("="*95)
    header = f"{'Threshold (theta)':<18} | {'OOS Refusal Rate':<18} | {'False Refusal Rate':<20} | {'In-Scope Accuracy':<18} | {'Status':<15}"
    print(header)
    print("-" * 95)
    
    best_theta = None
    for row in sweep_results:
        t = row["threshold"]
        oos = f"{row['out_of_scope_refusal_rate']*100:.1f}%"
        frr = f"{row['false_refusal_rate']*100:.1f}%"
        acc = f"{row['in_scope_accuracy']*100:.1f}%"
        
        # Criterion: OOS Refusal >= 95% and False Refusal <= 5%
        is_viable = (row["out_of_scope_refusal_rate"] >= 0.95 and row["false_refusal_rate"] <= 0.05)
        status = "RECOMMENDED" if is_viable and (best_theta is None) else ("VIABLE" if is_viable else "SUB-OPTIMAL")
        if status == "RECOMMENDED":
            best_theta = t

        print(f"{t:<18} | {oos:<18} | {frr:<20} | {acc:<18} | {status:<15}")
    print("="*95)
    print(f"Optimal Calibrated Threshold: theta = {best_theta or 0.35}")
    print("="*95)

    # Save machine readable results
    with open("calibration_results.json", "w") as f:
        json.dump({"best_threshold": best_theta or 0.35, "sweep": sweep_results}, f, indent=2)


if __name__ == "__main__":
    main()
