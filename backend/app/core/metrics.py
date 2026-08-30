"""Prometheus Metrics Collector and Exporter for Nyaya Legal RAG (Part D).

Zero-dependency, thread-safe Prometheus metrics collector exposing official text/plain format.
Maintains strict low-cardinality label rules and enforces token cost accounting.
"""

from dataclasses import dataclass, field
import logging
import threading
import time
from typing import Dict, List, Optional, Tuple

from backend.app.core.config import settings

logger = logging.getLogger("nyaya.core.metrics")


class MetricsCollector:
    """Thread-safe Prometheus metrics collector with zero external dependencies."""

    def __init__(self):
        self._lock = threading.Lock()

        # HTTP counters & histograms
        self._http_requests_total: Dict[Tuple[str, str, int], int] = {}
        self._http_duration_sum: Dict[Tuple[str, str], float] = {}
        self._http_duration_count: Dict[Tuple[str, str], int] = {}

        # Domain counters
        self._chat_requests_total: Dict[str, int] = {}       # intent -> count
        self._chat_duration_sum: Dict[str, float] = {}
        self._chat_duration_count: Dict[str, int] = {}

        self._document_uploads_total: Dict[str, int] = {}    # status -> count
        self._ingestion_jobs_total: Dict[Tuple[str, str], int] = {}  # (status, stage) -> count
        self._ingestion_failures_total: int = 0

        self._retrieval_duration_sum: Dict[str, float] = {}  # retriever_type -> sum
        self._retrieval_duration_count: Dict[str, int] = {}

        self._embedding_duration_sum: float = 0.0
        self._embedding_duration_count: int = 0

        self._refusal_count_total: Dict[str, int] = {}       # reason -> count

        # LLM token & cost accounting
        self._prompt_tokens_total: int = 0
        self._completion_tokens_total: int = 0
        self._total_tokens_total: int = 0
        self._estimated_cost_usd_total: float = 0.0

        # Vector DB Availability Gauge (1.0 = healthy, 0.0 = unavailable)
        self._qdrant_available: float = 1.0

    def record_http_request(self, method: str, endpoint: str, status_code: int, duration_seconds: float) -> None:
        """Record HTTP request observation."""
        with self._lock:
            key = (method, endpoint, status_code)
            self._http_requests_total[key] = self._http_requests_total.get(key, 0) + 1

            dur_key = (method, endpoint)
            self._http_duration_sum[dur_key] = self._http_duration_sum.get(dur_key, 0.0) + duration_seconds
            self._http_duration_count[dur_key] = self._http_duration_count.get(dur_key, 0) + 1

    def record_chat_request(self, intent: str, duration_seconds: float) -> None:
        """Record legal chat query execution."""
        with self._lock:
            self._chat_requests_total[intent] = self._chat_requests_total.get(intent, 0) + 1
            self._chat_duration_sum[intent] = self._chat_duration_sum.get(intent, 0.0) + duration_seconds
            self._chat_duration_count[intent] = self._chat_duration_count.get(intent, 0) + 1

    def record_document_upload(self, status: str) -> None:
        """Record document upload submission."""
        with self._lock:
            self._document_uploads_total[status] = self._document_uploads_total.get(status, 0) + 1

    def record_ingestion_job(self, status: str, stage: str) -> None:
        """Record ingestion job state transition."""
        with self._lock:
            key = (status, stage)
            self._ingestion_jobs_total[key] = self._ingestion_jobs_total.get(key, 0) + 1

    def record_ingestion_failure(self) -> None:
        """Record background ingestion failure."""
        with self._lock:
            self._ingestion_failures_total += 1

    def record_retrieval(self, retriever_type: str, duration_seconds: float) -> None:
        """Record retrieval operation latency."""
        with self._lock:
            self._retrieval_duration_sum[retriever_type] = self._retrieval_duration_sum.get(retriever_type, 0.0) + duration_seconds
            self._retrieval_duration_count[retriever_type] = self._retrieval_duration_count.get(retriever_type, 0) + 1

    def record_embedding(self, duration_seconds: float) -> None:
        """Record embedding model latency."""
        with self._lock:
            self._embedding_duration_sum += duration_seconds
            self._embedding_duration_count += 1

    def record_refusal(self, reason: str) -> None:
        """Record grounded refusal verdict."""
        with self._lock:
            self._refusal_count_total[reason] = self._refusal_count_total.get(reason, 0) + 1

    def record_tokens_and_cost(self, prompt_tokens: int, completion_tokens: int) -> None:
        """Record LLM prompt and completion token counts and compute cost."""
        with self._lock:
            self._prompt_tokens_total += prompt_tokens
            self._completion_tokens_total += completion_tokens
            self._total_tokens_total += (prompt_tokens + completion_tokens)

            # Calculation: (prompt_tokens / 1000 * cost_per_1k_input) + (completion_tokens / 1000 * cost_per_1k_output)
            cost_in = (prompt_tokens / 1000.0) * settings.llm_cost_per_1k_input_tokens
            cost_out = (completion_tokens / 1000.0) * settings.llm_cost_per_1k_output_tokens
            self._estimated_cost_usd_total += (cost_in + cost_out)

    def set_qdrant_availability(self, available: bool) -> None:
        """Update vector DB health gauge."""
        with self._lock:
            self._qdrant_available = 1.0 if available else 0.0

    def generate_prometheus_exposition(self) -> str:
        """Export metrics in official Prometheus exposition text/plain format."""
        lines: List[str] = []

        with self._lock:
            # 1. nyaya_http_requests_total
            lines.append("# HELP nyaya_http_requests_total Total HTTP requests handled.")
            lines.append("# TYPE nyaya_http_requests_total counter")
            if not self._http_requests_total:
                lines.append('nyaya_http_requests_total{endpoint="/api/v1/health",method="GET",status_code="200"} 0')
            else:
                for (m, ep, st), val in self._http_requests_total.items():
                    lines.append(f'nyaya_http_requests_total{{endpoint="{ep}",method="{m}",status_code="{st}"}} {val}')

            # 2. nyaya_http_request_duration_seconds
            lines.append("# HELP nyaya_http_request_duration_seconds_sum Total sum of HTTP request latency in seconds.")
            lines.append("# TYPE nyaya_http_request_duration_seconds_sum counter")
            for (m, ep), s_val in self._http_duration_sum.items():
                lines.append(f'nyaya_http_request_duration_seconds_sum{{endpoint="{ep}",method="{m}"}} {s_val:.6f}')

            lines.append("# HELP nyaya_http_request_duration_seconds_count Total count of HTTP request duration observations.")
            lines.append("# TYPE nyaya_http_request_duration_seconds_count counter")
            for (m, ep), c_val in self._http_duration_count.items():
                lines.append(f'nyaya_http_request_duration_seconds_count{{endpoint="{ep}",method="{m}"}} {c_val}')

            # 3. nyaya_chat_requests_total
            lines.append("# HELP nyaya_chat_requests_total Total legal query chat requests.")
            lines.append("# TYPE nyaya_chat_requests_total counter")
            if not self._chat_requests_total:
                lines.append('nyaya_chat_requests_total{intent="STATUTORY_ONLY"} 0')
            else:
                for intent, cnt in self._chat_requests_total.items():
                    lines.append(f'nyaya_chat_requests_total{{intent="{intent}"}} {cnt}')

            # 4. nyaya_chat_duration_seconds
            lines.append("# HELP nyaya_chat_duration_seconds_sum Total chat processing latency sum in seconds.")
            lines.append("# TYPE nyaya_chat_duration_seconds_sum counter")
            for intent, s_val in self._chat_duration_sum.items():
                lines.append(f'nyaya_chat_duration_seconds_sum{{intent="{intent}"}} {s_val:.6f}')

            # 5. nyaya_document_uploads_total
            lines.append("# HELP nyaya_document_uploads_total Total document upload attempts.")
            lines.append("# TYPE nyaya_document_uploads_total counter")
            if not self._document_uploads_total:
                lines.append('nyaya_document_uploads_total{status="QUEUED"} 0')
            else:
                for st, cnt in self._document_uploads_total.items():
                    lines.append(f'nyaya_document_uploads_total{{status="{st}"}} {cnt}')

            # 6. nyaya_document_ingestion_jobs_total
            lines.append("# HELP nyaya_document_ingestion_jobs_total Total background ingestion job stage transitions.")
            lines.append("# TYPE nyaya_document_ingestion_jobs_total counter")
            if not self._ingestion_jobs_total:
                lines.append('nyaya_document_ingestion_jobs_total{stage="complete",status="READY"} 0')
            else:
                for (st, stg), cnt in self._ingestion_jobs_total.items():
                    lines.append(f'nyaya_document_ingestion_jobs_total{{stage="{stg}",status="{st}"}} {cnt}')

            # 7. nyaya_document_ingestion_failures_total
            lines.append("# HELP nyaya_document_ingestion_failures_total Total background document ingestion failures.")
            lines.append("# TYPE nyaya_document_ingestion_failures_total counter")
            lines.append(f"nyaya_document_ingestion_failures_total {self._ingestion_failures_total}")

            # 8. nyaya_retrieval_duration_seconds
            lines.append("# HELP nyaya_retrieval_duration_seconds_sum Total retrieval latency sum in seconds.")
            lines.append("# TYPE nyaya_retrieval_duration_seconds_sum counter")
            if not self._retrieval_duration_sum:
                lines.append('nyaya_retrieval_duration_seconds_sum{retriever_type="HYBRID"} 0.000000')
            else:
                for r_type, s_val in self._retrieval_duration_sum.items():
                    lines.append(f'nyaya_retrieval_duration_seconds_sum{{retriever_type="{r_type}"}} {s_val:.6f}')

            # 9. nyaya_embedding_duration_seconds
            lines.append("# HELP nyaya_embedding_duration_seconds_sum Total embedding computation latency sum in seconds.")
            lines.append("# TYPE nyaya_embedding_duration_seconds_sum counter")
            lines.append(f"nyaya_embedding_duration_seconds_sum {self._embedding_duration_sum:.6f}")

            # 10. nyaya_refusal_count_total
            lines.append("# HELP nyaya_refusal_count_total Total out-of-scope or ungrounded refusal responses.")
            lines.append("# TYPE nyaya_refusal_count_total counter")
            if not self._refusal_count_total:
                lines.append('nyaya_refusal_count_total{reason="OUT_OF_SCOPE"} 0')
            else:
                for r_reason, cnt in self._refusal_count_total.items():
                    lines.append(f'nyaya_refusal_count_total{{reason="{r_reason}"}} {cnt}')

            # 11. Token accounting
            lines.append("# HELP nyaya_llm_prompt_tokens_total Cumulative count of prompt tokens passed to LLM.")
            lines.append("# TYPE nyaya_llm_prompt_tokens_total counter")
            lines.append(f"nyaya_llm_prompt_tokens_total {self._prompt_tokens_total}")

            lines.append("# HELP nyaya_llm_completion_tokens_total Cumulative count of completion tokens generated by LLM.")
            lines.append("# TYPE nyaya_llm_completion_tokens_total counter")
            lines.append(f"nyaya_llm_completion_tokens_total {self._completion_tokens_total}")

            lines.append("# HELP nyaya_llm_tokens_total Cumulative count of total LLM tokens.")
            lines.append("# TYPE nyaya_llm_tokens_total counter")
            lines.append(f"nyaya_llm_tokens_total {self._total_tokens_total}")

            lines.append("# HELP nyaya_llm_estimated_cost_usd_total Estimated cumulative cost in USD based on provider pricing configuration.")
            lines.append("# TYPE nyaya_llm_estimated_cost_usd_total counter")
            lines.append(f"nyaya_llm_estimated_cost_usd_total {self._estimated_cost_usd_total:.6f}")

            # 12. Qdrant availability gauge
            lines.append("# HELP nyaya_qdrant_available Vector DB availability gauge (1 = available, 0 = unavailable).")
            lines.append("# TYPE nyaya_qdrant_available gauge")
            lines.append(f"nyaya_qdrant_available {self._qdrant_available}")

        return "\n".join(lines) + "\n"


_metrics_collector_instance: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """Singleton provider for MetricsCollector."""
    global _metrics_collector_instance
    if _metrics_collector_instance is None:
        _metrics_collector_instance = MetricsCollector()
    return _metrics_collector_instance
