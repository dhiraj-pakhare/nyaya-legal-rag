"""Prometheus Metrics API Route (Part D)."""

import logging
from fastapi import APIRouter, Response

from backend.app.core.metrics import get_metrics_collector
from backend.app.core.qdrant_repo import get_qdrant_repository

logger = logging.getLogger("nyaya.api.metrics")

router = APIRouter(prefix="/metrics", tags=["Diagnostics"])


@router.get(
    "",
    summary="Get Prometheus Metrics",
    description="Exposes application operational metrics in official Prometheus exposition format (text/plain)."
)
def get_prometheus_metrics() -> Response:
    """Export operational metrics in text/plain Prometheus exposition format."""
    collector = get_metrics_collector()

    # Update Qdrant availability gauge
    try:
        qdrant_repo = get_qdrant_repository()
        qdrant_ok = qdrant_repo.count() >= 0
        collector.set_qdrant_availability(qdrant_ok)
    except Exception:
        collector.set_qdrant_availability(False)

    content = collector.generate_prometheus_exposition()
    return Response(
        content=content,
        media_type="text/plain; version=0.0.4; charset=utf-8"
    )
