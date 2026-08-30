"""Diagnostics and Health Application Service (Phase 8).

Implements:
1. /health: Lightweight liveness probe (<1ms)
2. /ready: Deep dependency readiness probe inspecting Qdrant, embeddings, forms registry, and LLM
"""

from datetime import datetime, timezone
import logging
from typing import Any, Dict, List, Optional

from backend.app.api.schemas.common import (
    DependencyStatusDTO,
    HealthResponseDTO,
    ReadinessResponseDTO,
)
from backend.app.core.config import settings
from backend.app.core.embeddings import get_embedding_model
from backend.app.core.qdrant_repo import get_qdrant_repository
from backend.app.forms.repository import get_form_registry
from backend.app.generation.providers import get_llm_provider

logger = logging.getLogger("nyaya.services.diagnostics")


class DiagnosticsService:
    """Application service for system health, readiness, and operational telemetry."""

    def __init__(
        self,
        qdrant_repo: Optional[Any] = None,
        forms_registry: Optional[Any] = None,
        embedding_model: Optional[Any] = None,
        llm_provider: Optional[Any] = None
    ):
        self._qdrant_repo = qdrant_repo
        self._forms_registry = forms_registry
        self._embedding_model = embedding_model
        self._llm_provider = llm_provider

    def check_liveness(self) -> HealthResponseDTO:
        """Perform cheap, non-blocking process liveness check."""
        now_utc = datetime.now(timezone.utc).isoformat()
        return HealthResponseDTO(
            status="UP",
            timestamp=now_utc,
            version="1.0.0"
        )

    def check_readiness(self) -> ReadinessResponseDTO:
        """Perform deep dependency readiness check across storage, vector indices, and registries."""
        now_utc = datetime.now(timezone.utc).isoformat()
        deps: List[DependencyStatusDTO] = []
        all_ready = True

        # 1. Qdrant Statutory Vector DB Check
        try:
            qdrant_repo = self._qdrant_repo or get_qdrant_repository()
            chunk_count = qdrant_repo.count()
            if chunk_count > 0:
                deps.append(
                    DependencyStatusDTO(
                        name="qdrant_vector_store",
                        status="READY",
                        details={"collection": settings.qdrant_collection, "points_indexed": chunk_count}
                    )
                )
            else:
                deps.append(
                    DependencyStatusDTO(
                        name="qdrant_vector_store",
                        status="DEGRADED",
                        details={"collection": settings.qdrant_collection, "points_indexed": 0}
                    )
                )
        except Exception as e:
            logger.error(f"Qdrant readiness probe failed: {str(e)}")
            all_ready = False
            deps.append(
                DependencyStatusDTO(
                    name="qdrant_vector_store",
                    status="UNAVAILABLE",
                    details={"error": "Vector database unreachable"}
                )
            )

        # 2. Statutory Forms Registry Check (Must contain exactly 58 forms)
        try:
            form_reg = self._forms_registry or get_form_registry()
            f_count = form_reg.count()
            if f_count == 58:
                deps.append(
                    DependencyStatusDTO(
                        name="statutory_forms_registry",
                        status="READY",
                        details={"forms_loaded": f_count}
                    )
                )
            else:
                all_ready = False
                deps.append(
                    DependencyStatusDTO(
                        name="statutory_forms_registry",
                        status="DEGRADED",
                        details={"forms_loaded": f_count, "expected": 58}
                    )
                )
        except Exception as e:
            logger.error(f"Forms registry readiness probe failed: {str(e)}")
            all_ready = False
            deps.append(
                DependencyStatusDTO(
                    name="statutory_forms_registry",
                    status="UNAVAILABLE",
                    details={"error": "Forms registry failed initialization"}
                )
            )

        # 3. Dense Embedding Engine Check
        try:
            embed_model = self._embedding_model or get_embedding_model()
            deps.append(
                DependencyStatusDTO(
                    name="embedding_engine",
                    status="READY",
                    details={"dimension": embed_model.dimension, "model_name": settings.embedding_model_name}
                )
            )
        except Exception as e:
            logger.error(f"Embedding model readiness probe failed: {str(e)}")
            all_ready = False
            deps.append(
                DependencyStatusDTO(
                    name="embedding_engine",
                    status="UNAVAILABLE",
                    details={"error": "Embedding model failed initialization"}
                )
            )

        # 4. LLM Provider Configuration Check
        try:
            llm_prov = self._llm_provider or get_llm_provider()
            deps.append(
                DependencyStatusDTO(
                    name="llm_provider",
                    status="READY",
                    details={"provider": getattr(llm_prov, "provider_name", "configured"), "model": getattr(llm_prov, "model", "configured")}
                )
            )
        except Exception as e:
            logger.warning(f"LLM provider check warning: {str(e)}")
            deps.append(
                DependencyStatusDTO(
                    name="llm_provider",
                    status="DEGRADED",
                    details={"error": "LLM provider warning"}
                )
            )

        overall_status = "READY" if all_ready else "UNAVAILABLE"
        return ReadinessResponseDTO(
            status=overall_status,
            dependencies=deps,
            timestamp=now_utc
        )


_diagnostics_service_instance: Optional[DiagnosticsService] = None


def get_diagnostics_service() -> DiagnosticsService:
    """Singleton provider for DiagnosticsService."""
    global _diagnostics_service_instance
    if _diagnostics_service_instance is None:
        _diagnostics_service_instance = DiagnosticsService()
    return _diagnostics_service_instance
