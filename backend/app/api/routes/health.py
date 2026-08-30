"""Health and Readiness Diagnostic API Routes (Phase 8)."""

from fastapi import APIRouter, Depends, Response, status

from backend.app.api.schemas.common import HealthResponseDTO, ReadinessResponseDTO
from backend.app.services.diagnostics_service import (
    DiagnosticsService,
    get_diagnostics_service,
)

router = APIRouter(tags=["Diagnostics"])


@router.get(
    "/health",
    response_model=HealthResponseDTO,
    summary="Process Liveness Probe",
    description="Cheap, non-blocking probe verifying the application process is alive."
)
def get_liveness(
    service: DiagnosticsService = Depends(get_diagnostics_service)
) -> HealthResponseDTO:
    return service.check_liveness()


@router.get(
    "/ready",
    response_model=ReadinessResponseDTO,
    summary="System Readiness Probe",
    description="Deep diagnostic probe inspecting Qdrant, embeddings, forms registry, and LLM provider readiness."
)
def get_readiness(
    response: Response,
    service: DiagnosticsService = Depends(get_diagnostics_service)
) -> ReadinessResponseDTO:
    res = service.check_readiness()
    if res.status != "READY":
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return res
