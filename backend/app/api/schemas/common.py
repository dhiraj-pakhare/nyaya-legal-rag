"""Common DTO schemas for API error handling and health diagnostics (Phase 8)."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ErrorDetailDTO(BaseModel):
    """Structured error payload details."""
    code: str = Field(..., description="Machine-readable error code")
    message: str = Field(..., description="Human-readable safe error message")
    status_code: int = Field(..., description="HTTP status code")
    details: Optional[Any] = Field(default=None, description="Optional safe contextual details")


class ErrorResponseDTO(BaseModel):
    """Standardized error envelope preventing internal path or stack trace leakage."""
    error: ErrorDetailDTO


class HealthResponseDTO(BaseModel):
    """Liveness probe response payload."""
    status: str = Field(default="UP", description="Process liveness state")
    timestamp: str = Field(..., description="ISO 8601 UTC timestamp")
    version: str = Field(default="1.0.0", description="Application version")


class DependencyStatusDTO(BaseModel):
    """Individual system component readiness status."""
    name: str
    status: str                         # "READY" | "DEGRADED" | "UNAVAILABLE"
    details: Optional[Dict[str, Any]] = None


class ReadinessResponseDTO(BaseModel):
    """Readiness probe response payload."""
    status: str                         # "READY" | "UNAVAILABLE"
    dependencies: List[DependencyStatusDTO] = Field(default_factory=list)
    timestamp: str
