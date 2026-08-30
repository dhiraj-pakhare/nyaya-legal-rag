"""Nyaya Legal RAG - Application Services Module (Phase 8)."""

from backend.app.services.diagnostics_service import (
    DiagnosticsService,
    get_diagnostics_service,
)
from backend.app.services.document_service import (
    DocumentManagementService,
    get_document_service,
)
from backend.app.services.forms_service import (
    StatutoryFormsService,
    get_forms_service,
)
from backend.app.services.query_service import (
    LegalQueryService,
    get_query_service,
)

__all__ = [
    "DiagnosticsService",
    "get_diagnostics_service",
    "DocumentManagementService",
    "get_document_service",
    "StatutoryFormsService",
    "get_forms_service",
    "LegalQueryService",
    "get_query_service",
]
