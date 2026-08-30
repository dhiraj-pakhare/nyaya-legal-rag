"""Nyaya Legal RAG - API DTO Schemas Module (Phase 8)."""

from backend.app.api.schemas.common import (
    DependencyStatusDTO,
    ErrorDetailDTO,
    ErrorResponseDTO,
    HealthResponseDTO,
    ReadinessResponseDTO,
)
from backend.app.api.schemas.query import (
    BaseCitationDTO,
    CitationDTO,
    CitationType,
    DocumentCitationDTO,
    FormCitationDTO,
    QueryRequestDTO,
    QueryResponseDTO,
    StatutoryCitationDTO,
    StreamEventDTO,
)
from backend.app.api.schemas.documents import (
    DocumentDetailDTO,
    DocumentIngestResponseDTO,
    DocumentListItemDTO,
)
from backend.app.api.schemas.forms import (
    FormFieldDTO,
    FormLookupRequestDTO,
    FormLookupResponseDTO,
    FormSignatureDTO,
    FormTableHeadDTO,
    StatutoryFormDTO,
)

__all__ = [
    "DependencyStatusDTO",
    "ErrorDetailDTO",
    "ErrorResponseDTO",
    "HealthResponseDTO",
    "ReadinessResponseDTO",
    "BaseCitationDTO",
    "CitationDTO",
    "CitationType",
    "DocumentCitationDTO",
    "FormCitationDTO",
    "QueryRequestDTO",
    "QueryResponseDTO",
    "StatutoryCitationDTO",
    "StreamEventDTO",
    "DocumentDetailDTO",
    "DocumentIngestResponseDTO",
    "DocumentListItemDTO",
    "FormFieldDTO",
    "FormLookupRequestDTO",
    "FormLookupResponseDTO",
    "FormSignatureDTO",
    "FormTableHeadDTO",
    "StatutoryFormDTO",
]
