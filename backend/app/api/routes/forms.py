"""Statutory Forms API Routes (Phase 8)."""

from fastapi import APIRouter, Depends

from backend.app.api.schemas.forms import (
    FormLookupRequestDTO,
    FormLookupResponseDTO,
    StatutoryFormDTO,
)
from backend.app.services.forms_service import (
    StatutoryFormsService,
    get_forms_service,
)

router = APIRouter(prefix="/forms", tags=["Statutory Forms"])


@router.post(
    "/lookup",
    response_model=FormLookupResponseDTO,
    summary="Deterministic Statutory Form Lookup",
    description="Identifies Second Schedule statutory forms via exact number, section, title, or fuzzy match."
)
def lookup_statutory_form(
    request: FormLookupRequestDTO,
    service: StatutoryFormsService = Depends(get_forms_service)
) -> FormLookupResponseDTO:
    return service.lookup_form(request.query)


@router.get(
    "/{id_or_number}",
    response_model=StatutoryFormDTO,
    summary="Direct Statutory Form Retrieval",
    description="Retrieves a complete statutory form by canonical Form ID (e.g. 'BNSS_FORM_01') or Form Number (e.g. '1')."
)
def get_statutory_form(
    id_or_number: str,
    service: StatutoryFormsService = Depends(get_forms_service)
) -> StatutoryFormDTO:
    return service.get_form_by_id_or_number(id_or_number)
