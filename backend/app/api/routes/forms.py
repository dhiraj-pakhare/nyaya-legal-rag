"""Statutory Forms API Routes (Part B & Phase 8)."""

from fastapi import APIRouter, Depends, Query, Response
from fastapi.responses import Response

from backend.app.api.schemas.forms import (
    FormLookupRequestDTO,
    FormLookupResponseDTO,
    StatutoryFormDTO,
    StatutoryFormListResponseDTO,
)
from backend.app.services.forms_service import (
    StatutoryFormsService,
    get_forms_service,
)

router = APIRouter(prefix="/forms", tags=["Statutory Forms"])


@router.get(
    "",
    response_model=StatutoryFormListResponseDTO,
    summary="List All Statutory Forms",
    description="Returns all 58 Second Schedule statutory forms with metadata, byte sizes, hashes, and download links."
)
def list_forms(
    service: StatutoryFormsService = Depends(get_forms_service)
) -> StatutoryFormListResponseDTO:
    return service.list_forms()


@router.get(
    "/search",
    response_model=FormLookupResponseDTO,
    summary="Deterministic Search for Statutory Forms",
    description="Searches forms via query parameter by number, section, title, or fuzzy alias."
)
def search_forms(
    q: str = Query(..., min_length=1, max_length=500, description="Search query e.g. 'Form 1', 'Section 35(3)', 'Bail bond'"),
    service: StatutoryFormsService = Depends(get_forms_service)
) -> FormLookupResponseDTO:
    return service.search_forms(q)


@router.get(
    "/download-all",
    summary="Bulk Download All Statutory Forms as ZIP",
    description="Returns a ZIP archive containing all 58 Second Schedule statutory form PDFs."
)
def download_all_forms(
    service: StatutoryFormsService = Depends(get_forms_service)
) -> Response:
    filename, zip_bytes = service.get_bulk_forms_zip()
    return Response(
        content=zip_bytes,
        media_type="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"'
        }
    )


@router.post(
    "/lookup",
    response_model=FormLookupResponseDTO,
    summary="Deterministic Statutory Form Lookup (POST)",
    description="Identifies Second Schedule statutory forms via exact number, section, title, or fuzzy match."
)
def lookup_statutory_form(
    request: FormLookupRequestDTO,
    service: StatutoryFormsService = Depends(get_forms_service)
) -> FormLookupResponseDTO:
    return service.lookup_form(request.query)


@router.get(
    "/{id_or_number}/download",
    summary="Download Individual Statutory Form PDF",
    description="Downloads the vector/text PDF file for a specific statutory form (e.g. '1', '33', 'BNSS_FORM_01')."
)
def download_form_pdf(
    id_or_number: str,
    service: StatutoryFormsService = Depends(get_forms_service)
) -> Response:
    filename, pdf_bytes = service.get_form_pdf_bytes(id_or_number)
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"'
        }
    )


@router.get(
    "/{id_or_number}",
    response_model=StatutoryFormDTO,
    summary="Direct Statutory Form JSON Metadata Retrieval",
    description="Retrieves structured JSON model for a specific statutory form by ID (e.g. 'BNSS_FORM_01') or number ('1')."
)
def get_statutory_form(
    id_or_number: str,
    service: StatutoryFormsService = Depends(get_forms_service)
) -> StatutoryFormDTO:
    return service.get_form_by_id_or_number(id_or_number)
