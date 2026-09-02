"""User Document Lifecycle Management API Routes (Part D & Phase 8)."""

from typing import Any, Dict, List
from fastapi import APIRouter, Depends, File, UploadFile, status

from backend.app.api.deps import get_session_scope
from backend.app.api.schemas.documents import (
    DocumentCancellationResponseDTO,
    DocumentDetailDTO,
    DocumentListItemDTO,
    DocumentStatusDTO,
    DocumentUploadResponseDTO,
)
from backend.app.document_rag.models import UserDocumentSessionScope
from backend.app.services.document_service import (
    DocumentManagementService,
    get_document_service,
)

from backend.app.core.rate_limiter import enforce_rate_limit

router = APIRouter(prefix="/documents", tags=["User Documents"])


@router.post(
    "/upload",
    response_model=DocumentUploadResponseDTO,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(enforce_rate_limit)],
    summary="Asynchronous User PDF Upload",
    description="Submits an uploaded PDF document for asynchronous background extraction, chunking, embedding, and vector indexing."
)
@router.post(
    "",
    response_model=DocumentUploadResponseDTO,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(enforce_rate_limit)],
    summary="Asynchronous User PDF Upload (Alias)",
    description="Submits an uploaded PDF document for asynchronous background extraction, chunking, embedding, and vector indexing."
)
async def upload_document(
    file: UploadFile = File(...),
    scope: UserDocumentSessionScope = Depends(get_session_scope),
    service: DocumentManagementService = Depends(get_document_service)
) -> DocumentUploadResponseDTO:
    return await service.upload_and_ingest(scope=scope, file=file)


@router.get(
    "/{document_id}/status",
    response_model=DocumentStatusDTO,
    summary="Get Ingestion Job Status",
    description="Polls async background ingestion job status and progress stage for a document. Enforces uniform 404 on unowned or missing IDs."
)
def get_document_status(
    document_id: str,
    scope: UserDocumentSessionScope = Depends(get_session_scope),
    service: DocumentManagementService = Depends(get_document_service)
) -> DocumentStatusDTO:
    return service.get_document_status(scope=scope, document_id_or_job_id=document_id)


@router.post(
    "/{job_id}/cancel",
    response_model=DocumentCancellationResponseDTO,
    summary="Cancel Ingestion Job",
    description="Cancels an active or pending background document ingestion job. Returns idempotent response if already finished, failed, or cancelled."
)
def cancel_document_ingestion(
    job_id: str,
    scope: UserDocumentSessionScope = Depends(get_session_scope),
    service: DocumentManagementService = Depends(get_document_service)
) -> DocumentCancellationResponseDTO:
    return service.cancel_document_ingestion(scope=scope, document_id_or_job_id=job_id)


@router.get(
    "",
    response_model=List[DocumentListItemDTO],
    summary="List User Documents",
    description="Returns all active documents owned by the authenticated principal."
)
def list_documents(
    scope: UserDocumentSessionScope = Depends(get_session_scope),
    service: DocumentManagementService = Depends(get_document_service)
) -> List[DocumentListItemDTO]:
    return service.list_documents(scope=scope)


@router.get(
    "/{document_id}",
    response_model=DocumentDetailDTO,
    summary="Get Document Metadata",
    description="Retrieves metadata for a specific document. Returns uniform 404 if not found or unowned."
)
def get_document(
    document_id: str,
    scope: UserDocumentSessionScope = Depends(get_session_scope),
    service: DocumentManagementService = Depends(get_document_service)
) -> DocumentDetailDTO:
    return service.get_document(scope=scope, document_id=document_id)


@router.delete(
    "/{document_id}",
    response_model=Dict[str, Any],
    summary="Delete Document",
    description="Deletes document vectors, BM25 indices, and metadata. Returns uniform 404 if not found or unowned."
)
def delete_document(
    document_id: str,
    scope: UserDocumentSessionScope = Depends(get_session_scope),
    service: DocumentManagementService = Depends(get_document_service)
) -> Dict[str, Any]:
    return service.delete_document(scope=scope, document_id=document_id)
