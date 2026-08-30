"""User Document Lifecycle Management API Routes (Phase 8)."""

from typing import Any, Dict, List
from fastapi import APIRouter, Depends, File, UploadFile, status

from backend.app.api.deps import get_session_scope
from backend.app.api.schemas.documents import (
    DocumentDetailDTO,
    DocumentIngestResponseDTO,
    DocumentListItemDTO,
)
from backend.app.document_rag.models import UserDocumentSessionScope
from backend.app.services.document_service import (
    DocumentManagementService,
    get_document_service,
)

router = APIRouter(prefix="/documents", tags=["User Documents"])


@router.post(
    "",
    response_model=DocumentIngestResponseDTO,
    status_code=status.HTTP_201_CREATED,
    summary="Upload and Ingest User PDF Document",
    description="Synchronously validates, chunks, embeds, and indexes an uploaded PDF document into tenant-isolated storage."
)
async def upload_document(
    file: UploadFile = File(...),
    scope: UserDocumentSessionScope = Depends(get_session_scope),
    service: DocumentManagementService = Depends(get_document_service)
) -> DocumentIngestResponseDTO:
    return await service.upload_and_ingest(scope=scope, file=file)


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
