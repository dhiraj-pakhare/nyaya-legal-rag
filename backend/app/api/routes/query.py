"""Unified Legal Query and Streaming API Routes (Phase 8)."""

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from backend.app.api.deps import get_session_scope
from backend.app.api.schemas.query import QueryRequestDTO, QueryResponseDTO
from backend.app.document_rag.models import UserDocumentSessionScope
from backend.app.services.query_service import (
    LegalQueryService,
    get_query_service,
)

router = APIRouter(prefix="/query", tags=["Legal Query"])


@router.post(
    "",
    response_model=QueryResponseDTO,
    summary="Execute Unified Legal Query",
    description="Executes grounded legal reasoning across statutory penal/procedural codes, user documents, or statutory forms with AST citation verification."
)
def execute_legal_query(
    request: QueryRequestDTO,
    scope: UserDocumentSessionScope = Depends(get_session_scope),
    service: LegalQueryService = Depends(get_query_service)
) -> QueryResponseDTO:
    return service.execute_query(scope=scope, request=request)


@router.post(
    "/stream",
    summary="Stream Legal Query with Server-Sent Events (SSE)",
    description="Streams legal generation events using Server-Sent Events (SSE). Emits tokens only after AST citation verification succeeds."
)
async def stream_legal_query(
    request: QueryRequestDTO,
    scope: UserDocumentSessionScope = Depends(get_session_scope),
    service: LegalQueryService = Depends(get_query_service)
) -> StreamingResponse:
    event_generator = service.stream_query(scope=scope, request=request)
    return StreamingResponse(
        event_generator,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )
