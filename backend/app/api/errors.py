"""Standardized API Error Classes and Global Exception Handlers (Phase 8).

Ensures all errors follow a uniform JSON structure and prevents internal filesystem
paths, credentials, and raw stack traces from leaking to clients.
"""

import logging
from typing import Any, Dict, Optional
from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from backend.app.api.schemas.common import ErrorDetailDTO, ErrorResponseDTO

logger = logging.getLogger("nyaya.api.errors")


class APIError(Exception):
    """Base API exception with typed error code and status."""
    def __init__(
        self,
        message: str,
        code: str = "INTERNAL_ERROR",
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
        details: Optional[Any] = None
    ):
        super().__init__(message)
        self.message = message
        self.code = code
        self.status_code = status_code
        self.details = details


class UnauthorizedError(APIError):
    def __init__(self, message: str = "Authentication required.", details: Optional[Any] = None):
        super().__init__(
            message=message,
            code="UNAUTHORIZED",
            status_code=status.HTTP_401_UNAUTHORIZED,
            details=details
        )


class ForbiddenError(APIError):
    def __init__(self, message: str = "Access forbidden.", details: Optional[Any] = None):
        super().__init__(
            message=message,
            code="FORBIDDEN",
            status_code=status.HTTP_403_FORBIDDEN,
            details=details
        )


class NotFoundError(APIError):
    def __init__(self, message: str = "Resource not found or inaccessible.", details: Optional[Any] = None):
        super().__init__(
            message=message,
            code="NOT_FOUND",
            status_code=status.HTTP_404_NOT_FOUND,
            details=details
        )


class DocumentNotFoundError(NotFoundError):
    """Uniform 404 for unowned or missing documents to prevent existence enumeration."""
    def __init__(self, message: str = "Document not found or inaccessible."):
        super().__init__(message=message, details=None)
        self.code = "DOCUMENT_NOT_FOUND"


class PayloadTooLargeError(APIError):
    def __init__(self, message: str = "File size exceeds maximum permitted limit.", details: Optional[Any] = None):
        super().__init__(
            message=message,
            code="FILE_TOO_LARGE",
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            details=details
        )


class UnsupportedMediaTypeError(APIError):
    def __init__(self, message: str = "Unsupported media type. Only PDF files are supported.", details: Optional[Any] = None):
        super().__init__(
            message=message,
            code="UNSUPPORTED_MEDIA_TYPE",
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            details=details
        )


class ValidationFailedError(APIError):
    def __init__(self, message: str = "Citation or claim validation failed.", details: Optional[Any] = None):
        super().__init__(
            message=message,
            code="VALIDATION_FAILED",
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            details=details
        )


class RateLimitExceededError(APIError):
    def __init__(self, message: str = "Rate limit exceeded. Please retry later.", retry_after_seconds: int = 60):
        super().__init__(
            message=message,
            code="RATE_LIMIT_EXCEEDED",
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            details={"retry_after_seconds": retry_after_seconds}
        )
        self.retry_after_seconds = retry_after_seconds


class ServiceUnavailableError(APIError):
    def __init__(self, message: str = "Required system dependencies are unavailable.", details: Optional[Any] = None):
        super().__init__(
            message=message,
            code="SERVICE_UNAVAILABLE",
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            details=details
        )


def register_error_handlers(app: FastAPI) -> None:
    """Attach global exception handlers to the FastAPI application."""

    @app.exception_handler(APIError)
    async def api_error_handler(request: Request, exc: APIError) -> JSONResponse:
        logger.warning(f"API Error [{exc.code}] on {request.method} {request.url.path}: {exc.message}")
        payload = ErrorResponseDTO(
            error=ErrorDetailDTO(
                code=exc.code,
                message=exc.message,
                status_code=exc.status_code,
                details=exc.details
            )
        )
        headers = {}
        if hasattr(exc, "retry_after_seconds") and exc.retry_after_seconds:
            headers["Retry-After"] = str(exc.retry_after_seconds)

        return JSONResponse(
            status_code=exc.status_code,
            content=payload.model_dump(),
            headers=headers if headers else None
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        code_map = {
            400: "INVALID_REQUEST",
            401: "UNAUTHORIZED",
            403: "FORBIDDEN",
            404: "NOT_FOUND",
            405: "METHOD_NOT_ALLOWED",
            413: "FILE_TOO_LARGE",
            415: "UNSUPPORTED_MEDIA_TYPE",
            422: "VALIDATION_FAILED",
            500: "INTERNAL_ERROR",
            503: "SERVICE_UNAVAILABLE",
        }
        err_code = code_map.get(exc.status_code, "HTTP_ERROR")
        payload = ErrorResponseDTO(
            error=ErrorDetailDTO(
                code=err_code,
                message=str(exc.detail),
                status_code=exc.status_code,
                details=None
            )
        )
        return JSONResponse(
            status_code=exc.status_code,
            content=payload.model_dump()
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        logger.info(f"Request Validation Error on {request.method} {request.url.path}: {exc.errors()}")
        # Sanitize validation errors to avoid leaking internal structures
        sanitized_details = []
        for err in exc.errors():
            loc = " -> ".join(str(l) for l in err.get("loc", []))
            sanitized_details.append({"location": loc, "message": err.get("msg", "Invalid parameter")})

        payload = ErrorResponseDTO(
            error=ErrorDetailDTO(
                code="INVALID_REQUEST",
                message="Request parameter validation failed.",
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                details=sanitized_details
            )
        )
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=payload.model_dump()
        )

    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception(f"Unhandled Exception on {request.method} {request.url.path}: {str(exc)}")
        payload = ErrorResponseDTO(
            error=ErrorDetailDTO(
                code="INTERNAL_ERROR",
                message="An internal server error occurred.",
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                details=None
            )
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=payload.model_dump()
        )
