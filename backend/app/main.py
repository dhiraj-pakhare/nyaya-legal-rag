"""FastAPI Main Application Entrypoint for Nyaya Legal RAG (Phase 8).

Assembles API routers, registers global error handlers, configures CORS, and
manages application lifespan events.
"""

from contextlib import asynccontextmanager
import logging
from typing import AsyncGenerator
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api.errors import register_error_handlers
from backend.app.api.routes import api_router, health
from backend.app.core.config import settings
from backend.app.forms.repository import get_form_registry

logger = logging.getLogger("nyaya.main")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan context manager for startup initialization and graceful shutdown."""
    logger.info("Initializing Nyaya Legal RAG Application...")
    # Pre-load Second Schedule statutory forms registry
    try:
        registry = get_form_registry(settings.pdf_path)
        logger.info(f"Loaded {registry.count()} statutory forms from {settings.pdf_path}")
    except Exception as e:
        logger.warning(f"Could not pre-load statutory forms registry on startup: {str(e)}")

    yield

    logger.info("Shutting down Nyaya Legal RAG Application.")


import time
from fastapi import Request
from backend.app.core.metrics import get_metrics_collector


def create_app() -> FastAPI:
    """Application factory for Nyaya Legal RAG API Gateway."""
    app = FastAPI(
        title="Nyaya Legal RAG API",
        description=(
            "Enterprise-grade statutory retrieval-augmented generation and legal forms engine "
            "for the Bharatiya Nagarik Suraksha Sanhita (BNSS) and Bharatiya Nyaya Sanhita (BNS)."
        ),
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc"
    )

    # 1. Configure CORS
    cors_origins = [o.strip() for o in settings.cors_origins.split(",") if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins if cors_origins else ["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 2. HTTP Metrics Middleware (Low cardinality endpoints)
    @app.middleware("http")
    async def metrics_middleware(request: Request, call_next):
        start_time = time.monotonic()
        response = await call_next(request)
        duration = time.monotonic() - start_time

        endpoint = request.url.path
        if endpoint.startswith("/api/v1/documents/") and endpoint != "/api/v1/documents/upload":
            parts = endpoint.split("/")
            if len(parts) >= 5 and parts[4] == "status":
                endpoint = "/api/v1/documents/{id}/status"
            elif len(parts) >= 4:
                endpoint = "/api/v1/documents/{id}"
        elif endpoint.startswith("/api/v1/forms/"):
            endpoint = "/api/v1/forms/{id}"

        collector = get_metrics_collector()
        collector.record_http_request(request.method, endpoint, response.status_code, duration)
        return response

    # 3. Register Global Error Handlers (Prevents stack trace / path leakage)
    register_error_handlers(app)

    # 4. Mount API Routers under configured prefix (/api/v1)
    app.include_router(api_router, prefix=settings.api_prefix)
    # Also mount health router at root for direct /health and /ready probes
    app.include_router(health.router)

    return app


app = create_app()
