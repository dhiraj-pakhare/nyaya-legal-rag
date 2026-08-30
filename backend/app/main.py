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
from backend.app.api.routes import api_router
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

    # 2. Register Global Error Handlers (Prevents stack trace / path leakage)
    register_error_handlers(app)

    # 3. Mount API Routers under configured prefix (/api/v1)
    app.include_router(api_router, prefix=settings.api_prefix)

    return app


app = create_app()
