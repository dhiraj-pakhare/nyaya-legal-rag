"""Nyaya Legal RAG - API Router Assembly (Phase 8 & Part D)."""

from fastapi import APIRouter

from backend.app.api.routes import documents, forms, health, metrics, query

api_router = APIRouter()

api_router.include_router(health.router)
api_router.include_router(query.router)
api_router.include_router(query.chat_router)
api_router.include_router(documents.router)
api_router.include_router(forms.router)
api_router.include_router(metrics.router)
