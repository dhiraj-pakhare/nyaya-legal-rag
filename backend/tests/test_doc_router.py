"""Tests for Query Intent Router."""

from backend.app.document_rag.models import QueryIntent, UserDocumentSessionScope
from backend.app.document_rag.router import QueryIntentRouter


def test_router_statutory_only():
    """Test routing pure statutory questions."""
    router = QueryIntentRouter()
    scope = UserDocumentSessionScope(user_id="user_1", active_document_ids=["doc_1"])

    decision = router.route("What is the punishment for murder under section 103 BNS?", scope)
    assert decision.intent == QueryIntent.STATUTORY_ONLY
    assert decision.confidence >= 0.90


def test_router_document_only():
    """Test routing pure user-document questions."""
    router = QueryIntentRouter()
    scope = UserDocumentSessionScope(user_id="user_1", active_document_ids=["doc_1"])

    decision = router.route("What does my uploaded notice say about the arbitration clause?", scope)
    assert decision.intent == QueryIntent.DOCUMENT_ONLY
    assert decision.confidence >= 0.90


def test_router_combined():
    """Test routing questions connecting document facts to statutory law."""
    router = QueryIntentRouter()
    scope = UserDocumentSessionScope(user_id="user_1", active_document_ids=["doc_1"])

    decision = router.route("Does the extortion allegation in my FIR fall under section 308 BNS?", scope)
    assert decision.intent == QueryIntent.COMBINED
    assert decision.confidence >= 0.90


def test_router_no_active_documents_defaults_to_statutory():
    """Test that if user has 0 active documents, queries default strictly to statutory."""
    router = QueryIntentRouter()
    scope = UserDocumentSessionScope(user_id="user_1", active_document_ids=[])

    decision = router.route("What does the notice say?", scope)
    assert decision.intent == QueryIntent.STATUTORY_ONLY


def test_router_ambiguous_query_with_active_documents():
    """Test that ambiguous queries with active documents safely route to combined."""
    router = QueryIntentRouter()
    scope = UserDocumentSessionScope(user_id="user_1", active_document_ids=["doc_1"])

    decision = router.route("Is this an offence?", scope)
    assert decision.intent == QueryIntent.COMBINED
