"""Security helpers, trusted identity resolvers, and log sanitizers."""

import hashlib
import os
import re
from typing import Any, Dict, Optional

from backend.app.document_rag.models import SecurityScopeError, UserDocumentSessionScope


def resolve_trusted_identity(
    auth_principal: Optional[str] = None,
    session_id: Optional[str] = None,
    active_document_ids: Optional[list] = None
) -> UserDocumentSessionScope:
    """Resolve and enforce trusted server-side identity context.
    
    CRITICAL SECURITY INVARIANT:
    `user_id` MUST NEVER be extracted from unauthenticated client request bodies.
    It must be supplied exclusively by server middleware or verified auth token.
    """
    if not auth_principal or not auth_principal.strip():
        raise SecurityScopeError("Authentication required: Missing or empty server-side user principal.")

    cleaned_user_id = auth_principal.strip()
    cleaned_session_id = session_id.strip() if session_id and session_id.strip() else None
    doc_ids = list(active_document_ids or [])

    scope = UserDocumentSessionScope(
        user_id=cleaned_user_id,
        session_id=cleaned_session_id,
        active_document_ids=doc_ids
    )
    scope.validate_scope()
    return scope


def sanitize_filename(filename: str) -> str:
    """Sanitize client-provided filename to prevent directory traversal or control characters."""
    if not filename:
        return "unnamed_document.pdf"
    # Take basename only
    base = os.path.basename(filename)
    # Remove null bytes and non-printable characters
    clean = re.sub(r'[^\w\s\.-]', '_', base).strip()
    return clean or "unnamed_document.pdf"


def sanitize_for_logs(data: Dict[str, Any]) -> Dict[str, Any]:
    """Sanitize log metadata by stripping raw document text, PII, and sensitive payloads."""
    sanitized: Dict[str, Any] = {}
    sensitive_keys = {"text", "raw_text", "content", "file_bytes", "password", "auth_token"}

    for key, value in data.items():
        if key in sensitive_keys:
            sanitized[key] = "<REDACTED_CONTENT>"
        elif key == "user_id" and isinstance(value, str):
            # Anonymize user_id for public logs using short hash
            sanitized["user_id_hash"] = hashlib.sha256(value.encode()).hexdigest()[:12]
        else:
            sanitized[key] = value

    return sanitized
