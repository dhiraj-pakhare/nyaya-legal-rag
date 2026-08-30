"""Thread-Safe Rate Limiter for Scoped Identity Rate Control (Part D)."""

import logging
import threading
import time
from typing import Dict, List, Optional, Tuple
from fastapi import Depends

from backend.app.api.deps import get_session_scope
from backend.app.api.errors import RateLimitExceededError
from backend.app.core.config import settings
from backend.app.document_rag.models import UserDocumentSessionScope

logger = logging.getLogger("nyaya.core.rate_limiter")


class APIRateLimiter:
    """Thread-safe sliding-window rate limiter keying on authenticated principal scope."""

    def __init__(
        self,
        requests_per_minute: Optional[int] = None,
        time_window_seconds: Optional[int] = None,
        enabled: Optional[bool] = None
    ):
        self.requests_per_minute = requests_per_minute or settings.rate_limit_requests_per_minute
        self.time_window_seconds = time_window_seconds or settings.rate_limit_time_window_seconds
        self.enabled = enabled if enabled is not None else settings.rate_limit_enabled
        self._history: Dict[str, List[float]] = {}
        self._lock = threading.Lock()

    def check_rate_limit(self, user_id: str) -> Tuple[bool, int]:
        """Check if request for user_id is within quota. Returns (is_allowed, retry_after_seconds)."""
        if not self.enabled:
            return True, 0

        now = time.monotonic()
        cutoff = now - self.time_window_seconds

        with self._lock:
            timestamps = self._history.get(user_id, [])
            # Prune expired timestamps outside sliding window
            timestamps = [ts for ts in timestamps if ts > cutoff]

            if len(timestamps) >= self.requests_per_minute:
                # Limit exceeded. Calculate retry_after from oldest timestamp in window
                oldest = timestamps[0]
                retry_after = max(1, int(self.time_window_seconds - (now - oldest)))
                self._history[user_id] = timestamps
                return False, retry_after

            # Allow request and record timestamp
            timestamps.append(now)
            self._history[user_id] = timestamps
            return True, 0

    def reset(self) -> None:
        """Reset rate limiter state (useful for test isolation)."""
        with self._lock:
            self._history.clear()


_rate_limiter_instance: Optional[APIRateLimiter] = None


def get_rate_limiter() -> APIRateLimiter:
    """Singleton provider for APIRateLimiter."""
    global _rate_limiter_instance
    if _rate_limiter_instance is None:
        _rate_limiter_instance = APIRateLimiter()
    return _rate_limiter_instance


def enforce_rate_limit(
    scope: UserDocumentSessionScope = Depends(get_session_scope),
    limiter: APIRateLimiter = Depends(get_rate_limiter)
) -> None:
    """FastAPI dependency enforcing rate limits against the authenticated principal identity."""
    is_allowed, retry_after = limiter.check_rate_limit(scope.user_id)
    if not is_allowed:
        logger.warning(f"Rate limit exceeded for user '{scope.user_id}'. Retry after {retry_after}s.")
        raise RateLimitExceededError(
            message=f"Rate limit exceeded for user identity. Please retry after {retry_after} seconds.",
            retry_after_seconds=retry_after
        )
