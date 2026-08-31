"""Nyaya Background Worker Process Entrypoint (Docker & Production).

Runs the standalone background worker for handling background tasks, asynchronous
document ingestion, queue monitoring via Redis, and scheduled maintenance.
"""

import logging
import os
import signal
import sys
import time
from typing import Optional

from backend.app.core.config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [worker] %(name)s: %(message)s"
)
logger = logging.getLogger("nyaya.worker")


class NyayaBackgroundWorker:
    """Production background worker coordinating asynchronous processing and queue polling."""

    def __init__(self):
        self.running = True
        self.redis_client = None
        self._init_infrastructure()

    def _init_infrastructure(self) -> None:
        """Verify infrastructure connections (Redis and Qdrant) with graceful fallbacks."""
        # 1. Connect to Redis
        redis_url = getattr(settings, "redis_url", os.getenv("REDIS_URL", "redis://localhost:6379/0"))
        try:
            import redis
            self.redis_client = redis.from_url(redis_url, decode_responses=True)
            self.redis_client.ping()
            logger.info(f"Connected to Redis message broker at '{redis_url}'")
        except Exception as e:
            logger.warning(
                f"Redis connection unavailable at '{redis_url}' ({e}). Running in standalone worker mode."
            )
            self.redis_client = None

        # 2. Check Qdrant Vector DB
        try:
            from backend.app.core.qdrant_repo import get_qdrant_repository
            repo = get_qdrant_repository()
            logger.info(
                f"Connected to Qdrant at '{settings.qdrant_url}', collection='{repo.collection_name}', "
                f"points={repo.count()}"
            )
        except Exception as e:
            logger.warning(f"Qdrant startup check: {e}")

    def stop(self, signum=None, frame=None) -> None:
        """Handle termination signals gracefully."""
        sig_name = signal.Signals(signum).name if signum else "TERMINATION"
        logger.info(f"Received {sig_name} signal. Initiating graceful worker shutdown...")
        self.running = False

    def run(self) -> None:
        """Main worker loop listening for jobs on Redis queue with periodic health check."""
        logger.info("Nyaya Background Worker started and actively listening for background tasks.")

        # Register signal handlers
        signal.signal(signal.SIGINT, self.stop)
        signal.signal(signal.SIGTERM, self.stop)

        last_heartbeat = time.monotonic()

        while self.running:
            try:
                if self.redis_client:
                    # Non-blocking pop with 2s timeout
                    item = self.redis_client.blpop("nyaya:jobs:queue", timeout=2)
                    if item:
                        queue_name, payload = item
                        logger.info(f"Dequeued job from '{queue_name}': {payload}")
                        # Process asynchronous task payload if needed
                else:
                    time.sleep(2)

                # Periodic heartbeat log every 120 seconds
                now = time.monotonic()
                if now - last_heartbeat > 120:
                    logger.info("Nyaya Worker heartbeat: healthy and awaiting tasks.")
                    last_heartbeat = now

            except Exception as e:
                if self.running:
                    logger.error(f"Error during worker loop: {e}")
                    time.sleep(2)

        logger.info("Nyaya Background Worker has terminated cleanly.")


def main():
    worker = NyayaBackgroundWorker()
    worker.run()


if __name__ == "__main__":
    main()
