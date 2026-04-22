"""Connection pool and rate limiting for external service calls.

Provides semaphores and connection pooling to prevent resource exhaustion
when calling external services (LM Studio, ChromaDB, etc.).
"""

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from app.core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class RateLimiterConfig:
    """Configuration for rate limiter."""

    max_concurrent: int = 10  # Max concurrent requests
    rate_per_second: float = 0  # Rate limit (0 = unlimited)
    burst_size: int = 0  # Burst capacity (0 = unlimited)
    timeout_seconds: float = 30.0  # Max wait time for semaphore


class RateLimiter:
    """Async rate limiter with token bucket algorithm.

    Supports both simple concurrency limiting and rate limiting.
    """

    def __init__(self, name: str, config: Optional[RateLimiterConfig] = None):
        self.name = name
        self.config = config or RateLimiterConfig()
        self._semaphore: Optional[asyncio.Semaphore] = None
        self._tokens: float = 0
        self._last_update: float = time.time()
        self._lock = asyncio.Lock()
        self._waiters: deque[tuple[asyncio.Future, float]] = deque()
        self._active_requests: int = 0

        # Initialize semaphore for concurrency limiting
        if self.config.max_concurrent > 0:
            self._semaphore = asyncio.Semaphore(self.config.max_concurrent)

        # Initialize token bucket
        if self.config.rate_per_second > 0:
            self._tokens = float(self.config.burst_size or self.config.max_concurrent)

        logger.info(
            f"Rate limiter '{name}' initialized: max_concurrent={self.config.max_concurrent}, "
            f"rate={self.config.rate_per_second}/s, burst={self.config.burst_size}"
        )

    @property
    def active_requests(self) -> int:
        return self._active_requests

    def _refill_tokens(self):
        """Refill token bucket based on elapsed time."""
        now = time.time()
        elapsed = now - self._last_update
        self._last_update = now

        if self.config.rate_per_second > 0:
            self._tokens = min(
                self.config.burst_size or float(self.config.max_concurrent),
                self._tokens + elapsed * self.config.rate_per_second,
            )

    async def acquire(self) -> bool:
        """Acquire permission to make a request.

        Returns:
            True if permission granted, False if timeout

        Raises:
            asyncio.TimeoutError: If timeout exceeded
        """
        # First acquire semaphore (concurrency limit)
        if self._semaphore:
            try:
                await asyncio.wait_for(
                    self._semaphore.acquire(),
                    timeout=self.config.timeout_seconds,
                )
            except asyncio.TimeoutError:
                logger.warning(f"Rate limiter '{self.name}': semaphore timeout")
                return False

        # Then check token bucket (rate limit)
        if self.config.rate_per_second > 0:
            async with self._lock:
                self._refill_tokens()
                if self._tokens >= 1:
                    self._tokens -= 1
                else:
                    # Wait for token
                    wait_time = (1 - self._tokens) / self.config.rate_per_second
                    if wait_time > self.config.timeout_seconds:
                        logger.warning(f"Rate limiter '{self.name}': rate limit timeout")
                        if self._semaphore:
                            self._semaphore.release()
                        return False

                    await asyncio.sleep(wait_time)
                    self._tokens = 0

        self._active_requests += 1
        return True

    def release(self):
        """Release resources after request completion."""
        self._active_requests -= 1
        if self._semaphore:
            self._semaphore.release()

    async def __aenter__(self):
        await self.acquire()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.release()
        return False


# Global rate limiters for each service
_rate_limiters: dict[str, RateLimiter] = {}


def get_rate_limiter(name: str) -> RateLimiter:
    """Get or create a rate limiter for a service."""
    if name not in _rate_limiters:
        # Default configurations per service type
        configs = {
            "vlm": RateLimiterConfig(
                max_concurrent=settings.MAX_CONCURRENT_VLM_CALLS,
                timeout_seconds=settings.VLM_TIMEOUT_SECONDS,
            ),
            "rag": RateLimiterConfig(
                max_concurrent=settings.MAX_CONCURRENT_RAG_CALLS,
                timeout_seconds=settings.RAG_TIMEOUT_SECONDS,
            ),
            "embedding": RateLimiterConfig(
                max_concurrent=settings.MAX_CONCURRENT_RAG_CALLS,
                timeout_seconds=10.0,
            ),
            "llm": RateLimiterConfig(
                max_concurrent=settings.MAX_CONCURRENT_REQUESTS,
                timeout_seconds=settings.REQUEST_TIMEOUT,
            ),
        }

        config = configs.get(name, RateLimiterConfig())
        _rate_limiters[name] = RateLimiter(name, config)
        logger.info(f"Created rate limiter for '{name}'")

    return _rate_limiters[name]


def get_all_rate_limiters() -> dict[str, dict]:
    """Get status of all rate limiters."""
    return {
        name: {"active_requests": limiter.active_requests, "config": limiter.config}
        for name, limiter in _rate_limiters.items()
    }


class ConnectionPool:
    """Simple connection pool for HTTP clients.

    Manages a pool of reusable connections to external services.
    """

    def __init__(
        self,
        name: str,
        max_size: int = 10,
        max_idle_seconds: float = 60.0,
    ):
        self.name = name
        self.max_size = max_size
        self.max_idle_seconds = max_idle_seconds
        self._pool: asyncio.Queue = asyncio.Queue(maxsize=max_size)
        self._created: int = 0
        self._in_use: int = 0
        self._lock = asyncio.Lock()
        self._last_cleanup = time.time()

    async def acquire(self):
        """Acquire a connection from the pool."""
        # Try to get existing connection
        try:
            conn = self._pool.get_nowait()
            # Check if stale
            if hasattr(conn, "_created") and time.time() - conn._created > self.max_idle_seconds:
                await self._close_connection(conn)
                conn = None
        except asyncio.QueueEmpty:
            conn = None

        # Create new if needed
        if conn is None:
            async with self._lock:
                if self._created < self.max_size:
                    conn = await self._create_connection()
                    self._created += 1

        if conn:
            self._in_use += 1
            return conn

        # Wait for available connection
        conn = await asyncio.wait_for(self._pool.get(), timeout=30.0)
        self._in_use += 1
        return conn

    async def release(self, conn):
        """Release a connection back to the pool."""
        self._in_use -= 1
        try:
            self._pool.put_nowait(conn)
        except asyncio.QueueFull:
            await self._close_connection(conn)

    async def _create_connection(self):
        """Create a new connection (override in subclass)."""
        return {"_created": time.time()}

    async def _close_connection(self, conn):
        """Close a connection (override in subclass)."""
        pass

    @property
    def stats(self) -> dict:
        return {
            "name": self.name,
            "max_size": self.max_size,
            "created": self._created,
            "in_use": self._in_use,
            "available": self._pool.qsize(),
        }

    async def cleanup_stale(self):
        """Clean up stale connections."""
        now = time.time()
        if now - self._last_update < 60:
            return

        async with self._lock:
            self._last_update = now

            # Remove excess connections
            while self._created > self.max_size // 2:
                try:
                    conn = self._pool.get_nowait()
                    await self._close_connection(conn)
                    self._created -= 1
                except asyncio.QueueEmpty:
                    break
