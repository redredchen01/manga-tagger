"""Rate limiting middleware using token bucket algorithm."""

import time
from threading import Lock
from typing import Dict, Tuple

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from app.config import settings
from app.logging_config import get_logger

logger = get_logger(__name__)


class TokenBucket:
    """Token bucket implementation for rate limiting."""

    def __init__(self, capacity: int, refill_rate: float):
        """
        Initialize token bucket.

        Args:
            capacity: Maximum number of tokens (burst capacity)
            refill_rate: Tokens added per second
        """
        self.capacity = capacity
        self.refill_rate = refill_rate
        self._tokens = float(capacity)
        self._last_refill = time.time()
        self._lock = Lock()

    def consume(self, tokens: int = 1) -> Tuple[bool, float]:
        """
        Try to consume tokens from the bucket.

        Args:
            tokens: Number of tokens to consume

        Returns:
            Tuple of (success: bool, retry_after: float)
        """
        with self._lock:
            self._refill()

            if self._tokens >= tokens:
                self._tokens -= tokens
                return True, 0.0

            # Calculate retry after time
            tokens_needed = tokens - self._tokens
            retry_after = tokens_needed / self.refill_rate
            return False, retry_after

    def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self._last_refill

        # Add tokens based on elapsed time
        new_tokens = elapsed * self.refill_rate
        self._tokens = min(self.capacity, self._tokens + new_tokens)
        self._last_refill = now


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware using token bucket algorithm."""

    def __init__(self, app):
        super().__init__(app)
        self.buckets: Dict[str, TokenBucket] = {}
        self.lock = Lock()

        # Get settings
        self.enabled = settings.RATE_LIMIT_ENABLED
        self.requests_per_second = (
            settings.RATE_LIMIT_REQUESTS / 60.0
        )  # Convert per minute to per second
        self.burst = settings.RATE_LIMIT_BURST

        if self.enabled:
            logger.info(
                f"Rate limiting enabled: {settings.RATE_LIMIT_REQUESTS} requests/min, "
                f"burst={self.burst}"
            )

    def _get_client_ip(self, request: Request) -> str:
        """
        Get client IP address, considering X-Forwarded-For header.

        Args:
            request: FastAPI request

        Returns:
            Client IP address
        """
        # Check X-Forwarded-For header (for reverse proxy)
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            # Take the first IP in the chain
            return forwarded_for.split(",")[0].strip()

        # Fallback to direct client IP
        if request.client:
            return request.client.host

        return "unknown"

    def _get_bucket(self, client_ip: str) -> TokenBucket:
        """
        Get or create token bucket for client IP.

        Args:
            client_ip: Client IP address

        Returns:
            TokenBucket instance
        """
        with self.lock:
            if client_ip not in self.buckets:
                self.buckets[client_ip] = TokenBucket(
                    capacity=self.burst, refill_rate=self.requests_per_second
                )
            return self.buckets[client_ip]

    async def dispatch(self, request: Request, call_next):
        """
        Process request with rate limiting.

        Args:
            request: FastAPI request
            call_next: Next middleware/handler

        Returns:
            Response or 429 error
        """
        # Skip rate limiting if disabled
        if not self.enabled:
            return await call_next(request)

        client_ip = self._get_client_ip(request)
        bucket = self._get_bucket(client_ip)

        success, retry_after = bucket.consume()

        if not success:
            logger.warning(f"Rate limit exceeded for IP: {client_ip}")
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Rate limit exceeded",
                    "message": f"Too many requests. Please try again later.",
                    "retry_after": round(retry_after, 2),
                },
                headers={"Retry-After": str(int(retry_after) + 1)},
            )

        response = await call_next(request)
        return response
