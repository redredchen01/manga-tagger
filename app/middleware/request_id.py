"""Request ID middleware for structured logging.

Adds unique request IDs to all requests for distributed tracing.
"""

import uuid
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from app.core.logging_config import clear_request_context, set_request_context


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Middleware that adds unique request ID to each request.

    The request ID is:
    - Added to response headers (X-Request-ID)
    - Bound to structlog context for structured logging
    - Used for distributed tracing across services
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Generate unique request ID
        request_id = str(uuid.uuid4())

        # Get correlation ID from incoming request if present
        correlation_id = request.headers.get("X-Correlation-ID", request_id)

        # Bind to logging context
        set_request_context(request_id=request_id, correlation_id=correlation_id)

        # Process request
        try:
            response = await call_next(request)
        finally:
            # Clean up context after request
            clear_request_context()

        # Add request ID to response headers
        response.headers["X-Request-ID"] = request_id
        if correlation_id != request_id:
            response.headers["X-Correlation-ID"] = correlation_id

        return response
