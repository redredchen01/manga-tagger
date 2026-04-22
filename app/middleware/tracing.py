import uuid
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
import logging
import structlog

logger = structlog.get_logger(__name__)


class TracingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Generate request ID
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())

        # Generate correlation ID (for tracing across services)
        correlation_id = request.headers.get("X-Correlation-ID") or request_id

        # Store in request state
        request.state.request_id = request_id
        request.state.correlation_id = correlation_id

        # Bind context variables for logging
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(
            request_id=request_id,
            correlation_id=correlation_id,
        )

        # Log request start
        logger.info(
            "Request started",
            method=request.method,
            path=request.url.path,
        )

        # Process request
        response = await call_next(request)

        # Add headers to response
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Correlation-ID"] = correlation_id

        # Log request completion
        logger.info(
            "Request completed",
            status=response.status_code,
        )

        # Clear context after request
        structlog.contextvars.clear_contextvars()

        return response
