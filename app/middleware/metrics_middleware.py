"""Metrics middleware for Prometheus monitoring."""

from starlette.middleware.base import BaseHTTPMiddleware

from app.metrics import REQUEST_COUNT, REQUEST_LATENCY


class MetricsMiddleware(BaseHTTPMiddleware):
    """Middleware to track HTTP request metrics."""

    async def dispatch(self, request, call_next):
        import time

        start_time = time.time()

        response = await call_next(request)

        duration = time.time() - start_time

        # Extract endpoint pattern (normalize paths with IDs)
        endpoint = request.url.path
        # Simplify path for metrics (replace UUIDs, numeric IDs, etc.)
        if endpoint and endpoint[0] == "/":
            parts = endpoint.split("/")
            # Replace numeric segments with :id placeholder
            normalized_parts = []
            for part in parts:
                if part.isdigit():
                    normalized_parts.append(":id")
                else:
                    normalized_parts.append(part)
            endpoint = "/".join(normalized_parts)

        REQUEST_COUNT.labels(
            method=request.method, endpoint=endpoint, status=response.status_code
        ).inc()

        REQUEST_LATENCY.labels(method=request.method, endpoint=endpoint).observe(duration)

        return response
