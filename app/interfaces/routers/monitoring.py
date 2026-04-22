"""Monitoring and metrics endpoints for the API."""

import logging

from fastapi import APIRouter
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Monitoring"])


@router.get(
    "/metrics",
    summary="Prometheus metrics",
    description="""
Prometheus metrics endpoint for monitoring.

Returns metrics in Prometheus exposition format including:
- HTTP request counts and latencies
- VLM request counts and latencies
- RAG request counts and latencies
- Tag recommendation counts and latencies
- Cache hits and misses

Access this endpoint at: `/api/v1/metrics`
    """,
    responses={
        200: {
            "description": "Prometheus metrics",
            "content": {
                "text/plain": {
                    "example": '# HELP http_requests_total Total HTTP requests\n# TYPE http_requests_total counter\nhttp_requests_total{method="GET",endpoint="/api/v1/health",status="200"} 123.0',
                }
            },
        }
    },
)
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@router.get(
    "/performance",
    include_in_schema=False,
    summary="Performance metrics",
    description="""
Internal performance metrics endpoint.

Returns current process metrics including:
- Memory usage (RSS in MB)
- CPU percentage
- Thread count
- GC statistics

**Note:** This is an internal endpoint, not exposed in OpenAPI schema.
    """,
)
async def performance_metrics():
    """Performance metrics endpoint (internal use)."""
    import gc
    import os

    import psutil

    process = psutil.Process(os.getpid())

    return {
        "memory_mb": process.memory_info().rss / 1024 / 1024,
        "cpu_percent": process.cpu_percent(),
        "threads": process.num_threads(),
        "gc_stats": gc.get_stats(),
    }
