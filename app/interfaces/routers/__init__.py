"""Routers package for API endpoints.

This module provides modular router implementations for different API concerns:
- health.py: Health check endpoints
- rag.py: RAG (Retrieval-Augmented Generation) endpoints
- tagging.py: Tagging and image analysis endpoints
- monitoring.py: Metrics and monitoring endpoints
- websocket.py: WebSocket for real-time updates
- query.py: Query endpoints (tagged-images)
"""

from app.interfaces.routers.health import router as health_router
from app.interfaces.routers.rag import router as rag_router
from app.interfaces.routers.tagging import router as tagging_router
from app.interfaces.routers.monitoring import router as monitoring_router
from app.interfaces.routers.websocket import router as websocket_router
from app.interfaces.routers.query import router as query_router

__all__ = [
    "health_router",
    "rag_router",
    "tagging_router",
    "monitoring_router",
    "websocket_router",
    "query_router",
]
