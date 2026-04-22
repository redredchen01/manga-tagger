"""API routes for Manga Tagger - Optimized Version.

This module has been refactored into modular routers in app.interfaces.routers.
The endpoints are now organized by concern:
- health.py: Health check endpoints
- rag.py: RAG (Retrieval-Augmented Generation) endpoints
- tagging.py: Tagging and image analysis endpoints
- monitoring.py: Metrics and monitoring endpoints
- websocket.py: WebSocket for real-time updates
- query.py: Query endpoints (placeholder)

This file now re-exports all routers for backward compatibility.
Service singletons are also exposed here so existing tests can still patch them.
"""

from fastapi import APIRouter

from app.domain.tag.library import TagLibraryService, get_tag_library_service
from app.domain.tag.recommender import TagRecommenderService, get_tag_recommender_service
from app.infrastructure.lm_studio.llm_service import LMStudioLLMService
from app.infrastructure.lm_studio.vlm_service import LMStudioVLMService
from app.infrastructure.rag.rag_service import RAGService
from app.interfaces.routers import (  # noqa: F401
    health_router,
    monitoring_router,
    query_router,
    rag_router,
    tagging_router,
    websocket_router,
)

# ---------------------------------------------------------------------------
# Service singletons (module-level, used by tests for patching)
# Tests patch routes_v2.get_vlm_service, etc. — keep these accessible here.
# ---------------------------------------------------------------------------

_vlm_service: LMStudioVLMService | None = None
_llm_service: LMStudioLLMService | None = None
_rag_service: RAGService | None = None
_tag_library: TagLibraryService | None = None
_tag_recommender: TagRecommenderService | None = None


def get_vlm_service():
    """Legacy singleton — respect USE_OLLAMA, mirroring app.dependencies.get_vlm_service."""
    global _vlm_service
    if _vlm_service is None:
        from app.core.config import settings

        if settings.USE_OLLAMA:
            from app.infrastructure.ollama.ollama_vlm_service import OllamaVLMService

            _vlm_service = OllamaVLMService()
        else:
            _vlm_service = LMStudioVLMService()
    return _vlm_service


def get_llm_service() -> LMStudioLLMService:
    global _llm_service
    if _llm_service is None:
        _llm_service = LMStudioLLMService()
    return _llm_service


def get_rag_service() -> RAGService:
    global _rag_service
    if _rag_service is None:
        _rag_service = RAGService()
    return _rag_service


def get_tag_library() -> TagLibraryService:
    global _tag_library
    if _tag_library is None:
        _tag_library = get_tag_library_service()
    return _tag_library


def get_tag_recommender() -> TagRecommenderService:
    global _tag_recommender
    if _tag_recommender is None:
        _tag_recommender = get_tag_recommender_service()
    return _tag_recommender


# ---------------------------------------------------------------------------
# Main router that aggregates all sub-routers for backward compatibility
# ---------------------------------------------------------------------------

router = APIRouter()

# Include all routers
router.include_router(health_router)
router.include_router(rag_router)
router.include_router(tagging_router)
router.include_router(monitoring_router)
router.include_router(websocket_router)
router.include_router(query_router)

__all__ = [
    "router",
    "get_vlm_service",
    "get_llm_service",
    "get_rag_service",
    "get_tag_library",
    "get_tag_recommender",
]
