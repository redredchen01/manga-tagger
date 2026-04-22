"""FastAPI dependency injection providers for Manga Tagger services.

This module provides dependency injection providers that replace the global
singleton pattern in routes_v2.py. Routes can use these providers with FastAPI's
Depends() to access services via request.app.state.

Example:
    @router.post("/tag-cover")
    async def tag_cover(
        file: UploadFile,
        vlm_service: LMStudioVLMService = Depends(get_vlm_service),
    ):
        ...

The original singleton functions in routes_v2.py still work for backward compatibility.
"""

from fastapi import Request

from app.domain.tag.library import TagLibraryService, get_tag_library_service
from app.domain.tag.recommender import TagRecommenderService, get_tag_recommender_service
from app.infrastructure.lm_studio.vlm_service import LMStudioVLMService
from app.infrastructure.lm_studio.llm_service import LMStudioLLMService
from app.infrastructure.rag.rag_service import RAGService


def get_vlm_service(request: Request):
    """Get VLM service — prefers app.state, falls back to lazy singleton."""
    svc = getattr(request.app.state, "vlm_service", None)
    if svc is not None:
        return svc
    # Fallback: lazy singleton (used when lifespan hasn't run, e.g. in tests)
    from app.core.config import settings

    if settings.USE_OLLAMA:
        from app.infrastructure.ollama.ollama_vlm_service import OllamaVLMService

        return OllamaVLMService()
    elif settings.USE_LM_STUDIO:
        return LMStudioVLMService()
    return None


def get_llm_service(request: Request) -> LMStudioLLMService | None:
    """Get LLM service — prefers app.state, falls back to lazy singleton."""
    svc = getattr(request.app.state, "llm_service", None)
    if svc is not None:
        return svc
    from app.core.config import settings

    if settings.USE_LM_STUDIO:
        return LMStudioLLMService()
    return None


def get_rag_service(request: Request) -> RAGService:
    """Get RAG service — prefers app.state, falls back to lazy singleton."""
    svc = getattr(request.app.state, "rag_service", None)
    if svc is not None:
        return svc
    return RAGService()


def get_tag_library_service(request: Request) -> TagLibraryService:
    """Get tag library service — prefers app.state, falls back to lazy singleton."""
    svc = getattr(request.app.state, "tag_library", None)
    if svc is not None:
        return svc
    from app.domain.tag.library import get_tag_library_service as _get

    return _get()


def get_tag_recommender(request: Request) -> TagRecommenderService:
    """Get tag recommender — prefers app.state, falls back to lazy singleton."""
    svc = getattr(request.app.state, "tag_recommender", None)
    if svc is not None:
        return svc
    from app.domain.tag.recommender import get_tag_recommender_service as _get

    return _get()
