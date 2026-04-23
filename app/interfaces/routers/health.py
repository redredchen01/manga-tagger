"""Health check endpoints for the API."""

import logging
import time
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException

from app.config import settings
from app.models import HealthResponse
from app.services.tag_library_service import get_tag_library_service

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Health"])

# Module-level service instances (singleton pattern from original routes_v2.py)
_vlm_service = None
_llm_service = None
_rag_service = None
_tag_library = None
_tag_recommender = None


def get_vlm_service():
    """Get or create VLM service singleton."""
    global _vlm_service
    from app.core.config import settings

    if settings.USE_OLLAMA:
        if _vlm_service is None:
            from app.infrastructure.ollama.ollama_vlm_service import OllamaVLMService

            _vlm_service = OllamaVLMService()
        return _vlm_service
    elif settings.USE_LM_STUDIO:
        if _vlm_service is None:
            from app.services.lm_studio_vlm_service_v2 import LMStudioVLMService

            _vlm_service = LMStudioVLMService()
        return _vlm_service
    return None


def get_llm_service():
    """Get or create LLM service singleton."""
    global _llm_service
    if settings.USE_LM_STUDIO:
        if _llm_service is None:
            from app.services.lm_studio_llm_service import LMStudioLLMService

            _llm_service = LMStudioLLMService()
        return _llm_service
    return None


def get_rag_service():
    """Get or create RAG service singleton."""
    global _rag_service
    if _rag_service is None:
        from app.services.rag_service import RAGService

        _rag_service = RAGService()
    return _rag_service


def get_tag_library():
    """Get or create tag library singleton."""
    global _tag_library
    if _tag_library is None:
        _tag_library = get_tag_library_service()
    return _tag_library


def get_tag_recommender():
    """Get or create tag recommender singleton."""
    global _tag_recommender
    if _tag_recommender is None:
        from app.services.tag_recommender_service import get_tag_recommender_service

        _tag_recommender = get_tag_recommender_service()
    return _tag_recommender


def get_resilience_status() -> Dict[str, Any]:
    """Get circuit breaker and rate limiter status."""
    from app.core.circuit_breaker import get_all_circuit_breakers, CircuitState
    from app.core.rate_limiter import get_all_rate_limiters

    # Circuit breakers
    circuit_breakers = get_all_circuit_breakers()
    resilience = {"circuit_breakers": {}, "rate_limiters": {}}

    for name, stats in circuit_breakers.items():
        resilience["circuit_breakers"][name] = {
            "state": stats.state.value if hasattr(stats, "state") else "unknown",
            "total_calls": stats.total_calls,
            "successful_calls": stats.successful_calls,
            "failed_calls": stats.failed_calls,
            "rejected_calls": stats.rejected_calls,
        }

    # Rate limiters
    limiters = get_all_rate_limiters()
    for name, info in limiters.items():
        # info["config"] is a RateLimiterConfig dataclass
        config = info.get("config")
        max_concurrent = config.max_concurrent if config else 0
        resilience["rate_limiters"][name] = {
            "active_requests": info.get("active_requests", 0),
            "max_concurrent": max_concurrent,
        }

    return resilience


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="""
Check the health status of the Manga Cover Auto-Tagger service.

This endpoint provides:
- Service status (healthy/degraded)
- API version information
- Status of all loaded models (VLM, LLM, RAG, tag library)
- Configuration details
- Resilience status (circuit breakers, rate limiters)

**Note:** This is a lightweight endpoint that does not initialize heavy services.
It is used by startup scripts, Kubernetes probes, and monitoring systems.
    """,
    responses={
        200: {
            "description": "Service is healthy",
            "content": {
                "application/json": {
                    "example": {
                        "status": "healthy",
                        "version": "2.0.0",
                        "models_loaded": {
                            "vlm": "qwen3.6-35b-a3b-uncensored",
                            "vlm_initialized": True,
                            "llm": "qwen3.6-35b-a3b-uncensored",
                            "llm_initialized": True,
                            "rag": "ChromaDB (Local)",
                            "rag_initialized": True,
                            "lm_studio_mode": True,
                            "tag_library": 611,
                        },
                        "resilience": {
                            "circuit_breakers": {
                                "vlm": {"state": "closed", "total_calls": 100, "rejected_calls": 0}
                            },
                            "rate_limiters": {"vlm": {"active_requests": 0, "max_concurrent": 1}},
                        },
                    }
                }
            },
        }
    },
)
async def health_check():
    """Lightweight health check endpoint.

    Do not initialize heavy services here. Health checks are used by startup
    scripts, probes, and the frontend, so they must stay fast and side-effect free.
    """
    total_tags = 0
    try:
        if _tag_library is not None and hasattr(_tag_library, "tag_names"):
            total_tags = len(_tag_library.tag_names) if _tag_library.tag_names else 0
        else:
            tag_lib = get_tag_library()
            if tag_lib and hasattr(tag_lib, "tag_names"):
                total_tags = len(tag_lib.tag_names) if tag_lib.tag_names else 0
    except Exception as e:
        logger.warning(f"Could not get tag library count: {e}")

    models_dict: Dict[str, Any] = {
        "vlm": settings.OLLAMA_VISION_MODEL
        if settings.USE_OLLAMA
        else (settings.LM_STUDIO_VISION_MODEL if settings.USE_LM_STUDIO else "Disabled"),
        "vlm_initialized": _vlm_service is not None,
        "llm": settings.LM_STUDIO_TEXT_MODEL if settings.USE_LM_STUDIO else "Disabled",
        "llm_initialized": _llm_service is not None,
        "rag": "ChromaDB (Local)",
        "rag_initialized": _rag_service is not None,
        "lm_studio_mode": settings.USE_LM_STUDIO,
        "ollama_mode": settings.USE_OLLAMA,
        "tag_library": total_tags,
    }

    # Get resilience status
    try:
        resilience = get_resilience_status()
    except Exception as e:
        logger.warning(f"Could not get resilience status: {e}")
        resilience = {"error": str(e)}

    return HealthResponse(
        status="healthy",
        version="2.0.0",
        models_loaded=models_dict,
    )


@router.get(
    "/health/detailed",
    summary="Detailed health check",
    description="""
Detailed health check with full system status including:
- All service states
- Circuit breaker status
- Rate limiter status
- Memory usage
- Uptime
    """,
)
async def detailed_health_check():
    """Detailed health check with full system status."""
    import psutil
    import os

    # Get basic health first
    basic = await health_check()

    # Additional detailed info
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()

    # Get resilience
    resilience = get_resilience_status()

    return {
        "status": basic.status,
        "version": basic.version,
        "uptime_seconds": time.time() - os.path.getctime(__file__)
        if os.path.exists(__file__)
        else 0,
        "models": basic.models_loaded,
        "resilience": resilience,
        "system": {
            "memory_rss_mb": memory_info.rss / 1024 / 1024,
            "memory_vms_mb": memory_info.vms / 1024 / 1024,
            "cpu_percent": process.cpu_percent(),
            "open_files": len(process.open_files()),
        },
    }
