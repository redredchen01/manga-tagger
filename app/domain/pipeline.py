"""Unified tagging pipeline orchestrator.

Provides the core 3-stage tagging pipeline:
1. VLM analysis (Vision Language Model)
2. RAG search (Similar image matching)
3. Tag recommendation (Library matching + synthesis)

This module consolidates functionality from both run_tagging_pipeline and
run_tagging_pipeline_with_progress into a single function with optional
progress callback support.
"""

import asyncio
import logging
from typing import Any, Awaitable, Callable, Dict, List, Optional

from app.core.config import settings
from app.core.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerOpen,
    get_circuit_breaker,
    CircuitState,
)
from app.core.rate_limiter import get_rate_limiter
from app.core.logging_config import get_logger
from app.core.performance.memory import track_memory
from app.domain.models import TagCoverResponse, TagResult
from app.domain.tag.recommender import TagRecommenderService, get_tag_recommender_service
from app.infrastructure.lm_studio.vlm_service import LMStudioVLMService
from app.infrastructure.rag.rag_service import RAGService
from app.utils import validate_image
from app.performance.async_utils import gather_with_limit, timeout_after

logger = get_logger(__name__)


# Type for progress callback - matches task specification: (job_id, progress, message)
ProgressCallback = Optional[Callable[[str, int, str], Awaitable[None]]]


async def _emit_progress(
    callback: Optional[Callable[[str, int, str], Awaitable[None]]],
    job_id: str,
    progress: int,
    message: str,
) -> None:
    """Emit progress if callback is provided.

    Args:
        callback: Optional progress callback function
        job_id: Job identifier for tracking
        progress: Progress percentage (0-100)
        message: Status message
    """
    if callback:
        await callback(job_id, progress, message)


@track_memory
async def run_tagging_pipeline(
    image_bytes: bytes,
    image_url: Optional[str] = None,
    file_hash: Optional[str] = None,
    top_k: int = 5,
    confidence_threshold: float = 0.5,
    filename: Optional[str] = None,
    batch_size: int = 20,
    progress_callback: Optional[ProgressCallback] = None,
    vlm_service=None,
    rag_service: Optional[RAGService] = None,
    tag_recommender: Optional[TagRecommenderService] = None,
    job_id: Optional[str] = None,
    **kwargs,
) -> TagCoverResponse:
    """Run the 3-stage tagging pipeline.

    Stage 1 & 2: Parallel VLM + RAG execution
    Stage 3: Tag recommendation with library matching

    Args:
        image_bytes: The image data to tag
        image_url: Optional URL of the image (for metadata)
        file_hash: Optional hash of the image file (for caching)
        top_k: Number of tags to return
        confidence_threshold: Minimum confidence for tags
        filename: Optional filename for the image
        batch_size: Batch size for processing (default 20)
        progress_callback: Optional async callback for progress updates.
                           If None, works like run_tagging_pipeline.
                           If provided, works like run_tagging_pipeline_with_progress.
                           Signature: async def callback(job_id: str, progress: int, message: str)
        vlm_service: Optional VLM service instance
        rag_service: Optional RAG service instance
        tag_recommender: Optional tag recommender instance

    Returns:
        TagCoverResponse with tags and metadata
    """
    # Determine job_id for progress tracking — explicit job_id wins
    job_id = job_id or file_hash or filename or "default"

    # Emit initial progress if callback provided
    await _emit_progress(progress_callback, job_id, 0, "Starting pipeline...")

    # Get services if not provided
    from app.core.config import settings

    if vlm_service is None:
        if settings.USE_OLLAMA:
            from app.infrastructure.ollama.ollama_vlm_service import OllamaVLMService

            vlm_service = OllamaVLMService()
        elif settings.USE_LM_STUDIO:
            vlm_service = LMStudioVLMService()

    if rag_service is None:
        rag_service = RAGService()

    if tag_recommender is None:
        tag_recommender = get_tag_recommender_service()

    # Validate image
    is_valid, error_msg = validate_image(image_bytes)
    if not is_valid:
        raise ValueError(error_msg)

    # Emit validation complete progress
    await _emit_progress(progress_callback, job_id, 10, "Image validated")

    # Stage 1 & 2: Parallel VLM + RAG with progress updates
    await _emit_progress(progress_callback, job_id, 20, "Starting VLM + RAG analysis...")
    logger.info("Stage 1 & 2: Parallel VLM + RAG execution")

    # Get circuit breakers for resilience
    vlm_cb = get_circuit_breaker("vlm")
    rag_cb = get_circuit_breaker("rag")

    # Get rate limiters to prevent resource exhaustion
    vlm_limiter = get_rate_limiter("vlm")
    rag_limiter = get_rate_limiter("rag")

    # Create tasks with circuit breaker and rate limiter protection
    vlm_task = None
    rag_task = None

    if vlm_service:
        # Wrap VLM call with circuit breaker and rate limiter
        async def protected_vlm():
            async with vlm_limiter:
                # Use circuit breaker call() to properly track failures
                return await vlm_cb.call(
                    timeout_after,
                    settings.VLM_TIMEOUT_SECONDS,
                    vlm_service.extract_metadata(image_bytes),
                    fallback=lambda: {"description": "VLM circuit breaker open - using fallback"},
                )

        vlm_task = protected_vlm()

    if rag_service:
        # Wrap RAG call with circuit breaker and rate limiter
        async def protected_rag():
            async with rag_limiter:
                # Use circuit breaker call() to properly track failures
                return await rag_cb.call(
                    timeout_after,
                    settings.RAG_TIMEOUT_SECONDS,
                    rag_service.search_similar(image_bytes, top_k=10),
                    fallback=lambda: [],
                )

        rag_task = protected_rag()

    # Run both in parallel with concurrency limit
    vlm_analysis, rag_matches = await gather_with_limit(
        settings.MAX_CONCURRENT_VLM_CALLS,
        vlm_task or asyncio.sleep(0, result=None),
        rag_task or asyncio.sleep(0, result=[]),
    )

    # Handle VLM results
    vlm_description = ""
    if vlm_analysis is None:
        vlm_description = "VLM service not available"
        vlm_analysis = {"description": vlm_description}
    else:
        vlm_description = vlm_analysis.get("description", "")

    # Emit VLM complete progress
    await _emit_progress(
        progress_callback, job_id, 50, f"VLM analysis complete: {vlm_description[:50]}..."
    )

    if rag_matches is None:
        rag_matches = []

    # Emit RAG complete progress
    rag_count = len(rag_matches) if isinstance(rag_matches, list) else 0
    await _emit_progress(
        progress_callback, job_id, 60, f"RAG search complete: {rag_count} matches found"
    )

    # Stage 3: Tag recommendation using tag library
    await _emit_progress(progress_callback, job_id, 70, "Starting tag recommendation...")
    logger.info("Stage 3: Tag recommendation with library matching")

    from app.services.tag_recommender_service import TagRecommendation

    recommendations: List[TagRecommendation] = []

    # Phase 1: RAG influence on scoring is gated by config.
    # When disabled, the recommender sees an empty list (no RAG -> tags),
    # but the API response still reports what RAG found in metadata for
    # visibility/debugging.
    rag_matches_for_scoring = rag_matches if settings.RAG_INFLUENCE_ENABLED else []

    if vlm_analysis:
        recommendations = await tag_recommender.recommend_tags(
            vlm_analysis=vlm_analysis,
            rag_matches=rag_matches_for_scoring,
            top_k=top_k,
            confidence_threshold=confidence_threshold,
            vlm_service=vlm_service,
            image_bytes=image_bytes,
        )

    # Emit recommendation complete progress
    await _emit_progress(
        progress_callback, job_id, 90, f"Generated {len(recommendations)} tag recommendations"
    )

    # Convert recommendations to TagResult format
    tags = [
        TagResult(
            tag=rec.tag,
            confidence=rec.confidence,
            source=rec.source,
            reason=rec.reason,
        )
        for rec in recommendations
    ]

    # Prepare metadata
    metadata = {
        # VLM info
        "vlm_description": vlm_description,
        "vlm_analysis": {
            "character_types": vlm_analysis.get("character_types", []) if vlm_analysis else [],
            "clothing": vlm_analysis.get("clothing", []) if vlm_analysis else [],
            "body_features": vlm_analysis.get("body_features", []) if vlm_analysis else [],
            "actions": vlm_analysis.get("actions", []) if vlm_analysis else [],
            "themes": vlm_analysis.get("themes", []) if vlm_analysis else [],
        }
        if vlm_analysis
        else {},
        # RAG info
        "rag_matches_count": len(rag_matches) if isinstance(rag_matches, list) else 0,
        "rag_matches": rag_matches if isinstance(rag_matches, list) else [],
        # Library info
        "library_tags_available": (
            len(tag_recommender.tag_library.tag_names)
            if tag_recommender
            and hasattr(tag_recommender, "tag_library")
            and tag_recommender.tag_library
            and tag_recommender.tag_library.tag_names
            else 0
        ),
        # API info
        "api_version": "2.0.0",
        # Job info (only if progress callback was used)
        **({"job_id": job_id} if progress_callback else {}),
    }

    # Emit completion progress
    await _emit_progress(progress_callback, job_id, 100, "Pipeline complete")

    return TagCoverResponse(tags=tags, metadata=metadata)
