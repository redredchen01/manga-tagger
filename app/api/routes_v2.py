"""API routes for Manga Tagger - Optimized Version.

Integrates tag library matching with VLM analysis and RAG for intelligent tag recommendations.
"""

import time
import json
from typing import List, Optional, Dict, Any
import logging
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from fastapi.responses import JSONResponse

from app.models import (
    TagCoverResponse,
    TagResult,
    RAGAddResponse,
    TagsListResponse,
    TagInfo,
    HealthResponse,
)
from app.services.lm_studio_vlm_service_v2 import LMStudioVLMService
from app.services.lm_studio_llm_service import LMStudioLLMService
from app.services.rag_service import RAGService
from app.services.tag_library_service import get_tag_library_service
from app.services.tag_recommender_service import (
    get_tag_recommender_service,
    TagRecommendation,
)
from app.config import settings
from app.utils import safe_confidence

logger = logging.getLogger(__name__)

router = APIRouter()

# Service instances (singletons)
_vlm_service: Optional[LMStudioVLMService] = None
_llm_service: Optional[LMStudioLLMService] = None
_rag_service: Optional[RAGService] = None
_tag_library = None
_tag_recommender = None


def get_vlm_service():
    """Get or create VLM service singleton."""
    global _vlm_service
    if settings.USE_LM_STUDIO:
        if _vlm_service is None:
            _vlm_service = LMStudioVLMService()
        return _vlm_service
    return None


def get_llm_service():
    """Get or create LLM service singleton."""
    global _llm_service
    if settings.USE_LM_STUDIO:
        if _llm_service is None:
            _llm_service = LMStudioLLMService()
        return _llm_service
    return None


def get_rag_service():
    """Get or create RAG service singleton."""
    global _rag_service
    if _rag_service is None:
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
        _tag_recommender = get_tag_recommender_service()
    return _tag_recommender


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    # Ensure services are initialized to report correct status
    vlm = get_vlm_service()
    llm = get_llm_service()
    rag = get_rag_service()
    
    # Get tag library count - safe access with fallback
    total_tags = 0
    try:
        tag_lib = get_tag_library()
        if tag_lib and hasattr(tag_lib, 'tag_names'):
            total_tags = len(tag_lib.tag_names) if tag_lib.tag_names else 0
    except Exception as e:
        logger.warning(f"Could not get tag library count: {e}")

    models_dict: Dict[str, Any] = {
        "vlm": settings.LM_STUDIO_VISION_MODEL if vlm else "Disabled",
        "llm": settings.LM_STUDIO_TEXT_MODEL if llm else "Disabled",
        "rag": "ChromaDB (Local)" if rag else "Not Initialized",
        "lm_studio_mode": settings.USE_LM_STUDIO,
        "tag_library": total_tags,
    }

    return HealthResponse(
        status="healthy",
        version="2.0.0",
        models_loaded=models_dict,
    )


@router.post("/rag/add", response_model=RAGAddResponse)
async def add_to_rag(
    file: UploadFile = File(..., description="Reference image"),
    tags: str = Form(..., description="JSON array of tags"),
    metadata: str = Form("{}", description="JSON metadata object"),
):
    """
    Add an image to the RAG dataset.

    - **file**: Reference image
    - **tags**: JSON array of tag strings (e.g., '["貓娘", "蘿莉"]')
    - **metadata**: Optional JSON metadata
    """
    try:
        # Parse tags
        try:
            tag_list = json.loads(tags)
            if not isinstance(tag_list, list):
                raise ValueError("Tags must be a JSON array")
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON in tags field")

        # Parse metadata
        try:
            metadata_dict = json.loads(metadata)
            if not isinstance(metadata_dict, dict):
                raise ValueError("Metadata must be a JSON object")
        except json.JSONDecodeError:
            raise HTTPException(
                status_code=400, detail="Invalid JSON in metadata field"
            )

        # Validate file type
        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")

        # Read image
        image_bytes = await file.read()

        # Get RAG service
        rag_service = get_rag_service()

        # Add to RAG
        doc_id = await rag_service.add_image(image_bytes, tag_list, metadata_dict)

        return RAGAddResponse(
            success=True,
            id=doc_id,
            message=f"Successfully added image with {len(tag_list)} tags",
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to add image: {str(e)}")


@router.post("/tag-cover", response_model=TagCoverResponse)
async def tag_cover(
    file: UploadFile = File(..., description="Manga cover image to tag"),
    top_k: int = Form(5, ge=1, le=20, description="Number of tags to return"),
    confidence_threshold: float = Form(
        0.5, ge=0.0, le=1.0, description="Minimum confidence threshold"
    ),
    include_metadata: str = Form(
        "true", description="Include processing metadata in response"
    ),
):
    """
    Tag a manga cover image using AI analysis.

    This endpoint:
    1. Analyzes the image using VLM (Vision Language Model)
    2. Searches for similar images in RAG database
    3. Matches extracted features against the 611-tag library
    4. Returns the best matching tags with confidence scores
    """
    start_time = time.time()

    try:
        # Validate file type
        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")

        # Read image
        image_bytes = await file.read()

        # Get services
        vlm_service = get_vlm_service()
        rag_service = get_rag_service()
        tag_recommender = get_tag_recommender()

        # Stage 1: VLM analysis
        logger.info("Stage 1: VLM analysis")
        vlm_analysis = None
        vlm_description = ""
        if vlm_service:
            vlm_analysis = await vlm_service.extract_metadata(image_bytes)
            vlm_description = vlm_analysis.get("description", "")
        else:
            vlm_description = "VLM service not available"

        # Stage 2: RAG similarity search
        logger.info("Stage 2: RAG similarity search")
        rag_matches = []
        if rag_service:
            rag_matches = await rag_service.search_similar(image_bytes, top_k=10)

        # Stage 3: Tag recommendation using tag library
        logger.info("Stage 3: Tag recommendation with library matching")
        recommendations: List[TagRecommendation] = []

        if vlm_analysis:
            recommendations = await tag_recommender.recommend_tags(
                vlm_analysis=vlm_analysis,
                rag_matches=rag_matches,
                top_k=top_k,
                confidence_threshold=confidence_threshold,
                vlm_service=vlm_service,
                image_bytes=image_bytes,
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

        # Calculate processing time
        processing_time = time.time() - start_time

        # Prepare response metadata
        response_metadata = None
        if include_metadata.lower() == "true":
            response_metadata = {
                "processing_time": round(processing_time, 2),
                "vlm_description": vlm_description,
                "vlm_analysis": {
                    "character_types": vlm_analysis.get("character_types", [])
                    if vlm_analysis
                    else [],
                    "clothing": vlm_analysis.get("clothing", [])
                    if vlm_analysis
                    else [],
                    "body_features": vlm_analysis.get("body_features", [])
                    if vlm_analysis
                    else [],
                    "actions": vlm_analysis.get("actions", []) if vlm_analysis else [],
                    "themes": vlm_analysis.get("themes", []) if vlm_analysis else [],
                }
                if vlm_analysis
                else {},
                "rag_matches": rag_matches if isinstance(rag_matches, list) else [],
                "library_tags_available": (
                    len(tag_recommender.tag_library.tag_names) 
                    if tag_recommender and hasattr(tag_recommender, 'tag_library') and tag_recommender.tag_library and tag_recommender.tag_library.tag_names
                    else 0
                ),
            }

        logger.info(
            f"Tagging completed in {processing_time:.2f}s, returned {len(tags)} tags"
        )
        return TagCoverResponse(tags=tags, metadata=response_metadata)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to tag image: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to tag image: {str(e)}")


@router.get("/tags", response_model=TagsListResponse)
async def list_tags(
    category: Optional[str] = None, search: Optional[str] = None, limit: int = 100
):
    """
    List available tags from the 611-tag library.

    - **category**: Filter by category (character, clothing, body, action, theme, other)
    - **search**: Search for specific tags
    - **limit**: Maximum number of tags to return
    """
    try:
        tag_lib = get_tag_library()

        if not tag_lib.tags:
            return TagsListResponse(tags=[], total=0)

        # Get tags based on filters
        if search:
            tag_names = tag_lib.search_tags(search, limit=limit)
            tags_data = [
                TagInfo(tag_name=name, description=tag_lib.get_tag_description(name))
                for name in tag_names
            ]
        elif category:
            tag_names = tag_lib.get_tags_by_category(category)[:limit]
            tags_data = [
                TagInfo(tag_name=name, description=tag_lib.get_tag_description(name))
                for name in tag_names
            ]
        else:
            tags_data = [
                TagInfo(tag_name=tag["tag_name"], description=tag.get("description"))
                for tag in tag_lib.tags[:limit]
            ]

        return TagsListResponse(tags=tags_data, total=len(tag_lib.tag_names))

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load tags: {str(e)}")


@router.get("/rag/stats")
async def get_rag_stats():
    """Get RAG dataset statistics."""
    try:
        rag_service = get_rag_service()
        stats = rag_service.get_stats()

        # Add tag library info
        tag_lib = get_tag_library()
        stats["tag_library_total"] = len(tag_lib.tag_names)

        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")


@router.get("/tags/categories")
async def get_tag_categories():
    """Get tag categories and their counts."""
    try:
        tag_lib = get_tag_library()

        categories = {}
        for cat_name, tags in tag_lib.tag_categories.items():
            categories[cat_name] = len(tags)

        return {"categories": categories, "total_tags": len(tag_lib.tag_names)}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get categories: {str(e)}"
        )
