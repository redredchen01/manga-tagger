"""API routes for Manga Tagger."""

import time
import json
from typing import List, Optional
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
    ProcessingMetadata,
    VLMMetadata,
    RAGMatch,
)
from app.services.lm_studio_vlm_service import LMStudioVLMService
from app.services.lm_studio_llm_service import LMStudioLLMService
from app.services.rag_service import RAGService
from app.services.tag_library_service import get_tag_library_service
from app.services.tag_mapper import get_tag_mapper
from app.config import settings
from app.utils import safe_confidence

# Import tag_matcher for enhanced tag matching
from tag_matcher import TagMatcher

logger = logging.getLogger(__name__)

router = APIRouter()

# Service instances (singletons)
_lm_studio_vlm_service: Optional[LMStudioVLMService] = None
_lm_studio_llm_service: Optional[LMStudioLLMService] = None
_rag_service: Optional[RAGService] = None


def get_vlm_service():
    """Get or create VLM service singleton."""
    global _lm_studio_vlm_service
    if settings.USE_LM_STUDIO:
        if _lm_studio_vlm_service is None:
            _lm_studio_vlm_service = LMStudioVLMService()
        return _lm_studio_vlm_service
    else:
        return None


def get_llm_service():
    """Get or create LLM service singleton."""
    global _lm_studio_llm_service
    if settings.USE_LM_STUDIO:
        if _lm_studio_llm_service is None:
            _lm_studio_llm_service = LMStudioLLMService()
        return _lm_studio_llm_service
    else:
        return None


def get_rag_service():
    """Get or create RAG service singleton."""
    global _rag_service
    if _rag_service is None:
        from app.services.rag_service import RAGService

        _rag_service = RAGService()
    return _rag_service


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        models_loaded={
            "vlm": _lm_studio_vlm_service is not None,
            "rag": _rag_service is not None,
            "llm": _lm_studio_llm_service is not None,
            "lm_studio_mode": settings.USE_LM_STUDIO,
        },
    )


@router.post("/rag/add", response_model=RAGAddResponse)
async def add_to_rag(
    file: UploadFile = File(..., description="Reference image"),
    tags: str = Form(..., description="JSON array of tags"),
    metadata: str = Form("{}", description="JSON metadata object"),
    rag_service: RAGService = Depends(get_rag_service),
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

    try:
        # Get services
        vlm_service = get_vlm_service()
        rag_service = get_rag_service()

        # Validate file type
        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")

        # Read image
        image_bytes = await file.read()

        # Stage 1: VLM extraction
        vlm_metadata = None
        vlm_description = ""
        if vlm_service:
            vlm_metadata = await vlm_service.extract_metadata(image_bytes)
            vlm_description = vlm_metadata.description
        else:
            vlm_description = "VLM service not available"

        # Stage 2: RAG similarity search
        rag_matches = []
        if rag_service:
            rag_matches = await rag_service.search_similar(image_bytes, top_k)

        # Stage 3: Synthesize tags using LLM if available, otherwise use VLM + RAG
        tags = []
        processing_time = 0.0

        # Get LLM service
        llm_service = get_llm_service()

        if llm_service and vlm_metadata:
            # Use LLM for intelligent tag synthesis
            try:
                llm_tags = await llm_service.synthesize_tags(
                    vlm_metadata=vlm_metadata,
                    rag_matches=rag_matches,
                    top_k=top_k,
                    confidence_threshold=confidence_threshold,
                )
                tags.extend(llm_tags)
            except Exception as e:
                logger.warning(f"LLM synthesis failed: {e}, falling back to VLM+RAG")

        # Fallback: Add tags from VLM results - SKIPPED tag_matcher due to embedding model loading issue
        # TODO: Fix sentence-transformers import issue
        if not tags and vlm_metadata:
            # Skip enhanced matching for now - will use basic VLM fallback below
            logger.info("Skipping enhanced tag matching (embedding model not available)")

            # Basic fallback: Add tags from VLM characters
            if not tags:
                for character in vlm_metadata.characters[:2]:
                    tags.append(
                        TagResult(
                            tag=character,
                            confidence=safe_confidence(0.8),
                            source="vlm",
                            reason=f"Detected character type: {character}",
                        )
                    )

        # Add tags from RAG results if still need more
        for match in rag_matches[:3]:
            for tag in match["tags"][:2]:
                if len(tags) < top_k:
                    # Check if tag already exists
                    if not any(t.tag == tag for t in tags):
                        tags.append(
                            TagResult(
                                tag=tag,
                                confidence=safe_confidence(match["score"]),
                                source="vlm+rag",
                                reason=f"Similar to reference image (score: {match['score']:.2f})",
                            )
                        )

        # Prepare response metadata
        response_metadata = None
        if include_metadata:
            response_metadata = {
                "processing_time": processing_time,
                "vlm_description": vlm_description,
                "rag_matches": rag_matches,
            }

        return TagCoverResponse(tags=tags[:top_k], metadata=response_metadata)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to tag image: {str(e)}")


@router.get("/tags", response_model=TagsListResponse)
async def list_tags():
    """List available tags from tag library."""
    try:
        from pathlib import Path
        import json

        tags_path = Path(settings.TAG_LIBRARY_PATH)
        if not tags_path.exists():
            return TagsListResponse(tags=[], total=0)

        with open(tags_path, "r", encoding="utf-8") as f:
            tag_data = json.load(f)

        tags = [
            TagInfo(
                tag_name=item["tag_name"],
                description=item.get("description"),
                category=item.get("category"),
            )
            for item in tag_data
        ]

        return TagsListResponse(tags=tags, total=len(tags))

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load tags: {str(e)}")


@router.get("/rag/stats")
async def get_rag_stats(rag_service: RAGService = Depends(get_rag_service)):
    """Get RAG dataset statistics."""
    try:
        stats = rag_service.get_stats()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")
