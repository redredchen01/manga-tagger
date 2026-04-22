"""Tagging and image analysis endpoints for the API."""

import asyncio
import logging
import time
import uuid
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile

from app.auth import verify_api_key
from app.core.config import settings
from app.core.exceptions import ServiceUnavailableError, ValidationError
from app.dependencies import (
    get_rag_service,
    get_tag_library_service,
    get_tag_recommender,
    get_vlm_service,
)
from app.domain.models import MangaDescriptionResponse, TagCoverResponse, TagInfo, TagsListResponse
from app.domain.pipeline import run_tagging_pipeline
from app.interfaces.websocket.connection_manager import manager
from app.utils import validate_image

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Tagging"])


async def validate_and_read_image(file: UploadFile) -> bytes:
    """Validate image file and return bytes."""
    # Validate file type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise ValidationError(
            detail=f"Invalid file type. Must be an image. Got: {file.content_type}",
        )

    # Read image
    image_bytes = await file.read()

    # Validate file size (max 10MB)
    max_file_size = 10 * 1024 * 1024
    if len(image_bytes) > max_file_size:
        raise ValidationError(
            detail=f"File too large. Max size is 10MB. Got: {len(image_bytes) / 1024 / 1024:.1f}MB",
        )

    # Validate minimum file size
    if len(image_bytes) < 1024:
        raise ValidationError(detail="File too small. Must be at least 1KB")

    return image_bytes


def _stage_from_progress(progress: int) -> str:
    """Map progress percentage to coarse pipeline stage label."""
    if progress < 20:
        return "init"
    if progress < 50:
        return "vlm"
    if progress < 70:
        return "rag"
    if progress < 100:
        return "recommend"
    return "complete"


async def send_progress_update(job_id: str, progress: int, message: str):
    """Send progress update via WebSocket — matches ProgressCallback signature."""
    await manager.send_update(
        job_id,
        {
            "job_id": job_id,
            "stage": _stage_from_progress(progress),
            "progress": progress,
            "message": message,
            "timestamp": datetime.utcnow().isoformat(),
        },
    )


@router.post(
    "/tag-cover",
    response_model=TagCoverResponse,
    dependencies=[Depends(verify_api_key)],
    summary="Tag a manga cover image",
    description="""
Analyze a manga cover image and return matching tags.

**Pipeline:**
1. VLM analysis extracts visual features
2. RAG search finds similar images
3. Tag recommendation synthesizes results

**Supported formats:** JPEG, PNG, WebP, GIF
**Max file size:** 10MB

**Authentication:** Requires X-API-Key header
    """,
    responses={
        200: {
            "description": "Tags successfully generated",
            "content": {
                "application/json": {
                    "example": {
                        "tags": [
                            {
                                "tag": "catgirl",
                                "confidence": 0.92,
                                "source": "vlm+rag",
                                "reason": "Matched by visual features and supporting RAG results",
                            }
                        ],
                        "metadata": {
                            "processing_time": 2.34,
                            "rag_matches_count": 3,
                            "api_version": "2.0.0",
                        },
                    }
                }
            },
        },
        400: {
            "description": "Invalid input",
            "content": {
                "application/json": {
                    "example": {
                        "error": {
                            "code": "VALIDATION_ERROR",
                            "message": "File must be an image",
                            "status": 400,
                        }
                    }
                }
            },
        },
        401: {
            "description": "Authentication required",
            "content": {
                "application/json": {
                    "example": {
                        "error": {
                            "code": "AUTHENTICATION_FAILED",
                            "message": "X-API-Key header is required",
                            "status": 401,
                        }
                    }
                }
            },
        },
        500: {
            "description": "Server error",
            "content": {
                "application/json": {
                    "example": {
                        "error": {
                            "code": "INTERNAL_ERROR",
                            "message": "Failed to tag image",
                            "status": 500,
                        }
                    }
                }
            },
        },
        503: {
            "description": "Service unavailable",
            "content": {
                "application/json": {
                    "example": {
                        "error": {
                            "code": "SERVICE_UNAVAILABLE",
                            "message": "VLM service is unavailable",
                            "status": 503,
                        }
                    }
                }
            },
        },
    },
)
async def tag_cover(
    file: UploadFile = File(..., description="Manga cover image to tag"),
    top_k: int = Form(5, ge=1, le=20, description="Number of tags to return"),
    confidence_threshold: float = Form(
        0.5, ge=0.0, le=1.0, description="Minimum confidence threshold"
    ),
    include_metadata: str = Form("true", description="Include processing metadata in response"),
    vlm_service=Depends(get_vlm_service),
    rag_service=Depends(get_rag_service),
    tag_recommender=Depends(get_tag_recommender),
):
    start_time = time.time()

    try:
        image_bytes = await validate_and_read_image(file)

        # Run the tagging pipeline
        result = await run_tagging_pipeline(
            image_bytes=image_bytes,
            top_k=top_k,
            confidence_threshold=confidence_threshold,
            vlm_service=vlm_service,
            rag_service=rag_service,
            tag_recommender=tag_recommender,
        )

        # Calculate processing time and add to metadata
        processing_time = time.time() - start_time

        # Handle include_metadata parameter
        if isinstance(include_metadata, bool):
            include_meta = include_metadata
        else:
            include_meta = include_metadata.lower() == "true" if include_metadata else False

        if include_meta and result.metadata:
            result.metadata["processing_time"] = round(processing_time, 2)
            result.metadata["timestamp"] = time.time()
        elif not include_meta:
            result.metadata = None

        logger.info(
            f"Tagging completed in {processing_time:.2f}s, returned {len(result.tags)} tags"
        )
        return result

    except ValidationError:
        raise
    except HTTPException:
        raise
    except ValueError as e:
        raise ValidationError(detail=str(e))
    except Exception as e:
        logger.error(f"Failed to tag image: {e}")
        raise HTTPException(status_code=500, detail="Failed to tag image") from None


@router.post(
    "/generate-manga-description",
    response_model=MangaDescriptionResponse,
    dependencies=[Depends(verify_api_key)],
    summary="Generate manga description",
    description="""
Generate a detailed visual description of a manga cover using VLM.

This endpoint:
1. Analyzes the manga cover using Vision Language Model
2. Extracts detailed visual information including characters, themes, and art style
3. Returns a comprehensive text description of the cover

**Supported formats:** JPEG, PNG, WebP, GIF
**Max file size:** 10MB

**Authentication:** Requires X-API-Key header
    """,
    responses={
        200: {
            "description": "Description successfully generated",
            "content": {
                "application/json": {
                    "example": {
                        "description": "A young girl with blonde hair and cat ears, wearing a navy blue school uniform with a red ribbon. She has a playful expression and is holding a cat plushie.",
                        "metadata": {
                            "processing_time": 3.21,
                            "vlm_analysis": {
                                "description": "A young girl with blonde hair and cat ears...",
                                "characters": ["catgirl"],
                                "themes": ["slice_of_life"],
                                "art_style": "Modern anime style",
                                "genre_indicators": ["comedy", "romance"],
                            },
                        },
                    }
                }
            },
        },
        400: {
            "description": "Invalid input",
            "content": {
                "application/json": {
                    "example": {
                        "error": {
                            "code": "VALIDATION_ERROR",
                            "message": "File must be an image",
                            "status": 400,
                        }
                    }
                }
            },
        },
        401: {
            "description": "Authentication required",
            "content": {
                "application/json": {
                    "example": {
                        "error": {
                            "code": "AUTHENTICATION_FAILED",
                            "message": "X-API-Key header is required",
                            "status": 401,
                        }
                    }
                }
            },
        },
        503: {
            "description": "Service unavailable",
            "content": {
                "application/json": {
                    "example": {
                        "error": {
                            "code": "SERVICE_UNAVAILABLE",
                            "message": "VLM service is unavailable",
                            "status": 503,
                        }
                    }
                }
            },
        },
    },
)
async def generate_manga_description(
    file: UploadFile = File(..., description="Manga cover image to analyze"),
    include_metadata: str = Form("true", description="Include processing metadata in response"),
    vlm_service=Depends(get_vlm_service),
):
    start_time = time.time()
    try:
        # Validate file type
        if not file.content_type or not file.content_type.startswith("image/"):
            raise ValidationError(detail="File must be an image")

        # Read and validate image
        image_bytes = await file.read()
        is_valid, error_msg = validate_image(image_bytes)
        if not is_valid:
            raise ValidationError(detail=error_msg)

        # Check VLM service availability
        if not vlm_service:
            raise ServiceUnavailableError(service="VLM")

        # Generate description via VLM
        vlm_analysis = await vlm_service.extract_metadata(image_bytes)
        description = vlm_analysis.get("description", "Failed to generate description")

        processing_time = time.time() - start_time

        # Prepare response metadata
        metadata = None
        if include_metadata.lower() == "true":
            metadata = {"processing_time": round(processing_time, 2), "vlm_analysis": vlm_analysis}

        logger.info(f"Description generated in {processing_time:.2f}s")
        return MangaDescriptionResponse(description=description, metadata=metadata)

    except ValidationError:
        raise
    except ServiceUnavailableError:
        raise
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to generate description: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate description") from None


@router.get(
    "/tags",
    response_model=TagsListResponse,
    summary="List available tags",
    description="""
List available tags from the 611-tag library.

This endpoint provides:
- Full list of all tags
- Filter by category
- Search by tag name
- Pagination support

**Categories:**
- character: Character traits and types
- clothing: Clothing and accessories
- body: Body features
- action: Actions and poses
- theme: Themes and settings
- other: Miscellaneous tags
    """,
    responses={
        200: {
            "description": "Tags retrieved successfully",
            "content": {
                "application/json": {
                    "example": {
                        "tags": [
                            {
                                "tag_name": "catgirl",
                                "description": "A character with cat ears and tail",
                                "category": "character",
                            },
                            {
                                "tag_name": "school_uniform",
                                "description": "Standard school clothing",
                                "category": "clothing",
                            },
                        ],
                        "total": 611,
                    }
                }
            },
        },
        400: {
            "description": "Invalid parameters",
            "content": {
                "application/json": {
                    "example": {
                        "error": {
                            "code": "VALIDATION_ERROR",
                            "message": "limit must be between 1 and 500",
                            "status": 400,
                        }
                    }
                }
            },
        },
    },
    tags=["Tags"],
)
async def list_tags(
    category: Optional[str] = None,
    search: Optional[str] = None,
    limit: int = 100,
    tag_lib=Depends(get_tag_library_service),
):
    # Validate limit parameter
    if limit < 1 or limit > 500:
        raise HTTPException(status_code=400, detail="limit must be between 1 and 500")

    # Sanitize search parameter
    if search is not None:
        # Strip leading/trailing whitespace
        search = search.strip()

        # Check for empty string after stripping
        if not search:
            search = None
        else:
            # Check length limit
            if len(search) > 200:
                raise HTTPException(status_code=400, detail="search must be 200 characters or less")

            # Check for null bytes and control characters (ASCII < 32 except tab, newline, carriage return)
            for char in search:
                char_ord = ord(char)
                if char_ord < 32 and char_ord not in (9, 10, 13):
                    raise HTTPException(
                        status_code=400, detail="search must not contain control characters"
                    )

    try:
        if not tag_lib.tags:
            return TagsListResponse(tags=[], total=0)

        # Get tags based on filters
        if search:
            tag_names = tag_lib.search_tags(search, limit=limit)
            tags_data = [
                TagInfo(tag_name=name, description=tag_lib.get_tag_description(name), category=None)
                for name in tag_names
            ]
        elif category:
            tag_names = tag_lib.get_tags_by_category(category)[:limit]
            tags_data = [
                TagInfo(tag_name=name, description=tag_lib.get_tag_description(name), category=None)
                for name in tag_names
            ]
        else:
            tags_data = [
                TagInfo(tag_name=tag["tag_name"], description=tag.get("description"), category=None)
                for tag in tag_lib.tags[:limit]
            ]

        return TagsListResponse(tags=tags_data, total=len(tag_lib.tag_names))

    except Exception as e:
        logger.error(f"Failed to load tags: {e}")
        raise HTTPException(status_code=500, detail="Failed to load tags") from None


@router.get(
    "/tags/categories",
    summary="Get tag categories",
    description="""
Get all tag categories with their counts.

This endpoint returns:
- List of all categories
- Number of tags in each category
- Total number of tags
    """,
    responses={
        200: {
            "description": "Categories retrieved successfully",
            "content": {
                "application/json": {
                    "example": {
                        "categories": {
                            "character": 120,
                            "clothing": 85,
                            "body": 45,
                            "action": 60,
                            "theme": 180,
                            "other": 121,
                        },
                        "total_tags": 611,
                    }
                }
            },
        },
        500: {
            "description": "Server error",
            "content": {
                "application/json": {
                    "example": {
                        "error": {
                            "code": "INTERNAL_ERROR",
                            "message": "Failed to get tag categories",
                            "status": 500,
                        }
                    }
                }
            },
        },
    },
    tags=["Tags"],
)
async def get_tag_categories(
    tag_lib=Depends(get_tag_library_service),
):
    """Get tag categories and their counts."""
    try:
        categories = {}
        for cat_name, tags in tag_lib.tag_categories.items():
            categories[cat_name] = len(tags)

        return {"categories": categories, "total_tags": len(tag_lib.tag_names)}
    except Exception as e:
        logger.error(f"Failed to get tag categories: {e}")
        raise HTTPException(status_code=500, detail="Failed to get tag categories")


@router.post(
    "/tag-cover/async",
    response_model=dict,
    dependencies=[Depends(verify_api_key)],
    summary="Tag a manga cover asynchronously with WebSocket updates",
    description="""
Submit a tagging job for asynchronous processing.

This endpoint:
1. Accepts an image and returns a job ID immediately
2. Processes the image in the background
3. Sends progress updates via WebSocket to `/ws/jobs/{job_id}`

**Steps to use:**
1. Submit the job with this endpoint to get a job_id
2. Connect to WebSocket at `/ws/jobs/{job_id}`
3. Receive real-time progress updates

**WebSocket message format:**
```json
{
    "job_id": "uuid",
    "stage": "vlm|rag|recommend|complete",
    "progress": 0.0-1.0,
    "message": "Status message",
    "timestamp": "2024-01-01T00:00:00"
}
```

**Supported formats:** JPEG, PNG, WebP, GIF
**Max file size:** 10MB

**Authentication:** Requires X-API-Key header
    """,
    responses={
        200: {
            "description": "Job submitted successfully",
            "content": {
                "application/json": {
                    "example": {
                        "job_id": "550e8400-e29b-41d4-a716-446655440000",
                        "status": "processing",
                        "websocket_url": "ws://localhost:8000/api/v1/ws/jobs/550e8400-e29b-41d4-a716-446655440000",
                    }
                }
            },
        },
    },
)
async def tag_cover_async(
    request: Request,
    file: UploadFile = File(..., description="Manga cover image to tag"),
    top_k: int = Form(5, ge=1, le=20, description="Number of tags to return"),
    confidence_threshold: float = Form(
        0.5, ge=0.0, le=1.0, description="Minimum confidence threshold"
    ),
    vlm_service=Depends(get_vlm_service),
    rag_service=Depends(get_rag_service),
    tag_recommender=Depends(get_tag_recommender),
):
    """Submit an async tagging job with WebSocket progress updates."""
    # Generate job ID
    job_id = str(uuid.uuid4())

    image_bytes = await validate_and_read_image(file)

    # Start background task — pass job_id so progress updates reach the right WS subscriber
    asyncio.create_task(
        run_tagging_pipeline(
            image_bytes=image_bytes,
            top_k=top_k,
            confidence_threshold=confidence_threshold,
            progress_callback=send_progress_update,
            vlm_service=vlm_service,
            rag_service=rag_service,
            tag_recommender=tag_recommender,
            job_id=job_id,
        )
    )

    # Build websocket URL from the actual request host so it works on any port
    host = request.url.netloc
    scheme = "wss" if request.url.scheme == "https" else "ws"
    return {
        "job_id": job_id,
        "status": "processing",
        "websocket_url": f"{scheme}://{host}/api/v1/ws/jobs/{job_id}",
    }
