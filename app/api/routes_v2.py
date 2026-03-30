"""API routes for Manga Tagger - Optimized Version.

Integrates tag library matching with VLM analysis and RAG for intelligent tag recommendations.
"""

import json
import logging
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response

from app.auth import verify_api_key
from app.config import settings
from app.exceptions import ValidationError, ServiceUnavailableError
from app.models import (
    HealthResponse,
    JobResponse,
    JobStatusResponse,
    MangaDescriptionResponse,
    RAGAddResponse,
    TagCoverResponse,
    TagInfo,
    TagResult,
    TagsListResponse,
)
from app.services.lm_studio_llm_service import LMStudioLLMService
from app.services.lm_studio_vlm_service_v2 import LMStudioVLMService
from app.services.pipeline import run_tagging_pipeline, run_tagging_pipeline_with_progress
from app.services.rag_service import RAGService
from app.services.tag_library_service import get_tag_library_service
from app.services.tag_recommender_service import get_tag_recommender_service
from app.utils import validate_image
from app.websocket.connection_manager import manager

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
                            "vlm": "glm-4.6v-flash",
                            "vlm_initialized": True,
                            "llm": "llama-3.2-8b-instruct",
                            "llm_initialized": True,
                            "rag": "ChromaDB (Local)",
                            "rag_initialized": True,
                            "lm_studio_mode": True,
                            "tag_library": 611,
                        },
                    }
                }
            },
        }
    },
    tags=["Health"],
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
        "vlm": settings.LM_STUDIO_VISION_MODEL if settings.USE_LM_STUDIO else "Disabled",
        "vlm_initialized": _vlm_service is not None,
        "llm": settings.LM_STUDIO_TEXT_MODEL if settings.USE_LM_STUDIO else "Disabled",
        "llm_initialized": _llm_service is not None,
        "rag": "ChromaDB (Local)",
        "rag_initialized": _rag_service is not None,
        "lm_studio_mode": settings.USE_LM_STUDIO,
        "tag_library": total_tags,
    }

    return HealthResponse(
        status="healthy",
        version="2.0.0",
        models_loaded=models_dict,
    )


@router.post(
    "/rag/add",
    response_model=RAGAddResponse,
    dependencies=[Depends(verify_api_key)],
    summary="Add image to RAG dataset",
    description="""
Add a reference image to the RAG (Retrieval-Augmented Generation) dataset.

This endpoint:
1. Validates the uploaded image
2. Extracts CLIP embeddings from the image
3. Stores the image with associated tags in ChromaDB
4. Returns a document ID for future retrieval

**Supported formats:** JPEG, PNG, WebP, GIF
**Max file size:** 10MB

**Authentication:** Requires X-API-Key header
    """,
    responses={
        200: {
            "description": "Image successfully added to RAG",
            "content": {
                "application/json": {
                    "example": {
                        "success": True,
                        "id": "img_abc123def456",
                        "message": "Successfully added image with 3 tags",
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
                            "message": "Failed to add image to RAG dataset",
                            "status": 500,
                        }
                    }
                }
            },
        },
    },
    tags=["RAG"],
)
async def add_to_rag(
    file: UploadFile = File(..., description="Reference image to add to RAG"),
    tags: str = Form(
        ...,
        description='JSON array of tags (e.g., \'["catgirl", "school_uniform"]\')',
    ),
    metadata: str = Form(
        "{}", description='Optional JSON metadata object (e.g., \'{"source": "manual"}\')'
    ),
):
    try:
        # Parse tags
        try:
            tag_list = json.loads(tags)
            if not isinstance(tag_list, list):
                raise ValueError("Tags must be a JSON array")
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON in tags field") from None

        # Parse metadata
        try:
            metadata_dict = json.loads(metadata)
            if not isinstance(metadata_dict, dict):
                raise ValueError("Metadata must be a JSON object")
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON in metadata field") from None

        # Validate file type
        if not file.content_type or not file.content_type.startswith("image/"):
            raise ValidationError(detail="File must be an image")

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

    except ValidationError:
        raise
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to add image to RAG dataset: {e}")
        raise HTTPException(status_code=500, detail="Failed to add image to RAG dataset") from None


@router.post("/upload", response_model=RAGAddResponse, dependencies=[Depends(verify_api_key)])
async def upload_image(
    file: UploadFile = File(..., description="Reference image"),
    tags: str = Form(..., description="JSON array of tags"),
    metadata: str = Form("{}", description="JSON metadata object"),
):
    """
    Alias for /rag/add. Add an image to the RAG dataset.
    """
    return await add_to_rag(file=file, tags=tags, metadata=metadata)


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
    tags=["Tagging"],
)
async def tag_cover(
    file: UploadFile = File(..., description="Manga cover image to tag"),
    top_k: int = Form(5, ge=1, le=20, description="Number of tags to return"),
    confidence_threshold: float = Form(
        0.5, ge=0.0, le=1.0, description="Minimum confidence threshold"
    ),
    include_metadata: str = Form("true", description="Include processing metadata in response"),
):
    start_time = time.time()

    try:
        image_bytes = await validate_and_read_image(file)

        # Get services
        vlm_service = get_vlm_service()
        rag_service = get_rag_service()
        tag_recommender = get_tag_recommender()

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
    tags=["Tagging"],
)
async def generate_manga_description(
    file: UploadFile = File(..., description="Manga cover image to analyze"),
    include_metadata: str = Form("true", description="Include processing metadata in response"),
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

        # Get service
        vlm_service = get_vlm_service()
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
        tag_lib = get_tag_library()

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
    "/rag/stats",
    summary="Get RAG statistics",
    description="""
Get statistics about the RAG (Retrieval-Augmented Generation) dataset.

Returns:
- Total number of images in the collection
- Embedding dimension
- Number of unique tags
- Tag library total count
    """,
    responses={
        200: {
            "description": "RAG statistics retrieved successfully",
            "content": {
                "application/json": {
                    "example": {
                        "total_images": 150,
                        "embedding_dimension": 512,
                        "unique_tags": 89,
                        "tag_library_total": 611,
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
                            "message": "Failed to get RAG stats",
                            "status": 500,
                        }
                    }
                }
            },
        },
    },
    tags=["RAG"],
)
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
        logger.error(f"Failed to get RAG stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to get RAG stats") from None


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
async def get_tag_categories():
    """Get tag categories and their counts."""
    try:
        tag_lib = get_tag_library()

        categories = {}
        for cat_name, tags in tag_lib.tag_categories.items():
            categories[cat_name] = len(tags)

        return {"categories": categories, "total_tags": len(tag_lib.tag_names)}
    except Exception as e:
        logger.error(f"Failed to get tag categories: {e}")
        raise HTTPException(status_code=500, detail="Failed to get tag categories")


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
    tags=["Monitoring"],
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
    tags=["Monitoring"],
)
async def performance_metrics():
    """Performance metrics endpoint (internal use)."""
    import gc
    import psutil
    import os

    process = psutil.Process(os.getpid())

    return {
        "memory_mb": process.memory_info().rss / 1024 / 1024,
        "cpu_percent": process.cpu_percent(),
        "threads": process.num_threads(),
        "gc_stats": gc.get_stats(),
    }


# =============================================================================
# WebSocket endpoints for real-time updates
# =============================================================================


async def send_progress_update(job_id: str, stage: str, progress: float, message: str):
    """Send progress update via WebSocket."""
    await manager.send_update(
        job_id,
        {
            "job_id": job_id,
            "stage": stage,
            "progress": progress,
            "message": message,
            "timestamp": datetime.utcnow().isoformat(),
        },
    )


@router.websocket("/ws/jobs/{job_id}")
async def websocket_endpoint(websocket: WebSocket, job_id: str):
    """WebSocket endpoint for real-time job updates.

    Connect to receive progress updates for a specific job.

    Example:
        wscat -c ws://localhost:8000/api/v1/ws/jobs/your-job-id
    """
    await manager.connect(websocket, job_id)
    try:
        while True:
            # Keep connection alive - can receive client messages if needed
            data = await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket, job_id)


@router.post(
    "/tag-cover/async",
    response_model=Dict[str, str],
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
    tags=["Tagging"],
)
async def tag_cover_async(
    file: UploadFile = File(..., description="Manga cover image to tag"),
    top_k: int = Form(5, ge=1, le=20, description="Number of tags to return"),
    confidence_threshold: float = Form(
        0.5, ge=0.0, le=1.0, description="Minimum confidence threshold"
    ),
):
    """Submit an async tagging job with WebSocket progress updates."""
    # Generate job ID
    job_id = str(uuid.uuid4())

    image_bytes = await validate_and_read_image(file)

    # Get services
    vlm_service = get_vlm_service()
    rag_service = get_rag_service()
    tag_recommender = get_tag_recommender()

    # Start background task
    import asyncio

    asyncio.create_task(
        run_tagging_pipeline_with_progress(
            image_bytes=image_bytes,
            top_k=top_k,
            confidence_threshold=confidence_threshold,
            job_id=job_id,
            progress_callback=send_progress_update,
            vlm_service=vlm_service,
            rag_service=rag_service,
            tag_recommender=tag_recommender,
        )
    )

    return {
        "job_id": job_id,
        "status": "processing",
        "websocket_url": f"ws://localhost:8000/api/v1/ws/jobs/{job_id}",
    }
