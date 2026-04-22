"""RAG (Retrieval-Augmented Generation) endpoints for the API."""

import json
import logging
from typing import Dict

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from app.auth import verify_api_key
from app.config import settings
from app.exceptions import ValidationError
from app.models import RAGAddResponse

logger = logging.getLogger(__name__)

router = APIRouter(tags=["RAG"])

# Module-level service instance (singleton pattern from original routes_v2.py)
_rag_service = None


def get_rag_service():
    """Get or create RAG service singleton."""
    global _rag_service
    if _rag_service is None:
        from app.services.rag_service import RAGService

        _rag_service = RAGService()
    return _rag_service


def get_tag_library():
    """Get or create tag library singleton."""
    from app.services.tag_library_service import get_tag_library_service

    return get_tag_library_service()


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
