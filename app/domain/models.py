"""Pydantic models for request/response validation."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class TagResult(BaseModel):
    """Single tag result with confidence and reasoning."""

    tag: str = Field(..., description="Tag name", example="catgirl")
    confidence: float = Field(
        ..., description="Confidence score (0-1)", example=0.92, ge=0.0, le=1.0
    )
    source: str = Field(
        ...,
        description="Source of the tag (vlm, rag, llm, vlm+rag)",
        example="vlm+rag",
    )
    reason: str | None = Field(
        None,
        description="Explanation for why this tag was assigned",
        example="Matched by visual features and supporting RAG results",
    )


class TagCoverRequest(BaseModel):
    """Request model for tag-cover endpoint."""

    top_k: int = Field(5, ge=1, le=20, description="Number of tags to return")
    confidence_threshold: float = Field(
        0.5, ge=0.0, le=1.0, description="Minimum confidence threshold"
    )
    include_metadata: bool = Field(True, description="Include processing metadata in response")


class TagCoverResponse(BaseModel):
    """Response model for tag-cover endpoint."""

    tags: list[TagResult] = Field(..., description="List of assigned tags")
    metadata: dict | None = Field(None, description="Processing metadata")

    model_config = {
        "json_schema_extra": {
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
    }


class RAGAddRequest(BaseModel):
    """Request model for adding image to RAG dataset."""

    tags: list[str] = Field(
        ..., description="List of tags for this image", example=["catgirl", "school_uniform"]
    )
    metadata: dict | None = Field(
        None, description="Optional metadata", example={"source": "manual"}
    )


class RAGAddResponse(BaseModel):
    """Response model for RAG add endpoint."""

    success: bool = Field(..., description="Whether the operation succeeded", example=True)
    id: str = Field(..., description="ID of the added document", example="img_abc123")
    message: str = Field(
        ..., description="Status message", example="Successfully added image with 2 tags"
    )


class TagInfo(BaseModel):
    """Tag information from tag library."""

    tag_name: str = Field(..., description="Tag name", example="catgirl")
    description: str | None = Field(
        None, description="Tag description", example="A character with cat ears and tail"
    )
    category: str | None = Field(None, description="Tag category", example="character")


class TagsListResponse(BaseModel):
    """Response model for tags list endpoint."""

    tags: list[TagInfo] = Field(..., description="List of available tags")
    total: int = Field(..., description="Total number of tags", example=611)

    model_config = {
        "json_schema_extra": {
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
    }


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Service status", example="healthy")
    version: str = Field(..., description="API version", example="2.0.0")
    models_loaded: dict = Field(..., description="Which models are loaded")

    model_config = {
        "json_schema_extra": {
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
            }
        }
    }


class VLMMetadata(BaseModel):
    """Metadata extracted by VLM."""

    description: str = Field(
        ...,
        description="Visual description of the cover",
        example="A girl with cat ears wearing school uniform",
    )
    characters: list[str] = Field(
        default_factory=list, description="Detected characters", example=["catgirl"]
    )
    themes: list[str] = Field(
        default_factory=list, description="Detected themes", example=["slice_of_life"]
    )
    art_style: str | None = Field(
        None, description="Art style description", example="Modern anime style"
    )
    genre_indicators: list[str] = Field(
        default_factory=list,
        description="Genre indicators",
        example=["comedy", "romance"],
    )
    tag_definitions: dict[str, str] = Field(
        default_factory=dict,
        description="Definitions for candidate tags for semantic matching",
        example={"catgirl": "A character with cat ears, tail, or other feline features"},
    )


class RAGMatch(BaseModel):
    """Single RAG match result."""

    id: str = Field(..., description="Document ID", example="img_abc123")
    score: float = Field(..., description="Similarity score", example=0.95, ge=0.0, le=1.0)
    tags: list[str] = Field(..., description="Tags from matched document", example=["catgirl"])
    metadata: dict | None = Field(None, description="Match metadata")


class ProcessingMetadata(BaseModel):
    """Detailed processing metadata."""

    processing_time: float = Field(
        ..., description="Total processing time in seconds", example=2.34
    )
    vlm_description: str = Field(
        ..., description="VLM-generated description", example="A girl with cat ears"
    )
    rag_matches: list[RAGMatch] = Field(default_factory=list, description="RAG matches")
    vlm_metadata: VLMMetadata | None = None


class MangaDescriptionResponse(BaseModel):
    """Response model for generate-manga-description endpoint."""

    description: str = Field(
        ...,
        description="VLM-generated description",
        example="A young girl with blonde hair and cat ears, wearing a navy blue school uniform with a red ribbon. She has a playful expression and is holding a cat plushie.",
    )
    metadata: dict | None = Field(None, description="Processing metadata")

    model_config = {
        "json_schema_extra": {
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
    }


# Error response models for consistent error handling
class ErrorDetail(BaseModel):
    """Error detail model."""

    code: str = Field(..., description="Error code", example="VALIDATION_ERROR")
    message: str = Field(
        ..., description="Human-readable error message", example="File must be an image"
    )
    status: int = Field(..., description="HTTP status code", example=400)


class ErrorResponse(BaseModel):
    """Standard error response model."""

    error: ErrorDetail = Field(..., description="Error details")

    model_config = {
        "json_schema_extra": {
            "example": {
                "error": {
                    "code": "VALIDATION_ERROR",
                    "message": "File must be an image",
                    "status": 400,
                }
            }
        }
    }


class JobResponse(BaseModel):
    """Response model for async job submission."""

    job_id: str = Field(..., description="Unique job identifier")
    status: str = Field(..., description="Job status (queued, processing)")
    status_url: str = Field(..., description="URL to check job status")
    created_at: datetime = Field(..., description="Job creation timestamp")

    model_config = {
        "json_schema_extra": {
            "example": {
                "job_id": "550e8400-e29b-41d4-a716-446655440000",
                "status": "queued",
                "status_url": "http://localhost:8000/api/v1/jobs/550e8400-e29b-41d4-a716-446655440000",
                "created_at": "2024-01-01T00:00:00",
            }
        }
    }


class JobStatusResponse(BaseModel):
    """Response model for job status check."""

    job_id: str = Field(..., description="Unique job identifier")
    status: str = Field(..., description="Job status (queued, processing, completed, failed)")
    result: Optional[TagCoverResponse] = Field(None, description="Tagging result if completed")
    error: Optional[str] = Field(None, description="Error message if failed")

    model_config = {
        "json_schema_extra": {
            "example": {
                "job_id": "550e8400-e29b-41d4-a716-446655440000",
                "status": "completed",
                "result": {
                    "tags": [
                        {
                            "tag": "catgirl",
                            "confidence": 0.92,
                            "source": "vlm+rag",
                            "reason": "Matched by visual features",
                        }
                    ],
                    "metadata": {
                        "processing_time": 2.34,
                        "rag_matches_count": 3,
                        "api_version": "2.0.0",
                    },
                },
                "error": None,
            }
        }
    }


# Additional response models for progress tracking
class ReadyResponse(BaseModel):
    """Response model for /ready endpoint."""

    ready: bool = Field(..., description="Whether service is ready")
    message: str | None = Field(None, description="Status message")


class ProgressUpdate(BaseModel):
    """Progress update for long-running operations."""

    job_id: str = Field(..., description="Job identifier")
    progress: float = Field(..., description="Progress percentage (0-100)", ge=0.0, le=100.0)
    status: str = Field(..., description="Current status")
    message: str | None = Field(None, description="Status message")


class ProgressMessage(BaseModel):
    """WebSocket progress message."""

    type: str = Field(..., description="Message type")
    job_id: str = Field(..., description="Job identifier")
    progress: float = Field(..., description="Progress percentage")
    message: str | None = Field(None, description="Status message")
    data: dict | None = Field(None, description="Additional data")
