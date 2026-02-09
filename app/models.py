"""Pydantic models for request/response validation."""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class TagResult(BaseModel):
    """Single tag result with confidence and reasoning."""

    tag: str = Field(..., description="Tag name")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score (0-1)")
    source: str = Field(..., description="Source of the tag (vlm, rag, llm, vlm+rag)")
    reason: Optional[str] = Field(
        None, description="Explanation for why this tag was assigned"
    )


class TagCoverRequest(BaseModel):
    """Request model for tag-cover endpoint."""

    top_k: int = Field(5, ge=1, le=20, description="Number of tags to return")
    confidence_threshold: float = Field(
        0.5, ge=0.0, le=1.0, description="Minimum confidence threshold"
    )
    include_metadata: bool = Field(
        True, description="Include processing metadata in response"
    )


class TagCoverResponse(BaseModel):
    """Response model for tag-cover endpoint."""

    tags: List[TagResult] = Field(..., description="List of assigned tags")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Processing metadata")


class RAGAddRequest(BaseModel):
    """Request model for adding image to RAG dataset."""

    tags: List[str] = Field(..., description="List of tags for this image")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Optional metadata")


class RAGAddResponse(BaseModel):
    """Response model for RAG add endpoint."""

    success: bool = Field(..., description="Whether the operation succeeded")
    id: str = Field(..., description="ID of the added document")
    message: str = Field(..., description="Status message")


class TagInfo(BaseModel):
    """Tag information from tag library."""

    tag_name: str = Field(..., description="Tag name")
    description: Optional[str] = Field(None, description="Tag description")
    category: Optional[str] = Field(None, description="Tag category")


class TagsListResponse(BaseModel):
    """Response model for tags list endpoint."""

    tags: List[TagInfo] = Field(..., description="List of available tags")
    total: int = Field(..., description="Total number of tags")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Service status")
    version: str = Field("1.0.0", description="API version")
    models_loaded: Dict[str, Any] = Field(..., description="Which models are loaded")


class VLMMetadata(BaseModel):
    """Metadata extracted by VLM."""

    description: str = Field(..., description="Visual description of the cover")
    characters: List[str] = Field(
        default_factory=list, description="Detected characters"
    )
    themes: List[str] = Field(default_factory=list, description="Detected themes")
    art_style: Optional[str] = Field(None, description="Art style description")
    genre_indicators: List[str] = Field(
        default_factory=list, description="Genre indicators"
    )
    tag_definitions: Dict[str, str] = Field(
        default_factory=dict, description="Definitions for candidate tags for semantic matching"
    )


class RAGMatch(BaseModel):
    """Single RAG match result."""

    id: str = Field(..., description="Document ID")
    score: float = Field(..., description="Similarity score")
    tags: List[str] = Field(..., description="Tags from matched document")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Match metadata")


class ProcessingMetadata(BaseModel):
    """Detailed processing metadata."""

    processing_time: float = Field(..., description="Total processing time in seconds")
    vlm_description: str = Field(..., description="VLM-generated description")
    rag_matches: List[RAGMatch] = Field(default_factory=list, description="RAG matches")
    vlm_metadata: Optional[VLMMetadata] = None
