"""Pydantic models for request/response validation.

This module re-exports all models from app.domain.models for backward compatibility.
"""

from app.domain.models import (
    ErrorDetail,
    ErrorResponse,
    HealthResponse,
    JobResponse,
    JobStatusResponse,
    MangaDescriptionResponse,
    ProcessingMetadata,
    ProgressMessage,
    ProgressUpdate,
    RAGAddRequest,
    RAGAddResponse,
    RAGMatch,
    ReadyResponse,
    TagCoverRequest,
    TagCoverResponse,
    TagInfo,
    TagResult,
    TagsListResponse,
    VLMMetadata,
)

__all__ = [
    "ErrorDetail",
    "ErrorResponse",
    "HealthResponse",
    "JobResponse",
    "JobStatusResponse",
    "MangaDescriptionResponse",
    "ProcessingMetadata",
    "ProgressMessage",
    "ProgressUpdate",
    "RAGAddRequest",
    "RAGAddResponse",
    "RAGMatch",
    "ReadyResponse",
    "TagCoverRequest",
    "TagCoverResponse",
    "TagInfo",
    "TagResult",
    "TagsListResponse",
    "VLMMetadata",
]
