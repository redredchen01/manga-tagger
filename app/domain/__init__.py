"""Domain business logic layer.

This module contains core domain models, constants, and prompts that are
used throughout the application.
"""

# Re-export models
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

# Re-export constants
from app.domain.constants import (
    EXACT_MATCH_BOOST,
    MIN_ACCEPTABLE_CONFIDENCE,
    PARTIAL_MATCH_BOOST,
    RAG_SIMILARITY_THRESHOLD,
    RAG_SUPPORT_BOOST,
    RAG_SUPPORT_DECAY,
    SEMANTIC_MATCH_PENALTY,
    SEMANTIC_SIBLING_THRESHOLD,
    SEMANTIC_SIBLINGS,
    SENSITIVE_TAGS,
    TAG_FREQUENCY_CALIBRATION,
)

# Re-export prompts
from app.domain.prompts import get_optimized_prompt, get_safe_prompt

__all__ = [
    # Models
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
    # Constants
    "SENSITIVE_TAGS",
    "TAG_FREQUENCY_CALIBRATION",
    "SEMANTIC_SIBLINGS",
    "SEMANTIC_SIBLING_THRESHOLD",
    "EXACT_MATCH_BOOST",
    "PARTIAL_MATCH_BOOST",
    "SEMANTIC_MATCH_PENALTY",
    "RAG_SUPPORT_BOOST",
    "RAG_SUPPORT_DECAY",
    "MIN_ACCEPTABLE_CONFIDENCE",
    "RAG_SIMILARITY_THRESHOLD",
    # Prompts
    "get_safe_prompt",
    "get_optimized_prompt",
]
