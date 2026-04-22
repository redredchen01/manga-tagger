"""Re-export stub for backward compatibility.

This module now re-exports from app.infrastructure.lm_studio.embedding_service.
"""

from app.infrastructure.lm_studio.embedding_service import (
    LMStudioEmbeddingService,
    get_embedding_service,
)

__all__ = ["LMStudioEmbeddingService", "get_embedding_service"]
