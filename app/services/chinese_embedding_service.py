"""Re-export stub for backward compatibility.

This module now re-exports from app.infrastructure.embedding.chinese_embedding_service.
"""

from app.infrastructure.embedding.chinese_embedding_service import (
    ChineseEmbeddingService,
    get_chinese_embedding_service,
)

__all__ = ["ChineseEmbeddingService", "get_chinese_embedding_service"]
