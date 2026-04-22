"""Chinese embedding infrastructure service."""

from app.infrastructure.embedding.chinese_embedding_service import (
    ChineseEmbeddingService,
    get_chinese_embedding_service,
)

__all__ = ["ChineseEmbeddingService", "get_chinese_embedding_service"]
