"""Abstract interface for embedding services.

Defines the contract for all embedding implementations in the system.
"""

from abc import ABC, abstractmethod
from typing import Any, List, Protocol


class EmbeddingService(Protocol):
    """Protocol defining the embedding service interface."""

    def is_available(self) -> bool:
        """Check if the embedding service is available."""
        ...

    @abstractmethod
    async def generate_embedding(self, data: Any) -> List[float]:
        """Generate embedding for data.

        Args:
            data: Input data (image bytes, text, etc.)

        Returns:
            List of embedding values
        """
        ...


class TextEmbeddingService(Protocol):
    """Protocol for text-based embedding services."""

    def is_available(self) -> bool:
        """Check if the service is available."""
        ...

    async def encode(self, texts: List[str]) -> List[List[float]]:
        """Encode texts to embeddings.

        Args:
            texts: List of text strings

        Returns:
            List of embedding vectors
        """
        ...


class ImageEmbeddingService(Protocol):
    """Protocol for image-based embedding services."""

    def is_available(self) -> bool:
        """Check if the service is available."""
        ...

    async def encode(self, image_bytes: bytes) -> List[float]:
        """Encode image to embedding.

        Args:
            image_bytes: Raw image bytes

        Returns:
            Embedding vector
        """
        ...


# Service type identifiers for runtime dispatch
class EmbeddingServiceType:
    """Enum-like class for embedding service types."""

    CHINESE_TEXT = "chinese_text"  # ChineseEmbeddingService
    LM_STUDIO_IMAGE = "lm_studio_image"  # LMStudioEmbeddingService
    CLIP = "clip"  # Future: CLIP-based
    SENTENCE_TRANSFORMER = "sentence_transformer"  # Future: General text
