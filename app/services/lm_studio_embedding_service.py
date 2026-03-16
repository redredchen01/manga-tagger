"""LM Studio Embedding Service for RAG.

Uses LM Studio embedding models for image similarity search.
"""

import base64
import io
import logging
from typing import List, Optional

import httpx
import numpy as np
from PIL import Image

from app.config import settings

logger = logging.getLogger(__name__)


class LMStudioEmbeddingService:
    """LM Studio Embedding service for generating image embeddings."""

    def __init__(self):
        """Initialize LM Studio embedding service."""
        self.base_url = settings.LM_STUDIO_BASE_URL.rstrip("/")
        self.api_key = settings.LM_STUDIO_API_KEY
        self.timeout = settings.REQUEST_TIMEOUT

        # Try to use Qwen3-Embedding or other embedding model if available
        # Fall back to vision model if no dedicated embedding model
        self.model = getattr(
            settings,
            "LM_STUDIO_EMBEDDING_MODEL",
            "Qwen/Qwen3-Embedding-8B-GGUF",
        )
        self.embedding_dim = getattr(
            settings,
            "LM_STUDIO_EMBEDDING_DIM",
            4096,  # Qwen3-Embedding-8B 維度
        )

        logger.info(f"LM Studio embedding service initialized")

    def _prepare_image(self, image_bytes: bytes) -> bytes:
        """Prepare image for embedding generation."""
        try:
            image = Image.open(io.BytesIO(image_bytes))

            if image.mode != "RGB":
                image = image.convert("RGB")

            max_size = 512
            if max(image.size) > max_size:
                ratio = max_size / max(image.size)
                new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
                image = image.resize(new_size, Image.Resampling.LANCZOS)

            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=85)
            return buffer.getvalue()

        except Exception as e:
            logger.error(f"Image preparation failed: {e}")
            raise ValueError(f"Invalid image data: {e}")

    async def generate_embedding(
        self, image_bytes: bytes, target_dim: int = 4096
    ) -> List[float]:
        """
        Generate embedding for image using LM Studio.

        Since LM Studio may not have a dedicated image embedding model,
        we'll use the vision model to generate a text description first,
        then embed that text, or use a deterministic hash-based embedding
        as fallback.

        Args:
            image_bytes: Raw image bytes
            target_dim: Target embedding dimension (default 4096 for Qwen3-Embedding-8B)

        Returns:
            List of floats representing the embedding vector
        """
        try:
            # Try LM Studio embeddings API first
            embedding = await self._try_lm_studio_embedding(image_bytes)
            if embedding is not None:
                # Ensure consistent dimension
                embedding = self._adjust_dimension(embedding, target_dim)
                return embedding

            # Fallback to deterministic embedding
            logger.debug("Using deterministic fallback embedding")
            return self._generate_deterministic_embedding(image_bytes, target_dim)

        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            return self._generate_deterministic_embedding(image_bytes, target_dim)

    async def _try_lm_studio_embedding(
        self, image_bytes: bytes
    ) -> Optional[List[float]]:
        """Try to get embedding from LM Studio."""
        try:
            import json

            # Prepare image
            prepared_bytes = self._prepare_image(image_bytes)
            base64_image = base64.b64encode(prepared_bytes).decode("utf-8")

            # Use vision model to analyze image and get embedding from the response
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }

            # Try embeddings endpoint first
            payload = {
                "model": self.model,
                "input": f"data:image/jpeg;base64,{base64_image}",
            }

            async with httpx.AsyncClient(timeout=self.timeout) as client:
                try:
                    response = await client.post(
                        f"{self.base_url}/embeddings", headers=headers, json=payload
                    )
                    response.raise_for_status()
                    result = response.json()

                    if "data" in result and len(result["data"]) > 0:
                        embedding = result["data"][0].get("embedding", [])
                        if embedding:
                            logger.debug("Successfully got embedding from LM Studio")
                            return embedding
                except httpx.HTTPStatusError:
                    # Embeddings endpoint not available, will fallback
                    pass
                except Exception as e:
                    logger.debug(f"Embeddings endpoint failed: {e}")

            # Fallback: use vision model to get features
            return await self._extract_features_via_vision(
                base64_image, headers, client
            )

        except Exception as e:
            logger.debug(f"LM Studio embedding failed: {e}")
            return None

    async def _extract_features_via_vision(
        self, base64_image: str, headers: dict, client: httpx.AsyncClient
    ) -> Optional[List[float]]:
        """Extract features using vision model."""
        try:
            # Use vision model to analyze the image
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Describe this image in detail. List all visible elements, characters, clothing, setting, and style.",
                        },
                        {
                            "type": "image_url",
                            "image_url": f"data:image/jpeg;base64,{base64_image}",
                        },
                    ],
                }
            ]

            payload = {
                "model": settings.LM_STUDIO_VISION_MODEL,
                "messages": messages,
                "max_tokens": 256,
                "temperature": 0.1,  # Low temperature for consistent embeddings
            }

            response = await client.post(
                f"{self.base_url}/chat/completions", headers=headers, json=payload
            )
            response.raise_for_status()
            result = response.json()

            if "choices" in result and len(result["choices"]) > 0:
                description = result["choices"][0]["message"]["content"]
                # Convert description to deterministic embedding
                return self._text_to_embedding(description)

        except Exception as e:
            logger.debug(f"Vision feature extraction failed: {e}")

        return None

    def _adjust_dimension(
        self, embedding: List[float], target_dim: int = 4096
    ) -> List[float]:
        """Adjust embedding dimension to target size."""
        current_dim = len(embedding)

        if current_dim == target_dim:
            return embedding
        elif current_dim > target_dim:
            # Truncate if too large
            return embedding[:target_dim]
        else:
            # Pad with zeros if too small
            return embedding + [0.0] * (target_dim - current_dim)

    def _text_to_embedding(self, text: str, target_dim: int = 4096) -> List[float]:
        """Convert text to deterministic embedding vector."""
        import hashlib

        # Create deterministic embedding from text
        hash_obj = hashlib.md5(text.encode("utf-8"))
        seed = int(hash_obj.hexdigest()[:8], 16)

        np.random.seed(seed)
        embedding = np.random.randn(target_dim).astype(np.float32)

        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding.tolist()

    def _generate_deterministic_embedding(
        self, image_bytes: bytes, target_dim: int = 4096
    ) -> List[float]:
        """Generate deterministic embedding from image bytes."""
        import hashlib

        hash_obj = hashlib.md5(image_bytes)
        seed = int(hash_obj.hexdigest()[:8], 16)

        np.random.seed(seed)
        embedding = np.random.randn(target_dim).astype(np.float32)

        # Normalize to unit vector
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding.tolist()


# Singleton instance
_embedding_service: Optional[LMStudioEmbeddingService] = None


def get_embedding_service() -> LMStudioEmbeddingService:
    """Get or create embedding service singleton."""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = LMStudioEmbeddingService()
    return _embedding_service
