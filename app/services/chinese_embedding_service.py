"""Chinese Embedding Service for text-based tag similarity search.

Uses sentence-transformers models for Chinese text embeddings.
Supports batch processing and similarity calculations.
"""

import logging
from typing import List, Dict, Any, Optional
import asyncio
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from app.config import settings

# Try to import sentence-transformers, but handle gracefully if not available
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError as e:
    logger = logging.getLogger(__name__)
    logger.warning(f"sentence-transformers not available: {e}")
    SentenceTransformer = None
    SENTENCE_TRANSFORMERS_AVAILABLE = False

logger = logging.getLogger(__name__)


class ChineseEmbeddingService:
    """Chinese embedding service for text-based tag similarity search."""

    def __init__(self):
        """Initialize Chinese embedding service."""
        self.model = None
        self.device = settings.DEVICE
        self.model_name = getattr(
            settings,
            "CHINESE_EMBEDDING_MODEL",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        )
        self.target_dim = 512
        self.executor = ThreadPoolExecutor(max_workers=2)
        self._initialized = False

        # Initialize model if enabled
        if getattr(settings, "USE_CHINESE_EMBEDDINGS", True):
            self._init_model()

def _init_model(self):
        """Initialize the sentence transformer model."""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            logger.warning("sentence-transformers not available, using fallback mode")
            self.model = None
            self._initialized = False
            return
            
        try:
            logger.info(f"Loading Chinese embedding model: {self.model_name}")
            
            # Load model with appropriate device
            if self.device == "cuda" and not self._cuda_available():
                logger.warning("CUDA not available, falling back to CPU")
                self.device = "cpu"
            
            self.model = SentenceTransformer(
                self.model_name,
                device=self.device
            )
            
            # Test the model
            test_embedding = self.model.encode(["测试"], convert_to_numpy=True)
            self.target_dim = test_embedding.shape[0]
            
            self._initialized = True
            logger.info(f"Chinese embedding model loaded successfully. Embedding dim: {self.target_dim}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Chinese embedding model: {e}")
            self.model = None
            self._initialized = False

        def _cuda_available(self) -> bool:
                """Check if CUDA is available."""
        try:
            import torch

            return torch.cuda.is_available()
        except ImportError:
            return False

        def is_available(self) -> bool:
        """Check if the service is available."""
        return self._initialized and self.model is not None

    async def encode_batch(self, texts: List[str]) -> np.ndarray:
        """
        Encode a batch of texts to embeddings.

        Args:
            texts: List of text strings to encode

        Returns:
            numpy array of embeddings with shape (len(texts), embedding_dim)
        """
        if not self.is_available():
            logger.warning("Chinese embedding service not available, using fallback")
            return self._generate_fallback_embeddings(texts)

        try:
            # Run encoding in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            embeddings = await loop.run_in_executor(
                self.executor, self._encode_texts_sync, texts
            )

            return embeddings

        except Exception as e:
            logger.error(f"Batch encoding failed: {e}")
            return self._generate_fallback_embeddings(texts)

    def _encode_texts_sync(self, texts: List[str]) -> np.ndarray:
        """Synchronous text encoding."""
        # Filter out empty texts
        valid_texts = [text if text else "" for text in texts]

        # Generate embeddings
        embeddings = self.model.encode(
            valid_texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            batch_size=32,
            show_progress_bar=False,
        )

        return embeddings

    async def encode_single(self, text: str) -> np.ndarray:
        """
        Encode a single text to embedding.

        Args:
            text: Text string to encode

        Returns:
            numpy array of embedding with shape (embedding_dim,)
        """
        embeddings = await self.encode_batch([text])
        return embeddings[0]

    def _generate_fallback_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate fallback embeddings using deterministic hashing."""
        import hashlib

        embeddings = []
        for text in texts:
            # Create deterministic embedding from text
            hash_obj = hashlib.md5(text.encode("utf-8"))
            seed = int(hash_obj.hexdigest()[:8], 16)

            np.random.seed(seed)
            embedding = np.random.randn(self.target_dim).astype(np.float32)

            # Normalize to unit vector
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm

            embeddings.append(embedding)

        return np.array(embeddings, dtype=np.float32)

    async def calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate cosine similarity between two texts.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Cosine similarity score (0-1)
        """
        try:
            embeddings = await self.encode_batch([text1, text2])

            # Calculate cosine similarity
            similarity = np.dot(embeddings[0], embeddings[1])

            # Ensure result is in valid range
            similarity = max(0.0, min(1.0, float(similarity)))

            return similarity

        except Exception as e:
            logger.error(f"Similarity calculation failed: {e}")
            return 0.0

    async def find_most_similar(
        self,
        query_text: str,
        candidate_texts: List[str],
        top_k: int = 5,
        threshold: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """
        Find most similar texts to query from candidates.

        Args:
            query_text: Query text
            candidate_texts: List of candidate texts
            top_k: Number of results to return
            threshold: Minimum similarity threshold

        Returns:
            List of dictionaries with text, similarity, and index
        """
        if not candidate_texts:
            return []

        try:
            # Encode all texts
            all_texts = [query_text] + candidate_texts
            embeddings = await self.encode_batch(all_texts)

            query_embedding = embeddings[0]
            candidate_embeddings = embeddings[1:]

            # Calculate similarities
            similarities = np.dot(candidate_embeddings, query_embedding)

            # Get top-k results
            top_indices = np.argsort(similarities)[::-1][:top_k]

            results = []
            for idx in top_indices:
                similarity = float(similarities[idx])
                if similarity >= threshold:
                    results.append(
                        {
                            "text": candidate_texts[idx],
                            "similarity": similarity,
                            "index": idx,
                        }
                    )

            return results

        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            return []

    async def search_tags_by_text(
        self,
        query_text: str,
        tag_list: List[str],
        top_k: int = 10,
        threshold: float = 0.3,
    ) -> List[Dict[str, Any]]:
        """
        Search for tags by text similarity.

        Args:
            query_text: Query text (e.g., image description)
            tag_list: List of available tags
            top_k: Number of results to return
            threshold: Minimum similarity threshold

        Returns:
            List of dictionaries with tag and similarity score
        """
        if not tag_list:
            return []

        try:
            # Use the general similarity search
            results = await self.find_most_similar(
                query_text=query_text,
                candidate_texts=tag_list,
                top_k=top_k,
                threshold=threshold,
            )

            # Convert to tag format
            tag_results = []
            for result in results:
                tag_results.append(
                    {"tag": result["text"], "similarity": result["similarity"]}
                )

            return tag_results

        except Exception as e:
            logger.error(f"Tag search failed: {e}")
            return []

    def get_stats(self) -> Dict[str, Any]:
        """Get service statistics."""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "target_dim": self.target_dim,
            "initialized": self._initialized,
            "available": self.is_available(),
        }

    def __del__(self):
        """Cleanup resources."""
        if hasattr(self, "executor"):
            self.executor.shutdown(wait=False)


# Singleton instance
_chinese_embedding_service: Optional[ChineseEmbeddingService] = None


def get_chinese_embedding_service() -> ChineseEmbeddingService:
    """Get or create Chinese embedding service singleton."""
    global _chinese_embedding_service
    if _chinese_embedding_service is None:
        _chinese_embedding_service = ChineseEmbeddingService()
    return _chinese_embedding_service
