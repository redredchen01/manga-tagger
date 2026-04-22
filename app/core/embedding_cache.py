"""Embedding cache service for RAG.

Provides a centralized LRU cache for embeddings to improve performance.
"""

import hashlib
from collections import OrderedDict
from typing import Optional

import numpy as np


class EmbeddingCache:
    """LRU-style cache for embeddings with configurable max size."""

    def __init__(self, max_size: int = 100):
        """
        Initialize the embedding cache.

        Args:
            max_size: Maximum number of embeddings to cache
        """
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._max_size = max_size

    def get_cache_key(self, data: bytes) -> str:
        """Generate cache key from data using MD5 hash."""
        return hashlib.md5(data).hexdigest()

    def get(self, data: bytes) -> Optional[np.ndarray]:
        """
        Get cached embedding for data.

        Args:
            data: Data to get cached embedding for

        Returns:
            Cached embedding or None if not found
        """
        key = self.get_cache_key(data)
        if key in self._cache:
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            return self._cache[key].copy()
        return None

    def put(self, data: bytes, embedding: np.ndarray) -> None:
        """
        Cache an embedding.

        Args:
            data: Data the embedding was generated from
            embedding: The embedding to cache
        """
        key = self.get_cache_key(data)

        # If key already exists, update and move to end
        if key in self._cache:
            self._cache.move_to_end(key)
            self._cache[key] = embedding.copy()
            return

        # Evict oldest if cache is full
        if len(self._cache) >= self._max_size:
            self._cache.popitem(last=False)

        self._cache[key] = embedding.copy()

    def clear(self) -> None:
        """Clear all cached embeddings."""
        self._cache.clear()

    def size(self) -> int:
        """Get current cache size."""
        return len(self._cache)

    def __len__(self) -> int:
        """Get current cache size."""
        return len(self._cache)

    def __contains__(self, data: bytes) -> bool:
        """Check if data is in cache."""
        return self.get_cache_key(data) in self._cache


# Global cache instance
_embedding_cache: Optional[EmbeddingCache] = None


def get_embedding_cache(max_size: int = 100) -> EmbeddingCache:
    """
    Get or create the global embedding cache instance.

    Args:
        max_size: Maximum cache size (only used on first creation)

    Returns:
        The global embedding cache instance
    """
    global _embedding_cache
    if _embedding_cache is None:
        _embedding_cache = EmbeddingCache(max_size=max_size)
    return _embedding_cache
