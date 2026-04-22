"""Unit tests for embedding cache service.

Tests the LRU embedding cache for performance and correctness.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.embedding_cache import EmbeddingCache, get_embedding_cache


class TestEmbeddingCache:
    """Test embedding cache functionality."""

    def test_cache_starts_empty(self):
        """Cache should start empty."""
        cache = EmbeddingCache(max_size=3)
        assert cache.size() == 0

    def test_put_and_get(self):
        """Should be able to store and retrieve embeddings."""
        cache = EmbeddingCache(max_size=3)
        data = b"test_image_data"
        embedding = np.array([1.0, 2.0, 3.0])

        cache.put(data, embedding)
        retrieved = cache.get(data)

        assert retrieved is not None
        np.testing.assert_array_equal(retrieved, embedding)

    def test_cache_miss(self):
        """Getting non-existent data should return None."""
        cache = EmbeddingCache(max_size=3)

        result = cache.get(b"nonexistent")
        assert result is None

    def test_lru_eviction(self):
        """Cache should evict oldest entry when full."""
        cache = EmbeddingCache(max_size=2)

        # Add two entries
        cache.put(b"data1", np.array([1.0]))
        cache.put(b"data2", np.array([2.0]))
        assert cache.size() == 2

        # Add third - should evict first
        cache.put(b"data3", np.array([3.0]))
        assert cache.size() == 2

        # First entry should be evicted
        assert cache.get(b"data1") is None
        # Second and third should still exist
        assert cache.get(b"data2") is not None
        assert cache.get(b"data3") is not None

    def test_lru_update_moves_to_end(self):
        """Updating an entry should move it to most recent."""
        cache = EmbeddingCache(max_size=3)
        data = b"test"

        cache.put(data, np.array([1.0]))
        cache.put(b"other", np.array([2.0]))
        cache.put(b"third", np.array([3.0]))

        # Update existing key
        cache.put(data, np.array([1.5]))

        # Add another - data should NOT be evicted since it was just used
        cache.put(b"fourth", np.array([4.0]))

        assert cache.get(data) is not None
        np.testing.assert_array_equal(cache.get(data), np.array([1.5]))

    def test_clear(self):
        """Clear should remove all entries."""
        cache = EmbeddingCache(max_size=3)
        cache.put(b"data1", np.array([1.0]))
        cache.put(b"data2", np.array([2.0]))

        cache.clear()
        assert cache.size() == 0
        assert cache.get(b"data1") is None

    def test_contains(self):
        """Should be able to check membership."""
        cache = EmbeddingCache(max_size=3)
        data = b"test"

        assert data not in cache

        cache.put(data, np.array([1.0]))
        assert data in cache

    def test_different_data_same_embedding(self):
        """Different data should have different cache keys."""
        cache = EmbeddingCache(max_size=3)
        embedding = np.array([1.0, 2.0])

        cache.put(b"data1", embedding)
        cache.put(b"data2", embedding)

        assert cache.size() == 2
        assert cache.get(b"data1") is not None
        assert cache.get(b"data2") is not None


class TestGetEmbeddingCacheSingleton:
    """Test the global cache singleton."""

    def test_returns_same_instance(self):
        """get_embedding_cache should return the same instance."""
        cache1 = get_embedding_cache()
        cache2 = get_embedding_cache()
        assert cache1 is cache2

    def test_singleton_maintains_state(self):
        """Singleton should maintain state between calls."""
        cache1 = get_embedding_cache()
        cache1.put(b"test", np.array([1.0, 2.0]))

        cache2 = get_embedding_cache()
        assert cache2.size() == 1
        np.testing.assert_array_equal(cache2.get(b"test"), np.array([1.0, 2.0]))

    def test_custom_max_size_only_on_first_call(self):
        """Custom max_size should only apply on first creation."""
        # Reset global singleton first
        import app.core.embedding_cache as ec

        original_cache = ec._embedding_cache
        ec._embedding_cache = None

        try:
            # First call with custom size
            cache1 = get_embedding_cache(max_size=50)
            assert cache1._max_size == 50

            # Second call should ignore max_size
            cache2 = get_embedding_cache(max_size=10)
            assert cache2._max_size == 50  # Still the original
        finally:
            # Restore original singleton
            ec._embedding_cache = original_cache


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
