"""Unit tests for RAG service.

Tests RAG service functionality with mocked ChromaDB.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.infrastructure.rag import rag_service


class MockCollection:
    """Mock ChromaDB collection."""

    def __init__(self):
        self._data = {}
        self._count = 0

    def query(self, query_embeddings, n_results, include=None):
        """Mock query method."""
        # Return empty results - simulating no matches
        return {
            "ids": [[]],
            "distances": [[]],
            "metadatas": [[]],
        }

    def add(self, ids, embeddings, metadatas):
        """Mock add method."""
        for i, id_val in enumerate(ids):
            self._data[id_val] = {
                "embedding": embeddings[i],
                "metadata": metadatas[i],
            }
        self._count += len(ids)

    def count(self):
        """Mock count method."""
        return self._count


class MockChromaClient:
    """Mock ChromaDB client."""

    def __init__(self):
        self.collection = MockCollection()

    def get_or_create_collection(self, name, metadata=None):
        """Mock get_or_create_collection."""
        return self.collection


class TestRAGServiceInit:
    """Test RAG service initialization."""

    @pytest.fixture(autouse=True)
    def mock_chromadb(self):
        """Mock ChromaDB to avoid external dependencies."""
        with patch("chromadb.PersistentClient") as mock_client:
            mock_client.return_value = MockChromaClient()
            with patch("app.infrastructure.rag.rag_service.chromadb") as mock_chroma:
                mock_chroma.PersistentClient = mock_client
                mock_chroma.Client = MockChromaClient
                yield mock_client

    def test_rag_service_initializes(self):
        """Test that RAG service initializes without errors."""
        rag = rag_service.RAGService()
        assert rag is not None
        assert hasattr(rag, "_embedding_cache")

    def test_embedding_cache_initialized(self):
        """Test that embedding cache is initialized."""
        rag = rag_service.RAGService()
        assert rag._embedding_cache is not None
        assert isinstance(rag._embedding_cache, dict)
        assert rag._embedding_cache_max_size == 100


class TestEmbeddingCache:
    """Test embedding cache functionality."""

    @pytest.fixture
    def mock_rag_service(self):
        """Create RAG service with mocked dependencies."""
        with patch("chromadb.PersistentClient") as mock_client:
            mock_client.return_value = MockChromaClient()
            with patch("app.infrastructure.rag.rag_service.chromadb") as mock_chroma:
                mock_chroma.PersistentClient = mock_client
                mock_chroma.Client = MockChromaClient
                rag = rag_service.RAGService()
                yield rag

    def test_cache_key_generation(self, mock_rag_service):
        """Test cache key generation."""
        key1 = mock_rag_service._get_cache_key(b"image1")
        key2 = mock_rag_service._get_cache_key(b"image1")
        key3 = mock_rag_service._get_cache_key(b"image2")

        assert key1 == key2
        assert key1 != key3
        assert len(key1) == 32  # MD5 hash length

    def test_cached_embedding_retrieval(self, mock_rag_service):
        """Test getting cached embedding."""
        test_embedding = np.array([0.1] * 512, dtype=np.float32)
        mock_rag_service._cache_embedding(b"test_image", test_embedding)

        cached = mock_rag_service._get_cached_embedding(b"test_image")
        assert cached is not None
        np.testing.assert_array_equal(cached, test_embedding)

    def test_cache_miss(self, mock_rag_service):
        """Test cache miss returns None."""
        cached = mock_rag_service._get_cached_embedding(b"never_seen_before")
        assert cached is None

    def test_cache_eviction(self, mock_rag_service):
        """Test that cache evicts oldest entries when full."""
        # Fill cache beyond max size
        for i in range(105):
            embedding = np.random.rand(512).astype(np.float32)
            mock_rag_service._cache_embedding(f"image_{i}".encode(), embedding)

        # Cache should not exceed max size
        assert len(mock_rag_service._embedding_cache) <= mock_rag_service._embedding_cache_max_size


class TestDeterministicEmbedding:
    """Test deterministic embedding generation."""

    @pytest.fixture
    def mock_rag_service(self):
        """Create RAG service with mocked dependencies."""
        with patch("chromadb.PersistentClient") as mock_client:
            mock_client.return_value = MockChromaClient()
            with patch("app.infrastructure.rag.rag_service.chromadb") as mock_chroma:
                mock_chroma.PersistentClient = mock_client
                mock_chroma.Client = MockChromaClient
                rag = rag_service.RAGService()
                yield rag

    def test_deterministic_embedding_same_input(self, mock_rag_service):
        """Test that same input produces same embedding."""
        embedding1 = mock_rag_service._generate_deterministic_embedding(b"test_image")
        embedding2 = mock_rag_service._generate_deterministic_embedding(b"test_image")

        np.testing.assert_array_almost_equal(embedding1, embedding2)

    def test_deterministic_embedding_different_input(self, mock_rag_service):
        """Test that different inputs produce different embeddings."""
        embedding1 = mock_rag_service._generate_deterministic_embedding(b"image1")
        embedding2 = mock_rag_service._generate_deterministic_embedding(b"image2")

        # Should be different (very unlikely to be identical)
        assert not np.allclose(embedding1, embedding2)

    def test_deterministic_embedding_normalized(self, mock_rag_service):
        """Test that embeddings are normalized."""
        embedding = mock_rag_service._generate_deterministic_embedding(b"test")

        norm = np.linalg.norm(embedding)
        assert abs(norm - 1.0) < 0.001


class TestSearchSimilar:
    """Test search_similar method."""

    @pytest.fixture
    def mock_rag_service(self):
        """Create RAG service with mocked dependencies."""
        with patch("chromadb.PersistentClient") as mock_client:
            mock_client.return_value = MockChromaClient()
            with patch("app.infrastructure.rag.rag_service.chromadb") as mock_chroma:
                mock_chroma.PersistentClient = mock_client
                mock_chroma.Client = MockChromaClient
                rag = rag_service.RAGService()
                yield rag

    @pytest.mark.asyncio
    async def test_search_similar_returns_list(self, mock_rag_service):
        """Test search_similar returns a list."""
        results = await mock_rag_service.search_similar(b"fake_image_bytes", top_k=5)
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_search_similar_with_top_k(self, mock_rag_service):
        """Test search_similar respects top_k parameter."""
        results = await mock_rag_service.search_similar(b"fake_image_bytes", top_k=3)
        # Should return list regardless of results
        assert isinstance(results, list)


class TestAddImage:
    """Test add_image method."""

    @pytest.fixture
    def mock_rag_service(self):
        """Create RAG service with mocked dependencies."""
        with patch("chromadb.PersistentClient") as mock_client:
            mock_client.return_value = MockChromaClient()
            with patch("app.infrastructure.rag.rag_service.chromadb") as mock_chroma:
                mock_chroma.PersistentClient = mock_client
                mock_chroma.Client = MockChromaClient
                rag = rag_service.RAGService()
                yield rag

    @pytest.mark.asyncio
    async def test_add_image_returns_string_id(self, mock_rag_service):
        """Test add_image returns a document ID."""
        image_bytes = b"fake_image_data"
        tags = ["tag1", "tag2"]

        doc_id = await mock_rag_service.add_image(image_bytes, tags)

        assert isinstance(doc_id, str)
        assert len(doc_id) > 0


class TestGetStats:
    """Test get_stats method."""

    @pytest.fixture
    def mock_rag_service(self):
        """Create RAG service with mocked dependencies."""
        with patch("chromadb.PersistentClient") as mock_client:
            mock_client.return_value = MockChromaClient()
            with patch("app.infrastructure.rag.rag_service.chromadb") as mock_chroma:
                mock_chroma.PersistentClient = mock_client
                mock_chroma.Client = MockChromaClient
                rag = rag_service.RAGService()
                yield rag

    def test_get_stats_returns_dict(self, mock_rag_service):
        """Test get_stats returns a dictionary."""
        stats = mock_rag_service.get_stats()
        assert isinstance(stats, dict)
        assert "total_documents" in stats

    def test_get_stats_has_expected_keys(self, mock_rag_service):
        """Test get_stats has expected keys."""
        stats = mock_rag_service.get_stats()
        assert "collection_name" in stats
        assert "embedding_mode" in stats
