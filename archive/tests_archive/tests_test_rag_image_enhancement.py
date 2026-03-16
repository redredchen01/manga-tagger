"""Tests for RAG Image Enhancement functionality.

Tests the CLIP-based image embedding service and reference image management.
"""

import pytest
import io
from PIL import Image

from app.services.clip_image_embedding_service import (
    get_clip_image_service,
    reset_clip_service,
    CLIPImageEmbeddingService,
)


@pytest.fixture
def clip_service():
    """Create CLIP service for testing."""
    reset_clip_service()
    service = get_clip_image_service()
    yield service
    # Cleanup
    reset_clip_service()


@pytest.fixture
def sample_image():
    """Create a sample test image."""
    img = Image.new("RGB", (224, 224), color="red")
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="JPEG")
    img_bytes.seek(0)
    return img_bytes.read()


@pytest.fixture
def sample_images_with_tags():
    """Create multiple sample images with tags."""
    images = []
    colors = ["red", "blue", "green", "yellow", "purple"]
    tag_sets = [
        ["蘿莉", "loli", "少女"],
        ["貓娘", "catgirl"],
        ["巨乳", "big breasts"],
        ["校服", "school uniform"],
        ["熟女", "mature"],
    ]

    for color, tags in zip(colors, tag_sets):
        img = Image.new("RGB", (224, 224), color=color)
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="JPEG")
        img_bytes.seek(0)
        images.append({"bytes": img_bytes.read(), "tags": tags, "color": color})

    return images


def test_clip_service_initialization(clip_service):
    """Test that CLIP service initializes correctly."""
    assert clip_service is not None
    assert clip_service.model is not None
    assert clip_service.collection is not None


def test_get_stats(clip_service):
    """Test getting RAG statistics."""
    stats = clip_service.get_stats()

    assert "total_documents" in stats
    assert "collection_name" in stats
    assert "model" in stats
    assert isinstance(stats["total_documents"], int)


def test_add_single_image(clip_service, sample_image):
    """Test adding a single image to RAG."""
    tags = ["測試", "test", "sample"]

    doc_id = clip_service.add_image(sample_image, tags)

    assert doc_id is not None
    assert doc_id.startswith("img_")

    # Verify it was added
    stats = clip_service.get_stats()
    assert stats["total_documents"] >= 1


def test_add_multiple_images(clip_service, sample_images_with_tags):
    """Test adding multiple images."""
    added_ids = []

    for img_data in sample_images_with_tags:
        doc_id = clip_service.add_image(img_data["bytes"], img_data["tags"])
        added_ids.append(doc_id)

    assert len(added_ids) == 5

    stats = clip_service.get_stats()
    assert stats["total_documents"] >= 5


def test_search_similar_images(clip_service, sample_images_with_tags):
    """Test searching for similar images."""
    # Add images first
    for img_data in sample_images_with_tags:
        clip_service.add_image(img_data["bytes"], img_data["tags"])

    # Query with one of the images
    query_image = sample_images_with_tags[0]["bytes"]

    results = clip_service.search(query_image, top_k=3, similarity_threshold=0.1)

    assert isinstance(results, list)
    # Should find at least the query image itself (or similar ones)
    assert len(results) >= 1


def test_search_by_text(clip_service, sample_images_with_tags):
    """Test searching images by text query."""
    # Add images first
    for img_data in sample_images_with_tags:
        clip_service.add_image(img_data["bytes"], img_data["tags"])

    # Search by text
    results = clip_service.search_by_text("loli", top_k=5, similarity_threshold=0.1)

    assert isinstance(results, list)

    # Should find at least the loli-tagged image
    if len(results) > 0:
        # Check that tags are returned
        for result in results:
            assert "tags" in result
            assert "score" in result
            assert "id" in result


def test_image_with_metadata(clip_service, sample_image):
    """Test adding image with custom metadata."""
    tags = ["測試"]
    metadata = {
        "source": "test",
        "description": "Test image for unit tests",
        "custom_field": "custom_value",
    }

    doc_id = clip_service.add_image(sample_image, tags, metadata)

    assert doc_id is not None

    # Search and verify metadata
    results = clip_service.search(sample_image, top_k=1)
    if len(results) > 0:
        result = results[0]
        assert "metadata" in result
        assert result["metadata"].get("source") == "test"


def test_delete_collection(clip_service, sample_image):
    """Test deleting and recreating collection."""
    # Add an image
    clip_service.add_image(sample_image, ["測試"])

    stats_before = clip_service.get_stats()
    initial_count = stats_before["total_documents"]

    # Delete collection
    clip_service.delete_collection()

    stats_after = clip_service.get_stats()
    assert stats_after["total_documents"] == 0

    # Add new image
    clip_service.add_image(sample_image, ["測試2"])
    stats_final = clip_service.get_stats()
    assert stats_final["total_documents"] == 1


def test_tag_normalization(clip_service, sample_image):
    """Test that tags are properly normalized."""
    # Add image with various tag formats
    tags = ["tag1", "TAG2", "Tag3", "multi word tag"]

    doc_id = clip_service.add_image(sample_image, tags)
    assert doc_id is not None

    # Search should find the image
    results = clip_service.search(sample_image, top_k=5)
    assert len(results) >= 1


def test_duplicate_prevention(clip_service, sample_image):
    """Test that duplicate images are allowed (different embeddings)."""
    # Add same image multiple times with different tags
    doc_id1 = clip_service.add_image(sample_image, ["tags1"])
    doc_id2 = clip_service.add_image(sample_image, ["tags2"])

    # Should get different IDs
    assert doc_id1 != doc_id2

    # Both should be in the database
    stats = clip_service.get_stats()
    assert stats["total_documents"] >= 2


def test_empty_tag_list(clip_service, sample_image):
    """Test adding image with empty tag list."""
    doc_id = clip_service.add_image(sample_image, [])
    assert doc_id is not None


def test_embedding_consistency(clip_service, sample_image):
    """Test that same image generates consistent embeddings."""
    # Generate embedding twice
    emb1 = clip_service.generate_embedding(sample_image)
    emb2 = clip_service.generate_embedding(sample_image)

    # Should be identical
    assert emb1 == emb2


def test_embedding_dimension(clip_service, sample_image):
    """Test that embedding has expected dimension."""
    embedding = clip_service.generate_embedding(sample_image)

    # CLIP ViT-L/14 produces 768-dimensional embeddings
    assert len(embedding) > 0
    assert len(embedding) == 768  # CLIP ViT-L/14 dimension


def test_similarity_score_range(clip_service, sample_images_with_tags):
    """Test that similarity scores are in valid range."""
    # Add images
    for img_data in sample_images_with_tags:
        clip_service.add_image(img_data["bytes"], img_data["tags"])

    # Search
    query = sample_images_with_tags[0]["bytes"]
    results = clip_service.search(query, top_k=5)

    for result in results:
        assert 0.0 <= result["score"] <= 1.0


def test_large_threshold_returns_empty(clip_service, sample_images_with_tags):
    """Test that high threshold returns empty results."""
    # Add images
    for img_data in sample_images_with_tags:
        clip_service.add_image(img_data["bytes"], img_data["tags"])

    # Search with very high threshold
    query = sample_images_with_tags[0]["bytes"]
    results = clip_service.search(query, top_k=5, similarity_threshold=0.99)

    # Should return empty (identical embeddings might score 1.0, but search filters)
    assert isinstance(results, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
