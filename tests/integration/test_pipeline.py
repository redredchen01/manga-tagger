"""End-to-end integration tests for the tagging pipeline.

These tests verify the complete request/response cycle including:
- Full tagging pipeline: Image → VLM → RAG → Tags
- Parameter handling (top_k, confidence_threshold)
- RAG add and search functionality
- Error handling for invalid inputs

Uses mock services to avoid external dependencies (LM Studio, ChromaDB).
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.api import routes_v2
from app.main import app
from app.dependencies import get_vlm_service, get_rag_service, get_tag_recommender
from app.services.mock_services import MockRAGService, MockVLMService


# Create test client
client = TestClient(app)


def reset_route_singletons() -> None:
    """Reset route singleton services between tests."""
    routes_v2._vlm_service = None
    routes_v2._llm_service = None
    routes_v2._rag_service = None
    routes_v2._tag_library = None
    routes_v2._tag_recommender = None


def create_test_image(
    width: int = 256, height: int = 256, color: tuple = (255, 0, 0)
) -> tuple[bytes, str]:
    """Create a test image and return (bytes, filename)."""
    img = Image.new("RGB", (width, height), color)
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    buffer.seek(0)
    return buffer.getvalue(), "test_cover.jpg"


def create_test_image_file(
    width: int = 256, height: int = 256, color: tuple = (255, 0, 0)
) -> tuple[io.BytesIO, str]:
    """Create a test image file for upload."""
    img = Image.new("RGB", (width, height), color)
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    buffer.seek(0)
    return buffer, "test_cover.jpg"


@pytest.fixture(autouse=True)
def setup_mock_services():
    """Setup mock services for all tests."""
    # Reset singletons before each test
    reset_route_singletons()

    # Mock VLM service
    mock_vlm = MockVLMService()

    # Mock RAG service
    mock_rag = MockRAGService()

    # Create mock tag recommender
    mock_rec = AsyncMock()
    mock_rec.tag_library = type(
        "MockTagLibrary",
        (),
        {
            "tags": [
                {"tag_name": "catgirl"},
                {"tag_name": "school_uniform"},
                {"tag_name": "maid"},
                {"tag_name": "swimsuit"},
                {"tag_name": "glasses"},
            ],
            "tag_names": ["catgirl", "school_uniform", "maid", "swimsuit", "glasses"],
            "tag_categories": {
                "character": ["catgirl"],
                "clothing": ["school_uniform", "maid", "swimsuit"],
                "body": ["glasses"],
            },
            "match_tags_by_keywords": lambda *args, **kwargs: [],
            "get_all_tags": lambda: [],
            "suggest_related_tags": lambda *args, **kwargs: [],
            "get_tag_definitions": lambda *args, **kwargs: {},
            "search_tags": lambda *args, **kwargs: [],
            "get_tag_description": lambda *args, **kwargs: None,
            "validate_tags": lambda *args, **kwargs: [],
        },
    )()

    async def mock_recommend_tags(*args, top_k=5, confidence_threshold=0.5, **kwargs):
        from app.services.tag_recommender_service import TagRecommendation

        # Create mock recommendations with varying confidence levels
        recommendations = [
            TagRecommendation(
                tag="catgirl",
                confidence=0.95,
                source="vlm+rag",
                reason="Matched by visual features and supporting RAG results",
            ),
            TagRecommendation(
                tag="school_uniform",
                confidence=0.88,
                source="vlm",
                reason="Character clothing detected",
            ),
            TagRecommendation(
                tag="maid",
                confidence=0.75,
                source="library_match",
                reason="Matched from tag library",
            ),
            TagRecommendation(
                tag="glasses", confidence=0.65, source="rag", reason="RAG match found"
            ),
        ]
        # Filter by confidence threshold
        filtered = [r for r in recommendations if r.confidence >= confidence_threshold]
        return filtered[:top_k]

    mock_rec.recommend_tags = mock_recommend_tags

    # Inject mocks into app.state for DI-based endpoints
    app.state.vlm_service = mock_vlm
    app.state.rag_service = mock_rag
    app.state.tag_recommender = mock_rec
    app.state.tag_library = mock_rec.tag_library
    app.state.llm_service = AsyncMock()

    # Also patch module-level functions for backward-compat endpoints
    with (
        patch.object(routes_v2, "get_vlm_service", return_value=mock_vlm),
        patch.object(routes_v2, "get_rag_service", return_value=mock_rag),
        patch.object(routes_v2, "get_tag_recommender") as mock_rc,
        patch("app.config.settings.USE_MOCK_SERVICES", True),
        patch("app.config.settings.USE_LM_STUDIO", False),
    ):
        mock_rc.return_value = mock_rec
        yield {"vlm_service": mock_vlm, "rag_service": mock_rag, "tag_recommender": mock_rec}

    # Cleanup after test
    reset_route_singletons()


class TestFullTaggingPipeline:
    """Tests for the full tagging pipeline."""

    @pytest.mark.asyncio
    async def test_full_tagging_pipeline_mock_mode(self, setup_mock_services):
        """Test complete pipeline: Image → VLM → RAG → Tags."""
        # Create test image
        image_bytes, filename = create_test_image()

        # Call the tag-cover endpoint
        response = client.post(
            "/api/v1/tag-cover",
            files={"file": (filename, image_bytes, "image/jpeg")},
            data={"top_k": "5", "confidence_threshold": "0.5", "include_metadata": "true"},
        )

        # Verify response
        assert response.status_code == 200
        data = response.json()

        # Verify response structure
        assert "tags" in data
        assert isinstance(data["tags"], list)
        assert len(data["tags"]) > 0

        # Verify tag structure
        first_tag = data["tags"][0]
        assert "tag" in first_tag
        assert "confidence" in first_tag
        assert "source" in first_tag
        assert "reason" in first_tag

        # Verify confidence is a valid float between 0 and 1
        assert 0.0 <= first_tag["confidence"] <= 1.0

        # Verify metadata
        assert "metadata" in data
        metadata = data["metadata"]
        assert metadata is not None
        assert "processing_time" in metadata
        assert "rag_matches_count" in metadata
        assert "api_version" in metadata

    @pytest.mark.asyncio
    async def test_pipeline_with_different_top_k(self, setup_mock_services):
        """Test pipeline respects top_k parameter."""
        image_bytes, filename = create_test_image()

        # Test with top_k=3
        response = client.post(
            "/api/v1/tag-cover",
            files={"file": (filename, image_bytes, "image/jpeg")},
            data={"top_k": "3", "confidence_threshold": "0.5", "include_metadata": "true"},
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data["tags"]) <= 3

        # Test with top_k=10
        response = client.post(
            "/api/v1/tag-cover",
            files={"file": (filename, image_bytes, "image/jpeg")},
            data={"top_k": "10", "confidence_threshold": "0.5", "include_metadata": "true"},
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data["tags"]) <= 10

    @pytest.mark.asyncio
    async def test_pipeline_with_confidence_threshold(self, setup_mock_services):
        """Test pipeline filters by confidence threshold."""
        image_bytes, filename = create_test_image()

        # Test with threshold=0.5
        response = client.post(
            "/api/v1/tag-cover",
            files={"file": (filename, image_bytes, "image/jpeg")},
            data={"top_k": "10", "confidence_threshold": "0.5", "include_metadata": "true"},
        )

        assert response.status_code == 200
        data = response.json()
        for tag in data["tags"]:
            assert tag["confidence"] >= 0.5

        # Test with threshold=0.9
        response = client.post(
            "/api/v1/tag-cover",
            files={"file": (filename, image_bytes, "image/jpeg")},
            data={"top_k": "10", "confidence_threshold": "0.9", "include_metadata": "true"},
        )

        assert response.status_code == 200
        data = response.json()
        for tag in data["tags"]:
            assert tag["confidence"] >= 0.9


class TestRAGIntegration:
    """Tests for RAG add and search functionality."""

    @pytest.mark.asyncio
    async def test_rag_add_and_search(self, setup_mock_services):
        """Test adding image to RAG and searching."""
        # Create test image
        image_bytes, filename = create_test_image()

        # Add image to RAG
        test_tags = ["catgirl", "school_uniform", "maid"]
        response = client.post(
            "/api/v1/rag/add",
            files={"file": (filename, image_bytes, "image/jpeg")},
            data={"tags": json.dumps(test_tags), "metadata": json.dumps({"source": "test"})},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "id" in data
        assert "Successfully added image" in data["message"]

        # Now call tag-cover and verify RAG matches in metadata
        response = client.post(
            "/api/v1/tag-cover",
            files={"file": (filename, image_bytes, "image/jpeg")},
            data={"top_k": "5", "confidence_threshold": "0.5", "include_metadata": "true"},
        )

        assert response.status_code == 200
        data = response.json()

        # Verify RAG matches in metadata
        assert "metadata" in data
        assert "rag_matches_count" in data["metadata"]

    @pytest.mark.asyncio
    async def test_rag_add_invalid_tags(self, setup_mock_services):
        """Test RAG add rejects invalid tags."""
        image_bytes, filename = create_test_image()

        # Send invalid JSON for tags
        response = client.post(
            "/api/v1/rag/add",
            files={"file": (filename, image_bytes, "image/jpeg")},
            data={"tags": "not valid json", "metadata": "{}"},
        )

        assert response.status_code == 400
        assert "Invalid JSON" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_rag_add_invalid_metadata(self, setup_mock_services):
        """Test RAG add rejects invalid metadata."""
        image_bytes, filename = create_test_image()

        # Send invalid JSON for metadata
        response = client.post(
            "/api/v1/rag/add",
            files={"file": (filename, image_bytes, "image/jpeg")},
            data={"tags": '["tag1", "tag2"]', "metadata": "not valid json"},
        )

        assert response.status_code == 400
        assert "Invalid JSON" in response.json()["detail"]


class TestErrorHandling:
    """Tests for error handling."""

    @pytest.mark.asyncio
    async def test_pipeline_invalid_image(self, setup_mock_services):
        """Test pipeline rejects invalid images."""
        # Send non-image file
        response = client.post(
            "/api/v1/tag-cover",
            files={"file": ("bad.txt", b"not an image", "text/plain")},
            data={"top_k": "5", "confidence_threshold": "0.5", "include_metadata": "true"},
        )

        assert response.status_code == 400
        # Handle both error formats: {"detail": ...} and {"error": {"message": ...}}
        error_data = response.json()
        error_msg = error_data.get("detail") or error_data.get("error", {}).get("message", "")
        assert "Invalid file type" in error_msg

    @pytest.mark.asyncio
    async def test_pipeline_too_small_file(self, setup_mock_services):
        """Test pipeline rejects too small files."""
        # Send too small file (less than 1KB)
        response = client.post(
            "/api/v1/tag-cover",
            files={"file": ("tiny.jpg", b"123", "image/jpeg")},
            data={"top_k": "5", "confidence_threshold": "0.5", "include_metadata": "true"},
        )

        assert response.status_code == 400
        # Handle both error formats
        error_data = response.json()
        error_msg = error_data.get("detail") or error_data.get("error", {}).get("message", "")
        assert "File too small" in error_msg

    @pytest.mark.asyncio
    async def test_pipeline_too_large_file(self, setup_mock_services):
        """Test pipeline rejects too large files."""
        # Create a large image (over 10MB)
        large_image_bytes = b"x" * (11 * 1024 * 1024)  # 11MB

        response = client.post(
            "/api/v1/tag-cover",
            files={"file": ("large.jpg", large_image_bytes, "image/jpeg")},
            data={"top_k": "5", "confidence_threshold": "0.5", "include_metadata": "true"},
        )

        assert response.status_code == 400
        # Handle both error formats
        error_data = response.json()
        error_msg = error_data.get("detail") or error_data.get("error", {}).get("message", "")
        assert "File too large" in error_msg

    @pytest.mark.asyncio
    async def test_pipeline_invalid_top_k(self, setup_mock_services):
        """Test pipeline rejects invalid top_k values."""
        image_bytes, filename = create_test_image()

        # Test with top_k=0 (below minimum)
        response = client.post(
            "/api/v1/tag-cover",
            files={"file": (filename, image_bytes, "image/jpeg")},
            data={"top_k": "0", "confidence_threshold": "0.5", "include_metadata": "true"},
        )

        assert response.status_code == 422  # FastAPI validation error

        # Test with top_k=25 (above maximum)
        response = client.post(
            "/api/v1/tag-cover",
            files={"file": (filename, image_bytes, "image/jpeg")},
            data={"top_k": "25", "confidence_threshold": "0.5", "include_metadata": "true"},
        )

        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_pipeline_invalid_confidence_threshold(self, setup_mock_services):
        """Test pipeline rejects invalid confidence_threshold values."""
        image_bytes, filename = create_test_image()

        # Test with threshold=-0.1 (below minimum)
        response = client.post(
            "/api/v1/tag-cover",
            files={"file": (filename, image_bytes, "image/jpeg")},
            data={"top_k": "5", "confidence_threshold": "-0.1", "include_metadata": "true"},
        )

        assert response.status_code == 422

        # Test with threshold=1.5 (above maximum)
        response = client.post(
            "/api/v1/tag-cover",
            files={"file": (filename, image_bytes, "image/jpeg")},
            data={"top_k": "5", "confidence_threshold": "1.5", "include_metadata": "true"},
        )

        assert response.status_code == 422


class TestResponseStructure:
    """Tests for response structure validation."""

    @pytest.mark.asyncio
    async def test_response_matches_pydantic_model(self, setup_mock_services):
        """Verify response structure matches Pydantic models."""
        image_bytes, filename = create_test_image()

        response = client.post(
            "/api/v1/tag-cover",
            files={"file": (filename, image_bytes, "image/jpeg")},
            data={"top_k": "5", "confidence_threshold": "0.5", "include_metadata": "true"},
        )

        assert response.status_code == 200
        data = response.json()

        # Verify TagCoverResponse structure
        assert "tags" in data
        assert "metadata" in data

        # Verify TagResult structure for each tag
        for tag in data["tags"]:
            assert "tag" in tag
            assert isinstance(tag["tag"], str)
            assert "confidence" in tag
            assert isinstance(tag["confidence"], (int, float))
            assert "source" in tag
            assert isinstance(tag["source"], str)
            assert "reason" in tag
            assert tag["reason"] is None or isinstance(tag["reason"], str)

    @pytest.mark.asyncio
    async def test_metadata_excluded_when_requested(self, setup_mock_services):
        """Test metadata can be excluded from response."""
        image_bytes, filename = create_test_image()

        response = client.post(
            "/api/v1/tag-cover",
            files={"file": (filename, image_bytes, "image/jpeg")},
            data={"top_k": "5", "confidence_threshold": "0.5", "include_metadata": "false"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["metadata"] is None


class TestEndpointCompatibility:
    """Tests for backward compatibility."""

    @pytest.mark.asyncio
    async def test_upload_alias_works(self, setup_mock_services):
        """Test /upload is an alias for /rag/add."""
        image_bytes, filename = create_test_image()

        # Use /upload endpoint
        response = client.post(
            "/api/v1/upload",
            files={"file": (filename, image_bytes, "image/jpeg")},
            data={"tags": '["test_tag"]', "metadata": "{}"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
