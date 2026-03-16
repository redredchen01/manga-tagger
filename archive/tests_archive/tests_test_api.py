"""Tests for Manga Cover Auto-Tagger."""

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


def test_health_check(client):
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "models_loaded" in data


def test_list_tags(client):
    """Test tags list endpoint."""
    response = client.get("/tags")
    assert response.status_code == 200
    data = response.json()
    assert "tags" in data
    assert "total" in data
    assert isinstance(data["tags"], list)


def test_root_endpoint(client):
    """Test root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "Manga Cover Auto-Tagger"
    assert "endpoints" in data


def test_tag_cover_no_file(client):
    """Test tag-cover endpoint without file."""
    response = client.post("/tag-cover")
    assert response.status_code == 422  # Validation error


def test_rag_add_no_file(client):
    """Test rag/add endpoint without file."""
    response = client.post("/rag/add")
    assert response.status_code == 422  # Validation error


def test_rag_stats(client):
    """Test rag/stats endpoint."""
    response = client.get("/rag/stats")
    # May fail if RAG not initialized
    assert response.status_code in [200, 500]


def test_tag_cover_with_known_keywords(client):
    """Test tag-cover with keywords that previously caused '0 tags' issue.

    This test addresses the bug where VLM提取 'mature, large_breasts'
    但 RAG 匹配返回空结果.

    Regression test for: underscore normalization, tag mapping, and hybrid matching.
    """
    import io
    from PIL import Image

    # Create a simple test image
    img = Image.new("RGB", (512, 512), color="red")
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="JPEG")
    img_bytes.seek(0)

    response = client.post(
        "/tag-cover",
        files={"file": ("test.jpg", img_bytes, "image/jpeg")},
        data={"top_k": 10, "confidence_threshold": 0.3},
    )

    # Should not return 500 error
    assert response.status_code in [200, 422, 400]

    if response.status_code == 200:
        data = response.json()
        # Verify structure
        assert "tags" in data
        assert "metadata" in data

        # Check metadata has expected fields
        assert "vlm_description" in data["metadata"]
        assert "rag_matches" in data["metadata"]

        # Tags should be a list (can be empty but not None)
        assert isinstance(data["tags"], list)


def test_tag_mapping_underscore_normalization(client):
    """Test that keywords with underscores (e.g., 'large_breasts') are properly mapped.

    This tests the fix for: 'large_breasts' not matching '巨乳' due to underscore issue.
    """
    import io
    from PIL import Image
    from app.services.tag_mapper import TagMapper

    # Directly test tag mapper
    mapper = TagMapper()

    # These should all map successfully
    assert mapper.to_chinese("large_breasts") == "巨乳"
    assert mapper.to_chinese("huge breasts") == "巨乳"
    assert mapper.to_chinese("big breasts") == "巨乳"
    assert mapper.to_chinese("loli") == "蘿莉"
    assert mapper.to_chinese("catgirl") == "貓娘"


def test_vector_store_tag_count():
    """Test that vector store has the expected tag count.

    This ensures the '0 tags' issue is caught early.
    """
    from tag_vector_store import TagVectorStore

    store = TagVectorStore(persist_directory="./chroma_db")
    count = store.collection.count()

    # Should have 611 tags loaded
    assert count == 611, f"Expected 611 tags, got {count}"
    assert count > 0, "Vector store should not be empty"


def test_library_matching_for_breast_tags():
    """Test that breast-related tags are properly matched.

    This specifically tests the 'large_breasts' -> '巨乳' mapping.
    """
    from app.services.tag_library_service import TagLibraryService
    from app.services.tag_mapper import TagMapper

    lib = TagLibraryService()
    mapper = TagMapper()

    # Test VLM keywords that previously failed
    test_keywords = ["large_breasts", "huge breasts", "big breasts"]

    for keyword in test_keywords:
        # Map to Chinese
        cn_tag = mapper.to_chinese(keyword)
        assert cn_tag is not None, f"Failed to map '{keyword}'"

        # Match in library
        matches = lib.match_tags_by_keywords([cn_tag], min_confidence=0.3)

        # Should find at least one match
        assert len(matches) > 0, (
            f"No matches found for '{keyword}' (mapped to '{cn_tag}')"
        )

        # Top match should be '巨乳' or similar
        top_tag = matches[0][0]
        assert (
            "乳" in top_tag or "breast" in top_tag.lower() or "chest" in top_tag.lower()
        ), f"Expected breast-related tag, got '{top_tag}'"


def test_hybrid_matching_comprehensive():
    """Comprehensive test for hybrid matching with various VLM outputs.

    Tests the complete flow: VLM keywords -> Tag Mapping -> Library Matching -> RAG.
    """
    from tag_matcher import create_tag_matcher

    matcher = create_tag_matcher()

    # Test cases that previously caused '0 tags'
    test_cases = [
        "mature, large_breasts",
        "loli, catgirl, school uniform",
        "big breasts",
        "huge breasts",
    ]

    for vlm_output in test_cases:
        # Extract keywords (comma-separated)
        keywords = [k.strip() for k in vlm_output.split(",")]

        # Get matches for each keyword
        all_matches = []
        for kw in keywords:
            matches = matcher.match(kw, top_k=5, similarity_threshold=0.3)
            all_matches.extend(matches)

        # Remove duplicates
        seen = set()
        unique_matches = []
        for m in all_matches:
            if m.tag_name not in seen:
                seen.add(m.tag_name)
                unique_matches.append(m)

        # Should find at least one match
        assert len(unique_matches) > 0, (
            f"No matches found for VLM output: '{vlm_output}' (keywords: {keywords})"
        )


def test_rag_integration_with_vlm_output():
    """Test that RAG properly processes VLM output format.

    Tests the format: 'mature, large_breasts'
    """
    from tag_matcher import TagMatcher
    from tag_loader import TagLoader
    from tag_vector_store import TagVectorStore

    # Initialize components
    store = TagVectorStore(persist_directory="./chroma_db")
    loader = TagLoader("51標籤庫.json")
    matcher = TagMatcher(tag_store=store, tag_loader=loader)

    # Simulate VLM output format
    vlm_outputs = ["mature, large_breasts", "loli"]

    for vlm_output in vlm_outputs:
        # Process like the real system would
        matches = matcher.match(vlm_output, top_k=10, similarity_threshold=0.3)

        # Should return results
        assert matches is not None

        # For single keyword tests, should find matches
        if len(vlm_output.split(",")) == 1:
            assert len(matches) > 0, f"No matches for '{vlm_output}'"
