"""Unit tests for tag recommender service.

Tests tag recommendation functionality with mock VLM/RAG input.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services import tag_recommender_service


class TestTagRecommenderService:
    """Test TagRecommenderService functionality."""

    @pytest.fixture
    def mock_settings(self):
        """Mock settings to avoid external dependencies."""
        with patch("app.services.tag_recommender_service.settings") as mock:
            mock.USE_MOCK_SERVICES = True
            mock.USE_LM_STUDIO = False
            mock.EXACT_MATCH_BOOST = 1.1
            mock.PARTIAL_MATCH_BOOST = 1.0
            mock.SEMANTIC_MATCH_PENALTY = 0.95
            mock.CHINESE_EMBEDDING_THRESHOLD = 0.50
            mock.RAG_SUPPORT_BOOST = 1.05
            mock.RAG_SUPPORT_DECAY = 0.95
            mock.MIN_ACCEPTABLE_CONFIDENCE = 0.35
            mock.TAG_FREQUENCY_CALIBRATION = {}
            mock.SENSITIVE_TAGS = set()
            mock.SEMANTIC_SIBLINGS = {}
            yield mock

    @pytest.fixture
    def recommender(self, mock_settings):
        """Create a fresh recommender instance."""
        # Reset singleton
        tag_recommender_service._recommender_service = None
        recommender = tag_recommender_service.get_tag_recommender_service()
        return recommender

    @pytest.mark.asyncio
    async def test_recommend_tags_with_mock_input(self, recommender):
        """Test recommend_tags with mock VLM/RAG input."""
        vlm_analysis = {
            "description": "test image",
            "raw_keywords": ["tag1", "tag2"],
            "character_types": [],
            "clothing": [],
            "body_features": [],
            "actions": [],
            "themes": [],
        }
        rag_matches = []

        results = await recommender.recommend_tags(
            vlm_analysis=vlm_analysis,
            rag_matches=rag_matches,
            top_k=5,
            confidence_threshold=0.5,
        )

        assert isinstance(results, list)
        # Should return some results even with mock input
        # (may be empty if no library matches found)

    @pytest.mark.asyncio
    async def test_recommend_tags_empty_vlm_analysis(self, recommender):
        """Test recommend_tags handles empty VLM analysis."""
        vlm_analysis = {"description": "", "raw_keywords": []}
        rag_matches = []

        results = await recommender.recommend_tags(
            vlm_analysis=vlm_analysis,
            rag_matches=rag_matches,
            top_k=5,
            confidence_threshold=0.5,
        )

        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_recommend_tags_with_rag_matches(self, recommender):
        """Test recommend_tags with RAG matches."""
        vlm_analysis = {
            "description": "test",
            "raw_keywords": [],
            "character_types": [],
            "clothing": [],
            "body_features": [],
            "actions": [],
            "themes": [],
        }
        rag_matches = [
            {"id": "1", "score": 0.9, "tags": ["loli", "catgirl"]},
            {"id": "2", "score": 0.8, "tags": ["school_uniform"]},
        ]

        results = await recommender.recommend_tags(
            vlm_analysis=vlm_analysis,
            rag_matches=rag_matches,
            top_k=5,
            confidence_threshold=0.5,
        )

        assert isinstance(results, list)

    def test_is_vlm_analysis_valid(self, recommender):
        """Test _is_vlm_analysis_valid method."""
        # Valid analysis
        valid_analysis = {
            "raw_keywords": ["tag1"],
            "character_types": ["loli"],
        }
        assert recommender._is_vlm_analysis_valid(valid_analysis) is True

        # Invalid - empty
        assert recommender._is_vlm_analysis_valid({}) is False

        # Invalid - failed
        failed_analysis = {
            "description": "failed to analyze",
            "raw_keywords": [],
        }
        assert recommender._is_vlm_analysis_valid(failed_analysis) is False

    def test_extract_vlm_keywords(self, recommender):
        """Test _extract_vlm_keywords method."""
        vlm_analysis = {
            "character_types": ["loli", "catgirl"],
            "clothing": ["school_uniform"],
            "raw_keywords": ["extra_tag"],
        }
        keywords = recommender._extract_vlm_keywords(vlm_analysis)
        assert isinstance(keywords, list)
        assert len(keywords) > 0

    @pytest.mark.asyncio
    async def test_sensitive_tags_with_string_format(self):
        """Test _verify_and_calibrate with SENSITIVE_TAGS as production string format.

        This test catches the str-vs-set asymmetry bug that normal mocking hides.
        In production, SENSITIVE_TAGS is a comma-separated string; settings.sensitive_tags
        is a @computed_field set derived from it. Line 581 should use the set, not the string.
        """
        with patch("app.services.tag_recommender_service.settings") as mock:
            mock.USE_MOCK_SERVICES = True
            mock.USE_LM_STUDIO = False
            # Production format: SENSITIVE_TAGS as comma-separated string
            mock.SENSITIVE_TAGS = "蘿莉,正太,嬰兒,強制,強姦,亂倫,獵奇,肛交,觸手,綁縛"
            # Corresponding set (what @computed_field produces)
            mock.sensitive_tags = {"蘿莉", "正太", "嬰兒", "強制", "強姦", "亂倫", "獵奇", "肛交", "觸手", "綁縛"}
            mock.SENSITIVE_SUBSTRING_FILTER_ENABLED = True
            mock.EXACT_MATCH_BOOST = 1.1
            mock.PARTIAL_MATCH_BOOST = 1.0
            mock.SEMANTIC_MATCH_PENALTY = 0.95
            mock.MIN_ACCEPTABLE_CONFIDENCE = 0.35
            mock.TAG_FREQUENCY_CALIBRATION = {}
            mock.SEMANTIC_SIBLINGS = {}
            mock.EXACT_MATCH_PENALTY = {}
            mock.VISUAL_FEATURE_BOOST = {}
            mock.RAG_SUPPORT_BOOST = 1.0
            mock.RAG_SUPPORT_DECAY = 1.0
            mock.MUTUAL_EXCLUSIVITY = {}
            mock.TAG_HIERARCHY = {}

            # Reset singleton and create fresh recommender
            tag_recommender_service._recommender_service = None
            recommender = tag_recommender_service.get_tag_recommender_service()

            # Test case 1: exact match should be detected correctly
            rec_exact = tag_recommender_service.TagRecommendation(
                tag="蘿莉",
                confidence=0.9,
                source="vlm",
                reason="from VLM"
            )

            # Test case 2: substring but not exact should NOT be flagged as exact_sensitive
            rec_substring = tag_recommender_service.TagRecommendation(
                tag="蘿莉蘿莉",  # contains "蘿莉" but is not exact match
                confidence=0.85,
                source="vlm",
                reason="from VLM"
            )

            # Verify the logic with mock VLM (async)
            mock_vlm = MagicMock()
            mock_vlm.verify_sensitive_tag = AsyncMock(return_value=True)

            recommendations = [rec_exact, rec_substring]

            # Call _verify_and_calibrate (will be async)
            result = await recommender._verify_and_calibrate(
                recommendations=recommendations,
                vlm_service=mock_vlm,
                image_bytes=b"fake_image_bytes",
                rag_matches=[],
                vlm_analysis=None
            )

            # Verify:
            # - rec_exact should have "Verified" in reason (exact match, verified by VLM)
            # - rec_substring should have "substring-verified" if verification passed
            assert any(r.tag == "蘿莉" and "Verified" in r.reason for r in result), \
                "Exact-match sensitive tag should be verified"

            # Check substring handling - should be in result with verification marker
            substring_recs = [r for r in result if r.tag == "蘿莉蘿莉"]
            assert len(substring_recs) > 0, "Substring-sensitive tag should be present after verification"
            assert any("substring-verified" in r.reason for r in substring_recs), \
                "Substring-sensitive tag should have substring-verified marker"


class TestTagRecommendation:
    """Test TagRecommendation dataclass."""

    def test_tag_recommendation_creation(self):
        """Test creating a TagRecommendation."""
        rec = tag_recommender_service.TagRecommendation(
            tag="catgirl",
            confidence=0.95,
            source="vlm",
            reason="Found in character types",
        )
        assert rec.tag == "catgirl"
        assert rec.confidence == 0.95
        assert rec.source == "vlm"
        assert rec.reason == "Found in character types"


class TestSingletonPattern:
    """Test tag recommender singleton pattern."""

    def test_get_tag_recommender_service_singleton(self):
        """Test that get_tag_recommender_service returns singleton."""
        # Reset singleton
        tag_recommender_service._recommender_service = None

        r1 = tag_recommender_service.get_tag_recommender_service()
        r2 = tag_recommender_service.get_tag_recommender_service()

        assert r1 is r2

    def test_singleton_returns_correct_type(self):
        """Test that singleton returns correct type."""
        tag_recommender_service._recommender_service = None
        recommender = tag_recommender_service.get_tag_recommender_service()
        assert isinstance(recommender, tag_recommender_service.TagRecommenderService)
