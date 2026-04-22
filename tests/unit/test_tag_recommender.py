"""Unit tests for tag recommender service.

Tests tag recommendation functionality with mock VLM/RAG input.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

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
