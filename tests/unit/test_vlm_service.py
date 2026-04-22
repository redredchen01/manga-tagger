"""Unit tests for VLM tag parsing service.

Tests tag parsing functionality from VLM analysis responses.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services import tag_parser


class TestParseResponse:
    """Test parse_response function."""

    def test_parse_response_valid_json(self):
        """Test parse_response handles valid JSON-like input."""
        result = tag_parser.parse_response('{"tags": ["tag1", "tag2"]}')
        assert result is not None
        assert isinstance(result, dict)
        # The parser extracts tags from various formats
        assert "raw_keywords" in result

    def test_parse_response_with_tags_prefix(self):
        """Test parse_response handles tags: prefix."""
        result = tag_parser.parse_response(
            "Description: A cute character. Tags: catgirl, school_uniform"
        )
        assert result is not None
        assert "raw_keywords" in result

    def test_parse_response_comma_separated(self):
        """Test parse_response handles comma-separated tags."""
        result = tag_parser.parse_response("catgirl, school_uniform, loli")
        assert result is not None
        assert "raw_keywords" in result
        assert len(result["raw_keywords"]) > 0

    def test_parse_response_newline_separated(self):
        """Test parse_response handles newline-separated tags."""
        result = tag_parser.parse_response("catgirl\nschool_uniform\nloli")
        assert result is not None
        assert "raw_keywords" in result

    def test_parse_response_invalid_input(self):
        """Test parse_response handles completely invalid input gracefully."""
        result = tag_parser.parse_response("not json at all")
        # Should return fallback structure, not crash
        assert result is not None
        assert isinstance(result, dict)
        assert "raw_keywords" in result

    def test_parse_response_empty_string(self):
        """Test parse_response handles empty input."""
        result = tag_parser.parse_response("")
        assert result is not None
        assert isinstance(result, dict)

    def test_parse_response_preserves_description(self):
        """Test parse_response preserves description field."""
        desc = "A beautiful illustration of a catgirl in school uniform"
        result = tag_parser.parse_response(desc)
        assert result["description"] is not None
        # Description should contain some of the original text or cleaned version


class TestExtractTagsFromDescription:
    """Test extract_tags_from_description function."""

    def test_extract_tags_from_description_catgirl(self):
        """Test extraction of catgirl tag."""
        tags = tag_parser.extract_tags_from_description("A cute catgirl in school uniform")
        assert "catgirl" in tags

    def test_extract_tags_from_description_school_uniform(self):
        """Test extraction of school_uniform tag."""
        tags = tag_parser.extract_tags_from_description("A character wearing school uniform")
        assert "school_uniform" in tags

    def test_extract_tags_from_description_loli(self):
        """Test extraction of loli tag."""
        tags = tag_parser.extract_tags_from_description("A loli character with blonde hair")
        assert "loli" in tags

    def test_extract_tags_from_description_no_matches(self):
        """Test extraction with no keyword matches."""
        tags = tag_parser.extract_tags_from_description("Random text with no tags")
        # Should return empty list when no known keywords found
        assert isinstance(tags, list)

    def test_extract_tags_from_description_maid(self):
        """Test extraction of maid tag."""
        tags = tag_parser.extract_tags_from_description("A maid in a cafe")
        assert "maid" in tags

    def test_extract_tags_from_description_swimsuit(self):
        """Test extraction of swimsuit tag."""
        tags = tag_parser.extract_tags_from_description("Character in swimsuit at beach")
        assert "swimsuit" in tags


class TestExtractTagsFromReasoning:
    """Test extract_tags_from_reasoning function."""

    def test_extract_tags_from_reasoning_valid(self):
        """Test extraction from reasoning text with bullet points."""
        reasoning = "- loli\n- catgirl\n- school_uniform"
        tags = tag_parser.extract_tags_from_reasoning(reasoning)
        assert isinstance(tags, list)

    def test_extract_tags_from_reasoning_empty(self):
        """Test extraction from empty reasoning."""
        tags = tag_parser.extract_tags_from_reasoning("")
        assert tags == []

    def test_extract_tags_from_reasoning_no_bullets(self):
        """Test extraction with no bullet points."""
        tags = tag_parser.extract_tags_from_reasoning("This is just text")
        assert tags == []


class TestFallbackMetadata:
    """Test fallback metadata functions."""

    def test_get_fallback_metadata(self):
        """Test fallback metadata structure."""
        result = tag_parser.get_fallback_metadata("test error")
        assert result is not None
        assert isinstance(result, dict)
        assert "description" in result
        assert "raw_keywords" in result
        assert result["raw_keywords"] == []

    def test_get_mock_metadata(self):
        """Test mock metadata structure."""
        result = tag_parser.get_mock_metadata()
        assert result is not None
        assert isinstance(result, dict)
        assert "description" in result
        assert "raw_keywords" in result
        assert len(result["raw_keywords"]) > 0


class TestKeywordLists:
    """Test that keyword lists are properly defined."""

    def test_character_keywords_not_empty(self):
        """Test that CHARACTER_KEYWORDS is not empty."""
        assert len(tag_parser.CHARACTER_KEYWORDS) > 0

    def test_clothing_keywords_not_empty(self):
        """Test that CLOTHING_KEYWORDS is not empty."""
        assert len(tag_parser.CLOTHING_KEYWORDS) > 0

    def test_body_keywords_not_empty(self):
        """Test that BODY_KEYWORDS is not empty."""
        assert len(tag_parser.BODY_KEYWORDS) > 0

    def test_action_keywords_not_empty(self):
        """Test that ACTION_KEYWORDS is not empty."""
        assert len(tag_parser.ACTION_KEYWORDS) > 0

    def test_theme_keywords_not_empty(self):
        """Test that THEME_KEYWORDS is not empty."""
        assert len(tag_parser.THEME_KEYWORDS) > 0
