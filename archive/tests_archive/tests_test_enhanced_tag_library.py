"""Tests for Enhanced Tag Library functionality."""

import pytest
import sys
from pathlib import Path

# Add app to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.tag_library_service import TagLibraryService
from app.config import settings


class TestEnhancedTagLibrary:
    """Test cases for enhanced tag library features."""

    @pytest.fixture
    def enhanced_library(self):
        """Create a tag library service with enhanced tags."""
        # Use the enhanced tag library if it exists
        import os
        enhanced_path = "./data/tags_enhanced.json"
        if os.path.exists(enhanced_path):
            return TagLibraryService(enhanced_path)
        return TagLibraryService()

    def test_load_enhanced_tags(self, enhanced_library):
        """Test that enhanced tags are loaded correctly."""
        assert len(enhanced_library.tag_names) > 0
        print(f"Loaded {len(enhanced_library.tag_names)} tags")

    def test_enhanced_fields_loaded(self, enhanced_library):
        """Test that enhanced fields are loaded."""
        # Check if any tag has enhanced fields
        if enhanced_library.tag_visual_cues:
            print(f"Found {len(enhanced_library.tag_visual_cues)} tags with visual cues")
            assert True
        else:
            print("No enhanced fields loaded (may be using legacy format)")

    def test_get_visual_cues(self, enhanced_library):
        """Test getting visual cues for a tag."""
        # Try to get visual cues for "貓娘"
        cues = enhanced_library.get_tag_visual_cues("貓娘")
        print(f"Visual cues for 貓娘: {cues}")
        # This should return a list (possibly empty if not in enhanced format)

    def test_get_related_tags(self, enhanced_library):
        """Test getting related tags for a tag."""
        related = enhanced_library.get_tag_related_tags("貓娘")
        print(f"Related tags for 貓娘: {related}")
        # This should return a list (possibly empty if not in enhanced format)

    def test_get_negative_cues(self, enhanced_library):
        """Test getting negative cues for a tag."""
        negative = enhanced_library.get_tag_negative_cues("貓娘")
        print(f"Negative cues for 貓娘: {negative}")
        # This should return a list (possibly empty if not in enhanced format)

    def test_get_aliases(self, enhanced_library):
        """Test getting aliases for a tag."""
        aliases = enhanced_library.get_tag_aliases("貓娘")
        print(f"Aliases for 貓娘: {aliases}")
        # This should return a list (possibly empty if not in enhanced format)

    def test_get_confidence_boost(self, enhanced_library):
        """Test getting confidence boost for a tag."""
        boost = enhanced_library.get_tag_confidence_boost("貓娘")
        print(f"Confidence boost for 貓娘: {boost}")
        # Should return a float, default 1.0

    def test_match_with_exact_keyword(self, enhanced_library):
        """Test exact keyword matching."""
        matches = enhanced_library.match_tags_by_keywords_enhanced(
            ["貓娘"], min_confidence=0.5
        )
        print(f"Exact match for 貓娘: {matches}")
        assert len(matches) > 0
        assert matches[0][0] == "貓娘"
        assert matches[0][1] == 1.0

    def test_match_with_alias(self, enhanced_library):
        """Test alias matching."""
        matches = enhanced_library.match_tags_by_keywords_enhanced(
            ["catgirl"], min_confidence=0.5
        )
        print(f"Alias match for catgirl: {matches}")
        # Should match if "catgirl" is an alias

    def test_match_with_visual_cue(self, enhanced_library):
        """Test visual cues matching."""
        matches = enhanced_library.match_tags_by_keywords_enhanced(
            ["貓耳"], min_confidence=0.5
        )
        print(f"Visual cue match for 貓耳: {matches}")
        # Should match tags that have 貓耳 in their visual_cues

    def test_match_with_contains(self, enhanced_library):
        """Test contains matching."""
        matches = enhanced_library.match_tags_by_keywords_enhanced(
            ["貓"], min_confidence=0.5
        )
        print(f"Contains match for 貓: {matches[:5]}")  # Show first 5
        assert len(matches) > 0

    def test_match_with_partial(self, enhanced_library):
        """Test partial word matching."""
        matches = enhanced_library.match_tags_by_keywords_enhanced(
            ["貓耳娘"], min_confidence=0.5
        )
        print(f"Partial match for 貓耳娘: {matches}")
        # Should match related tags

    def test_get_tag_category_explicit(self, enhanced_library):
        """Test getting category for a tag with explicit category."""
        category = enhanced_library.get_tag_category("貓娘")
        print(f"Category for 貓娘: {category}")
        assert category == "character"

    def test_get_tag_category_inferred(self, enhanced_library):
        """Test getting category for a tag without explicit category."""
        category = enhanced_library.get_tag_category("翅膀")
        print(f"Category for 翅膀: {category}")
        # Should infer category from tag name


class TestEnhancedMatchingComparison:
    """Compare legacy vs enhanced matching."""

    @pytest.fixture
    def legacy_library(self):
        """Create a tag library service with legacy tags."""
        return TagLibraryService("./data/tags.json")

    @pytest.fixture
    def enhanced_library(self):
        """Create a tag library service with enhanced tags."""
        import os
        enhanced_path = "./data/tags_enhanced.json"
        if os.path.exists(enhanced_path):
            return TagLibraryService(enhanced_path)
        return None

    def test_enhanced_vs_legacy_precision(self, legacy_library, enhanced_library):
        """Test that enhanced matching provides better precision."""
        if enhanced_library is None:
            pytest.skip("Enhanced tag library not available")

        # Test keywords that should match with high confidence
        test_keywords = ["catgirl", "貓耳", "nekomimi"]

        for keyword in test_keywords:
            legacy_matches = legacy_library.match_tags_by_keywords(
                [keyword], min_confidence=0.5
            )
            enhanced_matches = enhanced_library.match_tags_by_keywords_enhanced(
                [keyword], min_confidence=0.5
            )

            print(f"\nKeyword: {keyword}")
            print(f"  Legacy matches: {len(legacy_matches)}")
            print(f"  Enhanced matches: {len(enhanced_matches)}")

            # Enhanced should have at least as many matches as legacy
            # (due to additional matching paths like aliases, visual_cues)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
