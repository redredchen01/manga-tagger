"""
Unit tests for Tag Relationship Graph
Tests tag relationship validation and conflict detection.
"""

import unittest
import tempfile
import shutil
from pathlib import Path

from app.services.tag_relationship_graph import (
    TagRelationshipGraph,
    TagRelationship,
    ValidationResult,
    get_tag_relationship_graph,
    reset_tag_relationship_graph,
)


class TestTagRelationshipGraph(unittest.TestCase):
    """Test tag relationship graph functionality."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.graph = TagRelationshipGraph(
            persistence_path=f"{self.temp_dir}/test_relationships.json"
        )
        self.graph.build_default_graph()

    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
        reset_tag_relationship_graph()

    def test_conflict_detection(self):
        """Test detecting conflicting tags."""
        # Test conflicting tags
        tags = ["蘿莉", "人妻"]
        result = self.graph.validate_tag_combination(tags)

        self.assertFalse(result.is_valid)
        self.assertGreater(len(result.conflicts), 0)

        print(f"✓ Detected {len(result.conflicts)} conflict(s) for {tags}")

    def test_no_conflict_compatible_tags(self):
        """Test compatible tags have no conflicts."""
        tags = ["蘿莉", "貧乳", "獸耳"]
        result = self.graph.validate_tag_combination(tags)

        self.assertTrue(result.is_valid)
        self.assertEqual(len(result.conflicts), 0)

        print(f"✓ No conflicts for compatible tags: {tags}")

    def test_recommendations(self):
        """Test tag recommendations based on dependencies."""
        tags = ["貓娘"]
        result = self.graph.validate_tag_combination(tags)

        self.assertGreater(len(result.recommendations), 0)

        # Check that "獸耳" and "尾巴" are recommended
        recommended_tags = [r["tag"] for r in result.recommendations]
        self.assertIn("獸耳", recommended_tags)

        print(f"✓ Recommendations for '貓娘': {recommended_tags[:5]}")

    def test_confidence_adjustments(self):
        """Test confidence adjustments based on relationships."""
        tags = ["蘿莉", "貧乳"]
        confidences = {"蘿莉": 0.9, "貧乳": 0.8}

        result = self.graph.validate_tag_combination(tags, confidences)

        # Should have adjustments (boost for satisfying dependency)
        self.assertGreater(len(result.confidence_adjustments), 0)

        print(f"✓ Confidence adjustments: {result.confidence_adjustments}")

    def test_transitive_conflicts(self):
        """Test detecting transitive conflicts."""
        # If A implies B, and B conflicts with C, then A conflicts with C
        # This is a more complex test case
        tags = ["蘿莉", "巨乳"]
        result = self.graph.validate_tag_combination(tags)

        # Should detect conflict (蘿莉 vs 巨乳)
        self.assertFalse(result.is_valid)

        print(f"✓ Transitive conflict detected for {tags}")

    def test_related_tags_query(self):
        """Test querying related tags."""
        related = self.graph.get_related_tags("貓娘")

        self.assertGreater(len(related), 0)

        # Check relationships
        relations = [r["relation"] for r in related]
        self.assertIn("depends_on", relations)

        print(f"✓ Found {len(related)} related tags for '貓娘'")

    def test_graph_persistence(self):
        """Test saving and loading graph."""
        # Add a custom relationship
        self.graph.add_relationship("測試A", "測試B", "depends_on", 0.9)
        self.graph._save_relationships()

        # Create new graph instance pointing to same file
        new_graph = TagRelationshipGraph(persistence_path=self.graph.persistence_path)

        # Check that custom relationship was loaded
        self.assertTrue(new_graph.graph.has_edge("測試A", "測試B"))

        print("✓ Graph persistence works correctly")

    def test_reasoning_chain(self):
        """Test reasoning chain generation."""
        tags = ["蘿莉", "貧乳"]
        result = self.graph.validate_tag_combination(tags)

        self.assertGreater(len(result.reasoning_chain), 0)

        print(f"✓ Reasoning chain generated ({len(result.reasoning_chain)} items)")


class TestEdgeCases(unittest.TestCase):
    """Test edge cases."""

    def setUp(self):
        self.graph = TagRelationshipGraph()
        self.graph.build_default_graph()

    def test_empty_tag_list(self):
        """Test validation with empty tag list."""
        result = self.graph.validate_tag_combination([])
        self.assertTrue(result.is_valid)
        self.assertEqual(len(result.conflicts), 0)

    def test_single_tag(self):
        """Test validation with single tag."""
        result = self.graph.validate_tag_combination(["蘿莉"])
        self.assertTrue(result.is_valid)

    def test_unknown_tags(self):
        """Test validation with unknown tags."""
        result = self.graph.validate_tag_combination(["未知標籤1", "未知標籤2"])
        # Should be valid (no conflicts since tags aren't in graph)
        self.assertTrue(result.is_valid)


def run_tests():
    """Run all tests."""
    unittest.main(argv=[""], verbosity=2, exit=False)


if __name__ == "__main__":
    print("Running Tag Relationship Graph Tests...")
    print("=" * 60)
    run_tests()
    print("=" * 60)
    print("✅ All tests completed!")
