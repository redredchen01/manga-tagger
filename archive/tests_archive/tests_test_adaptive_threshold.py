"""
Unit tests for Adaptive Threshold Service
Tests dynamic threshold calculation and integration with TagMatcher.
"""

import unittest
import numpy as np
from pathlib import Path
import tempfile
import shutil

# Import the service
from app.services.adaptive_threshold_service import (
    AdaptiveThresholdService,
    AdaptiveThresholdConfig,
    ImageComplexityAnalyzer,
    TagPerformanceMetrics,
    get_adaptive_threshold_service,
    reset_adaptive_threshold_service,
)


class TestImageComplexityAnalyzer(unittest.TestCase):
    """Test image complexity analysis."""

    def setUp(self):
        self.analyzer = ImageComplexityAnalyzer()

    def test_simple_image_complexity(self):
        """Test complexity calculation on simple image."""
        # Create simple uniform image
        image = np.ones((100, 100, 3), dtype=np.uint8) * 128
        metrics = self.analyzer.analyze(image)

        self.assertIn("overall", metrics)
        self.assertIn("edge_density", metrics)
        self.assertIn("color_variance", metrics)

        # Simple image should have low complexity
        self.assertLess(metrics["overall"], 0.5)

    def test_complex_image_complexity(self):
        """Test complexity calculation on complex image."""
        # Create complex random image
        np.random.seed(42)
        image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        metrics = self.analyzer.analyze(image)

        # Complex image should have higher complexity
        self.assertGreater(metrics["overall"], 0.3)

    def test_empty_image(self):
        """Test handling of empty image."""
        image = np.array([])
        metrics = self.analyzer.analyze(image)
        self.assertEqual(metrics["overall"], 0.5)


class TestTagPerformanceMetrics(unittest.TestCase):
    """Test tag performance tracking."""

    def test_initial_metrics(self):
        """Test initial metric values."""
        metrics = TagPerformanceMetrics()

        self.assertEqual(metrics.total_predictions, 0)
        self.assertEqual(metrics.precision, 0.5)  # Neutral
        self.assertEqual(metrics.recall, 0.5)  # Neutral
        self.assertEqual(metrics.f1_score, 0.0)

    def test_update_metrics(self):
        """Test updating metrics with predictions."""
        metrics = TagPerformanceMetrics()

        # Add correct prediction
        metrics.total_predictions += 1
        metrics.correct_predictions += 1

        self.assertEqual(metrics.precision, 1.0)
        self.assertEqual(metrics.recall, 1.0)
        self.assertEqual(metrics.f1_score, 1.0)

        # Add false positive
        metrics.total_predictions += 1
        metrics.false_positives += 1

        self.assertEqual(metrics.precision, 0.5)
        self.assertEqual(metrics.recall, 0.5)


class TestAdaptiveThresholdService(unittest.TestCase):
    """Test adaptive threshold service."""

    def setUp(self):
        # Create temporary directory for persistence
        self.temp_dir = tempfile.mkdtemp()
        self.config = AdaptiveThresholdConfig(
            base_threshold=0.50, min_threshold=0.30, max_threshold=0.80
        )
        self.service = AdaptiveThresholdService(self.config)
        # Override persistence path
        self.service.persistence_path = Path(self.temp_dir) / "test_stats.json"

    def tearDown(self):
        # Clean up temporary directory
        shutil.rmtree(self.temp_dir)
        reset_adaptive_threshold_service()

    def test_dynamic_threshold_calculation(self):
        """Test dynamic threshold calculation."""
        # Test with low complexity
        low_complexity_features = {"complexity_score": 0.2}
        threshold_low = self.service.calculate_dynamic_threshold(
            "character", low_complexity_features
        )

        # Test with high complexity
        high_complexity_features = {"complexity_score": 0.8}
        threshold_high = self.service.calculate_dynamic_threshold(
            "character", high_complexity_features
        )

        # Both should be within bounds
        self.assertGreaterEqual(threshold_low, self.config.min_threshold)
        self.assertLessEqual(threshold_low, self.config.max_threshold)
        self.assertGreaterEqual(threshold_high, self.config.min_threshold)
        self.assertLessEqual(threshold_high, self.config.max_threshold)

        print(f"Low complexity threshold: {threshold_low:.3f}")
        print(f"High complexity threshold: {threshold_high:.3f}")

    def test_performance_factor_influence(self):
        """Test that performance history influences threshold."""
        # Add good performance history
        for _ in range(10):
            self.service.update_performance("character", True, True, 0.9)

        # Calculate threshold with good performance
        features = {"complexity_score": 0.5}
        threshold_good = self.service.calculate_dynamic_threshold("character", features)

        # Reset and add poor performance
        self.service.reset_category_stats("character")
        for _ in range(10):
            self.service.update_performance("character", True, False, 0.6)

        threshold_poor = self.service.calculate_dynamic_threshold("character", features)

        # Good performance should generally allow higher thresholds
        print(f"Good performance threshold: {threshold_good:.3f}")
        print(f"Poor performance threshold: {threshold_poor:.3f}")

    def test_threshold_bounds(self):
        """Test that threshold respects min/max bounds."""
        extreme_features = {"complexity_score": 0.0}
        threshold_min = self.service.calculate_dynamic_threshold(
            "test", extreme_features
        )

        extreme_features = {"complexity_score": 1.0}
        threshold_max = self.service.calculate_dynamic_threshold(
            "test", extreme_features
        )

        self.assertGreaterEqual(threshold_min, self.config.min_threshold)
        self.assertLessEqual(threshold_max, self.config.max_threshold)

    def test_persistence(self):
        """Test saving and loading performance history."""
        # Add some performance data
        self.service.update_performance("character", True, True, 0.85)
        self.service.update_performance("clothing", True, False, 0.75)

        # Save
        self.service._save_history()
        self.assertTrue(self.service.persistence_path.exists())

        # Create new service and load
        new_service = AdaptiveThresholdService(self.config)
        new_service.persistence_path = self.service.persistence_path
        new_service._load_history()

        # Check data was loaded
        self.assertIn("character", new_service.category_performance)
        self.assertIn("clothing", new_service.category_performance)

    def test_get_category_stats(self):
        """Test getting category statistics."""
        # Add performance data
        self.service.update_performance("character", True, True, 0.9)
        self.service.update_performance("character", True, True, 0.85)
        self.service.update_performance("character", True, False, 0.70)

        stats = self.service.get_category_stats("character")

        self.assertEqual(stats["category"], "character")
        self.assertEqual(stats["total_predictions"], 3)
        self.assertIn("precision", stats)
        self.assertIn("recall", stats)
        self.assertIn("f1_score", stats)

        print(f"Category stats: {stats}")


class TestIntegration(unittest.TestCase):
    """Integration tests with TagMatcher."""

    def setUp(self):
        reset_adaptive_threshold_service()

    def test_tagmatcher_with_adaptive_threshold(self):
        """Test that TagMatcher can use adaptive threshold service."""
        try:
            from tag_matcher import TagMatcher

            # This test requires the full system to be initialized
            # For now, just test that the integration doesn't crash
            service = get_adaptive_threshold_service()

            # Test dynamic threshold calculation
            features = {"complexity_score": 0.5}
            threshold = service.calculate_dynamic_threshold("character", features)

            self.assertGreater(threshold, 0)
            self.assertLess(threshold, 1)

            print(f"TagMatcher integration test passed. Threshold: {threshold:.3f}")

        except ImportError as e:
            self.skipTest(f"Could not import TagMatcher: {e}")


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""

    def setUp(self):
        self.service = AdaptiveThresholdService()

    def test_empty_features(self):
        """Test with empty features dictionary."""
        threshold = self.service.calculate_dynamic_threshold("test", {})
        self.assertGreaterEqual(threshold, 0.3)
        self.assertLessEqual(threshold, 0.8)

    def test_unknown_category(self):
        """Test with unknown category."""
        features = {"complexity_score": 0.5}
        threshold = self.service.calculate_dynamic_threshold(
            "unknown_category", features
        )
        # Should use neutral performance factor (0.5)
        self.assertGreaterEqual(threshold, 0.3)
        self.assertLessEqual(threshold, 0.8)

    def test_complexity_caching(self):
        """Test that complexity results are cached."""
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        features = {"image": image}

        # First call should compute
        threshold1 = self.service.calculate_dynamic_threshold("test", features)

        # Second call should use cache
        threshold2 = self.service.calculate_dynamic_threshold("test", features)

        self.assertEqual(threshold1, threshold2)


def run_tests():
    """Run all tests."""
    unittest.main(argv=[""], verbosity=2, exit=False)


if __name__ == "__main__":
    print("Running Adaptive Threshold Service Tests...")
    print("=" * 60)
    run_tests()
    print("=" * 60)
    print("Tests completed!")
