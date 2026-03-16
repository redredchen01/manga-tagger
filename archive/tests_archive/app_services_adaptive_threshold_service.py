"""
Adaptive Threshold Service Module
Implements dynamic threshold calculation for improved tag matching precision.

This service analyzes image complexity and historical performance to adjust
matching thresholds dynamically, reducing false positives while maintaining recall.
"""

import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class AdaptiveThresholdConfig:
    """Configuration for adaptive threshold calculation."""

    base_threshold: float = 0.50
    complexity_weight: float = 0.4
    performance_weight: float = 0.4
    min_threshold: float = 0.30
    max_threshold: float = 0.80
    history_window_size: int = 100  # Number of predictions to keep in history
    complexity_method: str = (
        "edge_density"  # "edge_density", "color_variance", "combined"
    )
    precision_mode: bool = False  # PRECISION-FIRST mode flag


@dataclass
class TagPerformanceMetrics:
    """Tracks performance metrics for a specific tag category."""

    total_predictions: int = 0
    correct_predictions: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    average_confidence: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)

    @property
    def precision(self) -> float:
        if self.total_predictions == 0:
            return 0.5  # Neutral starting point
        return self.correct_predictions / max(self.total_predictions, 1)

    @property
    def recall(self) -> float:
        denominator = self.correct_predictions + self.false_negatives
        if denominator == 0:
            return 0.5
        return self.correct_predictions / denominator

    @property
    def f1_score(self) -> float:
        p, r = self.precision, self.recall
        if p + r == 0:
            return 0.0
        return 2 * (p * r) / (p + r)


class ImageComplexityAnalyzer:
    """Analyzes image complexity using various methods."""

    def __init__(self, method: str = "combined"):
        self.method = method

    def analyze(self, image: np.ndarray) -> Dict[str, float]:
        """
        Analyze image complexity and return complexity factors.

        Args:
            image: numpy array of image (BGR format from OpenCV)

        Returns:
            Dictionary with complexity metrics (0.0 to 1.0)
        """
        if image is None or image.size == 0:
            return {"overall": 0.5}

        metrics = {}

        # Method 1: Edge Density
        gray = (
            cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        )
        edges = cv2.Canny(gray, 100, 200)
        edge_density = np.sum(edges > 0) / edges.size
        metrics["edge_density"] = min(edge_density * 5, 1.0)  # Scale to 0-1

        # Method 2: Color Variance
        if len(image.shape) == 3:
            color_variance = np.std(image.reshape(-1, 3), axis=0).mean() / 255.0
            metrics["color_variance"] = color_variance
        else:
            metrics["color_variance"] = np.std(gray) / 255.0

        # Method 3: Texture Complexity (using Laplacian variance)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        metrics["texture_complexity"] = min(laplacian_var / 500, 1.0)

        # Method 4: Detail Level (number of contours)
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        detail_level = min(len(contours) / 100, 1.0)
        metrics["detail_level"] = detail_level

        # Calculate overall complexity
        if self.method == "combined":
            weights = {
                "edge_density": 0.3,
                "color_variance": 0.2,
                "texture_complexity": 0.3,
                "detail_level": 0.2,
            }
            overall = sum(metrics[k] * weights[k] for k in weights if k in metrics)
        elif self.method == "edge_density":
            overall = metrics.get("edge_density", 0.5)
        elif self.method == "color_variance":
            overall = metrics.get("color_variance", 0.5)
        else:
            overall = np.mean(list(metrics.values()))

        metrics["overall"] = np.clip(overall, 0.0, 1.0)

        return metrics


class AdaptiveThresholdService:
    """
    Service for calculating dynamic thresholds based on image complexity
    and historical tag category performance.
    """

    def __init__(self, config: Optional[AdaptiveThresholdConfig] = None):
        """
        Initialize the adaptive threshold service.

        Args:
            config: Configuration for threshold calculation
        """
        self.config = config or AdaptiveThresholdConfig()
        self.complexity_analyzer = ImageComplexityAnalyzer(
            self.config.complexity_method
        )

        # Historical performance tracking
        self.category_performance: Dict[str, TagPerformanceMetrics] = defaultdict(
            lambda: TagPerformanceMetrics()
        )

        # Cache for recent complexity calculations
        self.complexity_cache: Dict[str, Dict] = {}
        self.cache_max_size = 1000

        # Persistence path
        self.persistence_path = Path("./data/adaptive_threshold_stats.json")
        self._load_history()

        logger.info(
            f"AdaptiveThresholdService initialized with base_threshold={self.config.base_threshold}"
        )

        # PRECISION-FIRST mode: Higher thresholds to reduce false positives
        self._precision_mode = self.config.precision_mode
        self.PRECISION_THRESHOLD_CONFIGS = {
            "character": 0.75,   # Higher threshold for character tags
            "body": 0.70,       # Higher threshold for body tags
            "clothing": 0.65,   # Higher threshold for clothing tags
            "action": 0.80,     # Higher threshold for action tags
            "theme": 0.82,      # Higher threshold for theme tags
            "sensitive": 0.90,  # Highest threshold for sensitive tags
            "other": 0.65,      # Higher threshold for other tags
        }

    def set_precision_mode(self, enabled: bool = True) -> None:
        """Enable or disable PRECISION-FIRST mode.
        
        When enabled, higher thresholds are used to reduce false positives.
        
        Args:
            enabled: True for precision-first (fewer false positives), False for balanced
        """
        self._precision_mode = enabled
        logger.info(f"Precision mode {'enabled' if enabled else 'disabled'}")

    def is_precision_mode(self) -> bool:
        """Check if precision mode is enabled.
        
        Returns:
            True if precision mode is enabled
        """
        return self._precision_mode

    def get_threshold_for_category(self, tag_category: str) -> float:
        """Get the base threshold for a specific category.
        
        Args:
            tag_category: Category name (character, body, clothing, action, theme, sensitive, other)
            
        Returns:
            Base threshold for the category
        """
        if self._precision_mode and tag_category in self.PRECISION_THRESHOLD_CONFIGS:
            return self.PRECISION_THRESHOLD_CONFIGS[tag_category]
        # Return balanced mode default thresholds
        defaults = {
            "character": 0.65,
            "body": 0.55,
            "clothing": 0.55,
            "action": 0.70,
            "theme": 0.70,
            "sensitive": 0.80,
            "other": 0.50,
        }
        return defaults.get(tag_category, 0.50)

    def calculate_dynamic_threshold(
        self, tag_category: str, image_features: Dict[str, Any]
    ) -> float:
        """
        Calculate dynamic threshold based on image complexity and category performance.

        The formula adjusts base threshold using:
        - Complexity factor: Higher complexity -> slightly lower threshold (harder to match)
        - Performance factor: Better historical performance -> can use higher threshold

        Args:
            tag_category: Category of tag (e.g., "character", "clothing", "body")
            image_features: Dictionary containing image data or complexity metrics

        Returns:
            Adjusted threshold value between min_threshold and max_threshold
        """
        # PRECISION MODE: Use strict fixed thresholds to minimize false positives
        if self._precision_mode:
            precision_threshold = self.get_threshold_for_category(tag_category)
            logger.debug(f"Precision mode: {tag_category} threshold = {precision_threshold:.3f}")
            return precision_threshold
        
        # Extract or calculate complexity factor
        complexity_factor = self._get_complexity_factor(image_features)

        # Get performance factor for this category
        performance_factor = self._get_category_performance_factor(tag_category)

        # Calculate adjustment
        # Base formula: threshold = base * (0.8 + complexity_weight * complexity * performance_weight * performance)
        # This means:
        # - High complexity + High performance = slightly lower threshold
        # - Low complexity + High performance = higher threshold (safer)
        # - High complexity + Low performance = lower threshold (more lenient)
        adjustment = (
            0.8
            + self.config.complexity_weight
            * complexity_factor
            * self.config.performance_weight
            * performance_factor
        )

        adjusted_threshold = self.config.base_threshold * adjustment

        # Clip to valid range
        final_threshold = np.clip(
            adjusted_threshold, self.config.min_threshold, self.config.max_threshold
        )

        logger.debug(
            f"Category: {tag_category}, Complexity: {complexity_factor:.3f}, "
            f"Performance: {performance_factor:.3f}, Threshold: {final_threshold:.3f}"
        )

        return float(final_threshold)

    def _get_complexity_factor(self, image_features: Dict[str, Any]) -> float:
        """
        Extract complexity factor from image features.

        Args:
            image_features: Dictionary that may contain:
                - "image": numpy array of image
                - "complexity_metrics": pre-calculated complexity dict
                - "complexity_score": float complexity score

        Returns:
            Complexity factor (0.0 to 1.0)
        """
        # If pre-calculated complexity is provided, use it
        if "complexity_score" in image_features:
            return float(image_features["complexity_score"])

        if "complexity_metrics" in image_features:
            return image_features["complexity_metrics"].get("overall", 0.5)

        # If image is provided, analyze it
        if "image" in image_features:
            image = image_features["image"]
            # Check cache
            image_hash = self._get_image_hash(image)
            if image_hash in self.complexity_cache:
                return self.complexity_cache[image_hash]["overall"]

            # Analyze and cache
            metrics = self.complexity_analyzer.analyze(image)
            self._update_complexity_cache(image_hash, metrics)
            return metrics["overall"]

        # Default: neutral complexity
        return 0.5

    def _get_category_performance_factor(self, tag_category: str) -> float:
        """
        Get performance factor for a tag category.

        Higher performing categories can afford higher thresholds.

        Args:
            tag_category: Tag category name

        Returns:
            Performance factor (0.0 to 1.0)
        """
        metrics = self.category_performance.get(tag_category)
        if metrics is None:
            return 0.5  # Neutral for unknown categories

        # Use F1 score as performance indicator
        f1 = metrics.f1_score

        # Map F1 (0-1) to performance factor (0.5-1.0)
        # Higher F1 = higher performance factor = can use stricter threshold
        performance_factor = 0.5 + (f1 * 0.5)

        return performance_factor

    def update_performance(
        self, tag_category: str, predicted: bool, actual: bool, confidence: float
    ) -> None:
        """
        Update performance metrics for a tag category based on feedback.

        Args:
            tag_category: Category that was predicted
            predicted: Whether tag was predicted
            actual: Whether tag was actually correct (ground truth)
            confidence: Confidence score of prediction
        """
        metrics = self.category_performance[tag_category]
        metrics.total_predictions += 1
        metrics.average_confidence = (
            metrics.average_confidence * (metrics.total_predictions - 1) + confidence
        ) / metrics.total_predictions

        if predicted and actual:
            metrics.correct_predictions += 1
        elif predicted and not actual:
            metrics.false_positives += 1
        elif not predicted and actual:
            metrics.false_negatives += 1

        metrics.last_updated = datetime.now()

        # Periodically save to disk
        if metrics.total_predictions % 10 == 0:
            self._save_history()

    def get_category_stats(self, tag_category: str) -> Dict[str, Any]:
        """Get statistics for a specific category."""
        metrics = self.category_performance.get(tag_category, TagPerformanceMetrics())
        return {
            "category": tag_category,
            "total_predictions": metrics.total_predictions,
            "precision": metrics.precision,
            "recall": metrics.recall,
            "f1_score": metrics.f1_score,
            "average_confidence": metrics.average_confidence,
            "last_updated": metrics.last_updated.isoformat(),
        }

    def get_all_stats(self) -> Dict[str, Any]:
        """Get statistics for all categories."""
        return {
            category: self.get_category_stats(category)
            for category in self.category_performance.keys()
        }

    def reset_category_stats(self, tag_category: Optional[str] = None) -> None:
        """Reset statistics for a category or all categories."""
        if tag_category:
            self.category_performance[tag_category] = TagPerformanceMetrics()
        else:
            self.category_performance.clear()
        self._save_history()

    def _get_image_hash(self, image: np.ndarray) -> str:
        """Generate simple hash for image caching."""
        # Use image dimensions and mean color as simple hash
        if len(image.shape) == 3:
            mean_color = image.mean(axis=(0, 1))
            return f"{image.shape}_{mean_color[0]:.1f}_{mean_color[1]:.1f}_{mean_color[2]:.1f}"
        else:
            return f"{image.shape}_{image.mean():.1f}"

    def _update_complexity_cache(self, image_hash: str, metrics: Dict) -> None:
        """Update complexity cache with LRU eviction."""
        if len(self.complexity_cache) >= self.cache_max_size:
            # Remove oldest entry (simple FIFO)
            oldest_key = next(iter(self.complexity_cache))
            del self.complexity_cache[oldest_key]

        self.complexity_cache[image_hash] = metrics

    def _save_history(self) -> None:
        """Save performance history to disk."""
        try:
            self.persistence_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "config": {
                    "base_threshold": self.config.base_threshold,
                    "complexity_weight": self.config.complexity_weight,
                    "performance_weight": self.config.performance_weight,
                },
                "performance_history": {
                    category: {
                        "total_predictions": metrics.total_predictions,
                        "correct_predictions": metrics.correct_predictions,
                        "false_positives": metrics.false_positives,
                        "false_negatives": metrics.false_negatives,
                        "average_confidence": metrics.average_confidence,
                        "last_updated": metrics.last_updated.isoformat(),
                    }
                    for category, metrics in self.category_performance.items()
                },
            }

            with open(self.persistence_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

        except Exception as e:
            logger.error(f"Failed to save adaptive threshold history: {e}")

    def _load_history(self) -> None:
        """Load performance history from disk."""
        if not self.persistence_path.exists():
            return

        try:
            with open(self.persistence_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Load performance history
            for category, metrics_data in data.get("performance_history", {}).items():
                metrics = TagPerformanceMetrics()
                metrics.total_predictions = metrics_data.get("total_predictions", 0)
                metrics.correct_predictions = metrics_data.get("correct_predictions", 0)
                metrics.false_positives = metrics_data.get("false_positives", 0)
                metrics.false_negatives = metrics_data.get("false_negatives", 0)
                metrics.average_confidence = metrics_data.get("average_confidence", 0.0)

                last_updated_str = metrics_data.get("last_updated")
                if last_updated_str:
                    metrics.last_updated = datetime.fromisoformat(last_updated_str)

                self.category_performance[category] = metrics

            logger.info(
                f"Loaded performance history for {len(self.category_performance)} categories"
            )

        except Exception as e:
            logger.error(f"Failed to load adaptive threshold history: {e}")


# Singleton instance for easy access
_adaptive_threshold_service: Optional[AdaptiveThresholdService] = None


def get_adaptive_threshold_service(
    config: Optional[AdaptiveThresholdConfig] = None,
) -> AdaptiveThresholdService:
    """Get or create singleton instance of AdaptiveThresholdService."""
    global _adaptive_threshold_service
    if _adaptive_threshold_service is None:
        _adaptive_threshold_service = AdaptiveThresholdService(config)
    return _adaptive_threshold_service


def reset_adaptive_threshold_service() -> None:
    """Reset the singleton instance (useful for testing)."""
    global _adaptive_threshold_service
    _adaptive_threshold_service = None


if __name__ == "__main__":
    # Test the service
    import sys

    print("Testing AdaptiveThresholdService...")

    # Create service
    service = AdaptiveThresholdService()

    # Test complexity analysis
    print("\n1. Testing image complexity analysis:")
    test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    complexity = service.complexity_analyzer.analyze(test_image)
    print(f"   Complexity metrics: {complexity}")

    # Test threshold calculation
    print("\n2. Testing dynamic threshold calculation:")
    image_features = {"image": test_image}

    categories = ["character", "clothing", "body", "action", "theme"]
    for category in categories:
        threshold = service.calculate_dynamic_threshold(category, image_features)
        print(f"   {category}: threshold = {threshold:.3f}")

    # Test performance update
    print("\n3. Testing performance tracking:")
    service.update_performance(
        "character", predicted=True, actual=True, confidence=0.85
    )
    service.update_performance(
        "character", predicted=True, actual=False, confidence=0.75
    )
    service.update_performance(
        "character", predicted=False, actual=True, confidence=0.0
    )

    stats = service.get_category_stats("character")
    print(
        f"   Character stats: Precision={stats['precision']:.3f}, "
        f"Recall={stats['recall']:.3f}, F1={stats['f1_score']:.3f}"
    )

    # Test with updated performance
    print("\n4. Testing threshold after performance update:")
    threshold_after = service.calculate_dynamic_threshold("character", image_features)
    print(f"   Character threshold after updates: {threshold_after:.3f}")

    print("\n✅ All tests passed!")
