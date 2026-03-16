"""Tag Outlier Detection Service.

Detects and filters potentially incorrect tags based on:
1. Statistical outliers in confidence distribution
2. Conflicting tag patterns
3. Confidence consistency with tag relationships
"""

import logging
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class OutlierResult:
    """Result of outlier detection."""

    is_outlier: bool
    reason: str
    confidence: float


class TagOutlierDetector:
    """Detects potentially incorrect tags using statistical methods."""

    def __init__(self, z_threshold: float = 2.0, min_samples: int = 3):
        """Initialize the outlier detector.

        Args:
            z_threshold: Z-score threshold for outlier detection (default: 2.0)
            min_samples: Minimum samples needed for statistical analysis
        """
        self.z_threshold = z_threshold
        self.min_samples = min_samples

        # Known conflicting tag pairs (should not appear together)
        # Format: (tag_a, tag_b) - if both have high confidence, flag for review
        self.mutual_exclusions = {
            # Body type conflicts
            ("巨乳", "貧乳"),
            ("大型胸部", "貧乳"),
            ("長腿", "短腿"),
            # Age conflicts
            ("蘿莉", "熟女"),
            ("蘿莉", "人妻"),
            ("正太", "大叔"),
            ("正太", "成年"),
            # Species conflicts
            ("貓娘", "犬娘"),
            ("狐娘", "人類"),
            # Clothing conflicts
            ("全裸", "衣服"),
            ("內衣", "全裸"),
        }

        # Tags that typically have LOW confidence when they appear
        # (prone to false positives)
        self.low_reliability_tags = {
            "蘿莉": 0.7,  # Often over-detected
            "正太": 0.7,
            "人妻": 0.65,
            "熟女": 0.65,
            "SM": 0.6,
            "調教": 0.6,
        }

    def detect_outliers(
        self, tags: List[str], confidences: Dict[str, float]
    ) -> Dict[str, OutlierResult]:
        """Detect outliers in tag confidence distribution.

        Args:
            tags: List of tag names
            confidences: Dict mapping tag names to confidence scores

        Returns:
            Dict mapping tag names to OutlierResult
        """
        if len(tags) < self.min_samples:
            return {}

        scores = [confidences.get(tag, 0.0) for tag in tags]

        # Calculate statistics
        mean_conf = np.mean(scores)
        std_conf = np.std(scores)

        if std_conf == 0:
            return {}

        results = {}

        for tag, score in zip(tags, scores):
            z_score = (score - mean_conf) / std_conf

            # Check for statistical outlier (very high or very low)
            if abs(z_score) > self.z_threshold:
                direction = "high" if z_score > 0 else "low"
                results[tag] = OutlierResult(
                    is_outlier=True,
                    reason=f"Statistical outlier: {direction} confidence (z={z_score:.2f})",
                    confidence=score,
                )

        return results

    def check_conflicts(
        self, tags: List[str], confidences: Dict[str, float]
    ) -> List[Tuple[str, str, float]]:
        """Check for conflicting tag pairs.

        Args:
            tags: List of predicted tags
            confidences: Dict mapping tag names to confidence scores

        Returns:
            List of (tag_a, tag_b, conflict_score) tuples
        """
        conflicts = []
        tag_set = set(tags)

        for tag_a, tag_b in self.mutual_exclusions:
            if tag_a in tag_set and tag_b in tag_set:
                conf_a = confidences.get(tag_a, 0)
                conf_b = confidences.get(tag_b, 0)

                # Higher conflict score = more likely to be wrong
                conflict_score = (conf_a + conf_b) / 2

                if conflict_score > 0.5:
                    conflicts.append((tag_a, tag_b, conflict_score))
                    logger.warning(
                        f"Conflict detected: {tag_a} <-> {tag_b} (score: {conflict_score:.2f})"
                    )

        return conflicts

    def check_low_reliability(
        self, tags: List[str], confidences: Dict[str, float]
    ) -> Dict[str, float]:
        """Check for tags that are prone to false positives.

        Args:
            tags: List of predicted tags
            confidences: Dict mapping tag names to confidence scores

        Returns:
            Dict mapping unreliable tags to their required minimum confidence
        """
        unreliable = {}

        for tag in tags:
            if tag in self.low_reliability_tags:
                min_conf = self.low_reliability_tags[tag]
                actual_conf = confidences.get(tag, 0)

                if actual_conf < min_conf:
                    unreliable[tag] = {
                        "required": min_conf,
                        "actual": actual_conf,
                        "gap": min_conf - actual_conf,
                    }

        return unreliable

    def filter_and_rank(
        self, tags: List[str], confidences: Dict[str, float]
    ) -> Tuple[List[str], Dict[str, float], List[str]]:
        """Filter out outliers and return cleaned results.

        Args:
            tags: List of predicted tags
            confidences: Dict mapping tag names to confidence scores

        Returns:
            Tuple of (filtered_tags, adjusted_confidences, warnings)
        """
        filtered_tags = list(tags)
        adjusted_confidences = dict(confidences)
        warnings = []

        # Step 1: Detect statistical outliers
        outliers = self.detect_outliers(filtered_tags, adjusted_confidences)
        for tag, result in outliers.items():
            # Downgrade confidence for outliers
            adjusted_confidences[tag] *= 0.7
            warnings.append(f"Downgraded '{tag}': {result.reason}")

        # Step 2: Check mutual exclusions
        conflicts = self.check_conflicts(filtered_tags, adjusted_confidences)
        for tag_a, tag_b, score in conflicts:
            # Reduce confidence of lower-confidence tag
            conf_a = adjusted_confidences.get(tag_a, 0)
            conf_b = adjusted_confidences.get(tag_b, 0)

            if conf_a < conf_b:
                adjusted_confidences[tag_a] *= 0.5
                warnings.append(
                    f"Conflict penalty: '{tag_a}' reduced due to conflict with '{tag_b}'"
                )
            else:
                adjusted_confidences[tag_b] *= 0.5
                warnings.append(
                    f"Conflict penalty: '{tag_b}' reduced due to conflict with '{tag_a}'"
                )

        # Step 3: Check low reliability tags
        unreliable = self.check_low_reliability(filtered_tags, adjusted_confidences)
        for tag, info in unreliable.items():
            # Only flag as warning, don't automatically remove
            warnings.append(
                f"Low reliability tag '{tag}': confidence {info['actual']:.2f} "
                f"below typical {info['required']:.2f}"
            )

        # Step 4: Remove tags with very low adjusted confidence
        min_threshold = 0.15
        final_tags = []
        for tag in filtered_tags:
            if adjusted_confidences.get(tag, 0) >= min_threshold:
                final_tags.append(tag)
            else:
                warnings.append(f"Filtered out '{tag}': adjusted confidence too low")

        return final_tags, adjusted_confidences, warnings


# Singleton
_outlier_detector: Optional[TagOutlierDetector] = None


def get_outlier_detector() -> TagOutlierDetector:
    """Get or create outlier detector singleton."""
    global _outlier_detector
    if _outlier_detector is None:
        _outlier_detector = TagOutlierDetector()
    return _outlier_detector
