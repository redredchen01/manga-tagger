"""
Dynamic Weight Calculator Module
Optimizes hybrid matching weights based on tag types and query characteristics.
"""

import numpy as np
from typing import Dict, Tuple, Optional, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class WeightConfig:
    """Configuration for dynamic weight calculation."""

    # Default weights
    default_lexical: float = 0.70
    default_semantic: float = 0.30

    # Category-specific base weights
    character_weights: Dict[str, float] = None
    clothing_weights: Dict[str, float] = None
    body_weights: Dict[str, float] = None
    action_weights: Dict[str, float] = None
    theme_weights: Dict[str, float] = None

    def __post_init__(self):
        if self.character_weights is None:
            self.character_weights = {"lexical": 0.80, "semantic": 0.20}
        if self.clothing_weights is None:
            self.clothing_weights = {"lexical": 0.60, "semantic": 0.40}
        if self.body_weights is None:
            self.body_weights = {"lexical": 0.40, "semantic": 0.60}
        if self.action_weights is None:
            self.action_weights = {"lexical": 0.50, "semantic": 0.50}
        if self.theme_weights is None:
            self.theme_weights = {"lexical": 0.65, "semantic": 0.35}


class DynamicWeightCalculator:
    """
    Calculates dynamic weights for hybrid matching based on tag category and query complexity.
    """

    def __init__(self, config: Optional[WeightConfig] = None):
        """Initialize the weight calculator."""
        self.config = config or WeightConfig()

        # Category mapping
        self.category_map = {
            "character": [
                "蘿莉",
                "少女",
                "人妻",
                "熟女",
                "御姐",
                "正太",
                "少年",
                "貓娘",
                "狐娘",
                "犬娘",
                "精靈",
                "魔物",
                "天使",
                "惡魔",
            ],
            "clothing": [
                "校服",
                "泳裝",
                "女僕",
                "和服",
                "兔女郎",
                "體操服",
                "內衣",
                "比基尼",
            ],
            "body": [
                "巨乳",
                "貧乳",
                "爆乳",
                "絲襪",
                "長腿",
                "獸耳",
                "尾巴",
                "長髮",
                "短髮",
            ],
            "action": ["做愛", "口交", "乳交", "手淫", "群交", "強姦", "綁架"],
            "theme": ["純愛", "百合", "耽美", "後宮", "學園", "NTR", "調教"],
        }

        logger.info("DynamicWeightCalculator initialized")

    def calculate_weights(
        self, tag_category: str, query_complexity: float
    ) -> Tuple[float, float]:
        """
        Calculate dynamic lexical and semantic weights.

        Args:
            tag_category: Category of tags (character, clothing, body, etc.)
            query_complexity: Complexity of the query (0.0 to 1.0)

        Returns:
            Tuple of (lexical_weight, semantic_weight)
        """
        # Get base weights for category
        base_weights = self._get_category_weights(tag_category)

        # Adjust based on query complexity
        # Higher complexity -> more weight on semantic
        # Lower complexity -> more weight on lexical
        complexity_adjustment = query_complexity * 0.2

        lexical_weight = base_weights["lexical"] - complexity_adjustment
        semantic_weight = base_weights["semantic"] + complexity_adjustment

        # Normalize to ensure sum is reasonable
        total = lexical_weight + semantic_weight
        lexical_weight = lexical_weight / total * 1.0
        semantic_weight = semantic_weight / total * 1.0

        # Clip to valid range
        lexical_weight = np.clip(lexical_weight, 0.1, 0.9)
        semantic_weight = np.clip(semantic_weight, 0.1, 0.9)

        return lexical_weight, semantic_weight

    def _get_category_weights(self, tag_category: str) -> Dict[str, float]:
        """Get base weights for a tag category."""
        category_weights = {
            "character": self.config.character_weights,
            "clothing": self.config.clothing_weights,
            "body": self.config.body_weights,
            "action": self.config.action_weights,
            "theme": self.config.theme_weights,
        }

        return category_weights.get(
            tag_category,
            {
                "lexical": self.config.default_lexical,
                "semantic": self.config.default_semantic,
            },
        )

    def get_tag_category(self, tag: str) -> str:
        """Determine the category of a tag."""
        for category, tags in self.category_map.items():
            if tag in tags:
                return category
        return "general"

    def calculate_combined_score(
        self,
        lexical_score: float,
        semantic_score: float,
        tag_category: str = "general",
        query_complexity: float = 0.5,
    ) -> float:
        """
        Calculate combined score using dynamic weights.

        Args:
            lexical_score: Score from lexical matching (0-1)
            semantic_score: Score from semantic matching (0-1)
            tag_category: Category of the tag
            query_complexity: Complexity of the query

        Returns:
            Combined score (0-1)
        """
        lexical_weight, semantic_weight = self.calculate_weights(
            tag_category, query_complexity
        )

        combined = lexical_score * lexical_weight + semantic_score * semantic_weight

        return float(combined)


# Singleton instance
_weight_calculator: Optional[DynamicWeightCalculator] = None


def get_dynamic_weight_calculator(
    config: Optional[WeightConfig] = None,
) -> DynamicWeightCalculator:
    """Get or create singleton instance."""
    global _weight_calculator
    if _weight_calculator is None:
        _weight_calculator = DynamicWeightCalculator(config)
    return _weight_calculator


def reset_dynamic_weight_calculator() -> None:
    """Reset the singleton instance."""
    global _weight_calculator
    _weight_calculator = None
