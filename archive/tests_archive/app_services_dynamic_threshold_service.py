"""Dynamic Threshold Service.

Adjusts matching thresholds based on tag category and context.
Supports PRECISION-FIRST mode for reducing false positives.
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ThresholdConfig:
    """Threshold configuration for a category."""
    category: str
    base_threshold: float
    min_confidence: float
    max_confidence: float
    description: str


class DynamicThresholdService:
    """Service for managing dynamic matching thresholds.
    
    Supports two modes:
    - BALANCED (default): Balanced precision/recall
    - PRECISION_FIRST: Higher thresholds to reduce false positives
    """
    
    # Category-specific thresholds (BALANCED mode)
    THRESHOLD_CONFIGS: Dict[str, ThresholdConfig] = {
        # Character tags - higher threshold due to sensitivity
        "character": ThresholdConfig(
            category="character",
            base_threshold=0.65,
            min_confidence=0.65,
            max_confidence=1.0,
            description="Character tags (age, type) require higher confidence"
        ),
        # Body features - medium threshold
        "body": ThresholdConfig(
            category="body",
            base_threshold=0.55,
            min_confidence=0.55,
            max_confidence=1.0,
            description="Body feature tags have medium confidence requirements"
        ),
        # Clothing tags - medium threshold
        "clothing": ThresholdConfig(
            category="clothing",
            base_threshold=0.55,
            min_confidence=0.55,
            max_confidence=1.0,
            description="Clothing tags have medium confidence requirements"
        ),
        # Action tags - higher threshold for accuracy
        "action": ThresholdConfig(
            category="action",
            base_threshold=0.70,
            min_confidence=0.70,
            max_confidence=1.0,
            description="Action tags require higher confidence due to complexity"
        ),
        # Theme tags - high threshold
        "theme": ThresholdConfig(
            category="theme",
            base_threshold=0.70,
            min_confidence=0.70,
            max_confidence=1.0,
            description="Theme tags require high confidence"
        ),
        # Other tags - standard threshold
        "other": ThresholdConfig(
            category="other",
            base_threshold=0.50,
            min_confidence=0.50,
            max_confidence=1.0,
            description="Other tags use standard threshold"
        ),
        # Sensitive tags - highest threshold
        "sensitive": ThresholdConfig(
            category="sensitive",
            base_threshold=0.80,
            min_confidence=0.80,
            max_confidence=1.0,
            description="Sensitive tags require very high confidence"
        ),
    }
    
    # PRECISION-FIRST thresholds (higher = fewer false positives)
    PRECISION_THRESHOLD_CONFIGS: Dict[str, ThresholdConfig] = {
        "character": ThresholdConfig(
            category="character",
            base_threshold=0.75,
            min_confidence=0.75,
            max_confidence=1.0,
            description="PRECISION: Character tags - strict threshold"
        ),
        "body": ThresholdConfig(
            category="body",
            base_threshold=0.70,
            min_confidence=0.70,
            max_confidence=1.0,
            description="PRECISION: Body features - strict threshold"
        ),
        "clothing": ThresholdConfig(
            category="clothing",
            base_threshold=0.65,
            min_confidence=0.65,
            max_confidence=1.0,
            description="PRECISION: Clothing tags - strict threshold"
        ),
        "action": ThresholdConfig(
            category="action",
            base_threshold=0.80,
            min_confidence=0.80,
            max_confidence=1.0,
            description="PRECISION: Action tags - very strict threshold"
        ),
        "theme": ThresholdConfig(
            category="theme",
            base_threshold=0.82,
            min_confidence=0.82,
            max_confidence=1.0,
            description="PRECISION: Theme tags - very strict threshold"
        ),
        "other": ThresholdConfig(
            category="other",
            base_threshold=0.65,
            min_confidence=0.65,
            max_confidence=1.0,
            description="PRECISION: Other tags - strict threshold"
        ),
        "sensitive": ThresholdConfig(
            category="sensitive",
            base_threshold=0.90,
            min_confidence=0.90,
            max_confidence=1.0,
            description="PRECISION: Sensitive tags - extremely strict"
        ),
    }
    
    # Sensitive tag keywords
    SENSITIVE_KEYWORDS = {
        "loli", "蘿莉", "shota", "正太", "幼女",
        "anal", "肛交", "rape", "強姦", "bestiality", "獸交",
        "ntr", "調教", "凌辱", "群交", "乱倫",
        "incest", "近親", "母子", "父女",
        "rape", "非自願",
    }
    
    def __init__(self):
        """Initialize dynamic threshold service."""
        self.configs = self.THRESHOLD_CONFIGS.copy()
        logger.info(f"DynamicThresholdService initialized with {len(self.configs)} categories")
    
    def get_threshold(
        self,
        tag: str,
        base_threshold: float = 0.50,
        context: Optional[Dict] = None
    ) -> float:
        """Get the appropriate threshold for a tag.
        
        Args:
            tag: Tag name
            base_threshold: Base threshold to use if no category match
            context: Optional context (e.g., image metadata)
            
        Returns:
            Adjusted threshold value
        """
        # Check if tag is sensitive
        tag_lower = tag.lower()
        for keyword in self.SENSITIVE_KEYWORDS:
            if keyword in tag_lower:
                return self.configs["sensitive"].base_threshold
        
        # Determine category
        category = self._categorize_tag(tag)
        
        if category in self.configs:
            return self.configs[category].base_threshold
        
        return base_threshold
    
    def filter_by_threshold(
        self,
        tags: List[str],
        scores: Dict[str, float],
        base_threshold: float = 0.50
    ) -> Tuple[List[str], Dict[str, float]]:
        """Filter tags based on dynamic thresholds.
        
        Args:
            tags: List of tag names
            scores: Dict of tag -> confidence score
            base_threshold: Base threshold
            
        Returns:
            Tuple of (filtered_tags, filtered_scores)
        """
        filtered_tags = []
        filtered_scores = {}
        
        for tag in tags:
            score = scores.get(tag, 0.5)
            threshold = self.get_threshold(tag, base_threshold)
            
            if score >= threshold:
                filtered_tags.append(tag)
                filtered_scores[tag] = score
            else:
                logger.debug(f"Filtered out '{tag}': score={score:.3f} < threshold={threshold:.3f}")
        
        return filtered_tags, filtered_scores
    
    def _categorize_tag(self, tag: str) -> str:
        """Categorize a tag based on keywords."""
        tag_lower = tag.lower()
        
        # Character keywords
        char_keywords = ["蘿莉", "正太", "少女", "人妻", "熟女", "御姐", "老太婆", " lolicon"]
        for kw in char_keywords:
            if kw in tag_lower:
                return "character"
        
        # Body keywords
        body_keywords = ["巨乳", "貧乳", "平胸", "肌肉", "大肌肉", "纖細", "瘦", "胸部", "乳房"]
        for kw in body_keywords:
            if kw in tag_lower:
                return "body"
        
        # Clothing keywords
        clothing_keywords = ["內衣", "泳裝", "校服", "女僕", "護士", "裸體", "全裸", "穿衣", "絲襪"]
        for kw in clothing_keywords:
            if kw in tag_lower:
                return "clothing"
        
        # Action keywords
        action_keywords = ["口交", "肛交", "做愛", "自慰", "手淫", "乳交", "足交", "顏射", "中出"]
        for kw in action_keywords:
            if kw in tag_lower:
                return "action"
        
        # Theme keywords
        theme_keywords = ["純愛", "NTR", "凌辱", "調教", "強姦", "綠帽", "乱倫", "近親"]
        for kw in theme_keywords:
            if kw in tag_lower:
                return "theme"
        
        return "other"
    
    def adjust_score(
        self,
        tag: str,
        original_score: float,
        factors: Optional[Dict[str, float]] = None
    ) -> float:
        """Adjust a confidence score based on various factors.
        
        Args:
            tag: Tag name
            original_score: Original confidence score
            factors: Optional adjustment factors
            
        Returns:
            Adjusted score
        """
        factors = factors or {}
        adjusted = original_score
        
        # Category adjustment
        category = self._categorize_tag(tag)
        if category == "character":
            adjusted *= 0.95  # Slightly reduce character tag scores
        elif category == "action":
            adjusted *= 0.90  # Reduce action tag scores more
        elif category == "theme":
            adjusted *= 0.85  # Reduce theme tag scores
        
        # Apply any provided factors
        for factor_name, factor_value in factors.items():
            if factor_name == "vlm_confidence":
                # VLM-provided confidence
                adjusted = adjusted * 0.7 + original_score * factor_value * 0.3
            elif factor_name == "tag_frequency":
                # Frequency-based adjustment (more common = slightly lower)
                adjusted *= (1.0 - min(factor_value * 0.1, 0.2))
        
        # Ensure within bounds
        return min(max(adjusted, 0.0), 1.0)
    
    def get_category_info(self, category: str) -> Optional[ThresholdConfig]:
        """Get threshold config for a category.
        
        Args:
            category: Category name
            
        Returns:
            ThresholdConfig or None
        """
        return self.configs.get(category)
    
    def list_categories(self) -> List[str]:
        """List all available categories."""
        return list(self.configs.keys())


# Singleton instance
_threshold_service: Optional[DynamicThresholdService] = None


def get_dynamic_threshold_service() -> DynamicThresholdService:
    """Get or create DynamicThresholdService singleton."""
    global _threshold_service
    if _threshold_service is None:
        _threshold_service = DynamicThresholdService()
    return _threshold_service
