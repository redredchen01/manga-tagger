"""Emergency Tag Service for fallback scenarios.

When all other tagging methods fail, this service provides basic tags
based on statistical analysis of common manga/anime tags.
"""

import logging
from typing import List, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class EmergencyTag:
    """A tag with its default confidence and category."""
    tag: str
    confidence: float
    category: str
    reason: str


class EmergencyTagService:
    """Service for providing emergency tags when VLM and RAG fail.
    
    This is the last resort fallback to ensure users always get
    at least some tag recommendations.
    """
    
    # Common tags based on manga/anime statistics
    # Ordered by frequency/popularity
    EMERGENCY_TAGS = [
        # Character types (most common)
        EmergencyTag("少女", 0.75, "character", "最常見角色類型"),
        EmergencyTag("女性", 0.85, "character", "基本角色性別"),
        
        # Body features
        EmergencyTag("巨乳", 0.55, "body", "常見體型特徵"),
        EmergencyTag("貧乳", 0.50, "body", "常見體型特徵"),
        
        # Clothing
        EmergencyTag("校服", 0.50, "clothing", "常見服裝"),
        EmergencyTag("泳裝", 0.45, "clothing", "常見服裝"),
        
        # Themes
        EmergencyTag("戀愛", 0.50, "theme", "常見主題"),
        EmergencyTag("校園", 0.45, "theme", "常見主題"),
        
        # Additional common tags
        EmergencyTag("長髮", 0.50, "body", "常見特徵"),
        EmergencyTag("單人", 0.60, "other", "常見構圖"),
    ]
    
    # Category priority for selection
    CATEGORY_PRIORITY = ["character", "body", "clothing", "theme", "other"]
    
    def __init__(self):
        """Initialize the emergency tag service."""
        self._tags_by_category = {}
        for tag in self.EMERGENCY_TAGS:
            if tag.category not in self._tags_by_category:
                self._tags_by_category[tag.category] = []
            self._tags_by_category[tag.category].append(tag)
        
        logger.info(f"EmergencyTagService initialized with {len(self.EMERGENCY_TAGS)} tags")
    
    def get_emergency_tags(
        self, 
        top_k: int = 5,
        exclude_tags: Optional[List[str]] = None
    ) -> List[Tuple[str, float, str]]:
        """Get emergency tags for fallback scenarios.
        
        Args:
            top_k: Number of tags to return
            exclude_tags: Tags to exclude (already selected)
            
        Returns:
            List of (tag_name, confidence, reason) tuples
        """
        exclude_set = set(exclude_tags or [])
        selected_tags = []
        selected_categories = set()
        
        # First pass: Get one tag from each priority category
        for category in self.CATEGORY_PRIORITY:
            if len(selected_tags) >= top_k:
                break
                
            if category not in self._tags_by_category:
                continue
                
            for tag in self._tags_by_category[category]:
                if tag.tag not in exclude_set:
                    selected_tags.append((tag.tag, tag.confidence, tag.reason))
                    selected_categories.add(category)
                    exclude_set.add(tag.tag)
                    break
        
        # Second pass: Fill remaining slots with highest confidence tags
        if len(selected_tags) < top_k:
            remaining = []
            for tag in self.EMERGENCY_TAGS:
                if tag.tag not in exclude_set:
                    remaining.append((tag.tag, tag.confidence, tag.reason))
            
            # Sort by confidence
            remaining.sort(key=lambda x: x[1], reverse=True)
            
            for tag_tuple in remaining:
                if len(selected_tags) >= top_k:
                    break
                selected_tags.append(tag_tuple)
        
        logger.info(f"Emergency fallback provided {len(selected_tags)} tags")
        return selected_tags
    
    def get_category_distribution(self) -> dict:
        """Get the distribution of tags by category."""
        return {
            category: len(tags) 
            for category, tags in self._tags_by_category.items()
        }


# Singleton instance
_emergency_tag_service: Optional[EmergencyTagService] = None


def get_emergency_tag_service() -> EmergencyTagService:
    """Get or create the emergency tag service singleton."""
    global _emergency_tag_service
    if _emergency_tag_service is None:
        _emergency_tag_service = EmergencyTagService()
    return _emergency_tag_service
