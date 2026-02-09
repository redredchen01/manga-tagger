"""Tag Library Service for managing and matching tags.

Loads the 611-tag library and provides intelligent matching capabilities.
"""

import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import re

logger = logging.getLogger(__name__)


class TagLibraryService:
    """Service for loading and matching tags from the tag library."""

    def __init__(self, tag_library_path: Optional[str] = None):
        """Initialize tag library service."""
        self.tag_library_path = tag_library_path or "./data/tags.json"
        self.tags: List[Dict[str, Any]] = []
        self.tag_names: List[str] = []
        self.tag_categories: Dict[str, List[str]] = {}

        self._load_tags()

    def _load_tags(self):
        """Load tags from JSON file."""
        try:
            tag_path = Path(self.tag_library_path)
            if not tag_path.exists():
                logger.warning(f"Tag library not found at {self.tag_library_path}")
                return

            with open(tag_path, "r", encoding="utf-8") as f:
                self.tags = json.load(f)

            self.tag_names = [tag["tag_name"] for tag in self.tags if "tag_name" in tag]

            # Categorize tags by type
            self._categorize_tags()

            logger.info(f"Loaded {len(self.tag_names)} tags from library")

        except Exception as e:
            logger.error(f"Failed to load tag library: {e}")
            self.tags = []
            self.tag_names = []

    def _categorize_tags(self):
        """Categorize tags by type for better matching."""
        # Character type tags
        character_keywords = [
            "蘿莉",
            "少女",
            "人妻",
            "熟女",
            "御姐",
            "正太",
            "少年",
            "貓娘",
            "犬娘",
            "狐娘",
            "獸人",
            "精靈",
            "魔物",
            "機娘",
            "天使",
            "惡魔",
            "幽靈",
            "喪屍",
            "喪尸",
            "機械人",
        ]

        # Clothing tags
        clothing_keywords = [
            "校服",
            "泳裝",
            "内衣",
            "和服",
            "女僕",
            "護士",
            "警察",
            "體操服",
            "兔女郎",
            "旗袍",
            "巫女",
            "婚纱",
            "西裝",
        ]

        # Body features
        body_keywords = [
            "巨乳",
            "貧乳",
            "長腿",
            "絲襪",
            "吊帶襪",
            "過膝襪",
            "裸足",
            "眼鏡",
            "義眼",
            "義肢",
            "疤痕",
            "紋身",
        ]

        # Action/Scene
        action_keywords = [
            "做愛",
            "口交",
            "手淫",
            "乳交",
            "足交",
            "肛交",
            "群交",
            "綁架",
            "監禁",
            "調教",
            "陵辱",
            "強姦",
            "痴漢",
        ]

        # Theme/Genre
        theme_keywords = [
            "純愛",
            "NTR",
            "凌辱",
            "調教",
            "奴隸",
            "主從",
            "百合",
            "耽美",
            "亂倫",
            "師生",
            "職場",
            "近親",
        ]

        self.tag_categories = {
            "character": [],
            "clothing": [],
            "body": [],
            "action": [],
            "theme": [],
            "other": [],
        }

        for tag_name in self.tag_names:
            categorized = False
            for keyword in character_keywords:
                if keyword in tag_name or tag_name in keyword:
                    self.tag_categories["character"].append(tag_name)
                    categorized = True
                    break

            if not categorized:
                for keyword in clothing_keywords:
                    if keyword in tag_name or tag_name in keyword:
                        self.tag_categories["clothing"].append(tag_name)
                        categorized = True
                        break

            if not categorized:
                for keyword in body_keywords:
                    if keyword in tag_name or tag_name in keyword:
                        self.tag_categories["body"].append(tag_name)
                        categorized = True
                        break

            if not categorized:
                for keyword in action_keywords:
                    if keyword in tag_name or tag_name in keyword:
                        self.tag_categories["action"].append(tag_name)
                        categorized = True
                        break

            if not categorized:
                for keyword in theme_keywords:
                    if keyword in tag_name or tag_name in keyword:
                        self.tag_categories["theme"].append(tag_name)
                        categorized = True
                        break

            if not categorized:
                self.tag_categories["other"].append(tag_name)

    def get_all_tags(self) -> List[str]:
        """Get all available tag names."""
        return self.tag_names.copy()

    def get_tags_by_category(self, category: str) -> List[str]:
        """Get tags by category."""
        return self.tag_categories.get(category, []).copy()

    def search_tags(self, query: str, limit: int = 10) -> List[str]:
        """Search for tags matching query."""
        if not query:
            return []

        query_lower = query.lower()
        matches = []

        for tag_name in self.tag_names:
            if query_lower in tag_name.lower():
                matches.append(tag_name)
                if len(matches) >= limit:
                    break

        return matches

    def get_tag_description(self, tag_name: str) -> Optional[str]:
        """Get description for a tag."""
        for tag in self.tags:
            if tag.get("tag_name") == tag_name:
                return tag.get("description")
        return None

    def get_tag_definitions(self, tag_names: List[str]) -> Dict[str, str]:
        """Get descriptions for multiple tags as a dictionary."""
        definitions = {}
        tags_to_find = set(tag_names)
        for tag_data in self.tags:
            name = tag_data.get("tag_name")
            if name in tags_to_find:
                description = tag_data.get("description", "")
                if description:
                    definitions[name] = description
                tags_to_find.remove(name)
                if not tags_to_find:
                    break
        return definitions

    def match_tags_by_keywords(
        self, keywords: List[str], min_confidence: float = 0.6
    ) -> List[Tuple[str, float]]:
        """Match keywords to tags from library with confidence scores.

        Returns list of (tag_name, confidence) tuples.
        """
        matches = []

        for keyword in keywords:
            keyword_lower = keyword.lower().strip()

            for tag_name in self.tag_names:
                tag_lower = tag_name.lower()

                # Exact match
                if keyword_lower == tag_lower:
                    matches.append((tag_name, 1.0))
                    continue

                # For very short keywords (< 4 chars), ONLY allow exact match
                # This prevents "to", "in", "2" from matching long tags like "少女" if they happen to contain them
                if len(keyword_lower) < 4:
                    continue

                # Contains match (only for keywords >= 4 chars)
                if keyword_lower in tag_lower or tag_lower in keyword_lower:
                    # Calculate confidence based on length ratio
                    ratio = (
                        len(keyword_lower) / len(tag_lower) if len(tag_lower) > 0 else 0
                    )
                    confidence = 0.8 + (ratio * 0.2)  # 0.8 to 1.0
                    matches.append((tag_name, min(confidence, 1.0)))
                    continue

                # Partial word match (only for keywords >= 4 chars)
                keyword_parts = keyword_lower.split()
                tag_parts = tag_lower.split()

                matching_parts = sum(
                    1 for kp in keyword_parts if any(kp in tp for tp in tag_parts)
                )
                if matching_parts > 0:
                    confidence = 0.6 + (matching_parts / len(keyword_parts)) * 0.2
                    if confidence >= min_confidence:
                        matches.append((tag_name, min(confidence, 0.9)))

        # Sort by confidence and remove duplicates
        matches.sort(key=lambda x: x[1], reverse=True)
        seen = set()
        unique_matches = []
        for tag, conf in matches:
            if tag not in seen:
                seen.add(tag)
                unique_matches.append((tag, conf))

        return unique_matches

    def validate_tags(self, tags: List[str]) -> List[Tuple[str, bool]]:
        """Validate if tags exist in library.

        Returns list of (tag_name, is_valid) tuples.
        """
        return [(tag, tag in self.tag_names) for tag in tags]

    def suggest_related_tags(
        self, existing_tags: List[str], limit: int = 5
    ) -> List[str]:
        """Suggest related tags based on existing ones."""
        suggestions = []

        # Find categories of existing tags
        categories = set()
        for tag in existing_tags:
            for cat, tags in self.tag_categories.items():
                if tag in tags:
                    categories.add(cat)

        # Suggest tags from same categories
        for cat in categories:
            for tag in self.tag_categories[cat]:
                if tag not in existing_tags and tag not in suggestions:
                    suggestions.append(tag)
                    if len(suggestions) >= limit:
                        return suggestions

        return suggestions


# Singleton instance
_tag_library_service: Optional[TagLibraryService] = None


def get_tag_library_service() -> TagLibraryService:
    """Get or create tag library service singleton."""
    global _tag_library_service
    if _tag_library_service is None:
        from app.config import settings

        _tag_library_service = TagLibraryService(settings.TAG_LIBRARY_PATH)
    return _tag_library_service
