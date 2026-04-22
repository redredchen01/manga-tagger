"""Tag Library Service for managing and matching tags."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from app.core.config import settings
from app.core.logging_config import get_logger

logger = get_logger(__name__)


class TagLibraryService:
    """Service for loading and matching tags from the tag library."""

    def __init__(self, tag_library_path: Optional[str] = None):
        self.tag_library_path = tag_library_path or "./data/tags.json"
        self.tags: List[Dict[str, Any]] = []
        self.tag_names: List[str] = []
        self.tag_categories: Dict[str, List[str]] = {}
        self._tag_name_lower_map: Dict[str, str] = {}
        self._tag_description_map: Dict[str, str] = {}

        self._load_tags()

    def _load_tags(self):
        try:
            tag_path = Path(self.tag_library_path)
            if not tag_path.exists():
                logger.warning("Tag library not found at %s", self.tag_library_path)
                return

            with open(tag_path, "r", encoding="utf-8") as f:
                self.tags = json.load(f)

            self.tag_names = [tag["tag_name"] for tag in self.tags if "tag_name" in tag]
            self._tag_name_lower_map = {tag.lower(): tag for tag in self.tag_names}
            self._tag_description_map = {
                tag["tag_name"]: tag.get("description", "")
                for tag in self.tags
                if "tag_name" in tag and tag.get("description")
            }
            self._categorize_tags()
            logger.info("Loaded %s tags from library", len(self.tag_names))

        except Exception as e:
            logger.error("Failed to load tag library: %s", e)
            self.tags = []
            self.tag_names = []
            self._tag_name_lower_map = {}
            self._tag_description_map = {}

    def _categorize_tags(self):
        category_keywords = {
            "character": frozenset(
                {
                    "蘿莉",
                    "人妻",
                    "老太婆",
                    "獸耳",
                    "精靈",
                    "天使",
                    "惡魔娘",
                    "吸血鬼",
                }
            ),
            "clothing": frozenset({"女生制服", "男生制服", "泳裝", "比基尼", "女僕裝"}),
            "body": frozenset(
                {"巨乳", "貧乳", "眼鏡", "大肌肉", "雙馬尾", "馬尾", "超長髮", "精靈短髮"}
            ),
            "action": frozenset(
                {
                    "口交",
                    "乳交",
                    "肛交",
                    "觸手",
                    "自慰",
                    "獸交",
                    "調教",
                    "中出",
                    "內射",
                    "精液",
                    "露吹",
                }
            ),
            "theme": frozenset({"NTR", "百合", "後宮", "奴隸", "純愛", "凌虐", "調教"}),
        }

        self.tag_categories = {cat: [] for cat in category_keywords}
        self.tag_categories["other"] = []

        keyword_to_category: dict[str, str] = {}
        for category, keywords in category_keywords.items():
            for keyword in keywords:
                keyword_to_category[keyword] = category

        for tag_name in self.tag_names:
            category = None
            for keyword, cat in keyword_to_category.items():
                if keyword in tag_name or tag_name in keyword:
                    category = cat
                    break
            if category:
                self.tag_categories[category].append(tag_name)
            else:
                self.tag_categories["other"].append(tag_name)

    def get_all_tags(self) -> List[str]:
        return self.tag_names.copy()

    def get_tags_by_category(self, category: str) -> List[str]:
        return self.tag_categories.get(category, []).copy()

    def search_tags(self, query: str, limit: int = 10) -> List[str]:
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
        return self._tag_description_map.get(tag_name)

    def get_tag_definitions(self, tag_names: List[str]) -> Dict[str, str]:
        definitions = {}
        for name in tag_names:
            description = self._tag_description_map.get(name)
            if description:
                definitions[name] = description
        return definitions

    def match_tags_by_keywords(
        self, keywords: List[str], min_confidence: float = 0.6
    ) -> List[Tuple[str, float]]:
        matches = []
        for keyword in keywords:
            keyword_lower = keyword.lower().strip()
            if keyword_lower in self._tag_name_lower_map:
                exact_tag = self._tag_name_lower_map[keyword_lower]
                matches.append((exact_tag, 1.0))
                continue
            if len(keyword_lower) < 2:
                continue
            for tag_name in self.tag_names:
                tag_lower = tag_name.lower()
                if keyword_lower in tag_lower or tag_lower in keyword_lower:
                    ratio = len(keyword_lower) / len(tag_lower) if len(tag_lower) > 0 else 0
                    confidence = 0.8 + (ratio * 0.2)
                    if confidence >= min_confidence:
                        matches.append((tag_name, min(confidence, 1.0)))
                        continue
                # Specific check for sub-strings for longer tags (e.g. 'seifuku' in 'school seifuku')
                if len(keyword_lower) >= 4 and keyword_lower in tag_lower:
                    confidence = 0.75 + (len(keyword_lower) / len(tag_lower) * 0.1)
                    matches.append((tag_name, min(confidence, 0.95)))
                    continue
                keyword_parts = keyword_lower.split()
                tag_parts = tag_lower.split()
                matching_parts = sum(1 for kp in keyword_parts if any(kp in tp for tp in tag_parts))
                if matching_parts > 0:
                    confidence = 0.6 + (matching_parts / max(len(keyword_parts), 1)) * 0.2
                    if confidence >= min_confidence:
                        matches.append((tag_name, min(confidence, 0.9)))
                        continue
                # 降低 char match 權重 (從 0.5 base + 0.3 ratio 改為 0.3 base + 0.2 ratio)
                # 這是最低優先級的匹配方式
                char_matches = sum(1 for c in keyword_lower if c in tag_lower)
                if char_matches >= max(1, int(len(keyword_lower) * 0.85)):  # 提高門檻從 0.7 到 0.85
                    confidence = 0.3 + (char_matches / len(keyword_lower)) * 0.2
                    if confidence >= min_confidence:
                        matches.append((tag_name, min(confidence, 0.7)))  # 降低上限從 0.8 到 0.7

            description_hits = [
                (
                    name,
                    0.65,
                )
                for name, description in self._tag_description_map.items()
                if keyword_lower in description.lower()
            ]
            matches.extend(description_hits)

        matches.sort(key=lambda x: x[1], reverse=True)
        seen = set()
        unique_matches = []
        for tag, conf in matches:
            if tag not in seen:
                seen.add(tag)
                unique_matches.append((tag, conf))
        return unique_matches

    def validate_tags(self, tags: List[str]) -> List[Tuple[str, bool]]:
        return [(tag, tag in self.tag_names) for tag in tags]

    def match_tags_fuzzy(
        self, keywords: List[str], min_confidence: float = 0.5
    ) -> List[Tuple[str, float]]:
        """使用 difflib fuzzy matching 進行標籤匹配（備用方法）。"""
        import difflib

        matches = []
        for keyword in keywords:
            keyword_lower = keyword.lower().strip()
            if len(keyword_lower) < 2:
                continue
            for tag_name in self.tag_names:
                tag_lower = tag_name.lower()
                # 使用 SequenceMatcher 計算相似度
                ratio = difflib.SequenceMatcher(None, keyword_lower, tag_lower).ratio()
                if ratio >= min_confidence:
                    matches.append((tag_name, ratio))
        # 排序並去重
        matches.sort(key=lambda x: x[1], reverse=True)
        seen = set()
        unique = []
        for tag, conf in matches:
            if tag not in seen:
                seen.add(tag)
                unique.append((tag, conf))
        return unique

    def suggest_related_tags(self, existing_tags: List[str], limit: int = 5) -> List[str]:
        suggestions = []
        categories = set()
        for tag in existing_tags:
            for cat, tags in self.tag_categories.items():
                if tag in tags:
                    categories.add(cat)
        for cat in categories:
            for tag in self.tag_categories[cat]:
                if tag not in existing_tags and tag not in suggestions:
                    suggestions.append(tag)
                    if len(suggestions) >= limit:
                        return suggestions
        return suggestions

    # =========================================================================
    # Enhanced matching methods for improved precision
    # =========================================================================

    def get_hierarchical_parent(self, tag: str) -> Optional[str]:
        """Get the parent/generic tag if this is a specific tag."""
        return settings.TAG_HIERARCHY.get(tag)

    def is_specific_tag(self, tag: str) -> bool:
        """Check if tag has a hierarchical relationship (is specific version)."""
        return tag in settings.TAG_HIERARCHY

    def apply_hierarchical_boost(self, tags: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        """Apply confidence boost for specific tags over generic ones.

        When we have both a specific tag (貓娘) and its generic (獸耳),
        boost the specific tag and keep only it.
        """
        if not tags:
            return tags

        # Group tags by their generic parent
        tag_dict = {tag: conf for tag, conf in tags}
        boosted = []
        processed_generics = set()

        for tag, conf in tags:
            parent = self.get_hierarchical_parent(tag)

            # If this is a specific tag, check if generic is also present
            if parent and parent in tag_dict:
                # Boost specific tag, penalize generic
                specific_boost = 1.10
                generic_penalty = 0.85

                boosted.append((tag, min(conf * specific_boost, 1.0)))
                # Mark parent as processed with penalty
                if parent not in processed_generics:
                    boosted.append((parent, tag_dict[parent] * generic_penalty))
                    processed_generics.add(parent)
            elif tag not in processed_generics:
                boosted.append((tag, conf))

        # Re-sort by confidence
        boosted.sort(key=lambda x: x[1], reverse=True)
        return boosted

    def apply_visual_feature_boost(
        self, tags: List[Tuple[str, float]], has_visual_support: bool = True
    ) -> List[Tuple[str, float]]:
        """Apply confidence boost for tags with visual feature support.

        Args:
            tags: List of (tag, confidence) tuples
            has_visual_support: Whether we have visual evidence for these tags
        """
        if not tags or not has_visual_support:
            return tags

        boosted = []
        for tag, conf in tags:
            if tag in settings.VISUAL_FEATURE_BOOST:
                boost = settings.VISUAL_FEATURE_BOOST[tag]
                boosted.append((tag, min(conf * boost, 1.0)))
                logger.debug(
                    f"Applied visual feature boost to {tag}: {conf:.2f} -> {min(conf * boost, 1.0):.2f}"
                )
            else:
                boosted.append((tag, conf))

        return boosted

    def check_mutual_exclusivity(self, tags: List[str]) -> List[Tuple[str, bool, Optional[str]]]:
        """Check for mutual exclusivity violations.

        Returns list of (tag, is_valid, conflict_tag) tuples.
        """
        results = []
        tag_set = set(tags)

        for tag in tags:
            conflicts = settings.MUTUAL_EXCLUSIVITY.get(tag, set())
            conflict = tag_set.intersection(conflicts)
            if conflict:
                conflict_tag = next(iter(conflict))
                results.append((tag, False, conflict_tag))
            else:
                results.append((tag, True, None))

        return results

    def resolve_mutual_exclusivity(self, tags: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        """Resolve mutual exclusivity by keeping higher confidence tag.

        When two mutually exclusive tags both have high confidence,
        keep only the one with higher confidence.
        """
        if not tags:
            return tags

        # Sort by confidence descending
        sorted_tags = sorted(tags, key=lambda x: x[1], reverse=True)
        kept = []
        blocked = set()

        for tag, conf in sorted_tags:
            if tag in blocked:
                continue

            kept.append((tag, conf))

            # Block conflicting tags
            conflicts = settings.MUTUAL_EXCLUSIVITY.get(tag, set())
            for conflict in conflicts:
                blocked.add(conflict)

        return kept


_tag_library_service: Optional[TagLibraryService] = None


def get_tag_library_service() -> TagLibraryService:
    global _tag_library_service
    if _tag_library_service is None:
        from app.core.config import settings

        _tag_library_service = TagLibraryService(settings.TAG_LIBRARY_PATH)
    return _tag_library_service
