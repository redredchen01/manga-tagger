"""
Tag Description Standardization Tool
Standardizes and enhances tag descriptions for better embedding quality.

This tool helps improve tag matching accuracy by:
1. Standardizing description format
2. Extracting visual keywords
3. Adding negative keywords
4. Enhancing description quality
"""

import json
import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import logging
from collections import Counter

logger = logging.getLogger(__name__)


@dataclass
class StandardizedTag:
    """Standardized tag format with enhanced metadata."""

    tag_name: str
    description: str
    enhanced_description: str
    visual_keywords: List[str]
    negative_keywords: List[str]
    category: str
    required_confidence: float
    related_tags: List[str]
    conflicting_tags: List[str]
    usage_notes: str
    version: str = "1.0"


class TagDescriptionStandardizer:
    """
    Standardizes tag descriptions for consistent quality.
    """

    def __init__(self):
        """Initialize the standardizer."""
        self.category_keywords = {
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
            "clothing": ["校服", "泳裝", "女僕", "和服", "兔女郎", "體操服", "內衣"],
            "action": ["做愛", "口交", "乳交", "手淫", "群交", "強姦", "綁架"],
            "theme": ["純愛", "百合", "耽美", "後宮", "學園", "NTR", "調教"],
        }

        # Visual keyword patterns by category
        self.visual_patterns = {
            "character": ["外觀", "身材", "年齡", "特徵", "氣質"],
            "body": ["尺寸", "形狀", "顏色", "長度", "比例"],
            "clothing": ["顏色", "款式", "材質", "狀態", "搭配"],
            "action": ["姿勢", "場景", "對象", "強度"],
            "theme": ["氛圍", "情節", "關係", "風格"],
        }

    def standardize_tag(self, tag_data: Dict[str, Any]) -> StandardizedTag:
        """
        Standardize a single tag.

        Args:
            tag_data: Raw tag data with tag_name and description

        Returns:
            StandardizedTag with enhanced metadata
        """
        tag_name = tag_data.get("tag_name", "")
        original_description = tag_data.get("description", "")

        # Determine category
        category = self._determine_category(tag_name)

        # Enhance description
        enhanced_description = self._enhance_description(
            tag_name, original_description, category
        )

        # Extract visual keywords
        visual_keywords = self._extract_visual_keywords(
            tag_name, enhanced_description, category
        )

        # Extract negative keywords
        negative_keywords = self._extract_negative_keywords(
            tag_name, enhanced_description, category
        )

        # Determine required confidence
        required_confidence = self._determine_required_confidence(tag_name, category)

        # Get related and conflicting tags
        related_tags = self._get_related_tags(tag_name, category)
        conflicting_tags = self._get_conflicting_tags(tag_name, category)

        # Generate usage notes
        usage_notes = self._generate_usage_notes(tag_name, category)

        return StandardizedTag(
            tag_name=tag_name,
            description=original_description,
            enhanced_description=enhanced_description,
            visual_keywords=visual_keywords,
            negative_keywords=negative_keywords,
            category=category,
            required_confidence=required_confidence,
            related_tags=related_tags,
            conflicting_tags=conflicting_tags,
            usage_notes=usage_notes,
        )

    def _determine_category(self, tag_name: str) -> str:
        """Determine tag category."""
        for category, keywords in self.category_keywords.items():
            if tag_name in keywords:
                return category
        return "other"

    def _enhance_description(
        self, tag_name: str, description: str, category: str
    ) -> str:
        """Enhance tag description with standardized format."""
        parts = [f"【定義】{tag_name}"]

        # Add original description
        if description:
            parts.append(f"【說明】{description}")

        # Add category-specific enhancements
        if category == "character":
            parts.append(self._enhance_character_description(tag_name))
        elif category == "body":
            parts.append(self._enhance_body_description(tag_name))
        elif category == "clothing":
            parts.append(self._enhance_clothing_description(tag_name))
        elif category == "action":
            parts.append(self._enhance_action_description(tag_name))
        elif category == "theme":
            parts.append(self._enhance_theme_description(tag_name))

        return "\n".join(parts)

    def _enhance_character_description(self, tag_name: str) -> str:
        """Enhance character tag descriptions."""
        enhancements = {
            "蘿莉": "【視覺特徵】嬰兒肥、小骨架、未發育胸部、稚氣面容\n【年齡外觀】8-14歲",
            "少女": "【視覺特徵】青春面容、發育中身材\n【年齡外觀】14-18歲",
            "人妻": "【視覺特徵】成熟韻味、已婚氣質\n【年齡外觀】25-35歲",
            "熟女": "【視覺特徵】成熟豐滿、女性魅力\n【年齡外觀】30-45歲",
            "貓娘": "【視覺特徵】貓耳、尾巴、可愛氣質",
            "狐娘": "【視覺特徵】狐耳、尾巴、妖嬈氣質",
        }
        return enhancements.get(tag_name, "【視覺特徵】待補充")

    def _enhance_body_description(self, tag_name: str) -> str:
        """Enhance body feature descriptions."""
        enhancements = {
            "巨乳": "【視覺特徵】乳房豐滿突出、明顯乳溝\n【判斷標準】超出正常比例的大胸部",
            "貧乳": "【視覺特徵】胸部平坦、未發育或微乳\n【判斷標準】胸部尺寸較小",
            "絲襪": "【視覺特徵】腿部覆蓋絲襪、光滑質感\n【常見顏色】黑色、白色、肉色",
            "長髮": "【視覺特徵】頭髮長度超過肩膀\n【常見款式】直髮、捲髮",
        }
        return enhancements.get(tag_name, "【視覺特徵】待補充")

    def _enhance_clothing_description(self, tag_name: str) -> str:
        """Enhance clothing descriptions."""
        enhancements = {
            "校服": "【視覺特徵】學生制服、正式校園服裝",
            "泳裝": "【視覺特徵】適合游泳的服裝、暴露度高",
            "女僕": "【視覺特徵】黑白配色、圍裙、頭飾",
        }
        return enhancements.get(tag_name, "【視覺特徵】服裝類標籤")

    def _enhance_action_description(self, tag_name: str) -> str:
        """Enhance action descriptions."""
        enhancements = {
            "做愛": "【行為特徵】性器官接合、親密行為",
            "口交": "【行為特徵】口腔與性器官接觸",
            "群交": "【行為特徵】多人性行為、3人或以上",
        }
        return enhancements.get(tag_name, "【行為特徵】待補充")

    def _enhance_theme_description(self, tag_name: str) -> str:
        """Enhance theme descriptions."""
        enhancements = {
            "純愛": "【主題特徵】純真愛情、雙方自願",
            "NTR": "【主題特徵】伴侶被奪、背叛情節",
            "百合": "【主題特徵】女性間戀愛、純粹情感",
        }
        return enhancements.get(tag_name, "【主題特徵】待補充")

    def _extract_visual_keywords(
        self, tag_name: str, description: str, category: str
    ) -> List[str]:
        """Extract visual keywords from description."""
        keywords = []

        # Extract Chinese terms related to visual features
        visual_terms = re.findall(r"【視覺特徵】(.*?)(?:\n|$)", description)
        for term in visual_terms:
            keywords.extend([t.strip() for t in term.split("、") if t.strip()])

        # Add category-specific keywords
        category_keywords = {
            "character": ["年輕", "成熟", "可愛", "美麗"],
            "body": ["大", "小", "長", "短", "豐滿", "修長"],
            "clothing": ["顏色", "款式", "搭配"],
            "action": ["姿勢", "動作", "場景"],
            "theme": ["氛圍", "情感", "關係"],
        }

        keywords.extend(category_keywords.get(category, []))

        # Remove duplicates while preserving order
        seen = set()
        unique_keywords = []
        for k in keywords:
            if k not in seen and len(k) > 1:
                seen.add(k)
                unique_keywords.append(k)

        return unique_keywords[:10]  # Limit to top 10

    def _extract_negative_keywords(
        self, tag_name: str, description: str, category: str
    ) -> List[str]:
        """Extract negative keywords (what this tag is NOT)."""
        negatives = []

        # Tag-specific negative keywords
        negative_mappings = {
            "蘿莉": ["成熟", "豐滿", "人妻"],
            "人妻": ["未成年", "蘿莉", "少女"],
            "巨乳": ["平胸", "貧乳", "微乳"],
            "貧乳": ["巨乳", "爆乳", "豐滿"],
            "純愛": ["NTR", "強姦", "凌辱"],
            "長髮": ["短髮", "光頭"],
        }

        negatives = negative_mappings.get(tag_name, [])

        # Extract from description if contains "不得與" or "不應"
        if "不得與" in description or "不應" in description:
            # Simple extraction
            pass

        return negatives[:5]

    def _determine_required_confidence(self, tag_name: str, category: str) -> float:
        """Determine minimum required confidence for this tag."""
        # Sensitive tags require higher confidence
        sensitive_tags = ["蘿莉", "正太", "強姦", "NTR", "凌辱"]
        if tag_name in sensitive_tags:
            return 0.80

        # Character tags
        if category == "character":
            return 0.75

        # Default
        return 0.65

    def _get_related_tags(self, tag_name: str, category: str) -> List[str]:
        """Get related tags based on category."""
        related = []

        if category == "character":
            if tag_name in ["貓娘", "狐娘", "犬娘"]:
                related = ["獸耳", "尾巴", "獸人"]
            elif tag_name == "蘿莉":
                related = ["少女", "貧乳", "可愛"]

        elif category == "body":
            if tag_name == "巨乳":
                related = ["豐滿", "大乳房"]
            elif tag_name == "貧乳":
                related = ["小乳房", "微乳"]

        return related[:5]

    def _get_conflicting_tags(self, tag_name: str, category: str) -> List[str]:
        """Get conflicting tags."""
        conflicts = []

        conflict_mappings = {
            "蘿莉": ["人妻", "熟女", "巨乳"],
            "人妻": ["蘿莉", "正太"],
            "巨乳": ["貧乳", "爆乳"],
            "貧乳": ["巨乳"],
            "純愛": ["NTR", "強姦"],
        }

        return conflict_mappings.get(tag_name, [])[:5]

    def _generate_usage_notes(self, tag_name: str, category: str) -> str:
        """Generate usage notes for the tag."""
        notes = f"【使用建議】此為{category}類標籤。"

        if category == "character":
            notes += "根據角色外觀判斷，優先考慮視覺特徵。"
        elif category == "body":
            notes += "根據身體部位特徵判斷，注意比例和尺寸。"
        elif category == "clothing":
            notes += "根據穿著服裝判斷，注意款式和搭配。"

        return notes

    def process_tag_library(
        self, input_path: str, output_path: str, limit: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Process entire tag library.

        Args:
            input_path: Path to input JSON file
            output_path: Path to output JSON file
            limit: Optional limit on number of tags to process

        Returns:
            Statistics about the processing
        """
        # Load input
        with open(input_path, "r", encoding="utf-8") as f:
            tags = json.load(f)

        if limit:
            tags = tags[:limit]

        logger.info(f"Processing {len(tags)} tags...")

        # Process each tag
        standardized_tags = []
        stats = {
            "total": len(tags),
            "by_category": Counter(),
            "avg_description_length": 0,
            "avg_keywords": 0,
        }

        total_desc_len = 0
        total_keywords = 0

        for tag_data in tags:
            try:
                standardized = self.standardize_tag(tag_data)
                standardized_tags.append(
                    {
                        "tag_name": standardized.tag_name,
                        "description": standardized.description,
                        "enhanced_description": standardized.enhanced_description,
                        "visual_keywords": standardized.visual_keywords,
                        "negative_keywords": standardized.negative_keywords,
                        "category": standardized.category,
                        "required_confidence": standardized.required_confidence,
                        "related_tags": standardized.related_tags,
                        "conflicting_tags": standardized.conflicting_tags,
                        "usage_notes": standardized.usage_notes,
                        "version": standardized.version,
                    }
                )

                # Update stats
                stats["by_category"][standardized.category] += 1
                total_desc_len += len(standardized.enhanced_description)
                total_keywords += len(standardized.visual_keywords)

            except Exception as e:
                logger.error(f"Error processing tag {tag_data.get('tag_name')}: {e}")

        # Calculate averages
        if standardized_tags:
            stats["avg_description_length"] = total_desc_len / len(standardized_tags)
            stats["avg_keywords"] = total_keywords / len(standardized_tags)

        # Save output
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(standardized_tags, f, indent=2, ensure_ascii=False)

        logger.info(f"Standardized {len(standardized_tags)} tags")
        logger.info(f"Stats: {stats}")

        return stats


if __name__ == "__main__":
    # Test the standardizer
    print("Testing Tag Description Standardizer...")
    print("=" * 60)

    standardizer = TagDescriptionStandardizer()

    # Test with sample tags
    test_tags = [
        {"tag_name": "蘿莉", "description": "有性暗示或裸體的未成年少女外觀角色"},
        {"tag_name": "巨乳", "description": "胸部豐滿的角色"},
        {"tag_name": "貓娘", "description": "貓耳少女"},
    ]

    for tag_data in test_tags:
        print(f"\n標籤: {tag_data['tag_name']}")
        result = standardizer.standardize_tag(tag_data)
        print(f"  類別: {result.category}")
        print(f"  視覺關鍵詞: {result.visual_keywords}")
        print(f"  負面關鍵詞: {result.negative_keywords}")
        print(f"  所需置信度: {result.required_confidence}")

    print("\n" + "=" * 60)
    print("✅ Standardizer test completed!")
