"""Tag Library Migration Script

Migrates existing tags.json to enhanced format with visual_cues,
related_tags, negative_cues, aliases, and confidence_boost.
"""

import json
import logging
import re
from pathlib import Path
from typing import List, Dict, Any, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TagMigrator:
    """Migrates tag library to enhanced format."""

    # Character type keywords for categorization
    CHARACTER_KEYWORDS = [
        "蘿莉", "少女", "人妻", "熟女", "御姐", "正太", "少年",
        "貓娘", "犬娘", "狐娘", "獸人", "精靈", "魔物", "機娘",
        "天使", "惡魔", "幽靈", "喪屍", "喪尸", "機械人",
    ]

    # Clothing keywords
    CLOTHING_KEYWORDS = [
        "校服", "泳裝", "内衣", "和服", "女僕", "護士", "警察",
        "體操服", "兔女郎", "旗袍", "巫女", "婚纱", "西裝",
    ]

    # Body feature keywords
    BODY_KEYWORDS = [
        "巨乳", "貧乳", "長腿", "絲襪", "吊帶襪", "過膝襪",
        "裸足", "眼鏡", "義眼", "義肢", "疤痕", "紋身",
    ]

    # Action keywords
    ACTION_KEYWORDS = [
        "做愛", "口交", "手淫", "乳交", "足交", "肛交", "群交",
        "綁架", "監禁", "調教", "陵辱", "強姦", "痴漢",
    ]

    # Theme keywords
    THEME_KEYWORDS = [
        "純愛", "NTR", "凌辱", "調教", "奴隸", "主從", "百合",
        "耽美", "亂倫", "師生", "職場", "近親",
    ]

    # Common visual cue patterns
    VISUAL_CUE_PATTERNS = {
        "ears": ["耳", "ears"],
        "tail": ["尾", "tail"],
        "wings": ["翅膀", "wing"],
        "horns": ["角", "horn"],
        "eyes": ["瞳", "眼", "eyes"],
        "hair": ["髮", "发", "hair"],
        "skin": ["皮膚", "皮肤", "skin"],
        "body": ["身體", "身体", "body"],
        "clothing": ["服裝", "服装", "clothing"],
        "accessories": ["配飾", "配饰", "accessories"],
    }

    # Common negative cue patterns
    NEGATIVE_CUE_PATTERNS = {
        "human": ["人類", "人类", "human"],
        "animal": ["動物", "动物", "animal"],
        "clothed": ["穿著", "穿着", "clothed"],
        "naked": ["裸體", "裸体", "naked"],
    }

    def __init__(self):
        """Initialize the migrator."""
        self.category_map = {
            "character": self.CHARACTER_KEYWORDS,
            "clothing": self.CLOTHING_KEYWORDS,
            "body": self.BODY_KEYWORDS,
            "action": self.ACTION_KEYWORDS,
            "theme": self.THEME_KEYWORDS,
        }

    def migrate_tags(self, input_path: str, output_path: str) -> List[Dict[str, Any]]:
        """
        Migrate tags from old format to enhanced format.

        Args:
            input_path: Path to original tags.json
            output_path: Path to save enhanced tags.json

        Returns:
            List of enhanced tag dictionaries
        """
        logger.info(f"Loading tags from {input_path}")

        # Load original tags
        with open(input_path, "r", encoding="utf-8") as f:
            original_tags = json.load(f)

        logger.info(f"Loaded {len(original_tags)} tags")

        # Migrate each tag
        enhanced_tags = []
        for tag_data in original_tags:
            tag_name = tag_data.get("tag_name", "")
            description = tag_data.get("description", "")

            if not tag_name:
                logger.warning(f"Skipping tag without tag_name: {tag_data}")
                continue

            enhanced_tag = {
                "tag": tag_name,
                "category": self.auto_categorize(tag_name, description),
                "description": description,
                "visual_cues": self.extract_visual_cues(description),
                "related_tags": self.infer_related_tags(tag_name, description),
                "negative_cues": self.generate_negative_cues(tag_name, description),
                "aliases": self.generate_aliases(tag_name, description),
                "confidence_boost": self.calculate_confidence_boost(tag_name, description),
            }

            enhanced_tags.append(enhanced_tag)

        # Save enhanced tags
        logger.info(f"Saving {len(enhanced_tags)} enhanced tags to {output_path}")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(enhanced_tags, f, ensure_ascii=False, indent=2)

        logger.info("Migration completed successfully")
        return enhanced_tags

    def auto_categorize(self, tag_name: str, description: str) -> str:
        """
        Automatically categorize a tag based on name and description.

        Args:
            tag_name: The tag name
            description: The tag description

        Returns:
            Category string: character/clothing/body/action/theme/other
        """
        text = f"{tag_name} {description}".lower()

        # Check each category
        for category, keywords in self.category_map.items():
            for keyword in keywords:
                if keyword in text:
                    return category

        return "other"

    def extract_visual_cues(self, description: str) -> List[str]:
        """
        Extract visual cues from description.

        Args:
            description: The tag description

        Returns:
            List of visual cue keywords
        """
        visual_cues = []

        # Extract body parts and features
        body_parts = [
            "貓耳", "貓尾", "貓瞳", "鬍鬚", "翅膀", "角", "尾巴",
            "眼睛", "瞳孔", "髮型", "頭髮", "皮膚", "身體", "胸部",
            "腿部", "手臂", "手", "腳", "臉", "嘴", "鼻子", "耳朵",
        ]

        for part in body_parts:
            if part in description:
                visual_cues.append(part)

        # Extract clothing items
        clothing_items = [
            "校服", "泳裝", "内衣", "和服", "女僕裝", "護士服", "警察制服",
            "體操服", "兔女郎裝", "旗袍", "巫女服", "婚纱", "西裝",
            "裙子", "褲子", "襯衫", "外套", "大衣", "連衣裙",
        ]

        for item in clothing_items:
            if item in description:
                visual_cues.append(item)

        # Extract accessories
        accessories = [
            "眼鏡", "項鍊", "耳環", "戒指", "手鐲", "髮飾", "帽子",
            "領帶", "領結", "手套", "襪子", "鞋子", "靴子",
        ]

        for accessory in accessories:
            if accessory in description:
                visual_cues.append(accessory)

        return list(set(visual_cues))

    def infer_related_tags(self, tag_name: str, description: str) -> List[str]:
        """
        Infer related tags based on tag name and description.

        Args:
            tag_name: The tag name
            description: The tag description

        Returns:
            List of related tag names
        """
        related_tags = []

        # Character type relationships
        character_relations = {
            "貓娘": ["動物娘", "獸耳", "貓耳", "犬娘", "狐娘"],
            "犬娘": ["動物娘", "獸耳", "狗耳", "貓娘", "狐娘"],
            "狐娘": ["動物娘", "獸耳", "狐耳", "貓娘", "犬娘"],
            "蘿莉": ["年輕女孩", "小女孩", "未成年", "貧乳"],
            "人妻": ["熟女", "御姐", "成熟女性"],
            "精靈": ["長耳", "魔法", "奇幻"],
            "天使": ["翅膀", "光環", "神聖"],
            "惡魔": ["角", "翅膀", "黑暗"],
        }

        # Check if tag_name is in character_relations
        if tag_name in character_relations:
            related_tags.extend(character_relations[tag_name])

        # Extract related terms from description
        related_terms = [
            "動物娘", "獸耳", "貓耳", "犬耳", "狐耳", "長耳",
            "年輕", "成熟", "未成年", "成年",
            "魔法", "奇幻", "神聖", "黑暗",
        ]

        for term in related_terms:
            if term in description and term not in related_tags:
                related_tags.append(term)

        return list(set(related_tags))

    def generate_negative_cues(self, tag_name: str, description: str) -> List[str]:
        """
        Generate negative cues (exclusion criteria) for a tag.

        Args:
            tag_name: The tag name
            description: The tag description

        Returns:
            List of negative cue keywords
        """
        negative_cues = []

        # Common negative cue patterns
        negative_patterns = {
            "貓娘": ["純人類", "無動物特徵", "無貓耳", "無貓尾"],
            "犬娘": ["純人類", "無動物特徵", "無狗耳", "無狗尾"],
            "狐娘": ["純人類", "無動物特徵", "無狐耳", "無狐尾"],
            "蘿莉": ["成熟女性", "巨乳", "豐滿身材", "成年女性"],
            "人妻": ["未成年", "年輕女孩", "小女孩"],
            "精靈": ["人類", "無長耳", "無魔法"],
            "天使": ["惡魔", "無翅膀", "無光環"],
            "惡魔": ["天使", "無角", "無翅膀"],
        }

        # Check if tag_name is in negative_patterns
        if tag_name in negative_patterns:
            negative_cues.extend(negative_patterns[tag_name])

        # Extract negative terms from description
        negative_terms = [
            "不應", "不得", "不", "無", "非", "排除", "不是",
        ]

        for term in negative_terms:
            if term in description:
                # Try to extract the full negative phrase
                pattern = rf"{term}([^，。；！？\n]+)"
                matches = re.findall(pattern, description)
                for match in matches:
                    if len(match) > 1 and match not in negative_cues:
                        negative_cues.append(match.strip())

        return list(set(negative_cues))

    def generate_aliases(self, tag_name: str, description: str) -> List[str]:
        """
        Generate aliases for a tag.

        Args:
            tag_name: The tag name
            description: The tag description

        Returns:
            List of alias strings
        """
        aliases = []

        # Common alias mappings
        alias_mappings = {
            "貓娘": ["catgirl", "nekomimi", "貓耳娘"],
            "犬娘": ["doggirl", "dog girl", "犬耳娘"],
            "狐娘": ["foxgirl", "fox girl", "狐耳娘", "kitsune"],
            "蘿莉": ["loli", "little girl", "young girl"],
            "人妻": ["milf", "mature woman", "wife"],
            "精靈": ["elf", "elven", "fairy"],
            "天使": ["angel", "angelic"],
            "惡魔": ["demon", "devil", "succubus"],
            "校服": ["school uniform", "制服", "學生服"],
            "泳裝": ["swimsuit", "swimwear", "bathing suit"],
            "眼鏡": ["glasses", "eyewear", "spectacles"],
        }

        # Check if tag_name is in alias_mappings
        if tag_name in alias_mappings:
            aliases.extend(alias_mappings[tag_name])

        # Extract English terms from description
        english_pattern = r'[a-zA-Z\s]{3,}'
        english_matches = re.findall(english_pattern, description)
        for match in english_matches:
            alias = match.strip()
            if alias and alias not in aliases and alias != tag_name:
                aliases.append(alias)

        return list(set(aliases))

    def calculate_confidence_boost(self, tag_name: str, description: str) -> float:
        """
        Calculate confidence boost factor for a tag.

        Args:
            tag_name: The tag name
            description: The tag description

        Returns:
            Confidence boost factor (default 1.0)
        """
        # High confidence tags (well-defined, common)
        high_confidence_tags = [
            "貓娘", "犬娘", "狐娘", "蘿莉", "人妻", "校服", "泳裝",
            "眼鏡", "巨乳", "貧乳", "長腿", "絲襪",
        ]

        # Medium confidence tags (less common, more specific)
        medium_confidence_tags = [
            "精靈", "天使", "惡魔", "機娘", "幽靈", "喪屍",
        ]

        if tag_name in high_confidence_tags:
            return 1.1
        elif tag_name in medium_confidence_tags:
            return 1.05
        else:
            return 1.0


def main():
    """Main entry point for the migration script."""
    import argparse

    parser = argparse.ArgumentParser(description="Migrate tag library to enhanced format")
    parser.add_argument(
        "--input",
        default="./data/tags.json",
        help="Path to original tags.json (default: ./data/tags.json)",
    )
    parser.add_argument(
        "--output",
        default="./data/tags_enhanced.json",
        help="Path to save enhanced tags.json (default: ./data/tags_enhanced.json)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run migration without saving output",
    )

    args = parser.parse_args()

    # Check if input file exists
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file not found: {args.input}")
        return

    # Create migrator
    migrator = TagMigrator()

    # Run migration
    if args.dry_run:
        logger.info("Running in dry-run mode (no output will be saved)")
        with open(input_path, "r", encoding="utf-8") as f:
            original_tags = json.load(f)
        logger.info(f"Would migrate {len(original_tags)} tags")
    else:
        migrator.migrate_tags(args.input, args.output)


if __name__ == "__main__":
    main()
