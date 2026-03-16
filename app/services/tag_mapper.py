"""Tag mapping service for English to Chinese tag translation."""

import logging
import re
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)


class TagMapper:
    """Maps English tags to Chinese tags in the library."""

    def __init__(self):
        """Initialize tag mappings."""
        self.en_to_cn: Dict[str, str] = {}
        self.cn_to_en: Dict[str, str] = {}
        # False positive blocklist - short common terms that cause spurious matches
        self.blocklist: set = {
            "bra",  # Matches in "library", "ibrarian"
            "sex",  # Too common, causes false positives
            "anal",  # Too short, partial match issues
            "rape",  # Too short, partial match issues
            "to",  # Common word, causes false matches
            "in",  # Common word, causes false matches
            "or",  # Common word, causes false matches
            "on",  # Common word, causes false matches
            "at",  # Common word, causes false matches
            "is",  # Common word, causes false matches
            "flat",  # Too generic
            "round",  # Too generic
            "open",  # Too generic
            "close",  # Too generic
        }
        self._build_mappings()

    def _build_mappings(self):
        """Build English to Chinese tag mappings."""
        # Character types
        character_mappings = {
            "loli": "蘿莉",
            "shota": "正太",
            "teen": "少女",
            "young girl": "少女",
            "young": "少女",
            "mature": "熟女",
            "milf": "人妻",
            "catgirl": "貓娘",
            "cat girl": "貓娘",
            "doggirl": "犬娘",
            "dog girl": "犬娘",
            "foxgirl": "狐娘",
            "fox girl": "狐娘",
            "elf": "精靈",
            "demon": "惡魔",
            "angel": "天使",
            "vampire": "吸血鬼",
            "monster girl": "魔物娘",
            "futanari": "扶他",
            "futa": "扶他",
            "shemale": "人妖",
            "trap": "偽娘",
            "boy": "少年",
            "girl": "少女",
        }

        # Clothing - only mappings for tags that exist in library
        clothing_mappings = {
            "school uniform": "女生制服",  # Map to existing tag
            "swimsuit": "泳裝",
            "bikini": "比基尼",
            "lingerie": "情趣內衣",  # Map to existing tag
            "kimono": "和服",
            "maid outfit": "女僕裝",
            "maid": "女僕裝",
            "nurse": "護士裝",
            "police uniform": "警服",
            "bunny suit": "兔女郎",
            "bunny girl": "兔女郎",
            "dress": "歌德蘿莉裝",  # Map to existing tag (closest match)
            "skirt": "熱褲",  # Map to existing tag (closest match)
            "panties": "內褲",
            "bra": "緊身胸衣",  # Map to existing tag
            "shirt": "緊身衣",  # Map to existing tag
            "glasses": "眼鏡",
        }

        # Body features - only mappings for tags that exist in library
        body_mappings = {
            "flat chest": "貧乳",
            "small breasts": "貧乳",
            "large breasts": "巨乳",
            "big breasts": "巨乳",
            "huge breasts": "巨乳",
            "huge_breasts": "巨乳",
            "large_breasts": "巨乳",
            "stockings": "長筒襪",  # Map to existing tag
            "pantyhose": "連褲襪",  # Map to existing tag
            "knee high socks": "過膝襪",
            "knee socks": "過膝襪",
            "tattoo": "紋身",
            "long hair": "超長髮",  # Map to existing tag (closest match)
            "short hair": "精靈短髮",  # Map to existing tag
            "twintails": "雙馬尾",
            "ponytail": "單馬尾",
            "heterochromia": "異色瞳",
        }

        # Actions
        action_mappings = {
            "sex": "做愛",
            "vaginal": "正常位",
            "oral": "口交",
            "blowjob": "口交",
            "paizuri": "乳交",
            "titfuck": "乳交",
            "handjob": "手淫",
            "anal": "肛交",
            "bondage": "綁架",
            "bdsm": "BDSM",
            "tentacles": "觸手",
            "rape": "強姦",
            "prostitution": "賣淫",
            "masturbation": "自慰",
            "group sex": "群交",
            "orgy": "群交",
            "threesome": "3P",
        }

        # Themes
        theme_mappings = {
            "vanilla": "純愛",
            "pure love": "純愛",
            "ntr": "NTR",
            "netorare": "NTR",
            "cheating": "外遇",
            "incest": "亂倫",
            "yuri": "百合",
            "yaoi": "耽美",
            "harem": "後宮",
            "school life": "學園",
            "romance": "戀愛",
            "comedy": "喜劇",
            "drama": "劇情",
            "training": "調教",
            "slave": "奴隸",
        }

        # Combine all mappings
        all_mappings = {}
        all_mappings.update(character_mappings)
        all_mappings.update(clothing_mappings)
        all_mappings.update(body_mappings)
        all_mappings.update(action_mappings)
        all_mappings.update(theme_mappings)

        self.en_to_cn = all_mappings
        self.cn_to_en = {v: k for k, v in all_mappings.items()}

        logger.info(f"Built {len(self.en_to_cn)} tag mappings")

    def to_chinese(self, english_tag: str) -> Optional[str]:
        """Convert English tag to Chinese."""
        # Normalize: underscore -> space, lowercase
        tag_normalized = english_tag.lower().strip().replace("_", " ")

        # PRECISION: Check blocklist first
        if tag_normalized in self.blocklist:
            return None

        # Direct match
        if tag_normalized in self.en_to_cn:
            return self.en_to_cn[tag_normalized]

        # Partial match - STRICT: min 5 chars to avoid false positives
        for en_tag, cn_tag in self.en_to_cn.items():
            if len(en_tag) >= 5 and len(tag_normalized) >= 5:
                if en_tag in tag_normalized or tag_normalized in en_tag:
                    return cn_tag
            # Word boundary matching
            if len(en_tag) >= 5:
                pattern = r"\b" + re.escape(en_tag) + r"\b"
                if re.search(pattern, tag_normalized):
                    return cn_tag

        # Language-specific mapping fixes for library tags
        fallback_mappings = {
            "mature": "熟女",
            "young girl": "少女",
            "teen": "少女",
            "shota": "少年",
            "teenager": "少女",
            "milf": "人妻",
            "sex": "做愛",
            "oral": "口交",
            "paizuri": "乳交",
            "handjob": "手淫",
            "anal": "肛交",
            "bondage": "綁架",
            "tentacles": "觸手",
            "rape": "強姦",
            "vanilla": "純愛",
            "ntr": "NTR",
            "incest": "亂倫",
            "yuri": "百合",
            "yaoi": "耽美",
            "harem": "後宮",
            "school life": "學園",
        }

        if tag_normalized in fallback_mappings:
            return fallback_mappings[tag_normalized]

        return None

    def to_english(self, chinese_tag: str) -> Optional[str]:
        """Convert Chinese tag to English."""
        return self.cn_to_en.get(chinese_tag)

    def map_keywords(self, keywords: List[str]) -> List[Tuple[str, Optional[str]]]:
        """Map a list of English keywords to Chinese tags.

        Returns list of (original_keyword, chinese_tag) tuples.
        """
        results = []
        for kw in keywords:
            cn_tag = self.to_chinese(kw)
            results.append((kw, cn_tag))
        return results

    def get_all_mappings(self) -> Dict[str, str]:
        """Get all English to Chinese mappings."""
        return self.en_to_cn.copy()


# Singleton instance
_tag_mapper: Optional[TagMapper] = None


def get_tag_mapper() -> TagMapper:
    """Get or create tag mapper singleton."""
    global _tag_mapper
    if _tag_mapper is None:
        _tag_mapper = TagMapper()
    return _tag_mapper
