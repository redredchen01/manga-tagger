"""Tag Alias Service for English-Chinese tag mapping.

Provides English to Chinese tag alias lookup for VLM output matching.
"""

import json
import logging
from typing import Dict, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class TagAliasService:
    """Service for managing and looking up tag aliases."""

    # Blocklist of words that should NOT be mapped (noise words)
    BLOCKLIST = {
        "bra",
        "sex",
        "anal",
        "rape",
        "to",
        "in",
        "on",
        "at",
        "is",
        "flat",
        "round",
        "open",
        "close",
        "big",
        "small",
        "long",
        "short",
        "the",
        "a",
        "an",
        "and",
        "or",
        "but",
        "if",
        "then",
        "else",
        "with",
        "without",
        "from",
        "by",
        "for",
        "of",
        "as",
        "that",
        "this",
        "it",
        "be",
        "are",
        "was",
        "were",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "must",
        "can",
        "need",
        "dare",
        "ought",
        "used",
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
        "ten",
        "zero",
    }

    def __init__(self, alias_path: Optional[str] = None):
        """Initialize tag alias service.

        Args:
            alias_path: Path to tag aliases JSON file
        """
        self.alias_path = alias_path or "./data/tag_aliases.json"
        self.aliases: Dict[str, List[str]] = {}
        self._load_aliases()

    def _load_aliases(self):
        """Load aliases from JSON file."""
        try:
            path = Path(self.alias_path)
            if not path.exists():
                logger.warning(f"Tag aliases not found at {self.alias_path}")
                return

            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.aliases = data.get("aliases", {})

            logger.info(f"Loaded {len(self.aliases)} English alias entries")

        except Exception as e:
            logger.error(f"Failed to load tag aliases: {e}")
            self.aliases = {}

    def to_chinese(self, english_keyword: str) -> Optional[str]:
        """Convert English keyword to Chinese tag(s).

        Args:
            english_keyword: English keyword from VLM

        Returns:
            Chinese tag name if found, None otherwise
        """
        key = english_keyword.lower().strip()

        # Blocklist check - reject noise words
        if key in self.BLOCKLIST:
            return None

        if key in self.aliases:
            # Return first (most common) Chinese translation
            return self.aliases[key][0]

        return None

    def to_chinese_all(self, english_keyword: str) -> List[str]:
        """Get all Chinese translations for an English keyword.

        Args:
            english_keyword: English keyword from VLM

        Returns:
            List of Chinese tag names
        """
        key = english_keyword.lower().strip()

        # Blocklist check - reject noise words
        if key in self.BLOCKLIST:
            return []

        return self.aliases.get(key, [])

    def get_english_variants(self, chinese_tag: str) -> List[str]:
        """Get all English variants for a Chinese tag.

        Args:
            chinese_tag: Chinese tag name

        Returns:
            List of English keywords that map to this Chinese tag
        """
        variants = []
        for eng, chn_list in self.aliases.items():
            if chinese_tag in chn_list:
                variants.append(eng)
        return variants

    def is_alias(self, keyword: str) -> bool:
        """Check if a keyword is a known alias.

        Args:
            keyword: Keyword to check

        Returns:
            True if keyword is a known alias
        """
        key = keyword.lower().strip()

        # Also check blocklist
        if key in self.BLOCKLIST:
            return False

        return key in self.aliases

    def match_keyword(self, keyword: str, chinese_tags: List[str]) -> List[str]:
        """Match an English keyword against a list of Chinese tags.

        Args:
            keyword: English keyword from VLM
            chinese_tags: List of Chinese tag names to match against

        Returns:
            List of matching Chinese tags
        """
        # Blocklist check
        key = keyword.lower().strip()
        if key in self.BLOCKLIST:
            return []

        matches = []
        chinese_map = {tag.lower(): tag for tag in chinese_tags}

        # Get all Chinese translations for this keyword
        chinese_versions = self.to_chinese_all(keyword)

        for cn in chinese_versions:
            cn_lower = cn.lower()
            if cn_lower in chinese_map:
                matches.append(chinese_map[cn_lower])

        return matches

    def expand_description(self, description: str) -> str:
        """Expand a description by adding Chinese equivalents.

        Args:
            description: English description from VLM

        Returns:
            Description with Chinese tag equivalents appended
        """
        words = description.replace(",", " ").split()
        expanded_parts = []

        for word in words:
            word_lower = word.lower().strip()

            # Skip blocklist words
            if word_lower in self.BLOCKLIST:
                continue

            expanded_parts.append(word)
            chinese = self.to_chinese(word)
            if chinese and chinese not in expanded_parts:
                expanded_parts.append(chinese)

        return " ".join(expanded_parts)


# Singleton instance
_tag_alias_service: Optional[TagAliasService] = None


def get_tag_alias_service() -> TagAliasService:
    """Get or create TagAliasService singleton."""
    global _tag_alias_service
    if _tag_alias_service is None:
        _tag_alias_service = TagAliasService()
    return _tag_alias_service
