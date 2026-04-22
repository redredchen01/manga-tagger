"""Tag parsing utilities for VLM responses.

Re-export from app.domain.tag.parser for backward compatibility.
"""

from app.domain.tag.parser import (  # noqa: F401
    ACTION_KEYWORDS,
    BODY_KEYWORDS,
    CHAR_KEYWORDS,
    CHARACTER_KEYWORDS,
    CLOTHING_KEYWORDS,
    THEME_KEYWORDS,
    STOP_WORDS,
    extract_tags_from_description,
    extract_tags_from_reasoning,
    get_fallback_metadata,
    get_mock_metadata,
    parse_response,
)

__all__ = [
    "parse_response",
    "extract_tags_from_description",
    "extract_tags_from_reasoning",
    "get_fallback_metadata",
    "get_mock_metadata",
    "CHARACTER_KEYWORDS",
    "CLOTHING_KEYWORDS",
    "BODY_KEYWORDS",
    "ACTION_KEYWORDS",
    "THEME_KEYWORDS",
    "STOP_WORDS",
]
