"""Tag domain package.

Re-exports all tag-related services for convenient importing.
"""

from app.domain.tag.library import TagLibraryService, get_tag_library_service
from app.domain.tag.mapper import TagMapper, get_tag_mapper
from app.domain.tag.parser import (
    ACTION_KEYWORDS,
    BODY_KEYWORDS,
    CHAR_KEYWORDS,
    CLOTHING_KEYWORDS,
    STOP_WORDS,
    THEME_KEYWORDS,
    extract_tags_from_description,
    extract_tags_from_reasoning,
    get_fallback_metadata,
    get_mock_metadata,
    parse_response,
)
from app.domain.tag.recommender import (
    TagRecommendation,
    TagRecommenderService,
    get_tag_recommender_service,
)

__all__ = [
    # Mapper
    "TagMapper",
    "get_tag_mapper",
    # Parser
    "parse_response",
    "extract_tags_from_description",
    "extract_tags_from_reasoning",
    "get_fallback_metadata",
    "get_mock_metadata",
    "CHAR_KEYWORDS",
    "CLOTHING_KEYWORDS",
    "BODY_KEYWORDS",
    "ACTION_KEYWORDS",
    "THEME_KEYWORDS",
    "STOP_WORDS",
    # Library
    "TagLibraryService",
    "get_tag_library_service",
    # Recommender
    "TagRecommendation",
    "TagRecommenderService",
    "get_tag_recommender_service",
]
