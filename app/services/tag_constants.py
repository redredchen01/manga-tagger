"""Constants for tag recommender service.

Re-exports from app.domain.tag.parser for backward compatibility.
"""

from app.domain.tag.parser import (  # noqa: F401
    BODY_KEYWORDS,
    CHAR_KEYWORDS,
    CLOTHING_KEYWORDS,
    STOP_WORDS,
)

__all__ = ["CHAR_KEYWORDS", "CLOTHING_KEYWORDS", "BODY_KEYWORDS", "STOP_WORDS"]
