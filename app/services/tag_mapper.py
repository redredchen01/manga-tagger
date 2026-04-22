"""Tag mapping service for English to Chinese tag translation.

Re-export from app.domain.tag.mapper for backward compatibility.
"""

from app.domain.tag.mapper import *  # noqa: F401, F403

__all__ = ["TagMapper", "get_tag_mapper"]
