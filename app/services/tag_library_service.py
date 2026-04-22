"""Tag Library Service for managing and matching tags.

Re-export from app.domain.tag.library for backward compatibility.
"""

from app.domain.tag.library import *  # noqa: F401, F403

__all__ = ["TagLibraryService", "get_tag_library_service"]
