"""Tag Recommender Service for intelligent tag suggestions.

Re-export from app.domain.tag.recommender for backward compatibility.
"""

from app.domain.tag.recommender import *  # noqa: F401, F403

__all__ = ["TagRecommendation", "TagRecommenderService", "get_tag_recommender_service"]
