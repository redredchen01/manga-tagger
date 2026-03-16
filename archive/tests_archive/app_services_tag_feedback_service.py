"""Tag Feedback Service for collecting and managing user feedback.

This service handles:
- Storing user feedback on tag predictions
- Computing tag accuracy statistics
- Managing dynamic tag weight adjustments based on feedback
"""

import json
import uuid
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class TagFeedbackService:
    """Service for managing tag feedback and computing accuracy statistics."""

    def __init__(self, feedback_path: Optional[str] = None):
        """Initialize the feedback service.

        Args:
            feedback_path: Path to store feedback data (default: ./data/feedback.json)
        """
        self.feedback_path = (
            Path(feedback_path) if feedback_path else Path("./data/feedback.json")
        )
        self.feedback_path.parent.mkdir(parents=True, exist_ok=True)

        # In-memory cache
        self._feedback_data: Dict[str, Any] = {"feedback": [], "tag_weights": {}}

        # Load existing feedback
        self._load_feedback()

    def _load_feedback(self):
        """Load feedback data from file."""
        try:
            if self.feedback_path.exists():
                with open(self.feedback_path, "r", encoding="utf-8") as f:
                    self._feedback_data = json.load(f)
                logger.info(
                    f"Loaded {len(self._feedback_data.get('feedback', []))} feedback entries"
                )
            else:
                self._feedback_data = {"feedback": [], "tag_weights": {}}
                self._save_feedback()
        except Exception as e:
            logger.error(f"Failed to load feedback data: {e}")
            self._feedback_data = {"feedback": [], "tag_weights": {}}

    def _save_feedback(self):
        """Save feedback data to file."""
        try:
            with open(self.feedback_path, "w", encoding="utf-8") as f:
                json.dump(self._feedback_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Failed to save feedback data: {e}")

    def compute_image_hash(self, image_bytes: bytes) -> str:
        """Compute hash for image to track feedback.

        Args:
            image_bytes: Image data

        Returns:
            SHA256 hash of the image
        """
        return hashlib.sha256(image_bytes).hexdigest()[:16]

    def add_feedback(
        self,
        image_hash: str,
        request_id: str,
        feedback_items: List[Dict[str, Any]],
        user_notes: Optional[str] = None,
        predicted_tags: Optional[List[str]] = None,
    ) -> str:
        """Add user feedback for a tag prediction.

        Args:
            image_hash: Hash of the original image
            request_id: ID of the original tagging request
            feedback_items: List of feedback items
            user_notes: Optional user notes
            predicted_tags: Tags that were predicted for this image

        Returns:
            Unique feedback ID
        """
        feedback_id = str(uuid.uuid4())[:8]

        feedback_entry = {
            "id": feedback_id,
            "image_hash": image_hash,
            "request_id": request_id,
            "timestamp": datetime.now().isoformat(),
            "feedback_items": feedback_items,
            "user_notes": user_notes,
            "predicted_tags": predicted_tags or [],
        }

        # Add to feedback list
        self._feedback_data["feedback"].append(feedback_entry)

        # Update tag weights based on feedback
        self._update_tag_weights(feedback_items)

        # Save to file
        self._save_feedback()

        logger.info(f"Added feedback {feedback_id} for image {image_hash}")

        return feedback_id

    def _update_tag_weights(self, feedback_items: List[Dict[str, Any]]):
        """Update tag weights based on feedback.

        Positive feedback increases weight, negative decreases it.

        Args:
            feedback_items: List of feedback items
        """
        tag_weights = self._feedback_data.get("tag_weights", {})

        for item in feedback_items:
            tag = item.get("tag")
            feedback_type = item.get("feedback_type")

            if not tag:
                continue

            # Initialize tag weight if not exists
            if tag not in tag_weights:
                tag_weights[tag] = {
                    "positive": 0,
                    "negative": 0,
                    "missing": 0,
                    "correction": 0,
                    "weight": 1.0,  # Default weight
                }

            # Update counts
            if feedback_type == "positive":
                tag_weights[tag]["positive"] += 1
            elif feedback_type == "negative":
                tag_weights[tag]["negative"] += 1
            elif feedback_type == "missing":
                tag_weights[tag]["missing"] += 1
            elif feedback_type == "correction":
                tag_weights[tag]["correction"] += 1
                # Track correction mappings
                suggested = item.get("suggested_tag")
                if suggested:
                    if "corrections" not in tag_weights[tag]:
                        tag_weights[tag]["corrections"] = {}
                    tag_weights[tag]["corrections"][suggested] = (
                        tag_weights[tag]["corrections"].get(suggested, 0) + 1
                    )

            # Recalculate weight
            pos = tag_weights[tag]["positive"]
            neg = tag_weights[tag]["negative"]
            total = pos + neg

            if total > 0:
                # Weight between 0.5 and 1.5 based on accuracy
                accuracy = pos / total
                tag_weights[tag]["weight"] = 0.5 + accuracy  # Range: 0.5 to 1.5

        self._feedback_data["tag_weights"] = tag_weights

    def get_tag_weight(self, tag: str) -> float:
        """Get weight adjustment for a specific tag.

        Args:
            tag: Tag name

        Returns:
            Weight multiplier (default 1.0)
        """
        tag_weights = self._feedback_data.get("tag_weights", {})
        return tag_weights.get(tag, {}).get("weight", 1.0)

    def get_tag_weights_batch(self, tags: List[str]) -> Dict[str, float]:
        """Get weight adjustments for multiple tags.

        Args:
            tags: List of tag names

        Returns:
            Dict mapping tag names to weights
        """
        tag_weights = self._feedback_data.get("tag_weights", {})
        return {tag: tag_weights.get(tag, {}).get("weight", 1.0) for tag in tags}

    def get_stats(self) -> Dict[str, Any]:
        """Get feedback statistics.

        Returns:
            Dict with statistics
        """
        feedback_list = self._feedback_data.get("feedback", [])

        if not feedback_list:
            return {
                "total_feedback": 0,
                "positive_count": 0,
                "negative_count": 0,
                "missing_count": 0,
                "correction_count": 0,
                "tag_accuracy": {},
                "overall_accuracy": 0.0,
            }

        # Count feedback types
        positive_count = 0
        negative_count = 0
        missing_count = 0
        correction_count = 0

        # Track per-tag accuracy
        tag_stats: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {"correct": 0, "incorrect": 0}
        )

        for entry in feedback_list:
            for item in entry.get("feedback_items", []):
                feedback_type = item.get("feedback_type")
                tag = item.get("tag")

                if feedback_type == "positive":
                    positive_count += 1
                    if tag:
                        tag_stats[tag]["correct"] += 1
                elif feedback_type == "negative":
                    negative_count += 1
                    if tag:
                        tag_stats[tag]["incorrect"] += 1
                elif feedback_type == "missing":
                    missing_count += 1
                elif feedback_type == "correction":
                    correction_count += 1

        # Calculate per-tag accuracy
        tag_accuracy = {}
        for tag, stats in tag_stats.items():
            total = stats["correct"] + stats["incorrect"]
            if total > 0:
                tag_accuracy[tag] = stats["correct"] / total

        # Calculate overall accuracy
        total_rated = positive_count + negative_count
        overall_accuracy = positive_count / total_rated if total_rated > 0 else 0.0

        return {
            "total_feedback": len(feedback_list),
            "positive_count": positive_count,
            "negative_count": negative_count,
            "missing_count": missing_count,
            "correction_count": correction_count,
            "tag_accuracy": tag_accuracy,
            "overall_accuracy": round(overall_accuracy, 3),
        }

    def get_recent_feedback(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent feedback entries.

        Args:
            limit: Maximum number of entries to return

        Returns:
            List of recent feedback entries
        """
        feedback_list = self._feedback_data.get("feedback", [])
        return sorted(
            feedback_list, key=lambda x: x.get("timestamp", ""), reverse=True
        )[:limit]

    def get_correction_suggestions(self, tag: str) -> Dict[str, int]:
        """Get common correction suggestions for a tag.

        Args:
            tag: Tag to get corrections for

        Returns:
            Dict mapping suggested tags to count
        """
        tag_weights = self._feedback_data.get("tag_weights", {})
        tag_data = tag_weights.get(tag, {})
        return tag_data.get("corrections", {})

    def reset_stats(self):
        """Reset all feedback data."""
        self._feedback_data = {"feedback": [], "tag_weights": {}}
        self._save_feedback()
        logger.info("Feedback statistics reset")


# Singleton instance
_feedback_service: Optional[TagFeedbackService] = None


def get_feedback_service() -> TagFeedbackService:
    """Get or create feedback service singleton."""
    global _feedback_service
    if _feedback_service is None:
        _feedback_service = TagFeedbackService()
    return _feedback_service
