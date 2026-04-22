"""RQ background tasks for Manga Tagger."""

from app.tasks.tagging_tasks import process_tagging_job

__all__ = ["process_tagging_job"]
