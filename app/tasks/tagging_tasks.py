"""Background tagging tasks for RQ worker."""

import asyncio
import json
import logging

from app.tasks.worker import get_redis_connection
from app.services.pipeline import run_tagging_pipeline

logger = logging.getLogger(__name__)


def process_tagging_job(
    job_id: str,
    image_bytes_hex: str,
    top_k: int,
    confidence_threshold: float,
) -> dict:
    """Background task for tagging pipeline.

    This function runs synchronously but calls the async tagging pipeline.
    It stores results in Redis for later retrieval via the jobs endpoint.

    Args:
        job_id: Unique job identifier
        image_bytes_hex: Image data as hex string
        top_k: Number of tags to return
        confidence_threshold: Minimum confidence threshold

    Returns:
        Dict containing the tagging result
    """
    conn = get_redis_connection()

    try:
        # Convert hex string back to bytes
        image_bytes = bytes.fromhex(image_bytes_hex)

        # Set initial status
        conn.setex(f"job:{job_id}:status", 3600, "processing")

        # Run the async pipeline
        result = asyncio.run(
            run_tagging_pipeline(
                image_bytes=image_bytes,
                top_k=top_k,
                confidence_threshold=confidence_threshold,
            )
        )

        # Store result
        result_dict = result.model_dump() if hasattr(result, "model_dump") else result.dict()
        conn.setex(f"job:{job_id}:result", 3600, json.dumps(result_dict))
        conn.setex(f"job:{job_id}:status", 3600, "completed")

        return result_dict

    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}")
        conn.setex(f"job:{job_id}:status", 3600, f"failed:{str(e)}")
        raise
