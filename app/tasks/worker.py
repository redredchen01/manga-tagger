"""RQ worker setup for background task processing."""

import redis
from rq import Queue, Worker

from app.config import settings


def get_redis_connection():
    """Get Redis connection for RQ."""
    return redis.Redis(
        host=settings.REDIS_HOST,
        port=settings.REDIS_PORT,
        db=settings.REDIS_DB,
        password=settings.REDIS_PASSWORD,
    )


def get_queue(name: str = "default") -> Queue:
    """Get or create an RQ queue.

    Args:
        name: Queue name (default: "default")

    Returns:
        RQ Queue instance
    """
    return Queue(name, connection=get_redis_connection())


def run_worker():
    """Run the RQ worker to process background jobs."""
    conn = get_redis_connection()
    worker = Worker(["default"], connection=conn)
    worker.work()
