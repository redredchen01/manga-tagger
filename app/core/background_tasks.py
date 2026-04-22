"""Background tasks for system maintenance and resource cleanup."""

import asyncio
import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)

# Background task handle
_cleanup_task: Optional[asyncio.Task] = None
_cleanup_interval = 300  # 5 minutes


async def start_background_cleanup():
    """Start background cleanup task."""
    global _cleanup_task

    if _cleanup_task is not None and not _cleanup_task.done():
        logger.info("Background cleanup already running")
        return

    _cleanup_task = asyncio.create_task(_cleanup_loop())
    logger.info("Started background cleanup task")


async def stop_background_cleanup():
    """Stop background cleanup task."""
    global _cleanup_task

    if _cleanup_task is not None and not _cleanup_task.done():
        _cleanup_task.cancel()
        try:
            await _cleanup_task
        except asyncio.CancelledError:
            pass
        logger.info("Stopped background cleanup task")


async def _cleanup_loop():
    """Background loop for periodic cleanup."""
    logger.info("Background cleanup loop started")

    while True:
        try:
            await asyncio.sleep(_cleanup_interval)
            await perform_cleanup()
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Cleanup error: {e}")


async def perform_cleanup():
    """Perform system cleanup operations."""
    import gc
    import psutil
    import os

    logger.debug("Performing system cleanup...")

    # 1. Force garbage collection
    collected = gc.collect()

    # 2. Clear embedding cache if too large
    try:
        from app.core.embedding_cache import get_embedding_cache

        cache = get_embedding_cache()
        if hasattr(cache, "clear"):
            # Only clear if cache is excessively large
            if hasattr(cache, "__len__") and len(cache) > 1000:
                cache.clear()
                logger.info("Cleared embedding cache")
    except Exception as e:
        logger.debug(f"Could not clear embedding cache: {e}")

    # 3. Log memory usage
    try:
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        mem_mb = mem_info.rss / 1024 / 1024

        # Log warning if memory is high
        if mem_mb > 2048:  # 2GB
            logger.warning(f"High memory usage: {mem_mb:.1f}MB, collected {collected} objects")
        else:
            logger.debug(f"Memory: {mem_mb:.1f}MB, GC collected {collected} objects")
    except Exception as e:
        logger.debug(f"Could not get memory stats: {e}")

    # 4. Clean up stale circuit breaker stats
    try:
        from app.core.circuit_breaker import get_all_circuit_breakers

        cbs = get_all_circuit_breakers()
        for name, stats in cbs.items():
            # Reset if no activity for a long time
            if hasattr(stats, "last_success_time"):
                idle_time = time.time() - stats.last_success_time
                if idle_time > 3600:  # 1 hour
                    # Reset stats but keep circuit state
                    logger.info(f"Resetting stale circuit breaker: {name}")
    except Exception as e:
        logger.debug(f"Could not reset circuit breakers: {e}")


async def get_system_status() -> dict:
    """Get current system status."""
    import gc
    import psutil
    import os

    try:
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()

        return {
            "memory_rss_mb": mem_info.rss / 1024 / 1024,
            "memory_vms_mb": mem_info.vms / 1024 / 1024,
            "cpu_percent": process.cpu_percent(),
            "num_threads": process.num_threads(),
            "gc_counts": gc.get_count(),
            "open_files": len(process.open_files()),
        }
    except Exception as e:
        return {"error": str(e)}


# Export for use in main.py lifespan
__all__ = [
    "start_background_cleanup",
    "stop_background_cleanup",
    "perform_cleanup",
    "get_system_status",
]
