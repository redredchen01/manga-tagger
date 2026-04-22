"""Memory monitoring utilities for Python 3.12+."""

import gc
import logging
import tracemalloc
from functools import wraps
from typing import Callable

logger = logging.getLogger(__name__)


def track_memory(func: Callable):
    """Decorator to track memory usage of functions.

    Uses tracemalloc to monitor peak memory usage. Logs a warning if peak
    exceeds 100MB threshold. Works with both sync and async functions.

    Args:
        func: The function to wrap

    Returns:
        Wrapped function with memory tracking
    """

    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        tracemalloc.start()

        try:
            result = await func(*args, **kwargs)
            return result
        finally:
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            if peak > 100 * 1024 * 1024:  # 100MB
                logger.warning(f"High memory usage in {func.__name__}: {peak / 1024 / 1024:.2f}MB")

    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        tracemalloc.start()

        try:
            result = func(*args, **kwargs)
            return result
        finally:
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            if peak > 100 * 1024 * 1024:  # 100MB
                logger.warning(f"High memory usage in {func.__name__}: {peak / 1024 / 1024:.2f}MB")

    # Return appropriate wrapper based on function type
    import asyncio

    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    return sync_wrapper


def get_memory_stats() -> dict:
    """Get current memory statistics.

    Returns:
        Dictionary with memory usage information
    """
    gc.collect()

    # Get GC stats if available (Python 3.12+)
    try:
        gc_stats = gc.get_stats()
    except AttributeError:
        gc_stats = []

    return {
        "gc_stats": gc_stats,
        "gc_counts": gc.get_count(),
    }


def force_garbage_collection() -> dict:
    """Force garbage collection and return statistics.

    Returns:
        Dictionary with collection results
    """
    collected = gc.collect()
    return {
        "objects_collected": collected,
        "gc_counts": gc.get_count(),
    }
