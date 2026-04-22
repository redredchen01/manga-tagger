"""Core infrastructure module - re-exports for backward compatibility."""

from app.core.config import settings
from app.core.config import Settings
from app.core.exceptions import AppException
from app.core.exceptions import AuthenticationError
from app.core.exceptions import NotFoundError
from app.core.exceptions import RateLimitError
from app.core.exceptions import ServiceUnavailableError
from app.core.exceptions import ValidationError
from app.core.exceptions import TaggingError
from app.core.exceptions import RAGError
from app.core.exceptions import LLMError
from app.core.logging_config import get_logger
from app.core.logging_config import configure_logging
from app.core.logging_config import setup_logging
from app.core.logging_config import set_request_context
from app.core.logging_config import clear_request_context
from app.core.http_client import get_http_client
from app.core.http_client import close_http_client
from app.core.cache import cache_manager
from app.core.cache import CacheManager
from app.core.metrics import (
    REQUEST_COUNT,
    REQUEST_LATENCY,
    VLM_REQUEST_COUNT,
    VLM_LATENCY,
    RAG_REQUEST_COUNT,
    RAG_LATENCY,
    TAG_RECOMMENDATION_COUNT,
    TAG_RECOMMENDATION_LATENCY,
    CACHE_HITS,
    CACHE_MISSES,
    ACTIVE_REQUESTS,
)

# Performance submodule
from app.core.performance.async_utils import (
    run_in_executor,
    gather_with_limit,
    timeout_after,
    run_with_retry,
    AsyncBatcher,
)
from app.core.performance.memory import (
    track_memory,
    get_memory_stats,
    force_garbage_collection,
)

__all__ = [
    # Config
    "settings",
    "Settings",
    # Exceptions
    "AppException",
    "AuthenticationError",
    "NotFoundError",
    "RateLimitError",
    "ServiceUnavailableError",
    "ValidationError",
    "TaggingError",
    "RAGError",
    "LLMError",
    # Logging
    "get_logger",
    "configure_logging",
    "setup_logging",
    "set_request_context",
    "clear_request_context",
    # HTTP Client
    "get_http_client",
    "close_http_client",
    # Cache
    "cache_manager",
    "CacheManager",
    # Metrics
    "REQUEST_COUNT",
    "REQUEST_LATENCY",
    "VLM_REQUEST_COUNT",
    "VLM_LATENCY",
    "RAG_REQUEST_COUNT",
    "RAG_LATENCY",
    "TAG_RECOMMENDATION_COUNT",
    "TAG_RECOMMENDATION_LATENCY",
    "CACHE_HITS",
    "CACHE_MISSES",
    "ACTIVE_REQUESTS",
    # Performance
    "run_in_executor",
    "gather_with_limit",
    "timeout_after",
    "run_with_retry",
    "AsyncBatcher",
    "track_memory",
    "get_memory_stats",
    "force_garbage_collection",
]
