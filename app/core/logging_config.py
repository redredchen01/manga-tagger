import logging
import structlog
from app.core.config import settings


def configure_logging():
    if settings.LOG_FORMAT == "json":
        structlog.configure(
            processors=[
                structlog.processors.add_log_level,
                structlog.processors.TimeStamper(fmt="iso", utc=True),
                structlog.contextvars.merge_contextvars,
                structlog.processors.JSONRenderer(),
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
        )
    else:
        structlog.configure(
            processors=[
                structlog.processors.add_log_level,
                structlog.contextvars.merge_contextvars,
                structlog.dev.ConsoleRenderer(),
            ],
            context_class=dict,
            cache_logger_on_first_use=True,
        )
    logging.basicConfig(
        level=getattr(logging, settings.LOG_LEVEL.upper()),
        format="%(message)s",
    )


def get_logger(name: str):
    return structlog.get_logger(name)


def set_request_context(request_id: str, correlation_id: str):
    """Set request context for logging."""
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(request_id=request_id, correlation_id=correlation_id)


def clear_request_context():
    """Clear request context after request completes."""
    structlog.contextvars.clear_contextvars()


# Setup logging on module import
setup_logging = configure_logging
