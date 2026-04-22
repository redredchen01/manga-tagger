from fastapi import Request
from fastapi.responses import JSONResponse
from app.exceptions import AppException
from app.logging_config import get_logger

logger = get_logger(__name__)


async def app_exception_handler(request: Request, exc: AppException):
    """Handle custom app exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {"code": exc.error_code, "message": exc.detail, "status": exc.status_code}
        },
    )


async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions with request context"""
    request_id = getattr(request.state, "request_id", None)
    correlation_id = getattr(request.state, "correlation_id", None)

    logger.error(
        "Unexpected error",
        error=str(exc),
        error_type=type(exc).__name__,
        request_id=request_id,
        correlation_id=correlation_id,
    )

    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "code": "INTERNAL_ERROR",
                "message": "An unexpected error occurred",
                "status": 500,
            }
        },
    )
