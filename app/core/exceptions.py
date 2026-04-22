from typing import Optional

from fastapi import HTTPException, status


class AppException(HTTPException):
    """Base application exception"""

    def __init__(self, status_code: int, detail: str, error_code: Optional[str] = None):
        super().__init__(status_code=status_code, detail=detail)
        self.error_code = error_code


class ValidationError(AppException):
    """Input validation error"""

    def __init__(self, detail: str):
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST, detail=detail, error_code="VALIDATION_ERROR"
        )


class ServiceUnavailableError(AppException):
    """External service unavailable"""

    def __init__(self, service: str):
        super().__init__(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"{service} service is unavailable",
            error_code="SERVICE_UNAVAILABLE",
        )


class RateLimitError(AppException):
    """Rate limit exceeded"""

    def __init__(self, retry_after: float):
        super().__init__(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded",
            error_code="RATE_LIMIT_EXCEEDED",
        )
        self.retry_after = retry_after


class AuthenticationError(AppException):
    """Authentication failed"""

    def __init__(self, detail: str = "Invalid API key"):
        super().__init__(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=detail,
            error_code="AUTHENTICATION_FAILED",
        )


class NotFoundError(AppException):
    """Resource not found"""

    def __init__(self, resource: str):
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"{resource} not found",
            error_code="NOT_FOUND",
        )


# Additional exception classes that may be used by the application
# These are re-exported from app/__init__.py for backward compatibility
class TaggingError(AppException):
    """Tagging operation failed"""

    def __init__(self, detail: str):
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=detail,
            error_code="TAGGING_ERROR",
        )


class RAGError(AppException):
    """RAG operation failed"""

    def __init__(self, detail: str):
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=detail,
            error_code="RAG_ERROR",
        )


class LLMError(AppException):
    """LLM operation failed"""

    def __init__(self, detail: str):
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=detail,
            error_code="LLM_ERROR",
        )
