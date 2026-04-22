"""Security headers middleware for OWASP recommended protections."""

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from app.config import settings


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Middleware to add OWASP recommended security headers to all responses."""

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)

        # Only apply security headers if enabled
        if not settings.SECURITY_HEADERS_ENABLED:
            return response

        # OWASP recommended security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        response.headers["Content-Security-Policy"] = "default-src 'self'"

        # Remove server header to avoid information disclosure
        if "server" in response.headers:
            del response.headers["server"]

        return response
