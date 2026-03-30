"""API Key authentication for Manga Tagger."""

import secrets

from fastapi import Depends, HTTPException, status
from fastapi.security import APIKeyHeader

from app.config import settings
from app.exceptions import AuthenticationError

# API Key header definition
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: str | None = Depends(api_key_header)) -> str | None:
    """Verify the API key from the X-API-Key header.

    Returns the API key if valid.
    Raises HTTPException 401 if no API key is provided when required.
    Raises HTTPException 403 if the API key is invalid.

    When API_KEY is None (not set), this function allows all requests through
    (returns the key as None), making authentication optional for backward compatibility.
    """
    # If API_KEY is not configured, skip authentication (backward compatible)
    if settings.API_KEY is None:
        return None

    # API_KEY is configured, require authentication
    if api_key is None:
        raise AuthenticationError(detail="X-API-Key header is required")

    # Verify the API key
    if not secrets.compare_digest(api_key, settings.API_KEY):
        raise AuthenticationError(detail="Invalid API key")

    return api_key


class RequireAPIKey:
    """Dependency for routes that require API key authentication.

    When API_KEY is not configured (None), this dependency allows all requests through.
    When API_KEY is configured, it enforces API key authentication.
    """

    def __init__(self):
        pass

    async def __call__(self, api_key: str | None = Depends(verify_api_key)) -> None:
        """Verify API key and return None if valid (or not required)."""
        # verify_api_key already handles the HTTPException logic
        # If we get here, the key is valid (or not required)
        return None


# Convenience dependency instance for use in routes
RequireAPIKey = RequireAPIKey
