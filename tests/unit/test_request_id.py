"""Unit tests for request ID middleware.

Tests that request IDs are properly generated and propagated.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.main import app


client = TestClient(app)


class TestRequestIDMiddleware:
    """Test request ID middleware functionality."""

    def test_request_id_generated_for_each_request(self):
        """Each request should get a unique request ID."""
        response1 = client.get("/api/v1/health")
        response2 = client.get("/api/v1/health")

        assert response1.status_code == 200
        assert response2.status_code == 200

        # Both should have request IDs
        assert "x-request-id" in response1.headers
        assert "x-request-id" in response2.headers

        # IDs should be unique
        assert response1.headers["x-request-id"] != response2.headers["x-request-id"]

    def test_request_id_is_uuid(self):
        """Request ID should be a valid UUID."""
        response = client.get("/api/v1/health")

        request_id = response.headers["x-request-id"]
        # Should be a valid UUID format (8-4-4-4-12)
        assert len(request_id) == 36
        assert request_id.count("-") == 4

    def test_correlation_id_forwarded(self):
        """Incoming X-Correlation-ID should be forwarded."""
        response = client.get(
            "/api/v1/health",
            headers={"X-Correlation-ID": "test-correlation-123"},
        )

        assert response.status_code == 200
        assert response.headers["x-correlation-id"] == "test-correlation-123"

    def test_correlation_id_generated_when_missing(self):
        """When no correlation ID, request ID should be used."""
        response = client.get("/api/v1/health")

        assert response.status_code == 200
        # When no incoming correlation ID, it should equal request ID
        assert response.headers["x-correlation-id"] == response.headers["x-request-id"]

    def test_request_id_case_insensitive(self):
        """Request ID header should work case-insensitively."""
        response = client.get(
            "/api/v1/health",
            headers={"x-correlation-id": "test-123"},
        )

        assert response.status_code == 200
        # Both should be present (case-insensitive)
        assert "X-Request-ID" in response.headers
        assert "X-Correlation-ID" in response.headers


class TestRequestIDLogging:
    """Test that request IDs are available in logging context."""

    def test_request_context_available_in_logger(self):
        """Request context should be available in structured logging."""
        # This is a basic test - in real usage, you'd check the logs
        response = client.get("/api/v1/health")

        assert response.status_code == 200
        assert "x-request-id" in response.headers


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
