"""Unit tests for configuration settings.

Tests configuration defaults, environment variable handling, and security settings.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import Settings


class TestConfigSecurity:
    """Test security-related configuration."""

    def test_cors_empty_default_is_secure(self):
        """Test that empty CORS_ORIGINS defaults to empty list (secure)."""
        # Create fresh settings with empty CORS_ORIGINS
        s = Settings(CORS_ORIGINS=[])
        assert s.cors_origins == []

    def test_cors_list_configured_correctly(self):
        """Test that CORS_ORIGINS can be configured as a list."""
        s = Settings(CORS_ORIGINS=["https://example.com", "https://api.example.com"])
        assert s.cors_origins == ["https://example.com", "https://api.example.com"]

    def test_cors_string_configured_correctly(self):
        """Test that CORS_ORIGINS can be configured as comma-separated string."""
        s = Settings(CORS_ORIGINS="https://example.com,https://api.example.com")
        assert s.cors_origins == ["https://example.com", "https://api.example.com"]


class TestSensitiveTags:
    """Test sensitive tags configuration."""

    def test_sensitive_tags_default(self):
        """Test that default sensitive tags are loaded."""
        s = Settings()
        assert s.sensitive_tags is not None
        assert isinstance(s.sensitive_tags, set)
        assert len(s.sensitive_tags) > 0

    def test_sensitive_tags_env_override(self):
        """Test that SENSITIVE_TAGS can be overridden via environment variable."""
        original = os.environ.get("SENSITIVE_TAGS")
        try:
            os.environ["SENSITIVE_TAGS"] = "tag1,tag2,tag3"
            s = Settings()
            assert s.sensitive_tags == {"tag1", "tag2", "tag3"}
        finally:
            if original is not None:
                os.environ["SENSITIVE_TAGS"] = original
            elif "SENSITIVE_TAGS" in os.environ:
                del os.environ["SENSITIVE_TAGS"]

    def test_sensitive_tags_whitespace_handling(self):
        """Test that sensitive tags handle whitespace correctly."""
        original = os.environ.get("SENSITIVE_TAGS")
        try:
            os.environ["SENSITIVE_TAGS"] = "  tag1  ,  tag2  ,  tag3  "
            s = Settings()
            assert s.sensitive_tags == {"tag1", "tag2", "tag3"}
        finally:
            if original is not None:
                os.environ["SENSITIVE_TAGS"] = original
            elif "SENSITIVE_TAGS" in os.environ:
                del os.environ["SENSITIVE_TAGS"]

    def test_sensitive_substring_filter_enabled_default(self):
        """The substring post-filter flag defaults to True."""
        from app.core.config import Settings
        s = Settings()
        assert s.SENSITIVE_SUBSTRING_FILTER_ENABLED is True


class TestConcurrencyDefaults:
    """Test concurrency configuration defaults."""

    def test_default_concurrency_increased(self):
        """Test that concurrency defaults meet the minimum requirements."""
        s = Settings()
        assert s.MAX_CONCURRENT_REQUESTS >= 10, "MAX_CONCURRENT_REQUESTS should be >= 10"
        assert s.API_WORKERS >= 4, "API_WORKERS should be >= 4"

    def test_custom_concurrency_values(self):
        """Test that custom values can override defaults."""
        s = Settings(MAX_CONCURRENT_REQUESTS=20, API_WORKERS=8)
        assert s.MAX_CONCURRENT_REQUESTS == 20
        assert s.API_WORKERS == 8


class TestLMStudioConfig:
    """Test LM Studio configuration."""

    def test_lm_studio_defaults(self):
        """Test that LM Studio defaults are sensible."""
        s = Settings()
        assert s.LM_STUDIO_BASE_URL == "http://127.0.0.1:1234/v1"
        assert s.LM_STUDIO_VISION_MODEL is not None
        assert s.LM_STUDIO_EMBEDDING_MODEL is not None
        assert s.LM_STUDIO_EMBEDDING_DIM == 4096

    def test_lm_studio_can_be_disabled(self):
        """Test that LM Studio can be disabled."""
        s = Settings(USE_LM_STUDIO=False)
        assert s.USE_LM_STUDIO is False


class TestMockServices:
    """Test mock services configuration."""

    def test_mock_services_default_value(self):
        """Test the default value for mock services in Settings class."""
        # Check the class default, not runtime value (which may be overridden by .env)
        # The default in Settings class should be False
        assert Settings.model_fields["USE_MOCK_SERVICES"].default is False

    def test_mock_services_can_be_enabled(self):
        """Test that mock services can be enabled."""
        s = Settings(USE_MOCK_SERVICES=True)
        assert s.USE_MOCK_SERVICES is True


class TestTagLibraryPath:
    """Test tag library path configuration."""

    def test_tag_library_path_resolved(self):
        """Test that tag library path is properly resolved."""
        s = Settings()
        assert s.tag_library_path is not None
        assert isinstance(s.tag_library_path, Path)

    def test_chroma_db_path_resolved(self):
        """Test that ChromaDB path is properly resolved."""
        s = Settings()
        assert s.chroma_db_path is not None
        assert isinstance(s.chroma_db_path, Path)
