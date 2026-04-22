"""Shared pytest fixtures for Manga Tagger tests.

Cleans up app.state between test modules to prevent state leakage.
"""

import pytest

from app.main import app


@pytest.fixture(autouse=True, scope="module")
def cleanup_app_state():
    """Reset app.state between test modules to prevent state leakage."""
    # Snapshot attributes before test module
    original_attrs = set(vars(app.state).keys())
    yield
    # Remove any attributes added during test
    for attr in list(vars(app.state).keys()):
        if attr not in original_attrs:
            delattr(app.state, attr)
