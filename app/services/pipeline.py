"""Pipeline module — re-exports from domain.

This module provides backward compatibility by re-exporting the unified
pipeline function from app.domain.pipeline.

Deprecated: Use app.domain.pipeline directly.
"""

from app.domain.pipeline import run_tagging_pipeline  # noqa: F401

# Import the new function for backward compatibility
from app.domain.pipeline import (  # noqa: F401
    run_tagging_pipeline as run_tagging_pipeline_with_progress,
)
