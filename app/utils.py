"""Utility functions for handling floating point precision issues."""


def safe_confidence(value: float) -> float:
    """Ensure confidence is within valid [0, 1] range with proper precision."""
    # Round to 6 decimal places, then clamp to [0, 1]
    rounded = round(float(value), 6)
    return max(0.0, min(1.0, rounded))
