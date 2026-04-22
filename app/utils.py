"""Utility functions for handling floating point precision issues and image validation."""

import io

from PIL import Image


def safe_confidence(value: float) -> float:
    """Ensure confidence is within valid [0, 1] range with proper precision."""
    rounded = round(float(value), 6)
    return max(0.0, min(1.0, rounded))


def validate_image(
    image_bytes: bytes,
    max_size_mb: float = 10.0,
    min_size_bytes: int = 1024,
    min_dimension: int = 32,
    max_dimension: int = 4096,
) -> tuple[bool, str]:
    """Validate image bytes and return (is_valid, error_message).

    Args:
        image_bytes: Raw image bytes
        max_size_mb: Maximum file size in MB
        min_size_bytes: Minimum file size in bytes
        min_dimension: Minimum width/height in pixels
        max_dimension: Maximum width/height in pixels

    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check file size
    size_mb = len(image_bytes) / (1024 * 1024)
    if size_mb > max_size_mb:
        return False, f"File too large. Max: {max_size_mb}MB, Got: {size_mb:.1f}MB"

    if len(image_bytes) < min_size_bytes:
        return False, f"File too small. Min: {min_size_bytes} bytes"

    # Validate image format
    try:
        img = Image.open(io.BytesIO(image_bytes))
        width, height = img.size

        if width < min_dimension or height < min_dimension:
            return (
                False,
                f"Image too small. Min: {min_dimension}x{min_dimension}, Got: {width}x{height}",
            )

        if width > max_dimension or height > max_dimension:
            return (
                False,
                f"Image too large. Max: {max_dimension}x{max_dimension}, Got: {width}x{height}",
            )

        return True, ""

    except Exception as e:
        return False, f"Invalid image: {str(e)}"
