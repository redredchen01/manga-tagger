"""
Test the fixed VLM service
"""

import asyncio
import io
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from PIL import Image
from app.services.lm_studio_vlm_service_v2 import LMStudioVLMService


def create_test_image():
    """Create a test image with visual content."""
    from PIL import ImageDraw

    # Create a more interesting image
    img = Image.new("RGB", (512, 512), color="white")
    draw = ImageDraw.Draw(img)

    # Draw a simple face-like pattern
    draw.ellipse([150, 100, 350, 300], outline="black", width=3)
    draw.ellipse([200, 160, 230, 190], fill="blue")
    draw.ellipse([270, 160, 300, 190], fill="blue")
    draw.arc([200, 200, 300, 280], start=0, end=180, fill="red", width=3)
    draw.text((180, 350), "Anime Character", fill="black")

    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=85)
    return buffer.getvalue()


async def test_fixed_vlm():
    """Test the fixed VLM service."""
    print("=" * 60)
    print("TESTING FIXED VLM SERVICE")
    print("=" * 60)

    # Create test image
    print("\n[1] Creating test image...")
    image_data = create_test_image()
    print(f"    Image size: {len(image_data)} bytes")

    # Initialize service
    print("\n[2] Initializing VLM service...")
    vlm = LMStudioVLMService()
    print(f"    Model: {vlm.model}")

    # Test extraction
    print("\n[3] Extracting metadata...")
    print("    (This may take 10-30 seconds...)")

    try:
        result = await asyncio.wait_for(vlm.extract_metadata(image_data), timeout=60)

        print("\n" + "=" * 60)
        print("RESULT")
        print("=" * 60)
        print(f"\nDescription: {result.get('description', 'N/A')[:200]}...")
        print(f"\nCharacter types: {result.get('character_types', [])}")
        print(f"Clothing: {result.get('clothing', [])}")
        print(f"Body features: {result.get('body_features', [])}")
        print(f"Actions: {result.get('actions', [])}")
        print(f"Themes: {result.get('themes', [])}")
        print(f"\nRaw keywords: {result.get('raw_keywords', [])}")

        # Check if we got any tags
        total_tags = (
            len(result.get("character_types", []))
            + len(result.get("clothing", []))
            + len(result.get("body_features", []))
            + len(result.get("actions", []))
            + len(result.get("themes", []))
        )

        print("\n" + "=" * 60)
        if total_tags > 0:
            print(f"SUCCESS! Extracted {total_tags} categorized tags")
        else:
            print("No categorized tags extracted")
            print("But raw keywords were found and should be mapped by tag_mapper")
        print("=" * 60)

    except asyncio.TimeoutError:
        print("\n[ERROR] Timeout after 60 seconds")
        print("VLM service took too long to respond")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_fixed_vlm())
