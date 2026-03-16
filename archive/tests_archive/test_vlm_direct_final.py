# Test VLM directly
import asyncio
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from PIL import Image, ImageDraw
from app.services.lm_studio_vlm_service_v2 import LMStudioVLMService


def create_test_image():
    """Create test image."""
    img = Image.new("RGB", (512, 512), color="white")
    draw = ImageDraw.Draw(img)
    draw.ellipse([150, 100, 350, 300], outline="black", width=3)
    draw.ellipse([200, 160, 230, 190], fill="blue")
    draw.ellipse([270, 160, 300, 190], fill="blue")
    draw.rectangle([150, 300, 350, 480], outline="red", width=3)
    draw.text((180, 350), "Anime Girl", fill="black")
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=85)
    return buffer.getvalue()


async def test_vlm():
    """Test VLM service."""
    print("Testing VLM service...")

    vlm = LMStudioVLMService()
    image_data = create_test_image()

    print("Sending image to VLM...")
    result = await vlm.extract_metadata(image_data)

    print("\nResult:")
    print(f"  Description: {result.get('description', 'N/A')[:100]}...")
    print(f"  Character types: {result.get('character_types', [])}")
    print(f"  Clothing: {result.get('clothing', [])}")
    print(f"  Body features: {result.get('body_features', [])}")
    print(f"  Raw keywords: {result.get('raw_keywords', [])}")

    has_data = (
        result.get("character_types")
        or result.get("clothing")
        or result.get("body_features")
        or result.get("raw_keywords")
    )

    if has_data:
        print("\n[OK] VLM is working and returning tags!")
    else:
        print("\n[FAIL] VLM returned empty data")


if __name__ == "__main__":
    asyncio.run(test_vlm())
