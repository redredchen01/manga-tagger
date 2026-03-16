# Test mock VLM mode
import sys
import asyncio
import io
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))

# Enable mock mode
import app.config

app.config.settings.USE_MOCK_SERVICES = True

from app.services.lm_studio_vlm_service_v2 import LMStudioVLMService


async def test_mock_vlm():
    """Test VLM service in mock mode."""
    print("=== Testing Mock VLM Service ===\n")

    # Create a simple test image
    print("Creating test image...")
    img = Image.new("RGB", (512, 512), color="red")
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="JPEG")
    image_data = img_bytes.getvalue()
    print(f"Image size: {len(image_data)} bytes\n")

    # Initialize VLM service
    print("Initializing VLM service (MOCK MODE)...")
    vlm = LMStudioVLMService()
    print(f"Model: {vlm.model}")
    print(f"Mock mode: ENABLED\n")

    # Test metadata extraction
    print("Extracting metadata...")
    result = await vlm.extract_metadata(image_data)

    print("\n=== Mock VLM Result ===")
    print(f"Description: {result.get('description', 'N/A')}")
    print(f"Character types: {result.get('character_types', [])}")
    print(f"Clothing: {result.get('clothing', [])}")
    print(f"Body features: {result.get('body_features', [])}")
    print(f"Actions: {result.get('actions', [])}")
    print(f"Themes: {result.get('themes', [])}")
    print(f"Raw keywords: {result.get('raw_keywords', [])}")

    # Check if result is valid
    has_content = (
        result.get("character_types")
        or result.get("clothing")
        or result.get("body_features")
        or result.get("raw_keywords")
    )

    if has_content:
        print("\n[OK] Mock VLM working! Tags will be generated.")
    else:
        print("\n[ERROR] Mock VLM returned empty results!")


if __name__ == "__main__":
    asyncio.run(test_mock_vlm())
