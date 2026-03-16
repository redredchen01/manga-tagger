# Test VLM service directly
import sys
import asyncio
import io
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))

from app.services.lm_studio_vlm_service_v2 import LMStudioVLMService


async def test_vlm_service():
    """Test VLM service with a simple image."""
    print("=== Testing VLM Service ===\n")

    # Create a simple test image
    print("Creating test image...")
    img = Image.new("RGB", (512, 512), color="red")
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="JPEG")
    image_data = img_bytes.getvalue()
    print(f"Image size: {len(image_data)} bytes\n")

    # Initialize VLM service
    print("Initializing VLM service...")
    vlm = LMStudioVLMService()
    print(f"Model: {vlm.model}")
    print(f"Base URL: {vlm.base_url}")
    print(f"Timeout: {vlm.timeout}s\n")

    # Test metadata extraction
    print("Extracting metadata (this may take a while)...")
    try:
        result = await asyncio.wait_for(
            vlm.extract_metadata(image_data),
            timeout=60,  # 60 second timeout for testing
        )

        print("\n=== VLM Result ===")
        print(f"Description: {result.get('description', 'N/A')[:200]}...")
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
            print("\n[OK] VLM returned valid results!")
        else:
            print("\n[WARNING] VLM returned empty results!")
            print("This is likely why no tags are generated.")

    except asyncio.TimeoutError:
        print("\n[ERROR] VLM service timed out after 30 seconds!")
        print("LM Studio may be too slow or not responding.")
    except Exception as e:
        print(f"\n[ERROR] VLM service failed: {e}")


if __name__ == "__main__":
    asyncio.run(test_vlm_service())
