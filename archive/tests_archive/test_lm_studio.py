"""Test script for LM Studio integration."""

import asyncio
import requests
import sys
from pathlib import Path

# Add the project root to path
sys.path.insert(0, str(Path(__file__).parent))

from app.config import settings
from app.services.lm_studio_vlm_service import LMStudioVLMService
from app.services.lm_studio_llm_service import LMStudioLLMService


async def test_lm_studio_connectivity():
    """Test basic connectivity to LM Studio."""
    print("Testing LM Studio connectivity...")

    try:
        response = requests.get(
            f"{settings.LM_STUDIO_BASE_URL}/models",
            headers={"Authorization": f"Bearer {settings.LM_STUDIO_API_KEY}"},
            timeout=5,
        )
        if response.status_code == 200:
            models = response.json()
            print(f"✅ LM Studio is reachable at {settings.LM_STUDIO_BASE_URL}")
            print(
                f"Available models: {[model['id'] for model in models.get('data', [])]}"
            )
            return True
        else:
            print(f"❌ LM Studio returned status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to LM Studio: {e}")
        print(f"   Make sure LM Studio is running at {settings.LM_STUDIO_BASE_URL}")
        return False


async def test_vlm_service():
    """Test VLM service with a simple image."""
    print("\nTesting VLM service...")

    try:
        vlm_service = LMStudioVLMService()

        # Create a simple test image (1x1 pixel)
        from PIL import Image
        import io

        # Create a test image
        test_image = Image.new("RGB", (100, 100), color="red")
        buffer = io.BytesIO()
        test_image.save(buffer, format="JPEG")
        image_bytes = buffer.getvalue()

        # Test metadata extraction
        metadata = await vlm_service.extract_metadata(image_bytes)

        print(f"✅ VLM service working")
        print(f"   Description: {metadata.description[:100]}...")
        print(f"   Characters: {metadata.characters}")
        print(f"   Themes: {metadata.themes}")
        return True

    except Exception as e:
        print(f"❌ VLM service failed: {e}")
        return False


async def test_llm_service():
    """Test LLM service."""
    print("\nTesting LLM service...")

    try:
        llm_service = LMStudioLLMService()

        # Create mock data
        from app.models import VLMMetadata

        vlm_metadata = VLMMetadata(
            description="Test image with a girl in school uniform",
            characters=["少女"],
            themes=["學校"],
            art_style="anime style",
            genre_indicators=["校服"],
        )

        rag_matches = [
            {"score": 0.8, "tags": ["蘿莉", "校服", "雙馬尾"]},
            {"score": 0.7, "tags": ["少女", "學校", "制服"]},
        ]

        # Test tag synthesis
        tags = await llm_service.synthesize_tags(
            vlm_metadata=vlm_metadata,
            rag_matches=rag_matches,
            top_k=5,
            confidence_threshold=0.5,
        )

        print(f"✅ LLM service working")
        print(f"   Generated {len(tags)} tags:")
        for tag in tags:
            print(f"   - {tag.tag} (confidence: {tag.confidence:.2f})")
        return True

    except Exception as e:
        print(f"❌ LLM service failed: {e}")
        return False


async def main():
    """Run all tests."""
    print("=" * 50)
    print("LM Studio Integration Test")
    print("=" * 50)

    # Check configuration
    print(f"Configuration:")
    print(f"  USE_LM_STUDIO: {settings.USE_LM_STUDIO}")
    print(f"  LM_STUDIO_BASE_URL: {settings.LM_STUDIO_BASE_URL}")
    print(f"  LM_STUDIO_VISION_MODEL: {settings.LM_STUDIO_VISION_MODEL}")
    print(f"  LM_STUDIO_TEXT_MODEL: {settings.LM_STUDIO_TEXT_MODEL}")
    print()

    # Run tests
    tests = [test_lm_studio_connectivity(), test_vlm_service(), test_llm_service()]

    results = await asyncio.gather(*tests, return_exceptions=True)

    # Summary
    print("\n" + "=" * 50)
    print("Test Summary:")
    print("=" * 50)

    passed = sum(1 for result in results if result is True)
    total = len(results)

    print(f"Passed: {passed}/{total}")

    if passed == total:
        print("🎉 All tests passed! LM Studio integration is working.")
    else:
        print("❌ Some tests failed. Check the errors above.")

    return passed == total


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
