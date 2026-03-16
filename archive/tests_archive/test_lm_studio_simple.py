"""Simple test script to verify LM Studio integration works."""

import sys
import io
import asyncio
import requests
import base64
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


async def test_simple_chat():
    """Test simple chat completion with GLM-4.6V-flash."""
    print("\nTesting simple chat completion...")

    try:
        import httpx

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{settings.LM_STUDIO_BASE_URL}/chat/completions",
                headers={"Authorization": f"Bearer {settings.LM_STUDIO_API_KEY}"},
                json={
                    "model": settings.LM_STUDIO_VISION_MODEL,
                    "messages": [{"role": "user", "content": "Hello, how are you?"}],
                    "max_tokens": 50,
                },
                timeout=10,
            )

            if response.status_code == 200:
                result = response.json()
                print(f"✅ Chat completion successful")
                print(
                    f"   Response: {result.get('choices', [{}])[0].get('message', {}).get('content', 'No response')}"
                )
                return True
            else:
                print(f"❌ Chat completion failed: {response.status_code}")
                return False

    except Exception as e:
        print(f"❌ Chat completion failed: {e}")
        return False


async def test_image_analysis():
    """Test image analysis with GLM-4.6V-flash."""
    print("\nTesting image analysis...")

    try:
        import httpx

        # Create a simple test image
        from PIL import Image
        import io

        test_image = Image.new("RGB", (100, 100), color="red")
        buffer = io.BytesIO()
        test_image.save(buffer, format="JPEG")
        image_bytes = buffer.getvalue()

        # Convert to base64
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{settings.LM_STUDIO_BASE_URL}/chat/completions",
                headers={
                    "Authorization": f"Bearer {settings.LM_STUDIO_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": settings.LM_STUDIO_VISION_MODEL,
                    "messages": [
                        {
                            "role": "user",
                            "content": "Analyze this image and describe what you see:",
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{image_b64}"
                                    },
                                }
                            ],
                        },
                    ],
                    "max_tokens": 100,
                },
                timeout=10,
            )

            if response.status_code == 200:
                result = response.json()
                print(f"✅ Image analysis successful")
                print(
                    f"   Response: {result.get('choices', [{}])[0].get('message', {}).get('content', 'No response')}"
                )
                return True
            else:
                print(f"❌ Image analysis failed: {response.status_code}")
                return False

    except Exception as e:
        print(f"❌ Image analysis failed: {e}")
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
    print()

    # Run tests
    tests = [test_lm_studio_connectivity(), test_simple_chat(), test_image_analysis()]

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
