"""
Direct VLM diagnostic test - Tests multiple API formats to find working solution
"""

import asyncio
import base64
import io
import json
import httpx
from PIL import Image

# LM Studio settings
BASE_URL = "http://127.0.0.1:1234/v1"
API_KEY = "lm-studio"
MODEL = "zai-org/glm-4.6v-flash"


def create_test_image():
    """Create a simple test image."""
    img = Image.new("RGB", (512, 512), color="red")
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=85)
    return buffer.getvalue()


def encode_image(image_bytes):
    """Encode image to base64."""
    return base64.b64encode(image_bytes).decode("utf-8")


async def test_format_1_openai_compatible():
    """Test OpenAI-compatible format (current implementation)."""
    print("\n" + "=" * 60)
    print("TEST 1: OpenAI-Compatible Format (Current)")
    print("=" * 60)

    image_bytes = create_test_image()
    base64_image = encode_image(image_bytes)

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "List tags for this anime image: loli, catgirl, school_uniform",
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                },
            ],
        }
    ]

    payload = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": 512,
        "temperature": 0.7,
    }

    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(
                f"{BASE_URL}/chat/completions", headers=headers, json=payload
            )
            response.raise_for_status()
            result = response.json()

            print(f"Status: {response.status_code}")
            print(
                f"Response: {json.dumps(result, indent=2, ensure_ascii=False)[:1000]}"
            )

            if "choices" in result and len(result["choices"]) > 0:
                content = result["choices"][0]["message"]["content"]
                print(f"\nContent: {content}")
                return content
            else:
                print("\nNo choices in response!")
                return None

    except Exception as e:
        print(f"Error: {e}")
        return None


async def test_format_2_simple_prompt():
    """Test with simpler prompt."""
    print("\n" + "=" * 60)
    print("TEST 2: Simple Prompt Format")
    print("=" * 60)

    image_bytes = create_test_image()
    base64_image = encode_image(image_bytes)

    # Try with text-only content first to see if model responds
    messages = [{"role": "user", "content": "What do you see in this image?"}]

    # Alternative: inline image in text
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image in detail."},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                },
            ],
        }
    ]

    payload = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": 256,
        "temperature": 0.5,
    }

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(
                f"{BASE_URL}/chat/completions",
                headers={"Content-Type": "application/json"},
                json=payload,
            )
            response.raise_for_status()
            result = response.json()

            print(f"Status: {response.status_code}")
            if "choices" in result and len(result["choices"]) > 0:
                content = result["choices"][0]["message"]["content"]
                print(f"Content: {content[:500]}")
                return content
            else:
                print(f"Response: {json.dumps(result, indent=2)[:500]}")
                return None

    except Exception as e:
        print(f"Error: {e}")
        return None


async def test_format_3_different_model():
    """Test with different available model."""
    print("\n" + "=" * 60)
    print("TEST 3: Testing Available Models")
    print("=" * 60)

    # First check what models are available
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get(f"{BASE_URL}/models")
            response.raise_for_status()
            models = response.json()
            print(f"Available models: {json.dumps(models, indent=2)}")
    except Exception as e:
        print(f"Error getting models: {e}")


async def test_format_4_lmstudio_native():
    """Test LM Studio native completions endpoint."""
    print("\n" + "=" * 60)
    print("TEST 4: LM Studio Native Format")
    print("=" * 60)

    image_bytes = create_test_image()
    base64_image = encode_image(image_bytes)

    # Some vision models work better with different payload structure
    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant that analyzes anime/manga images.",
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Analyze this image and list relevant tags separated by commas.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    },
                ],
            },
        ],
        "max_tokens": 512,
        "temperature": 0.7,
        "stream": False,
    }

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(
                f"{BASE_URL}/chat/completions",
                headers={"Content-Type": "application/json"},
                json=payload,
            )
            response.raise_for_status()
            result = response.json()

            print(f"Status: {response.status_code}")
            if "choices" in result and len(result["choices"]) > 0:
                content = result["choices"][0]["message"]["content"]
                print(f"Content: {content[:500]}")
                return content
            else:
                print(f"Response: {json.dumps(result, indent=2)[:500]}")
                return None

    except Exception as e:
        print(f"Error: {e}")
        return None


async def main():
    """Run all diagnostic tests."""
    print("=" * 60)
    print("VLM DIAGNOSTIC TESTS")
    print("=" * 60)
    print(f"Testing model: {MODEL}")
    print(f"LM Studio URL: {BASE_URL}")

    # Test 1: Current OpenAI format
    result1 = await test_format_1_openai_compatible()

    # Test 2: Simple prompt
    result2 = await test_format_2_simple_prompt()

    # Test 3: Check available models
    await test_format_3_different_model()

    # Test 4: Native format with system message
    result4 = await test_format_4_lmstudio_native()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    results = [
        ("OpenAI Format", result1),
        ("Simple Prompt", result2),
        ("LM Studio Native", result4),
    ]

    for name, result in results:
        if (
            result
            and len(result.strip()) > 10
            and not all(c in "<|>" for c in result.strip())
        ):
            print(f"✓ {name}: WORKING")
            print(f"  Sample: {result[:100]}...")
        else:
            print(f"✗ {name}: FAILED or EMPTY")
            if result:
                print(f"  Got: {result[:100]}...")


if __name__ == "__main__":
    asyncio.run(main())
