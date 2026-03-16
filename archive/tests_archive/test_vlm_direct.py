# Direct test of VLM with detailed logging
import asyncio
import base64
import io
import json
import httpx
from PIL import Image

BASE_URL = "http://127.0.0.1:1234/v1"
MODEL = "zai-org/glm-4.6v-flash"


def create_test_image():
    """Create a test image with some visual content."""
    from PIL import ImageDraw

    # Create a more interesting image than just red
    img = Image.new("RGB", (512, 512), color="white")
    draw = ImageDraw.Draw(img)

    # Draw a simple face-like pattern
    draw.ellipse([150, 100, 350, 300], outline="black", width=3)  # Face
    draw.ellipse([200, 160, 230, 190], fill="blue")  # Left eye
    draw.ellipse([270, 160, 300, 190], fill="blue")  # Right eye
    draw.arc([200, 200, 300, 280], start=0, end=180, fill="red", width=3)  # Smile

    # Add some text
    draw.text((180, 350), "Anime Character", fill="black")

    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=85)
    return buffer.getvalue()


async def test_vlm_detailed():
    """Test VLM with detailed logging."""
    print("Creating test image with content...")
    image_bytes = create_test_image()
    base64_image = base64.b64encode(image_bytes).decode("utf-8")
    print(f"Image encoded: {len(base64_image)} chars")

    # Try the exact format from the working diagnostic test
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
        "max_tokens": 1024,
        "temperature": 0.5,
    }

    print(f"\nSending request to {BASE_URL}/chat/completions")
    print(f"Model: {MODEL}")
    print("Waiting for response...")

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(
                f"{BASE_URL}/chat/completions",
                headers={"Content-Type": "application/json"},
                json=payload,
            )
            response.raise_for_status()
            result = response.json()

            print(f"\nStatus: {response.status_code}")
            print(f"Full response:")
            print(json.dumps(result, indent=2, ensure_ascii=False))

            if "choices" in result and len(result["choices"]) > 0:
                content = result["choices"][0]["message"]["content"]
                print(f"\nContent: {content}")
            else:
                print("\nNo content in response!")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_vlm_detailed())
