# Final integration test
import requests
import io
from PIL import Image, ImageDraw
import json


def create_test_image(color="red", tags_hint=None):
    """Create a test image with visual elements."""
    img = Image.new("RGB", (512, 512), color="white")
    draw = ImageDraw.Draw(img)

    # Draw face
    draw.ellipse([150, 100, 350, 300], outline="black", width=3)
    draw.ellipse([200, 160, 230, 190], fill="blue")
    draw.ellipse([270, 160, 300, 190], fill="blue")
    draw.arc([200, 200, 300, 280], start=0, end=180, fill="red", width=3)

    # Draw clothing based on color
    if color == "red":
        draw.rectangle([150, 300, 350, 480], outline="red", width=3)
    else:
        draw.rectangle([150, 300, 350, 480], outline="blue", width=3)

    if tags_hint:
        draw.text((180, 400), tags_hint, fill="black")

    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=85)
    buffer.seek(0)
    return buffer


def test_api():
    """Test the API endpoint."""
    print("=" * 70)
    print("FINAL INTEGRATION TEST - Manga Tagger API")
    print("=" * 70)

    # Test 1: Health check
    print("\n[1] Health Check...")
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            health = response.json()
            print(f"    Status: {health['status']}")
            print(f"    Tag Library: {health['models_loaded']['tag_library']} tags")
            print(f"    LM Studio Mode: {health['models_loaded']['lm_studio_mode']}")
    except Exception as e:
        print(f"    ERROR: {e}")
        return

    # Test 2: Tag an image
    print("\n[2] Tagging Image...")
    img_bytes = create_test_image("red", "Test Character")

    try:
        response = requests.post(
            "http://localhost:8000/tag-cover",
            files={"file": ("test.jpg", img_bytes, "image/jpeg")},
            data={"top_k": 5, "confidence_threshold": 0.3, "include_metadata": "true"},
            timeout=60,
        )

        if response.status_code == 200:
            result = response.json()
            print(f"    Status: 200 OK")
            print(f"    Tags Found: {len(result.get('tags', []))}")

            print("\n    Generated Tags:")
            for i, tag in enumerate(result.get("tags", []), 1):
                print(
                    f"      {i}. {tag['tag']} (confidence: {tag['confidence']:.2f}, source: {tag['source']})"
                )

            metadata = result.get("metadata", {})
            print(f"\n    Processing Time: {metadata.get('processing_time', 'N/A')}s")
            print(f"    RAG Matches: {metadata.get('rag_matches', 0)}")

            # Show VLM status
            vlm_desc = metadata.get("vlm_description", "")
            if "failed" in vlm_desc.lower():
                print(f"\n    ⚠️  VLM Status: Not working (using RAG fallback)")
            else:
                print(f"\n    ✓ VLM Status: Working")

            if result.get("tags"):
                print("\n" + "=" * 70)
                print("✅ SUCCESS! Tagging system is working!")
                print("=" * 70)
                return True
            else:
                print("\n    ⚠️  No tags generated")
                return False
        else:
            print(f"    ERROR: Status {response.status_code}")
            print(f"    {response.text[:200]}")
            return False

    except Exception as e:
        print(f"    ERROR: {e}")
        return False


if __name__ == "__main__":
    success = test_api()
    exit(0 if success else 1)
