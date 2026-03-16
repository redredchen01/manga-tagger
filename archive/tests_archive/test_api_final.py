# Test API endpoint
import requests
import io
from PIL import Image, ImageDraw

# Create test image
img = Image.new("RGB", (512, 512), color="white")
draw = ImageDraw.Draw(img)
draw.ellipse([150, 100, 350, 300], outline="black", width=3)
draw.ellipse([200, 160, 230, 190], fill="blue")
draw.ellipse([270, 160, 300, 190], fill="blue")
draw.rectangle([150, 300, 350, 480], outline="blue", width=2)
img_bytes = io.BytesIO()
img.save(img_bytes, format="JPEG")
img_bytes.seek(0)

# Test API
print("Testing API endpoint...")
response = requests.post(
    "http://localhost:8000/tag-cover",
    files={"file": ("test.jpg", img_bytes, "image/jpeg")},
    data={"top_k": 5, "confidence_threshold": 0.3, "include_metadata": "true"},
    timeout=120,
)

print(f"Status: {response.status_code}")
if response.status_code == 200:
    result = response.json()
    print(f"\nTags count: {len(result.get('tags', []))}")
    for tag in result.get("tags", []):
        print(f"  - {tag['tag']}: {tag['confidence']:.2f}")

    if result.get("tags"):
        print("\nSUCCESS! API is returning tags!")
    else:
        print("\nNo tags returned")
else:
    print(f"Error: {response.text}")
