# Test API endpoint with a simple image
import requests
import io
from PIL import Image
import json

# Create a simple test image
print("Creating test image...")
img = Image.new("RGB", (512, 512), color="red")
img_bytes = io.BytesIO()
img.save(img_bytes, format="JPEG")
img_bytes.seek(0)

# Test health endpoint
print("\n=== Testing Health Endpoint ===")
response = requests.get("http://localhost:8000/health", timeout=10)
print(f"Status: {response.status_code}")
health_data = response.json()
print(f"Health: {json.dumps(health_data, indent=2)}")

# Test tag-cover endpoint
print("\n=== Testing Tag Cover Endpoint ===")
files = {"file": ("test.jpg", img_bytes, "image/jpeg")}
data = {"top_k": 5, "confidence_threshold": 0.3, "include_metadata": "true"}

response = requests.post(
    "http://localhost:8000/tag-cover", files=files, data=data, timeout=60
)

print(f"Status: {response.status_code}")
if response.status_code == 200:
    result = response.json()
    print(f"\nResponse:")
    print(f"  Tags count: {len(result.get('tags', []))}")
    print(f"  Tags: {result.get('tags', [])}")
    print(f"\n  Metadata:")
    metadata = result.get("metadata", {})
    print(f"    VLM description: {metadata.get('vlm_description', 'N/A')[:100]}...")
    print(f"    RAG matches: {metadata.get('rag_matches', 0)}")
    vlm_analysis = metadata.get("vlm_analysis", {})
    print(f"    Character types: {vlm_analysis.get('character_types', [])}")
    print(f"    Clothing: {vlm_analysis.get('clothing', [])}")
    print(f"    Body features: {vlm_analysis.get('body_features', [])}")
else:
    print(f"Error: {response.text}")
