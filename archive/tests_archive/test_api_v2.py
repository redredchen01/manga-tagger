"""Test the complete API with the new system."""

import requests
from PIL import Image
import io
import time

API_URL = "http://127.0.0.1:8000"

print("=" * 70)
print("API Integration Test - New Tag System")
print("=" * 70)

# Test 1: Health check
print("\n[Test 1] Health Check")
try:
    response = requests.get(f"{API_URL}/health", timeout=5)
    if response.status_code == 200:
        data = response.json()
        print(f"  Status: {data.get('status')}")
        print(f"  Version: {data.get('version')}")
        print(f"  Models: {data.get('models_loaded')}")
    else:
        print(f"  Error: {response.status_code}")
except Exception as e:
    print(f"  Error: {e}")
    print("  Make sure the API server is running!")
    exit(1)

# Test 2: Tag library info
print("\n[Test 2] Tag Library Info")
try:
    response = requests.get(f"{API_URL}/tags?limit=5", timeout=5)
    if response.status_code == 200:
        data = response.json()
        print(f"  Total tags available: {data.get('total')}")
        print(f"  Sample tags: {[t['tag_name'] for t in data.get('tags', [])[:3]]}")
    else:
        print(f"  Error: {response.status_code} - {response.text}")
except Exception as e:
    print(f"  Error: {e}")

# Test 3: Tag categories
print("\n[Test 3] Tag Categories")
try:
    response = requests.get(f"{API_URL}/tags/categories", timeout=5)
    if response.status_code == 200:
        data = response.json()
        print(f"  Categories: {data.get('categories')}")
        print(f"  Total: {data.get('total_tags')}")
    else:
        print(f"  Error: {response.status_code}")
except Exception as e:
    print(f"  Error: {e}")

# Test 4: Tag an image
print("\n[Test 4] Tag Cover Endpoint")
print("  Creating test image...")
img = Image.new("RGB", (224, 224), color="blue")
buffer = io.BytesIO()
img.save(buffer, format="JPEG")
img_bytes = buffer.getvalue()

print("  Sending request to /tag-cover...")
try:
    start_time = time.time()
    response = requests.post(
        f"{API_URL}/tag-cover",
        files={"file": ("test.jpg", img_bytes, "image/jpeg")},
        data={"top_k": "5", "confidence_threshold": "0.3", "include_metadata": "true"},
        timeout=60,
    )
    elapsed = time.time() - start_time

    print(f"  Response time: {elapsed:.2f}s")
    print(f"  Status: {response.status_code}")

    if response.status_code == 200:
        data = response.json()
        tags = data.get("tags", [])
        metadata = data.get("metadata", {})

        print(f"\n  Generated {len(tags)} tags:")
        for i, tag in enumerate(tags, 1):
            print(f"    {i}. {tag['tag']} ({tag['confidence']:.2f}) - {tag['source']}")

        print(f"\n  Metadata:")
        print(f"    Processing time: {metadata.get('processing_time', 'N/A')}s")
        print(
            f"    VLM characters: {metadata.get('vlm_analysis', {}).get('character_types', [])}"
        )
        print(
            f"    VLM clothing: {metadata.get('vlm_analysis', {}).get('clothing', [])}"
        )
        print(f"    RAG matches: {metadata.get('rag_matches', 0)}")
        print(f"    Library tags: {metadata.get('library_tags_available', 0)}")
    else:
        print(f"  Error: {response.text[:500]}")
except Exception as e:
    print(f"  Error: {e}")

# Test 5: RAG stats
print("\n[Test 5] RAG Stats")
try:
    response = requests.get(f"{API_URL}/rag/stats", timeout=5)
    if response.status_code == 200:
        data = response.json()
        print(f"  Documents: {data.get('total_documents')}")
        print(f"  Embedding mode: {data.get('embedding_mode')}")
        print(f"  Tag library: {data.get('tag_library_total')}")
    else:
        print(f"  Error: {response.status_code}")
except Exception as e:
    print(f"  Error: {e}")

print("\n" + "=" * 70)
print("API Test Completed!")
print("=" * 70)
