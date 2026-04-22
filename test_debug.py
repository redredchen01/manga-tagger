"""Quick debug test for tag-cover endpoint."""

import json
import os
import time

import requests

# Default API base URL - can be overridden via environment variable
API_HOST = os.environ.get("API_HOST", "127.0.0.1")
API_PORT = os.environ.get("API_PORT", "8000")
BASE_URL = f"http://{API_HOST}:{API_PORT}/api/v1"

# Test health
print("=== Health Check ===")
r = requests.get(f"{BASE_URL}/health")
print(f"Status: {r.status_code}")
print(json.dumps(r.json(), indent=2, ensure_ascii=False))

# Test tag-cover
print("\n=== Tag Cover ===")
start = time.time()
with open("test_anime.jpg", "rb") as f:
    files = {"file": ("test_anime.jpg", f, "image/jpeg")}
    data = {"top_k": 5, "confidence_threshold": 0.5, "include_metadata": "true"}
    r = requests.post(f"{BASE_URL}/tag-cover", files=files, data=data)
elapsed = time.time() - start
print(f"Status: {r.status_code} (took {elapsed:.2f}s)")
if r.status_code == 200:
    print(f"Tags found: {len(r.json().get('tags', []))}")

# Test generate-manga-description
print("\n=== Generate Description ===")
with open("test_anime.jpg", "rb") as f:
    files = {"file": ("test_anime.jpg", f, "image/jpeg")}
    data = {"include_metadata": "true"}
    r = requests.post(f"{BASE_URL}/generate-manga-description", files=files, data=data)
print(f"Status: {r.status_code}")
if r.status_code == 200:
    print(f"Description: {r.json().get('description')[:100]}...")

# Test upload (RAG add)
print("\n=== Upload (RAG Add) ===")
with open("test_anime.jpg", "rb") as f:
    files = {"file": ("test_anime.jpg", f, "image/jpeg")}
    data = {"tags": '["test", "tag"]', "metadata": '{"source": "test_script"}'}
    r = requests.post(f"{BASE_URL}/upload", files=files, data=data)
print(f"Status: {r.status_code}")
if r.status_code == 200:
    print(f"Response: {r.json().get('message')}")
