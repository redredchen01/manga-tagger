#!/usr/bin/env python3
"""Start API server and test it."""

import subprocess
import time
import sys
import requests
from PIL import Image
import io
import os

print("=" * 60)
print("Starting API Server and Testing")
print("=" * 60)

# Start server in background
print("\n[1] Starting API server...")
server_process = subprocess.Popen(
    [
        sys.executable,
        "-m",
        "uvicorn",
        "app.main:app",
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
    ],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    cwd="C:\\tagger",
)

# Wait for server to start
print("[2] Waiting for server to start...")
time.sleep(5)

# Test health endpoint
print("\n[3] Testing health endpoint...")
try:
    response = requests.get("http://localhost:8000/health", timeout=5)
    print(f"    Status: {response.status_code}")
    print(f"    Response: {response.json()}")
except Exception as e:
    print(f"    Error: {e}")
    server_process.terminate()
    sys.exit(1)

# Create test image
print("\n[4] Creating test image...")
img = Image.new("RGB", (224, 224), color="blue")
buffer = io.BytesIO()
img.save(buffer, format="JPEG")
img_bytes = buffer.getvalue()
print(f"    Test image created: {len(img_bytes)} bytes")

# Test tag-cover endpoint
print("\n[5] Testing /tag-cover endpoint...")
try:
    response = requests.post(
        "http://localhost:8000/tag-cover",
        files={"file": ("test.jpg", img_bytes, "image/jpeg")},
        data={"top_k": "3", "confidence_threshold": "0.3", "include_metadata": "true"},
        timeout=60,
    )
    print(f"    Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print("    SUCCESS!")
        print(f"    Tags: {result.get('tags', [])}")
        print(
            f"    VLM Description: {result.get('metadata', {}).get('vlm_description', 'N/A')[:100]}..."
        )
    else:
        print(f"    Error: {response.text[:500]}")
except Exception as e:
    print(f"    Error: {e}")

# Cleanup
print("\n[6] Stopping server...")
server_process.terminate()
server_process.wait()
print("    Server stopped")

print("\n" + "=" * 60)
print("Test complete")
print("=" * 60)
