#!/usr/bin/env python3
"""
Comprehensive API Test Script
Tests the exact flow from image to tags
"""

import sys
import io
from PIL import Image
import base64
import requests
import json

sys.path.insert(0, ".")

print("=" * 80)
print("COMPREHENSIVE API TEST")
print("=" * 80)

# Step 1: Create test image
print("\n[1] Creating test image...")
img = Image.new("RGB", (512, 512), color="red")
img_bytes = io.BytesIO()
img.save(img_bytes, format="JPEG")
img_bytes.seek(0)
test_image = img_bytes.read()
print(f"    Image size: {len(test_image)} bytes")

# Step 2: Test VLM directly
print("\n[2] Testing VLM directly...")
base64_image = base64.b64encode(test_image).decode("utf-8")

url = "http://127.0.0.1:1234/v1/chat/completions"
headers = {"Authorization": "Bearer lm-studio", "Content-Type": "application/json"}

prompt = """Please analyze this manga cover image in detail and provide structured information in Chinese.

Analyze and categorize:
1. 角色特徵 (Character Features): Age, gender, special characteristics
2. 服裝與外觀 (Clothing & Appearance): Uniform, casual wear, special outfits
3. 場景與動作 (Scene & Action): Setting, pose, action, background elements
4. 風格與元素 (Style & Elements): Art style, mood, visual effects
5. 主题與類型 (Themes & Genres): Overall themes and genre indicators

Format your response as detailed analysis covering these categories."""

data = {
    "model": "zai-org/glm-4.6v-flash",
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                },
            ],
        }
    ],
    "max_tokens": 500,
    "temperature": 0.1,
}

response = requests.post(url, headers=headers, json=data, timeout=60)
result = response.json()

if "choices" in result:
    content = result["choices"][0]["message"]["content"]

    with open("vlm_direct_test.json", "w", encoding="utf-8") as f:
        json.dump(
            {"raw_content": content, "length": len(content)},
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"    VLM Response:")
    print(f"    Length: {len(content)} characters")
    preview = content[:100] + "..." if len(content) > 100 else content
    print(f"    Preview: {preview}")
else:
    print(f"    Error: {result}")
    sys.exit(1)

# Step 3: Test API endpoint
print("\n[3] Testing /tag-cover API endpoint...")
files = {"file": ("test.jpg", test_image, "image/jpeg")}
data = {"top_k": 5, "confidence_threshold": 0.3, "include_metadata": "true"}

response = requests.post(
    "http://localhost:8000/tag-cover", files=files, data=data, timeout=60
)

print(f"    Status: {response.status_code}")

api_result = response.json()

with open("api_test_result.json", "w", encoding="utf-8") as f:
    json.dump(api_result, f, ensure_ascii=False, indent=2)

tags_count = len(api_result.get("tags", []))
vlm_desc = api_result.get("metadata", {}).get("vlm_description", "N/A")

print(f"    Tags count: {tags_count}")
print(
    f"    VLM description: {vlm_desc[:50]}..."
    if len(vlm_desc) > 50
    else f"    VLM description: {vlm_desc}"
)

# Step 4: Analysis
print("\n[4] Analysis...")
if tags_count == 0:
    print("    ISSUE: No tags generated!")
    print(f"    VLM Description: [{vlm_desc}]")
    print(f"    VLM Description length: {len(vlm_desc)}")
else:
    print("    SUCCESS: Tags generated!")
    for tag in api_result.get("tags", [])[:5]:
        tag_name = tag.get("tag", "?")
        confidence = tag.get("confidence", 0)
        print(f"    - {tag_name}: {confidence:.2f}")

print("\n" + "=" * 80)
