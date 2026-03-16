"""Test script that mimics the Streamlit flow."""

import requests
from PIL import Image, ImageDraw
import io

API_URL = "http://127.0.0.1:8000"

print("=" * 70)
print("STREAMLIT FLOW TEST")
print("=" * 70)

# Create test image (same as what user would upload)
print("\n[1] Creating test image (anime character)...")
img = Image.new("RGB", (512, 700), color="#FFB6C1")
draw = ImageDraw.Draw(img)
draw.ellipse([156, 150, 356, 400], fill="#FFE4C4", outline="black", width=2)
draw.ellipse([200, 220, 240, 260], fill="white", outline="black", width=1)
draw.ellipse([272, 220, 312, 260], fill="white", outline="black", width=1)
draw.ellipse([212, 230, 228, 252], fill="#4169E1")
draw.ellipse([284, 230, 300, 252], fill="#4169E1")
draw.arc([226, 310, 286, 350], 0, 180, fill="red", width=2)
draw.polygon(
    [(176, 150), (120, 60), (200, 110)], fill="#FFD700", outline="black", width=2
)
draw.polygon(
    [(336, 150), (392, 60), (312, 110)], fill="#FFD700", outline="black", width=2
)
draw.rectangle([180, 400, 332, 650], fill="white", outline="black", width=2)
draw.polygon(
    [(180, 550), (332, 550), (360, 700), (152, 700)],
    fill="#4169E1",
    outline="black",
    width=2,
)
draw.polygon([(236, 430), (276, 430), (256, 470)], fill="red", outline="black", width=1)

buffer = io.BytesIO()
img.save(buffer, format="JPEG", quality=95)
img_bytes = buffer.getvalue()
print(f"    Image created: {len(img_bytes)} bytes")

# Test different confidence thresholds
print("\n[2] Testing with different confidence thresholds...")

thresholds = [0.9, 0.5, 0.3, 0.1, 0.01]

for threshold in thresholds:
    response = requests.post(
        f"{API_URL}/tag-cover",
        files={"file": ("test.jpg", img_bytes, "image/jpeg")},
        data={
            "top_k": "10",
            "confidence_threshold": str(threshold),
            "include_metadata": "true",
        },
        timeout=90,
    )

    if response.status_code == 200:
        result = response.json()
        tags = result.get("tags", [])
        print(f"    Threshold {threshold}: {len(tags)} tags")
        if tags:
            print(f"       Top tag: {tags[0]['tag']} ({tags[0]['confidence']:.2f})")
    else:
        print(f"    Threshold {threshold}: ERROR {response.status_code}")

print("\n" + "=" * 70)
print("HOW TO USE:")
print("=" * 70)
print("""
1. Open http://127.0.0.1:8501 in browser
2. Upload an anime/manga cover image
3. Set confidence threshold to 0.1 (lower = more tags)
4. Click '開始標籤' button
5. Wait for AI analysis (5-10 seconds)
6. View generated tags on the right side

If still showing 0 tags:
- Check browser console for errors (F12)
- Refresh the page
- Try lowering confidence threshold to 0.1
- Make sure LM Studio is running on port 1234
""")
print("=" * 70)
