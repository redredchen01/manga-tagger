"""Direct test of LM Studio API."""

import requests
import base64
from PIL import Image, ImageDraw
import io

print("=" * 70)
print("Direct LM Studio API Test")
print("=" * 70)

# Create a simple anime-style image
print("\n[1] Creating test image...")
img = Image.new("RGB", (512, 512), color="white")
draw = ImageDraw.Draw(img)

# Draw simple anime character
draw.ellipse([150, 100, 350, 350], fill="#FFE4C4", outline="black", width=2)  # Face
draw.ellipse([200, 180, 230, 210], fill="white", outline="black", width=1)  # Left eye
draw.ellipse([270, 180, 300, 210], fill="white", outline="black", width=1)  # Right eye
draw.ellipse([210, 190, 220, 200], fill="blue")  # Left pupil
draw.ellipse([280, 190, 290, 200], fill="blue")  # Right pupil
draw.arc([220, 250, 280, 290], 0, 180, fill="red", width=2)  # Mouth

# Cat ears
draw.polygon(
    [(170, 100), (130, 50), (190, 80)], fill="#FFD700", outline="black", width=2
)
draw.polygon(
    [(330, 100), (370, 50), (310, 80)], fill="#FFD700", outline="black", width=2
)

# Convert to base64
buffer = io.BytesIO()
img.save(buffer, format="JPEG", quality=95)
img_base64 = base64.b64encode(buffer.getvalue()).decode()

print(f"    Image size: {len(buffer.getvalue())} bytes")

# Build request
print("\n[2] Sending request to LM Studio...")

payload = {
    "model": "zai-org/glm-4.6v-flash",
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "List all anime/manga tags that apply to this image. Focus on: 1) Character type (loli, catgirl, etc.), 2) Clothing (school uniform, etc.), 3) Features (glasses, etc.). Return as a simple list.",
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"},
                },
            ],
        }
    ],
    "max_tokens": 512,
    "temperature": 0.7,
}

try:
    response = requests.post(
        "http://127.0.0.1:1234/v1/chat/completions", json=payload, timeout=60
    )

    print(f"    Status: {response.status_code}")

    if response.status_code == 200:
        result = response.json()
        content = result["choices"][0]["message"]["content"]
        print(f"\n    Response:")
        print(f"    {content}")
    else:
        print(f"    Error: {response.text[:500]}")

except Exception as e:
    print(f"    Error: {e}")

print("\n" + "=" * 70)
print("Test Complete")
print("=" * 70)
