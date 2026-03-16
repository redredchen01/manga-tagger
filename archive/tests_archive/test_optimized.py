"""Test the optimized VLM service."""

import sys

sys.path.insert(0, ".")

import asyncio
from PIL import Image, ImageDraw
import io

from app.services.lm_studio_vlm_service_v2 import LMStudioVLMService
from app.services.tag_library_service import get_tag_library_service
from app.services.tag_recommender_service import get_tag_recommender_service

print("=" * 70)
print("Test Optimized VLM Service")
print("=" * 70)

# Create test image with anime character
print("\n[1] Creating test image...")
img = Image.new("RGB", (512, 512), color="white")
draw = ImageDraw.Draw(img)

# Draw anime face
draw.ellipse([150, 100, 350, 350], fill="#FFE4C4", outline="black", width=2)
draw.ellipse([200, 180, 230, 210], fill="white", outline="black", width=1)
draw.ellipse([270, 180, 300, 210], fill="white", outline="black", width=1)
draw.ellipse([210, 190, 220, 200], fill="blue")
draw.ellipse([280, 190, 290, 200], fill="blue")
draw.arc([220, 250, 280, 290], 0, 180, fill="red", width=2)

# Cat ears
draw.polygon(
    [(170, 100), (130, 50), (190, 80)], fill="#FFD700", outline="black", width=2
)
draw.polygon(
    [(330, 100), (370, 50), (310, 80)], fill="#FFD700", outline="black", width=2
)

# School uniform
draw.rectangle([180, 350, 320, 550], fill="white", outline="black", width=2)
draw.polygon(
    [(180, 550), (320, 550), (340, 700), (160, 700)],
    fill="#4169E1",
    outline="black",
    width=2,
)
draw.polygon([(230, 380), (270, 380), (250, 420)], fill="red", outline="black", width=1)

buffer = io.BytesIO()
img.save(buffer, format="JPEG", quality=95)
img_bytes = buffer.getvalue()

print(f"    Image size: {len(img_bytes)} bytes")

# Test VLM extraction
print("\n[2] Testing VLM extraction...")
vlm = LMStudioVLMService()
vlm_result = asyncio.run(vlm.extract_metadata(img_bytes))

print(f"    Description: {vlm_result.get('description', 'N/A')[:100]}...")
print(f"    Character types: {vlm_result.get('character_types', [])}")
print(f"    Clothing: {vlm_result.get('clothing', [])}")
print(f"    Body features: {vlm_result.get('body_features', [])}")
print(f"    Actions: {vlm_result.get('actions', [])}")
print(f"    Themes: {vlm_result.get('themes', [])}")
print(
    f"    Raw keywords ({len(vlm_result.get('raw_keywords', []))}): {vlm_result.get('raw_keywords', [])}"
)

# Test tag matching
print("\n[3] Testing tag library matching...")
tag_lib = get_tag_library_service()
keywords = vlm_result.get("raw_keywords", [])

if keywords:
    matches = tag_lib.match_tags_by_keywords(keywords, min_confidence=0.5)
    print(f"    Matches found: {len(matches)}")
    for tag, conf in matches[:10]:
        print(f"      - {tag} ({conf:.2f})")
else:
    print("    No keywords to match!")

# Test recommender
print("\n[4] Testing tag recommender...")
recommender = get_tag_recommender_service()
recommendations = recommender.recommend_tags(
    vlm_analysis=vlm_result, rag_matches=[], top_k=5, confidence_threshold=0.3
)

print(f"    Recommendations: {len(recommendations)}")
for i, rec in enumerate(recommendations, 1):
    print(f"      {i}. {rec.tag} ({rec.confidence:.2f}) - {rec.source}")
    print(f"         Reason: {rec.reason}")

print("\n" + "=" * 70)
print("Test Complete")
print("=" * 70)
