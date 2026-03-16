"""Test with a more realistic image."""

import sys

sys.path.insert(0, ".")

import asyncio
from PIL import Image, ImageDraw, ImageFont
import io

from app.services.lm_studio_vlm_service_v2 import LMStudioVLMService
from app.services.tag_library_service import get_tag_library_service
from app.services.tag_recommender_service import get_tag_recommender_service

print("=" * 70)
print("Test with Realistic Anime-style Image")
print("=" * 70)

# Create a more realistic test image with anime character elements
print("\n[1] Creating realistic test image...")
img = Image.new("RGB", (512, 768), color="pink")
draw = ImageDraw.Draw(img)

# Draw a simple anime-style face outline
draw.ellipse([150, 100, 350, 350], fill="#FFE4C4", outline="black", width=2)  # Face
draw.ellipse([200, 180, 230, 210], fill="white", outline="black", width=1)  # Left eye
draw.ellipse([270, 180, 300, 210], fill="white", outline="black", width=1)  # Right eye
draw.ellipse([210, 190, 220, 200], fill="blue")  # Left pupil
draw.ellipse([280, 190, 290, 200], fill="blue")  # Right pupil
draw.arc([220, 250, 280, 290], 0, 180, fill="red", width=2)  # Mouth

# Draw hair
draw.polygon(
    [(150, 200), (100, 50), (250, 100)], fill="#FFD700", outline="black"
)  # Left hair
draw.polygon(
    [(350, 200), (400, 50), (250, 100)], fill="#FFD700", outline="black"
)  # Right hair

# Draw school uniform
# White shirt
draw.rectangle([180, 350, 320, 550], fill="white", outline="black", width=2)
# Blue skirt
draw.polygon(
    [(180, 550), (320, 550), (340, 700), (160, 700)],
    fill="#4169E1",
    outline="black",
    width=2,
)
# Red ribbon
draw.polygon([(230, 380), (270, 380), (250, 420)], fill="red", outline="black", width=1)

# Draw cat ears
draw.polygon(
    [(170, 100), (130, 50), (190, 80)], fill="#FFD700", outline="black", width=2
)  # Left ear
draw.polygon(
    [(330, 100), (370, 50), (310, 80)], fill="#FFD700", outline="black", width=2
)  # Right ear
draw.ellipse([135, 45, 155, 65], fill="pink")  # Left inner ear
draw.ellipse([345, 45, 365, 65], fill="pink")  # Right inner ear

# Save and convert to bytes
buffer = io.BytesIO()
img.save(buffer, format="JPEG", quality=95)
img_bytes = buffer.getvalue()

# Save for visual inspection
img.save("test_anime.jpg")
print(f"    Image created: {len(img_bytes)} bytes")
print(f"    Saved as: test_anime.jpg")

# Test VLM extraction
print("\n[2] Testing VLM extraction...")
vlm = LMStudioVLMService()
vlm_result = asyncio.run(vlm.extract_metadata(img_bytes))

print(f"    Description: {vlm_result.get('description', 'N/A')[:200]}...")
print(f"    Character types: {vlm_result.get('character_types', [])}")
print(f"    Clothing: {vlm_result.get('clothing', [])}")
print(f"    Body features: {vlm_result.get('body_features', [])}")
print(f"    Actions: {vlm_result.get('actions', [])}")
print(f"    Themes: {vlm_result.get('themes', [])}")
print(
    f"    Raw keywords ({len(vlm_result.get('raw_keywords', []))}): {vlm_result.get('raw_keywords', [])[:10]}"
)

# Test tag library matching
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

# Test tag recommender
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
