"""Test script for the new tag recommendation system."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import asyncio
from PIL import Image
import io

from app.services.tag_library_service import get_tag_library_service
from app.services.tag_recommender_service import get_tag_recommender_service
from app.services.lm_studio_vlm_service_v2 import LMStudioVLMService

print("=" * 70)
print("Tag Library & Recommendation System Test")
print("=" * 70)

# Test 1: Tag Library Loading
print("\n[Test 1] Tag Library Loading")
tag_lib = get_tag_library_service()
print(f"  Total tags loaded: {len(tag_lib.tag_names)}")
print(f"  Character tags: {len(tag_lib.tag_categories['character'])}")
print(f"  Clothing tags: {len(tag_lib.tag_categories['clothing'])}")
print(f"  Body tags: {len(tag_lib.tag_categories['body'])}")
print(f"  Action tags: {len(tag_lib.tag_categories['action'])}")
print(f"  Theme tags: {len(tag_lib.tag_categories['theme'])}")
print(f"  Other tags: {len(tag_lib.tag_categories['other'])}")

# Test 2: Tag Search
print("\n[Test 2] Tag Search")
results = tag_lib.search_tags("loli", limit=5)
print(f"  Search 'loli': {results}")

results = tag_lib.search_tags("school", limit=5)
print(f"  Search 'school': {results}")

# Test 3: Tag Matching
print("\n[Test 3] Tag Matching by Keywords")
keywords = ["loli", "catgirl", "school uniform", "big breasts"]
matches = tag_lib.match_tags_by_keywords(keywords, min_confidence=0.6)
print(f"  Keywords: {keywords}")
print(f"  Matches: {matches[:5]}")

# Test 4: Tag Recommender
print("\n[Test 4] Tag Recommender")
recommender = get_tag_recommender_service()

# Mock VLM analysis
vlm_analysis = {
    "description": "A young catgirl in school uniform with glasses",
    "character_types": ["loli", "catgirl"],
    "clothing": ["school uniform", "glasses"],
    "body_features": ["flat chest"],
    "actions": [],
    "themes": ["school", "romance"],
    "settings": ["classroom"],
    "raw_keywords": [
        "loli",
        "catgirl",
        "school uniform",
        "glasses",
        "flat chest",
        "classroom",
    ],
}

rag_matches = [
    {"score": 0.85, "tags": ["蘿莉", "校服"]},
    {"score": 0.75, "tags": ["貓娘", "眼鏡"]},
]

recommendations = recommender.recommend_tags(
    vlm_analysis=vlm_analysis,
    rag_matches=rag_matches,
    top_k=5,
    confidence_threshold=0.5,
)

print(f"  Input keywords: {vlm_analysis['raw_keywords']}")
print(f"  Recommendations:")
for i, rec in enumerate(recommendations, 1):
    print(f"    {i}. {rec.tag} ({rec.confidence:.2f}) - {rec.source}")
    print(f"       Reason: {rec.reason}")

# Test 5: VLM Service (if LM Studio is available)
print("\n[Test 5] VLM Service")
try:
    vlm = LMStudioVLMService()
    print(f"  VLM Service initialized: {vlm.model}")

    # Create a test image
    img = Image.new("RGB", (224, 224), color="red")
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    img_bytes = buffer.getvalue()

    print("  Testing VLM extraction...")
    result = asyncio.run(vlm.extract_metadata(img_bytes))

    print(f"  Description: {result.get('description', 'N/A')[:100]}...")
    print(f"  Character types: {result.get('character_types', [])}")
    print(f"  Clothing: {result.get('clothing', [])}")
    print(f"  Keywords extracted: {len(result.get('raw_keywords', []))}")

except Exception as e:
    print(f"  VLM test skipped: {e}")

print("\n" + "=" * 70)
print("All tests completed!")
print("=" * 70)
