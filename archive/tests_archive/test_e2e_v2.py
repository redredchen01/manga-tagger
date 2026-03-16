# End-to-End Pipeline Test
import sys
import asyncio
import io
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))

# Enable mock mode
import app.config

app.config.settings.USE_MOCK_SERVICES = True

from app.services.lm_studio_vlm_service_v2 import LMStudioVLMService
from app.services.tag_recommender_service import get_tag_recommender_service


async def test_complete_pipeline():
    print("=" * 60)
    print("[MOCK MODE] Tagging System Test")
    print("=" * 60)

    # 1. Create test image
    print("\n[1/3] Creating test image...")
    img = Image.new("RGB", (512, 512), color=(255, 100, 150))
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="JPEG")
    image_data = img_bytes.getvalue()
    print(f"      Image size: {len(image_data)} bytes")

    # 2. VLM Analysis
    print("\n[2/3] VLM Image Analysis...")
    vlm_service = LMStudioVLMService()
    vlm_result = await vlm_service.extract_metadata(image_data)
    print(f"      Description: {vlm_result.get('description', 'N/A')[:60]}...")
    print(f"      Character types: {vlm_result.get('character_types', [])}")
    print(f"      Clothing: {vlm_result.get('clothing', [])}")
    print(f"      Body features: {vlm_result.get('body_features', [])}")
    print(f"      Raw keywords: {vlm_result.get('raw_keywords', [])}")

    # 3. Tag Recommendation
    print("\n[3/3] Tag Recommendation...")
    recommender = get_tag_recommender_service()
    recommendations = await recommender.recommend_tags(
        vlm_analysis=vlm_result,
        rag_matches=[],
        top_k=5,
        confidence_threshold=0.3,
        vlm_service=vlm_service,
        image_bytes=image_data,
    )

    print("\n" + "=" * 60)
    print("FINAL RECOMMENDED TAGS")
    print("=" * 60)
    if recommendations:
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec.tag} (confidence: {rec.confidence:.2f})")
            print(f"     source: {rec.source}")
            print(f"     reason: {rec.reason[:50]}...")
    else:
        print("  [ERROR] No tags recommended!")

    print("\n" + "=" * 60)
    if recommendations:
        print("[OK] Tagging system is working!")
    else:
        print("[ERROR] Tagging system needs debugging")
    print("=" * 60)

    return len(recommendations) > 0


if __name__ == "__main__":
    success = asyncio.run(test_complete_pipeline())
    sys.exit(0 if success else 1)
