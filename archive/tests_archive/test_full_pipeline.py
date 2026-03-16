# Full end-to-end test of tagging system
import asyncio
import io
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from PIL import Image, ImageDraw
from app.services.lm_studio_vlm_service_v2 import LMStudioVLMService
from app.services.tag_recommender_service import get_tag_recommender_service


def create_anime_test_image():
    """Create a test image resembling anime character."""
    img = Image.new("RGB", (512, 512), color="white")
    draw = ImageDraw.Draw(img)

    # Draw anime-style face
    draw.ellipse([150, 100, 350, 300], outline="black", width=3)  # Face outline
    draw.ellipse([200, 160, 230, 190], fill="blue")  # Left eye
    draw.ellipse([270, 160, 300, 190], fill="blue")  # Right eye
    draw.arc([200, 200, 300, 280], start=0, end=180, fill="red", width=3)  # Smile

    # Add clothing hint
    draw.rectangle([150, 300, 350, 480], outline="blue", width=2)  # School uniform body

    draw.text((180, 350), "Anime Girl", fill="black")

    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=85)
    return buffer.getvalue()


async def test_full_pipeline():
    """Test complete tagging pipeline."""
    print("=" * 60)
    print("FULL PIPELINE TEST")
    print("=" * 60)

    # Create test image
    print("\n[1] Creating test image...")
    image_data = create_anime_test_image()
    print(f"    Image size: {len(image_data)} bytes")

    # Step 1: VLM extraction
    print("\n[2] VLM Analysis...")
    vlm = LMStudioVLMService()
    vlm_result = await vlm.extract_metadata(image_data)

    print(f"    VLM Keywords: {vlm_result.get('raw_keywords', [])}")
    print(f"    Character types: {vlm_result.get('character_types', [])}")
    print(f"    Clothing: {vlm_result.get('clothing', [])}")

    # Step 2: Tag recommendation
    print("\n[3] Tag Recommendation...")
    recommender = get_tag_recommender_service()

    recommendations = recommender.recommend_tags(
        vlm_analysis=vlm_result, rag_matches=[], top_k=5, confidence_threshold=0.3
    )

    print(f"\n    Recommended tags ({len(recommendations)}):")
    for rec in recommendations:
        print(f"      - {rec.tag}: {rec.confidence:.2f} ({rec.source})")

    # Summary
    print("\n" + "=" * 60)
    if recommendations:
        print(f"SUCCESS! Generated {len(recommendations)} tag recommendations")
        print("\nThe tagging system is now WORKING!")
    else:
        print("WARNING: No recommendations generated")
        print("This could be due to:")
        print("  - VLM keywords not matching library tags")
        print("  - Confidence threshold too high")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(test_full_pipeline())
