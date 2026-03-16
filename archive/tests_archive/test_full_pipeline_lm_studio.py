"""Integration test for the full LM Studio tagging pipeline."""

import sys
import asyncio
import logging
from pathlib import Path

# Add the project root to path
sys.path.insert(0, str(Path(__file__).parent))

from app.services.tag_recommender_service import get_tag_recommender_service
from app.config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_full_recommendation():
    """Test full recommendation flow with LM Studio synthesis."""
    print("=" * 60)
    print("Testing Full Recommendation Pipeline (LM Studio Synthesis)")
    print("=" * 60)

    recommender = get_tag_recommender_service()

    # Mock VLM Analysis (Stage 1 result)
    vlm_analysis = {
        "description": "A high-quality manga cover showing a young girl with cat ears wearing a school uniform. She is holding a book in a library setting. The art style is clean and modern.",
        "character_types": ["蘿莉", "貓娘"],
        "clothing": ["校服"],
        "themes": ["學校", "校園"],
        "art_style": "Modern manga",
        "genre_indicators": ["戀愛", "喜劇"]
    }

    # Mock RAG Matches (Stage 2 result)
    rag_matches = [
        {
            "id": "match_1",
            "score": 0.85,
            "tags": ["少女", "可愛", "校服"],
            "metadata": {"title": "Similar Manga 1"}
        },
        {
            "id": "match_2",
            "score": 0.72,
            "tags": ["貓娘", "幻想"],
            "metadata": {"title": "Similar Manga 2"}
        }
    ]

    print("\nStarting recommendation with LM Studio synthesis...")
    print(f"USE_LM_STUDIO: {settings.USE_LM_STUDIO}")
    
    try:
        recommendations = await recommender.recommend_tags(
            vlm_analysis=vlm_analysis,
            rag_matches=rag_matches,
            top_k=5,
            confidence_threshold=0.3
        )

        print("\n" + "=" * 60)
        print("Final Recommendations:")
        print("=" * 60)
        
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. Tag: {rec.tag}")
            print(f"   Confidence: {rec.confidence:.2f}")
            print(f"   Source: {rec.source}")
            print(f"   Reason: {rec.reason}")
            print("-" * 30)

        if recommendations:
            print(f"\n✅ Successfully generated {len(recommendations)} recommendations.")
            # Check if source includes 'lm_studio_llm'
            has_llm = any(rec.source == "lm_studio_llm" for rec in recommendations)
            if has_llm:
                print("✅ LM Studio synthesis was used for the final tags.")
            else:
                print("⚠️ LM Studio synthesis results were not found in final tags. Check the service logic.")
            return True
        else:
            print("\n❌ No recommendations generated.")
            return False

    except Exception as e:
        print(f"\n❌ Error during recommendation: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_full_recommendation())
    sys.exit(0 if success else 1)
