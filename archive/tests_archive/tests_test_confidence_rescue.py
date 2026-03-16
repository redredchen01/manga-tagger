
import asyncio
import os
import sys

# Add the project root to the python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.services.tag_recommender_service import TagRecommenderService
from app.config import settings

async def test_confidence_rescue():
    # Force USE_LM_STUDIO to False for this test to isolate hybrid scoring
    settings.USE_LM_STUDIO = False
    
    service = TagRecommenderService()
    
    # Mock VLM analysis with a keyword that will match '後宮' (harem)
    vlm_analysis = {
        "description": "A classic harem story.",
        "character_types": ["harem"],
        "themes": [],
        "clothing": [],
        "body_features": [],
        "actions": []
    }
    
    print(f"Testing with HYBRID_SCORING_ALPHA = {settings.HYBRID_SCORING_ALPHA}")
    print("Running recommend_tags (LLM disabled)...")
    
    recs = await service.recommend_tags(vlm_analysis, [], top_k=5)
    
    harem_rec = next((r for r in recs if r.tag == "後宮"), None)
    
    if harem_rec:
        print(f"Found '後宮' with confidence {harem_rec.confidence}")
        print(f"Source: {harem_rec.source}")
        print(f"Reason: {harem_rec.reason}")
        
        if harem_rec.confidence > settings.HYBRID_SCORING_ALPHA:
            print("SUCCESS: Confidence Rescue is working! (Confidence > Alpha)")
        else:
            print("FAILURE: Confidence is still being pulled down to Alpha or below.")
    else:
        print("FAILURE: '後宮' not found in recommendations.")

if __name__ == "__main__":
    asyncio.run(test_confidence_rescue())
