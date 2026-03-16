import asyncio
import logging
import sys
from typing import Dict, Any

# Mocking settings and services to test logic
from app.services.tag_recommender_service import get_tag_recommender_service
from app.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VerifyPrecision")

async def test_precision_logic():
    print("\n--- Testing Tag Precision Optimization Logic ---\n")
    
    recommender = get_tag_recommender_service()
    
    # Test Case 1: Hybrid Scoring & Category Boosting
    # Description says "猫耳" (Cat ears), lexical match should be high, 
    # and character category (猫娘) should get boosted.
    vlm_analysis = {
        "description": "A character with cat ears wearing a school uniform.",
        "character_types": ["cat girl"],
        "clothing": ["school uniform"],
        "body_features": ["cat ears"],
        "actions": [],
        "themes": []
    }
    
    print(f"Testing Analysis: {vlm_analysis['description']}")
    
    # Note: Using mock_rag_matches to simulate database results
    mock_rag_matches = [
        {"score": 0.85, "tags": ["女生制服", "貓娘"]},
        {"score": 0.70, "tags": ["少女"]}
    ]
    
    # Run recommendation (LM Studio synthesis will be bypassed if not running or mocked)
    results = await recommender.recommend_tags(
        vlm_analysis=vlm_analysis,
        rag_matches=mock_rag_matches,
        top_k=5,
        confidence_threshold=0.3
    )
    
    print("\nRecommended Tags (Top 5):")
    for i, res in enumerate(results, 1):
        print(f"{i}. {res.tag} (Conf: {res.confidence:.2f}) - {res.reason}")

    # Verify if "貓娘" and "女生制服" are present and have high scores
    found_cat_girl = any(r.tag == "貓娘" for r in results)
    found_uniform = any(r.tag == "女生制服" for r in results)
    
    if found_cat_girl and found_uniform:
        print("\n✅ SUCCESS: Core tags '貓娘' and '女生制服' correctly identified.")
    else:
        print("\n❌ FAILURE: Missing core tags.")

    # Test Case 2: Sensitive Tag Verification (Logic Check)
    print("\n--- Testing Sensitive Tag Verification ---")
    # We will simulate a sensitive tag and see if it triggers the check
    # (Checking if 'anal' is filtered if not in description)
    # Since we can't easily mock the VLM service's vision check here, 
    # we just check if any sensitive tags appear without evidence.
    
    vlm_analysis_clean = {
        "description": "Safe image of a girl reading a book.",
        "character_types": ["girl"],
        "clothing": [],
        "body_features": [],
        "actions": [],
        "themes": []
    }
    
    # RAG match contains a false positive sensitive tag
    mock_rag_dirty = [{"score": 0.9, "tags": ["肛交"]}] 
    
    results_safe = await recommender.recommend_tags(
        vlm_analysis=vlm_analysis_clean,
        rag_matches=mock_rag_dirty,
        top_k=5,
        confidence_threshold=0.5
    )
    
    has_sensitive = any(r.tag == "肛交" for r in results_safe)
    if not has_sensitive:
        print("✅ SUCCESS: Sensitive tag '肛交' correctly filtered/not matched without evidence.")
    else:
        print("⚠️ WARNING: Sensitive tag '肛交' appeared. Check scoring logic.")

if __name__ == "__main__":
    asyncio.run(test_precision_logic())
