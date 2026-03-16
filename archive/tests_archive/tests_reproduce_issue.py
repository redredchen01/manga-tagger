import asyncio
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.services.tag_recommender_service import TagRecommenderService, TagRecommendation
from app.config import settings

# Mocking necessary parts if needed, but we can try to use the real service if environment allows
# straightforward usage.

async def reproduce():
    print("Initializing TagRecommenderService...")
    service = TagRecommenderService()
    
    # Simulate a VLM analysis that finds "harem"
    # "Harem" (後宮) should be in the tag library.
    # We assume 'harem' maps to '後宮' or is recognized directly.
    vlm_analysis = {
        "description": "A scene showing a harem situation.",
        "themes": ["harem"], # Structured keyword
        "raw_keywords": ["harem"]
    }
    
    rag_matches = [] # No RAG matches
    
    print("\nRunning recommend_tags with 'harem' as VLM keyword...")
    recommendations = await service.recommend_tags(
        vlm_analysis=vlm_analysis,
        rag_matches=rag_matches,
        top_k=5,
        confidence_threshold=0.1 # Low threshold to see what we get
    )
    
    found = False
    for rec in recommendations:
        if "後宮" in rec.tag or "harem" in rec.tag.lower():
            print(f"\nFound Tag: {rec.tag}")
            print(f"Confidence: {rec.confidence}")
            print(f"Reason: {rec.reason}")
            found = True
            
            # Check if confidence is limited by alpha
            expected_max = settings.HYBRID_SCORING_ALPHA
            if abs(rec.confidence - expected_max) < 0.05:
                print(f"Issue Reproduced: Confidence {rec.confidence} is close to alpha {expected_max}")
                print("The pure lexical match is being pulled down by semantic/RAG zeros.")
            else:
                 print(f"Confidence {rec.confidence} is different from alpha {expected_max}")

    if not found:
        print("\nTarget tag not found in recommendations.")
        # Try to find what tags WERE recommended to understand why
        print("Top recommendations:")
        for rec in recommendations:
            print(f"- {rec.tag}: {rec.confidence}")

if __name__ == "__main__":
    asyncio.run(reproduce())
