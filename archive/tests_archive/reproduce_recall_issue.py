
import asyncio
import logging
from app.services.tag_recommender_service import TagRecommenderService
from app.models import VLMMetadata

# Configure logging
logging.basicConfig(level=logging.INFO)
logging.getLogger("app.services.tag_recommender_service").setLevel(logging.DEBUG)

async def reproduce_recall_issue():
    service = TagRecommenderService()
    
    # Mock VLM Analysis with "Bunny Girl" (High confidence) and "Bodysuit" (Medium confidence keywords)
    vlm_analysis = {
        "description": "A bunny girl in a black bodysuit with rabbit ears.",
        "character_types": ["bunny_girl"],
        "themes": [],
        "clothing": ["bodysuit", "pantyhose"],
        "body_features": ["rabbit_ears"],
        "raw_keywords": ["bunny", "girl", "bodysuit", "ears", "black"]
    }
    
    # Mock RAG Matches (Weak secondary matches)
    rag_matches = [
        {"score": 0.85, "tags": ["bunny_girl"]},    # Strong match
        {"score": 0.45, "tags": ["bodysuit"]},      # Weak match
        {"score": 0.40, "tags": ["rabbit_ears"]},   # Weak match
        {"score": 0.35, "tags": ["pantyhose"]}      # Very weak match
    ]
    
    print("--- Starting Recall Reproduction ---")
    
    # Run recommendation with current strict thresholds
    recommendations = await service.recommend_tags(
        vlm_analysis=vlm_analysis,
        rag_matches=rag_matches,
        top_k=10,
        confidence_threshold=0.5  # Current high threshold
    )
    
    print(f"\nTotal Recommendations: {len(recommendations)}")
    for rec in recommendations:
        print(f"- {rec.tag} ({rec.confidence:.2f}) [{rec.source}]")

if __name__ == "__main__":
    asyncio.run(reproduce_recall_issue())
