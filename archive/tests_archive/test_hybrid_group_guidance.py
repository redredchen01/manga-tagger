import asyncio
import logging
import json
import os
import sys

# Add current directory to path
sys.path.append(os.getcwd())

from app.services.lm_studio_vlm_service_v4 import LMStudioVLMService
from app.services.tag_recommender_service import TagRecommenderService
from app.services.lm_studio_vlm_service_v4 import LMStudioVLMService
from app.services.tag_recommender_service import TagRecommenderService
from app.services.rag_service import RAGService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_hybrid_group_guidance():
    """Test script for Hybrid Group Guidance tagging."""
    
    # 1. Initialize services
    logger.info("Initializing services...")
    try:
        vlm_service = LMStudioVLMService()
        recommender = TagRecommenderService()
        rag_service = RAGService()
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        # Continue with fallback if possible
        vlm_service = LMStudioVLMService()
        recommender = TagRecommenderService()
        rag_service = None
    
    # 2. Load test image
    test_image_path = "test_anime.jpg"
    if not os.path.exists(test_image_path):
        # Try finding any jpg in current dir
        import glob
        jpgs = glob.glob("*.jpg")
        if jpgs:
            test_image_path = jpgs[0]
            logger.info(f"Using alternative test image: {test_image_path}")
        else:
            logger.error("No test image found.")
            return
        
    with open(test_image_path, "rb") as f:
        image_bytes = f.read()
        
    logger.info("--- Step 1: VLM Analysis (Grouped Guidance) ---")
    vlm_analysis = await vlm_service.extract_metadata(image_bytes)
    print("\nVLM Grouped Results:")
    print(json.dumps(vlm_analysis, indent=2, ensure_ascii=False))
    
    logger.info("\n--- Step 2: RAG Matching ---")
    rag_matches = []
    if rag_service:
        try:
            rag_matches = await rag_service.search_similar(image_bytes, top_k=2)
            print(f"Found {len(rag_matches)} RAG matches")
        except Exception as e:
            logger.warning(f"RAG matching failed: {e}")
    else:
        logger.warning("RAG service not available, skipping RAG step.")
    
    logger.info("\n--- Step 3: Tag Recommendation (Parallel Matching + Group Boost) ---")
    recommendations = await recommender.recommend_tags(
        vlm_analysis=vlm_analysis,
        rag_matches=rag_matches,
        top_k=10,
        image_bytes=image_bytes
    )
    
    print("\nFinal Recommended Tags:")
    if not recommendations:
        print("No tags recommended.")
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec.tag} (置信度: {rec.confidence:.2f})")
        print(f"   來源: {rec.source}")
        print(f"   原因: {rec.reason}")

if __name__ == "__main__":
    asyncio.run(test_hybrid_group_guidance())
