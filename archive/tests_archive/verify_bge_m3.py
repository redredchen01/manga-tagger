import asyncio
import logging
import sys
import os

# Add parent directory to path to import app modules
sys.path.append(os.getcwd())

from app.services.chinese_embedding_service import get_chinese_embedding_service
from app.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def verify_bge_m3():
    print(f"--- Verifying BGE-M3 Upgrade ---")
    print(f"Model Name: {settings.CHINESE_EMBEDDING_MODEL}")
    print(f"Target Dimension: {settings.CHINESE_EMBEDDING_DIM}")
    
    service = get_chinese_embedding_service()
    
    if not service.is_available():
        print("Error: Chinese Embedding Service is not available.")
        return
        
    print(f"Actual Dimension: {service.target_dim}")
    
    # Test semantic similarity
    # Magical Girl (English) -> 魔法少女 (Chinese)
    test_tags = ["魔法少女", "貓娘", "校服", "人妻", "巨乳"]
    print(f"Caching test tags: {test_tags}")
    await service.cache_tag_embeddings(test_tags)
    
    queries = ["magical girl", "cat girl", "JK", "mature woman"]
    
    print("\nTesting Semantic Search:")
    for query in queries:
        results = await service.search_cached_tags(query, top_k=1, threshold=0.1)
        if results:
            match = results[0]
            print(f"Query: '{query}' -> Match: '{match['tag']}' (Score: {match['similarity']:.4f})")
        else:
            print(f"Query: '{query}' -> No match found")

if __name__ == "__main__":
    asyncio.run(verify_bge_m3())
