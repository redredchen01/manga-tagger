import asyncio
import sys
import os
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.services.tag_recommender_service import TagRecommenderService, TagRecommendation
from app.models import TagResult

async def test_rag_rescue():
    print("Testing RAG Rescue Logic...")
    
    # 1. Setup service
    service = TagRecommenderService()
    
    # 2. Mock VLM Analysis (generic)
    vlm_analysis = {
        "description": "A character is standing.",
        "character_types": ["少女"],
    }
    
    # 3. Mock RAG Matches (High confidence)
    # Use tags that ARE in the library
    rag_matches = [
        {
            "score": 0.98,
            "tags": ["女生制服", "人妻", "兔女郎"]
        }
    ]
    
    # 4. Mock LLM synthesis to be too strict (only one tag)
    mock_llm_results = [
        TagResult(tag="少女", confidence=0.9, source="lm_studio_llm", reason="VLM description")
    ]
    
    print("Mocking LLM synthesis to return sparse results...")
    from app.services.lm_studio_llm_service import LMStudioLLMService
    with patch.object(LMStudioLLMService, 'synthesize_tags', return_value=mock_llm_results):
        # Also mock the validator to always return True for simplicity
        with patch('app.services.tag_validator.TagValidator.validate_tag', return_value=True):
            with patch('app.services.tag_validator.TagValidator.check_conflicts', side_effect=lambda x: x):
                
                recommendations = await service.recommend_tags(
                    vlm_analysis=vlm_analysis,
                    rag_matches=rag_matches,
                    top_k=10,
                    confidence_threshold=0.5
                )
                
                print(f"\nFinal Recommendations ({len(recommendations)}):")
                found_tags = {r.tag for r in recommendations}
                for rec in recommendations:
                    print(f"- {rec.tag} ({rec.confidence:.2f}) [Source: {rec.source}]")
                
                # Assertions
                target_tags = ["少女", "女生制服", "人妻", "兔女郎"]
                missing = [t for t in target_tags if t not in found_tags]
                
                if not missing:
                    print("\nSUCCESS: All target tags recovered via RAG Rescue!")
                else:
                    print(f"\nFAILURE: Missing tags: {missing}")
                    sys.exit(1)

if __name__ == "__main__":
    asyncio.run(test_rag_rescue())
