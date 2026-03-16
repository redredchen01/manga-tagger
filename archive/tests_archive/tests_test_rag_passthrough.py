import asyncio
import sys
import os
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.services.tag_recommender_service import TagRecommenderService
from app.models import TagResult

async def test_library_enforcement():
    print("Testing Library-Only Enforcement...")
    
    # 1. Setup service
    service = TagRecommenderService()
    print(f"Library loaded with {len(service.tag_library.tag_names)} tags.")
    
    # 2. Mock VLM Analysis (generic)
    vlm_analysis = {
        "description": "A character with red hair and school uniform.",
        "character_types": ["少女"],
    }
    
    # 3. Mock RAG Matches (High confidence)
    # Include tags that are NOW in the library
    rag_matches = [
        {
            "score": 0.98,
            "tags": ["紅髮", "校服", "bunny_girl"]
        }
    ]
    
    # 4. Mock LLM synthesis
    from app.services.lm_studio_llm_service import LMStudioLLMService
    
    async def mock_synthesize(*args, **kwargs):
        # Handle both positional and keyword arguments
        # args might be: (vlm_analysis, rag_matches, candidates, top_k)
        candidates = args[2] if len(args) > 2 else kwargs.get('candidates', [])
        print(f"LLM received candidates: {candidates}")
        results = []
        for c in candidates:
            # All these should be library tags now
            if c in ["紅髮", "女生制服", "兔女郎", "少女"]:
                results.append(TagResult(tag=c, confidence=0.9, source="lm_studio_llm", reason="Consistent"))
        return results

    with patch.object(LMStudioLLMService, 'synthesize_tags', side_effect=mock_synthesize):
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
                
                # Verify that ALL tags are in library
                # Correct import for library service
                from app.services.tag_library_service import get_tag_library_service
                library = get_tag_library_service()
                
                for tag in found_tags:
                    if tag in library.tag_names:
                        print(f"Verified: '{tag}' is in library.")
                    else:
                        print(f"FAILURE: '{tag}' is NOT in library but was recommended!")
                        sys.exit(1)
                
                # Check for "紅髮" (Now a library tag)
                if "紅髮" in found_tags:
                    print("SUCCESS: '紅髮' (now in library) correctly found!")
                else:
                    print("FAILURE: '紅髮' was not found.")
                    sys.exit(1)

if __name__ == "__main__":
    asyncio.run(test_library_enforcement())
