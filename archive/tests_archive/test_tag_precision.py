
import pytest
import asyncio
import sys
import os
from unittest.mock import MagicMock, patch

# Add parent directory to path
sys.path.append(os.getcwd())

from app.services.tag_recommender_service import TagRecommenderService
from app.services.tag_library_service import TagLibraryService
from app.services.tag_mapper import get_tag_mapper
from app.services.tag_conflict_resolver import get_conflict_resolver
from app.services.tag_relationship_graph import get_tag_relationship_graph

@pytest.mark.asyncio
async def test_precision_upgrade():
    # Setup services
    tag_library = TagLibraryService()
    # Mock tag library to have our target tags
    tag_library.tag_names = [
        "女僕", "圍裙", "洋裝", "絲襪", # Maid related
        "校服", "學園", # School related
        "貓娘", "獸耳" # Catgirl related
    ]
    tag_library.match_tags_by_keywords = MagicMock(return_value=[])
    tag_library.suggest_related_tags = MagicMock(return_value=[])
    
    # Initialize reconciler
    service = TagRecommenderService()
    service.tag_library = tag_library
    service.tag_mapper = get_tag_mapper()
    service.conflict_resolver = get_conflict_resolver()
    service.relationship_graph = get_tag_relationship_graph()
    
    # Reset graph and build defaults to ensure new rules are present
    service.relationship_graph.build_default_graph()

    # Scenario 1: Maid -> Boost Apron
    # VLM gives "Maid" (0.9) and "Apron" (0.4)
    # "Apron" 0.4 is below default threshold 0.5.
    # But "Maid" implies "Apron" (confidence 0.85).
    # Confidence adjustment should boost Apron.
    
    # Mock VLM keywords
    # We must mock _extract_vlm_keywords to return our test scenario
    mock_keywords = [
        ("女僕", 0.9, "vlm"),
        ("圍裙", 0.4, "vlm"), 
        ("洋裝", 0.7, "vlm")
    ]
    
    # Need to mock the pipeline to accept low confidence tags initially?
    # No, matched_tags filters by threshold.
    # Wait, recommend_tags filters by threshold at step 10.
    # Adjustments happen at step 11 (after filtering).
    # So "Apron" must survive initial threshold to be boosted.
    # Or we test "Apron" (0.6) -> Boost to 0.7?
    
    # Let's test AUTO-ADD first.
    # Scenario 2: "School Uniform" (0.9) -> Auto-add "School" (missing)
    
    vlm_analysis = {
        "character_types": ["Person"],
        "clothing": [],
        "description": "A girl in school uniform."
    }
    
    with patch.object(service, '_extract_vlm_keywords', return_value=[("校服", 0.9, "vlm"), ("少女", 0.8, "vlm")]):
        with patch.object(service, '_extract_rag_tags', return_value=[]):
             with patch("app.services.tag_recommender_service.settings") as mock_settings:
                mock_settings.USE_LM_STUDIO = False
                # Debug: Check graph suggestions directly
                print(f"\nDirect Graph Suggestions: {service.relationship_graph._suggest_related_tags(['校服'])}")

                recommendations = await service.recommend_tags(
                    vlm_analysis=vlm_analysis,
                    rag_matches=[],
                    top_k=10,
                    confidence_threshold=0.5
                )

                # Debug: Check graph state
                print(f"\nRelated to '校服': {service.relationship_graph.get_related_tags('校服')}")
                
    tags = {r.tag for r in recommendations}
    print(f"\nAuto-Add Test Tags: {tags}")
    
    assert "校服" in tags
    assert "學園" in tags, "Should auto-add 'School' implied by 'School Uniform'"
    
    # Check source of "學園"
    school_rec = next(r for r in recommendations if r.tag == "學園")
    assert school_rec.source == "relationship_graph"
    print(f"School Tag Source: {school_rec.source}")

    # Scenario 3: Confidence Boost
    # "Maid" (0.9) and "Apron" (0.6).
    # Expect Apron confidence > 0.6
    
    vlm_analysis_maid = {
        "character_types": [],
        "clothing": [],
        "description": "A maid in apron."
    }
    
    with patch.object(service, '_extract_vlm_keywords', return_value=[("女僕", 0.9, "vlm"), ("圍裙", 0.6, "vlm")]):
        with patch.object(service, '_extract_rag_tags', return_value=[]):
             with patch("app.services.tag_recommender_service.settings") as mock_settings:
                mock_settings.USE_LM_STUDIO = False
                
                recs_boost = await service.recommend_tags(
                    vlm_analysis=vlm_analysis_maid,
                    rag_matches=[],
                    top_k=10
                )
    
    print(f"\nBoost Test Tags: {[r.tag for r in recs_boost]}")
    apron_rec = next(r for r in recs_boost if r.tag == "圍裙")
    print(f"\nOriginal Apron Conf: 0.6, Final: {apron_rec.confidence}")
    assert apron_rec.confidence > 0.601, "Should be boosted by Maid implication"

if __name__ == "__main__":
    asyncio.run(test_precision_upgrade())
