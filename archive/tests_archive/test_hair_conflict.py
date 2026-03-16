
import pytest
import asyncio
import sys
import os
from unittest.mock import MagicMock, patch

sys.path.append(os.getcwd())

from app.services.tag_recommender_service import TagRecommenderService
from app.services.tag_library_service import TagLibraryService
from app.services.tag_mapper import get_tag_mapper
from app.services.tag_conflict_resolver import get_conflict_resolver
from app.services.tag_relationship_graph import get_tag_relationship_graph

@pytest.mark.asyncio
async def test_hair_conflicts():
    service = TagRecommenderService()
    # Mock services
    service.tag_library = TagLibraryService()
    service.tag_library.tag_names = ["單馬尾", "雙馬尾", "短髮", "長髮"]
    service.tag_mapper = get_tag_mapper()
    service.conflict_resolver = get_conflict_resolver()
    service.relationship_graph = get_tag_relationship_graph()
    service.relationship_graph.build_default_graph()

    # Scenario 1: Short Hair (0.9) vs Ponytail (0.8)
    # VLM gives both. Short Hair should win.
    
    vlm_analysis = {
        "character_types": [],
        "clothing": [],
        "description": "Girl with short hair and ponytail." # VLM hallucination
    }
    
    # Mock keywords extraction
    with patch.object(service, '_extract_vlm_keywords', return_value=[("短髮", 0.9, "vlm"), ("單馬尾", 0.8, "vlm")]):
        with patch.object(service, '_extract_rag_tags', return_value=[]):
             with patch("app.services.tag_recommender_service.settings") as mock_settings:
                mock_settings.USE_LM_STUDIO = False
                
                recs = await service.recommend_tags(vlm_analysis, [], 10)
    
    tags = {r.tag for r in recs}
    print(f"\nScenario 1 Expected: {{'短髮'}} | Actual: {tags}")
    
    if "單馬尾" in tags and "短髮" in tags:
        print("FAIL: Conflict NOT resolved automatically.")
    else:
        print("PASS: Conflict resolved.")

    # Scenario 2: Twintails (0.9) vs Ponytail (0.6)
    vlm_analysis_2 = {
        "character_types": [],
        "clothing": [],
        "description": "Girl with twintails." 
    }
    
    with patch.object(service, '_extract_vlm_keywords', return_value=[("雙馬尾", 0.9, "vlm"), ("單馬尾", 0.6, "vlm")]):
        with patch.object(service, '_extract_rag_tags', return_value=[]):
             with patch("app.services.tag_recommender_service.settings") as mock_settings:
                mock_settings.USE_LM_STUDIO = False
                recs = await service.recommend_tags(vlm_analysis_2, [], 10)
                
    tags = {r.tag for r in recs}
    print(f"\nScenario 2 Expected: {{'雙馬尾'}} | Actual: {tags}")

if __name__ == "__main__":
    asyncio.run(test_hair_conflicts())
