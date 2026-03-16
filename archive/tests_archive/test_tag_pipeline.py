
import pytest
import asyncio
import sys
import os
from unittest.mock import MagicMock, AsyncMock

# Add parent directory to path
sys.path.append(os.getcwd())

from app.services.tag_recommender_service import TagRecommenderService, TagRecommendation
from app.services.tag_library_service import TagLibraryService
from app.services.tag_mapper import get_tag_mapper
from app.services.tag_conflict_resolver import get_conflict_resolver
from unittest.mock import patch

# Mock settings
with patch("app.services.tag_recommender_service.settings") as mock_settings:
    mock_settings.USE_LM_STUDIO = False

@pytest.mark.asyncio
async def test_smart_backfill_and_conflict():
    # Setup services
    tag_library = TagLibraryService()
    # Mock tag library to have our target tags
    tag_library.tag_names = [
        "紫髮", "少女", "校服", "年齡增長", "年齡回溯", "巨乳", "貧乳"
    ]
    tag_library.match_tags_by_keywords = MagicMock(return_value=[])
    tag_library.suggest_related_tags = MagicMock(return_value=[])
    
    # Initialize reconciler
    service = TagRecommenderService()
    service.tag_library = tag_library
    service.tag_mapper = get_tag_mapper()
    service.conflict_resolver = get_conflict_resolver()
    
    # Mock relationship graph to be passive
    service.relationship_graph = MagicMock()
    service.relationship_graph.validate_tag_combination.return_value.is_valid = True
    service.relationship_graph.validate_tag_combination.return_value.conflicts = []
    service.relationship_graph.validate_tag_combination.return_value.confidence_adjustments = {}
    service.relationship_graph.validate_tag_combination.return_value.recommendations = []

    # Scenario: VLM gives high confidence "Purple Hair" but missing others
    # Description contains keywords for "Girl" and "School Uniform"
    
    vlm_analysis = {
        "character_types": ["Person"],
        "clothing": [],
        "description": "A teenage school girl with long purple hair wearing a school uniform. She looks about 16 years old."
    }
    
    # Mock VLM extracted keywords to return Purple Hair
    service._extract_vlm_keywords = MagicMock(return_value=[("紫髮", 0.9, "vlm")])
    service._extract_rag_tags = MagicMock(return_value=[])

    # Run recommendation
    # Request 5 tags. We only have 1 from VLM keywords.
    # Expect Backfill to find "少女" (from 'school girl'/'teenage') and "校服" (from 'school uniform')
    
    recommendations = await service.recommend_tags(
        vlm_analysis=vlm_analysis,
        rag_matches=[],
        top_k=5,
        confidence_threshold=0.3
    )
    
    tags = {r.tag for r in recommendations}
    print(f"\nResult Tags: {tags}")
    
    # Assertions
    assert "紫髮" in tags, "Should have VLM keyword tag"
    assert "少女" in tags, "Should detect 'school girl' -> 少女"
    assert "校服" in tags, "Should detect 'school uniform' -> 校服"
    
    # Check for conflicts (Age)
    # Let's force a conflict scenario
    # Add "Age Regression" to the initial keywords to see if it gets removed/resolved
    service._extract_vlm_keywords = MagicMock(return_value=[
        ("紫髮", 0.9, "vlm"),
        ("年齡增長", 0.6, "vlm"),
        ("年齡回溯", 0.5, "vlm") # Should be removed by conflict resolver
    ])
    
    # Debug conflict resolver state
    print(f"\nLoaded Conflict Groups: {len(service.conflict_resolver.conflict_groups)}")
    # Check if Age Shift group is present
    age_groups = [g for g in service.conflict_resolver.conflict_groups if "年齡增長" in g.get("tags", [])]
    print(f"Age Conflict Groups found: {len(age_groups)}")
    for g in age_groups:
        print(f"  - Group: {g.get('tags')}")

    recommendations_conflict = await service.recommend_tags(
        vlm_analysis=vlm_analysis,
        rag_matches=[],
        top_k=5
    )
    
    conflict_tags = {r.tag for r in recommendations_conflict}
    print(f"\nConflict Test Tags: {conflict_tags}", flush=True)
    print(f"Recommendations: {[ (r.tag, r.confidence) for r in recommendations_conflict ]}", flush=True)
    
    if "年齡增長" not in conflict_tags:
        print("!! FAILURE: '年齡增長' missing from result !!")
        # Check why it might be missing
        # Manually run resolve check
        test_tags = ["紫髮", "年齡增長", "年齡回溯"]
        test_scores = {"紫髮": 0.9, "年齡增長": 0.6, "年齡回溯": 0.5}
        res, _ = service.conflict_resolver.resolve(test_tags, test_scores)
        print(f"Manual Resolve Check: {res}", flush=True)

    assert "年齡增長" in conflict_tags
    assert "年齡回溯" not in conflict_tags, "Lower confidence conflicting tag should be removed"
    assert len(conflict_tags) >= 3, "Should still backfill other tags"

if __name__ == "__main__":
    asyncio.run(test_smart_backfill_and_conflict())
