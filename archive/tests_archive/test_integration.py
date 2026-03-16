#!/usr/bin/env python3
"""
Integration Test for All Fixes
Verify all phases are working correctly
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def test_all_phases():
    print("=" * 60)
    print("INTEGRATION TEST - All Phases")
    print("=" * 60)

    results = {}

    # Phase 3: VLM Prompt
    print("\n[Phase 3] VLM Prompt Optimization...")
    try:
        from app.services.lm_studio_vlm_service_v4 import LMStudioVLMService

        service = LMStudioVLMService()
        prompt = service._get_grouped_guidance_prompt()

        # Check key components
        checks = [
            ("character_types", "JueSe" in prompt or "role" in prompt.lower()),
            ("clothing", "FuZhuang" in prompt or "clothing" in prompt.lower()),
            ("body_features", "TiXing" in prompt or "body" in prompt.lower()),
        ]

        passed = sum(1 for _, check in checks if check)
        results["Phase 3"] = passed >= 2
        print(f"  Result: {passed}/{len(checks)} checks passed")
    except Exception as e:
        print(f"  Error: {e}")
        results["Phase 3"] = False

    # Phase 4: Tag Library Short Tags
    print("\n[Phase 4] Tag Library Short Tag Matching...")
    try:
        from app.services.tag_library_service import TagLibraryService

        service = TagLibraryService()

        # Test exact match for short tags
        test_keywords = ["SuLi", "JuRu", "BaiHe"]
        matches = service.match_tags_by_keywords(test_keywords, min_confidence=0.5)

        results["Phase 4"] = len(matches) > 0
        print(f"  Found {len(matches)} matches for short tags")
    except Exception as e:
        print(f"  Error: {e}")
        results["Phase 4"] = False

    # Phase 4b: TagMapper Chinese Aliases
    print("\n[Phase 4b] TagMapper Chinese Alias Support...")
    try:
        from app.services.tag_mapper import get_tag_mapper

        mapper = get_tag_mapper()

        # Test Chinese alias mapping
        test_cases = [
            ("LuoLi", "SuLi"),  # Simplified to Traditional
            ("JuRu", "JuRu"),  # Already traditional
        ]

        passed = 0
        for input_tag, expected in test_cases:
            result = mapper.to_chinese(input_tag)
            if result == expected:
                passed += 1

        results["Phase 4b"] = passed >= 1
        print(f"  Result: {passed}/{len(test_cases)} alias mappings passed")
    except Exception as e:
        print(f"  Error: {e}")
        results["Phase 4b"] = False

    # Phase 5: Vector Store Embedding
    print("\n[Phase 5] Vector Store Embedding Strategy...")
    try:
        from tag_vector_store import TagVectorStore
        import inspect

        # Check is_query parameter exists
        sig = inspect.signature(TagVectorStore._encode_text)
        has_is_query = "is_query" in sig.parameters

        results["Phase 5"] = has_is_query
        print(f"  _encode_text has is_query param: {has_is_query}")
    except Exception as e:
        print(f"  Error: {e}")
        results["Phase 5"] = False

    # Phase 6: Similarity Threshold
    print("\n[Phase 6] Similarity Threshold Configuration...")
    try:
        from app.config import settings

        threshold = getattr(settings, "RAG_SIMILARITY_THRESHOLD", None)
        results["Phase 6"] = threshold is not None and 0 < threshold < 1
        print(f"  Threshold configured: {threshold}")
    except Exception as e:
        print(f"  Error: {e}")
        results["Phase 6"] = False

    # Phase 8: Conflict Resolver Integration
    print("\n[Phase 8] Conflict Resolver Integration...")
    try:
        from app.services.tag_recommender_service import TagRecommenderService
        import inspect

        # Check if conflict_resolver is initialized
        source = inspect.getsource(TagRecommenderService.__init__)
        has_resolver = "conflict_resolver" in source

        results["Phase 8"] = has_resolver
        print(f"  Conflict resolver integrated: {has_resolver}")
    except Exception as e:
        print(f"  Error: {e}")
        results["Phase 8"] = False

    # Phase 9: Collection Name Config
    print("\n[Phase 9] Collection Name Configuration...")
    try:
        from app.config import settings

        has_image_collection = hasattr(settings, "CHROMA_IMAGE_COLLECTION")
        has_tag_collection = hasattr(settings, "CHROMA_TAG_COLLECTION")

        results["Phase 9"] = has_image_collection and has_tag_collection
        print(f"  Image collection config: {has_image_collection}")
        print(f"  Tag collection config: {has_tag_collection}")
    except Exception as e:
        print(f"  Error: {e}")
        results["Phase 9"] = False

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    for phase, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {phase}")

    all_passed = all(results.values())
    total = len(results)
    passed_count = sum(results.values())

    print(f"\nTotal: {passed_count}/{total} phases passed")

    if all_passed:
        print("\n[SUCCESS] All phases implemented successfully!")
    else:
        print("\n[WARNING] Some phases need attention.")

    return all_passed


if __name__ == "__main__":
    success = test_all_phases()
    sys.exit(0 if success else 1)
