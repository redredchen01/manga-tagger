#!/usr/bin/env python3
"""
Emergency Fix Validation Test
Test Phase 1 fixes
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def test_tag_mapper():
    print("=" * 60)
    print("Test 1: TagMapper - Chinese/English Mapping")
    print("=" * 60)

    from app.services.tag_mapper import get_tag_mapper

    mapper = get_tag_mapper()

    test_cases = [
        ("loli", "SuLi"),
        ("catgirl", "MaoNiang"),
        ("JuRu", "JuRu"),
        ("LuoLi", "SuLi"),
        ("PinRu", "PinRu"),
        ("BaiHe", "BaiHe"),
    ]

    passed = 0
    failed = 0

    for input_tag, expected in test_cases:
        result = mapper.to_chinese(input_tag)
        status = "OK" if result == expected else "FAIL"
        if result == expected:
            passed += 1
            print(f"  {status}: '{input_tag}' -> '{result}'")
        else:
            failed += 1
            print(f"  {status}: '{input_tag}' -> '{result}' (expected: '{expected}')")

    print(f"\nResult: {passed}/{len(test_cases)} passed, {failed} failed")
    return failed == 0


def test_tag_library_matching():
    print("\n" + "=" * 60)
    print("Test 2: TagLibraryService - Short Tag Matching")
    print("=" * 60)

    from app.services.tag_library_service import TagLibraryService

    service = TagLibraryService()

    test_cases = [
        (["SuLi"], "SuLi"),
        (["JuRu"], "JuRu"),
        (["PinRu"], "PinRu"),
        (["BaiHe"], "BaiHe"),
        (["YanJing"], "YanJing"),
        (["XiaoFu"], "XiaoFu"),
    ]

    passed = 0
    failed = 0

    for keywords, expected_tag in test_cases:
        matches = service.match_tags_by_keywords(keywords, min_confidence=0.5)
        matched_tags = [tag for tag, _ in matches]

        if expected_tag in matched_tags:
            passed += 1
            print(f"  OK: {keywords} -> found '{expected_tag}'")
        else:
            failed += 1
            print(f"  FAIL: {keywords} -> not found '{expected_tag}'")
            print(f"    Actual matches: {matched_tags[:3]}")

    print(f"\nResult: {passed}/{len(test_cases)} passed, {failed} failed")
    return failed == 0


def test_vlm_prompt():
    print("\n" + "=" * 60)
    print("Test 3: VLM Prompt - Contains Tag Options")
    print("=" * 60)

    from app.services.lm_studio_vlm_service_v4 import LMStudioVLMService

    service = LMStudioVLMService()
    prompt = service._get_grouped_guidance_prompt()

    required_tags = ["SuLi", "MaoNiang", "JuRu", "PinRu", "XiaoFu", "BaiHe"]
    missing = []

    for tag in required_tags:
        if tag not in prompt:
            missing.append(tag)

    if missing:
        print(f"  FAIL: Prompt missing tags: {missing}")
        return False
    else:
        print(f"  OK: Prompt contains all key tags")
        print(f"  OK: Prompt length: {len(prompt)} chars")
        return True


def test_vector_store_config():
    print("\n" + "=" * 60)
    print("Test 4: TagVectorStore - Config Consistency")
    print("=" * 60)

    try:
        from tag_vector_store import TagVectorStore

        print("  OK: TagVectorStore module imported")

        import inspect

        encode_sig = inspect.signature(TagVectorStore._encode_text)
        if "is_query" in encode_sig.parameters:
            print("  OK: _encode_text supports is_query param")
        else:
            print("  FAIL: _encode_text missing is_query param")
            return False

        search_sig = inspect.signature(TagVectorStore.search)
        if "similarity_threshold" in search_sig.parameters:
            print("  OK: search supports similarity_threshold param")
        else:
            print("  FAIL: search missing similarity_threshold param")
            return False

        return True

    except Exception as e:
        print(f"  FAIL: Test error: {e}")
        return False


def run_all_tests():
    print("\n" + "=" * 60)
    print("Tag System Emergency Fix Validation")
    print("=" * 60 + "\n")

    results = {
        "TagMapper": test_tag_mapper(),
        "TagLibrary": test_tag_library_matching(),
        "VLMPrompt": test_vlm_prompt(),
        "VectorStore": test_vector_store_config(),
    }

    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")

    all_passed = all(results.values())

    if all_passed:
        print("\n[SUCCESS] All tests passed! Fixes implemented successfully.")
    else:
        print("\n[WARNING] Some tests failed. Please check the implementation.")

    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
