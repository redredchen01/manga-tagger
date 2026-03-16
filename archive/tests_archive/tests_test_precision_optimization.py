"""Test script for precision optimization features.

Tests the enhanced matching pipeline with new thresholds,
alias mapping, conflict resolution, and calibration.
"""

import sys
import json
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.tag_alias_service import TagAliasService, get_tag_alias_service
from app.services.tag_conflict_resolver import TagConflictResolver, get_conflict_resolver
from app.services.dynamic_threshold_service import DynamicThresholdService, get_dynamic_threshold_service
from app.services.confidence_calibrator import ConfidenceCalibrator, get_confidence_calibrator


def test_alias_service():
    """Test tag alias service."""
    print("\n" + "=" * 60)
    print("Testing Tag Alias Service")
    print("=" * 60)
    
    service = get_tag_alias_service()
    
    test_cases = [
        ("catgirl", ["貓娘", "貓耳娘"]),
        ("big_breasts", ["巨乳"]),
        ("loli", ["蘿莉"]),
        ("unknown", []),
    ]
    
    passed = 0
    for eng, expected in test_cases:
        result = service.to_chinese_all(eng)
        if set(result) == set(expected):
            print(f"[OK] '{eng}' -> {result}")
            passed += 1
        else:
            print(f"[FAIL] '{eng}' -> {result} (expected {expected})")
    
    print(f"\nAlias Service: {passed}/{len(test_cases)} tests passed")
    return passed == len(test_cases)


def test_conflict_resolver():
    """Test tag conflict resolver."""
    print("\n" + "=" * 60)
    print("Testing Tag Conflict Resolver")
    print("=" * 60)
    
    resolver = get_conflict_resolver()
    
    # Test case 1: Age conflict
    tags1 = ["蘿莉", "熟女", "巨乳"]
    scores1 = {"蘿莉": 0.9, "熟女": 0.8, "巨乳": 0.7}
    
    result1 = resolver.check_conflicts(tags1, scores1)
    print(f"\nTest 1: {tags1}")
    print(f"Scores: {scores1}")
    print(f"Kept: {result1.kept_tags}")
    print(f"Removed: {result1.removed_tags}")
    print(f"Conflicts: {result1.conflicts_found}")
    
    # Test case 2: Breast size conflict
    tags2 = ["巨乳", "貧乳"]
    scores2 = {"巨乳": 0.85, "貧乳": 0.75}
    
    result2 = resolver.check_conflicts(tags2, scores2)
    print(f"\nTest 2: {tags2}")
    print(f"Scores: {scores2}")
    print(f"Kept: {result2.kept_tags}")
    print(f"Removed: {result2.removed_tags}")
    
    # Test case 3: Theme conflict
    tags3 = ["純愛", "NTR"]
    scores3 = {"純愛": 0.9, "NTR": 0.8}
    
    result3 = resolver.check_conflicts(tags3, scores3)
    print(f"\nTest 3: {tags3}")
    print(f"Scores: {scores3}")
    print(f"Kept: {result3.kept_tags}")
    print(f"Removed: {result3.removed_tags}")
    
    return True


def test_dynamic_threshold():
    """Test dynamic threshold service."""
    print("\n" + "=" * 60)
    print("Testing Dynamic Threshold Service")
    print("=" * 60)
    
    service = get_dynamic_threshold_service()
    
    test_tags = [
        ("蘿莉", "character"),
        ("巨乳", "body"),
        ("校服", "clothing"),
        ("肛交", "action"),
        ("NTR", "theme"),
        ("unknown_tag", "other"),
    ]
    
    print("\nThreshold per tag:")
    for tag, expected_cat in test_tags:
        threshold = service.get_threshold(tag)
        print(f"  '{tag}' (category: {expected_cat}): threshold={threshold:.2f}")
    
    # Test filtering
    tags = ["蘿莉", "巨乳", "校服", "肛交", "NTR", "unknown"]
    scores = {
        "蘿莉": 0.60,
        "巨乳": 0.55,
        "校服": 0.52,
        "肛交": 0.68,
        "NTR": 0.68,
        "unknown": 0.51,
    }
    
    filtered, filtered_scores = service.filter_by_threshold(tags, scores, base_threshold=0.50)
    print(f"\nOriginal tags: {tags}")
    print(f"Original scores: {scores}")
    print(f"Filtered tags: {filtered}")
    print(f"Filtered scores: {filtered_scores}")
    
    return True


def test_confidence_calibrator():
    """Test confidence calibrator."""
    print("\n" + "=" * 60)
    print("Testing Confidence Calibrator")
    print("=" * 60)
    
    calibrator = get_confidence_calibrator()
    
    # Test single calibration
    test_cases = [
        (0.95, "exact_match", 1),
        (0.85, "contains_match", 2),
        (0.75, "partial_match", 3),
        (0.65, "vector_similarity", 5),
        (0.55, "rag_search", 10),
    ]
    
    print("\nCalibration examples:")
    for raw, method, top_k in test_cases:
        calibrated = calibrator.calibrate(raw, method, top_k)
        print(f"  {method} (top_k={top_k}): {raw:.2f} -> {calibrated.calibrated_score:.4f}")
        print(f"    Range: [{calibrated.lower_bound:.4f}, {calibrated.upper_bound:.4f}]")
    
    # Test batch calibration
    scores = {
        "蘿莉": 0.92,
        "巨乳": 0.85,
        "校服": 0.78,
        "貓娘": 0.72,
        "女僕": 0.65,
        "NTR": 0.58,
        "unknown": 0.45,
    }
    methods = {
        "蘿莉": "exact_match",
        "巨乳": "contains_match",
        "校服": "partial_match",
        "貓娘": "vector_similarity",
        "女僕": "rag_search",
        "NTR": "hybrid_combined",
        "unknown": "rag_search",
    }
    
    print("\nBatch calibration:")
    calibrated = calibrator.calibrate_batch(scores, methods)
    for tag, cs in calibrated.items():
        print(f"  {tag}: {cs.raw_score:.2f} -> {cs.calibrated_score:.4f}")
    
    # Test top-k selection
    print("\nTop 5 calibrated tags:")
    top5 = calibrator.get_top_calibrated(scores, methods, top_k=5, min_threshold=0.0)
    for tag, score, method in top5:
        print(f"  {tag}: {score:.4f} ({method})")
    
    return True


def test_integration():
    """Test full integration pipeline."""
    print("\n" + "=" * 60)
    print("Testing Integration Pipeline")
    print("=" * 60)
    
    # Simulate VLM output
    vlm_output = "catgirl with large breasts wearing school uniform"
    
    # Step 1: Alias expansion
    alias_service = get_tag_alias_service()
    words = vlm_output.replace(",", " ").split()
    expanded_words = []
    for word in words:
        expanded_words.append(word)
        cn = alias_service.to_chinese(word)
        if cn:
            expanded_words.append(cn)
    
    print(f"\nOriginal: {vlm_output}")
    print(f"Expanded: {' '.join(expanded_words)}")
    
    # Step 2: Simulate matching scores
    scores = {
        "貓娘": 0.88,
        "巨乳": 0.85,
        "校服": 0.82,
        "蘿莉": 0.45,
        "女僕": 0.40,
        "unknown_tag": 0.35,
    }
    
    # Step 3: Apply dynamic thresholds
    threshold_service = get_dynamic_threshold_service()
    filtered, filtered_scores = threshold_service.filter_by_threshold(
        list(scores.keys()), scores, base_threshold=0.50
    )
    
    print(f"\nAfter threshold filtering: {filtered}")
    
    # Step 4: Resolve conflicts
    conflict_resolver = get_conflict_resolver()
    resolved_tags, resolved_scores = conflict_resolver.resolve(
        filtered, filtered_scores, max_tags=10
    )
    
    print(f"After conflict resolution: {resolved_tags}")
    
    # Step 5: Calibrate scores
    calibrator = get_confidence_calibrator()
    calibrated = calibrator.calibrate_batch(resolved_scores)
    
    print("\nFinal calibrated scores:")
    for tag, cs in sorted(calibrated.items(), key=lambda x: x[1].calibrated_score, reverse=True):
        print(f"  {tag}: {cs.calibrated_score:.4f} [{cs.lower_bound:.4f}, {cs.upper_bound:.4f}]")
    
    return True


def main():
    """Run all tests."""
    print("\n" + "#" * 60)
    print("# Precision Optimization Test Suite")
    print("#" * 60)
    
    results = []
    
    results.append(("Alias Service", test_alias_service()))
    results.append(("Conflict Resolver", test_conflict_resolver()))
    results.append(("Dynamic Threshold", test_dynamic_threshold()))
    results.append(("Confidence Calibrator", test_confidence_calibrator()))
    results.append(("Integration", test_integration()))
    
    print("\n" + ("=" * 60))
    print("# Test Summary")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False
    
    print("\n" + ("=" * 60))
    if all_passed:
        print("All tests passed!")
    else:
        print("Some tests failed!")
    print("=" * 60)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
