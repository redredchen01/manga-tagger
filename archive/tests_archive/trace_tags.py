# Debug script to trace tag generation flow and identify where tags are lost.
import sys
import json
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from app.services.tag_library_service import get_tag_library_service
from app.services.tag_mapper import get_tag_mapper
from app.services.tag_recommender_service import get_tag_recommender_service


def debug_tag_library():
    """Debug tag library loading."""
    print("\n=== DEBUG: Tag Library ===")
    lib = get_tag_library_service()
    print(f"[OK] Loaded {len(lib.tag_names)} tags")
    print(f"[OK] Categories: {list(lib.tag_categories.keys())}")
    print(f"[OK] Character tags: {len(lib.tag_categories.get('character', []))}")
    print(f"[OK] Body tags: {len(lib.tag_categories.get('body', []))}")

    # Check specific tags
    test_tags = ["蘿莉", "巨乳", "貓娘", "校服", "口交"]
    print("\n[OK] Checking specific tags:")
    for tag in test_tags:
        exists = tag in lib.tag_names
        status = "EXISTS" if exists else "NOT FOUND"
        print(f"  - '{tag}': {status}")

    return lib


def debug_tag_mapper():
    """Debug tag mapping."""
    print("\n=== DEBUG: Tag Mapper ===")
    mapper = get_tag_mapper()
    print(f"[OK] Built {len(mapper.en_to_cn)} mappings")

    # Test mappings
    test_keywords = [
        "loli",
        "catgirl",
        "large_breasts",
        "huge breasts",
        "school_uniform",
        "mature",
        "teen",
    ]

    print("\n[OK] Testing keyword mappings:")
    for kw in test_keywords:
        cn = mapper.to_chinese(kw)
        print(f"  - '{kw}' -> '{cn}'")

    return mapper


def debug_tag_matching(lib, mapper):
    """Debug tag matching logic."""
    print("\n=== DEBUG: Tag Matching ===")

    # Simulate VLM output keywords
    vlm_keywords = ["loli", "catgirl", "large_breasts", "school_uniform"]
    print(f"\n[OK] VLM Keywords: {vlm_keywords}")

    # Map to Chinese
    mapped_keywords = []
    for kw in vlm_keywords:
        cn_tag = mapper.to_chinese(kw)
        if cn_tag:
            mapped_keywords.append(cn_tag)
            print(f"  [MAP] '{kw}' -> '{cn_tag}'")
        else:
            mapped_keywords.append(kw)
            print(f"  [NO MAP] '{kw}' -> using as-is")

    print(f"\n[OK] Mapped Keywords: {mapped_keywords}")

    # Match in library
    matches = lib.match_tags_by_keywords(mapped_keywords, min_confidence=0.5)
    print(f"\n[OK] Library Matches ({len(matches)}):")
    for tag, conf in matches[:10]:
        print(f"  - {tag}: {conf:.3f}")

    return matches


def debug_recommender():
    """Debug tag recommender service."""
    print("\n=== DEBUG: Tag Recommender ===")

    recommender = get_tag_recommender_service()

    # Simulate VLM analysis output
    vlm_analysis = {
        "description": "A loli catgirl with large breasts wearing school uniform",
        "character_types": ["loli", "catgirl"],
        "clothing": ["school_uniform"],
        "body_features": ["large_breasts"],
        "actions": [],
        "themes": [],
        "settings": [],
        "raw_keywords": ["loli", "catgirl", "large_breasts", "school_uniform"],
    }

    print(f"\n[OK] VLM Analysis:")
    print(f"  - Character types: {vlm_analysis['character_types']}")
    print(f"  - Clothing: {vlm_analysis['clothing']}")
    print(f"  - Body features: {vlm_analysis['body_features']}")

    # Get recommendations
    recommendations = recommender.recommend_tags(
        vlm_analysis=vlm_analysis, rag_matches=[], top_k=5, confidence_threshold=0.5
    )

    print(f"\n[OK] Recommendations ({len(recommendations)}):")
    for rec in recommendations:
        print(f"  - {rec.tag}: {rec.confidence:.3f} (source: {rec.source})")

    if not recommendations:
        print("  [ERROR] NO RECOMMENDATIONS - This is the problem!")

    return recommendations


def main():
    """Run all debug tests."""
    print("=" * 60)
    print("TAG SYSTEM DEBUG")
    print("=" * 60)

    # Step 1: Check tag library
    lib = debug_tag_library()

    # Step 2: Check tag mapper
    mapper = debug_tag_mapper()

    # Step 3: Check tag matching
    matches = debug_tag_matching(lib, mapper)

    # Step 4: Check recommender
    recommendations = debug_recommender()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"[OK] Tag library: {len(lib.tag_names)} tags loaded")
    print(f"[OK] Tag mapper: {len(mapper.en_to_cn)} mappings")
    print(f"[OK] Library matches: {len(matches)} matches")
    status = "OK" if recommendations else "ERROR"
    print(f"[{status}] Recommendations: {len(recommendations)} tags")

    if not recommendations:
        print("\n[WARNING] ISSUE IDENTIFIED: No tag recommendations generated!")
        print("\nPossible causes:")
        print("  1. VLM keywords not mapping to Chinese tags")
        print("  2. Library matching returning no results")
        print("  3. Confidence threshold too high")
        print("  4. Tag library path incorrect")

    return recommendations


if __name__ == "__main__":
    main()
