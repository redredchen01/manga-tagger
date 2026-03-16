"""
Test Chinese Embedding semantic search for opposite tag conflicts
"""
import sys
import asyncio
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from app.services.chinese_embedding_service import get_chinese_embedding_service
from app.services.tag_library_service import get_tag_library_service


async def test_semantic_search_conflicts():
    """Test if semantic search returns opposite meaning tags"""
    
    print("=" * 70)
    print("Test: Chinese Embedding Semantic Search Conflict")
    print("=" * 70)
    
    embedding_service = get_chinese_embedding_service()
    tag_library = get_tag_library_service()
    
    # Define opposite tag pairs
    conflicting_pairs = [
        ("巨乳", "貧乳"),  # Large breasts vs Small breasts
        ("長髮", "短髮"),  # Long hair vs Short hair
        ("蘿莉", "熟女"),  # Young vs Mature
    ]
    
    # Get all tags
    all_tags = tag_library.get_all_tags()
    print(f"\nTag library has {len(all_tags)} tags")
    
    # Check if embedding service is available
    if not embedding_service.is_available():
        print("\n[WARNING] Chinese Embedding service not available, cannot test")
        return
    
    print("\n[OK] Chinese Embedding service initialized")
    
    # Cache all tag embeddings
    await embedding_service.cache_tag_embeddings(all_tags)
    print(f"[OK] Cached embeddings for {len(all_tags)} tags")
    
    # Test each opposite pair
    print("\n" + "=" * 70)
    print("Testing opposite tag pairs semantic similarity")
    print("=" * 70)
    
    issues_found = []
    
    for tag1, tag2 in conflicting_pairs:
        print(f"\n[Test] '{tag1}' semantic search results")
        
        # Search for tags similar to tag1
        matches = await embedding_service.search_cached_tags(
            tag1, top_k=5, threshold=0.3
        )
        
        print(f"   Search '{tag1}' returned similar tags:")
        for match in matches:
            similarity = match["similarity"]
            tag = match["tag"]
            marker = "[WARNING] " if tag == tag2 else "           "
            print(f"   {marker}- {tag}: {similarity:.3f}")
        
        # Check if opposite tag appears in results
        opposite_found = any(m["tag"] == tag2 for m in matches)
        if opposite_found:
            opposite_match = next(m for m in matches if m["tag"] == tag2)
            issues_found.append(
                {
                    "query": tag1,
                    "opposite": tag2,
                    "similarity": opposite_match["similarity"],
                }
            )
            print(f"   [ERROR] Issue found: '{tag1}' search includes opposite tag '{tag2}'!")
    
    # Summary
    print("\n" + "=" * 70)
    print("Test Result Summary")
    print("=" * 70)
    
    if issues_found:
        print(f"\n[ERROR] Found {len(issues_found)} issues:")
        for issue in issues_found:
            print(
                f"   - Search '{issue['query']}' returned opposite tag '{issue['opposite']}' "
                f"(similarity: {issue['similarity']:.3f})"
            )
        print("\n[WARNING] This causes the tag system to recommend wrong opposite tags!")
        print("   Example: Large breast images may be tagged as small breasts")
        return False
    else:
        print("\n[OK] No opposite tag conflicts found")
        return True


def main():
    result = asyncio.run(test_semantic_search_conflicts())
    sys.exit(0 if result else 1)


if __name__ == "__main__":
    main()
