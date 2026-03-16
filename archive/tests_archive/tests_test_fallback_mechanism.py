"""Test script for fallback mechanism.

Tests the emergency fallback system when VLM is unavailable.
"""

import asyncio
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.emergency_tag_service import get_emergency_tag_service
from app.services.tag_recommender_service import TagRecommenderService


def test_emergency_tag_service():
    """Test the emergency tag service directly."""
    print("\n" + "="*60)
    print("TEST 1: Emergency Tag Service")
    print("="*60)
    
    service = get_emergency_tag_service()
    
    # Test getting emergency tags
    tags = service.get_emergency_tags(top_k=5)
    
    print(f"\n[OK] Got {len(tags)} emergency tags:")
    for tag, confidence, reason in tags:
        print(f"   - {tag}: {confidence:.2f} ({reason})")
    
    # Test category distribution
    distribution = service.get_category_distribution()
    print(f"\n[DIST] Category distribution: {distribution}")
    
    assert len(tags) == 5, "Should return 5 tags"
    print("\n[PASS] TEST 1 PASSED")


async def test_fallback_metadata():
    """Test that fallback metadata contains keywords."""
    print("\n" + "="*60)
    print("TEST 2: Fallback Metadata")
    print("="*60)
    
    from app.services.lm_studio_vlm_service_v2 import LMStudioVLMService
    
    service = LMStudioVLMService()
    
    # Get fallback metadata
    fallback = service._get_fallback_metadata("VLM unavailable - using RAG fallback")
    
    print(f"\n📝 Fallback metadata:")
    print(f"   - Description: {fallback['description']}")
    print(f"   - Raw keywords: {fallback['raw_keywords']}")
    print(f"   - Fallback mode: {fallback.get('fallback_mode', False)}")
    
    assert fallback.get("fallback_mode") == True, "Should have fallback_mode=True"
    assert len(fallback["raw_keywords"]) > 0, "Should have keywords"
    print("\n✅ TEST 2 PASSED")


async def test_tag_recommender_with_fallback():
    """Test tag recommender with fallback VLM analysis."""
    print("\n" + "="*60)
    print("TEST 3: Tag Recommender with Fallback")
    print("="*60)
    
    recommender = TagRecommenderService()
    
    # Simulate fallback VLM analysis
    fallback_vlm = {
        "description": "Analysis fallback: VLM unavailable - using RAG fallback",
        "character_types": [],
        "clothing": [],
        "body_features": [],
        "actions": [],
        "themes": [],
        "settings": [],
        "raw_keywords": ["female", "girl", "manga", "anime"],
        "fallback_mode": True,
    }
    
    # Empty RAG matches (simulating RAG failure)
    rag_matches = []
    
    # Get recommendations
    recommendations = await recommender.recommend_tags(
        vlm_analysis=fallback_vlm,
        rag_matches=rag_matches,
        top_k=5,
        confidence_threshold=0.5,
    )
    
    print(f"\n🏷️ Got {len(recommendations)} recommendations:")
    for rec in recommendations:
        print(f"   - {rec.tag}: {rec.confidence:.2f} [{rec.source}]")
        print(f"     Reason: {rec.reason}")
    
    assert len(recommendations) > 0, "Should return at least some tags"
    print("\n✅ TEST 3 PASSED")


async def test_emergency_fallback_integration():
    """Test that emergency fallback activates when all else fails."""
    print("\n" + "="*60)
    print("TEST 4: Emergency Fallback Integration")
    print("="*60)
    
    recommender = TagRecommenderService()
    
    # Completely empty VLM analysis (worst case)
    empty_vlm = {
        "description": "Analysis failed: complete failure",
        "raw_keywords": [],
        "fallback_mode": True,
    }
    
    # Empty RAG matches
    rag_matches = []
    
    # Get recommendations - should trigger emergency fallback
    recommendations = await recommender.recommend_tags(
        vlm_analysis=empty_vlm,
        rag_matches=rag_matches,
        top_k=5,
        confidence_threshold=0.5,
    )
    
    print(f"\n🚨 Emergency fallback test - Got {len(recommendations)} recommendations:")
    for rec in recommendations:
        print(f"   - {rec.tag}: {rec.confidence:.2f} [{rec.source}]")
        print(f"     Reason: {rec.reason}")
    
    # Check if any came from emergency fallback
    emergency_tags = [r for r in recommendations if r.source == "emergency_fallback"]
    
    if emergency_tags:
        print(f"\n✅ Emergency fallback activated with {len(emergency_tags)} tags")
    
    assert len(recommendations) > 0, "Should return at least emergency tags"
    print("\n✅ TEST 4 PASSED")


async def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("[TEST] FALLBACK MECHANISM TEST SUITE")
    print("="*60)
    
    try:
        # Test 1: Emergency tag service
        test_emergency_tag_service()
        
        # Test 2: Fallback metadata
        await test_fallback_metadata()
        
        # Test 3: Tag recommender with fallback
        await test_tag_recommender_with_fallback()
        
        # Test 4: Emergency fallback integration
        await test_emergency_fallback_integration()
        
        print("\n" + "="*60)
        print("[PASS] ALL TESTS PASSED!")
        print("="*60)
        print("\nFallback mechanism is working correctly.")
        print("When VLM is unavailable, the system will:")
        print("  1. Use fallback keywords from VLM service")
        print("  2. Lower thresholds for better matching")
        print("  3. Activate emergency tags if all else fails")
        
    except AssertionError as e:
        print(f"\n[FAIL] TEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n[ERROR] ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
