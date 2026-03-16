import asyncio
import sys
import os

# Add parent directory to path to import app modules
sys.path.append(os.getcwd())

from app.services.tag_conflict_resolver import get_conflict_resolver

async def test_age_conflict():
    resolver = get_conflict_resolver()
    
    # Test 1: Age Progression vs Age Regression
    # Age progression has higher score
    tags = ["紫髮", "年齡增長", "年齡回溯", "返嬰癖"]
    scores = {
        "紫髮": 0.95,
        "年齡增長": 0.501,
        "年齡回溯": 0.500,
        "返嬰癖": 0.499
    }
    
    print(f"\nTesting Age Conflict resolution...")
    print(f"Input tags: {tags}")
    print(f"Scores: {scores}")
    
    resolved_tags, resolved_scores = resolver.resolve(tags, scores)
    
    print(f"Resolved tags: {resolved_tags}")
    print(f"Resolved scores: {resolved_scores}")
    
    assert "年齡增長" in resolved_tags
    assert "年齡回溯" not in resolved_tags
    assert "返嬰癖" not in resolved_tags
    print("SUCCESS: Age progression correctly won over regression/infantilization.")

async def test_breast_size_conflict():
    resolver = get_conflict_resolver()
    
    # Test 2: Multiple breast sizes
    tags = ["巨乳", "貧乳", "平胸"]
    scores = {
        "巨乳": 0.6,
        "貧乳": 0.8,
        "平胸": 0.4
    }
    
    print(f"\nTesting Breast Size Conflict resolution...")
    print(f"Input tags: {tags}")
    
    resolved_tags, _ = resolver.resolve(tags, scores)
    
    print(f"Resolved tags: {resolved_tags}")
    
    assert "貧乳" in resolved_tags
    assert "巨乳" not in resolved_tags
    assert "平胸" not in resolved_tags
    print("SUCCESS: Highest score breast size kept.")

async def test_complementary_conflict():
    resolver = get_conflict_resolver()
    
    # Test 3: Complementary exclusion (Pure vs Rape)
    tags = ["純愛", "強姦", "校服"]
    scores = {
        "純愛": 0.3,
        "強姦": 0.7,
        "校服": 0.9
    }
    
    print(f"\nTesting Complementary Conflict (Pure vs Rape)...")
    
    resolved_tags, _ = resolver.resolve(tags, scores)
    
    print(f"Resolved tags: {resolved_tags}")
    
    assert "強姦" in resolved_tags
    assert "純愛" not in resolved_tags
    print("SUCCESS: Rape won over Pure Love due to higher score.")

if __name__ == "__main__":
    asyncio.run(test_age_conflict())
    asyncio.run(test_breast_size_conflict())
    asyncio.run(test_complementary_conflict())
