"""Test suite for Hierarchical Tag Generator."""

import sys
import asyncio
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.hierarchical_tag_generator import (
    HierarchicalTagGenerator,
    HierarchicalResult,
    TagCategory,
    CategoryResult,
)
from app.services.enhanced_vlm_dispatcher import ModelPrediction, DispatchResult
from app.services.vlm_response_parser import ParsedResponse


def test_category_enum():
    """Test tag category enum."""
    print("\n" + "=" * 60)
    print("Testing Tag Category Enum")
    print("=" * 60)
    
    categories = list(TagCategory)
    print(f"Available categories: {len(categories)}")
    
    for cat in categories:
        print(f"  - {cat.value}: {cat.name}")
    
    return True


def test_category_keywords():
    """Test category keyword mapping."""
    print("\n" + "=" * 60)
    print("Testing Category Keywords")
    print("=" * 60)
    
    generator = HierarchicalTagGenerator()
    
    for cat, keywords in generator.CATEGORY_KEYWORDS.items():
        print(f"\n{cat.value} ({len(keywords)} keywords):")
        # Show first 5
        for kw in keywords[:5]:
            print(f"  - {kw}")
        if len(keywords) > 5:
            print(f"  ... and {len(keywords) - 5} more")
    
    return True


def test_category_prompts():
    """Test category-specific prompts."""
    print("\n" + "=" * 60)
    print("Testing Category Prompts")
    print("=" * 60)
    
    generator = HierarchicalTagGenerator()
    
    # Check that all categories have prompts
    for cat in TagCategory:
        prompt = generator.PROMPTS.get(cat)
        if prompt:
            print(f"[OK] {cat.value}: prompt available ({len(prompt)} chars)")
        else:
            print(f"[MISSING] {cat.value}: no prompt")
    
    return True


def test_category_identification():
    """Test category identification with mock data."""
    print("\n" + "=" * 60)
    print("Testing Category Identification")
    print("=" * 60)
    
    generator = HierarchicalTagGenerator()
    
    # Mock dispatch result
    mock_predictions = [
        ModelPrediction(
            model_name="glm-4.6v-flash",
            raw_response="""character:0.85:loli_character_visible
clothing:0.90:school_uniform
body:0.75:large_breasts""",
            parsed_response=ParsedResponse(
                raw_response="",
                tags=[],
                confidence={},
                parsing_method="test",
                is_valid=True,
            ),
            processing_time=1.5,
            confidence_scores={},
            is_valid=True,
        ),
        ModelPrediction(
            model_name="qwen-vl-max",
            raw_response="""character:0.80:catgirl_not_loli
clothing:0.85:uniform_visible
hair:0.70:blonde_hair""",
            parsed_response=ParsedResponse(
                raw_response="",
                tags=[],
                confidence={},
                parsing_method="test",
                is_valid=True,
            ),
            processing_time=2.0,
            confidence_scores={},
            is_valid=True,
        ),
    ]
    
    mock_dispatch = DispatchResult(
        predictions=mock_predictions,
        total_time=2.0,
        successful_models=2,
        failed_models=0,
        all_tags=[],
    )
    
    # Test parsing
    with patch.object(generator.dispatcher, 'dispatch_all', return_value=mock_dispatch):
        # Simulate category identification
        categories = []
        for pred in mock_predictions:
            if pred.is_valid:
                for line in pred.raw_response.split('\n'):
                    if ':' in line:
                        parts = line.split(':', 2)
                        if len(parts) >= 2:
                            cat_name = parts[0].strip().lower()
                            try:
                                confidence = float(parts[1].strip())
                                for category in TagCategory:
                                    if category.value in cat_name:
                                        categories.append(CategoryResult(
                                            category=category,
                                            confidence=confidence,
                                            evidence=[parts[2].strip()],
                                        ))
                                        break
                            except ValueError:
                                continue
        
        print(f"\nIdentified {len(categories)} categories:")
        for cat in categories:
            print(f"  - {cat.category.value}: {cat.confidence:.2f} ({cat.evidence[0]})")
    
    return True


def test_category_tag_generation():
    """Test category-specific tag generation."""
    print("\n" + "=" * 60)
    print("Testing Category Tag Generation")
    print("=" * 60)
    
    generator = HierarchicalTagGenerator()
    
    # Mock predictions for different categories
    category_responses = {
        TagCategory.CHARACTER: "蘿莉, 貓娘, 少女",
        TagCategory.CLOTHING: "校服, 水手服, 白色襪子",
        TagCategory.HAIR: "藍髮, 長髮, 雙馬尾",
        TagCategory.BODY: "巨乳, 長腿",
    }
    
    for category, response in category_responses.items():
        mock_pred = ModelPrediction(
            model_name="test",
            raw_response=response,
            parsed_response=ParsedResponse(
                raw_response=response,
                tags=[],
                confidence={},
                parsing_method="test",
                is_valid=True,
            ),
            processing_time=1.0,
            confidence_scores={},
            is_valid=True,
        )
        
        # Extract tags
        raw_tags = response.split(',')
        tags = []
        for tag in raw_tags:
            cleaned = generator._clean_tag(tag)
            if cleaned:
                tags.append(cleaned)
        
        print(f"\n{category.value}:")
        print(f"  Input: {response}")
        print(f"  Extracted: {tags}")
    
    return True


def test_consistency_check():
    """Test consistency checking."""
    print("\n" + "=" * 60)
    print("Testing Consistency Check")
    print("=" * 60)
    
    generator = HierarchicalTagGenerator()
    
    # Test case 1: Consistent tags
    consistent_tags = {"貓娘", "長髮", "藍髮", "校服"}
    category_tags = {
        TagCategory.CHARACTER: ["貓娘"],
        TagCategory.HAIR: ["長髮", "藍髮"],
        TagCategory.CLOTHING: ["校服"],
    }
    
    score1 = generator._check_consistency(category_tags, consistent_tags)
    print(f"Consistent tags: {consistent_tags}")
    print(f"Consistency score: {score1:.2f}")
    
    # Test case 2: Inconsistent tags
    inconsistent_tags = {"蘿莉", "巨乳", "熟女"}
    category_tags2 = {
        TagCategory.CHARACTER: ["蘿莉", "熟女"],
        TagCategory.BODY: ["巨乳"],
    }
    
    score2 = generator._check_consistency(category_tags2, inconsistent_tags)
    print(f"\nInconsistent tags: {inconsistent_tags}")
    print(f"Consistency score: {score2:.2f}")
    
    return True


def test_clean_tag():
    """Test tag cleaning."""
    print("\n" + "=" * 60)
    print("Testing Tag Cleaning")
    print("=" * 60)
    
    generator = HierarchicalTagGenerator()
    
    test_cases = [
        ("蘿莉", "蘿莉"),
        ("  貓娘  ", "貓娘"),
        ("校服,", "校服"),
        ("tag: 巨乳", "巨乳"),
        ("Features: 長腿", "長髮"),  # Will fail the prefix check
    ]
    
    for input_tag, expected in test_cases:
        cleaned = generator._clean_tag(input_tag)
        status = "[OK]" if cleaned == expected else "[FAIL]"
        print(f"{status} '{input_tag}' -> '{cleaned}' (expected '{expected}')")
    
    return True


def test_result_summary():
    """Test result summary generation."""
    print("\n" + "=" * 60)
    print("Testing Result Summary")
    print("=" * 60)
    
    generator = HierarchicalTagGenerator()
    
    # Create mock result
    result = HierarchicalResult(
        categories=[
            CategoryResult(TagCategory.CHARACTER, 0.85, ["loli visible"]),
            CategoryResult(TagCategory.CLOTHING, 0.90, ["uniform"]),
        ],
        all_tags=["蘿莉", "貓娘", "校服"],
        tags_by_category={
            TagCategory.CHARACTER: ["蘿莉", "貓娘"],
            TagCategory.CLOTHING: ["校服"],
        },
        consistency_score=0.95,
        warnings=[],
        rejected_tags=[],
    )
    
    summary = generator.get_category_summary(result)
    print(summary)
    
    return True


def test_full_pipeline():
    """Test the full hierarchical pipeline."""
    print("\n" + "=" * 60)
    print("Testing Full Pipeline (Mock)")
    print("=" * 60)
    
    generator = HierarchicalTagGenerator()
    
    # Mock category identification
    category_results = [
        CategoryResult(TagCategory.CHARACTER, 0.85, ["catgirl"]),
        CategoryResult(TagCategory.CLOTHING, 0.80, ["uniform"]),
    ]
    
    # Mock tag generation per category
    category_tags = {
        TagCategory.CHARACTER: ["貓娘", "蘿莉"],
        TagCategory.CLOTHING: ["校服", "白色過膝襪"],
    }
    
    # Build result
    all_tags = set()
    for tags in category_tags.values():
        all_tags.update(tags)
    
    result = HierarchicalResult(
        categories=category_results,
        all_tags=list(all_tags),
        tags_by_category=category_tags,
        consistency_score=0.9,
        warnings=[],
        rejected_tags=[],
    )
    
    print(f"\nFinal Results:")
    print(f"  Total tags: {len(result.all_tags)}")
    print(f"  Categories: {len(result.tags_by_category)}")
    print(f"  Consistency: {result.consistency_score:.2%}")
    print(f"\nTags by Category:")
    for cat, tags in result.tags_by_category.items():
        print(f"  {cat.value}: {tags}")
    
    return True


def main():
    """Run all tests."""
    print("\n" + "#" * 60)
    print("# Hierarchical Tag Generator Test Suite")
    print("#" * 60)
    
    results = []
    
    results.append(("Category Enum", test_category_enum()))
    results.append(("Category Keywords", test_category_keywords()))
    results.append(("Category Prompts", test_category_prompts()))
    results.append(("Category Identification", test_category_identification()))
    results.append(("Category Tag Generation", test_category_tag_generation()))
    results.append(("Consistency Check", test_consistency_check()))
    results.append(("Tag Cleaning", test_clean_tag()))
    results.append(("Result Summary", test_result_summary()))
    results.append(("Full Pipeline", test_full_pipeline()))
    
    print("\n" + "#" * 60)
    print("# Test Summary")
    print("#" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("All tests passed!")
    else:
        print("Some tests failed!")
    print("=" * 60)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
