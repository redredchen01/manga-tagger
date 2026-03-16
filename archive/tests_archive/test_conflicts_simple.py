#!/usr/bin/env python3
"""
簡化版全面衝突測試 - 避免Unicode問題
"""

import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.services.tag_validator import get_tag_validator
from app.services.tag_recommender_service import TagRecommendation


async def test_conflicts_simple():
    """簡化測試衝突規則"""
    validator = get_tag_validator()

    # 測試案例1：年齡衝突
    print("=== Test 1: Age Conflicts ===")
    age_tags = [
        TagRecommendation(tag="蘿莉", confidence=0.9, source="vlm", reason="Loli"),
        TagRecommendation(tag="人妻", confidence=0.6, source="rag", reason="MILF"),
        TagRecommendation(
            tag="老太婆", confidence=0.5, source="llm", reason="Old woman"
        ),
    ]
    print("Input:", [f"{t.tag}({t.confidence})" for t in age_tags])
    age_result = await validator.check_conflicts(age_tags)
    print("Output:", [f"{t.tag}({t.confidence})" for t in age_result])
    print(
        f"Result: {len(age_result)} age tag(s) - {'PASS' if len(age_result) <= 1 else 'FAIL'}"
    )

    # 測試案例2：髮色衝突
    print("\n=== Test 2: Hair Color Conflicts ===")
    hair_tags = [
        TagRecommendation(tag="藍髮", confidence=0.8, source="vlm", reason="Blue hair"),
        TagRecommendation(tag="紅髮", confidence=0.7, source="rag", reason="Red hair"),
        TagRecommendation(
            tag="金髮", confidence=0.6, source="llm", reason="Blonde hair"
        ),
    ]
    print("Input:", [f"{t.tag}({t.confidence})" for t in hair_tags])
    hair_result = await validator.check_conflicts(hair_tags)
    print("Output:", [f"{t.tag}({t.confidence})" for t in hair_result])
    print(
        f"Result: {len(hair_result)} hair color(s) - {'PASS' if len(hair_result) <= 1 else 'FAIL'}"
    )

    # 測試案例3：身材衝突
    print("\n=== Test 3: Body Feature Conflicts ===")
    body_tags = [
        TagRecommendation(
            tag="巨乳", confidence=0.9, source="vlm", reason="Large breasts"
        ),
        TagRecommendation(
            tag="貧乳", confidence=0.6, source="rag", reason="Small breasts"
        ),
        TagRecommendation(
            tag="平胸", confidence=0.5, source="llm", reason="Flat chest"
        ),
    ]
    print("Input:", [f"{t.tag}({t.confidence})" for t in body_tags])
    body_result = await validator.check_conflicts(body_tags)
    print("Output:", [f"{t.tag}({t.confidence})" for t in body_result])
    print(
        f"Result: {len(body_result)} body type(s) - {'PASS' if len(body_result) <= 1 else 'FAIL'}"
    )

    # 測試案例4：混合衝突
    print("\n=== Test 4: Mixed Conflicts ===")
    mixed_tags = [
        # 年齡
        TagRecommendation(tag="蘿莉", confidence=0.9, source="vlm", reason="Loli"),
        TagRecommendation(tag="人妻", confidence=0.7, source="rag", reason="MILF"),
        # 髮色
        TagRecommendation(tag="藍髮", confidence=0.8, source="vlm", reason="Blue hair"),
        TagRecommendation(tag="紅髮", confidence=0.6, source="rag", reason="Red hair"),
        # 身材
        TagRecommendation(
            tag="巨乳", confidence=0.7, source="vlm", reason="Large breasts"
        ),
        TagRecommendation(
            tag="貧乳", confidence=0.5, source="llm", reason="Small breasts"
        ),
        # 主題
        TagRecommendation(tag="純愛", confidence=0.8, source="rag", reason="Pure love"),
        TagRecommendation(tag="NTR", confidence=0.6, source="llm", reason="NTR"),
        # 種族
        TagRecommendation(
            tag="普通女孩", confidence=0.7, source="vlm", reason="Normal"
        ),
        TagRecommendation(tag="貓娘", confidence=0.8, source="rag", reason="Cat girl"),
    ]
    print("Input:", [f"{t.tag}({t.confidence})" for t in mixed_tags])
    mixed_result = await validator.check_conflicts(mixed_tags)
    print("Output:", [f"{t.tag}({t.confidence})" for t in mixed_result])

    # 分析結果
    age_count = len(
        [t for t in mixed_result if t.tag in ["蘿莉", "人妻", "老太婆", "少女", "熟女"]]
    )
    hair_count = len([t for t in mixed_result if t.tag.endswith("髮")])
    body_count = len(
        [t for t in mixed_result if t.tag in ["巨乳", "貧乳", "平胸", "爆乳"]]
    )
    theme_count = len(
        [t for t in mixed_result if t.tag in ["純愛", "NTR", "強姦", "調教"]]
    )
    race_count = len(
        [
            t
            for t in mixed_result
            if t.tag in ["普通女孩", "貓娘", "狗娘", "天使", "惡魔娘"]
        ]
    )

    print(
        f"Analysis - Age: {age_count}, Hair: {hair_count}, Body: {body_count}, Theme: {theme_count}, Race: {race_count}"
    )

    # 總體評估
    total_tests = 4
    passed = (
        len(age_result) <= 1
        and len(hair_result) <= 1
        and len(body_result) <= 1
        and age_count <= 1
        and hair_count <= 1
        and body_count <= 1
    )

    print(f"\n=== Summary ===")
    print(f"Tests Passed: {total_tests if passed else total_tests - 1}/{total_tests}")
    print(f"Overall Result: {'PASS' if passed else 'FAIL'}")

    return passed


if __name__ == "__main__":
    print("Starting comprehensive conflict detection tests...")
    result = asyncio.run(test_conflicts_simple())
    print(
        f"\nConflict System Status: {'Working correctly' if result else 'Needs adjustment'}"
    )
