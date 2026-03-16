#!/usr/bin/env python3
"""
簡化測試髮色衝突檢測功能
"""

import asyncio
import sys
import os

# 添加項目根目錄到 Python 路徑
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.services.tag_validator import get_tag_validator
from app.services.tag_recommender_service import TagRecommendation


async def test_hair_conflict_simple():
    """簡化測試髮色衝突"""
    validator = get_tag_validator()

    # 測試藍髮和紅髮衝突
    tags = [
        TagRecommendation(tag="藍髮", confidence=0.8, source="vlm", reason="Blue hair"),
        TagRecommendation(tag="紅髮", confidence=0.7, source="rag", reason="Red hair"),
        TagRecommendation(
            tag="金髮", confidence=0.6, source="vlm", reason="Blonde hair"
        ),
    ]

    print("Input tags:")
    for tag in tags:
        print(f"  - {tag.tag}: {tag.confidence}")

    resolved = await validator.check_conflicts(tags)

    print("\nAfter conflict resolution:")
    for tag in resolved:
        print(f"  - {tag.tag}: {tag.confidence}")

    # 檢查結果
    hair_colors = ["藍髮", "紅髮", "金髮"]
    hair_count = len([tag for tag in resolved if tag.tag in hair_colors])

    print(f"\nResult: {hair_count} hair color(s) remaining")

    if hair_count <= 1:
        print("SUCCESS: Conflict detection working!")
        return True
    else:
        print("FAILED: Multiple hair colors still present")
        return False


if __name__ == "__main__":
    print("Testing hair color conflict detection...")
    result = asyncio.run(test_hair_conflict_simple())
    print(f"\nTest result: {'PASSED' if result else 'FAILED'}")
