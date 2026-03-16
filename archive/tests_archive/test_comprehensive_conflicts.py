#!/usr/bin/env python3
"""
測試完整的互斥規則系統
"""

import asyncio
import sys
import os

# 添加項目根目錄到 Python 路徑
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.services.tag_validator import get_tag_validator
from app.services.tag_recommender_service import TagRecommendation


async def test_comprehensive_conflicts():
    """測試全面的衝突規則"""
    validator = get_tag_validator()

    # 測試案例：故意創造衝突
    test_cases = [
        {
            "name": "年齡衝突",
            "tags": [
                TagRecommendation(
                    tag="蘿莉", confidence=0.9, source="vlm", reason="Young character"
                ),
                TagRecommendation(
                    tag="人妻", confidence=0.6, source="rag", reason="Adult woman"
                ),
                TagRecommendation(
                    tag="老太婆", confidence=0.5, source="llm", reason="Old woman"
                ),
            ],
            "expected_max": 1,  # 只保留一種年齡標籤
        },
        {
            "name": "髮色衝突",
            "tags": [
                TagRecommendation(
                    tag="藍髮", confidence=0.8, source="vlm", reason="Blue hair"
                ),
                TagRecommendation(
                    tag="紅髮", confidence=0.7, source="rag", reason="Red hair"
                ),
                TagRecommendation(
                    tag="金髮", confidence=0.6, source="llm", reason="Blonde hair"
                ),
            ],
            "expected_max": 1,  # 只保留一種髮色
        },
        {
            "name": "身體特徵衝突",
            "tags": [
                TagRecommendation(
                    tag="巨乳", confidence=0.9, source="vlm", reason="Large breasts"
                ),
                TagRecommendation(
                    tag="貧乳", confidence=0.6, source="rag", reason="Small breasts"
                ),
                TagRecommendation(
                    tag="平胸", confidence=0.5, source="llm", reason="Flat chest"
                ),
            ],
            "expected_max": 1,  # 只保留一種身材類型
        },
        {
            "name": "主題衝突",
            "tags": [
                TagRecommendation(
                    tag="純愛", confidence=0.8, source="vlm", reason="Pure love"
                ),
                TagRecommendation(
                    tag="強姦", confidence=0.6, source="rag", reason="Rape content"
                ),
                TagRecommendation(
                    tag="NTR", confidence=0.7, source="llm", reason="Netorare"
                ),
            ],
            "expected_max": 2,  # 可能保留非純愛的主題
        },
        {
            "name": "種族衝突",
            "tags": [
                TagRecommendation(
                    tag="普通女孩", confidence=0.7, source="vlm", reason="Normal girl"
                ),
                TagRecommendation(
                    tag="貓娘", confidence=0.8, source="rag", reason="Cat girl"
                ),
                TagRecommendation(
                    tag="天使", confidence=0.6, source="llm", reason="Angel"
                ),
            ],
            "expected_max": 2,  # 可能保留非普通女孩的種族
        },
        {
            "name": "混合衝突（複雜案例）",
            "tags": [
                # 年齡衝突
                TagRecommendation(
                    tag="蘿莉", confidence=0.9, source="vlm", reason="Loli"
                ),
                TagRecommendation(
                    tag="人妻", confidence=0.7, source="rag", reason="MILF"
                ),
                # 髮色衝突
                TagRecommendation(
                    tag="藍髮", confidence=0.8, source="vlm", reason="Blue hair"
                ),
                TagRecommendation(
                    tag="紅髮", confidence=0.6, source="rag", reason="Red hair"
                ),
                # 身體衝突
                TagRecommendation(
                    tag="巨乳", confidence=0.7, source="vlm", reason="Large breasts"
                ),
                TagRecommendation(
                    tag="貧乳", confidence=0.5, source="llm", reason="Small breasts"
                ),
                # 主題衝突
                TagRecommendation(
                    tag="純愛", confidence=0.8, source="rag", reason="Pure love"
                ),
                TagRecommendation(
                    tag="NTR", confidence=0.6, source="llm", reason="NTR"
                ),
                # 種族衝突
                TagRecommendation(
                    tag="普通女孩", confidence=0.7, source="vlm", reason="Normal"
                ),
                TagRecommendation(
                    tag="貓娘", confidence=0.8, source="rag", reason="Cat girl"
                ),
            ],
            "expected_max": 8,  # 蘿莉+藍髮+巨乳+純愛+貓娘 + 其他非衝突標籤
        },
    ]

    print("=== 全面衝突規則測試 ===")
    passed = 0
    total = len(test_cases)

    for i, case in enumerate(test_cases, 1):
        print(f"\n【測試 {i}: {case['name']}】")
        print("輸入標籤:")
        for tag in case["tags"]:
            print(f"  - {tag.tag} ({tag.confidence:.2f})")

        # 執行衝突檢查
        resolved = await validator.check_conflicts(case["tags"])

        print("處理後標籤:")
        for tag in resolved:
            print(f"  - {tag.tag} ({tag.confidence:.2f})")

        # 分析結果
        conflict_types = {
            "年齡": ["蘿莉", "正太", "少女", "熟女", "人妻", "老太婆"],
            "髮色": [
                "金髮",
                "黑髮",
                "銀髮",
                "藍髮",
                "紅髮",
                "綠髮",
                "紫髮",
                "粉髮",
                "白髮",
                "棕髮",
                "灰髮",
                "彩髮",
                "漸層髮",
            ],
            "身材": ["巨乳", "貧乳", "平胸", "爆乳"],
            "主題": ["純愛", "強姦", "NTR", "調教", "凌辱", "群交"],
            "種族": [
                "普通女孩",
                "普通男孩",
                "貓娘",
                "狗娘",
                "狐狸娘",
                "惡魔娘",
                "天使",
                "精靈",
                "機器人",
            ],
        }

        analysis = ""
        for cat, tags in conflict_types.items():
            count = len([tag for tag in resolved if tag.tag in tags])
            if count > 1:
                analysis += f"❌{cat}衝突未解決({count}個); "
            elif count == 1:
                analysis += f"✅{cat}衝突已解決; "

        print(f"分析: {analysis}")

        if len(resolved) <= case["expected_max"]:
            print("✅ 測試通過")
            passed += 1
        else:
            print(
                f"❌ 測試失敗 - 預期最多{case['expected_max']}個，實際{len(resolved)}個"
            )

    print(f"\n=== 測試結果總結 ===")
    print(f"通過: {passed}/{total}")
    print(f"成功率: {passed / total * 100:.1f}%")

    return passed >= total * 0.8  # 80%通過率即算成功


async def test_specific_edge_cases():
    """測試特殊邊緣案例"""
    validator = get_tag_validator()

    print("\n=== 邊緣案例測試 ===")

    # 測試空列表
    empty_result = await validator.check_conflicts([])
    print(f"空列表測試: {'✅ 通過' if len(empty_result) == 0 else '❌ 失敗'}")

    # 測試無衝突標籤
    no_conflict_tags = [
        TagRecommendation(
            tag="校服", confidence=0.8, source="vlm", reason="School uniform"
        ),
        TagRecommendation(tag="眼鏡", confidence=0.7, source="rag", reason="Glasses"),
        TagRecommendation(tag="微笑", confidence=0.9, source="llm", reason="Smiling"),
    ]
    no_conflict_result = await validator.check_conflicts(no_conflict_tags)
    print(f"無衝突測試: {'✅ 通過' if len(no_conflict_result) == 3 else '❌ 失敗'}")

    # 測試相同置信度衝突
    same_conflict_tags = [
        TagRecommendation(tag="藍髮", confidence=0.8, source="vlm", reason="Blue hair"),
        TagRecommendation(tag="紅髮", confidence=0.8, source="rag", reason="Red hair"),
    ]
    same_conflict_result = await validator.check_conflicts(same_conflict_tags)
    print(
        f"相同置信度測試: {'✅ 通過' if len(same_conflict_result) == 1 else '❌ 失敗'}"
    )


if __name__ == "__main__":
    print("開始全面衝突規則測試...")

    async def run_all_tests():
        basic_passed = await test_comprehensive_conflicts()
        await test_specific_edge_cases()
        return basic_passed

    result = asyncio.run(run_all_tests())
    print(f"\n🎯 總體評估: {'衝突系統運行良好' if result else '需要進一步調整'}")
