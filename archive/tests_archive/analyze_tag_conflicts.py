#!/usr/bin/env python3
"""
分析完整標籤庫，識別所有可能的互斥關係
"""

import json
import re
from collections import defaultdict


def analyze_tag_library():
    """深入分析標籤庫的所有標籤"""
    with open("data/tags.json", encoding="utf-8") as f:
        data = json.load(f)

    # 按類別分組標籤
    categories = {
        "age": [],  # 年齡相關
        "body": [],  # 身體特徵
        "hair_color": [],  # 髮色
        "eye_color": [],  # 眼色
        "species": [],  # 種族/角色類型
        "clothing": [],  # 服裝
        "themes": [],  # 主題/場景
        "actions": [],  # 動作
        "body_parts": [],  # 身體部位
        "relationships": [],  # 關係
        "settings": [],  # 場景設定
    }

    # 分類邏輯
    for tag in data:
        tag_name = tag["tag_name"]
        desc = tag["description"]

        # 年齡相關
        if any(
            keyword in tag_name
            for keyword in ["蘿莉", "正太", "少女", "熟女", "人妻", "老太婆", "年齡"]
        ):
            categories["age"].append(tag)

        # 髮色
        elif tag_name.endswith("髮"):
            categories["hair_color"].append(tag)

        # 身體特徵
        elif any(
            keyword in tag_name
            for keyword in [
                "巨乳",
                "貧乳",
                "平胸",
                "爆乳",
                "肌肉",
                "大肌肉",
                "孕婦",
                "懷孕",
            ]
        ):
            categories["body"].append(tag)

        # 種族/角色類型
        elif any(
            keyword in tag_name
            for keyword in [
                "娘",
                "女孩",
                "女",
                "精靈",
                "惡魔",
                "天使",
                "怪",
                "獸",
                "福瑞",
                "機器人",
            ]
        ):
            categories["species"].append(tag)

        # 服裝
        elif any(
            keyword in tag_name
            for keyword in ["服", "裝", "制服", "泳裝", "內衣", "校服", "女僕"]
        ):
            categories["clothing"].append(tag)

        # 主題/場景
        elif any(
            keyword in tag_name
            for keyword in ["強姦", "肛交", "純愛", "NTR", "群交", "調教"]
        ):
            categories["themes"].append(tag)

        # 身體部位
        elif any(
            keyword in tag_name for keyword in ["乳", "胸", "臀", "腿", "足", "手"]
        ):
            if tag not in categories["body"]:
                categories["body_parts"].append(tag)

    return categories, data


def find_conflicts_by_logic(categories):
    """基於邏輯推導衝突關係"""
    conflicts = {}

    # 1. 年齡衝突（已存在，但可以完善）
    age_tags = [tag["tag_name"] for tag in categories["age"]]
    age_conflicts = {
        "蘿莉": ["熟女", "人妻", "御姐", "老太婆"],
        "正太": ["人妻", "熟女", "老太婆"],
        "少女": ["熟女", "人妻", "老太婆"],
        "熟女": ["蘿莉", "正太", "少女"],
        "人妻": ["蘿莉", "正太", "少女"],
        "老太婆": ["蘿莉", "正太", "少女", "熟女", "人妻"],
    }
    conflicts.update(age_conflicts)

    # 2. 身體特徵衝突（擴展現有）
    body_tags = [tag["tag_name"] for tag in categories["body"]]
    body_conflicts = {
        "巨乳": ["貧乳", "平胸"],
        "貧乳": ["巨乳", "爆乳"],
        "平胸": ["巨乳", "爆乳"],
        "爆乳": ["貧乳", "平胸"],
        "懷孕": ["男性角色", " futanari"],  # 邏輯上懷孕不能與男性角色同時存在
    }
    conflicts.update(body_conflicts)

    # 3. 髮色衝突（已實施）
    hair_tags = [tag["tag_name"] for tag in categories["hair_color"]]
    for i, tag1 in enumerate(hair_tags):
        conflicts[tag1] = [tag2 for j, tag2 in enumerate(hair_tags) if i != j]

    # 4. 種族/角色類型衝突
    species_conflicts = {}
    species_tags = [tag["tag_name"] for tag in categories["species"]]

    # 人類 vs 非人類
    human_like = ["人妻", "熟女", "蘿莉", "正太", "少女"]
    non_human = [
        tag
        for tag in species_tags
        if any(
            keyword in tag
            for keyword in ["娘", "精靈", "惡魔", "天使", "怪", "獸", "福瑞", "機器人"]
        )
    ]

    for human in human_like:
        if human in species_tags:
            conflicts[human] = non_human[:5]  # 限制數量避免過多衝突

    # 特定種族衝突
    species_pairs = [
        ("天使", ["惡魔", "蝙蝠女", "哥布林"]),
        ("惡魔", ["天使"]),
        ("精靈", ["哥布林", "怪物"]),
        ("機器人", ["植物娘", "幽靈", "屍體"]),
    ]

    for species, conflict_list in species_pairs:
        if species in species_tags:
            conflicts[species] = conflict_list

    # 5. 主題/場景衝突（擴展現有）
    theme_conflicts = {
        "純愛": ["強姦", "NTR", "調教", "群交"],
        "強姦": ["純愛"],
        "NTR": ["純愛"],
        "調教": ["純愛"],
        "群交": ["純愛"],
    }
    conflicts.update(theme_conflicts)

    return conflicts


def main():
    print("=== 標籤庫分析 ===")
    categories, all_tags = analyze_tag_library()

    print(f"總標籤數量: {len(all_tags)}")
    print("\n分類統計:")
    for cat, tags in categories.items():
        if tags:
            print(f"  {cat}: {len(tags)} 個標籤")

    print("\n=== 各類別標籤詳情 ===")
    for cat, tags in categories.items():
        if tags:
            print(f"\n【{cat}】 ({len(tags)}個):")
            for tag in tags[:10]:  # 只顯示前10個
                print(f"  - {tag['tag_name']}")
            if len(tags) > 10:
                print(f"  ... 還有 {len(tags) - 10} 個")

    print("\n=== 生成互斥規則 ===")
    conflicts = find_conflicts_by_logic(categories)

    total_conflicts = sum(len(v) for v in conflicts.values())
    print(f"總衝突規則數: {len(conflicts)} 組")
    print(f"總衝突關係數: {total_conflicts} 條")

    print("\n=== 衝突規則預覽 ===")
    for tag, conflict_list in list(conflicts.items())[:15]:
        print(f"{tag}: {conflict_list}")

    if len(conflicts) > 15:
        print(f"... 還有 {len(conflicts) - 15} 組衝突規則")

    return conflicts


if __name__ == "__main__":
    conflicts = main()
