"""Test Enhanced Tag Matching

測試增強後的標籤庫匹配效果
"""

import sys
import json
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.tag_library_service import TagLibraryService


def test_enhanced_matching():
    """測試增強匹配功能"""
    print("="*60)
    print("增強標籤庫匹配測試")
    print("="*60)
    
    # 使用增強格式的標籤庫
    service = TagLibraryService("./data/tags_enhanced.json")
    
    print(f"\n載入標籤數量: {len(service.tag_names)}")
    print(f"有視覺線索的標籤數量: {len([t for t in service.tag_names if service.get_tag_visual_cues(t)])}")
    print(f"有相關標籤的數量: {len([t for t in service.tag_names if service.get_tag_related_tags(t)])}")
    print(f"有別名的標籤數量: {len([t for t in service.tag_names if service.get_tag_aliases(t)])}")
    
    # 測試案例
    test_cases = [
        # 基本名稱匹配
        ("蘿莉", "測試: 蘿莉"),
        ("巨乳", "測試: 巨乳"),
        ("貓娘", "測試: 貓娘"),
        
        # 英文別名匹配
        ("loli", "測試: loli (蘿莉的英文別名)"),
        ("catgirl", "測試: catgirl (貓娘的英文別名)"),
        ("oppai", "測試: oppai (巨乳的英文)"),
        ("paizuri", "測試: paizuri (乳交的英文)"),
        ("milf", "測試: milf (人妻的英文)"),
        
        # 視覺線索匹配
        ("貓耳", "測試: 貓耳 (視覺線索)"),
        ("尾巴", "測試: 尾巴 (視覺線索)"),
        ("翅膀", "測試: 翅膀 (視覺線索)"),
        
        # 相關標籤匹配
        ("貧乳", "測試: 貧乳 (蘿莉的相關標籤)"),
        ("乳溝", "測試: 乳溝 (巨乳的相關標籤)"),
        
        # 部分匹配
        ("娘", "測試: 娘 (部分匹配)"),
        ("乳房", "測試: 乳房 (胸部標籤)"),
    ]
    
    print("\n" + "="*60)
    print("匹配測試結果")
    print("="*60)
    
    for keyword, description in test_cases:
        # 使用增強匹配
        matches_enhanced = service.match_tags_by_keywords_enhanced([keyword], min_confidence=0.5)
        
        # 使用普通匹配
        matches_normal = service.match_tags_by_keywords([keyword], min_confidence=0.5)
        
        print(f"\n關鍵詞: '{keyword}' ({description})")
        print("-"*50)
        
        if matches_enhanced:
            print(f"  增強匹配 (前3):")
            for tag, conf in matches_enhanced[:3]:
                print(f"    - {tag}: {conf:.2f}")
                cues = service.get_tag_visual_cues(tag)
                aliases = service.get_tag_aliases(tag)
                if cues:
                    print(f"      視覺線索: {cues}")
                if aliases:
                    print(f"      別名: {aliases}")
        else:
            print(f"  增強匹配: 無匹配")
        
        if matches_normal:
            print(f"  普通匹配 (前3):")
            for tag, conf in matches_normal[:3]:
                print(f"    - {tag}: {conf:.2f}")
        else:
            print(f"  普通匹配: 無匹配")
    
    # 測試多關鍵詞匹配
    print("\n" + "="*60)
    print("多關鍵詞測試")
    print("="*60)
    
    multi_keywords = [
        ["貓耳", "尾巴"],
        ["巨乳", "乳溝"],
        ["loli", "flat chest"],
    ]
    
    for keywords in multi_keywords:
        matches = service.match_tags_by_keywords_enhanced(keywords, min_confidence=0.5)
        print(f"\n關鍵詞組合: {keywords}")
        print("-"*40)
        for tag, conf in matches[:5]:
            print(f"  - {tag}: {conf:.2f}")
    
    print("\n" + "="*60)
    print("測試完成")
    print("="*60)


def show_tag_details():
    """顯示特定標籤的詳細資訊"""
    print("\n" + "="*60)
    print("標籤詳細資訊")
    print("="*60)
    
    service = TagLibraryService("./data/tags_enhanced.json")
    
    tags_to_show = ["蘿莉", "貓娘", "巨乳", "乳交", "殭屍"]
    
    for tag_name in tags_to_show:
        if tag_name in service.tag_names:
            print(f"\n標籤: {tag_name}")
            print("-"*40)
            print(f"  分類: {service.get_tag_category(tag_name)}")
            print(f"  視覺線索: {service.get_tag_visual_cues(tag_name)}")
            print(f"  相關標籤: {service.get_tag_related_tags(tag_name)}")
            print(f"  負向線索: {service.get_tag_negative_cues(tag_name)}")
            print(f"  別名: {service.get_tag_aliases(tag_name)}")
            print(f"  信心度加成: {service.get_tag_confidence_boost(tag_name)}")


if __name__ == "__main__":
    test_enhanced_matching()
    show_tag_details()
