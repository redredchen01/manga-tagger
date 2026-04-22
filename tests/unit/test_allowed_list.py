"""Tests for allowed_list.build_prompt_fragment."""
from app.domain.tag.allowed_list import build_prompt_fragment, group_by_category


def test_group_by_category_returns_all_six_categories():
    library = [
        {"tag_name": "貓娘", "category": "character", "description": "有貓耳和貓尾巴的角色"},
        {"tag_name": "和服", "category": "clothing", "description": "日式傳統服裝"},
        {"tag_name": "雙馬尾", "category": "body", "description": "兩束髮型"},
    ]
    grouped = group_by_category(library)
    assert "character" in grouped
    assert "clothing" in grouped
    assert "body" in grouped
    assert grouped["character"][0]["tag_name"] == "貓娘"


def test_build_prompt_fragment_includes_all_tags():
    library = [
        {"tag_name": "貓娘", "category": "character", "description": "貓耳貓尾"},
        {"tag_name": "和服", "category": "clothing", "description": "日式"},
    ]
    fragment = build_prompt_fragment(library)
    assert "貓娘" in fragment
    assert "和服" in fragment
    # Each category present as a header
    assert "character" in fragment.lower() or "角色" in fragment
    assert "clothing" in fragment.lower() or "服裝" in fragment


def test_build_prompt_fragment_groups_per_category():
    library = [
        {"tag_name": "貓娘", "category": "character", "description": "x"},
        {"tag_name": "狐娘", "category": "character", "description": "y"},
        {"tag_name": "和服", "category": "clothing", "description": "z"},
    ]
    fragment = build_prompt_fragment(library)
    # Catgirl and foxgirl appear before 和服 (character section comes first)
    assert fragment.find("貓娘") < fragment.find("和服")
    assert fragment.find("狐娘") < fragment.find("和服")


def test_build_prompt_fragment_omits_other_category():
    library = [
        {"tag_name": "useful", "category": "character", "description": "x"},
        {"tag_name": "junk", "category": "other", "description": "uncategorized"},
    ]
    fragment = build_prompt_fragment(library)
    assert "useful" in fragment
    # 'other' bucket excluded from prompt — those tags are too noisy without proper category
    assert "junk" not in fragment
