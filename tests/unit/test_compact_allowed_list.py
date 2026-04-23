"""build_compact_prompt_fragment outputs tag-names-only by category."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.domain.tag.allowed_list import (
    CATEGORY_LABEL_ZH,
    CATEGORY_ORDER,
    build_compact_prompt_fragment,
)


def _sample_library():
    return [
        {"tag_name": "蘿莉", "description": "ignored in compact", "category": "character"},
        {"tag_name": "貓娘", "description": "ignored", "category": "character"},
        {"tag_name": "女生制服", "description": "ignored", "category": "clothing"},
        {"tag_name": "雙馬尾", "description": "ignored", "category": "body"},
        {"tag_name": "站立", "description": "ignored", "category": "action"},
        {"tag_name": "純愛", "description": "ignored", "category": "theme"},
        {"tag_name": "動漫", "description": "ignored", "category": "style"},
        {"tag_name": "noise_tag", "description": "", "category": "other"},  # dropped
    ]


def test_compact_fragment_groups_by_category():
    frag = build_compact_prompt_fragment(_sample_library())
    # Each category in CATEGORY_ORDER that has entries appears as "### {label_zh}"
    for cat in CATEGORY_ORDER:
        label = CATEGORY_LABEL_ZH[cat].split(" (")[0]  # Chinese portion only in compact
        assert f"### {label}" in frag, f"missing header for {cat}"
    # "other" bucket is excluded
    assert "noise_tag" not in frag


def test_compact_fragment_tag_names_only():
    frag = build_compact_prompt_fragment(_sample_library())
    # Tag names present
    assert "蘿莉" in frag
    assert "貓娘" in frag
    # Descriptions are NOT present
    assert "ignored" not in frag
    assert "ignored in compact" not in frag


def test_compact_fragment_each_tag_once():
    frag = build_compact_prompt_fragment(_sample_library())
    assert frag.count("蘿莉") == 1
    assert frag.count("貓娘") == 1


def test_compact_fragment_size_under_5k_on_real_library():
    """With the real 611-tag library the fragment must be < 5,000 chars."""
    from app.domain.tag.library import get_tag_library_service

    lib = get_tag_library_service()
    frag = build_compact_prompt_fragment(lib.tags)
    assert 0 < len(frag) < 5_000, f"compact fragment size out of bounds: {len(frag)}"


def test_compact_fragment_sensitive_tags_all_present():
    """No sensitive tag can be trimmed out — they're safety-critical."""
    from app.domain.tag.library import get_tag_library_service

    lib = get_tag_library_service()
    frag = build_compact_prompt_fragment(lib.tags)
    sensitive = {
        "蘿莉", "正太", "強制", "強姦", "亂倫", "肛交", "觸手", "綁縛", "輪姦",
        "獸交", "中出", "顏射", "口交", "乳交", "自慰",
    }
    for tag in sensitive:
        if tag in lib.tag_names:  # only assert for tags present in library
            assert tag in frag, f"sensitive tag '{tag}' missing from compact fragment"
