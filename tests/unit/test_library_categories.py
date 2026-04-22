"""Verify every library entry has a valid category and sensitive tags
land in the expected categories."""
import json
from pathlib import Path

import pytest

CATEGORIES = {"character", "clothing", "body", "action", "theme", "style", "other"}
SENSITIVE_EXPECT_CATEGORY = {
    "蘿莉": {"character"},
    "正太": {"character"},
    "嬰兒": {"character"},
    "強制": {"action", "theme"},
    "肛交": {"action"},
    "觸手": {"theme", "body"},
    "綁縛": {"action", "theme"},
    "輪姦": {"action"},
    "中出": {"action"},
    "顏射": {"action"},
    "口交": {"action"},
    "乳交": {"action"},
    "自慰": {"action"},
}


@pytest.fixture(scope="module")
def library():
    return json.loads(Path("51標籤庫.json").read_text(encoding="utf-8"))


def test_every_entry_has_valid_category(library):
    for entry in library:
        assert "category" in entry, f"missing category: {entry.get('tag_name')!r}"
        assert entry["category"] in CATEGORIES, (
            f"invalid category {entry['category']!r} for tag {entry.get('tag_name')!r}"
        )


def test_other_category_is_minority(library):
    other_count = sum(1 for e in library if e.get("category") == "other")
    assert other_count < len(library) * 0.5, (
        f"too many uncategorized: {other_count}/{len(library)} in 'other'"
    )


def test_sensitive_tags_categorized_correctly(library):
    by_name = {e["tag_name"]: e for e in library}
    for tag, expected_cats in SENSITIVE_EXPECT_CATEGORY.items():
        if tag not in by_name:
            continue  # tag may not exist in this library
        actual = by_name[tag]["category"]
        assert actual in expected_cats, (
            f"sensitive tag {tag!r} categorized as {actual!r}, expected one of {expected_cats}"
        )
