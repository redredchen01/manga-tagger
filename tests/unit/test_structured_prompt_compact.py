"""get_structured_prompt with a compact fragment stays small and keeps contract."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.domain.prompts import get_structured_prompt
from app.domain.tag.allowed_list import build_compact_prompt_fragment
from app.domain.tag.library import get_tag_library_service


def test_prompt_total_size_under_5k_with_compact_fragment():
    lib = get_tag_library_service()
    frag = build_compact_prompt_fragment(lib.tags)
    prompt = get_structured_prompt(frag)
    assert len(prompt) < 5_000, f"prompt too large: {len(prompt)} chars"


def test_prompt_requires_json_only_output():
    prompt = get_structured_prompt("### 角色\n蘿莉, 貓娘")
    # Must instruct JSON-only
    assert "JSON" in prompt
    assert "只輸出 JSON" in prompt or "JSON 物件" in prompt


def test_prompt_category_is_optional_self_check():
    """Schema should label `category` as optional / self-check so VLMs that
    drop it don't fail downstream. Match is done by `tag` name alone."""
    prompt = get_structured_prompt("### 角色\n蘿莉")
    # Prompt should contain wording that indicates category is optional
    # or self-check only, not strict. Accept either form.
    lowered = prompt.lower()
    assert "optional" in lowered or "self-check" in lowered or "可選" in prompt or "自我檢查" in prompt


def test_prompt_keeps_hedge_prohibition():
    """Precision rules from spec §3.2 must not be lost."""
    prompt = get_structured_prompt("### 角色\n蘿莉")
    assert "hedge" in prompt.lower() or "需要更多" in prompt
