"""Test JSON parsing of VLM output, including markdown stripping."""
import pytest

from app.infrastructure.lm_studio.vlm_service import parse_vlm_json


def test_parse_vlm_json_clean():
    raw = '{"description": "a girl", "tags": [{"tag": "貓娘", "category": "character", "confidence": 0.9, "evidence": "ears"}]}'
    parsed = parse_vlm_json(raw)
    assert parsed["description"] == "a girl"
    assert len(parsed["tags"]) == 1
    assert parsed["tags"][0]["tag"] == "貓娘"


def test_parse_vlm_json_strips_markdown_fence():
    raw = '```json\n{"description": "x", "tags": []}\n```'
    parsed = parse_vlm_json(raw)
    assert parsed["description"] == "x"
    assert parsed["tags"] == []


def test_parse_vlm_json_strips_plain_fence():
    raw = '```\n{"description": "x", "tags": []}\n```'
    parsed = parse_vlm_json(raw)
    assert parsed["description"] == "x"


def test_parse_vlm_json_handles_leading_prose():
    raw = 'Sure, here is the result:\n\n{"description": "x", "tags": []}'
    parsed = parse_vlm_json(raw)
    assert parsed["description"] == "x"


def test_parse_vlm_json_returns_none_on_garbage():
    assert parse_vlm_json("this is not json at all") is None
    assert parse_vlm_json("") is None


def test_parse_vlm_json_drops_invalid_tags_silently():
    raw = '{"description": "x", "tags": [{"no_tag_field": true}, {"tag": "貓娘", "confidence": 0.8}]}'
    parsed = parse_vlm_json(raw)
    # Entries missing the required "tag" key are dropped
    assert len(parsed["tags"]) == 1
    assert parsed["tags"][0]["tag"] == "貓娘"


def test_parse_vlm_json_extracts_fenced_json_after_reasoning():
    """qwen3.6 reasoning mode: long prose followed by a fenced JSON block."""
    raw = (
        "讓我分析這張圖。\n"
        "- 角色:金髮少女\n"
        "- 服裝:制服\n"
        "這是一個簡單的分析,最終答案如下:\n\n"
        '```json\n'
        '{"description": "金髮少女穿制服", "tags": [{"tag": "雙馬尾", "confidence": 0.9}]}\n'
        '```\n'
    )
    parsed = parse_vlm_json(raw)
    assert parsed is not None
    assert parsed["description"] == "金髮少女穿制服"
    assert parsed["tags"][0]["tag"] == "雙馬尾"


def test_parse_vlm_json_picks_last_valid_when_multiple_json_blocks():
    """If reasoning contains earlier candidate objects that aren't the final
    answer, prefer the LAST valid one with a tags key."""
    raw = (
        "先想想 {\"tag\": \"draft\"} 這個看起來不太對。\n"
        "再想想。最終答案:\n"
        '{"description": "final", "tags": [{"tag": "貓娘"}]}'
    )
    parsed = parse_vlm_json(raw)
    assert parsed is not None
    assert parsed["description"] == "final"
    assert parsed["tags"][0]["tag"] == "貓娘"


def test_parse_vlm_json_unfenced_json_at_end_of_reasoning():
    """Reasoning prose that ends with a bare JSON object (no markdown fence)."""
    raw = (
        "Looking at the image I see a girl with twintails.\n"
        "Wait, checking again — yes twintails. Final:\n"
        '{"description": "twintails girl", "tags": [{"tag": "雙馬尾"}]}'
    )
    parsed = parse_vlm_json(raw)
    assert parsed is not None
    assert parsed["tags"][0]["tag"] == "雙馬尾"


def test_parse_vlm_json_returns_none_when_reasoning_has_no_final_json():
    """If the model only reasons and never commits to a JSON answer, we
    legitimately cannot parse — returning None lets the VLM-service retry."""
    raw = (
        "Let me think. The image shows a cat girl. Loli? Maybe. "
        "Actually I'm not sure. More analysis needed."
    )
    assert parse_vlm_json(raw) is None
