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
