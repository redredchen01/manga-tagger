"""Tests for the two-stage VLM prompt functions."""
from app.domain.prompts import get_stage1_description_prompt, get_stage2_tag_selection_prompt


def test_stage1_prompt_is_nonempty_string():
    prompt = get_stage1_description_prompt()
    assert isinstance(prompt, str)
    assert len(prompt) > 50


def test_stage2_prompt_embeds_description():
    desc = "一個貓娘坐在教室裡。"
    prompt = get_stage2_tag_selection_prompt(desc, "### 角色\n貓娘, 蘿莉")
    assert desc in prompt


def test_stage2_prompt_embeds_allowed_fragment():
    prompt = get_stage2_tag_selection_prompt("some desc", "### 角色\n貓娘, 蘿莉")
    assert "貓娘" in prompt


def test_stage2_prompt_requests_json_output():
    prompt = get_stage2_tag_selection_prompt("desc", "fragment")
    assert "tags" in prompt
