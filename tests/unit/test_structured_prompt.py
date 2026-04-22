from app.domain.prompts import get_structured_prompt


def test_structured_prompt_includes_allowed_list():
    fragment = "### 角色 (Character)\n- 貓娘：x\n\n### 服裝 (Clothing)\n- 和服：y"
    prompt = get_structured_prompt(fragment)
    assert "貓娘" in prompt
    assert "和服" in prompt


def test_structured_prompt_demands_strict_json():
    prompt = get_structured_prompt("### 角色\n- 貓娘")
    # The prompt must explicitly require JSON output
    assert "json" in prompt.lower() or "JSON" in prompt
    # Must require the tags array shape
    assert "tags" in prompt
    assert "confidence" in prompt
    assert "evidence" in prompt


def test_structured_prompt_warns_against_inventing_tags():
    prompt = get_structured_prompt("### 角色\n- 貓娘")
    # Must contain anti-invention language
    assert any(p in prompt for p in ["不要創造", "不可創造", "must not invent", "Do not invent"])


def test_structured_prompt_demands_no_hedging():
    prompt = get_structured_prompt("### 角色\n- 貓娘")
    # Explicit anti-hedge language
    assert any(p in prompt for p in ["hedge", "需要更多", "證據不足", "Do not hedge"])
