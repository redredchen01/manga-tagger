# Two-Stage VLM Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace BAAI/bge-m3 embedding rescue paths with a pure qwen3.6-only two-stage pipeline: Stage 1 (image → description) and Stage 2 (description → tags, text-only, temperature=0).

**Architecture:** `extract_metadata()` is refactored internally into two sequential qwen3.6 calls. The recommender's DESC_RESCUE (Stage 1b) and `_search_semantic` (Stage 4) blocks are deleted entirely. All other pipeline stages are unchanged.

**Tech Stack:** Python 3.12, httpx via `get_http_client()`, qwen3.6-35b-a3b via LM Studio REST at `http://127.0.0.1:1234/v1`, existing `parse_vlm_json()` reused for Stage 2.

---

### Task 1: Add Stage 1 and Stage 2 Prompt Functions

**Files:**
- Modify: `app/domain/prompts.py`
- Create: `tests/unit/test_two_stage_prompts.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_two_stage_prompts.py`:

```python
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
```

- [ ] **Step 2: Run to verify failure**

```
pytest tests/unit/test_two_stage_prompts.py -v
```

Expected: `ImportError: cannot import name 'get_stage1_description_prompt'`

- [ ] **Step 3: Append the two functions to `app/domain/prompts.py`**

```python
def get_stage1_description_prompt() -> str:
    """Stage 1: image → rich Chinese description. No tag list included."""
    return """你是漫畫圖像分析師。仔細觀察圖像，用中文寫出詳細描述。

請涵蓋以下六個面向（若有）：
1. 角色：外觀年齡、性別、物種/類型（例如：蘿莉、貓娘、人妻）
2. 服裝：具體衣物（例如：女生制服、比基尼、女僕裝）
3. 身體特徵：髮型、髮色、胸部大小、特殊特徵（例如：獸耳、翅膀）
4. 動作與互動：姿勢、動作、是否有性行為（如有請如實描述）
5. 藝術風格：黑白/彩色、草圖/完稿
6. 主題與氛圍：例如純愛、NTR、恐怖、奇幻

只描述你明確看到的。不要猜測。3–6 句話。

/no_think"""


def get_stage2_tag_selection_prompt(description: str, allowed_fragment: str) -> str:
    """Stage 2: description + allowed list → strict JSON tag selection.

    Args:
        description: plain text output from Stage 1
        allowed_fragment: compact tag list from build_compact_prompt_fragment()
    """
    return f"""根據以下圖像描述，從允許標籤列表中選出所有適用標籤。

圖像描述：
{description}

允許的標籤：
{allowed_fragment}

規則：
1. 只能選允許列表中存在的標籤，不可創造新標籤
2. confidence < 0.6 的標籤不要列出
3. 同一標籤不要重複列出
4. 只輸出 JSON 物件，不要任何其他文字

/no_think

{{"tags": [
  {{"tag": "<允許列表中的標籤名>", "confidence": 0.0-1.0, "evidence": "<簡短視覺證據>"}}
]}}"""
```

- [ ] **Step 4: Run to verify passing**

```
pytest tests/unit/test_two_stage_prompts.py -v
```

Expected: 4 PASSED

- [ ] **Step 5: Commit**

```bash
git add app/domain/prompts.py tests/unit/test_two_stage_prompts.py
git commit -m "feat(prompts): add Stage 1 description and Stage 2 tag-selection prompts"
```

---

### Task 2: Add `_extract_description()` to VLM Service

**Files:**
- Modify: `app/infrastructure/lm_studio/vlm_service.py`
- Create: `tests/unit/test_two_stage_vlm.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_two_stage_vlm.py`:

```python
"""Unit tests for the two-stage VLM pipeline.

Stage 1: image → description (vision call)
Stage 2: description + allowed list → tags JSON (text-only call)
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.infrastructure.lm_studio.vlm_service import LMStudioVLMService


def _http_response(content: str):
    """Minimal synchronous mock for an httpx response returning `content`."""
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json = MagicMock(return_value={
        "choices": [{"message": {"content": content, "reasoning_content": ""}}]
    })
    return resp


@pytest.mark.asyncio
async def test_extract_description_returns_stripped_text(monkeypatch):
    """_extract_description returns the model's plain text response, stripped."""
    service = LMStudioVLMService()

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(
        return_value=_http_response("  一個藍髮少女穿著女生制服，坐在教室中。  ")
    )
    monkeypatch.setattr(
        "app.infrastructure.lm_studio.vlm_service.get_http_client",
        AsyncMock(return_value=mock_client),
    )
    monkeypatch.setattr(
        "app.infrastructure.lm_studio.vlm_service.LMStudioVLMService._prepare_image",
        MagicMock(return_value=b"prepared"),
    )

    result = await service._extract_description(b"fake-image-bytes")
    assert result == "一個藍髮少女穿著女生制服，坐在教室中。"


@pytest.mark.asyncio
async def test_extract_description_returns_empty_on_http_error(monkeypatch):
    """_extract_description returns '' when HTTP raises."""
    service = LMStudioVLMService()

    monkeypatch.setattr(
        "app.infrastructure.lm_studio.vlm_service.get_http_client",
        AsyncMock(side_effect=Exception("connection refused")),
    )
    monkeypatch.setattr(
        "app.infrastructure.lm_studio.vlm_service.LMStudioVLMService._prepare_image",
        MagicMock(return_value=b"prepared"),
    )

    result = await service._extract_description(b"fake-image-bytes")
    assert result == ""


@pytest.mark.asyncio
async def test_extract_description_returns_empty_when_choices_empty(monkeypatch):
    """_extract_description returns '' when model returns empty choices."""
    service = LMStudioVLMService()

    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json = MagicMock(return_value={"choices": []})
    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=resp)
    monkeypatch.setattr(
        "app.infrastructure.lm_studio.vlm_service.get_http_client",
        AsyncMock(return_value=mock_client),
    )
    monkeypatch.setattr(
        "app.infrastructure.lm_studio.vlm_service.LMStudioVLMService._prepare_image",
        MagicMock(return_value=b"prepared"),
    )

    result = await service._extract_description(b"fake-image-bytes")
    assert result == ""
```

- [ ] **Step 2: Run to verify failure**

```
pytest tests/unit/test_two_stage_vlm.py -v
```

Expected: `AttributeError: 'LMStudioVLMService' object has no attribute '_extract_description'`

- [ ] **Step 3: Add `_extract_description()` to `app/infrastructure/lm_studio/vlm_service.py`**

Insert after `_encode_image_to_base64` (before `extract_metadata`):

```python
async def _extract_description(self, image_bytes: bytes) -> str:
    """Stage 1: image → plain text description. Returns '' on any failure."""
    try:
        prepared = self._prepare_image(image_bytes)
        b64 = self._encode_image_to_base64(prepared)

        from app.domain.prompts import get_stage1_description_prompt
        prompt = get_stage1_description_prompt()

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    ],
                }
            ],
            "max_tokens": 1024,
            "temperature": self.temperature,
            "stream": False,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        client = await get_http_client()
        resp = await client.post(
            f"{self.base_url}/chat/completions", headers=headers, json=payload
        )
        resp.raise_for_status()
        data = resp.json()

        if data.get("choices"):
            msg = data["choices"][0]["message"]
            content = msg.get("content", "") or msg.get("reasoning_content", "")
            return content.strip()

        return ""

    except Exception as e:
        logger.warning("Stage 1 description extraction failed: %s: %s", type(e).__name__, e)
        return ""
```

- [ ] **Step 4: Run to verify passing**

```
pytest tests/unit/test_two_stage_vlm.py -v
```

Expected: 3 PASSED

- [ ] **Step 5: Commit**

```bash
git add app/infrastructure/lm_studio/vlm_service.py tests/unit/test_two_stage_vlm.py
git commit -m "feat(vlm): add _extract_description() — Stage 1 image→description"
```

---

### Task 3: Add `_select_tags_from_description()` to VLM Service

**Files:**
- Modify: `app/infrastructure/lm_studio/vlm_service.py`
- Modify: `tests/unit/test_two_stage_vlm.py`

- [ ] **Step 1: Append the failing tests to `tests/unit/test_two_stage_vlm.py`**

```python
@pytest.mark.asyncio
async def test_select_tags_parses_valid_json(monkeypatch):
    """_select_tags_from_description returns parsed tags list."""
    service = LMStudioVLMService()

    json_body = '{"tags": [{"tag": "貓娘", "confidence": 0.9, "evidence": "貓耳"}]}'
    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=_http_response(json_body))
    monkeypatch.setattr(
        "app.infrastructure.lm_studio.vlm_service.get_http_client",
        AsyncMock(return_value=mock_client),
    )

    tags = await service._select_tags_from_description(
        "一個有貓耳的少女。", "### 角色\n貓娘, 蘿莉"
    )
    assert len(tags) == 1
    assert tags[0]["tag"] == "貓娘"
    assert tags[0]["confidence"] == 0.9


@pytest.mark.asyncio
async def test_select_tags_returns_empty_on_http_error(monkeypatch):
    """_select_tags_from_description returns [] when HTTP raises."""
    service = LMStudioVLMService()

    monkeypatch.setattr(
        "app.infrastructure.lm_studio.vlm_service.get_http_client",
        AsyncMock(side_effect=Exception("network error")),
    )

    tags = await service._select_tags_from_description("desc", "fragment")
    assert tags == []


@pytest.mark.asyncio
async def test_select_tags_retries_once_on_parse_failure(monkeypatch):
    """_select_tags_from_description retries once when first response is not valid JSON."""
    service = LMStudioVLMService()

    valid_json = '{"tags": [{"tag": "雙馬尾", "confidence": 0.85, "evidence": "clearly visible"}]}'
    mock_client = AsyncMock()
    mock_client.post = AsyncMock(side_effect=[
        _http_response("Sorry I cannot answer that."),  # first call: garbage
        _http_response(valid_json),                      # retry: valid JSON
    ])
    monkeypatch.setattr(
        "app.infrastructure.lm_studio.vlm_service.get_http_client",
        AsyncMock(return_value=mock_client),
    )

    tags = await service._select_tags_from_description("雙馬尾少女", "### 身體特徵\n雙馬尾")
    assert any(t["tag"] == "雙馬尾" for t in tags)
    assert mock_client.post.call_count == 2
```

- [ ] **Step 2: Run to verify failure**

```
pytest tests/unit/test_two_stage_vlm.py::test_select_tags_parses_valid_json tests/unit/test_two_stage_vlm.py::test_select_tags_returns_empty_on_http_error tests/unit/test_two_stage_vlm.py::test_select_tags_retries_once_on_parse_failure -v
```

Expected: `AttributeError: 'LMStudioVLMService' object has no attribute '_select_tags_from_description'`

- [ ] **Step 3: Add `_select_tags_from_description()` to `app/infrastructure/lm_studio/vlm_service.py`**

Insert after `_extract_description()`:

```python
async def _select_tags_from_description(
    self, description: str, allowed_fragment: str
) -> list[dict]:
    """Stage 2: text-only description → tags list. Returns [] on failure.

    Uses temperature=0 for deterministic selection. Retries once if the
    first response cannot be parsed as JSON.
    """
    try:
        from app.domain.prompts import get_stage2_tag_selection_prompt
        prompt = get_stage2_tag_selection_prompt(description, allowed_fragment)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1024,
            "temperature": 0.0,
            "stream": False,
        }

        client = await get_http_client()
        resp = await client.post(
            f"{self.base_url}/chat/completions", headers=headers, json=payload
        )
        resp.raise_for_status()
        data = resp.json()

        if not data.get("choices"):
            return []

        msg = data["choices"][0]["message"]
        content = msg.get("content", "") or msg.get("reasoning_content", "")
        parsed = parse_vlm_json(content)
        if parsed is not None:
            return parsed.get("tags", [])

        # Retry once with stricter reminder
        logger.warning("Stage 2 JSON parse failed; retrying with stricter prompt")
        payload["messages"][0]["content"] += (
            "\n\nIMPORTANT: Output ONLY the JSON object. Start with { and end with }. No prose."
        )
        resp2 = await client.post(
            f"{self.base_url}/chat/completions", headers=headers, json=payload
        )
        resp2.raise_for_status()
        data2 = resp2.json()
        if data2.get("choices"):
            msg2 = data2["choices"][0]["message"]
            content2 = msg2.get("content", "") or msg2.get("reasoning_content", "")
            parsed2 = parse_vlm_json(content2)
            if parsed2 is not None:
                return parsed2.get("tags", [])

        logger.error("Stage 2 tag selection failed after retry; returning []")
        return []

    except Exception as e:
        logger.warning("Stage 2 tag selection failed: %s: %s", type(e).__name__, e)
        return []
```

- [ ] **Step 4: Run all tests in the file**

```
pytest tests/unit/test_two_stage_vlm.py -v
```

Expected: 6 PASSED

- [ ] **Step 5: Commit**

```bash
git add app/infrastructure/lm_studio/vlm_service.py tests/unit/test_two_stage_vlm.py
git commit -m "feat(vlm): add _select_tags_from_description() — Stage 2 text→tags"
```

---

### Task 4: Refactor `extract_metadata()` to Orchestrate Two Stages

**Files:**
- Modify: `app/infrastructure/lm_studio/vlm_service.py`
- Modify: `tests/unit/test_two_stage_vlm.py`

- [ ] **Step 1: Append integration tests for `extract_metadata`**

Append to `tests/unit/test_two_stage_vlm.py`:

```python
@pytest.mark.asyncio
async def test_extract_metadata_returns_two_stage_result(monkeypatch):
    """extract_metadata combines Stage 1 description and Stage 2 tags."""
    service = LMStudioVLMService()

    stage1_desc = "一個藍髮貓娘穿著女生制服。"
    stage2_tags = [{"tag": "貓娘", "confidence": 0.9, "evidence": "貓耳"}]

    monkeypatch.setattr(
        "app.infrastructure.lm_studio.vlm_service.LMStudioVLMService._extract_description",
        AsyncMock(return_value=stage1_desc),
    )
    monkeypatch.setattr(
        "app.infrastructure.lm_studio.vlm_service.LMStudioVLMService._select_tags_from_description",
        AsyncMock(return_value=stage2_tags),
    )
    monkeypatch.setattr(
        "app.infrastructure.lm_studio.vlm_service.cache_manager.get",
        AsyncMock(return_value=None),
    )
    monkeypatch.setattr(
        "app.infrastructure.lm_studio.vlm_service.cache_manager.set",
        AsyncMock(),
    )
    monkeypatch.setattr(
        "app.infrastructure.lm_studio.vlm_service.cache_manager._make_key",
        MagicMock(return_value="test-key"),
    )
    monkeypatch.setattr("app.core.config.settings.USE_MOCK_SERVICES", False)

    result = await service.extract_metadata(b"fake-image-bytes")

    assert result["description"] == stage1_desc
    assert result["tags"] == stage2_tags
    assert result["source"] == "two_stage"


@pytest.mark.asyncio
async def test_extract_metadata_falls_back_when_stage1_empty(monkeypatch):
    """extract_metadata returns fallback (not crash) when Stage 1 returns ''."""
    service = LMStudioVLMService()

    monkeypatch.setattr(
        "app.infrastructure.lm_studio.vlm_service.LMStudioVLMService._extract_description",
        AsyncMock(return_value=""),
    )
    monkeypatch.setattr(
        "app.infrastructure.lm_studio.vlm_service.cache_manager.get",
        AsyncMock(return_value=None),
    )
    monkeypatch.setattr(
        "app.infrastructure.lm_studio.vlm_service.cache_manager._make_key",
        MagicMock(return_value="test-key"),
    )
    monkeypatch.setattr("app.core.config.settings.USE_MOCK_SERVICES", False)

    result = await service.extract_metadata(b"fake-image-bytes")
    assert "description" in result
    assert "tags" in result
```

- [ ] **Step 2: Run to verify failure**

```
pytest tests/unit/test_two_stage_vlm.py::test_extract_metadata_returns_two_stage_result tests/unit/test_two_stage_vlm.py::test_extract_metadata_falls_back_when_stage1_empty -v
```

Expected: FAIL (current `extract_metadata` does not call `_extract_description`)

- [ ] **Step 3: Replace the body of `extract_metadata()` in `vlm_service.py`**

The method signature (`async def extract_metadata(self, image_bytes: bytes) -> Dict[str, Any]:`) stays unchanged. Replace everything inside it with:

```python
    async def extract_metadata(self, image_bytes: bytes) -> Dict[str, Any]:
        """Extract visual metadata using a two-stage qwen3.6 pipeline.

        Stage 1 (vision): image → rich Chinese description
        Stage 2 (text-only, temperature=0): description → tags from allowed list
        """
        start_time = time.time()
        status = "success"

        if settings.USE_MOCK_SERVICES:
            logger.info("Using mock VLM service for testing")
            return get_mock_metadata()

        cache_key = cache_manager._make_key("vlm", image_bytes.hex())
        cached = await cache_manager.get(cache_key)
        if cached:
            logger.debug("VLM cache hit")
            CACHE_HITS.labels(cache_type="vlm").inc()
            return cached
        CACHE_MISSES.labels(cache_type="vlm").inc()

        try:
            # Stage 1: image → description
            description = await self._extract_description(image_bytes)
            if not description:
                logger.warning("Stage 1 returned empty description; returning fallback")
                return get_fallback_metadata("Stage 1: no description returned")

            # Stage 2: description → tags (text only, no image)
            from app.domain.tag.allowed_list import build_compact_prompt_fragment
            from app.domain.tag.library import get_tag_library_service
            tag_lib = get_tag_library_service()
            allowed_fragment = build_compact_prompt_fragment(tag_lib.tags)
            tags = await self._select_tags_from_description(description, allowed_fragment)

            logger.info("Two-stage VLM: Stage 2 selected %d tags", len(tags))

            result = {
                "description": description,
                "tags": tags,
                "source": "two_stage",
            }
            await cache_manager.set(cache_key, result)
            return result

        except Exception as e:
            logger.warning("Two-stage VLM pipeline failed: %s: %s", type(e).__name__, e)
            status = "error"
            return get_fallback_metadata(f"VLM pipeline failed: {e}")

        finally:
            duration = time.time() - start_time
            VLM_REQUEST_COUNT.labels(status=status).inc()
            VLM_LATENCY.observe(duration)
```

- [ ] **Step 4: Run all two-stage tests**

```
pytest tests/unit/test_two_stage_vlm.py -v
```

Expected: 8 PASSED

- [ ] **Step 5: Run the full unit test suite to check for regressions**

```
pytest tests/ -v --ignore=tests/integration -x
```

Expected: all PASSED (mock services path returns `get_mock_metadata()` before any staging logic)

- [ ] **Step 6: Commit**

```bash
git add app/infrastructure/lm_studio/vlm_service.py tests/unit/test_two_stage_vlm.py
git commit -m "feat(vlm): refactor extract_metadata() to two-stage pipeline"
```

---

### Task 5: Remove DESC_RESCUE Block from Recommender

**Files:**
- Modify: `app/domain/tag/recommender.py`
- Delete: `tests/unit/test_description_rescue.py`

- [ ] **Step 1: Delete the DESC_RESCUE test file**

```bash
git rm tests/unit/test_description_rescue.py
```

- [ ] **Step 2: Delete Stage 1b from `recommend_tags()` in `recommender.py`**

Remove lines 125–200 — the entire block from:

```python
            # Stage 1b: Description-embedding rescue path.
            # Always compute; merge is conditional on VLM's delivery.
            desc_candidates: List[TagRecommendation] = []
```

through to and including:

```python
            # Dual-source agreement boost for VLM tags also seen in embedding
            for r in recommendations:
                if r.source == "vlm_json" and r.tag in desc_tag_set:
                    # +0.10 matches the dual-source boost pattern used elsewhere (e.g. _merge_rag_tags)
                    r.confidence = safe_confidence(min(r.confidence + 0.10, 1.0))
                    r.reason = (r.reason or "") + " (+desc agreement)"
```

After the deletion the Stage 1 block ends at the `else:` branch closing `recommendations = self._match_with_library(...)`, and the next line should be the Stage 4 comment (which will be removed in Task 6):

```python
            # Stage 4: Semantic search (if available and needed)
            recommendations = await self._search_semantic(mapped_keywords, recommendations, top_k)
```

- [ ] **Step 3: Run the unit test suite**

```
pytest tests/ -v --ignore=tests/integration -x
```

Expected: all PASSED

- [ ] **Step 4: Commit**

```bash
git add app/domain/tag/recommender.py
git commit -m "remove(recommender): delete DESC_RESCUE Stage 1b and its tests"
```

---

### Task 6: Remove `_search_semantic` from Recommender

**Files:**
- Modify: `app/domain/tag/recommender.py`
- Delete: `tests/unit/test_semantic_fallback.py`

- [ ] **Step 1: Delete the semantic fallback test file**

```bash
git rm tests/unit/test_semantic_fallback.py
```

- [ ] **Step 2: Remove the call site in `recommend_tags()`**

Delete these two lines (currently at line ~202 after Task 5):

```python
            # Stage 4: Semantic search (if available and needed)
            recommendations = await self._search_semantic(mapped_keywords, recommendations, top_k)
```

- [ ] **Step 3: Delete the `_search_semantic()` method**

Delete the entire `async def _search_semantic(...)` method (~50 lines starting at `async def _search_semantic(`).

- [ ] **Step 4: Remove the stale comment on mapped_keywords**

Find this line in the Stage 1 JSON path:

```python
                    mapped_keywords.append(name)  # used by semantic fallback only
```

Change it to:

```python
                    mapped_keywords.append(name)
```

- [ ] **Step 5: Run the unit test suite**

```
pytest tests/ -v --ignore=tests/integration -x
```

Expected: all PASSED

- [ ] **Step 6: Commit**

```bash
git add app/domain/tag/recommender.py
git commit -m "remove(recommender): delete _search_semantic Stage 4 and its tests"
```

---

### Task 7: Remove Stale Settings from Config

**Files:**
- Modify: `app/core/config.py`

- [ ] **Step 1: Delete eight fields from `Settings` in `app/core/config.py`**

Remove these lines from the `Settings` class body:

```python
    DESC_RESCUE_ENABLED: bool = True
    DESC_RESCUE_TOP_K: int = 8
    DESC_RESCUE_THRESHOLD: float = 0.60  # looser than semantic_fallback 0.75
    DESC_RESCUE_PENALTY: float = 0.7  # confidence multiplier for description-only candidates
    DESC_RESCUE_UNDERDELIVER_THRESHOLD: int = 3  # if VLM delivered fewer than this, rescue fills freely
    DESC_RESCUE_MAX_ADDITIONS: int = 2  # cap rescue additions when VLM delivered >= threshold

    SEMANTIC_FALLBACK_TRIGGER_COUNT: int = 3  # only run semantic if VLM gave fewer than this
    SEMANTIC_FALLBACK_MAX_ADDITIONS: int = 2  # cap how many semantic tags to add
```

Leave these unchanged (still used by RAG / other paths):

```python
    USE_CHINESE_EMBEDDINGS: bool = True
    CHINESE_EMBEDDING_THRESHOLD: float = 0.75
    CHINESE_EMBEDDING_TOP_K: int = 10
```

- [ ] **Step 2: Verify no remaining references to the removed settings**

```bash
grep -rn "DESC_RESCUE\|SEMANTIC_FALLBACK_TRIGGER\|SEMANTIC_FALLBACK_MAX" app/ tests/
```

Expected: zero matches

- [ ] **Step 3: Run the full unit test suite**

```
pytest tests/ -v --ignore=tests/integration -x
```

Expected: all PASSED

- [ ] **Step 4: Commit**

```bash
git add app/core/config.py
git commit -m "remove(config): delete DESC_RESCUE_* and SEMANTIC_FALLBACK_* settings"
```

---

### Task 8: Final Verification

**Files:** read-only

- [ ] **Step 1: Confirm no embedding imports in recommender**

```bash
grep -n "embedding\|chinese_embedding\|ChineseEmbedding" app/domain/tag/recommender.py
```

Expected: zero matches

- [ ] **Step 2: Run the complete unit test suite**

```
pytest tests/ -v --ignore=tests/integration
```

Expected: all PASSED

- [ ] **Step 3: Final commit**

```bash
git add -A
git status
git commit -m "chore: two-stage VLM pipeline complete — bge-m3 embedding removed from recommender"
```
