# VLM Stability + Description Rescue — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Unblock Phase 1 acceptance by (1) shrinking the VLM allowed-tag prompt from ~14k → ~3.5k chars and (2) adding a description-embedding rescue path that recovers tags when VLM returns `{"tags": []}`.

**Architecture:** Add a "compact" allowed-list builder that emits only tag names grouped by category. Wire the VLM service to it. Add Stage 1b in the tag recommender: always embed the VLM's description via bge-m3, cosine-match against the cached 611-tag matrix, and merge as secondary candidates with a confidence × 0.7 penalty and a +0.10 dual-source agreement boost. No new modules, no model swap, no two-stage VLM.

**Tech Stack:** Python 3.13, pytest / pytest-asyncio, existing bge-m3 embedding service (`app.services.chinese_embedding_service`), LM Studio qwen3.6-35b-a3b-uncensored via httpx.

**Spec:** `docs/superpowers/specs/2026-04-23-vlm-stability-rescue-design.md` (commit `2bf9cf3`)

---

## File Structure

**New files (tests):**
- `tests/unit/test_compact_allowed_list.py` — covers `build_compact_prompt_fragment`
- `tests/unit/test_structured_prompt_compact.py` — covers `get_structured_prompt` with compact fragment
- `tests/unit/test_description_rescue.py` — covers Stage 1b (5 cases)

**Modified files:**
- `app/domain/tag/allowed_list.py` — add `build_compact_prompt_fragment`
- `app/domain/prompts.py` — mark `category` field as optional self-check in schema
- `app/infrastructure/lm_studio/vlm_service.py` — switch call from `build_prompt_fragment` → `build_compact_prompt_fragment` (line 156-159)
- `app/core/config.py` — add `DESC_RESCUE_*` three constants (after line 118)
- `app/domain/tag/recommender.py` — insert Stage 1b between line 123 (end of current Stage 1) and line 125 (Stage 4 semantic)
- `tests/unit/test_semantic_fallback.py` — update docstring comments to reflect that description_rescue runs first

**Untouched (intentional):**
- Legacy keyword-extraction path in recommender (`_extract_vlm_keywords` etc)
- `app/services/*` re-export shims
- RAG gate / LLM refinement gate / sensitive verification
- Any other test file beyond the two noted

---

## Pre-flight checks

Before starting, confirm:

```bash
# Unit suite is green on current master
cd C:/tagger && python -m pytest tests/unit/ -q 2>&1 | tail -3
# Expected: "115 passed, 2 warnings"

# Canonical tag library is loadable
python -c "from app.domain.tag.library import get_tag_library_service; lib=get_tag_library_service(); print(len(lib.tags), lib.tag_library_path)"
# Expected: "611 C:\tagger\51標籤庫.json"

# LM Studio is up with qwen3.6-35b (used at eval time, not during TDD)
curl -s http://127.0.0.1:1234/v1/models | grep -q qwen3.6-35b && echo OK || echo "Load model in LM Studio first"
```

If the suite isn't 115/115, stop and fix before starting — the plan assumes that baseline.

---

## Task 1: Compact allowed-tag list builder

**Files:**
- Create: `tests/unit/test_compact_allowed_list.py`
- Modify: `app/domain/tag/allowed_list.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_compact_allowed_list.py`:

```python
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
```

- [ ] **Step 2: Run the test to confirm it fails**

```bash
python -m pytest tests/unit/test_compact_allowed_list.py -v
```

Expected: `ImportError: cannot import name 'build_compact_prompt_fragment'` (all 5 tests error).

- [ ] **Step 3: Add `build_compact_prompt_fragment` to `app/domain/tag/allowed_list.py`**

Append to `app/domain/tag/allowed_list.py` (after the existing `build_prompt_fragment`, do NOT delete the old function — it is kept for legacy / tests that still call it):

```python
def build_compact_prompt_fragment(library: List[Dict]) -> str:
    """Build a compact allowed-tag fragment: tag names only, grouped by category.

    Format:
        ### 角色
        蘿莉, 貓娘, 狐娘, ...

        ### 服裝
        女生制服, 男生制服, ...

    For the 611-tag library this emits ~3,500 chars (vs ~14,675 for the
    verbose build_prompt_fragment). The verbose descriptions drop out;
    the VLM's pretrained knowledge covers semantics, only tag names
    matter for library matching.
    """
    grouped = group_by_category(library)
    out_lines: List[str] = []
    for cat in CATEGORY_ORDER:
        entries = grouped.get(cat, [])
        if not entries:
            continue
        # Use the Chinese portion of the category label only — drop the
        # parenthesised English to save tokens.
        label_zh = CATEGORY_LABEL_ZH[cat].split(" (")[0]
        out_lines.append(f"### {label_zh}")
        names = []
        seen_names = set()
        for e in entries:
            name = (e.get("tag_name") or "").strip()
            if not name or name in seen_names:
                continue
            names.append(name)
            seen_names.add(name)
        out_lines.append(", ".join(names))
        out_lines.append("")  # blank line between sections
    return "\n".join(out_lines).strip()
```

- [ ] **Step 4: Run the test to confirm it passes**

```bash
python -m pytest tests/unit/test_compact_allowed_list.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add app/domain/tag/allowed_list.py tests/unit/test_compact_allowed_list.py
git commit -m "feat(tag): add compact allowed-tag prompt fragment builder

Outputs tag-names-only grouped by category (~3,500 chars for the full
611-tag library, vs ~14,675 for the verbose build_prompt_fragment).
Keeps the 6 category headers for VLM orientation, drops per-tag
descriptions. Legacy build_prompt_fragment retained for existing
callers; compact version will be wired into the VLM service in a
follow-up task."
```

---

## Task 2: Structured prompt + compact fragment

**Files:**
- Create: `tests/unit/test_structured_prompt_compact.py`
- Modify: `app/domain/prompts.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_structured_prompt_compact.py`:

```python
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
```

- [ ] **Step 2: Run the test to confirm it fails**

```bash
python -m pytest tests/unit/test_structured_prompt_compact.py -v
```

Expected: `test_prompt_category_is_optional_self_check` FAILS; other three may pass today or fail depending on current prompt size. Confirm at least the category test fails.

- [ ] **Step 3: Update `get_structured_prompt` in `app/domain/prompts.py`**

Locate the schema block in `get_structured_prompt` and modify the `category` line. Find:

```python
  "tags": [
    {{
      "tag": "<必須在上方允許列表中>",
      "category": "<character|clothing|body|action|theme|style>",
      "confidence": 0.0-1.0,
      "evidence": "<簡短視覺證據,10 字以內>"
    }}
  ]
```

Replace with:

```python
  "tags": [
    {{
      "tag": "<必須在上方允許列表中>",
      "category": "<character|clothing|body|action|theme|style> — optional self-check",
      "confidence": 0.0-1.0,
      "evidence": "<簡短視覺證據,10 字以內>"
    }}
  ]
```

And add a new bullet to the "精準度規則" section after rule 3, renumbering subsequent rules is NOT required. Find:

```
3. tag 欄位必須**完全等於**允許列表中的某個標籤名稱(包含中英文與標點)
4. 同一標籤不要重複列出
```

Replace with:

```
3. tag 欄位必須**完全等於**允許列表中的某個標籤名稱(包含中英文與標點)
4. category 欄位為 optional self-check——寫不出或不確定時可省略,library 匹配只看 tag 欄位
5. 同一標籤不要重複列出
```

And renumber rule 5/6 down one each:

```
6. 角色年齡標記極嚴格:
   - 「蘿莉」:僅當角色有明顯兒童特徵(身形嬌小、兒童面容、無胸部發育)
   - 「正太」:僅當角色有明顯青少年男孩特徵
   - 大眼睛是藝術風格、非年齡指標
7. 性感內容嚴格證據:
   - 「肛交」「強姦」「觸手」等:僅當有明確視覺證據
```

- [ ] **Step 4: Run the test to confirm it passes**

```bash
python -m pytest tests/unit/test_structured_prompt_compact.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add app/domain/prompts.py tests/unit/test_structured_prompt_compact.py
git commit -m "feat(prompts): mark category as optional self-check in VLM schema

qwen3.6 sometimes omits category on tags it's unsure how to classify;
the library matcher already ignores category (only tag name matters),
so the schema was misleadingly strict. Relabel category as optional
self-check and document in the precision rules."
```

---

## Task 3: Wire VLM service to the compact fragment

**Files:**
- Modify: `app/infrastructure/lm_studio/vlm_service.py` (lines 156-159)

- [ ] **Step 1: Update the VLM service import and call**

In `app/infrastructure/lm_studio/vlm_service.py`, locate lines 156-159:

```python
            from app.domain.tag.allowed_list import build_prompt_fragment
            from app.domain.tag.library import get_tag_library_service
            tag_lib = get_tag_library_service()
            allowed_fragment = build_prompt_fragment(tag_lib.tags)
```

Replace with:

```python
            from app.domain.tag.allowed_list import build_compact_prompt_fragment
            from app.domain.tag.library import get_tag_library_service
            tag_lib = get_tag_library_service()
            allowed_fragment = build_compact_prompt_fragment(tag_lib.tags)
```

- [ ] **Step 2: Run the full unit suite**

```bash
python -m pytest tests/unit/ -q
```

Expected: all passing. (Previous tests covered `build_prompt_fragment` directly; the VLM service integration is tested live in Task 8.)

- [ ] **Step 3: Commit**

```bash
git add app/infrastructure/lm_studio/vlm_service.py
git commit -m "feat(vlm): switch to compact allowed-tag fragment

Reduces the allowed-tag list in the structured-JSON prompt from
~14,675 chars to ~3,500 chars. qwen3.6-35b-a3b's attention was
unstable on the long NSFW-heavy prompt, frequently returning
tags:[]. The descriptions per tag are dropped; the VLM's pretrained
knowledge carries the semantics, and tag-name-only is what the
library matcher needs."
```

---

## Task 4: Config flags for description rescue

**Files:**
- Modify: `app/core/config.py` (after line 118)

- [ ] **Step 1: Add three config constants**

In `app/core/config.py`, locate line 118:

```python
    RAG_INFLUENCE_ENABLED: bool = False  # Phase 1: RAG library too small to trust for scoring
```

Immediately after that line, insert:

```python

    # Phase 1 v2: description-embedding rescue path. When VLM returns empty
    # or under-delivered tags, rescue candidates from cosine-matching the
    # VLM's description against the cached 611-tag matrix.
    DESC_RESCUE_ENABLED: bool = True
    DESC_RESCUE_TOP_K: int = 8
    DESC_RESCUE_THRESHOLD: float = 0.60  # looser than semantic_fallback 0.75
```

- [ ] **Step 2: Run the config test**

```bash
python -m pytest tests/unit/test_config.py -v
```

Expected: all passing. Adding new config fields with defaults should not break existing config tests.

- [ ] **Step 3: Commit**

```bash
git add app/core/config.py
git commit -m "feat(config): add DESC_RESCUE_* flags for description-embedding rescue"
```

---

## Task 5: Description rescue path (Stage 1b) in recommender

**Files:**
- Create: `tests/unit/test_description_rescue.py`
- Modify: `app/domain/tag/recommender.py` (insert Stage 1b between lines 123 and 125)

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_description_rescue.py`:

```python
"""Stage 1b: VLM description is always embedded via bge-m3 and cosine-matched
against the cached 611-tag matrix. Results are merged as secondary candidates."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.domain.tag.recommender import TagRecommenderService


def _build_service(tag_names):
    tag_lib = MagicMock()
    tag_lib.tag_names = set(tag_names)
    tag_lib.tags = []
    tag_lib.get_all_tags = MagicMock(return_value=list(tag_names))
    tag_lib.match_tags_by_keywords = MagicMock(return_value=[])
    tag_lib.suggest_related_tags = MagicMock(return_value=[])

    tag_mapper = MagicMock()
    tag_mapper.to_chinese = MagicMock(return_value=None)

    with patch(
        "app.domain.tag.recommender.get_tag_library_service", return_value=tag_lib
    ), patch(
        "app.domain.tag.recommender.get_tag_mapper", return_value=tag_mapper
    ):
        return TagRecommenderService()


def _patch_embedding(monkeypatch, matches):
    """Install a fake embedding service that returns `matches`."""
    fake_embed = MagicMock()
    fake_embed.is_available = MagicMock(return_value=True)
    fake_embed._tag_matrix_cache = "not-none"
    fake_embed.search_cached_tags = AsyncMock(return_value=matches)
    fake_embed.cache_tag_embeddings = AsyncMock()
    monkeypatch.setattr(
        "app.services.chinese_embedding_service.get_chinese_embedding_service",
        lambda: fake_embed,
    )
    monkeypatch.setattr("app.core.config.settings.USE_MOCK_SERVICES", False)
    return fake_embed


@pytest.mark.asyncio
async def test_rescue_skipped_when_description_empty(monkeypatch):
    service = _build_service({"雙馬尾", "貓娘"})
    fake_embed = _patch_embedding(monkeypatch, [])
    service._search_semantic = AsyncMock(side_effect=lambda kws, recs, k: recs)
    service._refine_with_llm = AsyncMock(side_effect=lambda recs, *a, **kw: recs)
    service._verify_and_calibrate = AsyncMock(side_effect=lambda recs, *a, **kw: recs)

    vlm_analysis = {
        "description": "",  # empty → rescue must be skipped
        "tags": [{"tag": "雙馬尾", "category": "body", "confidence": 0.9, "evidence": "x"}],
        "source": "vlm_json",
    }
    await service.recommend_tags(vlm_analysis=vlm_analysis, rag_matches=[], top_k=5)
    fake_embed.search_cached_tags.assert_not_called()


@pytest.mark.asyncio
async def test_rescue_becomes_main_when_vlm_empty(monkeypatch):
    service = _build_service({"雙馬尾", "貓娘", "少女"})
    _patch_embedding(monkeypatch, [
        {"tag": "少女", "similarity": 0.82},
        {"tag": "雙馬尾", "similarity": 0.78},
    ])
    service._search_semantic = AsyncMock(side_effect=lambda kws, recs, k: recs)
    service._refine_with_llm = AsyncMock(side_effect=lambda recs, *a, **kw: recs)
    service._verify_and_calibrate = AsyncMock(side_effect=lambda recs, *a, **kw: recs)

    vlm_analysis = {
        "description": "一個綁雙馬尾的少女,穿著制服在校園。",
        "tags": [],  # VLM gave nothing
        "source": "vlm_json",
    }
    recs = await service.recommend_tags(
        vlm_analysis=vlm_analysis, rag_matches=[], top_k=5, confidence_threshold=0.3
    )
    tags = {r.tag for r in recs}
    assert "少女" in tags
    assert "雙馬尾" in tags
    for r in recs:
        if r.tag in {"少女", "雙馬尾"}:
            assert r.source == "description_rescue"


@pytest.mark.asyncio
async def test_rescue_caps_at_2_when_vlm_delivered(monkeypatch):
    service = _build_service({"雙馬尾", "貓娘", "少女", "校服", "微笑", "室內"})
    _patch_embedding(monkeypatch, [
        {"tag": "少女", "similarity": 0.80},
        {"tag": "校服", "similarity": 0.78},
        {"tag": "微笑", "similarity": 0.76},
        {"tag": "室內", "similarity": 0.72},
    ])
    service._search_semantic = AsyncMock(side_effect=lambda kws, recs, k: recs)
    service._refine_with_llm = AsyncMock(side_effect=lambda recs, *a, **kw: recs)
    service._verify_and_calibrate = AsyncMock(side_effect=lambda recs, *a, **kw: recs)

    vlm_analysis = {
        "description": "一個雙馬尾少女在教室,微笑中。",
        "tags": [
            {"tag": "雙馬尾", "category": "body", "confidence": 0.9, "evidence": "x"},
            {"tag": "貓娘", "category": "character", "confidence": 0.85, "evidence": "y"},
            {"tag": "微笑", "category": "action", "confidence": 0.8, "evidence": "z"},
        ],
        "source": "vlm_json",
    }
    recs = await service.recommend_tags(
        vlm_analysis=vlm_analysis, rag_matches=[], top_k=10, confidence_threshold=0.3
    )
    rescue_sources = [r for r in recs if r.source == "description_rescue"]
    assert len(rescue_sources) <= 2, f"rescue must cap at 2 additions, got {len(rescue_sources)}"


@pytest.mark.asyncio
async def test_rescue_dual_source_agreement_boosts_vlm_confidence(monkeypatch):
    service = _build_service({"雙馬尾", "貓娘"})
    _patch_embedding(monkeypatch, [
        {"tag": "雙馬尾", "similarity": 0.85},  # also in VLM's list
    ])
    service._search_semantic = AsyncMock(side_effect=lambda kws, recs, k: recs)
    service._refine_with_llm = AsyncMock(side_effect=lambda recs, *a, **kw: recs)
    service._verify_and_calibrate = AsyncMock(side_effect=lambda recs, *a, **kw: recs)

    vlm_analysis = {
        "description": "雙馬尾少女",
        "tags": [
            {"tag": "雙馬尾", "category": "body", "confidence": 0.7, "evidence": "obvious"},
            {"tag": "貓娘", "category": "character", "confidence": 0.7, "evidence": "ears"},
            {"tag": "微笑", "category": "action", "confidence": 0.7, "evidence": "mouth"},
        ],
        "source": "vlm_json",
    }
    recs = await service.recommend_tags(
        vlm_analysis=vlm_analysis, rag_matches=[], top_k=10, confidence_threshold=0.3
    )
    vlm_twintail = next(r for r in recs if r.tag == "雙馬尾")
    # 0.7 baseline + 0.10 agreement boost = 0.80 (after safe_confidence)
    assert vlm_twintail.confidence >= 0.79, (
        f"expected +0.10 boost, got confidence={vlm_twintail.confidence}"
    )
    assert "+desc agreement" in vlm_twintail.reason


@pytest.mark.asyncio
async def test_rescue_survives_embedding_service_failure(monkeypatch):
    service = _build_service({"雙馬尾"})

    def _broken():
        raise ImportError("simulated")
    monkeypatch.setattr(
        "app.services.chinese_embedding_service.get_chinese_embedding_service",
        _broken,
    )
    monkeypatch.setattr("app.core.config.settings.USE_MOCK_SERVICES", False)
    service._search_semantic = AsyncMock(side_effect=lambda kws, recs, k: recs)
    service._refine_with_llm = AsyncMock(side_effect=lambda recs, *a, **kw: recs)
    service._verify_and_calibrate = AsyncMock(side_effect=lambda recs, *a, **kw: recs)

    vlm_analysis = {
        "description": "雙馬尾少女",
        "tags": [{"tag": "雙馬尾", "category": "body", "confidence": 0.9, "evidence": "x"}],
        "source": "vlm_json",
    }
    # Should not raise
    recs = await service.recommend_tags(
        vlm_analysis=vlm_analysis, rag_matches=[], top_k=5, confidence_threshold=0.3
    )
    assert any(r.tag == "雙馬尾" for r in recs)
```

- [ ] **Step 2: Run the tests to confirm they fail**

```bash
python -m pytest tests/unit/test_description_rescue.py -v
```

Expected: at least 4 of 5 FAIL (the source label is "vlm_json", not "description_rescue"; the boost / cap / empty-tag-rescue behavior doesn't exist yet). `test_rescue_skipped_when_description_empty` may pass by luck because empty description means the rescue loop is trivially a no-op on today's code — but will fail once the code path is added without the guard.

- [ ] **Step 3: Implement Stage 1b in `app/domain/tag/recommender.py`**

Locate the existing Stage 1 block ending at line 123, and the Stage 4 comment at line 125:

```python
                recommendations = self._match_with_library(mapped_keywords, confidence_threshold)

            # Stage 4: Semantic search (if available and needed)
            recommendations = await self._search_semantic(mapped_keywords, recommendations, top_k)
```

Insert the Stage 1b block between them. Replace those two lines with:

```python
                recommendations = self._match_with_library(mapped_keywords, confidence_threshold)

            # Stage 1b: Description-embedding rescue path.
            # Always compute; merge is conditional on VLM's delivery.
            desc_candidates: List[TagRecommendation] = []
            description = (
                vlm_analysis.get("description", "") if isinstance(vlm_analysis, dict) else ""
            )
            if (
                settings.DESC_RESCUE_ENABLED
                and not settings.USE_MOCK_SERVICES
                and isinstance(description, str)
                and len(description) >= 10
            ):
                try:
                    from app.services.chinese_embedding_service import (
                        get_chinese_embedding_service,
                    )

                    embed_service = get_chinese_embedding_service()
                    if embed_service and embed_service.is_available():
                        if (
                            not hasattr(embed_service, "_tag_matrix_cache")
                            or embed_service._tag_matrix_cache is None
                        ):
                            await embed_service.cache_tag_embeddings(
                                self.tag_library.get_all_tags()
                            )
                        matches = await embed_service.search_cached_tags(
                            description,
                            top_k=settings.DESC_RESCUE_TOP_K,
                            threshold=settings.DESC_RESCUE_THRESHOLD,
                        )
                        for m in matches:
                            name = m.get("tag", "")
                            if not name or name not in self.tag_library.tag_names:
                                continue
                            sim = float(m.get("similarity", 0.0))
                            desc_candidates.append(
                                TagRecommendation(
                                    tag=name,
                                    confidence=safe_confidence(sim * 0.7),
                                    source="description_rescue",
                                    reason=f"desc embed match (sim={sim:.2f})",
                                )
                            )
                except (ImportError, AttributeError, RuntimeError) as e:
                    logger.warning(
                        "Description rescue unavailable: %s: %s", type(e).__name__, e
                    )

            # Merge rescue into primary recommendations
            vlm_tag_set = {r.tag for r in recommendations}
            desc_tag_set = {dc.tag for dc in desc_candidates}

            if len(recommendations) < 3:
                # VLM under-delivered → rescue becomes main source
                for dc in desc_candidates:
                    if dc.tag not in vlm_tag_set:
                        recommendations.append(dc)
                        vlm_tag_set.add(dc.tag)
            else:
                # VLM delivered → rescue adds at most 2 non-duplicates
                added = 0
                for dc in desc_candidates:
                    if added >= 2:
                        break
                    if dc.tag not in vlm_tag_set:
                        recommendations.append(dc)
                        vlm_tag_set.add(dc.tag)
                        added += 1

            # Dual-source agreement boost for VLM tags also seen in embedding
            for r in recommendations:
                if r.source == "vlm_json" and r.tag in desc_tag_set:
                    r.confidence = safe_confidence(min(r.confidence + 0.10, 1.0))
                    r.reason = (r.reason or "") + " (+desc agreement)"

            # Stage 4: Semantic search (if available and needed)
            recommendations = await self._search_semantic(mapped_keywords, recommendations, top_k)
```

- [ ] **Step 4: Run the test to confirm it passes**

```bash
python -m pytest tests/unit/test_description_rescue.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Run the full unit suite**

```bash
python -m pytest tests/unit/ -q
```

Expected: all passing (prior tests + 9 new from Tasks 1, 2, 5 = ~124 passed).

- [ ] **Step 6: Commit**

```bash
git add app/domain/tag/recommender.py tests/unit/test_description_rescue.py
git commit -m "feat(recommender): Stage 1b description-embedding rescue path

Always embed the VLM's description via bge-m3, cosine-match against
the cached 611-tag matrix, merge as secondary candidates with
confidence × 0.7 penalty. When VLM delivered ≥ 3 tags, rescue adds
at most 2 non-duplicates. When VLM under-delivered (< 3), rescue
becomes the main source. Tags found in both VLM and embedding get
a +0.10 dual-source agreement boost.

Solves the Phase 1 failure mode where qwen3.6-35b returns tags:[]
and the pipeline had no way to recover.

Gated by DESC_RESCUE_ENABLED (default True)."
```

---

## Task 6: Update semantic_fallback test docstring for the new ordering

**Files:**
- Modify: `tests/unit/test_semantic_fallback.py` (docstring at top + any comment that now lies)

- [ ] **Step 1: Update the module docstring and inline comments**

In `tests/unit/test_semantic_fallback.py`, locate the module docstring:

```python
"""Semantic search must only kick in when VLM under-delivered."""
```

Replace with:

```python
"""Semantic search is a second-tier fallback. In the full pipeline,
Stage 1b's description_rescue already runs first and may top up
`current_recs` before `_search_semantic` is called. These tests
exercise `_search_semantic` in isolation, so the unit-level trigger
(current_recs count vs SEMANTIC_FALLBACK_TRIGGER_COUNT) is unchanged."""
```

- [ ] **Step 2: Run the test**

```bash
python -m pytest tests/unit/test_semantic_fallback.py -v
```

Expected: 3 passed (tests themselves didn't change; only docstring).

- [ ] **Step 3: Commit**

```bash
git add tests/unit/test_semantic_fallback.py
git commit -m "test: clarify semantic_fallback docstring post-rescue-path

Stage 1b description_rescue now runs before _search_semantic, which
may top up current_recs and prevent the trigger from firing. Tests
still pass at unit level; add the note to avoid future confusion."
```

---

## Task 7: Full unit suite green check before live eval

**Files:** none

- [ ] **Step 1: Run the full unit suite one more time**

```bash
python -m pytest tests/unit/ -q
```

Expected: `124 passed` (115 existing + 5 compact + 4 prompt compact + 5 rescue = 129 max; adjust for exact count).

If anything fails, do NOT proceed to live eval. Stop and triage.

---

## Task 8: Live eval + acceptance check

**Files:**
- Update: `eval_reports/phase1_pre_rescue.json` (backup of current phase1)
- Update: `eval_reports/phase1_v2.json` (new output)

- [ ] **Step 1: Snapshot the prior phase1 report as the pre-rescue baseline**

```bash
cp eval_reports/phase1.json eval_reports/phase1_pre_rescue.json
```

- [ ] **Step 2: Start the server**

```bash
# Make sure LM Studio is running with qwen3.6-35b-a3b loaded first
taskkill //F //IM python.exe //FI "MEMUSAGE gt 1000000" 2>/dev/null
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --log-level warning &
# Wait for readiness — model loading + bge-m3 load can take ~45s
for i in 1 2 3 4 5 6 7 8 9 10 11 12; do
  sleep 5
  curl -s --max-time 3 http://127.0.0.1:8000/api/v1/health > /dev/null 2>&1 && echo "UP" && break
  echo "attempt $i"
done
```

- [ ] **Step 3: Run the eval**

```bash
python scripts/eval_accuracy.py --out eval_reports/phase1_v2.json
```

Expected console output ends with a summary block. Three valid images (1 errors out on size validation — expected for the 825-byte starter file).

- [ ] **Step 4: Compare against acceptance gates**

```bash
python - <<'PY'
import json
p = json.load(open("eval_reports/phase1_v2.json", encoding="utf-8"))["summary"]
pre = json.load(open("eval_reports/phase1_pre_rescue.json", encoding="utf-8"))["summary"]

print(f"{'metric':28s} {'pre-rescue':>10s} {'v2':>10s} {'target':>10s} {'pass':>6s}")
gates = [
    ("mean_precision",       "mean_precision",          0.7,  ">="),
    ("mean_recall",          "mean_recall",             0.5,  ">="),
    ("sensitive_fp_per_img", "sensitive_fp_per_scored_image", 0.3, "<="),
    ("median_latency_s",     "median_latency_s",        60.0, "<="),  # adjusted from spec 30s
]
all_pass = True
for label, key, target, op in gates:
    pv, pr = p.get(key, 0), pre.get(key, 0)
    ok = (pv >= target) if op == ">=" else (pv <= target)
    all_pass &= ok
    print(f"{label:28s} {pr:>10} {pv:>10} {target:>10} {'OK' if ok else 'FAIL':>6s}")

print()
print("ACCEPTANCE:", "PASS" if all_pass else "FAIL — see escalation E1/E2/E3 in spec")
PY
```

If **PASS**: proceed to Step 5.

If **FAIL**: do not commit a "passed" milestone. Record observations in a follow-up note and surface to user. Possible next moves are described in spec §7.2 (E1 two-stage VLM / E2 model swap / E3 library trim).

- [ ] **Step 5: Stop the server**

```bash
taskkill //F //IM python.exe //FI "MEMUSAGE gt 1000000" 2>/dev/null
```

- [ ] **Step 6: Milestone commit**

Compose the milestone based on the acceptance result.

If PASS:

```bash
python - <<'PY' > /tmp/phase1_v2_summary.txt
import json
p = json.load(open("eval_reports/phase1_v2.json", encoding="utf-8"))["summary"]
print(json.dumps(p, indent=2))
PY

git commit --allow-empty -m "$(cat <<EOF
milestone: Phase 1 acceptance (v2) PASSED

Adjusted acceptance (spec 2026-04-23 §6.3) on 3 valid starter images:

$(cat /tmp/phase1_v2_summary.txt)

Gates:
- mean_precision ≥ 0.7 ✓
- mean_recall ≥ 0.5 ✓
- sensitive_fp/image ≤ 0.3 ✓
- median_latency ≤ 60s ✓ (relaxed from spec §6.2's 30s, qwen3.6-35b-a3b
  consumer-GPU hard limit; see spec 2026-04-23 §6.3 for the adjustment)

Known risks (not blocking):
- Small-sample noise: 3 valid images, spec §6.5 footnote calls for ≥30
  before taking the numbers as durable. Expand golden set before Phase 2.

Next: Phase 2 (sensitive verification with double-sampling consensus).
EOF
)"
```

If FAIL:

```bash
git commit --allow-empty -m "$(cat <<EOF
eval: Phase 1 v2 still below acceptance gates

See eval_reports/phase1_v2.json for per-image breakdown. Pipeline
fixes from this branch (compact prompt + description rescue) are
real, but metrics remain below spec §6.2 targets. Candidate next
steps from spec 2026-04-23 §7.2:

- E1: two-stage VLM (free-form description → embed → small allowed
  list → confirm)
- E2: model swap (needs user sign-off; qwen3.6-35b was user-locked)
- E3: library trim to top-200 tags (sacrifices long-tail coverage)

No acceptance claim on this branch.
EOF
)"
```

---

## Spec coverage check (self-review result)

Spec section → task mapping:

| Spec section | Implemented by |
|---|---|
| §2.1 Architecture overview | Tasks 3, 5 |
| §3.1–3.4 Compact prompt format + size + keep/drop | Task 1 |
| §3.5 Schema `category` optional | Task 2 |
| §3.6 Precision rules preserved | Task 2 (ensures hedge prohibition survives) |
| §4.1 Always-compute rescue | Task 5 |
| §4.2 bge-m3 via existing embedding service | Task 5 |
| §4.3 Stage 1b pseudo-code | Task 5 |
| §4.4 Three config flags | Task 4 |
| §4.5 Relation to semantic_fallback | Task 6 (test docstring update) |
| §5 Error handling matrix | Covered in Task 5 code + `test_rescue_survives_embedding_service_failure` |
| §6.1 Three new test files | Tasks 1, 2, 5 |
| §6.2 Existing tests must stay green | Task 7 |
| §6.3 Adjusted acceptance gates | Task 8 Step 4 |
| §6.4 Eval flow | Task 8 |
| §6.5 No golden-set expansion this round | Documented in Task 8 milestone |
| §7.2 Escalation if failed | Task 8 Step 6 FAIL branch |
| §7.3 Kill-switch via `DESC_RESCUE_ENABLED` | Task 4 (default True, can be flipped to False) |
| §8 Range-out list | None required — plan does not touch those files |

All spec requirements mapped to tasks. No gaps.
