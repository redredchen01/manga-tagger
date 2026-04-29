# Two-Stage VLM Pipeline Design

**Date:** 2026-04-29  
**Status:** Approved  
**Goal:** Replace BAAI/bge-m3 embedding rescue paths with a pure qwen3.6-only two-stage pipeline.

---

## Problem

The current pipeline loads BAAI/bge-m3 (SentenceTransformer) as a fourth model alongside three LM Studio models, causing:

- **VRAM pressure** — qwen3.6-35b-a3b already saturates the consumer GPU; bge-m3 contends for memory
- **Latency** — embedding model warmup adds overhead even when the rescue path never fires
- **Poor rescue quality** — cosine similarity on description text does not reliably recover the right tags
- **Complexity** — two separate embedding rescue paths (DESC_RESCUE + semantic_fallback) with overlapping trigger conditions

---

## Solution

Two-stage VLM pipeline using only qwen3.6-35b-a3b-uncensored-hauhaucs-aggressive:

- **Stage 1 (Vision):** Image → rich Chinese description (no tag list, free-form)
- **Stage 2 (Text only):** Description + compact allowed-tag list → strict JSON tag selection at temperature=0

All tags in the output must exist in the 51標籤庫 tag library. The recommender's existing filter (`tag not in self.tag_library.tag_names`) enforces this.

---

## Architecture

### Current Pipeline

```
Image ──┬── VLM (image + tag list → description + tags JSON)
        └── RAG search (parallel)
                     │
               Recommender
               ├── Stage 1:  VLM JSON tags (primary)
               ├── Stage 1b: DESC_RESCUE via bge-m3       ← DELETE
               ├── Stage 4:  _search_semantic via bge-m3  ← DELETE
               └── Stages 5–11: RAG merge, calibration, sensitive tag verify
```

### New Pipeline

```
Image ──┬── VLM Stage 1 (image → description only)    ─┐
        └── RAG search (parallel)                       │ after Stage 1
                                                        ↓
                     VLM Stage 2 (text only → tags JSON, temp=0)
                                                        │
               Recommender (lean)
               ├── Stage 1: Stage 2 tags (same format as before)
               └── Stages 5–11: unchanged
```

### Timing

- RAG search runs in parallel with VLM Stage 1 (same as today)
- Stage 2 fires after Stage 1 completes (needs description)
- Stage 2 is text-only (no image encoding) — expected latency 5–15s vs. ~90s for Stage 1
- RAG results are typically ready before Stage 2 completes

---

## File Changes

### `app/infrastructure/lm_studio/vlm_service.py`

Add two private methods:

**`_extract_description(image_bytes: bytes) → str`**

- Sends image with a description-focused prompt (no tag list)
- Prompt asks for: character types, clothing, body features, actions, art style, themes
- temperature=0.3, max_tokens=1024
- Returns plain text description; returns `""` on failure

**`_select_tags_from_description(description: str, allowed_fragment: str) → list[dict]`**

- Text-only API call (no image)
- Prompt: "Based on the following description, select all applicable tags from the allowed list. Output JSON only."
- temperature=0, max_tokens=1024
- Reuses existing `parse_vlm_json()` for parsing
- Returns list of `{"tag": ..., "confidence": ..., "evidence": ...}` dicts

Modify **`extract_metadata(image_bytes: bytes) → dict`**:

```python
async def extract_metadata(image_bytes):
    description = await self._extract_description(image_bytes)
    if not description:
        return get_fallback_metadata("Stage 1 failed")

    allowed_fragment = build_compact_prompt_fragment(tag_lib.tags)
    tags = await self._select_tags_from_description(description, allowed_fragment)

    return {
        "description": description,
        "tags": tags,
        "source": "two_stage",
    }
```

Failure handling:

| Failure point | Action |
|---|---|
| Stage 1 no response / parse fail | Return `get_fallback_metadata(...)` |
| Stage 1 OK, Stage 2 fail | Return `{"description": ..., "tags": [], "source": "stage2_fail"}` |
| Stage 2 JSON parse fail on first attempt | Retry once with stricter prompt (same pattern as current VLM retry) |
| Stage 2 tag not in library | Dropped by recommender's existing filter |

`_select_tags_from_description` calls `parse_vlm_json()` (reused) and returns only the `tags` list from the result; `description` field in the Stage 2 response is ignored.

### `app/domain/prompts.py`

Add two new functions:

**`get_stage1_description_prompt() → str`**

Chinese prompt asking for thorough image description across six dimensions:
1. Character types (apparent age, gender, species/type)
2. Clothing and accessories
3. Body features (hair, chest, distinctive traits)
4. Actions and interactions
5. Art style (B&W/color, sketch/finished)
6. Themes and atmosphere

No tag list included — purpose is free-form perceptual description.

**`get_stage2_tag_selection_prompt(description: str, allowed_fragment: str) → str`**

Chinese prompt: "Based on the description below, select all applicable tags from the allowed list. Output JSON only with no extra text."

Embeds `description` and `allowed_fragment`. Same JSON schema as current `get_structured_prompt`:
```json
{"tags": [{"tag": "...", "confidence": 0.0-1.0, "evidence": "..."}]}
```

### `app/domain/tag/recommender.py`

**Remove entirely:**

- Stage 1b block (~50 lines): `desc_candidates`, DESC_RESCUE conditional, dual-source agreement boost
- `_search_semantic()` method (~50 lines)
- Call to `_search_semantic()` in `recommend_tags()`
- All `from app.services.chinese_embedding_service import ...` imports in recommender

**Keep unchanged:**

- Stage 1 VLM JSON path (reads `vlm_analysis["tags"]` — same format)
- Legacy keyword extraction fallback (for mock services)
- Stages 5–11 (RAG merge, calibration, sensitive tag verification, etc.)

### `app/core/config.py`

**Remove these settings:**

```python
DESC_RESCUE_ENABLED: bool
DESC_RESCUE_TOP_K: int
DESC_RESCUE_THRESHOLD: float
DESC_RESCUE_PENALTY: float
DESC_RESCUE_UNDERDELIVER_THRESHOLD: int
DESC_RESCUE_MAX_ADDITIONS: int
SEMANTIC_FALLBACK_TRIGGER_COUNT: int   # only used by _search_semantic
SEMANTIC_FALLBACK_MAX_ADDITIONS: int   # only used by _search_semantic
```

**Keep (still used by RAG or other paths):**

```python
USE_CHINESE_EMBEDDINGS: bool
CHINESE_EMBEDDING_THRESHOLD: float
CHINESE_EMBEDDING_TOP_K: int
```

### `app/domain/pipeline.py`

No functional changes. Update one progress log string for clarity:

```python
# Before:
"Starting VLM + RAG analysis..."
# After:
"Starting VLM Stage 1 (vision) + RAG in parallel..."
```

---

## Prompt Design

### Stage 1 — Description Prompt

```
你是漫畫圖像分析師。仔細觀察圖像，用中文寫出詳細描述。

請涵蓋以下六個面向：
1. 角色：外觀年齡、性別、物種/類型（如貓娘、蘿莉、人妻等）
2. 服裝：具體衣物（制服、泳裝、女僕裝等）
3. 身體特徵：髮型、髮色、胸部大小、特殊特徵
4. 動作與互動：單人動作、雙人互動、性行為（如有）
5. 藝術風格：黑白/彩色、草圖/完稿、風格類型
6. 主題與氛圍：純愛、NTR、百合、恐怖等

只描述你明確看到的。不要猜測。描述長度 3–6 句。
```

### Stage 2 — Tag Selection Prompt

```
根據以下圖像描述，從允許標籤列表中選出所有適用標籤。

描述：
{description}

允許的標籤：
{allowed_fragment}

規則：
1. 只能選允許列表中存在的標籤
2. confidence < 0.6 的標籤不要列出
3. 輸出純 JSON，不要任何其他文字

輸出格式：
{"tags": [{"tag": "標籤名", "confidence": 0.0-1.0, "evidence": "簡短視覺證據"}]}
```

---

## What Is NOT Changed

- `verify_sensitive_tag()` — already a qwen3.6 call, unchanged
- RAG service — unchanged
- Tag library loading — unchanged
- Recommender calibration, mutual exclusivity, hierarchy boost — unchanged
- `ChineseEmbeddingService` class — kept (used by RAG), just not called from recommender
- Cache logic in `extract_metadata` — cache key reused, cache stores two-stage result

---

## Acceptance Criteria

1. `extract_metadata()` returns `{"description": ..., "tags": [...], "source": "two_stage"}` with tags all present in `tag_library.tag_names`
2. No import or call to `ChineseEmbeddingService` in `recommender.py`
3. No `DESC_RESCUE_*` or `SEMANTIC_FALLBACK_*` settings in `config.py`
4. `_search_semantic()` method deleted from `recommender.py`
5. Existing tests pass (mock services path unaffected)
6. Manual smoke test on one real image returns ≥ 1 tag
