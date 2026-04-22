# Phase 1 — Tagging Pipeline Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate the four bug-class root causes in the current tagging pipeline (hedge-string contamination, low semantic threshold, RAG noise, prompt/library mismatch) and switch the VLM backend to LM Studio glm-4.6v-flash. Result: a dramatically more accurate tagger ready for the Phase 2 sensitive-verification layer.

**Architecture:** Replace the free-form description→keyword path with a structured-JSON output contract. VLM emits JSON with explicit `tags[]` constrained to the full library; pipeline trusts only that array. RAG is disabled as a scoring source (kept as metadata-only) until its index grows. Semantic embedding search becomes a fallback that triggers only when VLM under-delivers.

**Tech Stack:** Python 3.13, FastAPI, httpx (LM Studio HTTP), pytest, sentence-transformers (bge-m3, already loaded), LM Studio glm-4.6v-flash for VLM, glm-4.7-flash text model for one-shot library categorization.

**Spec:** `docs/superpowers/specs/2026-04-22-tagging-accuracy-design.md` (commit `4af020b`)

---

## File Structure

**New files:**
- `tests/golden/expected.json` — golden-set ground truth (must_have / must_not_have / nice_to_have per image)
- `tests/golden/images/*.jpg` — copies of the 4 starter test images
- `scripts/eval_accuracy.py` — runs API against golden set, computes precision/recall/sensitive metrics
- `scripts/fix_tag_categories.py` — one-shot offline library categorization (heuristic + LLM fallback)
- `app/domain/tag/allowed_list.py` — builds prompt fragment from library, grouped by category
- `app/domain/tag/constants.py` — already exists; will host `SENSITIVE_SET` (Phase 2 will populate; Phase 1 adds the file pattern only if not present)
- `tests/unit/test_allowed_list.py`
- `tests/unit/test_vlm_json_parse.py`
- `tests/unit/test_pipeline_no_hedge.py`
- `tests/unit/test_semantic_fallback.py`
- `tests/unit/test_rag_disabled.py`

**Modified files:**
- `.env` — set `USE_OLLAMA=false`, `USE_LM_STUDIO=true`
- `.env.example` — same defaults; add new toggles
- `app/core/config.py` — add `RAG_INFLUENCE_ENABLED`, `SEMANTIC_FALLBACK_TRIGGER_COUNT`, raise `CHINESE_EMBEDDING_THRESHOLD` to 0.75
- `app/domain/prompts.py` — replace `get_optimized_prompt` with `get_structured_prompt(allowed_list)`
- `app/infrastructure/lm_studio/vlm_service.py` — JSON parse with retry + drop description→keyword fallback
- `app/infrastructure/ollama/ollama_vlm_service.py` — same JSON parse logic (kept for dev)
- `app/domain/tag/recommender.py` — gate `_search_semantic`; route VLM JSON tags directly into `recommendations` instead of re-extracting from description
- `app/domain/pipeline.py` — gate RAG merge on `settings.RAG_INFLUENCE_ENABLED`
- `app/main.py` — fail-fast on LM Studio connectivity at startup
- `51標籤庫.json` — populated `category` field on every entry (output of Task 3 script)

**Untouched (intentional for Phase 1):**
- `app/services/*` legacy shim modules
- `app/interfaces/routers/*` route definitions (only their dependencies change)
- Streamlit frontend
- `app/infrastructure/lm_studio/llm_service.py` (Phase 1 doesn't use LLM synthesis)

---

## Pre-flight checks

Before starting, confirm:

```bash
# Both backends must be reachable
curl -s http://127.0.0.1:1234/v1/models | head -5      # LM Studio
curl -s http://127.0.0.1:11434/api/tags | head -5      # Ollama (still used by tests)

# glm-4.6v-flash must be loaded in LM Studio
curl -s http://127.0.0.1:1234/v1/models | grep -q glm-4.6v && echo "OK" || echo "Load glm-4.6v-flash in LM Studio first"

# glm-4.7-flash text model needed for Task 3
curl -s http://127.0.0.1:1234/v1/models | grep -q glm-4.7 && echo "OK" || echo "Load glm-4.7-flash in LM Studio first"
```

If any of the above fails, stop and fix before proceeding — the plan assumes both LM Studio models are loaded.

---

## Task 1: Phase 0 baseline — golden set + eval script

**Files:**
- Create: `tests/golden/expected.json`
- Create: `tests/golden/images/test_anime.jpg` (copy from repo root)
- Create: `tests/golden/images/test_anime_detailed.jpg`
- Create: `tests/golden/images/test_image.jpg`
- Create: `tests/golden/images/test_real_image.jpg`
- Create: `scripts/eval_accuracy.py`
- Create: `eval_reports/baseline.json` (output, gitignored)

- [ ] **Step 1: Copy starter images into golden set**

```bash
cp test_anime.jpg          tests/golden/images/test_anime.jpg
cp test_anime_detailed.jpg tests/golden/images/test_anime_detailed.jpg
cp test_image.jpg          tests/golden/images/test_image.jpg
cp test_real_image.jpg     tests/golden/images/test_real_image.jpg
```

- [ ] **Step 2: Visually inspect the 4 images and write `tests/golden/expected.json`**

Open each image in an image viewer. For each, fill in:
- `must_have`: tags you are *highly confident* the system should produce (be conservative — only obvious things)
- `must_not_have`: tags that should NOT come up. ALWAYS list the entire SFW-sensitive blocklist for SFW images: `["蘿莉", "正太", "嬰兒", "強制", "強姦", "亂倫", "獵奇", "肛交", "觸手", "綁縛", "輪姦", "獸交", "中出", "顏射", "口交", "乳交"]`
- `nice_to_have`: tags that would be good but optional

Starter content (revise after visual inspection — tags below are guesses based on filenames and earlier test runs):

```json
{
  "test_anime.jpg": {
    "description": "anime girl with twintails / animal ears",
    "must_have": ["雙馬尾", "獸耳"],
    "must_not_have": ["蘿莉", "正太", "嬰兒", "強制", "強姦", "亂倫", "獵奇", "肛交", "觸手", "綁縛", "輪姦", "獸交", "中出", "顏射", "口交", "乳交", "自慰"],
    "nice_to_have": ["尾巴", "微笑", "動漫"]
  },
  "test_anime_detailed.jpg": {
    "description": "fill after visual inspection",
    "must_have": [],
    "must_not_have": ["蘿莉", "正太", "嬰兒", "強制", "強姦", "亂倫", "獵奇", "肛交", "觸手", "綁縛", "輪姦", "獸交", "中出", "顏射", "口交", "乳交", "自慰"],
    "nice_to_have": []
  },
  "test_image.jpg": {
    "description": "fill after visual inspection",
    "must_have": [],
    "must_not_have": ["蘿莉", "正太", "嬰兒", "強制", "強姦", "亂倫", "獵奇", "肛交", "觸手", "綁縛", "輪姦", "獸交", "中出", "顏射", "口交", "乳交", "自慰"],
    "nice_to_have": []
  },
  "test_real_image.jpg": {
    "description": "fill after visual inspection",
    "must_have": [],
    "must_not_have": ["蘿莉", "正太", "嬰兒", "強制", "強姦", "亂倫", "獵奇", "肛交", "觸手", "綁縛", "輪姦", "獸交", "中出", "顏射", "口交", "乳交", "自慰"],
    "nice_to_have": []
  }
}
```

If any starter image is genuinely NSFW (review them), set `must_not_have` to exclude only the blocklist items not actually present, and add the actually-present sensitive tags to `must_have`.

- [ ] **Step 3: Write `scripts/eval_accuracy.py`**

```python
"""Evaluate tagging accuracy against the golden set.

Usage:
    python scripts/eval_accuracy.py [--api http://127.0.0.1:8000/api/v1] \\
                                    [--out eval_reports/<name>.json]
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import requests

SENSITIVE_SET = {
    "蘿莉", "正太", "嬰兒", "兒童",
    "強制", "強姦", "亂倫", "獵奇", "食人", "斷肢",
    "肛交", "觸手", "調教", "綁縛", "輪姦", "獸交",
    "中出", "顏射", "射精", "口交", "乳交", "手淫",
    "自慰", "足交", "戀足", "偷窺", "露出",
    "綠帽", "NTR", "少女", "少年", "偽娘",
}


def tag_image(api_base: str, image_path: Path) -> Dict[str, Any]:
    with open(image_path, "rb") as f:
        r = requests.post(
            f"{api_base}/tag-cover",
            files={"file": (image_path.name, f, "image/jpeg")},
            data={"top_k": "10", "confidence_threshold": "0.3"},
            timeout=240,
        )
    r.raise_for_status()
    return r.json()


def metrics(actual_tags: List[str], expected: Dict[str, List[str]]) -> Dict[str, float]:
    actual = set(actual_tags)
    must_have = set(expected.get("must_have", []))
    must_not = set(expected.get("must_not_have", []))

    tp = len(actual & must_have)
    fp = len(actual & must_not)
    fn = len(must_have - actual)

    sensitive_actual = actual & SENSITIVE_SET
    sensitive_must_not = must_not & SENSITIVE_SET
    sensitive_fp = len(sensitive_actual & sensitive_must_not)

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)

    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "sensitive_fp": sensitive_fp,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", default="http://127.0.0.1:8000/api/v1")
    parser.add_argument("--out", default="eval_reports/eval.json")
    parser.add_argument("--golden", default="tests/golden/expected.json")
    parser.add_argument("--images-dir", default="tests/golden/images")
    args = parser.parse_args()

    expected_all = json.loads(Path(args.golden).read_text(encoding="utf-8"))
    images_dir = Path(args.images_dir)

    per_image = []
    latencies = []
    for image_name, expected in expected_all.items():
        img_path = images_dir / image_name
        if not img_path.exists():
            print(f"SKIP {image_name}: not found", file=sys.stderr)
            continue
        t0 = time.time()
        try:
            response = tag_image(args.api, img_path)
        except Exception as e:
            print(f"FAIL {image_name}: {e}", file=sys.stderr)
            per_image.append({"image": image_name, "error": str(e)})
            continue
        latency = time.time() - t0
        latencies.append(latency)

        actual_tags = [t["tag"] for t in response.get("tags", [])]
        m = metrics(actual_tags, expected)
        m["latency_s"] = round(latency, 1)
        m["actual_tags"] = actual_tags
        m["image"] = image_name
        per_image.append(m)
        print(f"{image_name}  P={m['precision']} R={m['recall']} F1={m['f1']} "
              f"sens_fp={m['sensitive_fp']} {latency:.1f}s")

    summary = {
        "n": len(per_image),
        "median_latency_s": round(statistics.median(latencies), 1) if latencies else None,
        "mean_precision": round(statistics.mean(m["precision"] for m in per_image if "precision" in m), 3) if per_image else 0,
        "mean_recall": round(statistics.mean(m["recall"] for m in per_image if "recall" in m), 3) if per_image else 0,
        "mean_f1": round(statistics.mean(m["f1"] for m in per_image if "f1" in m), 3) if per_image else 0,
        "total_sensitive_fp": sum(m.get("sensitive_fp", 0) for m in per_image),
        "sensitive_fp_per_image": round(sum(m.get("sensitive_fp", 0) for m in per_image) / max(len(per_image), 1), 3),
    }

    out = {"summary": summary, "per_image": per_image}
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Add `eval_reports/` to `.gitignore`**

Append a single line to `.gitignore`:

```
eval_reports/
```

- [ ] **Step 5: Start the server (current state — Ollama backend) and run baseline**

```bash
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 > /tmp/srv.log 2>&1 &
# wait until /health returns 200 (~15-45s)
sleep 30
curl -s http://127.0.0.1:8000/api/v1/health

python scripts/eval_accuracy.py --out eval_reports/baseline.json
```

Expected: produces `eval_reports/baseline.json` with the per-image and summary metrics. Numbers will likely be low (precision ~0.3, sensitive_fp possibly >0). **Save these numbers** — they are what every subsequent phase must beat.

Stop the server: `kill $(pgrep -f "uvicorn app.main")`

- [ ] **Step 6: Commit**

```bash
git add tests/golden/expected.json \
        tests/golden/images/*.jpg \
        scripts/eval_accuracy.py \
        .gitignore
git commit -m "test: add golden set and accuracy eval script

Establishes baseline measurement infrastructure. Starter set is 4 images
copied from repo root. Eval script reports precision, recall, f1 and
sensitive_fp per image plus aggregate summary."
```

---

## Task 2: Switch VLM backend to LM Studio glm-4.6v-flash

**Files:**
- Modify: `.env`
- Modify: `.env.example`
- Modify: `app/main.py:36-49` (lifespan startup — add fail-fast on LM Studio connectivity)

- [ ] **Step 1: Update `.env` to LM Studio backend**

Replace the relevant block:

```
USE_LM_STUDIO=true
USE_MOCK_SERVICES=false
USE_OLLAMA=false

LM_STUDIO_BASE_URL=http://127.0.0.1:1234/v1
LM_STUDIO_API_KEY=lm-studio
LM_STUDIO_VISION_MODEL=zai-org/glm-4.6v-flash
LM_STUDIO_TEXT_MODEL=zai-org/glm-4.7-flash
```

- [ ] **Step 2: Update `.env.example` to mirror these defaults**

Make the same changes in `.env.example` so a fresh checkout starts on LM Studio.

- [ ] **Step 3: Add fail-fast LM Studio probe to `app/main.py` lifespan**

Currently the lifespan calls `LMStudioVLMService()` which only logs init; if LM Studio is down the service silently 0-tags. Add a probe:

In `app/main.py`, locate the lifespan function (around line 32) and modify the section after `Initializing service singletons...`:

```python
    # Populate app.state with service singletons for dependency injection
    logger.info("Initializing service singletons...")
    if settings.USE_OLLAMA:
        from app.infrastructure.ollama.ollama_vlm_service import OllamaVLMService
        app.state.vlm_service = OllamaVLMService()
        logger.info("VLM backend: Ollama")
    else:
        app.state.vlm_service = LMStudioVLMService()
        logger.info("VLM backend: LM Studio")

        # Fail-fast: confirm LM Studio is up and the vision model is loaded
        import httpx
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                r = await client.get(f"{settings.LM_STUDIO_BASE_URL.rstrip('/')}/models")
                r.raise_for_status()
                models = {m["id"] for m in r.json().get("data", [])}
                if settings.LM_STUDIO_VISION_MODEL not in models:
                    raise RuntimeError(
                        f"Vision model {settings.LM_STUDIO_VISION_MODEL} not loaded. "
                        f"Available: {sorted(models)}"
                    )
                logger.info(f"LM Studio reachable, model {settings.LM_STUDIO_VISION_MODEL} available")
        except Exception as e:
            logger.error(f"LM Studio probe failed: {e}")
            raise RuntimeError(
                f"LM Studio at {settings.LM_STUDIO_BASE_URL} is required when USE_OLLAMA=false. "
                f"Either start LM Studio with the vision model loaded, or set USE_OLLAMA=true."
            ) from e
```

- [ ] **Step 4: Smoke test — start server, verify backend, hit one image**

```bash
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 > /tmp/srv.log 2>&1 &
sleep 30
curl -s http://127.0.0.1:8000/api/v1/health | python -c "import sys,json; print(json.load(sys.stdin)['models_loaded'])"
```

Expected output should include `"vlm": "zai-org/glm-4.6v-flash"` and `"lm_studio_mode": true`.

Then a real call:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/tag-cover \
  -F "file=@test_anime.jpg" -F "top_k=5" --max-time 240 | python -m json.tool
```

Expected: response returns 200 with `vlm_description` containing detailed Chinese text (glm-4.6v output, much richer than llava). Stop the server.

- [ ] **Step 5: Commit**

```bash
git add .env .env.example app/main.py
git commit -m "config: switch VLM backend to LM Studio glm-4.6v-flash

Adds fail-fast startup probe: server refuses to start if LM Studio is
unreachable or the configured vision model is not loaded, instead of
silently returning empty tags."
```

---

## Task 3: One-shot library categorization

**Files:**
- Create: `scripts/fix_tag_categories.py`
- Modify: `51標籤庫.json` (output: every entry gains a `category` field)
- Create: `tests/unit/test_library_categories.py`

- [ ] **Step 1: Write the categorization script**

```python
"""Categorize every tag in 51標籤庫.json into one of:
  character | clothing | body | action | theme | style | other

Pass 1: keyword heuristics (fast, deterministic, ~70% coverage).
Pass 2: LLM (LM Studio glm-4.7-flash text model) for the rest.

Idempotent: skips entries that already have a non-empty `category`.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import httpx

LIBRARY = Path("51標籤庫.json")
LM_STUDIO_BASE = os.environ.get("LM_STUDIO_BASE_URL", "http://127.0.0.1:1234/v1").rstrip("/")
LM_TEXT_MODEL = os.environ.get("LM_STUDIO_TEXT_MODEL", "zai-org/glm-4.7-flash")

CATEGORIES = ["character", "clothing", "body", "action", "theme", "style", "other"]

# Heuristic keyword markers — order matters, first match wins.
HEURISTICS = [
    ("body",      ["乳", "胸", "陰", "肛", "髮", "髮型", "眼", "耳", "獸耳", "尾", "翅", "角",
                   "肌肉", "傷痕", "雀斑", "曬痕", "紋身", "雙馬尾", "馬尾", "金髮", "黑髮"]),
    ("clothing",  ["裝", "服", "衣", "裙", "褲", "鞋", "靴", "帽", "襪", "圍裙", "比基尼",
                   "和服", "浴衣", "泳衣", "緊身", "婚紗", "西裝", "制服", "內衣"]),
    ("action",    ["站", "坐", "躺", "跑", "戰", "吻", "哭", "抱", "睡", "吃", "喝",
                   "口交", "肛交", "手淫", "自慰", "強制", "強姦", "輪姦", "綁縛", "調教",
                   "凌虐", "中出", "顏射", "射精", "乳交", "獸交", "足交"]),
    ("character", ["娘", "精靈", "惡魔", "天使", "魔物", "吸血鬼", "鬼", "妖", "仙女",
                   "蘿莉", "正太", "少女", "少年", "御姐", "熟女", "人妻", "嬰兒",
                   "扶他", "偽娘", "機器人", "幽靈", "哥布林"]),
    ("theme",     ["百合", "耽美", "後宮", "NTR", "純愛", "奇幻", "科幻", "恐怖", "搞笑",
                   "溫馨", "治愈", "冒險", "懸疑", "歷史", "音樂", "運動", "亂倫", "獵奇",
                   "綠帽", "戀足", "偷窺", "露出", "綁縛", "調教"]),
    ("style",     ["黑白", "彩色", "草圖", "線稿", "完稿", "厚塗", "厚涂", "手繪",
                   "寫實", "動漫", "水墨"]),
]


def heuristic_classify(name: str, description: str) -> str | None:
    text = f"{name} {description}"
    for category, markers in HEURISTICS:
        if any(m in text for m in markers):
            return category
    return None


def llm_classify(name: str, description: str) -> str:
    prompt = f"""分類以下漫畫標籤到 7 類之一：
character / clothing / body / action / theme / style / other

標籤名稱：{name}
描述：{description}

只輸出類別名稱，不要任何其他文字。"""
    try:
        with httpx.Client(timeout=30.0) as client:
            r = client.post(
                f"{LM_STUDIO_BASE}/chat/completions",
                headers={"Authorization": "Bearer lm-studio", "Content-Type": "application/json"},
                json={
                    "model": LM_TEXT_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 16,
                    "temperature": 0.0,
                },
            )
            r.raise_for_status()
            answer = r.json()["choices"][0]["message"]["content"].strip().lower()
            for c in CATEGORIES:
                if c in answer:
                    return c
            return "other"
    except Exception as e:
        print(f"  LLM classify failed for {name!r}: {e}", file=sys.stderr)
        return "other"


def main():
    data = json.loads(LIBRARY.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        print(f"Expected JSON list at top level, got {type(data).__name__}", file=sys.stderr)
        sys.exit(1)

    counts = {c: 0 for c in CATEGORIES}
    pass_counts = {"already": 0, "heuristic": 0, "llm": 0}

    for i, entry in enumerate(data):
        name = entry.get("tag_name", "")
        desc = entry.get("description", "")

        if entry.get("category") in CATEGORIES:
            pass_counts["already"] += 1
            counts[entry["category"]] += 1
            continue

        cat = heuristic_classify(name, desc)
        if cat:
            entry["category"] = cat
            pass_counts["heuristic"] += 1
        else:
            cat = llm_classify(name, desc)
            entry["category"] = cat
            pass_counts["llm"] += 1
            time.sleep(0.05)  # be gentle on LM Studio

        counts[cat] += 1

        if (i + 1) % 50 == 0:
            print(f"  processed {i+1}/{len(data)}")

    LIBRARY.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== Category counts ===")
    for c in CATEGORIES:
        print(f"  {c:12s}: {counts[c]}")
    print(f"\n=== Source counts ===")
    for k, v in pass_counts.items():
        print(f"  {k:12s}: {v}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the script**

Confirm LM Studio is running with glm-4.7-flash loaded, then:

```bash
python scripts/fix_tag_categories.py
```

Expected: prints progress every 50 entries; finishes with category counts roughly distributed across all 7 categories (vs current 557/611 in "other"). Should take ~1-3 minutes depending on how many fall through to LLM.

- [ ] **Step 3: Spot-check sensitive tags landed in correct category**

```bash
python - <<'PY'
import json
data = json.load(open("51標籤庫.json", encoding="utf-8"))
sensitive = {"蘿莉","正太","嬰兒","強制","肛交","觸手","綁縛","輪姦","獸交","中出","顏射","口交","乳交","自慰"}
for e in data:
    if e.get("tag_name") in sensitive:
        print(f"  {e['tag_name']:8s}  → {e.get('category')}")
PY
```

All sensitive tags should land in `body`, `action`, or `character` — never `other`. If any landed wrong, manually edit and re-save.

- [ ] **Step 4: Write the regression test**

`tests/unit/test_library_categories.py`:

```python
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
```

- [ ] **Step 5: Run the test**

```bash
pytest tests/unit/test_library_categories.py -v
```

Expected: all three tests PASS. If `test_sensitive_tags_categorized_correctly` fails for any tag, manually edit `51標籤庫.json` to fix that entry's category, then re-run.

- [ ] **Step 6: Commit**

```bash
git add scripts/fix_tag_categories.py 51標籤庫.json tests/unit/test_library_categories.py
git commit -m "data: categorize all 611 tags into 7 categories

Heuristic + LLM offline pass. Replaces the prior state where 557/611
entries landed in 'other'. Sensitive tags verified in correct category."
```

---

## Task 4: Dynamic allowed-tag list builder

**Files:**
- Create: `app/domain/tag/allowed_list.py`
- Create: `tests/unit/test_allowed_list.py`

- [ ] **Step 1: Write the failing test**

`tests/unit/test_allowed_list.py`:

```python
"""Tests for allowed_list.build_prompt_fragment."""
from app.domain.tag.allowed_list import build_prompt_fragment, group_by_category


def test_group_by_category_returns_all_six_categories():
    library = [
        {"tag_name": "貓娘", "category": "character", "description": "有貓耳和貓尾巴的角色"},
        {"tag_name": "和服", "category": "clothing", "description": "日式傳統服裝"},
        {"tag_name": "雙馬尾", "category": "body", "description": "兩束髮型"},
    ]
    grouped = group_by_category(library)
    assert "character" in grouped
    assert "clothing" in grouped
    assert "body" in grouped
    assert grouped["character"][0]["tag_name"] == "貓娘"


def test_build_prompt_fragment_includes_all_tags():
    library = [
        {"tag_name": "貓娘", "category": "character", "description": "貓耳貓尾"},
        {"tag_name": "和服", "category": "clothing", "description": "日式"},
    ]
    fragment = build_prompt_fragment(library)
    assert "貓娘" in fragment
    assert "和服" in fragment
    # Each category present as a header
    assert "character" in fragment.lower() or "角色" in fragment
    assert "clothing" in fragment.lower() or "服裝" in fragment


def test_build_prompt_fragment_groups_per_category():
    library = [
        {"tag_name": "貓娘", "category": "character", "description": "x"},
        {"tag_name": "狐娘", "category": "character", "description": "y"},
        {"tag_name": "和服", "category": "clothing", "description": "z"},
    ]
    fragment = build_prompt_fragment(library)
    # Catgirl and foxgirl appear before 和服 (character section comes first)
    assert fragment.find("貓娘") < fragment.find("和服")
    assert fragment.find("狐娘") < fragment.find("和服")


def test_build_prompt_fragment_omits_other_category():
    library = [
        {"tag_name": "useful", "category": "character", "description": "x"},
        {"tag_name": "junk", "category": "other", "description": "uncategorized"},
    ]
    fragment = build_prompt_fragment(library)
    assert "useful" in fragment
    # 'other' bucket excluded from prompt — those tags are too noisy without proper category
    assert "junk" not in fragment
```

- [ ] **Step 2: Run the test to confirm it fails**

```bash
pytest tests/unit/test_allowed_list.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'app.domain.tag.allowed_list'`.

- [ ] **Step 3: Write the implementation**

`app/domain/tag/allowed_list.py`:

```python
"""Build the allowed-tag-list prompt fragment from the library.

Used by VLM prompts to constrain output to known canonical tags.
The library is grouped by category; the 'other' bucket is omitted because
those entries lack a meaningful category and would only add noise.
"""
from __future__ import annotations

from typing import Dict, List

# Display order — character first because it's most salient for tagging
CATEGORY_ORDER = ["character", "clothing", "body", "action", "theme", "style"]

CATEGORY_LABEL_ZH = {
    "character": "角色 (Character)",
    "clothing": "服裝 (Clothing)",
    "body": "身體特徵 (Body)",
    "action": "動作與互動 (Action)",
    "theme": "主題 (Theme)",
    "style": "藝術風格 (Style)",
}


def group_by_category(library: List[Dict]) -> Dict[str, List[Dict]]:
    """Group library entries by their category field."""
    grouped: Dict[str, List[Dict]] = {}
    for entry in library:
        cat = entry.get("category", "other")
        grouped.setdefault(cat, []).append(entry)
    return grouped


def build_prompt_fragment(library: List[Dict]) -> str:
    """Build a prompt fragment listing all allowed tags grouped by category.

    Format:
        ### 角色 (Character)
        - 貓娘：貓耳貓尾
        - 狐娘：狐耳狐尾

        ### 服裝 (Clothing)
        - 和服：日式
        ...
    """
    grouped = group_by_category(library)
    out_lines: List[str] = []
    for cat in CATEGORY_ORDER:
        entries = grouped.get(cat, [])
        if not entries:
            continue
        out_lines.append(f"### {CATEGORY_LABEL_ZH[cat]}")
        for e in entries:
            name = e.get("tag_name", "").strip()
            desc = (e.get("description", "") or "").strip().replace("\n", " ")
            if not name:
                continue
            if desc:
                # Truncate overly long descriptions to keep prompt size sane
                if len(desc) > 60:
                    desc = desc[:60] + "…"
                out_lines.append(f"- {name}：{desc}")
            else:
                out_lines.append(f"- {name}")
        out_lines.append("")  # blank line between sections
    return "\n".join(out_lines).strip()
```

- [ ] **Step 4: Run the test to confirm it passes**

```bash
pytest tests/unit/test_allowed_list.py -v
```

Expected: all 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add app/domain/tag/allowed_list.py tests/unit/test_allowed_list.py
git commit -m "feat(tag): build dynamic allowed-tag prompt fragment from library

Groups library entries by category and emits a clean prompt fragment.
'other' bucket excluded — those entries lack a proper category and
would only add noise to the VLM prompt."
```

---

## Task 5: New structured-JSON prompt

**Files:**
- Modify: `app/domain/prompts.py` (replace `get_optimized_prompt` with new builder)
- Create: `tests/unit/test_structured_prompt.py`

- [ ] **Step 1: Write the failing test**

`tests/unit/test_structured_prompt.py`:

```python
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
```

- [ ] **Step 2: Run the test to confirm it fails**

```bash
pytest tests/unit/test_structured_prompt.py -v
```

Expected: FAIL with `ImportError: cannot import name 'get_structured_prompt'`.

- [ ] **Step 3: Add `get_structured_prompt` to `app/domain/prompts.py`**

Append to `app/domain/prompts.py` (keep existing `get_safe_prompt` and `get_optimized_prompt` for backward compat with mock services and tests, but the pipeline will use the new one):

```python
def get_structured_prompt(allowed_list_fragment: str) -> str:
    """Build the strict-JSON prompt for VLM tag extraction.

    Args:
        allowed_list_fragment: pre-formatted allowed-tag list, typically
            from app.domain.tag.allowed_list.build_prompt_fragment.

    Returns:
        Full prompt string. The model is required to output ONLY a JSON
        object matching the schema described in the prompt.
    """
    return f"""你是漫畫圖像標籤系統。輸出**僅限**符合 JSON Schema 的結果。

## 允許的標籤（只能從以下選，不可創造新標籤）

{allowed_list_fragment}

## 輸出格式（嚴格 JSON，不可有任何其他文字）

```json
{{
  "description": "2-3 句中文描述，客觀陳述畫面內容",
  "tags": [
    {{
      "tag": "<必須在上方允許列表中>",
      "category": "<character|clothing|body|action|theme|style>",
      "confidence": 0.0-1.0,
      "evidence": "<簡短視覺證據，10 字以內>"
    }}
  ]
}}
```

## 精準度規則（嚴格遵守）

1. 視覺證據不足的標籤一律不要列（confidence < 0.6 的標籤直接拿掉，不要列出）
2. 不確定就不標。**Do not hedge**——不要寫「需要更多視覺證據」、「可能是」、「似乎」之類字串
3. tag 欄位必須**完全等於**允許列表中的某個標籤名稱（包含中英文與標點）
4. 同一標籤不要重複列出
5. 角色年齡標記極嚴格：
   - 「蘿莉」：僅當角色有明顯兒童特徵（身形嬌小、兒童面容、無胸部發育）
   - 「正太」：僅當角色有明顯青少年男孩特徵
   - 大眼睛是藝術風格、非年齡指標
6. 性感內容嚴格證據：
   - 「肛交」「強姦」「觸手」等：僅當有明確視覺證據

## 重要

只輸出 JSON 物件，不要任何 markdown 註解、解釋文字、開場白、結語。
從 `{{` 開始，以 `}}` 結束。

Response:"""
```

- [ ] **Step 4: Run the test to confirm it passes**

```bash
pytest tests/unit/test_structured_prompt.py -v
```

Expected: all 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add app/domain/prompts.py tests/unit/test_structured_prompt.py
git commit -m "feat(prompt): add structured-JSON VLM prompt builder

Replaces free-form description output with a strict JSON contract.
Existing get_optimized_prompt kept for mock/legacy paths."
```

---

## Task 6: VLM service — parse JSON output with retry, drop description-keyword fallback

**Files:**
- Modify: `app/infrastructure/lm_studio/vlm_service.py`
- Modify: `app/infrastructure/ollama/ollama_vlm_service.py` (mirror the changes for dev parity)
- Create: `tests/unit/test_vlm_json_parse.py`

- [ ] **Step 1: Write the failing test**

`tests/unit/test_vlm_json_parse.py`:

```python
"""Test JSON parsing of VLM output, including retry on malformed JSON."""
import pytest

from app.infrastructure.lm_studio.vlm_service import (
    LMStudioVLMService,
    parse_vlm_json,
)


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
    # Sometimes models emit a sentence before the JSON
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
```

- [ ] **Step 2: Run the test to confirm it fails**

```bash
pytest tests/unit/test_vlm_json_parse.py -v
```

Expected: FAIL with `ImportError: cannot import name 'parse_vlm_json'`.

- [ ] **Step 3: Add `parse_vlm_json` and refactor `extract_metadata`**

In `app/infrastructure/lm_studio/vlm_service.py`, add the parser as a module-level function (so tests can call it directly without instantiating the service) and rewrite the `extract_metadata` happy path to use it.

At the top of the module, after imports, add:

```python
import json
import re

_JSON_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*\n?(.*?)\n?```\s*$", re.DOTALL | re.IGNORECASE)
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_vlm_json(raw: str) -> dict | None:
    """Parse a VLM response into our canonical dict shape.

    Handles markdown fences and leading prose. Returns None if no valid
    JSON object can be extracted. Drops `tags` entries missing the `tag`
    field. Always returns at least `{"description": ..., "tags": [...]}`.
    """
    if not raw or not raw.strip():
        return None

    # Strip markdown fence if present
    fenced = _JSON_FENCE_RE.match(raw)
    if fenced:
        raw = fenced.group(1)

    # If there's leading prose, find the first { and last } and extract
    obj_match = _JSON_OBJECT_RE.search(raw)
    if not obj_match:
        return None

    try:
        data = json.loads(obj_match.group(0))
    except json.JSONDecodeError:
        return None

    if not isinstance(data, dict):
        return None

    description = data.get("description", "")
    if not isinstance(description, str):
        description = ""

    raw_tags = data.get("tags", [])
    if not isinstance(raw_tags, list):
        raw_tags = []

    cleaned_tags = []
    for t in raw_tags:
        if not isinstance(t, dict):
            continue
        tag_name = t.get("tag")
        if not isinstance(tag_name, str) or not tag_name.strip():
            continue
        cleaned_tags.append({
            "tag": tag_name.strip(),
            "category": t.get("category", ""),
            "confidence": float(t.get("confidence", 0.0)) if isinstance(t.get("confidence"), (int, float)) else 0.0,
            "evidence": t.get("evidence", ""),
        })

    return {"description": description, "tags": cleaned_tags}
```

Then locate the `extract_metadata` method (currently around line 70) and replace its body's "happy path" parsing block. Replace this region:

```python
            if "choices" in result and len(result["choices"]) > 0:
                message = result["choices"][0]["message"]
                content = message.get("content", "")
                reasoning = message.get("reasoning_content", "")

                # GLM models put analysis in reasoning_content
                effective_content = content if content else reasoning

                if effective_content and len(effective_content.strip()) > 5:
                    logger.info(f"VLM succeeded: {effective_content[:100]}...")

                    # Parse the response and clean up junk
                    parsed_metadata = parse_response(effective_content)

                    # If reasoning content had interesting bullet points, blend them in
                    if reasoning and reasoning != content:
                        reasoning_tags = extract_tags_from_reasoning(reasoning)
                        if reasoning_tags:
                            # Merge with raw_keywords and deduplicate
                            current_tags = parsed_metadata.get("raw_keywords", [])
                            combined = list(set(current_tags + reasoning_tags))
                            parsed_metadata["raw_keywords"] = combined
                            logger.info(f"Enriched with {len(reasoning_tags)} tags from reasoning")

                    # Cache successful result
                    await cache_manager.set(cache_key, parsed_metadata)

                    return parsed_metadata
```

With:

```python
            if "choices" in result and len(result["choices"]) > 0:
                message = result["choices"][0]["message"]
                content = message.get("content", "")
                reasoning = message.get("reasoning_content", "")
                # GLM-class models put analysis in reasoning_content if content is empty
                effective_content = content if content else reasoning

                parsed = parse_vlm_json(effective_content) if effective_content else None
                if parsed is not None:
                    logger.info(f"VLM succeeded: {len(parsed['tags'])} tags from JSON")
                    parsed["source"] = "vlm_json"
                    await cache_manager.set(cache_key, parsed)
                    return parsed

                # Parse failed — retry once with temperature=0.0 and a stricter system reminder
                logger.warning("VLM JSON parse failed; retrying once with temperature=0.0")
                payload["temperature"] = 0.0
                payload["messages"][0]["content"][0]["text"] += (
                    "\n\nIMPORTANT: previous attempt did not return valid JSON. "
                    "Output ONLY the JSON object, starting with { and ending with }. "
                    "No prose, no markdown."
                )
                response2 = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                )
                response2.raise_for_status()
                result2 = response2.json()
                if "choices" in result2 and result2["choices"]:
                    msg2 = result2["choices"][0]["message"]
                    content2 = msg2.get("content", "") or msg2.get("reasoning_content", "")
                    parsed2 = parse_vlm_json(content2)
                    if parsed2 is not None:
                        logger.info(f"VLM retry succeeded: {len(parsed2['tags'])} tags")
                        parsed2["source"] = "vlm_json_retry"
                        await cache_manager.set(cache_key, parsed2)
                        return parsed2

                logger.error("VLM JSON parse failed after retry; returning empty result")
                return {
                    "description": "",
                    "tags": [],
                    "source": "vlm_parse_fail",
                    "error": "VLM_PARSE_FAIL",
                }
```

Also at the top of `extract_metadata`, **change the prompt construction** from `get_optimized_prompt()` to the structured prompt:

Replace:

```python
            prompt = get_optimized_prompt()
```

With:

```python
            from app.domain.tag.allowed_list import build_prompt_fragment
            from app.domain.tag.library import get_tag_library_service
            tag_lib = get_tag_library_service()
            allowed_fragment = build_prompt_fragment(tag_lib.tags)
            prompt = get_structured_prompt(allowed_fragment)
```

Update the import at the top of the file:

```python
from app.domain.prompts import get_structured_prompt
```

(remove `from app.domain.prompts import get_optimized_prompt` if it's a sole import on its line; otherwise leave it)

Also remove the now-unused imports `extract_tags_from_description`, `extract_tags_from_reasoning`, `parse_response` from the top of the file since the new path doesn't use them.

- [ ] **Step 4: Mirror the same changes in `app/infrastructure/ollama/ollama_vlm_service.py`**

Apply the same `parse_vlm_json` import, prompt change, and parse-with-retry logic. Ollama uses `/api/generate` not `/v1/chat/completions`, so the response key is `result.get("response", "")` instead of `result["choices"][0]["message"]`. Adapt accordingly:

In `app/infrastructure/ollama/ollama_vlm_service.py`, change the prompt assignment:

```python
            from app.domain.tag.allowed_list import build_prompt_fragment
            from app.domain.tag.library import get_tag_library_service
            from app.infrastructure.lm_studio.vlm_service import parse_vlm_json

            tag_lib = get_tag_library_service()
            allowed_fragment = build_prompt_fragment(tag_lib.tags)
            prompt = get_structured_prompt(allowed_fragment)
```

(Add `from app.domain.prompts import get_structured_prompt` at the top.)

Then replace the success-path block:

```python
            content = result.get("response", "")

            if content and len(content.strip()) > 5:
                logger.info(f"VLM succeeded: {content[:100]}...")
                parsed_metadata = parse_response(content)
                await cache_manager.set(cache_key, parsed_metadata)
                return parsed_metadata
```

With:

```python
            content = result.get("response", "")
            parsed = parse_vlm_json(content) if content else None
            if parsed is not None:
                logger.info(f"Ollama VLM succeeded: {len(parsed['tags'])} tags from JSON")
                parsed["source"] = "vlm_json"
                await cache_manager.set(cache_key, parsed)
                return parsed

            logger.warning("Ollama VLM JSON parse failed; retrying once with temperature=0.0")
            payload["options"]["temperature"] = 0.0
            payload["prompt"] += (
                "\n\nIMPORTANT: previous attempt did not return valid JSON. "
                "Output ONLY the JSON object."
            )
            response2 = await client.post(
                f"{self.base_url}/api/generate", headers=headers, json=payload, timeout=self.timeout,
            )
            response2.raise_for_status()
            content2 = response2.json().get("response", "")
            parsed2 = parse_vlm_json(content2)
            if parsed2 is not None:
                logger.info(f"Ollama VLM retry succeeded: {len(parsed2['tags'])} tags")
                parsed2["source"] = "vlm_json_retry"
                await cache_manager.set(cache_key, parsed2)
                return parsed2

            logger.error("Ollama VLM JSON parse failed after retry")
            return {
                "description": "",
                "tags": [],
                "source": "vlm_parse_fail",
                "error": "VLM_PARSE_FAIL",
            }
```

- [ ] **Step 5: Run the test**

```bash
pytest tests/unit/test_vlm_json_parse.py -v
```

Expected: all 6 tests PASS.

- [ ] **Step 6: Smoke test against live LM Studio**

```bash
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 > /tmp/srv.log 2>&1 &
sleep 30
curl -X POST http://127.0.0.1:8000/api/v1/tag-cover \
  -F "file=@test_anime.jpg" -F "top_k=5" --max-time 240 > /tmp/out.json
python -c "
import json
d = json.load(open('/tmp/out.json', encoding='utf-8'))
print('vlm_description:', d['metadata'].get('vlm_description', '')[:200])
print('source:', d['metadata'].get('vlm_source'))
print('tags:', [t['tag'] for t in d['tags']])
"
```

Expected: `vlm_description` is non-empty, tags are valid library tags. Stop the server.

- [ ] **Step 7: Commit**

```bash
git add app/infrastructure/lm_studio/vlm_service.py \
        app/infrastructure/ollama/ollama_vlm_service.py \
        tests/unit/test_vlm_json_parse.py
git commit -m "feat(vlm): structured JSON output with retry; drop description-keyword fallback

VLM is now constrained to a JSON schema with tags array. On parse fail,
retries once with temperature=0.0 and a stricter system reminder.
Description-keyword extraction path removed from main flow — that path
was the source of hedge-string contamination."
```

---

## Task 7: Drop description-keyword extraction from recommender main path

**Files:**
- Modify: `app/domain/tag/recommender.py` (`recommend_tags`, `_extract_vlm_keywords`)
- Create: `tests/unit/test_pipeline_no_hedge.py`

- [ ] **Step 1: Write the failing test**

`tests/unit/test_pipeline_no_hedge.py`:

```python
"""Verify that hedge phrases never appear as final tags.

The system used to extract keywords from the VLM's free-form description,
which sometimes contained hedge phrases like '需要更多視覺證據'. After
Phase 1, only the structured JSON `tags` field is consumed — these
strings should never appear in final output."""
import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.domain.tag.recommender import TagRecommenderService


HEDGE_FRAGMENTS = ["需要更多", "證據不足", "可能是", "似乎", "failed", "error"]


@pytest.mark.asyncio
async def test_hedge_strings_never_become_tags():
    # Simulate a vlm_analysis from the new JSON path: explicit `tags`
    # array with curated names; description contains hedge text.
    vlm_analysis = {
        "description": "畫面中可能是少女，需要更多視覺證據才能確認年齡。",
        "tags": [
            {"tag": "雙馬尾", "category": "body", "confidence": 0.9, "evidence": "obvious"},
        ],
        "source": "vlm_json",
    }

    # Mock dependencies
    tag_lib = MagicMock()
    tag_lib.tag_names = {"雙馬尾", "貓娘", "蘿莉"}
    tag_lib.tags = []
    tag_lib.match_tags_by_keywords = MagicMock(return_value=[])

    service = TagRecommenderService(tag_lib)
    # Patch private methods that hit external services so this is a unit test
    service._search_semantic = AsyncMock(side_effect=lambda kws, recs, k: recs)
    service._refine_with_llm = AsyncMock(side_effect=lambda recs, *a, **kw: recs)
    service._verify_and_calibrate = AsyncMock(side_effect=lambda recs, *a, **kw: recs)

    recs = await service.recommend_tags(
        vlm_analysis=vlm_analysis,
        rag_matches=[],
        top_k=5,
        confidence_threshold=0.3,
    )

    final_tags = [r.tag for r in recs]
    # No hedge fragment should appear in any tag
    for tag in final_tags:
        for hedge in HEDGE_FRAGMENTS:
            assert hedge not in tag, f"hedge {hedge!r} leaked into tag {tag!r}"
    # The legitimate JSON tag should be present
    assert "雙馬尾" in final_tags
```

- [ ] **Step 2: Run the test to confirm it fails (or passes for wrong reason)**

```bash
pytest tests/unit/test_pipeline_no_hedge.py -v
```

It may pass spuriously if no hedge ever happens to match a library tag. We'll lock it in by changing the recommender to ONLY use JSON tags.

- [ ] **Step 3: Modify `recommend_tags` to consume `vlm_analysis["tags"]` directly**

In `app/domain/tag/recommender.py`, locate `recommend_tags` (around line 57). Replace its early stages (Stage 1-2 keyword extraction):

Find this block:

```python
            # Stage 1: Extract and validate keywords
            vlm_keywords = self._extract_vlm_keywords(vlm_analysis)
            vlm_is_valid = self._is_vlm_analysis_valid(vlm_analysis)

            if vlm_is_valid:
                logger.info(f"VLM valid. Extracted {len(vlm_keywords)} keywords")
            else:
                logger.warning("VLM invalid. Using RAG-only mode")
                description = vlm_analysis.get("description", "")
                if description and "failed" not in description.lower():
                    vlm_keywords = self._extract_keywords_from_text(description)

            # Stage 2: Map English keywords to Chinese
            mapped_keywords = self._map_keywords_to_chinese(vlm_keywords)

            # Stage 3: Match with tag library
            recommendations = self._match_with_library(mapped_keywords, confidence_threshold)
```

Replace with:

```python
            # Stage 1: VLM JSON tags are now authoritative; do NOT fall back to
            # description-keyword extraction (that path was the source of hedge
            # contamination).
            vlm_json_tags = vlm_analysis.get("tags", []) if isinstance(vlm_analysis, dict) else []
            mapped_keywords: List[str] = []

            if vlm_json_tags:
                # New path: use the tag names from the JSON contract directly
                logger.info(f"VLM JSON path: {len(vlm_json_tags)} tags from contract")
                for t in vlm_json_tags:
                    if not isinstance(t, dict):
                        continue
                    name = t.get("tag", "").strip()
                    if not name or name not in self.tag_library.tag_names:
                        continue
                    confidence = float(t.get("confidence", 0.6))
                    if confidence < 0.6:
                        continue
                    recommendations.append(
                        TagRecommendation(
                            tag=name,
                            confidence=safe_confidence(confidence),
                            source="vlm_json",
                            reason=t.get("evidence", "VLM JSON tag"),
                        )
                    )
                    mapped_keywords.append(name)  # used by semantic fallback only
            else:
                # Backward compat: legacy free-form analysis (mock services / Ollama dev)
                logger.warning("No VLM JSON tags; falling back to legacy keyword extraction")
                vlm_keywords = self._extract_vlm_keywords(vlm_analysis)
                mapped_keywords = self._map_keywords_to_chinese(vlm_keywords)
                recommendations = self._match_with_library(mapped_keywords, confidence_threshold)
```

- [ ] **Step 4: Run the test**

```bash
pytest tests/unit/test_pipeline_no_hedge.py -v
```

Expected: PASS. The recommender now consumes `vlm_analysis["tags"]` directly.

- [ ] **Step 5: Smoke test against live API**

```bash
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 > /tmp/srv.log 2>&1 &
sleep 30
curl -X POST http://127.0.0.1:8000/api/v1/tag-cover \
  -F "file=@test_anime.jpg" -F "top_k=5" --max-time 240 | python -m json.tool | head -30
```

Expected: tags like `雙馬尾`, `獸耳` etc. — no Chinese phrases like `需要更多視覺證據` or `卡在牆上`. Stop server.

- [ ] **Step 6: Commit**

```bash
git add app/domain/tag/recommender.py tests/unit/test_pipeline_no_hedge.py
git commit -m "fix(recommender): consume VLM JSON tags directly, drop hedge path

VLM JSON tags array is now authoritative. Legacy description-keyword
extraction kept only as fallback for mock services and dev. This eliminates
the class of bugs where hedge phrases like '需要更多視覺證據' were extracted
as keywords and matched to unrelated library tags."
```

---

## Task 8: Semantic match → fallback only

**Files:**
- Modify: `app/core/config.py`
- Modify: `app/domain/tag/recommender.py` (`_search_semantic`)
- Create: `tests/unit/test_semantic_fallback.py`

- [ ] **Step 1: Write the failing test**

`tests/unit/test_semantic_fallback.py`:

```python
"""Semantic search must only kick in when VLM under-delivered."""
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.domain.tag.recommender import TagRecommendation, TagRecommenderService


def _make_recs(n):
    return [TagRecommendation(tag=f"tag{i}", confidence=0.8, source="vlm_json", reason="x") for i in range(n)]


@pytest.mark.asyncio
async def test_semantic_skipped_when_vlm_has_enough(monkeypatch):
    tag_lib = MagicMock()
    tag_lib.tag_names = {f"tag{i}" for i in range(5)}
    tag_lib.tags = []

    service = TagRecommenderService(tag_lib)

    # Replace embedding service with one that records whether search was called
    fake_embed = MagicMock()
    fake_embed.is_available = MagicMock(return_value=True)
    fake_embed._tag_matrix_cache = "not-none"
    fake_embed.search_cached_tags = AsyncMock(return_value=[])

    import app.domain.tag.recommender as recmod
    monkeypatch.setattr(recmod, "get_chinese_embedding_service", lambda: fake_embed)

    # 5 VLM tags, threshold says trigger only if < 3
    recs = _make_recs(5)
    result = await service._search_semantic(["kw1", "kw2"], recs, top_k=10)

    # Embedding search must NOT have been called — VLM gave enough
    fake_embed.search_cached_tags.assert_not_called()
    assert len(result) == 5


@pytest.mark.asyncio
async def test_semantic_triggered_when_vlm_under_delivers(monkeypatch):
    tag_lib = MagicMock()
    tag_lib.tag_names = {"tag0", "extra"}
    tag_lib.tags = []
    tag_lib.get_all_tags = MagicMock(return_value=["tag0", "extra"])

    service = TagRecommenderService(tag_lib)

    fake_embed = MagicMock()
    fake_embed.is_available = MagicMock(return_value=True)
    fake_embed._tag_matrix_cache = "not-none"
    fake_embed.search_cached_tags = AsyncMock(return_value=[
        {"tag": "extra", "similarity": 0.82},
    ])
    fake_embed.cache_tag_embeddings = AsyncMock()

    import app.domain.tag.recommender as recmod
    monkeypatch.setattr(recmod, "get_chinese_embedding_service", lambda: fake_embed)

    # Only 1 VLM tag → under threshold
    recs = _make_recs(1)
    result = await service._search_semantic(["kw1"], recs, top_k=10)

    fake_embed.search_cached_tags.assert_called()
    tags = [r.tag for r in result]
    assert "extra" in tags  # semantic candidate added


@pytest.mark.asyncio
async def test_semantic_threshold_filters_low_similarity(monkeypatch):
    tag_lib = MagicMock()
    tag_lib.tag_names = {"low", "high"}
    tag_lib.tags = []
    tag_lib.get_all_tags = MagicMock(return_value=["low", "high"])

    service = TagRecommenderService(tag_lib)

    fake_embed = MagicMock()
    fake_embed.is_available = MagicMock(return_value=True)
    fake_embed._tag_matrix_cache = "not-none"
    # Returns one above threshold (0.80) and one below (0.65)
    fake_embed.search_cached_tags = AsyncMock(return_value=[
        {"tag": "high", "similarity": 0.80},
        {"tag": "low", "similarity": 0.65},
    ])
    fake_embed.cache_tag_embeddings = AsyncMock()

    import app.domain.tag.recommender as recmod
    monkeypatch.setattr(recmod, "get_chinese_embedding_service", lambda: fake_embed)

    recs = _make_recs(0)  # force trigger
    result = await service._search_semantic(["kw"], recs, top_k=10)

    tags = [r.tag for r in result]
    assert "high" in tags
    assert "low" not in tags  # 0.65 < 0.75 threshold
```

- [ ] **Step 2: Update config defaults**

In `app/core/config.py`, locate `CHINESE_EMBEDDING_THRESHOLD: float = 0.55` (around line 114) and change to:

```python
    CHINESE_EMBEDDING_THRESHOLD: float = 0.75
    SEMANTIC_FALLBACK_TRIGGER_COUNT: int = 3  # only run semantic if VLM gave fewer than this
    SEMANTIC_FALLBACK_MAX_ADDITIONS: int = 2  # cap how many semantic tags to add
```

- [ ] **Step 3: Modify `_search_semantic` in `app/domain/tag/recommender.py`**

Locate `_search_semantic` (around line 207). Add the trigger-count gate at the very top, after the existing mock check:

Replace the existing function body. Find:

```python
    async def _search_semantic(
        self,
        keywords: List[str],
        current_recs: List[TagRecommendation],
        top_k: int,
    ) -> List[TagRecommendation]:
        """Perform semantic search for additional tags."""
        if settings.USE_MOCK_SERVICES or len(current_recs) >= top_k:
            return current_recs
```

Change to:

```python
    async def _search_semantic(
        self,
        keywords: List[str],
        current_recs: List[TagRecommendation],
        top_k: int,
    ) -> List[TagRecommendation]:
        """Perform semantic search ONLY as a fallback when VLM under-delivered.

        Triggers when len(current_recs) < SEMANTIC_FALLBACK_TRIGGER_COUNT.
        Cap additions at SEMANTIC_FALLBACK_MAX_ADDITIONS.
        Filter results by CHINESE_EMBEDDING_THRESHOLD (0.75).
        """
        if settings.USE_MOCK_SERVICES:
            return current_recs
        if len(current_recs) >= settings.SEMANTIC_FALLBACK_TRIGGER_COUNT:
            return current_recs
        if len(current_recs) >= top_k:
            return current_recs
```

Then locate the existing per-keyword loop (around line 236):

```python
        for keyword in keywords:
            semantic_matches = await embedding_service.search_cached_tags(
                keyword, top_k=2, threshold=settings.CHINESE_EMBEDDING_THRESHOLD
            )
            for s_match in semantic_matches:
                tag_name = s_match["tag"]
                if not any(r.tag == tag_name for r in current_recs):
                    current_recs.append(
                        TagRecommendation(
                            tag=tag_name,
                            confidence=safe_confidence(
                                s_match["similarity"] * settings.SEMANTIC_MATCH_PENALTY
                            ),
                            source="semantic_match",
                            reason=f"Semantic match for '{keyword}' ({s_match['similarity']:.2f})",
                        )
                    )
        return current_recs
```

Replace with:

```python
        added_count = 0
        for keyword in keywords:
            if added_count >= settings.SEMANTIC_FALLBACK_MAX_ADDITIONS:
                break
            semantic_matches = await embedding_service.search_cached_tags(
                keyword, top_k=2, threshold=settings.CHINESE_EMBEDDING_THRESHOLD
            )
            for s_match in semantic_matches:
                if added_count >= settings.SEMANTIC_FALLBACK_MAX_ADDITIONS:
                    break
                tag_name = s_match["tag"]
                if any(r.tag == tag_name for r in current_recs):
                    continue
                current_recs.append(
                    TagRecommendation(
                        tag=tag_name,
                        confidence=safe_confidence(
                            s_match["similarity"] * settings.SEMANTIC_MATCH_PENALTY
                        ),
                        source="semantic_fallback",
                        reason=f"Semantic fallback for '{keyword}' (sim={s_match['similarity']:.2f})",
                    )
                )
                added_count += 1
        return current_recs
```

- [ ] **Step 4: Run the test**

```bash
pytest tests/unit/test_semantic_fallback.py -v
```

Expected: all 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add app/core/config.py app/domain/tag/recommender.py tests/unit/test_semantic_fallback.py
git commit -m "feat(recommender): semantic match becomes a triggered fallback

Only runs when VLM JSON gave fewer than SEMANTIC_FALLBACK_TRIGGER_COUNT
(3) tags. Threshold raised from 0.55 to 0.75. Caps additions at 2.
Renamed source label from 'semantic_match' to 'semantic_fallback' so
clients can distinguish the weaker signal."
```

---

## Task 9: Disable RAG influence on scoring

**Files:**
- Modify: `app/core/config.py`
- Modify: `app/domain/pipeline.py`
- Create: `tests/unit/test_rag_disabled.py`

- [ ] **Step 1: Write the failing test**

`tests/unit/test_rag_disabled.py`:

```python
"""When RAG_INFLUENCE_ENABLED is False, RAG matches must not become
final tags, even if their score is high. RAG metadata is still kept
for visibility."""
from unittest.mock import AsyncMock, MagicMock

import pytest


@pytest.mark.asyncio
async def test_rag_tags_excluded_when_disabled(monkeypatch):
    from app.domain import pipeline as pipemod
    from app.core.config import settings

    monkeypatch.setattr(settings, "RAG_INFLUENCE_ENABLED", False)

    # Mock services
    fake_vlm = AsyncMock()
    fake_vlm.extract_metadata = AsyncMock(return_value={
        "description": "x",
        "tags": [{"tag": "貓娘", "category": "character", "confidence": 0.9, "evidence": "ears"}],
        "source": "vlm_json",
    })
    fake_rag = AsyncMock()
    fake_rag.search_similar = AsyncMock(return_value=[
        {"id": "x", "score": 0.99, "tags": ["catgirl", "rag_only_tag"], "metadata": {}},
    ])
    fake_recommender = AsyncMock()
    fake_recommender.recommend_tags = AsyncMock(return_value=[])

    # Bypass image validation
    image_bytes = b"\xff\xd8" + b"\x00" * 4096  # not real image; validate_image is patched
    monkeypatch.setattr(pipemod, "validate_image", lambda _b: (True, None))

    result = await pipemod.run_tagging_pipeline(
        image_bytes=image_bytes,
        vlm_service=fake_vlm,
        rag_service=fake_rag,
        tag_recommender=fake_recommender,
    )

    # The recommender was called: check what rag_matches it received
    call_kwargs = fake_recommender.recommend_tags.call_args.kwargs
    rag_matches_passed = call_kwargs.get("rag_matches", [])
    assert rag_matches_passed == [], (
        f"RAG matches should be empty when RAG_INFLUENCE_ENABLED=False, "
        f"got {rag_matches_passed}"
    )
    # Metadata may still record what RAG found (informational only)
    assert "rag_matches" in result.metadata or result.metadata.get("rag_matches_count") is not None


@pytest.mark.asyncio
async def test_rag_tags_included_when_enabled(monkeypatch):
    from app.domain import pipeline as pipemod
    from app.core.config import settings

    monkeypatch.setattr(settings, "RAG_INFLUENCE_ENABLED", True)

    fake_vlm = AsyncMock()
    fake_vlm.extract_metadata = AsyncMock(return_value={
        "description": "x", "tags": [], "source": "vlm_json",
    })
    fake_rag = AsyncMock()
    fake_rag.search_similar = AsyncMock(return_value=[
        {"id": "x", "score": 0.99, "tags": ["catgirl"], "metadata": {}},
    ])
    fake_recommender = AsyncMock()
    fake_recommender.recommend_tags = AsyncMock(return_value=[])

    image_bytes = b"\xff\xd8" + b"\x00" * 4096
    monkeypatch.setattr(pipemod, "validate_image", lambda _b: (True, None))

    await pipemod.run_tagging_pipeline(
        image_bytes=image_bytes,
        vlm_service=fake_vlm,
        rag_service=fake_rag,
        tag_recommender=fake_recommender,
    )

    call_kwargs = fake_recommender.recommend_tags.call_args.kwargs
    rag_matches_passed = call_kwargs.get("rag_matches", [])
    assert len(rag_matches_passed) == 1
```

- [ ] **Step 2: Add config flag**

In `app/core/config.py`, add (next to the other tagging knobs):

```python
    RAG_INFLUENCE_ENABLED: bool = False  # Phase 1: RAG library too small to trust
```

- [ ] **Step 3: Modify the pipeline to honor the flag**

In `app/domain/pipeline.py`, locate the RAG result handling (around line 193-200):

```python
    if rag_matches is None:
        rag_matches = []
```

Add a flag-honoring wrapper right after that block, before the recommender call. Find the section that calls the recommender:

```python
    if vlm_analysis:
        recommendations = await tag_recommender.recommend_tags(
            vlm_analysis=vlm_analysis,
            rag_matches=rag_matches,
            ...
        )
```

Modify to:

```python
    # Phase 1: RAG influence on scoring is gated by config.
    # When disabled, the recommender sees an empty list (no RAG → tags),
    # but the API response still reports what RAG found in metadata for
    # visibility/debugging.
    rag_matches_for_scoring = rag_matches if settings.RAG_INFLUENCE_ENABLED else []

    if vlm_analysis:
        recommendations = await tag_recommender.recommend_tags(
            vlm_analysis=vlm_analysis,
            rag_matches=rag_matches_for_scoring,
            top_k=top_k,
            confidence_threshold=confidence_threshold,
            vlm_service=vlm_service,
            image_bytes=image_bytes,
        )
```

(Replace the existing `recommend_tags` call's `rag_matches=rag_matches` with `rag_matches=rag_matches_for_scoring`.)

- [ ] **Step 4: Run the test**

```bash
pytest tests/unit/test_rag_disabled.py -v
```

Expected: both tests PASS.

- [ ] **Step 5: Commit**

```bash
git add app/core/config.py app/domain/pipeline.py tests/unit/test_rag_disabled.py
git commit -m "feat(pipeline): gate RAG influence on scoring behind config flag

RAG_INFLUENCE_ENABLED defaults to False in Phase 1 — current 17-doc
library produces noisy contamination (every anime image gets
catgirl/yellow_hair). RAG endpoints and metadata-level visibility
preserved; only the scoring path is gated."
```

---

## Task 10: Final eval and acceptance check

**Files:**
- Update: `eval_reports/phase1.json` (output)

- [ ] **Step 1: Restart the server with all Phase 1 changes**

```bash
pkill -f "uvicorn app.main" 2>/dev/null; sleep 2
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 > /tmp/srv.log 2>&1 &
sleep 30
curl -s http://127.0.0.1:8000/api/v1/health | python -m json.tool
```

- [ ] **Step 2: Run the eval script**

```bash
python scripts/eval_accuracy.py --out eval_reports/phase1.json
```

- [ ] **Step 3: Compare against baseline and Phase 1 acceptance criteria**

```bash
python - <<'PY'
import json
b = json.load(open("eval_reports/baseline.json", encoding="utf-8"))["summary"]
p = json.load(open("eval_reports/phase1.json",   encoding="utf-8"))["summary"]

print(f"{'metric':25s} {'baseline':>10s} {'phase1':>10s} {'target':>10s} {'pass':>6s}")
checks = [
    ("mean_precision",      "mean_precision",       0.7,   "≥"),
    ("mean_recall",         "mean_recall",          0.5,   "≥"),
    ("sensitive_fp/image",  "sensitive_fp_per_image", 0.3, "≤"),
    ("median_latency_s",    "median_latency_s",    30.0,  "≤"),
]
all_pass = True
for label, key, target, op in checks:
    bv = b.get(key, 0)
    pv = p.get(key, 0)
    if op == "≥":
        passed = pv >= target
    else:
        passed = pv <= target
    all_pass &= passed
    print(f"{label:25s} {bv:>10} {pv:>10} {target:>10} {'OK' if passed else 'FAIL':>6s}")

print()
print("ACCEPTANCE:", "PASS" if all_pass else "FAIL — phase 1 not done")
PY
```

If acceptance fails:
- Check each failing metric, identify which task contributed to the regression
- Fix and re-run from Step 2
- DO NOT commit and proceed to Phase 2 until all metrics pass

If acceptance passes:

- [ ] **Step 4: Commit eval reports (baseline only — phase reports are gitignored)**

The baseline report already gitignored under `eval_reports/`. Make sure the baseline numbers are documented in the next commit message:

```bash
# Stop the server
pkill -f "uvicorn app.main" 2>/dev/null

# Read and document the metrics
python - <<'PY' > /tmp/phase1_summary.txt
import json
p = json.load(open("eval_reports/phase1.json", encoding="utf-8"))
print(json.dumps(p["summary"], indent=2))
PY
cat /tmp/phase1_summary.txt
```

Use the printed numbers in the commit message:

```bash
git commit --allow-empty -m "$(cat <<EOF
milestone: Phase 1 acceptance passed

Baseline → Phase 1 metrics on 4-image starter golden set:
$(cat /tmp/phase1_summary.txt)

All Phase 1 acceptance gates from spec §6.2 met:
- mean_precision ≥ 0.7 ✓
- mean_recall ≥ 0.5 ✓
- sensitive_fp_per_image ≤ 0.3 ✓
- median_latency_s ≤ 30 ✓

Ready to write Phase 2 plan (sensitive verification).
EOF
)"
```

- [ ] **Step 5: Stop the server**

```bash
pkill -f "uvicorn app.main" 2>/dev/null
echo "Phase 1 complete. Run /loop or invoke writing-plans for Phase 2."
```

---

## Phase boundary

**Phase 1 → Phase 2 handoff conditions:**

1. All 9 task commits landed on master
2. `eval_reports/phase1.json` shows acceptance criteria met
3. Milestone commit pushed

**Next plan:** After Phase 1 ships, invoke `superpowers:writing-plans` again with the spec, asking for Phase 2 (sensitive verification). Phase 2 builds directly on Phase 1's structured-JSON contract — there's no point starting it earlier.
