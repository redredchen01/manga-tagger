"""Categorize every tag in 51標籤庫.json into one of:
  character | clothing | body | action | theme | style | other

Strategy (offline, deterministic, no network):
  1. Explicit SENSITIVE_OVERRIDES dict for tags where heuristic ordering fails
     (e.g. "肛交" → action, not body; "蘿莉" → character, not body).
     Overrides force-reassign even over prior (possibly wrong) categories.
  2. Skip entries already carrying a valid category.
  3. Keyword-heuristic match.
  4. Default to "other".

The `llm_classify` function is retained for future use but is not called from
the main loop — a prior attempt with LM Studio took 30+ minutes for marginal
benefit and surfaced ordering bugs that are better fixed with explicit overrides.

Idempotent: re-running leaves explicitly-categorized entries unchanged except
where SENSITIVE_OVERRIDES corrects them.
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
LM_TEXT_MODEL = os.environ.get("LM_STUDIO_TEXT_MODEL", "qwen3.6-35b-a3b-uncensored-hauhaucs-aggressive")
LLM_TIMEOUT_S = 15.0
BATCH_SAVE_EVERY = 50

CATEGORIES = ["character", "clothing", "body", "action", "theme", "style", "other"]

# Sensitive tags get explicit category to bypass heuristic ordering issues
# (e.g. "肛交" should be 'action' but contains "肛" which matches 'body' marker first)
SENSITIVE_OVERRIDES = {
    # CSAM-equivalent → character
    "蘿莉": "character", "正太": "character", "嬰兒": "character", "兒童": "character",
    "少女": "character", "少年": "character",
    # Sex acts → action
    "肛交": "action", "口交": "action", "乳交": "action", "手淫": "action",
    "自慰": "action", "強姦": "action", "強制": "action", "輪姦": "action",
    "中出": "action", "顏射": "action", "射精": "action", "獸交": "action",
    "足交": "action", "綁縛": "action", "調教": "action", "凌虐": "action",
    # Themes/genres → theme
    "亂倫": "theme", "獵奇": "theme", "綠帽": "theme", "NTR": "theme",
    "戀足": "theme", "偷窺": "theme", "露出": "theme",
    "觸手": "theme", "食人": "theme",
    # Body parts mistaken-as-action by heuristic
    "斷肢": "body",
}

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
    prompt = (
        f"分類以下漫畫標籤到 7 類之一：\n"
        f"character / clothing / body / action / theme / style / other\n\n"
        f"標籤名稱：{name}\n描述：{description}\n\n"
        f"只輸出類別名稱，不要任何其他文字。"
    )
    try:
        with httpx.Client(timeout=LLM_TIMEOUT_S) as client:
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
        print(f"  LLM classify failed for {name!r}: {type(e).__name__}: {e}", file=sys.stderr)
        return "other"


def save_library(data: list) -> None:
    LIBRARY.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    data = json.loads(LIBRARY.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        print(f"Expected JSON list at top level, got {type(data).__name__}", file=sys.stderr)
        sys.exit(1)

    total = len(data)
    counts = {c: 0 for c in CATEGORIES}
    pass_counts = {"already": 0, "override": 0, "heuristic": 0, "default_other": 0, "llm": 0}

    # Single-pass: overrides → already-categorized → heuristics → default "other".
    # Overrides take precedence over prior (possibly wrong) categorizations
    # from partial runs, because the heuristic had ordering bugs.
    print(f"=== Categorizing {total} entries (offline, no network) ===")
    for entry in data:
        name = entry.get("tag_name", "")
        desc = entry.get("description", "")

        # 1. Explicit sensitive-tag override (exact tag_name match) — forces reassignment
        if name in SENSITIVE_OVERRIDES:
            cat = SENSITIVE_OVERRIDES[name]
            entry["category"] = cat
            pass_counts["override"] += 1
            counts[cat] += 1
            continue

        # 2. Skip if already categorized by a prior run
        if entry.get("category") in CATEGORIES:
            pass_counts["already"] += 1
            counts[entry["category"]] += 1
            continue

        # 3. Heuristic keyword match
        cat = heuristic_classify(name, desc)
        if cat:
            entry["category"] = cat
            pass_counts["heuristic"] += 1
            counts[cat] += 1
            continue

        # 4. Default to "other"
        entry["category"] = "other"
        pass_counts["default_other"] += 1
        counts["other"] += 1

    save_library(data)

    print("\n=== Category counts ===")
    for c in CATEGORIES:
        print(f"  {c:12s}: {counts[c]}")
    print("\n=== Source counts ===")
    for k, v in pass_counts.items():
        print(f"  {k:14s}: {v}")


if __name__ == "__main__":
    main()
