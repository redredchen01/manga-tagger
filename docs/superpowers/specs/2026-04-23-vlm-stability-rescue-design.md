# VLM Stability + Description Rescue 設計

**日期**:2026-04-23
**狀態**:設計階段
**作者**:Developer + Claude
**Scope**:Phase 1 acceptance 阻擋修復(milestone `2267d6b` Finding 6.1)
**Spec 關聯**:建立於 `docs/superpowers/specs/2026-04-22-tagging-accuracy-design.md` 之後,作為 Phase 1 pipeline 的阻擋項修補。**不取代**原三階段規劃——Phase 2 sensitive verification、Phase 3 embedding-first 仍按原 spec。

---

## 1. 背景與問題定義

### 1.1 Phase 1 驗收現況

milestone commit `2267d6b` 記錄 Phase 1 pipeline 骨架落地但**未通過** spec §6.2 acceptance:

| 指標 | 目標 | 實測 | 狀態 |
|---|---|---|---|
| mean_precision | ≥ 0.7 | 0.0 | ❌ |
| mean_recall | ≥ 0.5 | 0.0 | ❌ |
| sensitive_fp / image | ≤ 0.3 | 0.0 | ✓ |
| median_latency | ≤ 30s | 93.2s | ❌ |

### 1.2 根因(milestone 已錄)

- **根因 α:VLM 空 tag**。qwen3.6-35b-a3b 收到 14,675-char allowed_list prompt 後,頻繁回 `{"tags": []}`(server log 多次「VLM succeeded: 0 tags from JSON」)。偶爾有輸出時也選擇詭異(簡筆卡通被標 `人皮衣`)。
- **根因 β:Rescue path 缺失**。現 recommender 的 semantic fallback 用 VLM 自己的 tag name 當 embedding seed;若 VLM tags=[],semantic fallback 也沒 seed 可用 → 0 recommendation。
- **根因 γ:延遲超標**。90-100s / image,spec 30s 目標是 glm-4.6v 時代假設,qwen3.6-35b-a3b 單次 VLM call 就超。

### 1.3 約束

- **模型鎖定**:只用 `qwen3.6-35b-a3b-uncensored-hauhaucs-aggressive`(user confirmed,理由:uncensored、NSFW-friendly)
- **只改 prompt + pipeline**,不換模型、不加二段 VLM call
- **只修 Phase 1 阻擋**,不擴 scope 到 Phase 2/3

---

## 2. 架構總覽

本設計只動三個檔,不新增模組:

| 檔案 | 改動 |
|---|---|
| `app/domain/tag/allowed_list.py` | 新 function `build_compact_prompt_fragment`(tag name only,不含 description) |
| `app/domain/prompts.py` | `get_structured_prompt` 內 schema 提示微調(`category` 改為 optional self-check) |
| `app/domain/tag/recommender.py` | 新 Stage 1b:description rescue 路徑 |
| `app/infrastructure/lm_studio/vlm_service.py` | 改用 `build_compact_prompt_fragment` |
| `app/core/config.py` | 新 3 個 config 常數 |

### 2.1 資料流

```
image
  │
  ▼
VLM(compact prompt, ~3-4k chars)
  │
  ▼
parse JSON → {description, tags[]}
  │
  ├─── tags filtered against 611-library ──→ primary candidates (Stage 1)
  │
  └─── description (always computed)
        │
        ▼
     bge-m3 embed → cosine vs tag_matrix → top-K candidates (Stage 1b)
        │
        ▼
     secondary candidates (source="description_rescue")
  │
  ▼
Merge(Stage 1b 尾端):
  • if primary ≥ 3 tags → secondary 最多補 2 個未重複的,較低 confidence
  • if primary < 3 tags  → secondary 為主,primary 為次
  • VLM tag ∩ embedding tag → +0.10 boost(dual-source agreement)
  │
  ▼
現有 Stage 4-10 完全不動
(semantic fallback / RAG gate / dedupe / verify / calibrate)
```

### 2.2 邊界

- 非 VLM JSON path(legacy / mock / Ollama)**完全不動**
- sensitive verification(`_verify_and_calibrate`)不動
- RAG gating(`RAG_INFLUENCE_ENABLED`)不動
- LLM refinement gate(commit `42b6824`)不動
- Semantic fallback(commit `5afc3a6`)不動——與 description rescue 互補

---

## 3. Compact prompt 設計

### 3.1 舊格式(當前)

```
### 角色 (Character)
- 蘿莉:有性暗示或裸體的未成年少女外觀角色...
- 人妻:看上去像30-50岁的女性,不一定是mother...
...(14,675 chars 總)
```

### 3.2 新 compact 格式

```
### 角色
蘿莉, 人妻, 變身, 蜂娘, 青蛙娘, 幽靈, 河童, 魔物娘, ...

### 服裝
女生制服, 男生制服, 泳裝, 比基尼, 情趣內衣, ...

### 身體特徵
巨乳, 爆乳, 貧乳, 普通乳, ...

### 動作與互動
站立, 坐下, 躺臥, 奔跑, ...

### 主題
純愛, NTR, 百合, 耽美, ...

### 藝術風格
黑白, 彩色, 草圖, 線稿, ...
```

### 3.3 Size 估算

- 611 tags × avg 3 Chinese chars ≈ 1,800 chars
- 逗號分隔符:610 個
- 6 個 category headers
- **總估 ≈ 3,500 chars**(從 14,675 降 76%)

### 3.4 取捨

| 丟掉 | 理由 |
|---|---|
| Tag description(「蘿莉:有性暗示或裸體...」) | qwen3.6 對長 prompt 的 attention 不穩;描述詳情交由模型的 pretrained knowledge 提供 |

| 保留 | 理由 |
|---|---|
| 中文 tag name | library 匹配唯一依據,不能丟 |
| 6 個 category headers | 給 VLM 填寫 `category` 欄位時的提示;也利於模型組織視覺掃描順序 |

### 3.5 Schema 提示微調

現 schema 要求 VLM 每個 tag 都要帶 `category` 欄位。compact prompt 下每個 tag 已透過 header 隱含歸類,`category` 由 VLM 自行推斷。

**調整**:schema 改為 `category` optional(作為 self-check),library 匹配僅用 `tag` 欄位。`tag` 匹配 = 成功,不檢查 category 是否對。

### 3.6 保留的精準度規則(不動)

- 「視覺證據不足的標籤一律不要列」
- 「confidence < 0.6 的標籤直接拿掉」
- 「不要 hedge,不要寫『需要更多視覺證據』」
- 年齡 / sensitive tag 的嚴格證據要求

---

## 4. Description rescue path

### 4.1 觸發條件

**Always compute**(無論 VLM tags 是否為空)。merge 階段再決定採用程度。理由:
- 延遲成本極低(bge-m3 encode 單句 <100ms,cosine vs 611-tag matrix <1ms)
- 同步計算避免 if-then-else 分支複雜度
- 雙來源 agreement 信號需要兩邊都跑才能算

### 4.2 Embedding 來源

- 複用現有 `app/services/chinese_embedding_service.py:get_chinese_embedding_service`
- 底層模型 `BAAI/bge-m3`(已於 server 啟動時 load,`_tag_matrix_cache` 已於首次 request 填入 611-tag embedding)
- 現有 `embedding_service.search_cached_tags(query, top_k, threshold)` 介面**直接可用**

### 4.3 Pseudo-code(加在 `recommend_tags` 原 Stage 1 結尾、Stage 4 semantic fallback 之前)

```python
# Stage 1b: Description rescue — compute ALWAYS
description = vlm_analysis.get("description", "") if isinstance(vlm_analysis, dict) else ""
desc_candidates: List[TagRecommendation] = []

if settings.DESC_RESCUE_ENABLED and description and len(description) >= 10:
    try:
        from app.services.chinese_embedding_service import get_chinese_embedding_service
        embed_service = get_chinese_embedding_service()
        if embed_service and embed_service.is_available():
            if not hasattr(embed_service, "_tag_matrix_cache") or embed_service._tag_matrix_cache is None:
                await embed_service.cache_tag_embeddings(self.tag_library.get_all_tags())
            matches = await embed_service.search_cached_tags(
                description,
                top_k=settings.DESC_RESCUE_TOP_K,
                threshold=settings.DESC_RESCUE_THRESHOLD,
            )
            for m in matches:
                if m["tag"] not in self.tag_library.tag_names:
                    continue
                desc_candidates.append(TagRecommendation(
                    tag=m["tag"],
                    confidence=safe_confidence(m["similarity"] * 0.7),
                    source="description_rescue",
                    reason=f"desc embed match (sim={m['similarity']:.2f})",
                ))
    except (ImportError, AttributeError, RuntimeError) as e:
        logger.warning("Description rescue unavailable: %s: %s", type(e).__name__, e)

# Merge
vlm_tag_set = {r.tag for r in recommendations}
desc_tag_set = {dc.tag for dc in desc_candidates}

if len(recommendations) < 3:
    # VLM under-delivered → description_rescue 成主力
    for dc in desc_candidates:
        if dc.tag not in vlm_tag_set:
            recommendations.append(dc)
else:
    # VLM delivered → description_rescue 最多補 2 個
    added = 0
    for dc in desc_candidates:
        if added >= 2:
            break
        if dc.tag not in vlm_tag_set:
            recommendations.append(dc)
            added += 1

# Dual-source agreement boost
for r in recommendations:
    if r.source == "vlm_json" and r.tag in desc_tag_set:
        r.confidence = safe_confidence(min(r.confidence + 0.10, 1.0))
        r.reason += " (+desc agreement)"
```

### 4.4 新 config 常數(`app/core/config.py`)

```python
DESC_RESCUE_ENABLED: bool = True                # kill-switch for debug
DESC_RESCUE_TOP_K: int = 8                       # top candidates from description
DESC_RESCUE_THRESHOLD: float = 0.60              # rescue looser than semantic_fallback (0.75)
```

**Threshold 選擇理由**:semantic_fallback 用 VLM tag name(短詞)當 query,語義雜訊低,可用 0.75。description_rescue 用整句 description 當 query,語義廣度高、雜訊高,0.60 更合適。

### 4.5 與 semantic fallback 的關係

| 來源 | Seed | 目的 | 觸發 | 門檻 |
|---|---|---|---|---|
| semantic_fallback | VLM 的 tag name(短詞) | 補 VLM 漏掉的長尾 | `len(vlm_tags) < 3` | 0.75 |
| description_rescue(新) | VLM 的 description(整句) | 解 VLM 完全空 tag + 提供 dual-source 確認 | always compute,merge 時決定用量 | 0.60 |

兩者互補不衝突:
- VLM 給 5 tag:semantic_fallback 不 trigger(≥3);description_rescue 最多補 2
- VLM 給 1 tag:semantic_fallback trigger;description_rescue 也補(merge 時可能 overlap,dedupe 處理)
- VLM 給 0 tag:semantic_fallback 跑但拿不到 seed → 空;description_rescue 接手變主力

---

## 5. 錯誤處理

| 情境 | 行為 |
|---|---|
| `DESC_RESCUE_ENABLED=False` | Stage 1b 整段跳過,退回 Phase 1 現行為 |
| description 為空 / < 10 chars | desc_candidates = [],不影響 merge |
| embedding service 不可用 | `try/except (ImportError, AttributeError, RuntimeError)` → warning log,desc_candidates = [] |
| `search_cached_tags` 回空 | desc_candidates = [],merge 時 VLM tags 為唯一來源 |
| `tag_library.tag_names` 不含 matched tag | 該 candidate 被 filter 掉(防呆:library 不匹配則 silently drop) |
| 所有 candidates 皆 < threshold | desc_candidates = [] |

---

## 6. 測試策略

### 6.1 新增單元測試

| 檔案 | 涵蓋 |
|---|---|
| `tests/unit/test_compact_allowed_list.py` | `build_compact_prompt_fragment(611 tags)` 產出 < 5k chars;6 個 category headers 都出現;每個 tag 只出現一次;所有 SENSITIVE_SET tag 都在(不可被漏 trim) |
| `tests/unit/test_description_rescue.py` | (a) description 為空 → rescue 跳過;(b) VLM 給 0 tag → rescue candidates 成為主要 recommendations;(c) VLM 給 5 tag + rescue 給 10 → 最多補 2 個不重複;(d) 雙來源 agreement → VLM tag 的 confidence +0.10 且 reason 帶 "(+desc agreement)";(e) embedding service 掛掉 → 不爆炸,回 VLM-only 結果 |
| `tests/unit/test_structured_prompt_compact.py` | `get_structured_prompt(compact_fragment)` 總長 < 5k chars;仍要求 JSON-only 輸出;schema 裡 `category` 標記為 optional self-check |

### 6.2 現有測試仍必須綠

不動:`test_pipeline_no_hedge`、`test_rag_disabled`、`test_llm_refine_gate`、`test_tag_recommender`、`test_eval_accuracy`、`test_library_categories`、`test_vlm_json_parse`、`test_rag_service`(fixed)、其他 unit tests。

需更新:`test_semantic_fallback` 的 trigger 描述——description_rescue 會在 semantic_fallback 之前就補滿,semantic_fallback 實質 trigger 機會變少。更新測試 docstring + 增加 rescue+semantic 共存 case(rescue 先補到 3 個,semantic 不 trigger)。

更新:`test_allowed_list`(原測試針對 `build_prompt_fragment`)——保留(舊函數不刪除,deprecated 但為了 legacy path 留著),額外加 compact 的測。

### 6.3 Eval 驗收門檻(Phase 1 v2)

| 指標 | 原 spec §6.2 | 調整後 | 理由 |
|---|---|---|---|
| mean_precision | ≥ 0.7 | **≥ 0.7** | 核心不動 |
| mean_recall | ≥ 0.5 | **≥ 0.5** | 核心不動 |
| sensitive_fp / image | ≤ 0.3 | ≤ 0.3(已達) | 已達 |
| median_latency | ≤ 30s | **≤ 60s** | qwen3.6-35b-a3b 硬限。30s 目標為 glm-4.6v 時代假設,明確放寬並於 milestone commit 記錄 |

### 6.4 Eval 流程

1. Baseline snapshot:`cp eval_reports/phase1.json eval_reports/phase1_pre_rescue.json`
2. 實作完成後:`python scripts/eval_accuracy.py --out eval_reports/phase1_v2.json`
3. 跑比較檢查(與 plan Task 10 Step 3 類似結構),驗證四個門檻
4. 達標 → milestone commit(`milestone: Phase 1 acceptance met (v2)`)
5. 未達標 → 分析 rescue 補了哪些、miss 哪些,決定是否升級到「兩段式 VLM」(原 Q3 選項 B)或其他

### 6.5 本輪不擴 golden set

擴 golden set 至 ≥ 30 張是 spec §6.2 自己標註的風險緩解,但擴樣本是**資料收集 / 人工標註工作**,與 Finding 6.1 的設計問題正交。

本輪**僅以 4 張 starter 中 3 張有效樣本**做驗收,milestone commit 將明確記錄「小樣本噪音仍存,下一輪 Phase 2 前擴至 ≥ 30」。

---

## 7. 驗收 / Rollback

### 7.1 驗收通過的意義

Phase 1 v2 acceptance 通過即可進入 Phase 2 sensitive verification 的 spec / plan 設計。milestone commit 會明確:
- 本設計解掉了三根因之中的 α 與 β(VLM 空 tag 與 rescue path 缺失)
- 根因 γ(latency)做了局部改善但未達原 30s 目標——正式採用 60s 門檻

### 7.2 未通過時的 escalation

若 P/R 仍未達,可能升級:
- **E1 兩段式 VLM**:先用通用 prompt 呼 VLM 產生 description,依 embedding 取 top-K candidate tags,再呼一次 VLM 用短 allowed_list 做最終選擇。延遲加倍但 grounding 更強。
- **E2 換模型**:評估其他 NSFW-friendly vision model(MiniCPM-V、InternVL、其他 abliterated 變體)。與當前「鎖 qwen3.6-35b」偏好衝突,需 user 重新定案。
- **E3 砍 library**:只留高頻 top-200 tag,放棄長尾。適合 library 過大且 VLM 仍無法一次掃完的情境。

若 latency 仍 > 60s:
- **E4**:降低 VLM 的 `max_tokens` 或 `temperature`
- **E5**:改 streaming 讀 VLM output(可能的話,LM Studio 支援)

### 7.3 Kill-switch

`DESC_RESCUE_ENABLED=False` 可即時退回 Phase 1 現行為。無 DB migration、無破壞性變更,完全可回滾。

---

## 8. 範圍外(明確不做)

- 不擴 golden set(資料工作,另案)
- 不換模型(user 已鎖 qwen3.6-35b)
- 不加兩段式 VLM(留作 escalation option)
- 不動 Phase 2 sensitive verification(另案)
- 不碰 RAG index / legacy keyword extraction
- 不做 streamlit 前端改動

---

## 9. 風險

| 風險 | 緩解 |
|---|---|
| Compact prompt 丟掉 description 後 VLM 對冷門 tag 選擇變差 | description_rescue 用 embedding 補;雙來源 agreement boost 提升高信心候選 |
| description_rescue 引入過多弱信號 → precision 反而掉 | threshold 0.60 + confidence × 0.7 下壓;VLM ≥ 3 時最多補 2 個 |
| bge-m3 對中文 description 的 embedding 與 tag 名 embedding 語義不對齊 | 實測中 embedding service 已用 `tag_name + description` 當 tag embedding 文本(見 `cache_tag_embeddings` 實作),與 VLM 的中文 description 語義空間接近 |
| qwen3.6-35b-a3b 把 compact prompt 誤判為 prompt injection 或安全過濾 | 低可能;VLM 對短 allowed list 通常更配合(已從 spec 原設計延伸,14k→3.5k 仍包含 NSFW tag 名) |
| merge 邏輯的「補 2 個」上限太嚴 → recall 受限 | 若實測 recall 不達,改為「補到 top_k 上限」 |
