# 標籤精準度提升設計

**日期**：2026-04-22
**狀態**：設計階段
**作者**：Developer + Claude
**Use case**：NSFW 中文標籤系統（精準打標、不執行阻擋動作）

---

## 1. 背景與問題定義

### 1.1 目前 pipeline

`POST /api/v1/tag-cover` 經過：

1. **VLM**（Ollama llava 7B）→ 自然語言中文描述
2. **Parser** 從描述中爬 keyword（`extract_tags_from_description`）
3. **Library matching** keyword → tag（fuzzy）
4. **RAG search** image → similar images → 它們的 tags
5. **Semantic match** keyword → bge-m3 embedding → tag
6. 多源融合（library_match + rag + semantic_match）

### 1.2 觀察到的精準度問題

實測 `test_anime.jpg` 結果範例：

| 標籤 | 來源 | 問題 |
|---|---|---|
| `卡在牆上` | semantic_match | 來自 VLM 失敗訊息「failed」誤判為 keyword |
| `多對多` | semantic_match | 來自「using」誤判 |
| `人皮衣` | rag | RAG 17 筆庫的 1 筆 anime 蓋過所有判斷 |
| `需要更多視覺證據` | parser | VLM 的 hedge 字串被當 keyword |

歸納七類根因：

1. **VLM 模型能力不足**：llava 7B 在中文 + 細節辨識上明顯弱，常 hedge
2. **Prompt 與 library 不對齊**：prompt 內聯 ~120 tag、library 有 611，model 不知完整集合
3. **描述爬字路徑不可控**：hedge 字串會直接污染 final tags
4. **RAG 庫過小**：17 筆且偏 anime/catgirl，對任何動漫圖回傳同一組
5. **Semantic match 門檻過低**：0.55 撈一堆弱相似 tag
6. **無二階驗證**：`verify_sensitive_tag()` 已寫但 pipeline 沒呼叫
7. **Library categorization 失效**：557/611 在 "other" 類

各根因對應到的 phase：

| 根因 | 解法所在 phase |
|---|---|
| 1. VLM 能力不足 | Phase 1 §3.1 |
| 2. Prompt/library 不對齊 | Phase 1 §3.3 |
| 3. 描述爬字污染 | Phase 1 §3.2、§3.4 |
| 4. RAG 庫過小 | Phase 1 §3.6 |
| 5. Semantic 門檻過低 | Phase 1 §3.5 |
| 6. 無二階驗證 | Phase 2 §4 |
| 7. Library categorization | Phase 1 §3.8（同時造福 Phase 3 §5.2）|

### 1.3 Use case 與優先序

- **目標**：精準的中文 NSFW 標籤（含 sensitive 類別）
- **動作**：純打標，**不**自動阻擋。分類動作由人或別的系統處理
- **圖像來源**：漫畫封面、內頁、單張插畫（全包）
- **痛點優先序**：FP（誤標）≈ FN（漏標）= 顆粒度 = 各類均衡 > 排序

---

## 2. 整體 Pipeline 架構（三 phase 全完成後）

```
                ┌─────────────────────────────────────┐
                │    image upload (FastAPI)           │
                └───────────────┬─────────────────────┘
                                │
                                ▼
                ┌─────────────────────────────────────┐
                │  validate_image (size/format)        │
                └───────────────┬─────────────────────┘
                                │
                                ▼
                ┌─────────────────────────────────────┐
                │  Stage A: VLM (LM Studio glm-4.6v)   │
                │  Output: structured JSON             │
                │   {description, tags[]}              │
                └────────┬────────────────────────────┘
                         │
                         ├─► JSON tags ──────────────────────┐
                         │                                    │
                         ▼                                    │
              ┌──────────────────────────────┐                │
              │ Stage A.5 (Phase 3):         │                │
              │ embed(description)           │                │
              │ → cosine vs tag_matrix       │                │
              │ → top-K candidate tags       │                │
              └──────────┬───────────────────┘                │
                         │                                    │
                         ▼                                    ▼
                ┌─────────────────────────────────────────────────┐
                │  Stage B: Candidate merge                        │
                │  • VLM JSON tags         (conf × 1.0)            │
                │  • Embedding top-K        (conf × 0.85)          │
                │  • RAG (score ≥ 0.95 only) (conf × 0.6)          │
                │  • Semantic (only if VLM tags < 3)               │
                │  • Two-source agreement → +0.15 boost            │
                └────────────────────────┬────────────────────────┘
                                         │
                                         ▼
                ┌─────────────────────────────────────────────────┐
                │  Stage C: Sensitive verification (Phase 2)       │
                │  for tag in candidates:                          │
                │      if tag in SENSITIVE_SET:                    │
                │          run verify(image, tag) twice            │
                │          (temp=0.1 and 0.3)                      │
                │          keep only if 2/2 YES                    │
                │      else: pass through                          │
                └────────────────────────┬────────────────────────┘
                                         │
                                         ▼
                ┌─────────────────────────────────────────────────┐
                │  Stage D: Final ranking + audit log              │
                │  return top-K + per-tag evidence trail           │
                └─────────────────────────────────────────────────┘
```

**核心原則**：
1. **VLM 是主來源**，其他都是輔證或 fallback
2. **Sensitive 標籤一律雙驗**，非 sensitive 走快路徑
3. **每個 final tag 都有 evidence trail**，存到 metadata 供 debug

---

## 3. 階段 1：止血 + 換 VLM（1–2 天）

### 3.1 切換 VLM 後端到 LM Studio glm-4.6v-flash

- `.env`：`USE_OLLAMA=false`、`USE_LM_STUDIO=true`、`LM_STUDIO_VISION_MODEL=zai-org/glm-4.6v-flash`
- 啟動時實測 LM Studio 連通性，連不通 fail-fast（不要 fallback 到 mock）
- Ollama 路徑保留作 dev，但生產預設 LM Studio

### 3.2 強制 VLM 輸出結構化 JSON

新 prompt（取代 `app/domain/prompts.py:get_optimized_prompt`）：

```
你是漫畫圖像標籤系統。輸出**僅限**符合 JSON Schema 的結果。

允許的標籤（只能從以下選，不要創造）：
{ALLOWED_TAGS_BY_CATEGORY}   ← 由 library 動態注入

輸出格式（嚴格 JSON，不可有任何其他文字）：
{
  "description": "2-3 句中文描述",
  "tags": [
    {
      "tag": "<必須在允許列表>",
      "category": "<character|clothing|body|action|theme|style>",
      "confidence": 0.0-1.0,
      "evidence": "<簡短視覺證據>"
    }
  ]
}

精準度規則：
- 視覺證據不足的標籤一律不要列
- confidence < 0.6 的標籤直接拿掉
- 不確定就不標，不要 hedge
```

**Parse 策略**：
1. `json.loads` 直接 parse
2. 失敗則 strip markdown fence (` ```json ... ``` `) 再試
3. 再失敗 → 重試一次（temperature 0.0）
4. 仍失敗 → 回傳 `{"tags": [], "error": "VLM_PARSE_FAIL"}`，**不 fallback 去爬描述**

### 3.3 動態 allowed-tag list 注入

啟動時把 `51標籤庫.json` 按 6 類 group，**所有 611 個標籤全注入** prompt。glm-4.6v context window 128K 撐得住（estimated 5k–10k tokens）。

每個 tag 行格式：`<tag_name> — <short description>`

### 3.4 砍掉「描述 → keyword」路徑

- 從 `app/domain/pipeline.py` 與 `app/domain/tag/recommender.py` 移除 `extract_tags_from_description` 在主路徑的呼叫
- VLM JSON 給什麼 tag 就用什麼 tag
- `extract_tags_from_description` 函數本身保留，但只用於 mock/test fallback

### 3.5 Semantic match 改 fallback

- 觸發條件：`len(vlm_tags) < 3`
- 門檻：0.75（從 0.55 提升）
- 上限：每張圖最多補 2 個 semantic tag
- Source 標記：`semantic_fallback`（與一般 `semantic_match` 區分）

### 3.6 RAG 來源處理

- Phase 1：**直接停用 RAG 對打分的影響**（17 筆庫不可信）
- `rag/add` 端點與 `rag_matches` metadata 保留供日後擴庫
- 等庫長到 ≥ 500 筆再開回來，門檻 score ≥ 0.95，且 sensitive 類別永遠不採用 RAG 來源

### 3.7 多源融合改有優先級

```python
def merge_candidates(vlm_tags, rag_tags, semantic_tags):
    final = {}
    for t in vlm_tags:
        final[t.tag] = (t.confidence, "vlm", t.evidence)

    for t in rag_tags:
        if t.score < 0.95: continue
        if t.tag in SENSITIVE_SET: continue   # never trust RAG for sensitive
        if t.tag not in final:
            final[t.tag] = (t.score * 0.6, "rag", "RAG match")

    if len(final) < 3:   # only if VLM under-delivered
        for t in semantic_tags:
            if t.similarity < 0.75: continue
            if t.tag not in final:
                final[t.tag] = (t.similarity * 0.5, "semantic_fallback", t.keyword)
            if len(final) >= 5: break

    return sorted(final.items(), key=lambda x: -x[1][0])
```

### 3.8 Library categorization 修復

一次性離線腳本 `scripts/fix_tag_categories.py`：

1. **啟發式分類**先過一遍（含「乳/胸/陰/肛」→ body；含「裝/服/衣」→ clothing；etc.）
2. 沒被啟發式分類的 → 餵 LLM（LM Studio glm-4.7-flash 文字模型）自動分類
3. 結果寫進 `51標籤庫.json` 新欄位 `category`
4. 不卡在 100% 正確上——sensitive 相關 tag 必須對，其他可後續修

### 3.9 階段 1 預期效果

- Hedge 字串污染、低門檻 semantic 雜訊、RAG 1 筆蓋全部 → 全部消失
- VLM 升級 + 結構化輸出 → 標籤質量大幅提升
- 改動範圍：3–5 個檔案，~300 行

---

## 4. 階段 2：二階驗證（3–5 天）

### 4.1 SENSITIVE_SET 定義

放在 `app/domain/tag/constants.py` 作 single source of truth：

```python
SENSITIVE_SET = {
    # CSAM 等級
    "蘿莉", "正太", "嬰兒", "兒童",
    # 非自願 / 重度
    "強制", "強姦", "亂倫", "獵奇", "食人", "斷肢",
    # 特定癖好
    "肛交", "觸手", "調教", "綁縛", "輪姦", "獸交",
    "中出", "顏射", "射精", "口交", "乳交", "手淫",
    "自慰", "足交", "戀足", "偷窺", "露出",
    "綠帽", "NTR",
    # 角色年齡灰色（補強）
    "少女", "少年", "偽娘",
}
```

### 4.2 二階驗證流程

```python
async def verify_tag_with_consensus(image_bytes, tag, vlm_service):
    if tag not in SENSITIVE_SET:
        return True, "non-sensitive, no verification"

    # 雙採樣 + 2/2 共識
    result_a = await vlm_service.verify_sensitive_tag(image_bytes, tag, temperature=0.1)
    result_b = await vlm_service.verify_sensitive_tag(image_bytes, tag, temperature=0.3)

    if result_a and result_b:
        return True, "2/2 agreement"
    elif result_a or result_b:
        return False, "split decision (1/2) - rejected"
    else:
        return False, "both rejected"
```

**為什麼 2/2**：單次 YES/NO 在邊界 case 上 ~5% flip rate；兩次獨立採樣同意後降至 <0.5%。

### 4.3 修改 verify_sensitive_tag 簽名與 prompt

- 簽名加 `temperature: float = 0.1` 參數
- Prompt 改成 chain-of-thought 強制模型先列證據再給 YES/NO：

```
Examine this image and answer about: {tag}

Step 1: List all visual evidence relevant to "{tag}":
Step 2: Is the evidence sufficient to confirm "{tag}"? Answer YES or NO.
Step 3: Confidence (HIGH/MEDIUM/LOW)?

Output strictly:
EVIDENCE: <bullet list>
ANSWER: <YES|NO>
CONFIDENCE: <HIGH|MEDIUM|LOW>
```

只有 `ANSWER=YES` **且** `CONFIDENCE != LOW` 才視為通過。

### 4.4 Pipeline 整合點

在 `run_tagging_pipeline` 的 stage 3（recommendation）之後、return 之前插入 stage 3.5：

```python
# Stage 3.5: Sensitive verification
import asyncio

semaphore = asyncio.Semaphore(MAX_CONCURRENT_VERIFICATION)

async def _verify(rec):
    async with semaphore:
        return await verify_tag_with_consensus(image_bytes, rec.tag, vlm_service)

results = await asyncio.gather(*[_verify(r) for r in recommendations])

verified = []
audit = []
for rec, (ok, reason) in zip(recommendations, results):
    if ok:
        verified.append(rec)
    audit.append({"tag": rec.tag, "verified": ok, "reason": reason})

result.metadata["sensitive_verification"] = audit
```

### 4.5 並行優化

- 用 `asyncio.gather` + `asyncio.Semaphore` 限制同時驗證數
- `MAX_CONCURRENT_VERIFICATION = 4` 上限（避免打爆 LM Studio）
- 5 個 sensitive 候選 = 10 次驗證（每個雙採樣），並行下 ~20 秒可完成

### 4.6 Audit log 結構

```json
{
  "sensitive_verification": [
    {
      "tag": "蘿莉",
      "verified": false,
      "reason": "split decision (1/2) - rejected",
      "evidence_a": "...",
      "evidence_b": "..."
    },
    {
      "tag": "貓娘",
      "verified": true,
      "reason": "non-sensitive, no verification"
    }
  ]
}
```

### 4.7 階段 2 預期效果

- Sensitive 類別 FP 從 ~10% 降到 <1%
- 每張圖延遲增加 0–25 秒（取決於 sensitive 候選數）
- 完整 evidence trail，事後可追溯

---

## 5. 階段 3：Embedding-first（1–2 週）

### 5.1 動機

到 phase 2 結束，剩下的精度瓶頸是 **VLM 自身的 recall**——VLM 沒列的 tag，再怎麼驗證也救不回來。

Embedding-first 路徑解決：用 bge-m3 把 VLM 描述 embed，跟 library 全 611 tag embedding 做向量檢索，**獨立於 VLM 的 tag list**，等於多開一條候選來源。

**不取代 VLM JSON tags**，而是**第二條候選通道**，最後一起進二階驗證。

### 5.2 離線預備：tag embedding 索引

新腳本 `scripts/build_tag_index.py`：

```python
for tag in library.tags:
    repr = f"{tag.name}：{tag.description}"
    if tag.aliases:
        repr += "（" + "、".join(tag.aliases) + "）"
    embeddings[tag.name] = bge_m3.encode(repr)

np.save("data/tag_index/embeddings.npy", matrix)
json.dump(tag_names, "data/tag_index/names.json")
```

輸出：
- `data/tag_index/embeddings.npy` — shape `(611, 1024)`
- `data/tag_index/names.json` — 對應 tag 名

服務啟動時 mmap 進記憶體常駐。

**前置處理（library description 補強）**：
- 很多 library tag 沒 description 或太短
- 一次性離線跑 LLM 補強（生成 1-2 句中文定義），存回 library JSON
- 同時順便完成 phase 1.8 的 categorization metadata

### 5.3 Pipeline 變動：新增 Stage A.5

VLM 出來後，分兩條路徑：
1. JSON tags → 直接進 Stage B
2. description → bge-m3 embed → cosine vs tag_matrix → top-K（similarity ≥ 0.6）→ 進 Stage B

### 5.4 兩來源融合的加分機制

Phase 3 擴充 §3.7 的 `merge_candidates`，把 embedding 加為第三條候選來源（加在 vlm 之後、rag 之前）：

```python
def merge_candidates_v2(vlm_tags, embed_tags, rag_tags, semantic_tags):
    final = {}

    # 1. VLM 主來源
    for t in vlm_tags:
        final[t.tag] = {"conf": t.confidence, "sources": ["vlm"], "evidence": t.evidence}

    # 2. Embedding 來源（agreement 加分）
    for t in embed_tags:
        if t.similarity < 0.6: continue
        if t.tag in final:
            # agreement → boost
            final[t.tag]["conf"] = min(1.0, final[t.tag]["conf"] + 0.15)
            final[t.tag]["sources"].append("embedding")
            final[t.tag]["evidence"] += f"; embedding match (sim={t.similarity:.2f})"
        else:
            # embedding-only → 較低權重
            final[t.tag] = {
                "conf": t.similarity * 0.85,
                "sources": ["embedding"],
                "evidence": f"embedding-only match (sim={t.similarity:.2f})"
            }

    # 3. RAG 與 semantic_fallback 沿用 §3.7 邏輯（不變）
    # ...

    return final
```

### 5.5 性能

- 1 次 description embed：CPU ~50ms / GPU <10ms
- 1 次 matrix dot product (611×1024)：<1ms
- top-K 排序：µs 級
- **總共增加延遲 < 100ms**，可忽略

### 5.6 失敗處理

- `tag_index` 不存在 → 啟動 fail-fast，提示跑 `scripts/build_tag_index.py`
- bge-m3 暫時不可用 → 跳過 Stage A.5，純走 VLM JSON 路徑（degrade gracefully）
- Library 變動 → 提供 `scripts/update_tag_index.py` 增量更新

### 5.7 可解釋性

每個 final tag 在 metadata：

```json
{
  "tag": "雙馬尾",
  "confidence": 0.92,
  "sources": ["vlm", "embedding"],
  "evidence": "VLM observed: 雙馬尾髮型清晰可見; embedding match (sim=0.78)",
  "verified": true,
  "verification_reason": "non-sensitive, no verification"
}
```

### 5.8 階段 3 預期效果

- 補強 VLM 的 recall
- 兩來源 agreement → 高信心；單來源 → 標明為弱信號
- 新增 tag 不用改 prompt
- 額外延遲 < 100ms

---

## 6. 測試與評量策略

### 6.1 Golden test set

`tests/golden/`：

```
tests/golden/
├── images/                          # 30–50 張代表性圖
│   ├── 001_school_uniform.jpg
│   ├── 002_loli_borderline.jpg     # 邊緣 case：非蘿莉但偏童顏
│   ├── 003_loli_clear.jpg
│   ├── 004_tentacles_clear.jpg
│   ├── 005_tentacles_decoration.jpg # 邊緣：背景觸手裝飾無性行為
│   └── ...
└── expected.json
```

`expected.json` 結構：

```json
{
  "001_school_uniform.jpg": {
    "must_have": ["女生制服", "少女"],
    "must_not_have": ["蘿莉", "強制", "肛交"],
    "nice_to_have": ["雙馬尾", "微笑"]
  }
}
```

**Starter set**：用 repo 已有的 4 張圖（test_anime.jpg, test_anime_detailed.jpg, test_image.jpg, test_real_image.jpg），由 Developer 後續擴充至 30–50 張。

**取樣原則**：
- 1/3 純 SFW（驗證系統不會亂噴 sensitive 標籤）
- 1/3 明確 NSFW（驗證 sensitive 類別準確抓到）
- 1/3 邊緣 case（最重要）

### 6.2 評量指標與驗收門檻

```python
precision = TP / (TP + FP)
recall    = TP / (TP + FN)
f1        = 2*P*R / (P+R)

sensitive_precision = SP_TP / (SP_TP + SP_FP)
sensitive_fp_rate   = SP_FP / total_images
```

| 指標 | Baseline (現況) | Phase 1 後 | Phase 2 後 | Phase 3 後 |
|---|---|---|---|---|
| Overall Precision | 待測 | ≥ 0.7 | ≥ 0.75 | ≥ 0.8 |
| Overall Recall | 待測 | ≥ 0.5 | ≥ 0.55 | ≥ 0.7 |
| **Sensitive Precision** | 待測 | ≥ 0.85 | **≥ 0.97** | ≥ 0.97 |
| **Sensitive FP / image** | 待測 | ≤ 0.3 | **≤ 0.05** | ≤ 0.05 |
| 中位延遲 | ~45s | ≤ 30s | ≤ 50s | ≤ 50s |

> Phase 1 開始前必須先量真實 baseline。Starter set 4 張圖只能作 smoke test，正式驗收前須擴到 ≥ 30 張，否則 metric 統計噪音過大。

### 6.3 評量腳本

`scripts/eval_accuracy.py`：

```python
async def run_eval():
    results = []
    for img_path, expected in golden_set:
        actual = await tag_via_api(img_path)
        metrics = compute_metrics(actual, expected)
        results.append({"image": img_path, **metrics})

    print_summary(results)
    write_report("eval_report.json", results)
```

每次改完 pipeline 跑一次，自動比對指標。

### 6.4 Regression 單元測試

| Phase | 測試重點 |
|---|---|
| 1 | JSON parse 失敗時不會 fallback 去爬描述；hedge 字串不會出現在 final tags；semantic 只在 VLM 候選 < 3 時觸發；RAG 不影響打分 |
| 2 | Sensitive tag 一定走 verification；非 sensitive tag 不會 trigger（避免延遲）；split decision 一定 reject |
| 3 | Embedding 索引缺失時 fail-fast；兩來源 agreement 確實 boost；單來源 confidence 不變 |

### 6.5 監控

打進現有 Prometheus metrics：
- `tagger_sensitive_fp_total` — 累計 false sensitive
- `tagger_verification_split_total` — 累計 1/2 split decision
- `tagger_phase_latency_seconds{phase=vlm|verify|embed}` — 各階段延遲

split rate 突然飆高 = 模型不穩 / prompt 出問題 = 警報。

---

## 7. 實作順序與里程碑

| Phase | 工期 | 完成標準 |
|---|---|---|
| 0：建 baseline | 0.5 天 | golden set starter (4 張) + eval 腳本可跑、輸出當前 metrics |
| 1：止血 + 換 VLM | 1–2 天 | 6.2 表「Phase 1 後」門檻全達標 |
| 2：二階驗證 | 3–5 天 | 6.2 表「Phase 2 後」門檻全達標，sensitive 類達標尤為關鍵 |
| 3：Embedding-first | 1–2 週 | 6.2 表「Phase 3 後」門檻全達標 |

每 phase 完成後：
1. 跑 `scripts/eval_accuracy.py` 出報告
2. 沒達標 → 不進下一 phase
3. 達標 → commit + 開下一 phase 分支

---

## 8. 範圍外（明確不做）

- 不換 embedding 模型（bge-m3 夠用）
- 不換 LLM 文字合成模型（pipeline 不靠 LLM 合成）
- 不做 GUI 標註工具（golden set 用 JSON 手填）
- 不做 active learning / 模型 fine-tune
- 不擴 RAG 庫（前提是 RAG 在 phase 1 已停用，未來使用者自願擴）
- 不改 streamlit 前端（後端 API 維持 backward compatible）
- 不做自動阻擋動作（use case 已明確排除）

---

## 9. 風險與假設

| 風險 | 緩解 |
|---|---|
| glm-4.6v 在 sensitive 類別判斷可能保守 / 拒答 | Phase 2 chain-of-thought prompt 可導模型先列證據；若仍失敗，記錄到 audit log 後人工調整 prompt |
| 雙採樣帶來延遲倍增 | 並行化 + 上限 4；可接受 |
| Golden set 太小 → metric 變動大 | starter 4 張只供 smoke test；正式驗收前須擴到 ≥ 30 張 |
| Library description 補強 LLM 自動產生品質不一 | 一次性離線跑、人工抽查；不卡進 phase 1 |
| 拆 RAG 後若使用者擴庫想開回來 | 留 config flag `RAG_INFLUENCE_ENABLED=false`，未來改 true 即可 |
