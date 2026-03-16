# 精準貼標優化方案

## 問題診斷

### 當前問題
- **噪音過大**：RAG 閾值 0.25 過低，導致不相關結果被推薦
- **閾值不統一**：各匹配階段閾值不一致，缺乏層次過濾
- **缺少驗證**：標籤缺乏二次驗證機制

### 根本原因分析
1. `RAG_SIMILARITY_THRESHOLD = 0.25` → 過低
2. `match_tags_by_keywords(min_confidence=0.4)` → 過低
3. `embedding_service.search_cached_tags(threshold=0.3)` → 過低
4. 缺乏標籤間一致性檢查

---

## 優化方案

### 1. 提高閾值（核心改動）

| 參數 | 當前值 | 建議值 | 原因 |
|------|--------|--------|------|
| `RAG_SIMILARITY_THRESHOLD` | 0.25 | **0.5** | 過濾低質量 RAG 匹配 |
| `HYBRID_SCORING_ALPHA` | 0.6 | **0.7** | 增加詞法匹配權重 |
| `CHINESE_EMBEDDING_THRESHOLD` | 0.4 | **0.55** | 提高語義匹配精度 |

### 2. 標籤驗證層

```python
# 新增 TagValidator 類
class TagValidator:
    """標籤驗證器 - 確保標籤與圖片內容相關"""
    
    async def validate_tag_with_vlm(self, image_bytes, tag) -> bool:
        """使用 VLM 驗證標籤是否適用於圖片"""
        prompt = f"Does this image contain '{tag}'? Answer YES or NO."
        # ...
    
    def check_tag_conflict(self, tags: List[str]) -> List[str]:
        """檢查標籤衝突 - 例如同時有 '蘿莉' 和 '熟女'"""
        # ...
```

### 3. 敏感標籤過濾

```python
# 增強敏感標籤處理
SENSITIVE_TAGS_REQUIRING_VERIFICATION = {
    "loli": {"threshold": 0.9, "alternate_tags": ["少女", " teen"]},
    "anal": {"threshold": 0.9, "alternate_tags": ["正常位"]},
    "肛交": {"threshold": 0.9, "alternate_tags": []},
}
```

### 4. 改進 VLM 提示詞

```python
def _get_precision_prompt(self) -> str:
    """精準模式提示詞"""
    return """Analyze this image and ONLY output tags that are VISIBLE.
    
RULES:
1. If you CANNOT see it, do NOT include it
2. Be conservative - when in doubt, exclude
3. Only include adult tags if EXPLICITLY visible

Output format: tag1, tag2, tag3 (comma separated)
NO explanations. NO extra text."""
```

### 5. 標籤一致性檢查

```python
# 衝突標籤規則
TAG_CONFLICTS = {
    "蘿莉": ["熟女", "人妻", "御姐"],
    "正太": ["人妻", "熟女"],
    "肛交": [],
    "巨乳": ["貧乳", "貧乳"],
}

def remove_conflicting_tags(self, tags: List[str]) -> List[str]:
    """移除衝突標籤，保留置信度較高的"""
    # ...
```

---

## 實施順序

### Phase 1: 閾值調整（低風險）
1. 更新 `app/config.py` 中的閾值參數
2. 更新 `tag_recommender_service.py` 中的內部閾值
3. 測試基本功能

### Phase 2: 驗證層（中風險）
1. 新增 `TagValidator` 類
2. 整合到 `recommend_tags` 流程
3. 添加敏感標籤二次驗證

### Phase 3: 提示詞優化（中風險）
1. 改進 VLM 提示詞
2. 測試不同提示詞效果
3. 選擇最優版本

### Phase 4: 一致性檢查（低風險）
1. 定義衝突標籤規則
2. 實現一致性檢查邏輯
3. 整合到最終輸出

---

## 預期效果

| 指標 | 當前 | 目標 |
|------|------|------|
| 精確率 (Precision) | ~60% | **85%+** |
| 噪音率 | ~40% | **<15%** |
| 標籤數量/圖片 | 5-10 | **3-7** |

---

## 風險評估

| 風險 | 等級 | 緩解措施 |
|------|------|----------|
| 過濾過多有效標籤 | 中 | 設置最低標籤數量保障 |
| 敏感標籤誤判 | 中 | 二次驗證機制 |
| 性能下降 | 低 | 異步處理驗證 |

---

## 測試計劃

1. **基準測試**：記錄當前精確率
2. **閾值測試**：逐步調整閾值，記錄變化
3. **端到端測試**：使用標準測試集驗證
4. **A/B 測試**：比較不同提示詞效果
