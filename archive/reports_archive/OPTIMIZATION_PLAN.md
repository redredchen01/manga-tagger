# 視覺識別標籤系統優化方案

## 執行摘要

本方案針對漫畫封面自動標籤系統的標籤匹配問題，提出一套完整的優化策略。核心問題是**VLM 生成的描述與標籤庫無法有效匹配**，導致標籤與圖片內容不符。

---

## 一、問題診斷分析

### 1.1 發現的核心問題

根據代碼庫分析，我識別出以下關鍵問題：

#### 🔴 **問題 1：VLM 提示詞設計不當**
- **症狀**：VLM 生成的描述過於通用，無法準確識別具體標籤
- **證據**：`glm4v_client.py` 中的提示詞要求描述圖片內容，但沒有明確要求從標籤庫中選擇
- **影響**：生成的描述與標籤庫的語義空間不匹配

#### 🔴 **問題 2：標籤匹配流程斷裂**
- **症狀**：VLM 輸出 → 關鍵詞提取 → 標籤匹配的鏈條存在斷層
- **證據**：`tag_recommender_service.py` 中註釋顯示「Semantic search matching is disabled」
- **影響**：依賴簡單的字符串匹配，無法處理語義相似性

#### 🔴 **問題 3：中英文映射不完整**
- **症狀**：VLM 傾向輸出英文標籤，但標籤庫是中文
- **證據**：`tag_mapper.py` 只有約 100 個映射，標籤庫有 600+ 標籤
- **影響**：大量標籤無法正確映射，導致匹配失敗

#### 🔴 **問題 4：相似度閾值設置不當**
- **症狀**：閾值過高導致遺漏相關標籤，過低導致噪聲標籤
- **證據**：`FINAL_FIX_REPORT.json` 顯示閾值從 0.7 降至 0.25 才有結果
- **影響**：無法平衡精確率和召回率

#### 🔴 **問題 5：缺乏標籤語境理解**
- **症狀**：系統無法理解標籤之間的邏輯關係（如互斥、包含）
- **證據**：雖有 `conflict_learning_system.py`，但未在主要流程中啟用
- **影響**：可能輸出矛盾標籤（如同時標記「蘿莉」和「人妻」）

---

## 二、優化方案總覽

### 2.1 整體架構調整

```
┌─────────────────────────────────────────────────────────────┐
│                    優化後的標籤系統架構                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │   圖片輸入    │───▶│  VLM 分析    │───▶│ 結構化特徵   │  │
│  └──────────────┘    └──────────────┘    └──────────────┘  │
│         │                   │                     │        │
│         ▼                   ▼                     ▼        │
│  ┌─────────────────────────────────────────────────────┐  │
│  │              多階段標籤匹配引擎                      │  │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐          │  │
│  │  │ 精確匹配  │─▶│ 語義匹配  │─▶│ RAG 檢索  │          │  │
│  │  └──────────┘  └──────────┘  └──────────┘          │  │
│  └─────────────────────────────────────────────────────┘  │
│                              │                             │
│                              ▼                             │
│  ┌─────────────────────────────────────────────────────┐  │
│  │              標籤衝突解決與驗證                      │  │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐          │  │
│  │  │ 衝突檢測  │─▶│ 邏輯推理  │─▶│ 最終篩選  │          │  │
│  │  └──────────┘  └──────────┘  └──────────┘          │  │
│  └─────────────────────────────────────────────────────┘  │
│                              │                             │
│                              ▼                             │
│  ┌─────────────────────────────────────────────────────┐  │
│  │                 最終標籤輸出                         │  │
│  └─────────────────────────────────────────────────────┘  │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 優化策略矩陣

| 優先級 | 優化項目 | 預期改善 | 實施難度 |
|--------|----------|----------|----------|
| P0 | VLM 提示詞重構 | +40% 準確率 | 低 |
| P0 | 增強標籤映射 | +30% 匹配率 | 中 |
| P0 | 多階段匹配引擎 | +35% 召回率 | 中 |
| P1 | 動態閾值調整 | +15% 精確率 | 低 |
| P1 | 衝突檢測增強 | +20% 一致性 | 中 |
| P2 | 反饋學習機制 | 持續改善 | 高 |

---

## 三、Phase 3：VLM 提示詞優化

### 3.1 問題分析

當前提示詞的問題：
```python
# 當前提示詞 (glm4v_client.py:246-265)
"""Please analyze this image carefully and provide a detailed description..."""
```

**問題**：
1. 要求提供「描述」而非「標籤」
2. 沒有引用標籤庫的具體內容
3. 輸出格式不統一，難以解析

### 3.2 優化方案

#### 方案 A：標籤感知提示詞（推薦）

```python
TAG_AWARE_PROMPT = """你是一個專業的漫畫封面標籤專家。請分析這張圖片，並從以下標籤類別中選擇最適合的標籤：

【角色類型】
蘿莉、正太、少女、熟女、人妻、貓娘、犬娘、狐娘、精靈、天使、惡魔娘、吸血鬼、魔物娘、扶他、偽娘

【體型特徵】
貧乳、巨乳、極乳、長髮、短髮、雙馬尾、眼鏡、獸耳、尾巴、翅膀、觸手

【服裝類型】
校服、泳裝、比基尼、和服、女僕裝、護士服、兔女郎、內衣、制服、連身裙

【行為場景】
做愛、口交、乳交、手淫、肛交、群交、強姦、觸手、BDSM、綁架

【主題風格】
純愛、NTR、百合、後宮、調教、學園、戀愛、喜劇、劇情

請嚴格按照以下格式輸出：
角色特徵：[標籤1, 標籤2, ...]（最多3個）
體型特徵：[標籤1, 標籤2, ...]（最多3個）
服裝類型：[標籤1, 標籤2, ...]（最多3個）
行為場景：[標籤1, 標籤2, ...]（最多3個，如無則填"無"）
主題風格：[標籤1, 標籤2, ...]（最多2個）

重要規則：
1. 只從上述列表中選擇，不要添加新標籤
2. 如果不確定，寧願不選也不要猜測
3. 注意標籤之間的邏輯一致性（如選擇「蘿莉」不應選「人妻」）"""
```

#### 方案 B：漸進式提示詞

```python
PROGRESSIVE_PROMPT = """請分三步分析這張漫畫封面：

第一步 - 整體觀察（2-3句話）：
描述整體畫面內容和風格

第二步 - 特徵識別（列出關鍵詞）：
- 角色數量和類型：
- 明顯的身體特徵：
- 服裝和配件：
- 動作和互動：

第三步 - 標籤匹配（從標籤庫選擇）：
請從以下標籤中選擇5-8個最匹配的：
[此處動態插入標籤庫的Top 50熱門標籤]

輸出格式：
整體描述：...
特徵關鍵詞：...
匹配標籤：[標籤1, 標籤2, ...]"""
```

### 3.3 實施步驟

1. **創建提示詞測試框架**
   - 準備 20-50 張測試圖片
   - 人工標註正確標籤
   - A/B 測試不同提示詞效果

2. **動態標籤注入**
   ```python
   def build_tag_aware_prompt(tag_library: TagLibrary) -> str:
       """根據標籤庫動態構建提示詞"""
       categories = tag_library.get_categories()
       prompt_parts = []
       
       for category, tags in categories.items():
           tag_list = ", ".join(tags[:20])  # 每類取前20個
           prompt_parts.append(f"【{category}】\n{tag_list}")
       
       return BASE_PROMPT.format(tags="\n\n".join(prompt_parts))
   ```

3. **輸出解析器強化**
   ```python
   class StructuredOutputParser:
       """解析結構化 VLM 輸出"""
       
       def parse(self, text: str) -> Dict[str, List[str]]:
           patterns = {
               'character_types': r'角色特徵[：:]\s*\[(.*?)\]',
               'body_features': r'體型特徵[：:]\s*\[(.*?)\]',
               'clothing': r'服裝類型[：:]\s*\[(.*?)\]',
               'actions': r'行為場景[：:]\s*\[(.*?)\]',
               'themes': r'主題風格[：:]\s*\[(.*?)\]',
           }
           
           result = {}
           for key, pattern in patterns.items():
               match = re.search(pattern, text, re.DOTALL)
               if match:
                   items = [t.strip().strip('"') for t in match.group(1).split(',')]
                   result[key] = [t for t in items if t and t != '無']
               else:
                   result[key] = []
           
           return result
   ```

---

## 四、Phase 4：標籤匹配算法改進

### 4.1 當前流程問題

```python
# 當前流程 (tag_recommender_service.py)
vlm_keywords -> 英文轉中文 -> 字符串匹配 -> 輸出
```

**問題**：
- 單一階段匹配，容錯性低
- 沒有語義理解能力
- 無法處理標籤別名和同義詞

### 4.2 優化方案：多階段匹配引擎

```python
class MultiStageTagMatcher:
    """多階段標籤匹配引擎"""
    
    async def match(self, vlm_analysis: Dict, top_k: int = 10) -> List[TagMatch]:
        """
        四階段匹配流程：
        
        Stage 1: 精確匹配（高置信度）
        Stage 2: 語義匹配（中置信度）  
        Stage 3: RAG 檢索（輔助驗證）
        Stage 4: 組合排序與過濾
        """
        
        # Stage 1: 精確匹配
        exact_matches = self._stage1_exact_match(vlm_analysis)
        
        # Stage 2: 語義匹配
        semantic_matches = await self._stage2_semantic_match(vlm_analysis)
        
        # Stage 3: RAG 檢索
        rag_matches = await self._stage3_rag_retrieval(vlm_analysis)
        
        # Stage 4: 組合與排序
        final_matches = self._stage4_aggregate(
            exact_matches, 
            semantic_matches, 
            rag_matches,
            top_k=top_k
        )
        
        return final_matches
```

### 4.3 各階段詳細設計

#### Stage 1: 精確匹配

```python
def _stage1_exact_match(self, vlm_analysis: Dict) -> List[TagMatch]:
    """
    精確匹配策略：
    1. 直接字符串匹配
    2. 別名映射匹配
    3. 關鍵詞包含匹配
    """
    matches = []
    
    # 1. 直接匹配
    for keyword in vlm_analysis.get('raw_keywords', []):
        if self.tag_library.has_tag(keyword):
            matches.append(TagMatch(
                tag=keyword,
                confidence=1.0,
                source='exact',
                reason='直接匹配'
            ))
    
    # 2. 別名匹配
    for keyword in vlm_analysis.get('raw_keywords', []):
        aliases = self.alias_service.get_aliases(keyword)
        for alias in aliases:
            if self.tag_library.has_tag(alias):
                matches.append(TagMatch(
                    tag=alias,
                    confidence=0.95,
                    source='alias',
                    reason=f'別名匹配: {keyword}'
                ))
    
    return matches
```

#### Stage 2: 語義匹配

```python
async def _stage2_semantic_match(self, vlm_analysis: Dict) -> List[TagMatch]:
    """
    語義匹配策略：
    1. 使用中文嵌入模型計算相似度
    2. 結合描述文本進行語義搜索
    3. 引入閾值動態調整
    """
    matches = []
    
    # 構建查詢文本
    query_parts = []
    query_parts.extend(vlm_analysis.get('character_types', []))
    query_parts.extend(vlm_analysis.get('body_features', []))
    query_parts.extend(vlm_analysis.get('clothing', []))
    query_text = ' '.join(query_parts)
    
    # 使用中文嵌入服務進行語義搜索
    embedding_service = get_chinese_embedding_service()
    
    if embedding_service.is_available():
        # 批量搜索提高效率
        semantic_results = await embedding_service.search_cached_tags(
            query_text,
            top_k=20,
            threshold=self._get_dynamic_threshold()
        )
        
        for result in semantic_results:
            matches.append(TagMatch(
                tag=result['tag'],
                confidence=result['similarity'],
                source='semantic',
                reason=f'語義相似度: {result["similarity"]:.2f}'
            ))
    
    return matches


def _get_dynamic_threshold(self) -> float:
    """
    動態閾值調整：
    - 基於標籤類別調整
    - 基於歷史準確率調整
    """
    base_threshold = 0.6
    
    # 根據類別調整
    category_adjustments = {
        'character': -0.1,  # 角色類別可以更寬鬆
        'action': 0.05,     # 行為類別需要更嚴格
        'theme': -0.05,     # 主題類別適中
    }
    
    return base_threshold
```

#### Stage 3: RAG 檢索

```python
async def _stage3_rag_retrieval(self, vlm_analysis: Dict) -> List[TagMatch]:
    """
    RAG 檢索策略：
    1. 圖像相似度搜索
    2. 標籤協同過濾
    3. 結果可信度評估
    """
    matches = []
    
    # 獲取 RAG 相似圖片
    rag_service = get_rag_service()
    similar_images = await rag_service.search_similar(
        image_bytes=vlm_analysis.get('image_bytes'),
        top_k=10
    )
    
    # 統計相似圖片的標籤
    tag_votes = defaultdict(lambda: {'count': 0, 'scores': []})
    
    for img in similar_images:
        similarity = img.get('score', 0)
        tags = img.get('tags', [])
        
        for tag in tags:
            tag_votes[tag]['count'] += 1
            tag_votes[tag]['scores'].append(similarity)
    
    # 計算加權置信度
    for tag, votes in tag_votes.items():
        if votes['count'] >= 2:  # 至少2張圖片有該標籤
            avg_score = sum(votes['scores']) / len(votes['scores'])
            confidence = min(0.9, avg_score * (1 + votes['count'] * 0.1))
            
            matches.append(TagMatch(
                tag=tag,
                confidence=confidence,
                source='rag',
                reason=f'{votes["count"]} 張相似圖片包含此標籤'
            ))
    
    return matches
```

#### Stage 4: 組合排序與過濾

```python
def _stage4_aggregate(
    self,
    exact_matches: List[TagMatch],
    semantic_matches: List[TagMatch],
    rag_matches: List[TagMatch],
    top_k: int
) -> List[TagMatch]:
    """
    組合排序策略：
    1. 多源結果融合
    2. 置信度加權
    3. 多樣性保證
    4. 類別平衡
    """
    
    # 合併所有匹配
    all_matches = exact_matches + semantic_matches + rag_matches
    
    # 去重並合併置信度
    tag_scores = defaultdict(lambda: {'score': 0, 'sources': [], 'reasons': []})
    
    for match in all_matches:
        tag = match.tag
        
        # 根據來源給予不同權重
        source_weights = {
            'exact': 1.0,
            'alias': 0.95,
            'semantic': 0.8,
            'rag': 0.7
        }
        
        weight = source_weights.get(match.source, 0.5)
        weighted_score = match.confidence * weight
        
        # 累加多源證據
        tag_scores[tag]['score'] = max(
            tag_scores[tag]['score'],
            weighted_score
        )
        tag_scores[tag]['sources'].append(match.source)
        tag_scores[tag]['reasons'].append(match.reason)
    
    # 多樣性重排序
    final_results = self._diversity_rerank(tag_scores, top_k)
    
    return final_results


def _diversity_rerank(
    self,
    tag_scores: Dict,
    top_k: int
) -> List[TagMatch]:
    """
    多樣性重排序：
    - 保證每個類別都有代表
    - 避免同義標籤重複
    """
    results = []
    category_quota = {
        'character': 3,
        'body': 2,
        'clothing': 2,
        'action': 2,
        'theme': 1
    }
    
    category_counts = defaultdict(int)
    
    # 按分數排序
    sorted_tags = sorted(
        tag_scores.items(),
        key=lambda x: x[1]['score'],
        reverse=True
    )
    
    for tag, info in sorted_tags:
        category = self.tag_library.get_category(tag)
        
        # 檢查類別配額
        if category_counts[category] < category_quota.get(category, 1):
            results.append(TagMatch(
                tag=tag,
                confidence=info['score'],
                source='+'.join(set(info['sources'])),
                reason=' | '.join(info['reasons'][:2])
            ))
            category_counts[category] += 1
            
            if len(results) >= top_k:
                break
    
    return results
```

---

## 五、Phase 5：向量嵌入優化

### 5.1 當前問題

- 語義搜索被禁用（tag_recommender_service.py:144）
- 中文嵌入模型可能不適合標籤語義
- 缺乏標籤描述預處理

### 5.2 優化方案

#### 5.2.1 重新啟用並改進語義搜索

```python
# app/services/chinese_embedding_service.py

class ChineseEmbeddingService:
    """優化後的中文嵌入服務"""
    
    def __init__(self):
        self.model = None
        self._tag_embeddings_cache = {}
        self._load_model()
    
    def _load_model(self):
        """加載中文嵌入模型"""
        try:
            from sentence_transformers import SentenceTransformer
            
            # 使用專門的中文語義模型
            self.model = SentenceTransformer('BAAI/bge-m3')
            logger.info("Chinese embedding model loaded: BAAI/bge-m3")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
    
    async def cache_tag_embeddings(self, tags: List[Tag]):
        """
        預計算並緩存所有標籤的嵌入向量
        
        優化點：
        1. 結合標籤名稱和描述生成嵌入
        2. 批量處理提高效率
        3. 持久化緩存避免重複計算
        """
        if not self.model:
            return
        
        texts = []
        for tag in tags:
            # 組合標籤名稱和描述
            text = f"{tag.name}：{tag.description}"
            texts.append(text)
        
        # 批量編碼
        embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            convert_to_tensor=True
        )
        
        # 緩存結果
        for tag, embedding in zip(tags, embeddings):
            self._tag_embeddings_cache[tag.name] = {
                'embedding': embedding,
                'category': tag.category
            }
        
        logger.info(f"Cached {len(tags)} tag embeddings")
```

#### 5.2.2 標籤描述增強

```python
# app/services/tag_enhancement_service.py

class TagEnhancementService:
    """標籤描述增強服務"""
    
    def __init__(self):
        self.enhanced_descriptions = {}
    
    def enhance_tag_description(self, tag: Tag) -> str:
        """
        增強標籤描述以提高嵌入質量：
        1. 添加同義詞
        2. 添加上下文場景
        3. 添加視覺特徵描述
        """
        base_desc = tag.description
        
        # 根據類別添加上下文
        category_context = {
            'character': '漫畫角色類型，外貌特徵為',
            'body': '身體特徵，視覺上可見',
            'clothing': '服裝風格，穿著打扮',
            'action': '性行為或互動場景',
            'theme': '作品主題風格，故事類型'
        }
        
        context = category_context.get(tag.category, '')
        
        # 構建增強描述
        enhanced = f"{tag.name}是一個{context}。{base_desc}"
        
        # 添加相關關鍵詞
        related_keywords = self._get_related_keywords(tag)
        if related_keywords:
            enhanced += f" 相關特徵包括：{', '.join(related_keywords)}"
        
        return enhanced
```

---

## 六、Phase 6：衝突檢測增強

### 6.1 當前問題

- `conflict_learning_system.py` 已實現但未集成到主流程
- 缺乏實時衝突檢測

### 6.2 集成方案

```python
# app/services/tag_conflict_resolver.py

class TagConflictResolver:
    """標籤衝突解決器"""
    
    def __init__(self):
        self.conflict_rules = self._load_conflict_rules()
        self.learner = ConflictLearner()
    
    def _load_conflict_rules(self) -> List[ConflictRule]:
        """加載衝突規則"""
        return [
            # 年齡互斥
            ConflictRule(
                tags=['蘿莉', '人妻'],
                conflict_type='mutual_exclusive',
                reason='年齡特徵衝突'
            ),
            ConflictRule(
                tags=['蘿莉', '熟女'],
                conflict_type='mutual_exclusive',
                reason='年齡特徵衝突'
            ),
            # 體型互斥
            ConflictRule(
                tags=['貧乳', '巨乳'],
                conflict_type='mutual_exclusive',
                reason='胸部大小衝突'
            ),
            # 主題互斥
            ConflictRule(
                tags=['純愛', 'NTR'],
                conflict_type='mutual_exclusive',
                reason='主題風格衝突'
            ),
            # 包含關係（前置標籤）
            ConflictRule(
                tags=['貓娘', '獸耳'],
                conflict_type='implication',
                reason='貓娘應包含獸耳'
            ),
        ]
    
    def resolve_conflicts(self, tags: List[TagMatch]) -> List[TagMatch]:
        """
        解決標籤衝突：
        1. 檢測衝突
        2. 選擇最優組合
        3. 應用邏輯規則
        """
        resolved = tags.copy()
        
        # 1. 檢測互斥衝突
        for rule in self.conflict_rules:
            if rule.conflict_type == 'mutual_exclusive':
                conflicting = [t for t in resolved if t.tag in rule.tags]
                if len(conflicting) > 1:
                    # 保留置信度最高的
                    best = max(conflicting, key=lambda x: x.confidence)
                    for t in conflicting:
                        if t != best:
                            resolved.remove(t)
                            logger.info(f"Removed conflicting tag {t.tag}, kept {best.tag}")
        
        # 2. 應用包含關係
        for rule in self.conflict_rules:
            if rule.conflict_type == 'implication':
                main_tag, implied_tag = rule.tags
                if any(t.tag == main_tag for t in resolved):
                    # 如果有主要標籤，確保包含隱含標籤或移除隱含標籤
                    implied_in_list = any(t.tag == implied_tag for t in resolved)
                    if implied_in_list:
                        # 可以選擇移除隱含標籤以減少冗餘
                        pass
        
        return resolved
```

---

## 七、Phase 7：驗證與測試

### 7.1 測試框架設計

```python
# tests/test_tagging_pipeline.py

class TaggingPipelineTest:
    """標籤系統端到端測試"""
    
    def __init__(self):
        self.test_cases = self._load_test_cases()
        self.metrics = TaggingMetrics()
    
    def _load_test_cases(self) -> List[TestCase]:
        """加載測試用例"""
        return [
            TestCase(
                image_path="test_images/loli_catgirl.jpg",
                expected_tags=["蘿莉", "貓娘", "獸耳"],
                category="character"
            ),
            TestCase(
                image_path="test_images/big_breasts_uniform.jpg",
                expected_tags=["巨乳", "校服"],
                category="body_clothing"
            ),
            # ... 更多測試用例
        ]
    
    async def run_evaluation(self) -> EvaluationResult:
        """運行完整評估"""
        results = []
        
        for case in self.test_cases:
            # 執行標籤
            predicted = await self.tagger.tag_image(case.image_path)
            
            # 計算指標
            precision = self._calculate_precision(predicted, case.expected)
            recall = self._calculate_recall(predicted, case.expected)
            f1 = 2 * (precision * recall) / (precision + recall)
            
            results.append({
                'case': case,
                'predicted': predicted,
                'precision': precision,
                'recall': recall,
                'f1': f1
            })
        
        return EvaluationResult(
            avg_precision=mean([r['precision'] for r in results]),
            avg_recall=mean([r['recall'] for r in results]),
            avg_f1=mean([r['f1'] for r in results]),
            detailed_results=results
        )
```

### 7.2 關鍵指標定義

| 指標 | 定義 | 目標值 |
|------|------|--------|
| 精確率 (Precision) | 正確標籤 / 輸出標籤總數 | > 85% |
| 召回率 (Recall) | 正確標籤 / 期望標籤總數 | > 70% |
| F1 分數 | 2 * (P * R) / (P + R) | > 77% |
| 衝突率 | 衝突標籤對 / 總標籤對 | < 5% |
| 平均置信度 | 輸出標籤的平均信心分數 | > 0.75 |

---

## 八、實施路線圖

### Phase 1：基礎優化（Week 1-2）
- [ ] VLM 提示詞重構
- [ ] 增強標籤映射表
- [ ] 修復語義搜索

### Phase 2：核心改進（Week 3-4）
- [ ] 實現多階段匹配引擎
- [ ] 優化向量嵌入
- [ ] 集成衝突檢測

### Phase 3：驗證部署（Week 5-6）
- [ ] 構建測試套件
- [ ] A/B 測試驗證
- [ ] 性能監控部署

---

## 九、預期成果

### 量化指標

| 指標 | 當前 | 目標 | 提升 |
|------|------|------|------|
| 標籤匹配準確率 | ~45% | > 85% | +89% |
| 標籤召回率 | ~35% | > 70% | +100% |
| 衝突標籤比例 | ~25% | < 5% | -80% |
| 平均處理時間 | 3.5s | < 3s | -14% |

### 質化改善

1. **用戶體驗**：標籤與圖片內容高度相關
2. **維護成本**：自動化測試降低回歸風險
3. **可擴展性**：模塊化設計便於添加新標籤類別
4. **透明度**：每個標籤都有明確的匹配理由

---

## 十、風險與緩解

| 風險 | 影響 | 緩解措施 |
|------|------|----------|
| VLM 輸出不穩定 | 高 | 實施輸出驗證和重試機制 |
| 嵌入模型性能 | 中 | 使用輕量級模型，啟用緩存 |
| 標籤庫變更 | 低 | 設計動態標籤加載機制 |
| 測試覆蓋不足 | 中 | 建立持續集成測試流程 |

---

**文檔版本**: 1.0  
**創建日期**: 2026-02-13  
**作者**: Sisyphus AI Agent
