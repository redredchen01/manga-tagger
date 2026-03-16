# 優化後的標籤推薦流程

## 流程圖

```mermaid
flowchart TD
    A[圖片輸入] --> B[VLM 分析]
    B --> C{提取關鍵詞}
    C -->|結構化數據| D[詞法匹配]
    C -->|描述文本| E[語義匹配]
    
    D --> F{閾值檢查}
    E --> G{閾值檢查}
    
    F -->|分數 >= 0.6| H[候選標籤池]
    F -->|分數 < 0.6| I[丟棄]
    G -->|分數 >= 0.5| H
    G -->|分數 < 0.5| I
    
    H --> J[RAG 匹配]
    J --> K{閾值 >= 0.5}
    K -->|通過| L[候選標籤池]
    K -->|失敗| M[丟棄]
    
    L --> N{敏感標籤檢查}
    N -->|是敏感標籤| O[VLM 二次驗證]
    O -->|驗證通過| P[保留]
    O -->|驗證失敗| Q[移除]
    N -->|非敏感標籤| P
    
    P --> R{標籤衝突檢查}
    R -->|發現衝突| S[保留高分 移除低分]
    R -->|無衝突| T[保留]
    
    T --> U{最終分數計算}
    U --> V[混合評分]
    V --> W{confidence >= 0.5}
    W -->|通過| X[輸出標籤]
    W -->|失敗| Y[丟棄]
    
    X --> Z[最終結果]
```

## 閾值對照表

| 階段 | 參數 | 舊值 | 新值 | 說明 |
|------|------|------|------|------|
| 詞法匹配 | min_confidence | 0.4 | **0.6** | 提升精確度 |
| 語義匹配 | threshold | 0.3 | **0.5** | 減少噪音 |
| RAG 匹配 | threshold | 0.25 | **0.5** | 過濾低質量匹配 |
| 最終輸出 | confidence_threshold | 0.5 | **0.5** | 保持不變 |

## 新增組件

```mermaid
classDiagram
    class TagValidator {
        +validate_tag_with_vlm()
        +check_tag_conflict()
        +verify_sensitive_tags()
    }
    
    class TagRecommenderService {
        +recommend_tags()
        -_validate_candidates()
        -_filter_by_thresholds()
        -_check_conflicts()
    }
    
    TagRecommenderService --> TagValidator : 使用
```
