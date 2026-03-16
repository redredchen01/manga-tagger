# VLM 修復總結與替代方案

## 當前狀態

### ✅ 已修復的部分
1. **API 響應正常** - 服務器返回 200 OK
2. **標籤系統運作** - 返回 5 個標籤（基於 RAG 匹配和建議）
3. **標籤庫載入** - 611 個標籤正常載入
4. **RAG 系統運作** - 找到相似圖片匹配

### ⚠️ 已知問題
**VLM (Vision Language Model) 無法通過 REST API 正確處理圖片**

這是 LM Studio 的已知 Bug (#968)，影響響 GLM-4.6v/4.7v 模型的 REST API 圖片處理功能。

## 根本原因

1. **LM Studio Bug #968**: REST API `/v1/chat/completions` 無法正確處理 base64 圖片
2. **GLM Thinking 功能**: 預設開啟導致空響應（已嘗試禁用）

## 解決方案

### 方案 1: 更新 LM Studio（推薦）
```bash
# 下載最新版本 LM Studio 0.4.2+
# https://lmstudio.ai/download
```
新版本修復了 vision model 相關的 bug。

### 方案 2: 使用 LM Studio JS SDK（最穩定）
不修改 Python 程式碼，改用 LM Studio 的 JavaScript SDK：

```javascript
import { LMStudioClient } from "@lmstudio/sdk";

const client = new LMStudioClient();
const image = await client.files.prepareImageBase64(base64Image);
const model = await client.llm.model("zai-org/glm-4.6v-flash");

const result = await model.respond([
  { role: "user", content: "分析這張圖片", images: [image] }
]);
```

### 方案 3: 使用替代模型
嘗試其他支援 Vision 的模型：
- `Qwen/Qwen2-VL-7B-Instruct` 
- `llava-hf/llava-1.5-7b-hf`
- 或其他在 LM Studio 中正常運作的 VLM

### 方案 4: 修改為本地模型（不使用 LM Studio）
修改 `.env`：
```bash
USE_LM_STUDIO=false
USE_MOCK_SERVICES=false
VLM_MODEL=Qwen/Qwen2-VL-7B-Instruct
```

這會載入 HuggingFace transformers 直接運行模型（需要更多 VRAM）。

## 當前可用的功能

雖然 VLM 有問題，系統仍然可以運作：

1. **RAG 匹配**: 基於相似圖片推薦標籤 ✓
2. **標籤庫匹配**: 基於關鍵字匹配標籤 ✓
3. **標籤映射**: 英文到中文標籤轉換 ✓
4. **完整 API**: 所有端點正常運作 ✓

只是標籤不是由 AI 即時分析圖片產生，而是基於 RAG 數據庫中的相似圖片。

## 測試結果

```
Status: 200 OK
Tags count: 5
Processing time: 13.75s
RAG matches: 1
Library tags available: 611
```

## 建議

1. **短期**: 先使用當前版本，標籤基於 RAG 匹配已可正常運作
2. **中期**: 更新 LM Studio 到 0.4.2+ 版本
3. **長期**: 若仍有问题，考虑使用本地模型或其他 VLM 服務

## 修復的檔案

以下檔案已修改：
- `app/services/lm_studio_vlm_service_v2.py` - 添加 reasoning_content 支持、禁用 thinking
- `app/config.py` - 更新設定說明
- `.env` - 更新註解

## 如何使用當前版本

1. 上傳圖片到 RAG 數據庫（讓系統學習）
2. 上傳新圖片時，系統會基於相似度推薦標籤
3. 隨著 RAG 數據增加，標籤準確度會提升

---
**結論**: 貼標功能已可運作（基於 RAG），但即時 AI 分析需等待 LM Studio 更新或使用替代方案。
