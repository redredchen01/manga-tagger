# Manga Cover Auto-Tagger

快速入門指南

## 安裝

```bash
# 創建虛擬環境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安裝依賴
pip install -r requirements.txt

# 複製環境變數模板
cp .env.example .env
```

## 啟動服務

```bash
# 開發模式
uvicorn app.main:app --reload

# 生產模式
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1
```

## API 使用

### 標記漫畫封面

```bash
curl -X POST "http://localhost:8000/tag-cover" \
  -F "file=@cover.jpg" \
  -F "top_k=5" \
  -F "confidence_threshold=0.5"
```

### 查看所有標籤

```bash
curl "http://localhost:8000/tags"
```

### 健康檢查

```bash
curl "http://localhost:8000/health"
```

## 初始化 RAG 資料集

```bash
# 建立完整的 RAG 索引
python scripts/init_rag.py --dataset-path data/rag_dataset

# 添加單一圖片
python scripts/init_rag.py \
  --single-image path/to/image.jpg \
  --tags '["貓娘", "蘿莉"]'
```

## 專案結構

```
├── app/                    # 主應用程式
│   ├── main.py            # FastAPI 入口
│   ├── config.py          # 設定
│   ├── models.py          # Pydantic 模型
│   ├── api/               # API 路由
│   └── services/          # 業務邏輯
│       ├── vlm_service.py    # Stage 1: VLM
│       ├── rag_service.py    # Stage 2: RAG
│       └── llm_service.py    # Stage 3: LLM
├── data/
│   ├── rag_dataset/       # RAG 參考圖片
│   └── chroma_db/         # ChromaDB 資料
├── scripts/
│   └── init_rag.py        # RAG 初始化腳本
└── requirements.txt
```

## 環境變數

編輯 `.env` 文件：

```env
# 模型設定
VLM_MODEL=Qwen/Qwen2-VL-7B-Instruct
EMBEDDING_MODEL=openai/clip-vit-large-patch14
LLM_MODEL=meta-llama/Llama-3.2-8B-Instruct

# 使用量化模型（VRAM < 16GB）
# VLM_MODEL=Qwen/Qwen2-VL-7B-Instruct-GPTQ-Int4
# LLM_MODEL=meta-llama/Llama-3.2-8B-Instruct-GPTQ-Int4

# 設備設定
DEVICE=cuda  # 或 cpu
```

## 系統需求

- Python 3.9+
- CUDA GPU (推薦，16GB+ VRAM)
- 或 CPU (較慢，32GB+ RAM)
- 硬碟空間: 30GB+

## 技術架構

1. **Stage 1 - VLM**: Qwen2-VL 提取視覺元數據
2. **Stage 2 - RAG**: CLIP + ChromaDB 相似度搜尋
3. **Stage 3 - LLM**: Llama 3.2 合成最終標籤

## 注意事項

- 首次執行會自動下載模型（約 15-20GB）
- 建議使用量化模型以降低 VRAM 需求
- RAG 資料集越大，搜尋結果越準確
