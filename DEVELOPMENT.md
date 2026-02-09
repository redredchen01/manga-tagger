# 開發指南 (Development Guide)

## 快速開始

### 1. 環境設置

```bash
# 創建虛擬環境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安裝依賴
pip install -r requirements.txt

# 設置環境變數
cp .env.example .env
# 編輯 .env 文件
```

### 2. 開發模式

開發模式下使用 Mock 服務，無需下載大型模型：

```bash
# 確保 .env 中有 USE_MOCK_SERVICES=true
USE_MOCK_SERVICES=true
DEBUG=true

# 啟動開發服務器
python start_server.py

# 或手動啟動
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 3. 生產模式

生產模式下使用真實模型（需要大量 VRAM 和下載時間）：

```bash
# 編輯 .env
USE_MOCK_SERVICES=false

# 安裝完整依賴（包括量化支援）
pip install accelerate bitsandbytes optimum auto-gptq

# 啟動服務器
python start_server.py
```

## API 測試

### 使用測試客戶端

```bash
# 啟動服務器後，運行測試
python test_api.py
```

### 手動測試

```bash
# 健康檢查
curl http://localhost:8000/health

# 獲取標籤列表
curl http://localhost:8000/tags

# 標記圖片（需要真實圖片文件）
curl -X POST "http://localhost:8000/tag-cover" \
  -F "file=@test_image.jpg" \
  -F "top_k=5" \
  -F "confidence_threshold=0.5"
```

## 專案結構

```
manga-tagger/
├── app/                         # FastAPI 應用
│   ├── api/routes.py             # API 路由
│   ├── config.py                # 配置管理
│   ├── models.py               # Pydantic 模型
│   └── services/
│       ├── vlm_service.py       # VLM 服務（真實）
│       ├── rag_service.py       # RAG 服務（真實）
│       ├── llm_service.py      # LLM 服務（真實）
│       └── mock_services.py    # Mock 服務（開發）
├── data/                      # 數據目錄
│   ├── chroma_db/            # ChromaDB 數據
│   ├── rag_dataset/           # RAG 參考圖片
│   └── tags.json             # 標籤庫
├── tests/                    # 測試文件
├── scripts/                  # 腳本工具
├── start_server.py          # 啟動腳本
├── test_api.py              # API 測試
└── requirements.txt         # Python 依賴
```

## 配置選項

### .env 文件

```env
# API 設置
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=true
USE_MOCK_SERVICES=true    # 開發模式：true，生產模式：false

# 模型配置（生產模式）
VLM_MODEL=Qwen/Qwen2-VL-7B-Instruct
EMBEDDING_MODEL=openai/clip-vit-large-patch14
LLM_MODEL=meta-llama/Llama-3.2-8B-Instruct

# 設備配置
DEVICE=cuda    # 或 cpu

# RAG 設置
CHROMA_DB_PATH=./data/chroma_db
RAG_TOP_K=5
RAG_SIMILARITY_THRESHOLD=0.7
```

## 開發工作流程

### 1. 新增 API 端點

1. 在 `app/models.py` 中定義請求/響應模型
2. 在 `app/api/routes.py` 中實現端點
3. 在 `test_api.py` 中添加測試

### 2. 修改服務邏輯

- **開發模式**：修改 `app/services/mock_services.py`
- **生產模式**：修改對應的真實服務文件

### 3. 數據庫管理

```bash
# 初始化 RAG 數據庫
python scripts/init_rag.py --dataset-path data/rag_dataset

# 添加參考圖片到 RAG
curl -X POST "http://localhost:8000/rag/add" \
  -F "file=@reference.jpg" \
  -F "tags=[\"貓娘\",\"蘿莉\"]"
```

## 故障排除

### 1. 服務器無法啟動

```bash
# 檢查端口是否被占用
netstat -an | grep 8000

# 檢查配置
python -c "from app.config import settings; print(settings.dict())"
```

### 2. 模型加載失敗

- 確保有足夠的 VRAM（建議 16GB+）
- 檢查網路連接（需要下載模型）
- 使用量化模型減少 VRAM 使用

### 3. 依賴問題

```bash
# 重新安裝依賴
pip install -r requirements.txt --force-reinstall

# 檢查 PyTorch 安裝
python -c "import torch; print(torch.cuda.is_available())"
```

## 性能優化

### 1. Mock 模式
- 無需 GPU
- 啟動時間 < 5 秒
- 適合開發和測試

### 2. 生產模式
- 需要 GPU（建議 RTX 3090+）
- 首次啟動需要下載模型（15-20GB）
- 建議使用 4-bit 量化

## 測試覆蓋

```bash
# 運行所有測試
pytest tests/ -v

# 運行特定測試
pytest tests/test_api.py -v

# 檢查代碼風格
black app/ tests/
ruff check app/ tests/
mypy app/
```

## 部署

### Docker 部署

```bash
# 構建鏡像
docker build -t manga-tagger .

# 運行容器
docker run -p 8000:8000 manga-tagger
```

### 生產部署

1. 設置 `USE_MOCK_SERVICES=false`
2. 配置 GPU 支援
3. 設置反向代理（nginx）
4. 配置監控和日誌