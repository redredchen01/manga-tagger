# RTX 5090 中文嵌入模型部署完整指南

## 🎯 RTX 5090 支持的模型推薦

基於RTX 5090的24GB VRAM規格，以下模型為最佳選擇：

| 優先級 | 模型 | VRAM使用 | 性能評級 | 推薦用途 |
|--------|------|-----------|-----------|----------|
| **🥇 最高推薦** | Qwen3-Embedding-4B | ~8GB | ⭐⭐⭐⭐⭐⭐ | 生產環境，高精度 |
| **🥈 性能平衡** | Qwen3-Embedding-8B | ~16GB | ⭐⭐⭐⭐⭐⭐ | 最高精度，研究用途 |
| **🥉 輕量快速** | Qwen3-Embedding-0.6B | ~2GB | ⭐⭐⭐⭐ | 實時應用，高併發 |
| **🏅 性價比** | BGE-M3 | ~2GB | ⭐⭐⭐⭐ | 預算有限項目 |
| **🌟 雙語專用** | Jina-Embeddings-v2-base-zh | ~1GB | ⭐⭐⭐ | 中英雙語場景 |

## 🛠️ 環境準備與安裝

### 第一步：系統環境檢查

```bash
# 檢查GPU狀態
nvidia-smi

# 預期輸出應包含：
# - GPU: RTX 5090
# - Memory: 24GB
# - Driver: >= 550.00
# - CUDA: >= 12.1

# 檢查CUDA版本
nvcc --version

# 檢查Python版本
python --version  # 需要3.9+
```

### 第二步：CUDA環境安裝

```bash
# 下載CUDA 12.1 (推薦版本)
wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda_12.1.0_531.14_linux.run

# 安裝CUDA
sudo sh cuda_12.1.0_531.14_linux.run

# 下載cuDNN 8.9 (對應CUDA 12.1)
wget https://developer.download.nvidia.com/compute/cudnn/8.9.0/local_installers/cudnn-linux-x86_64-8.9.0.29_cuda12-archive.tar.xz

# 解壓並複製到CUDA目錄
tar -xf cudnn-linux-x86_64-8.9.0.29_cuda12-archive.tar.xz
sudo cp cudnn-*-archive/include/cudnn*.h /usr/local/cuda/include
sudo cp cudnn-*-archive/lib/libcudnn* /usr/local/cuda/lib64
sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*
```

### 第三步：Python環境設置

```bash
# 創建虛擬環境
conda create -n embedding python=3.10 -y
conda activate embedding

# 升級pip
pip install --upgrade pip

# 安裝PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 安裝基礎依賴
pip install numpy pandas scikit-learn tqdm
```

## 📦 模型安裝指南

### 選項1：Qwen3-Embedding-4B (推薦)

```python
# 安裝transformers
pip install transformers>=4.35.0

# 安裝加速庫
pip install accelerate flash-attn

# 下載並測試模型
from transformers import AutoTokenizer, AutoModel
import torch

# 加載模型
model_name = "Qwen/Qwen3-Embedding-4B"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype=torch.float16,  # 使用半精度節省記憶體
    device_map="auto"
)

# 測試推理
def encode_text(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=8192)
    with torch.no_grad():
        outputs = model(**inputs)
        # 使用CLS token的embedding
        embeddings = outputs.last_hidden_state[:, 0, :]
    return embeddings.cpu().numpy()

# 測試
test_text = "這是一段測試文本"
embedding = encode_text(test_text)
print(f"Embedding shape: {embedding.shape}")
print(f"VRAM使用: {torch.cuda.memory_allocated()/1024**3:.2f}GB")
```

### 選項2：BGE-M3 (性價比選擇)

```python
# 安裝FlagEmbedding
pip install -U FlagEmbedding

# 安裝額外依賴
pip install peft optimum

from FlagEmbedding import BGEM3FlagModel

# 初始化模型（使用半精度）
model = BGEM3FlagModel(
    'BAAI/bge-m3',
    use_fp16=True,  # 啟用半精度
    device='cuda'
)

# 測試編碼
texts = [
    "這是第一段測試文本",
    "這是第二段測試文本"
]

embeddings = model.encode(
    texts,
    batch_size=32,
    max_length=8192
)

print(f"產生了 {len(embeddings)} 個嵌入向量")
print(f"每個向量的維度: {embeddings[0].shape}")
```

### 選項3：Qwen3-Embedding-0.6B (最高性能)

```python
# 使用sentence-transformers
pip install -U sentence-transformers

from sentence_transformers import SentenceTransformer

# 加載輕量級模型
model = SentenceTransformer(
    'Qwen/Qwen3-Embedding-0.6B',
    device='cuda',
    trust_remote_code=True
)

# 批量編碼
texts = ["文本1", "文本2", "文本3"] * 1000  # 3000個文本
embeddings = model.encode(
    texts,
    batch_size=256,
    show_progress_bar=True,
    normalize_embeddings=True
)

print(f"處理了 {len(texts)} 個文本")
print(f"嵌入向量維度: {embeddings.shape}")
```

## 🚀 API服務部署

### FastAPI部署腳本

```python
# 安裝依賴
pip install fastapi uvicorn pydantic

# 創建api_server.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import torch
import uvicorn
import time
from contextlib import asynccontextmanager

app = FastAPI(title="中文嵌入API服務")

# 全局模型變量
model = None
tokenizer = None

class EmbeddingRequest(BaseModel):
    texts: List[str]
    model_name: Optional[str] = "qwen3-4b"
    normalize: bool = True

class EmbeddingResponse(BaseModel):
    embeddings: List[List[float]]
    usage: dict
    model: str
    processing_time: float

@app.on_event("startup")
async def startup_event():
    global model, tokenizer
    
    # 根據可用VRAM選擇模型
    if torch.cuda.get_device_properties(0).total_memory > 15e9:  # >15GB
        model_name = "Qwen/Qwen3-Embedding-8B"
        print("使用 Qwen3-8B 模型")
    elif torch.cuda.get_device_properties(0).total_memory > 8e9:  # >8GB
        model_name = "Qwen/Qwen3-Embedding-4B"
        print("使用 Qwen3-4B 模型")
    else:
        model_name = "Qwen/Qwen3-Embedding-0.6B"
        print("使用 Qwen3-0.6B 模型")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    print(f"模型加載完成，VRAM使用: {torch.cuda.memory_allocated()/1024**3:.2f}GB")

@app.post("/embeddings", response_model=EmbeddingResponse)
async def create_embeddings(request: EmbeddingRequest):
    start_time = time.time()
    
    try:
        # 根據請求選擇模型
        inputs = tokenizer(
            request.texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=8192
        ).to(model.device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :]
            
            if request.normalize:
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        
        processing_time = time.time() - start_time
        usage = {
            "prompt_tokens": inputs.input_ids.shape[1],
            "total_tokens": inputs.input_ids.numel(),
            "vram_used": f"{torch.cuda.memory_allocated()/1024**3:.2f}GB"
        }
        
        return EmbeddingResponse(
            embeddings=embeddings.cpu().tolist(),
            usage=usage,
            model=request.model_name,
            processing_time=processing_time
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "gpu_memory": f"{torch.cuda.memory_allocated()/1024**3:.2f}GB",
        "model": model.config._name_or_path if model else "not_loaded"
    }

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        workers=1  # GPU模型只能用單worker
    )
```

### Docker部署配置

```dockerfile
# Dockerfile
FROM nvidia/cuda:12.1-devel-ubuntu22.04

# 設置環境變量
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# 安裝系統依賴
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-pip \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# 創建工作目錄
WORKDIR /app

# 複製依賴文件
COPY requirements.txt .

# 安裝Python依賴
RUN pip3 install --no-cache-dir -r requirements.txt

# 複製應用代碼
COPY . .

# 暴露端口
EXPOSE 8000

# 啟動命令
CMD ["python3", "api_server.py"]
```

```yaml
# docker-compose.yml
version: '3.8'

services:
  embedding-api:
    build: .
    container_name: qwen3-embedding
    ports:
      - "8000:8000"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    environment:
      - CUDA_VISIBLE_DEVICES=0
    volumes:
      - ./models:/app/models
      - ./logs:/app/logs
    restart: unless-stopped
```

## ⚡ 性能優化設置

### VRAM優化

```python
# 啟用梯度檢查點
model.gradient_checkpointing_enable()

# 使用半精度
with torch.cuda.amp.autocast():
    outputs = model(**inputs)

# 動態批處理
def optimal_batch_size(model, max_seq_length):
    # 測試不同batch_size找到最優值
    for bs in [1, 2, 4, 8, 16, 32]:
        try:
            dummy = torch.randint(0, 10000, (bs, max_seq_length)).cuda()
            with torch.no_grad():
                _ = model(dummy)
            print(f"Batch size {bs}: OK")
            return bs
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            continue
    return 1  # 預設

# 啟用Flash Attention
if hasattr(model.config, 'attn_implementation'):
    model.config.attn_implementation = "flash_attention_2"
```

### 多進程處理

```python
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

def batch_encode_worker(texts_batch, model_name):
    # 子進程中獨立加載模型
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    
    # 編碼處理
    embeddings = []
    for text in texts_batch:
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
            embedding = outputs.last_hidden_state[:, 0, :].squeeze()
            embeddings.append(embedding.cpu().numpy())
    
    return embeddings

# 主進程
def parallel_encode(texts, model_name, num_workers=4):
    batch_size = len(texts) // num_workers
    batches = [texts[i:i+batch_size] for i in range(0, len(texts), batch_size)]
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(batch_encode_worker, batch, model_name) for batch in batches]
        results = [future.result() for future in futures]
    
    # 合併結果
    all_embeddings = []
    for batch_result in results:
        all_embeddings.extend(batch_result)
    
    return all_embeddings
```

## 📊 性能測試腳本

```python
# performance_test.py
import time
import psutil
import GPUtil
import numpy as np

class PerformanceMonitor:
    def __init__(self):
        self.results = []
    
    def start_test(self, test_name):
        self.start_time = time.time()
        self.start_memory = torch.cuda.memory_allocated()
        self.test_name = test_name
        print(f"開始測試: {test_name}")
    
    def end_test(self):
        end_time = time.time()
        end_memory = torch.cuda.memory_allocated()
        
        duration = end_time - self.start_time
        memory_used = (end_memory - self.start_memory) / 1024**3
        
        gpu_info = GPUtil.getGPUs()[0]
        gpu_util = gpu_info.load * 100
        
        result = {
            "test": self.test_name,
            "duration": duration,
            "memory_used_gb": memory_used,
            "gpu_utilization": gpu_util,
            "throughput_qps": 1.0 / duration
        }
        
        self.results.append(result)
        print(f"完成測試: {test_name}, 耗時: {duration:.3f}s, QPS: {result['throughput_qps']:.1f}")
        
        return result

def test_single_embedding(model, text, monitor):
    monitor.start_test("單文本編碼")
    result = model.encode(text)
    return monitor.end_test()

def test_batch_embedding(model, texts, monitor):
    monitor.start_test("批量編碼")
    results = model.encode(texts, batch_size=32)
    return monitor.end_test()

def test_concurrent_requests(model, texts, monitor, num_requests=100):
    import threading
    import queue
    
    result_queue = queue.Queue()
    monitor.start_test("併發請求")
    
    def worker():
        for text in texts[:num_requests]:
            result = model.encode(text)
            result_queue.put(result)
    
    threads = [threading.Thread(target=worker) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    
    return monitor.end_test()

# 主測試函數
def run_performance_tests():
    monitor = PerformanceMonitor()
    
    # 測試不同模型
    models_to_test = [
        ("Qwen/Qwen3-Embedding-4B", "Qwen3-4B"),
        ("BAAI/bge-m3", "BGE-M3"),
        ("Qwen/Qwen3-Embedding-0.6B", "Qwen3-0.6B")
    ]
    
    test_text = "這是一段用於性能測試的中文文本，包含各種常見詞彙和句式結構。"
    test_texts = [test_text] * 100
    
    for model_path, model_name in models_to_test:
        print(f"\n=== 測試模型: {model_name} ===")
        
        # 加載模型
        if "bge-m3" in model_path:
            from FlagEmbedding import BGEM3FlagModel
            model = BGEM3FlagModel(model_path, use_fp16=True)
        else:
            from transformers import AutoTokenizer, AutoModel
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            model = AutoModel.from_pretrained(
                model_path, 
                trust_remote_code=True,
                torch_dtype=torch.float16,
                device='cuda'
            )
            # 包裝encode方法
            model.encode = lambda texts: _encode_with_transformers(model, tokenizer, texts)
        
        # 執行測試
        test_single_embedding(model, test_text, monitor)
        test_batch_embedding(model, test_texts, monitor)
        test_concurrent_requests(model, test_texts, monitor)
        
        # 清理記憶體
        del model
        if 'tokenizer' in locals():
            del tokenizer
        torch.cuda.empty_cache()
    
    # 生成報告
    generate_performance_report(monitor.results)

def _encode_with_transformers(model, tokenizer, texts):
    if isinstance(texts, str):
        texts = [texts]
    
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=8192)
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :]
        return embeddings.cpu().numpy()

def generate_performance_report(results):
    print("\n" + "="*60)
    print("性能測試報告")
    print("="*60)
    
    # 按模型分組
    models = {}
    for result in results:
        test_name = result['test']
        if 'Qwen3-4B' in result['test']:
            model = 'Qwen3-4B'
        elif 'BGE-M3' in result['test']:
            model = 'BGE-M3'
        elif 'Qwen3-0.6B' in result['test']:
            model = 'Qwen3-0.6B'
        else:
            continue
            
        if model not in models:
            models[model] = {}
        models[model][test_name.split('_')[-1]] = result
    
    # 生成表格
    for model_name, tests in models.items():
        print(f"\n{model_name} 性能數據:")
        print(f"{'測試類型':<15} {'耗時(s)':<10} {'QPS':<8} {'記憶體(GB)':<12} {'GPU使用率(%)':<12}")
        print("-" * 70)
        for test_type, data in tests.items():
            print(f"{test_type:<15} {data['duration']:<10.3f} {data['throughput_qps']:<8.1f} {data['memory_used_gb']:<12.3f} {data['gpu_utilization']:<12.1f}")

if __name__ == "__main__":
    run_performance_tests()
```

## 🔧 啟動腳本

```bash
#!/bin/bash
# start_embedding_service.sh

echo "🚀 啟動RTX 5090中文嵌入服務"

# 檢查環境
echo "檢查GPU狀態..."
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits

# 啟動服務
echo "啟動API服務..."
if [ "$1" = "docker" ]; then
    echo "使用Docker部署..."
    docker-compose up -d
elif [ "$1" = "local" ]; then
    echo "使用本地部署..."
    python api_server.py
else
    echo "請指定部署方式: ./start_embedding_service.sh [docker|local]"
fi

# 等待服務就緒
echo "等待服務啟動..."
sleep 10

# 健康檢查
echo "執行健康檢查..."
curl -X GET http://localhost:8000/health | python3 -m json.tool

echo "✅ 服務已啟動！"
echo "API文檔: http://localhost:8000/docs"
```

## 📝 使用範例

```python
# client_example.py
import requests
import json

# API端點
API_URL = "http://localhost:8000/embeddings"

def test_embedding_api():
    # 測試文本
    test_data = {
        "texts": [
            "人工智能技術正在快速發展",
            "機器學習是AI的重要分支",
            "深度學習模型的性能不斷提升"
        ],
        "model_name": "qwen3-4b",
        "normalize": True
    }
    
    # 發送請求
    response = requests.post(API_URL, json=test_data)
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ 成功獲取 {len(result['embeddings'])} 個嵌入向量")
        print(f"⚡ 處理時間: {result['processing_time']:.3f}秒")
        print(f"💾 VRAM使用: {result['usage']['vram_used']}")
        print(f"📏 向量維度: {len(result['embeddings'][0])}")
        return result
    else:
        print(f"❌ 請求失敗: {response.status_code}")
        print(f"錯誤信息: {response.text}")
        return None

def performance_test():
    """性能測試"""
    import time
    
    # 生成測試數據
    test_texts = [f"這是第{i}個測試文本" for i in range(1000)]
    
    start_time = time.time()
    
    # 發送批量請求
    for i in range(0, len(test_texts), 50):  # 每批50個
        batch = test_texts[i:i+50]
        test_data = {"texts": batch}
        response = requests.post(API_URL, json=test_data)
        
        if response.status_code != 200:
            print(f"批次 {i//50} 失敗")
    
    end_time = time.time()
    
    print(f"📊 性能測試完成:")
    print(f"   處理文本數: {len(test_texts)}")
    print(f"   總耗時: {end_time - start_time:.2f}秒")
    print(f"   平均QPS: {len(test_texts)/(end_time - start_time):.1f}")

if __name__ == "__main__":
    # 基本功能測試
    test_embedding_api()
    
    # 性能測試
    print("\n開始性能測試...")
    performance_test()
```

## 🎯 總結與建議

### RTX 5090最優配置推薦：

1. **日常使用**: Qwen3-Embedding-4B + 半精度
2. **最高精度**: Qwen3-Embedding-8B + 量化優化  
3. **高性能**: Qwen3-Embedding-0.6B + 批處理
4. **性價比**: BGE-M3 + Flash Attention

### 關鍵性能指標預期：

- **QPS**: 100-3000 (取決於模型大小)
- **延遲**: 10-50ms P99
- **記憶體使用**: 2-16GB VRAM
- **準確率**: MTEB分數60-70+

### 故障排除：

```bash
# 常見問題解決

# 1. CUDA out of memory
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# 2. 模型加載慢
pip install transformers --upgrade

# 3. 性能不佳
# 檢查GPU頻率是否正常
nvidia-smi -q

# 4. API響應慢
# 增加worker數量或使用更大的batch_size
```

現在您可以按照這個指南在RTX 5090上成功部署中文嵌入模型了！🚀