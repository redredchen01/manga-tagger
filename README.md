# Manga Cover Auto-Tagger

A backend service for automatically tagging manga covers using Local VLM and RAG (Retrieval-Augmented Generation).

## ✨ Features

- **Multi-Stage Pipeline**: VLM → RAG → LLM synthesis
- **Local Models**: No API keys required, runs entirely on local hardware
- **RAG System**: CLIP embeddings + ChromaDB for similarity search
- **FastAPI**: RESTful API with async support
- **Development Mode**: Mock services for testing without heavy ML models
- **Extensible**: Easy to add new tags and reference images

## 🚀 Quick Start

### Development Mode (Recommended for testing)
No GPU required, starts in seconds!

1. **Clone and setup**:
```bash
git clone <repository-url>
cd manga-tagger
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Configure for development**:
```bash
cp .env.example .env
# Edit .env and set: USE_MOCK_SERVICES=true (for local testing without LM Studio)
# Or set: USE_LM_STUDIO=true and configure LM Studio settings (for production)
```

4. **Start the server**:
```bash
python start_server.py
# or: uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

5. **Test the API**:
```bash
python test_api.py
# or visit: http://localhost:8000/docs
```

### Production Mode (LM Studio)
Requires LM Studio and models:

1. **Install LM Studio and download models** (see LM Studio Integration section)
2. **Configure for production**:
```bash
# Edit .env and set: USE_LM_STUDIO=true
# Configure LM_STUDIO_BASE_URL and model settings
```

3. **Start server** (will use LM Studio models):
```bash
python start_server.py
```

### API Endpoints

#### Tag a Manga Cover

```bash
curl -X POST "http://localhost:8000/tag-cover" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@cover.jpg" \
  -F "top_k=5" \
  -F "confidence_threshold=0.5"
```

Response:
```json
{
  "tags": [
    {
      "tag": "貓姑",
      "confidence": 0.92,
      "source": "vlm+rag",
      "reason": "Character has cat ears and tail visible on cover"
    },
    {
      "tag": "蘋果",
      "confidence": 0.85,
      "source": "vlm",
      "reason": "Character appears to be young female with childlike features"
    }
  ],
  "metadata": {
    "processing_time": 2.34,
    "vlm_description": "A manga cover featuring a cat-eared girl...",
    "rag_matches": 5
  }
}
```

#### Health Check

```bash
curl "http://localhost:8000/health"
```

### API Endpoints

#### Tag a Manga Cover

```bash
curl -X POST "http://localhost:8000/tag-cover" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@cover.jpg" \
  -F "top_k=5" \
  -F "confidence_threshold=0.5"
```

Response:
```json
{
  "tags": [
    {
      "tag": "貓娘",
      "confidence": 0.92,
      "source": "vlm+rag",
      "reason": "Character has cat ears and tail visible on cover"
    },
    {
      "tag": "蘿莉",
      "confidence": 0.85,
      "source": "vlm",
      "reason": "Character appears to be young female with childlike features"
    }
  ],
  "metadata": {
    "processing_time": 2.34,
    "vlm_description": "A manga cover featuring a cat-eared girl...",
    "rag_matches": 5
  }
}
```

#### Health Check

```bash
curl "http://localhost:8000/health"
```

#### List Available Tags

```bash
curl "http://localhost:8000/tags"
```

#### Add Image to RAG Dataset

```bash
curl -X POST "http://localhost:8000/rag/add" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@reference.jpg" \
  -F "tags=[\"貓娘\",\"蘿莉\"]"
```

## LM Studio Integration (NEW - Recommended)

Instead of using local models, you can use LM Studio for all AI processing:

### Prerequisites

1. **Install LM Studio**: Download from https://lmstudio.ai/
2. **Download Models**: In LM Studio, download these models:
   - **GLM-4.6V-flash** (for vision analysis)
   - **Qwen3-coder-next** or similar (for text generation)
3. **Start LM Studio Server**: Go to Settings → Server and start the server
4. **Verify**: Check http://127.0.0.1:1234/v1/models to see available models

### Configuration

1. **Edit .env file**:
```env
# Enable LM Studio mode
USE_LM_STUDIO=true

# LM Studio settings
LM_STUDIO_BASE_URL=http://127.0.0.1:1234/v1
LM_STUDIO_API_KEY=lm-studio
LM_STUDIO_VISION_MODEL=zai-org/glm-4.6v-flash
LM_STUDIO_TEXT_MODEL=qwen/qwen3-coder-next
```

2. **Set USE_MOCK_SERVICES=false** in .env

### Testing

1. **Start the server**:
```bash
python start_server.py
```

2. **Test connectivity**:
```bash
curl http://127.0.0.1:8000/test
```

3. **Test image analysis**:
```bash
curl -X POST "http://127.0.0.1:8000/tag-cover" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@cover.jpg" \
  -F "top_k=5" \
  -F "confidence_threshold=0.5"
```

### Available Models

The service currently supports:
- **GLM-4.6V-flash**: Vision analysis (image understanding, tagging)
- **Qwen3-coder-next**: Text generation (tag synthesis, reasoning)

### Performance Tips

1. **Use GPU**: Ensure LM Studio is using GPU for faster processing
2. **Adjust Model Settings**: You can change models in .env if needed
3. **Monitor Resources**: Large images may require more VRAM

## Configuration

Edit `.env` file:

```env
# API Settings
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=1
DEBUG=false

# LM Studio Configuration (NEW - Recommended)
USE_LM_STUDIO=true
LM_STUDIO_BASE_URL=http://127.0.0.1:1234/v1
LM_STUDIO_API_KEY=lm-studio
LM_STUDIO_VISION_MODEL=zai-org/glm-4.6v-flash
LM_STUDIO_TEXT_MODEL=qwen/qwen3-coder-next

# Legacy Model Configuration (for local models - disabled when USE_LM_STUDIO=true)
VLM_MODEL=Qwen/Qwen2-VL-7B-Instruct
EMBEDDING_MODEL=openai/clip-vit-large-patch14
LLM_MODEL=meta-llama/Llama-3.2-8B-Instruct

# Model Inference Settings
VLM_MAX_TOKENS=512
LLM_MAX_TOKENS=1024
TEMPERATURE=0.7
TOP_P=0.9
DEVICE=cuda

# RAG / ChromaDB Settings
CHROMA_DB_PATH=./data/chroma_db
CHROMA_COLLECTION_NAME=manga_covers
RAG_TOP_K=5
RAG_SIMILARITY_THRESHOLD=0.7

# Tag Library
TAG_LIBRARY_PATH=./data/tags.json

# Performance Settings
BATCH_SIZE=1
MAX_CONCURRENT_REQUESTS=1
REQUEST_TIMEOUT=120

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json

# Mock Services (disabled for production)
USE_MOCK_SERVICES=false
```

## Performance Tips

1. **Use Quantized Models**: 4-bit quantization reduces VRAM by ~75%
2. **Adjust Batch Size**: Reduce if OOM errors occur
3. **Enable Model Offloading**: For CPU+GPU hybrid inference
4. **Cache Embeddings**: RAG embeddings are cached in ChromaDB

## Development

### Run Tests

```bash
pytest tests/
```

### Add Custom Tags

1. Edit `data/tags.json`
2. Add reference images to `data/rag_dataset/`
3. Rebuild RAG index: `python scripts/init_rag.py`

## License

MIT License

## Contributing

Pull requests welcome! Please ensure:
- Code follows PEP 8
- Tests pass
- Documentation updated
