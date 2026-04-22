# Manga Cover Auto-Tagger

Local manga cover tagging service built with FastAPI, LM Studio, and ChromaDB.

## Features

- Multi-stage pipeline: VLM analysis, RAG retrieval, and tag synthesis
- Local-first deployment with LM Studio support
- FastAPI API for tagging, health checks, tag listing, and RAG ingestion
- Streamlit UI for manual testing and dataset management
- Mock-service mode for fast development without loading heavy models

## Project Layout

- `app/`: API, config, models, and service layer
- `scripts/`: maintenance and initialization scripts
- `tests/`: active automated tests
- `data/`: tag library, RAG dataset, and ChromaDB data
- `streamlit_app.py`: local frontend for testing the API

## Quick Start

### 1. Create a virtual environment

```bash
python -m venv venv
venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment

```bash
copy .env.example .env
```

Development mode:

```env
USE_MOCK_SERVICES=true
USE_LM_STUDIO=false
```

LM Studio mode:

```env
USE_MOCK_SERVICES=false
USE_LM_STUDIO=true
LM_STUDIO_BASE_URL=http://127.0.0.1:1234/v1
LM_STUDIO_VISION_MODEL=glm-4.6v-flash
LM_STUDIO_TEXT_MODEL=llama-3.2-8b-instruct
LM_STUDIO_EMBEDDING_MODEL=Qwen/Qwen3-Embedding-8B-GGUF
```

### 4. Start the API

```bash
python start_server.py
```

Alternative:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 5. Open the docs

- Swagger UI: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/api/v1/health`

## API Endpoints

### Tag a manga cover

```bash
curl -X POST "http://localhost:8000/api/v1/tag-cover" ^
  -H "accept: application/json" ^
  -H "Content-Type: multipart/form-data" ^
  -F "file=@cover.jpg" ^
  -F "top_k=5" ^
  -F "confidence_threshold=0.5" ^
  -F "include_metadata=true"
```

Example response:

```json
{
  "tags": [
    {
      "tag": "catgirl",
      "confidence": 0.92,
      "source": "vlm+rag",
      "reason": "Matched by visual features and supporting RAG results"
    }
  ],
  "metadata": {
    "processing_time": 2.34,
    "rag_matches_count": 3,
    "api_version": "2.0.0"
  }
}
```

### Health check

```bash
curl "http://localhost:8000/api/v1/health"
```

### List tags

```bash
curl "http://localhost:8000/api/v1/tags"
```

### Add a reference image to RAG

```bash
curl -X POST "http://localhost:8000/api/v1/rag/add" ^
  -H "accept: application/json" ^
  -H "Content-Type: multipart/form-data" ^
  -F "file=@reference.jpg" ^
  -F "tags=[\"catgirl\",\"school_uniform\"]" ^
  -F "metadata={\"source\":\"manual\"}"
```

## Streamlit UI

Run the frontend locally:

```bash
streamlit run streamlit_app.py
```

The UI expects the FastAPI server to be available at `http://127.0.0.1:8000/api/v1` by default.

## Configuration Notes

Important settings in `.env`:

```env
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=false

USE_LM_STUDIO=true
USE_MOCK_SERVICES=false

LM_STUDIO_BASE_URL=http://127.0.0.1:1234/v1
LM_STUDIO_API_KEY=lm-studio
LM_STUDIO_VISION_MODEL=glm-4.6v-flash
LM_STUDIO_TEXT_MODEL=llama-3.2-8b-instruct
LM_STUDIO_EMBEDDING_MODEL=Qwen/Qwen3-Embedding-8B-GGUF
LM_STUDIO_EMBEDDING_DIM=4096

CHROMA_DB_PATH=./data/chroma_db
CHROMA_COLLECTION_NAME=manga_covers
TAG_LIBRARY_PATH=./data/tags.json

REQUEST_TIMEOUT=120
LOG_LEVEL=INFO
```

## Development

Run tests:

```bash
pytest tests/ -q
```

Initialize or rebuild the RAG collection:

```bash
python scripts/init_rag.py
```

## Current Priorities

- Keep route handlers thin and move orchestration into services
- Improve tag library matching performance as the dataset grows
- Add more targeted tests around LM Studio failure and fallback paths

