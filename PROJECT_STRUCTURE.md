# Project Structure

```
manga-tagger/
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI application
│   ├── config.py              # Configuration management
│   ├── models.py              # Pydantic models
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes.py          # API endpoints
│   └── services/
│       ├── __init__.py
│       ├── vlm_service.py     # Stage 1: VLM (Qwen2-VL)
│       ├── rag_service.py     # Stage 2: RAG + Embeddings (CLIP + ChromaDB)
│       └── llm_service.py     # Stage 3: LLM (Llama 3.2)
├── data/
│   ├── rag_dataset/           # RAG reference images
│   └── chroma_db/             # ChromaDB persistence
├── models/                    # Local model cache
├── scripts/
│   └── init_rag.py           # RAG dataset initialization
├── tests/
├── .env                      # Environment variables
├── .env.example
├── requirements.txt
└── README.md
```

## Architecture Overview

### Stage 1: VLM (Qwen2-VL) - Visual Metadata Extraction
- Extract detailed visual descriptions from manga covers
- Identify characters, themes, art style, genre indicators
- Output structured metadata

### Stage 2: RAG (CLIP + ChromaDB) - Similarity Search
- Generate embeddings for input image using CLIP
- Search ChromaDB for similar reference covers
- Retrieve relevant tags from top-K matches

### Stage 3: LLM (Llama 3.2) - Tag Synthesis
- Combine VLM metadata + RAG candidate tags
- Generate final tag list with confidence scores
- Explain reasoning for each tag

## API Endpoints

- `POST /tag-cover` - Main tagging endpoint
- `GET /health` - Health check
- `GET /tags` - List available tags
- `POST /rag/add` - Add image to RAG dataset
