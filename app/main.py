"""Main FastAPI application for Manga Cover Auto-Tagger."""

import logging
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

from app.config import settings
from app.api.routes_v2 import router

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# 效能優化: 添加請求 ID 中間件
class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting Manga Cover Auto-Tagger Service")
    settings.ensure_directories()
    logger.info(f"ChromaDB path: {settings.chroma_db_path}")
    logger.info(f"Device: {settings.DEVICE}")
    yield
    # Shutdown
    logger.info("Shutting down Manga Cover Auto-Tagger Service")


# Create FastAPI app
app = FastAPI(
    title="Manga Cover Auto-Tagger",
    description="""
    Automatically tag manga covers using Local VLM and RAG.
    
    ## Features
    
    - **Multi-Stage Pipeline**: VLM → RAG → LLM synthesis
    - **Local Models**: No API keys required
    - **RAG System**: CLIP embeddings + ChromaDB
    - **Manga Description Generation**: Specialized prompts for detailed manga content analysis
    
    ## Usage
    
    1. Upload a manga cover image to `/tag-cover` for automated tagging
    2. Upload a manga cover image to `/generate-manga-description` for detailed content analysis
    3. Receive JSON list of tags with confidence scores or detailed descriptions
    """,
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 效能優化: GZip 壓縮回應
app.add_middleware(GZipMiddleware, minimum_size=500)

# 效能優化: 請求 ID 追蹤
app.add_middleware(RequestIDMiddleware)


# Include API routes
app.include_router(router, prefix="/api/v1")


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "service": "Manga Cover Auto-Tagger",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "tag_cover": "/tag-cover",
            "upload": "/upload",
            "generate_manga_description": "/generate-manga-description",
            "health": "/health",
            "tags": "/tags",
            "rag_add": "/rag/add",
            "rag_stats": "/rag/stats",
        },
    }
