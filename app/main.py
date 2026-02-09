"""Main FastAPI application for Manga Cover Auto-Tagger."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.api.routes_v2 import router

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


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

# Include API routes
app.include_router(router, prefix="/api/v1")

# Also mount routes at root for convenience
app.include_router(router, prefix="")


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
