"""Main FastAPI application for Manga Cover Auto-Tagger.

DDD structure: core/ | domain/ | infrastructure/ | interfaces/
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

from app.api.routes_v2 import router
from app.core.background_tasks import start_background_cleanup, stop_background_cleanup
from app.core.cache import cache_manager
from app.core.config import settings
from app.core.exceptions import AppException
from app.core.http_client import close_http_client
from app.core.logging_config import configure_logging, get_logger
from app.domain.tag.library import get_tag_library_service
from app.domain.tag.recommender import get_tag_recommender_service
from app.infrastructure.lm_studio.vlm_service import LMStudioVLMService
from app.infrastructure.lm_studio.llm_service import LMStudioLLMService
from app.infrastructure.rag.rag_service import RAGService
from app.middleware.error_handler import app_exception_handler, general_exception_handler
from app.middleware.metrics_middleware import MetricsMiddleware
from app.middleware.tracing import TracingMiddleware

configure_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager — populates app.state with service singletons."""
    # Startup
    logger.info("Starting Manga Cover Auto-Tagger Service")
    settings.ensure_directories()
    logger.info(f"ChromaDB path: {settings.chroma_db_path}")
    logger.info(f"Device: {settings.DEVICE}")

    # Connect to Redis (optional, gracefully handles unavailability)
    await cache_manager.connect()
    if cache_manager.enabled:
        logger.info("Redis caching enabled")
    else:
        logger.info("Redis caching disabled (not available)")

    # Populate app.state with service singletons for dependency injection
    logger.info("Initializing service singletons...")
    if settings.USE_OLLAMA:
        from app.infrastructure.ollama.ollama_vlm_service import OllamaVLMService
        app.state.vlm_service = OllamaVLMService()
        logger.info("VLM backend: Ollama")
    else:
        app.state.vlm_service = LMStudioVLMService()
        logger.info("VLM backend: LM Studio")
    app.state.llm_service = LMStudioLLMService()
    app.state.rag_service = RAGService()
    app.state.tag_library = get_tag_library_service()
    app.state.tag_recommender = get_tag_recommender_service()
    logger.info("Service singletons initialized")

    # Start background cleanup task
    await start_background_cleanup()
    logger.info("Background cleanup started")

    yield

    # Shutdown
    await stop_background_cleanup()
    await cache_manager.disconnect()
    await close_http_client()
    logger.info("Shutting down Manga Cover Auto-Tagger Service")


# Create FastAPI app
app = FastAPI(
    title="Manga Cover Auto-Tagger",
    description="""
# Manga Cover Auto-Tagger API

Automatically tag manga covers using Local VLM and RAG.

## Features

- **Multi-Stage Pipeline**: VLM -> RAG -> LLM synthesis
- **Local Models**: No API keys required
- **RAG System**: CLIP embeddings + ChromaDB
- **Manga Description Generation**: Specialized prompts for detailed manga content analysis
- **Real-time Updates**: WebSocket support for async job progress

## Authentication

All write endpoints require an API key in the `X-API-Key` header.

## Rate Limiting

API requests are rate-limited to prevent abuse. See `/api/v1/health` for current limits.

## WebSocket Real-time Updates

Connect to WebSocket at `/api/v1/ws/jobs/{job_id}` to receive real-time progress updates for async tagging jobs.

### JavaScript Client Example

```javascript
// Submit async job
const response = await fetch('http://localhost:8000/api/v1/tag-cover/async', {
    method: 'POST',
    headers: {
        'X-API-Key': 'your-api-key',
    },
    body: formData
});
const { job_id, websocket_url } = await response.json();

// Connect to WebSocket
const ws = new WebSocket(websocket_url);
ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    console.log(`${data.stage}: ${data.progress * 100}% - ${data.message}`);
};
ws.onclose = () => console.log('Job completed');

// WebSocket message format:
// {
//     "job_id": "uuid",
//     "stage": "vlm|rag|recommend|complete",
//     "progress": 0.0-1.0,
//     "message": "Status message",
//     "timestamp": "2024-01-01T00:00:00"
// }
```

### Using wscat

```bash
# Connect to WebSocket
wscat -c ws://localhost:8000/api/v1/ws/jobs/your-job-id

# Submit background job in another terminal
curl -X POST http://localhost:8000/api/v1/tag-cover/async \
  -F "file=@test.jpg" \
  -H "X-API-Key: test"
```

## Error Handling

All errors follow a consistent format:
```json
{
    "error": {
        "code": "ERROR_CODE",
        "message": "Human-readable error message",
        "status": 400
    }
}
```
    """,
    version="2.0.0",
    contact={
        "name": "Manga Tagger Support",
        "url": "https://github.com/your-repo",
    },
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT",
    },
    lifespan=lifespan,
)

# Security check: warn if wildcard origins combined with credentials
_cors_origins = settings.cors_origins
_allow_credentials = True

# Check for insecure wildcard + credentials combination
if _cors_origins == "*" or (isinstance(_cors_origins, list) and "*" in _cors_origins):
    logger.warning(
        "SECURITY WARNING: CORS wildcard origins (*) with allow_credentials=True is insecure. "
        "This configuration allows any origin to access credentials (cookies, auth headers). "
        "Disabling credentials to prevent security vulnerability."
    )
    _allow_credentials = False

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=_allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 效能優化: GZip 壓縮回應
app.add_middleware(GZipMiddleware, minimum_size=500)

# Request ID middleware for distributed tracing (adds X-Request-ID header)
from app.middleware.request_id import RequestIDMiddleware

app.add_middleware(RequestIDMiddleware)

# Security headers middleware (OWASP recommended)
from app.middleware.security import SecurityHeadersMiddleware

app.add_middleware(SecurityHeadersMiddleware)

# Tracing middleware for distributed request tracking
app.add_middleware(TracingMiddleware)

# Metrics middleware for Prometheus
from app.middleware.metrics_middleware import MetricsMiddleware

app.add_middleware(MetricsMiddleware)

# Rate limiting middleware (conditionally enabled)
if settings.RATE_LIMIT_ENABLED:
    from app.middleware.rate_limit import RateLimitMiddleware

    app.add_middleware(RateLimitMiddleware)


# Exception handlers for custom error responses
app.add_exception_handler(AppException, app_exception_handler)
app.add_exception_handler(Exception, general_exception_handler)


# Include canonical API routes
app.include_router(router, prefix="/api/v1")

# Backward-compatible aliases for older clients and scripts still using root paths.
app.include_router(router, include_in_schema=False)


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "service": "Manga Cover Auto-Tagger",
        "version": "2.0.0",
        "docs": "/docs",
        "redoc": "/redoc",
        "openapi": "/openapi.json",
        "api_base": "/api/v1",
        "endpoints": {
            "tag_cover": "/api/v1/tag-cover",
            "tag_cover_async": "/api/v1/tag-cover/async",
            "upload": "/api/v1/upload",
            "generate_manga_description": "/api/v1/generate-manga-description",
            "health": "/api/v1/health",
            "tags": "/api/v1/tags",
            "tags_categories": "/api/v1/tags/categories",
            "rag_add": "/api/v1/rag/add",
            "rag_stats": "/api/v1/rag/stats",
            "websocket": "/api/v1/ws/jobs/{job_id}",
        },
    }
