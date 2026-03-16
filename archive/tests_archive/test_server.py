#!/usr/bin/env python3
"""
Simple test server to verify connectivity
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Manga Tagger Test Server", description="Simple test server for connectivity"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {
        "status": "Test Server Running",
        "message": "Manga Tagger Test Server",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
async def health():
    return {"status": "healthy", "version": "test-1.0.0", "test_mode": True}


@app.get("/docs")
async def docs():
    return {
        "message": "Test server documentation",
        "endpoints": ["/", "/health", "/docs"],
    }


if __name__ == "__main__":
    import uvicorn

    print("Starting Test Server on http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
