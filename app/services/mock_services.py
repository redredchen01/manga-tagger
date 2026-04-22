"""Mock services for testing without heavy ML models."""

import asyncio
import uuid
from typing import Any, Dict, List, Optional

from app.models import VLMMetadata


class MockVLMService:
    """Mock VLM service for testing."""

    async def extract_metadata(self, image_bytes: bytes):
        """Mock metadata extraction.

        Returns a dict to match what the pipeline expects.
        """
        await asyncio.sleep(0.1)  # Simulate processing time
        # Return a dict (not Pydantic model) to match pipeline expectations
        return {
            "description": "Mock analysis of manga cover image",
            "character_types": ["mock_character"],
            "clothing": ["school_uniform"],
            "body_features": [],
            "actions": [],
            "themes": ["mock_theme"],
            "art_style": "mock_style",
            "genre_indicators": ["mock_genre"],
            "raw_keywords": ["catgirl", "school_uniform"],
        }


class MockLLMService:
    """Mock LLM service for testing."""

    async def generate_response(self, prompt: str) -> str:
        """Mock response generation."""
        await asyncio.sleep(0.1)  # Simulate processing time
        return f"Mock response to: {prompt[:50]}..."


class MockRAGService:
    """Mock RAG service for testing."""

    def __init__(self):
        """Initialize mock RAG service."""
        self.documents = []

    async def search_similar(self, image_bytes: bytes, top_k: int = 5) -> List[Dict[str, Any]]:
        """Mock similarity search."""
        await asyncio.sleep(0.1)  # Simulate processing time
        return [
            {
                "id": f"mock_doc_{i}",
                "score": 0.9 - (i * 0.1),
                "tags": [f"mock_tag_{i}", f"mock_category_{i}"],
                "metadata": {"mock": True},
            }
            for i in range(min(top_k, 3))
        ]

    async def add_image(
        self,
        image_bytes: bytes,
        tags: List[str],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Mock image addition."""
        await asyncio.sleep(0.1)  # Simulate processing time
        doc_id = str(uuid.uuid4())
        self.documents.append({"id": doc_id, "tags": tags, "metadata": metadata or {}})
        return doc_id

    async def get_stats(self) -> Dict[str, Any]:
        """Mock stats."""
        return {
            "total_documents": len(self.documents),
            "collection_name": "mock_collection",
            "embedding_model": "mock_model",
        }
