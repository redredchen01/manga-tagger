"""When RAG_INFLUENCE_ENABLED is False, RAG matches must not flow into
the recommender (no RAG -> tags), even if their score is high. RAG
metadata is still preserved in the response for visibility."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.mark.asyncio
async def test_rag_tags_excluded_when_disabled(monkeypatch):
    from app.domain import pipeline as pipemod
    from app.core.config import settings

    monkeypatch.setattr(settings, "RAG_INFLUENCE_ENABLED", False)

    fake_vlm = MagicMock()
    fake_vlm.extract_metadata = AsyncMock(
        return_value={
            "description": "x",
            "tags": [
                {
                    "tag": "貓娘",
                    "category": "character",
                    "confidence": 0.9,
                    "evidence": "ears",
                }
            ],
            "source": "vlm_json",
        }
    )
    fake_rag = MagicMock()
    fake_rag.search_similar = AsyncMock(
        return_value=[
            {
                "id": "x",
                "score": 0.99,
                "tags": ["catgirl", "rag_only_tag"],
                "metadata": {},
            }
        ]
    )
    fake_recommender = MagicMock()
    fake_recommender.tag_library = MagicMock()
    fake_recommender.tag_library.tag_names = {"貓娘"}
    fake_recommender.recommend_tags = AsyncMock(return_value=[])

    # Bypass image validation and circuit breakers / rate limiters
    monkeypatch.setattr(pipemod, "validate_image", lambda _b: (True, None))

    class _NoopLimiter:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

    class _PassthroughCB:
        async def call(self, func, *args, fallback=None, **kwargs):
            return await func(*args, **kwargs)

    monkeypatch.setattr(pipemod, "get_circuit_breaker", lambda _n: _PassthroughCB())
    monkeypatch.setattr(pipemod, "get_rate_limiter", lambda _n: _NoopLimiter())

    image_bytes = b"\xff\xd8" + b"\x00" * 4096
    result = await pipemod.run_tagging_pipeline(
        image_bytes=image_bytes,
        vlm_service=fake_vlm,
        rag_service=fake_rag,
        tag_recommender=fake_recommender,
    )

    call_kwargs = fake_recommender.recommend_tags.call_args.kwargs
    rag_matches_passed = call_kwargs.get("rag_matches", [])
    assert rag_matches_passed == [], (
        f"RAG matches should be empty when RAG_INFLUENCE_ENABLED=False, "
        f"got {rag_matches_passed}"
    )
    # Metadata still reports what RAG found (informational only)
    assert result.metadata.get("rag_matches_count") == 1


@pytest.mark.asyncio
async def test_rag_tags_included_when_enabled(monkeypatch):
    from app.domain import pipeline as pipemod
    from app.core.config import settings

    monkeypatch.setattr(settings, "RAG_INFLUENCE_ENABLED", True)

    fake_vlm = MagicMock()
    fake_vlm.extract_metadata = AsyncMock(
        return_value={"description": "x", "tags": [], "source": "vlm_json"}
    )
    fake_rag = MagicMock()
    fake_rag.search_similar = AsyncMock(
        return_value=[
            {"id": "x", "score": 0.99, "tags": ["catgirl"], "metadata": {}}
        ]
    )
    fake_recommender = MagicMock()
    fake_recommender.tag_library = MagicMock()
    fake_recommender.tag_library.tag_names = set()
    fake_recommender.recommend_tags = AsyncMock(return_value=[])

    monkeypatch.setattr(pipemod, "validate_image", lambda _b: (True, None))

    class _NoopLimiter:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

    class _PassthroughCB:
        async def call(self, func, *args, fallback=None, **kwargs):
            return await func(*args, **kwargs)

    monkeypatch.setattr(pipemod, "get_circuit_breaker", lambda _n: _PassthroughCB())
    monkeypatch.setattr(pipemod, "get_rate_limiter", lambda _n: _NoopLimiter())

    image_bytes = b"\xff\xd8" + b"\x00" * 4096
    await pipemod.run_tagging_pipeline(
        image_bytes=image_bytes,
        vlm_service=fake_vlm,
        rag_service=fake_rag,
        tag_recommender=fake_recommender,
    )

    call_kwargs = fake_recommender.recommend_tags.call_args.kwargs
    rag_matches_passed = call_kwargs.get("rag_matches", [])
    assert len(rag_matches_passed) == 1
