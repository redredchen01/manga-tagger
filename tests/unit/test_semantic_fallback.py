"""Semantic search is a second-tier fallback. In the full pipeline,
Stage 1b's description_rescue already runs first and may top up
`current_recs` before `_search_semantic` is called. These tests
exercise `_search_semantic` in isolation, so the unit-level trigger
(current_recs count vs SEMANTIC_FALLBACK_TRIGGER_COUNT) is unchanged."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.domain.tag.recommender import TagRecommendation, TagRecommenderService


def _make_recs(n):
    return [
        TagRecommendation(tag=f"tag{i}", confidence=0.8, source="vlm_json", reason="x")
        for i in range(n)
    ]


def _build_service(tag_names):
    tag_lib = MagicMock()
    tag_lib.tag_names = set(tag_names)
    tag_lib.tags = []
    tag_lib.get_all_tags = MagicMock(return_value=list(tag_names))
    tag_lib.match_tags_by_keywords = MagicMock(return_value=[])
    tag_lib.suggest_related_tags = MagicMock(return_value=[])

    tag_mapper = MagicMock()
    tag_mapper.to_chinese = MagicMock(return_value=None)

    with patch(
        "app.domain.tag.recommender.get_tag_library_service",
        return_value=tag_lib,
    ), patch(
        "app.domain.tag.recommender.get_tag_mapper",
        return_value=tag_mapper,
    ):
        return TagRecommenderService()


@pytest.mark.asyncio
async def test_semantic_skipped_when_vlm_has_enough(monkeypatch):
    service = _build_service({f"tag{i}" for i in range(5)})

    fake_embed = MagicMock()
    fake_embed.is_available = MagicMock(return_value=True)
    fake_embed._tag_matrix_cache = "not-none"
    fake_embed.search_cached_tags = AsyncMock(return_value=[])

    monkeypatch.setattr(
        "app.services.chinese_embedding_service.get_chinese_embedding_service",
        lambda: fake_embed,
    )
    monkeypatch.setattr("app.core.config.settings.USE_MOCK_SERVICES", False)

    # 5 VLM tags — above trigger count (3), embedding must not be queried
    recs = _make_recs(5)
    result = await service._search_semantic(["kw1", "kw2"], recs, top_k=10)

    fake_embed.search_cached_tags.assert_not_called()
    assert len(result) == 5


@pytest.mark.asyncio
async def test_semantic_triggered_when_vlm_under_delivers(monkeypatch):
    service = _build_service({"tag0", "extra"})

    fake_embed = MagicMock()
    fake_embed.is_available = MagicMock(return_value=True)
    fake_embed._tag_matrix_cache = "not-none"
    fake_embed.search_cached_tags = AsyncMock(
        return_value=[{"tag": "extra", "similarity": 0.82}]
    )
    fake_embed.cache_tag_embeddings = AsyncMock()

    monkeypatch.setattr(
        "app.services.chinese_embedding_service.get_chinese_embedding_service",
        lambda: fake_embed,
    )
    monkeypatch.setattr("app.core.config.settings.USE_MOCK_SERVICES", False)

    # 1 VLM tag — below trigger count (3), should call embedding
    recs = _make_recs(1)
    result = await service._search_semantic(["kw1"], recs, top_k=10)

    fake_embed.search_cached_tags.assert_called()
    tags = [r.tag for r in result]
    assert "extra" in tags
    added = next(r for r in result if r.tag == "extra")
    assert added.source == "semantic_fallback"


@pytest.mark.asyncio
async def test_semantic_cap_on_additions(monkeypatch):
    service = _build_service({"a", "b", "c", "d"})

    fake_embed = MagicMock()
    fake_embed.is_available = MagicMock(return_value=True)
    fake_embed._tag_matrix_cache = "not-none"
    # Return many candidates so we can observe the cap
    fake_embed.search_cached_tags = AsyncMock(
        return_value=[
            {"tag": "a", "similarity": 0.90},
            {"tag": "b", "similarity": 0.88},
            {"tag": "c", "similarity": 0.85},
            {"tag": "d", "similarity": 0.82},
        ]
    )
    fake_embed.cache_tag_embeddings = AsyncMock()

    monkeypatch.setattr(
        "app.services.chinese_embedding_service.get_chinese_embedding_service",
        lambda: fake_embed,
    )
    monkeypatch.setattr("app.core.config.settings.USE_MOCK_SERVICES", False)
    monkeypatch.setattr("app.core.config.settings.SEMANTIC_FALLBACK_MAX_ADDITIONS", 2)

    recs = _make_recs(0)  # force trigger
    result = await service._search_semantic(["kw"], recs, top_k=10)

    # Only 2 additions despite 4 candidates
    assert len(result) == 2
    assert all(r.source == "semantic_fallback" for r in result)
