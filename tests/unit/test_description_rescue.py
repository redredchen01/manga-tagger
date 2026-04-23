"""Stage 1b: VLM description is always embedded via bge-m3 and cosine-matched
against the cached 611-tag matrix. Results are merged as secondary candidates."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.domain.tag.recommender import TagRecommenderService


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
        "app.domain.tag.recommender.get_tag_library_service", return_value=tag_lib
    ), patch(
        "app.domain.tag.recommender.get_tag_mapper", return_value=tag_mapper
    ):
        return TagRecommenderService()


def _patch_embedding(monkeypatch, matches):
    """Install a fake embedding service that returns `matches`."""
    fake_embed = MagicMock()
    fake_embed.is_available = MagicMock(return_value=True)
    fake_embed._tag_matrix_cache = "not-none"
    fake_embed.search_cached_tags = AsyncMock(return_value=matches)
    fake_embed.cache_tag_embeddings = AsyncMock()
    monkeypatch.setattr(
        "app.services.chinese_embedding_service.get_chinese_embedding_service",
        lambda: fake_embed,
    )
    monkeypatch.setattr("app.core.config.settings.USE_MOCK_SERVICES", False)
    return fake_embed


@pytest.mark.asyncio
async def test_rescue_skipped_when_description_empty(monkeypatch):
    service = _build_service({"雙馬尾", "貓娘"})
    fake_embed = _patch_embedding(monkeypatch, [])
    service._search_semantic = AsyncMock(side_effect=lambda kws, recs, k: recs)
    service._refine_with_llm = AsyncMock(side_effect=lambda recs, *a, **kw: recs)
    service._verify_and_calibrate = AsyncMock(side_effect=lambda recs, *a, **kw: recs)

    vlm_analysis = {
        "description": "",  # empty -> rescue must be skipped
        "tags": [{"tag": "雙馬尾", "category": "body", "confidence": 0.9, "evidence": "x"}],
        "source": "vlm_json",
    }
    await service.recommend_tags(vlm_analysis=vlm_analysis, rag_matches=[], top_k=5)
    fake_embed.search_cached_tags.assert_not_called()


@pytest.mark.asyncio
async def test_rescue_becomes_main_when_vlm_empty(monkeypatch):
    service = _build_service({"雙馬尾", "貓娘", "少女"})
    _patch_embedding(monkeypatch, [
        {"tag": "少女", "similarity": 0.82},
        {"tag": "雙馬尾", "similarity": 0.78},
    ])
    service._search_semantic = AsyncMock(side_effect=lambda kws, recs, k: recs)
    service._refine_with_llm = AsyncMock(side_effect=lambda recs, *a, **kw: recs)
    service._verify_and_calibrate = AsyncMock(side_effect=lambda recs, *a, **kw: recs)

    vlm_analysis = {
        "description": "一個綁雙馬尾的少女,穿著制服在校園。",
        "tags": [],  # VLM gave nothing
        "source": "vlm_json",
    }
    recs = await service.recommend_tags(
        vlm_analysis=vlm_analysis, rag_matches=[], top_k=5, confidence_threshold=0.3
    )
    tags = {r.tag for r in recs}
    assert "少女" in tags
    assert "雙馬尾" in tags
    for r in recs:
        if r.tag in {"少女", "雙馬尾"}:
            assert r.source == "description_rescue"


@pytest.mark.asyncio
async def test_rescue_caps_at_2_when_vlm_delivered(monkeypatch):
    service = _build_service({"雙馬尾", "貓娘", "少女", "校服", "微笑", "室內"})
    _patch_embedding(monkeypatch, [
        {"tag": "少女", "similarity": 0.80},
        {"tag": "校服", "similarity": 0.78},
        {"tag": "微笑", "similarity": 0.76},
        {"tag": "室內", "similarity": 0.72},
    ])
    service._search_semantic = AsyncMock(side_effect=lambda kws, recs, k: recs)
    service._refine_with_llm = AsyncMock(side_effect=lambda recs, *a, **kw: recs)
    service._verify_and_calibrate = AsyncMock(side_effect=lambda recs, *a, **kw: recs)

    vlm_analysis = {
        "description": "一個雙馬尾少女在教室,微笑中。",
        "tags": [
            {"tag": "雙馬尾", "category": "body", "confidence": 0.9, "evidence": "x"},
            {"tag": "貓娘", "category": "character", "confidence": 0.85, "evidence": "y"},
            {"tag": "微笑", "category": "action", "confidence": 0.8, "evidence": "z"},
        ],
        "source": "vlm_json",
    }
    recs = await service.recommend_tags(
        vlm_analysis=vlm_analysis, rag_matches=[], top_k=10, confidence_threshold=0.3
    )
    rescue_sources = [r for r in recs if r.source == "description_rescue"]
    assert len(rescue_sources) <= 2, f"rescue must cap at 2 additions, got {len(rescue_sources)}"


@pytest.mark.asyncio
async def test_rescue_dual_source_agreement_boosts_vlm_confidence(monkeypatch):
    service = _build_service({"雙馬尾", "貓娘"})
    _patch_embedding(monkeypatch, [
        {"tag": "雙馬尾", "similarity": 0.85},  # also in VLM's list
    ])
    service._search_semantic = AsyncMock(side_effect=lambda kws, recs, k: recs)
    service._refine_with_llm = AsyncMock(side_effect=lambda recs, *a, **kw: recs)
    service._verify_and_calibrate = AsyncMock(side_effect=lambda recs, *a, **kw: recs)

    vlm_analysis = {
        "description": "雙馬尾少女",
        "tags": [
            {"tag": "雙馬尾", "category": "body", "confidence": 0.7, "evidence": "obvious"},
            {"tag": "貓娘", "category": "character", "confidence": 0.7, "evidence": "ears"},
            {"tag": "微笑", "category": "action", "confidence": 0.7, "evidence": "mouth"},
        ],
        "source": "vlm_json",
    }
    recs = await service.recommend_tags(
        vlm_analysis=vlm_analysis, rag_matches=[], top_k=10, confidence_threshold=0.3
    )
    vlm_twintail = next(r for r in recs if r.tag == "雙馬尾")
    # 0.7 baseline + 0.10 agreement boost = 0.80 (after safe_confidence)
    assert vlm_twintail.confidence >= 0.79, (
        f"expected +0.10 boost, got confidence={vlm_twintail.confidence}"
    )
    assert "+desc agreement" in vlm_twintail.reason


@pytest.mark.asyncio
async def test_rescue_survives_embedding_service_failure(monkeypatch):
    service = _build_service({"雙馬尾"})

    def _broken():
        raise ImportError("simulated")
    monkeypatch.setattr(
        "app.services.chinese_embedding_service.get_chinese_embedding_service",
        _broken,
    )
    monkeypatch.setattr("app.core.config.settings.USE_MOCK_SERVICES", False)
    service._search_semantic = AsyncMock(side_effect=lambda kws, recs, k: recs)
    service._refine_with_llm = AsyncMock(side_effect=lambda recs, *a, **kw: recs)
    service._verify_and_calibrate = AsyncMock(side_effect=lambda recs, *a, **kw: recs)

    vlm_analysis = {
        "description": "雙馬尾少女",
        "tags": [{"tag": "雙馬尾", "category": "body", "confidence": 0.9, "evidence": "x"}],
        "source": "vlm_json",
    }
    # Should not raise
    recs = await service.recommend_tags(
        vlm_analysis=vlm_analysis, rag_matches=[], top_k=5, confidence_threshold=0.3
    )
    assert any(r.tag == "雙馬尾" for r in recs)
