"""Stage 8 LLM refinement must be skipped when VLM JSON path succeeded.

Regression guard: previously `synthesize_tags` was invoked unconditionally
after VLM JSON produced tags. With qwen3.6 returning 400 on the legacy
synthesis prompt, the refinement step wiped out the VLM's legitimate tags
and produced 0 recommendations. Per spec §3.4 / §3.7, VLM JSON is
authoritative; LLM refinement only runs on the legacy (non-JSON) fallback."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.domain.tag.recommender import TagRecommenderService


def _build_service():
    tag_lib = MagicMock()
    tag_lib.tag_names = {"雙馬尾", "貓娘"}
    tag_lib.tags = []
    tag_lib.match_tags_by_keywords = MagicMock(return_value=[])
    tag_lib.suggest_related_tags = MagicMock(return_value=[])
    tag_lib.get_all_tags = MagicMock(return_value=[])

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
async def test_llm_refine_skipped_when_vlm_json_path(monkeypatch):
    service = _build_service()

    # Force LM Studio path "enabled" so the only gate is the JSON path flag
    monkeypatch.setattr("app.core.config.settings.USE_LM_STUDIO", True)
    monkeypatch.setattr("app.core.config.settings.USE_MOCK_SERVICES", False)

    service._refine_with_llm = AsyncMock(side_effect=lambda recs, *a, **kw: [])
    service._verify_and_calibrate = AsyncMock(side_effect=lambda recs, *a, **kw: recs)

    vlm_analysis = {
        "description": "anime girl",
        "tags": [
            {"tag": "雙馬尾", "category": "body", "confidence": 0.9, "evidence": "obvious"},
            {"tag": "貓娘", "category": "character", "confidence": 0.85, "evidence": "cat ears"},
        ],
        "source": "vlm_json",
    }

    recs = await service.recommend_tags(
        vlm_analysis=vlm_analysis,
        rag_matches=[],
        top_k=5,
        confidence_threshold=0.3,
    )

    service._refine_with_llm.assert_not_called()
    final_tags = {r.tag for r in recs}
    assert "雙馬尾" in final_tags
    assert "貓娘" in final_tags


@pytest.mark.asyncio
async def test_llm_refine_still_runs_on_legacy_path(monkeypatch):
    service = _build_service()

    monkeypatch.setattr("app.core.config.settings.USE_LM_STUDIO", True)
    monkeypatch.setattr("app.core.config.settings.USE_MOCK_SERVICES", False)

    service._refine_with_llm = AsyncMock(side_effect=lambda recs, *a, **kw: recs)
    service._verify_and_calibrate = AsyncMock(side_effect=lambda recs, *a, **kw: recs)

    # No `tags` array -> legacy path -> refine should be called
    vlm_analysis = {
        "description": "some description",
        "character_types": ["catgirl"],
    }

    await service.recommend_tags(
        vlm_analysis=vlm_analysis,
        rag_matches=[],
        top_k=5,
        confidence_threshold=0.3,
    )

    service._refine_with_llm.assert_called_once()
