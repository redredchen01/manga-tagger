"""Verify that hedge phrases never appear as final tags.

The system used to extract keywords from the VLM's free-form description,
which sometimes contained hedge phrases like '需要更多視覺證據'. After
Phase 1, only the structured JSON `tags` field is consumed -- these
strings should never appear in final output."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.domain.tag.recommender import TagRecommenderService


HEDGE_FRAGMENTS = ["需要更多", "證據不足", "可能是", "似乎", "failed", "error"]


@pytest.mark.asyncio
async def test_hedge_strings_never_become_tags():
    # Simulate a vlm_analysis from the new JSON path: explicit `tags`
    # array with curated names; description contains hedge text.
    vlm_analysis = {
        "description": "畫面中可能是少女，需要更多視覺證據才能確認年齡。",
        "tags": [
            {"tag": "雙馬尾", "category": "body", "confidence": 0.9, "evidence": "obvious"},
        ],
        "source": "vlm_json",
    }

    # Mock dependencies
    tag_lib = MagicMock()
    tag_lib.tag_names = {"雙馬尾", "貓娘", "蘿莉"}
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
        service = TagRecommenderService()

    # Patch private methods that hit external services so this is a unit test
    service._refine_with_llm = AsyncMock(side_effect=lambda recs, *a, **kw: recs)
    service._verify_and_calibrate = AsyncMock(side_effect=lambda recs, *a, **kw: recs)

    recs = await service.recommend_tags(
        vlm_analysis=vlm_analysis,
        rag_matches=[],
        top_k=5,
        confidence_threshold=0.3,
    )

    final_tags = [r.tag for r in recs]
    # No hedge fragment should appear in any tag
    for tag in final_tags:
        for hedge in HEDGE_FRAGMENTS:
            assert hedge not in tag, f"hedge {hedge!r} leaked into tag {tag!r}"
    # The legitimate JSON tag should be present
    assert "雙馬尾" in final_tags
