"""Unit tests for the substring-sensitive post-filter in Stage 10.

Covers spec docs/superpowers/specs/2026-04-23-sensitive-post-filter-design.md:
B (drop unless verified) + R2 (verify_sensitive_tag is the only rescue) +
(a) exact-match path unchanged.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.domain.tag.recommender import TagRecommendation, TagRecommenderService


@pytest.fixture
def patched_settings():
    """Mock settings with a non-empty sensitive set containing only `蘿莉`.

    Both uppercase (used by the existing exact branch via `in`) and lowercase
    (used by the new substring branch via iteration) are populated.
    """
    with patch("app.domain.tag.recommender.settings") as mock:
        mock.SENSITIVE_TAGS = {"蘿莉"}
        mock.sensitive_tags = {"蘿莉"}
        mock.SENSITIVE_SUBSTRING_FILTER_ENABLED = True
        # Calibration / boost knobs - neutralize so we can assert on raw confidence
        mock.TAG_FREQUENCY_CALIBRATION = {}
        mock.EXACT_MATCH_PENALTY = {}
        mock.VISUAL_FEATURE_BOOST = {}
        mock.MIN_ACCEPTABLE_CONFIDENCE = 0.0
        mock.RAG_SUPPORT_BOOST = 1.0
        mock.RAG_SUPPORT_DECAY = 1.0
        mock.SEMANTIC_SIBLINGS = {}
        mock.MUTUAL_EXCLUSIVITY = {}
        mock.TAG_HIERARCHY = {}
        yield mock


def _make_rec(tag: str, confidence: float = 0.8, source: str = "vlm_json"):
    return TagRecommendation(
        tag=tag, confidence=confidence, source=source, reason="test"
    )


@pytest.mark.asyncio
async def test_compound_with_image_and_verify_true_is_kept(patched_settings):
    """巨乳蘿莉 + image_bytes + verify→True → retained, reason marks substring-verified."""
    svc = TagRecommenderService()
    vlm_service = MagicMock()
    vlm_service.verify_sensitive_tag = AsyncMock(return_value=True)

    out = await svc._verify_and_calibrate(
        recommendations=[_make_rec("巨乳蘿莉")],
        vlm_service=vlm_service,
        image_bytes=b"fake-image-bytes",
        rag_matches=[],
        vlm_analysis={},
    )

    assert any(r.tag == "巨乳蘿莉" for r in out)
    kept = next(r for r in out if r.tag == "巨乳蘿莉")
    assert "substring-verified" in kept.reason
    vlm_service.verify_sensitive_tag.assert_awaited_once_with(
        b"fake-image-bytes", "巨乳蘿莉"
    )


@pytest.mark.asyncio
async def test_compound_with_image_and_verify_false_is_dropped(patched_settings):
    """巨乳蘿莉 + image_bytes + verify→False → dropped."""
    svc = TagRecommenderService()
    vlm_service = MagicMock()
    vlm_service.verify_sensitive_tag = AsyncMock(return_value=False)

    out = await svc._verify_and_calibrate(
        recommendations=[_make_rec("巨乳蘿莉")],
        vlm_service=vlm_service,
        image_bytes=b"fake-image-bytes",
        rag_matches=[],
        vlm_analysis={},
    )

    assert all(r.tag != "巨乳蘿莉" for r in out)


@pytest.mark.asyncio
async def test_compound_without_image_bytes_is_dropped(patched_settings):
    """巨乳蘿莉 + no image_bytes → dropped (no rescue path available)."""
    svc = TagRecommenderService()

    out = await svc._verify_and_calibrate(
        recommendations=[_make_rec("巨乳蘿莉")],
        vlm_service=None,
        image_bytes=None,
        rag_matches=[],
        vlm_analysis={},
    )

    assert all(r.tag != "巨乳蘿莉" for r in out)


@pytest.mark.asyncio
async def test_exact_singleton_without_image_bytes_is_kept_with_penalty(
    patched_settings,
):
    """蘿莉 (exact match) + no image_bytes → kept, penalized ×0.7 (decision (a))."""
    svc = TagRecommenderService()

    out = await svc._verify_and_calibrate(
        recommendations=[_make_rec("蘿莉", confidence=0.8)],
        vlm_service=None,
        image_bytes=None,
        rag_matches=[],
        vlm_analysis={},
    )

    kept = next((r for r in out if r.tag == "蘿莉"), None)
    assert kept is not None
    # 0.8 * 0.7 = 0.56; safe_confidence rounds, so allow tolerance
    assert 0.55 <= kept.confidence <= 0.57
    assert "Unverified" in kept.reason


@pytest.mark.asyncio
async def test_compound_with_flag_disabled_is_kept(patched_settings):
    """Rollback path: flag=False → 巨乳蘿莉 retained even without verifier."""
    patched_settings.SENSITIVE_SUBSTRING_FILTER_ENABLED = False
    svc = TagRecommenderService()

    out = await svc._verify_and_calibrate(
        recommendations=[_make_rec("巨乳蘿莉")],
        vlm_service=None,
        image_bytes=None,
        rag_matches=[],
        vlm_analysis={},
    )

    assert any(r.tag == "巨乳蘿莉" for r in out)
