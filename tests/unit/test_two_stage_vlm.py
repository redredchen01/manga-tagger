"""Unit tests for the two-stage VLM pipeline.

Stage 1: image → description (vision call)
Stage 2: description + allowed list → tags JSON (text-only call)
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.infrastructure.lm_studio.vlm_service import LMStudioVLMService


def _http_response(content: str):
    """Minimal synchronous mock for an httpx response returning `content`."""
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json = MagicMock(return_value={
        "choices": [{"message": {"content": content, "reasoning_content": ""}}]
    })
    return resp


@pytest.mark.asyncio
async def test_extract_description_returns_stripped_text(monkeypatch):
    """_extract_description returns the model's plain text response, stripped."""
    service = LMStudioVLMService()

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(
        return_value=_http_response("  一個藍髮少女穿著女生制服，坐在教室中。  ")
    )
    monkeypatch.setattr(
        "app.infrastructure.lm_studio.vlm_service.get_http_client",
        AsyncMock(return_value=mock_client),
    )
    monkeypatch.setattr(
        "app.infrastructure.lm_studio.vlm_service.LMStudioVLMService._prepare_image",
        MagicMock(return_value=b"prepared"),
    )

    result = await service._extract_description(b"fake-image-bytes")
    assert result == "一個藍髮少女穿著女生制服，坐在教室中。"


@pytest.mark.asyncio
async def test_extract_description_returns_empty_on_http_error(monkeypatch):
    """_extract_description returns '' when HTTP raises."""
    service = LMStudioVLMService()

    monkeypatch.setattr(
        "app.infrastructure.lm_studio.vlm_service.get_http_client",
        AsyncMock(side_effect=Exception("connection refused")),
    )
    monkeypatch.setattr(
        "app.infrastructure.lm_studio.vlm_service.LMStudioVLMService._prepare_image",
        MagicMock(return_value=b"prepared"),
    )

    result = await service._extract_description(b"fake-image-bytes")
    assert result == ""


@pytest.mark.asyncio
async def test_extract_description_returns_empty_when_choices_empty(monkeypatch):
    """_extract_description returns '' when model returns empty choices."""
    service = LMStudioVLMService()

    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json = MagicMock(return_value={"choices": []})
    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=resp)
    monkeypatch.setattr(
        "app.infrastructure.lm_studio.vlm_service.get_http_client",
        AsyncMock(return_value=mock_client),
    )
    monkeypatch.setattr(
        "app.infrastructure.lm_studio.vlm_service.LMStudioVLMService._prepare_image",
        MagicMock(return_value=b"prepared"),
    )

    result = await service._extract_description(b"fake-image-bytes")
    assert result == ""
