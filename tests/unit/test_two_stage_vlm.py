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


@pytest.mark.asyncio
async def test_select_tags_parses_valid_json(monkeypatch):
    """_select_tags_from_description returns parsed tags list."""
    service = LMStudioVLMService()

    json_body = '{"tags": [{"tag": "貓娘", "confidence": 0.9, "evidence": "貓耳"}]}'
    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=_http_response(json_body))
    monkeypatch.setattr(
        "app.infrastructure.lm_studio.vlm_service.get_http_client",
        AsyncMock(return_value=mock_client),
    )

    tags = await service._select_tags_from_description(
        "一個有貓耳的少女。", "### 角色\n貓娘, 蘿莉"
    )
    assert len(tags) == 1
    assert tags[0]["tag"] == "貓娘"
    assert tags[0]["confidence"] == 0.9


@pytest.mark.asyncio
async def test_select_tags_returns_empty_on_http_error(monkeypatch):
    """_select_tags_from_description returns [] when HTTP raises."""
    service = LMStudioVLMService()

    monkeypatch.setattr(
        "app.infrastructure.lm_studio.vlm_service.get_http_client",
        AsyncMock(side_effect=Exception("network error")),
    )

    tags = await service._select_tags_from_description("desc", "fragment")
    assert tags == []


@pytest.mark.asyncio
async def test_select_tags_retries_once_on_parse_failure(monkeypatch):
    """_select_tags_from_description retries once when first response is not valid JSON."""
    service = LMStudioVLMService()

    valid_json = '{"tags": [{"tag": "雙馬尾", "confidence": 0.85, "evidence": "clearly visible"}]}'
    mock_client = AsyncMock()
    mock_client.post = AsyncMock(side_effect=[
        _http_response("Sorry I cannot answer that."),  # first call: garbage
        _http_response(valid_json),                      # retry: valid JSON
    ])
    monkeypatch.setattr(
        "app.infrastructure.lm_studio.vlm_service.get_http_client",
        AsyncMock(return_value=mock_client),
    )

    tags = await service._select_tags_from_description("雙馬尾少女", "### 身體特徵\n雙馬尾")
    assert any(t["tag"] == "雙馬尾" for t in tags)
    assert mock_client.post.call_count == 2


@pytest.mark.asyncio
async def test_select_tags_returns_empty_when_choices_empty(monkeypatch):
    """_select_tags_from_description returns [] when model returns empty choices."""
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

    tags = await service._select_tags_from_description("desc", "fragment")
    assert tags == []
