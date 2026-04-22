from __future__ import annotations

import sys
from pathlib import Path

from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.api import routes_v2
from app.main import app

# Use TestClient as context manager to trigger lifespan (populates app.state)
client = TestClient(app)


def reset_route_singletons() -> None:
    routes_v2._vlm_service = None
    routes_v2._llm_service = None
    routes_v2._rag_service = None
    routes_v2._tag_recommender = None


def test_health_is_lightweight_and_does_not_initialize_heavy_services() -> None:
    reset_route_singletons()

    response = client.get("/api/v1/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "healthy"
    assert payload["models_loaded"]["vlm_initialized"] is False
    assert payload["models_loaded"]["llm_initialized"] is False
    assert payload["models_loaded"]["rag_initialized"] is False
    assert payload["models_loaded"]["tag_library"] >= 1
    assert routes_v2._vlm_service is None
    assert routes_v2._llm_service is None
    assert routes_v2._rag_service is None


def test_compatibility_routes_remain_available() -> None:
    assert client.get("/health").status_code == 200
    assert client.get("/api/v1/health").status_code == 200
    assert client.get("/tags").status_code == 200
    assert client.get("/api/v1/tags").status_code == 200


def test_root_endpoint_points_to_canonical_api_paths() -> None:
    response = client.get("/")

    assert response.status_code == 200
    payload = response.json()
    assert payload["api_base"] == "/api/v1"
    assert payload["endpoints"]["tag_cover"] == "/api/v1/tag-cover"
    assert payload["endpoints"]["health"] == "/api/v1/health"


def test_tag_cover_rejects_tiny_file() -> None:
    response = client.post(
        "/api/v1/tag-cover",
        files={"file": ("tiny.jpg", b"123", "image/jpeg")},
        data={"top_k": "5", "confidence_threshold": "0.5", "include_metadata": "true"},
    )

    assert response.status_code == 400
    assert response.json()["error"]["message"] == "File too small. Must be at least 1KB"


def test_tag_cover_rejects_non_image_content_type() -> None:
    response = client.post(
        "/api/v1/tag-cover",
        files={"file": ("bad.txt", b"not an image", "text/plain")},
        data={"top_k": "5", "confidence_threshold": "0.5", "include_metadata": "true"},
    )

    assert response.status_code == 400
    assert (
        response.json()["error"]["message"]
        == "Invalid file type. Must be an image. Got: text/plain"
    )
