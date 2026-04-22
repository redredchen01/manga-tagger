#!/usr/bin/env python3
"""One-command smoke test for the Manga Tagger API."""

from __future__ import annotations

import sys
from pathlib import Path

from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.main import app


client = TestClient(app)


def check(name: str, ok: bool, detail: str = "") -> bool:
    marker = "PASS" if ok else "FAIL"
    suffix = f" - {detail}" if detail else ""
    print(f"[{marker}] {name}{suffix}")
    return ok


def main() -> int:
    results: list[bool] = []

    health = client.get("/api/v1/health")
    payload = health.json() if health.headers.get("content-type", "").startswith("application/json") else {}
    results.append(check("GET /api/v1/health", health.status_code == 200, str(health.status_code)))
    results.append(check("health status", payload.get("status") == "healthy", payload.get("status", "missing")))
    results.append(check("health is lightweight", payload.get("models_loaded", {}).get("rag_initialized") is False))

    root = client.get("/")
    root_payload = root.json()
    results.append(check("root api_base", root_payload.get("api_base") == "/api/v1", root_payload.get("api_base", "missing")))

    compat_health = client.get("/health")
    compat_tags = client.get("/tags")
    results.append(check("compat /health", compat_health.status_code == 200, str(compat_health.status_code)))
    results.append(check("compat /tags", compat_tags.status_code == 200, str(compat_tags.status_code)))

    tiny = client.post(
        "/api/v1/tag-cover",
        files={"file": ("tiny.jpg", b"123", "image/jpeg")},
        data={"top_k": "5", "confidence_threshold": "0.5", "include_metadata": "true"},
    )
    results.append(check("tiny image rejected", tiny.status_code == 400, tiny.json().get("detail", "")))

    bad_type = client.post(
        "/api/v1/tag-cover",
        files={"file": ("bad.txt", b"not an image", "text/plain")},
        data={"top_k": "5", "confidence_threshold": "0.5", "include_metadata": "true"},
    )
    results.append(check("non-image rejected", bad_type.status_code == 400, bad_type.json().get("detail", "")))

    tag_image = PROJECT_ROOT / "test_real_image.jpg"
    if tag_image.exists():
        with tag_image.open("rb") as fh:
            tagged = client.post(
                "/api/v1/tag-cover",
                files={"file": (tag_image.name, fh.read(), "image/jpeg")},
                data={"top_k": "5", "confidence_threshold": "0.5", "include_metadata": "true"},
            )
        tagged_payload = tagged.json() if tagged.status_code == 200 else {}
        results.append(check("real image tagging", tagged.status_code == 200, str(tagged.status_code)))
        results.append(check("real image returned tags", len(tagged_payload.get("tags", [])) > 0, str(len(tagged_payload.get("tags", [])))))
    else:
        results.append(check("real image fixture present", False, str(tag_image)))

    passed = sum(1 for item in results if item)
    total = len(results)
    print()
    print(f"Summary: {passed}/{total} checks passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
