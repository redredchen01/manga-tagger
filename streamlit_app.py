#!/usr/bin/env python3
"""Clean Streamlit frontend for Manga Tagger."""

from __future__ import annotations

import io
import json
import os
import socket
from typing import Any

import requests
import streamlit as st
from PIL import Image

# Default API base URL - can be overridden via environment variable
# Fallback to config settings if app is available, otherwise use env vars
DEFAULT_API_BASE = os.environ.get(
    "API_BASE_URL",
    f"http://{os.environ.get('API_HOST', '127.0.0.1')}:{os.environ.get('API_PORT', '8000')}/api/v1",
)
REQUEST_TIMEOUT = 30
TAG_TIMEOUT = 90


def is_private_ip(ip: str) -> bool:
    try:
        parts = ip.split(".")
        if len(parts) != 4:
            return False
        first = int(parts[0])
        second = int(parts[1])
        return (
            first == 10
            or first == 127
            or (first == 172 and 16 <= second <= 31)
            or (first == 192 and second == 168)
        )
    except ValueError:
        return False


def get_local_ip() -> str:
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.connect(("8.8.8.8", 80))
        ip = sock.getsockname()[0]
        sock.close()
        return ip
    except OSError:
        return "127.0.0.1"


def normalize_api_base(url: str) -> str:
    cleaned = (url or "").strip().rstrip("/")
    if not cleaned:
        return DEFAULT_API_BASE
    if cleaned.endswith("/api/v1"):
        return cleaned
    if cleaned.endswith("/api"):
        return f"{cleaned}/v1"
    return f"{cleaned}/api/v1"


def detect_api_base() -> str:
    if st.session_state.get("api_base_url"):
        return normalize_api_base(st.session_state["api_base_url"])

    env_url = (
        os.environ.get("EXTERNAL_URL")
        or os.environ.get("API_BASE_URL")
        or os.environ.get("STREAMLIT_API_URL")
    )
    if env_url:
        return normalize_api_base(env_url)

    server_ip = os.environ.get("SERVER_IP")
    if server_ip:
        return normalize_api_base(f"http://{server_ip}:8000")

    local_ip = get_local_ip()
    if local_ip and local_ip != "127.0.0.1":
        return normalize_api_base(f"http://{local_ip}:8000")

    return DEFAULT_API_BASE


def request_json(method: str, path: str, **kwargs: Any) -> tuple[int | None, Any]:
    url = f"{detect_api_base()}{path}"
    try:
        response = requests.request(
            method, url, timeout=kwargs.pop("timeout", REQUEST_TIMEOUT), **kwargs
        )
        try:
            return response.status_code, response.json()
        except ValueError:
            return response.status_code, response.text
    except requests.RequestException as exc:
        return None, str(exc)


def check_api_connection() -> bool:
    status_code, _ = request_json("GET", "/health")
    return status_code == 200


def get_api_health() -> dict[str, Any]:
    status_code, payload = request_json("GET", "/health")
    return payload if status_code == 200 and isinstance(payload, dict) else {}


def get_tags_list() -> list[dict[str, Any]]:
    status_code, payload = request_json("GET", "/tags")
    if status_code == 200 and isinstance(payload, dict):
        return payload.get("tags", [])
    return []


def get_rag_stats() -> dict[str, Any]:
    status_code, payload = request_json("GET", "/rag/stats")
    return payload if status_code == 200 and isinstance(payload, dict) else {}


def tag_image(image_bytes: bytes, top_k: int, confidence_threshold: float) -> dict[str, Any]:
    files = {"file": ("image.jpg", image_bytes, "image/jpeg")}
    data = {
        "top_k": top_k,
        "confidence_threshold": confidence_threshold,
        "include_metadata": "true",
    }
    status_code, payload = request_json(
        "POST", "/tag-cover", files=files, data=data, timeout=TAG_TIMEOUT
    )
    if status_code == 200 and isinstance(payload, dict):
        return payload
    return {"error": f"API error: {status_code} - {payload}"}


def add_to_rag(image_bytes: bytes, tags: list[str]) -> dict[str, Any]:
    files = {"file": ("reference.jpg", image_bytes, "image/jpeg")}
    data = {
        "tags": json.dumps(tags, ensure_ascii=False),
        "metadata": json.dumps({"source": "streamlit"}, ensure_ascii=False),
    }
    status_code, payload = request_json(
        "POST", "/rag/add", files=files, data=data, timeout=TAG_TIMEOUT
    )
    if status_code == 200 and isinstance(payload, dict):
        return payload
    return {"error": f"API error: {status_code} - {payload}"}


def set_page_config() -> None:
    st.set_page_config(
        page_title="Manga Tagger",
        page_icon="🏷️",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def render_sidebar() -> tuple[int, float] | None:
    with st.sidebar:
        st.header("連線設定")
        detected_url = detect_api_base()
        current_value = st.text_input("API Base URL", value=detected_url)
        normalized = normalize_api_base(current_value)
        st.session_state["api_base_url"] = normalized
        st.caption(f"目前使用: `{normalized}`")

        local_ip = get_local_ip()
        if local_ip != "127.0.0.1" and is_private_ip(local_ip):
            st.caption(f"本機 IP: `{local_ip}`")

        st.divider()
        st.subheader("API 狀態")
        if check_api_connection():
            health = get_api_health()
            st.success("API 已連線")
            models = health.get("models_loaded", {})
            st.metric("版本", health.get("version", "unknown"))
            st.metric("標籤數", models.get("tag_library", 0))
            if models:
                st.write(f"VLM: `{models.get('vlm', 'N/A')}`")
                st.write(f"LLM: `{models.get('llm', 'N/A')}`")
                st.write(f"RAG: `{models.get('rag', 'N/A')}`")
        else:
            st.error("無法連到 API")
            st.code("python start_server.py")
            return None

        st.divider()
        st.subheader("貼標參數")
        top_k = st.slider("返回標籤數", 1, 20, 5)
        confidence_threshold = st.slider("最低信心值", 0.0, 1.0, 0.5, 0.05)

        st.divider()
        rag_stats = get_rag_stats()
        if rag_stats:
            st.subheader("RAG 狀態")
            st.metric("文件數", rag_stats.get("total_documents", 0))
            if rag_stats.get("collection_name"):
                st.caption(f"Collection: `{rag_stats['collection_name']}`")

        return top_k, confidence_threshold


def render_tag_results(result: dict[str, Any]) -> None:
    tags = result.get("tags", [])
    metadata = result.get("metadata", {}) or {}

    st.subheader("貼標結果")
    if not tags:
        st.warning("沒有返回任何標籤")
        return

    st.metric("標籤數", len(tags))
    for index, tag in enumerate(tags, start=1):
        name = tag.get("tag", "")
        confidence = float(tag.get("confidence", 0.0))
        source = tag.get("source", "")
        reason = tag.get("reason", "")
        with st.expander(f"{index}. {name} ({confidence:.0%})", expanded=index <= 3):
            st.write(f"來源: `{source}`")
            st.progress(max(0.0, min(confidence, 1.0)))
            if reason:
                st.write(reason)

    if metadata:
        st.subheader("處理資訊")
        if metadata.get("processing_time") is not None:
            st.write(f"處理時間: `{metadata['processing_time']}s`")
        if metadata.get("vlm_description"):
            st.write("VLM 描述:")
            st.info(metadata["vlm_description"])
        if metadata.get("vlm_analysis"):
            st.json(metadata["vlm_analysis"])
        if metadata.get("rag_matches"):
            st.write("RAG 匹配:")
            st.json(metadata["rag_matches"])


def render_tagger_tab(top_k: int, confidence_threshold: float) -> None:
    st.header("漫畫封面貼標")
    uploaded_file = st.file_uploader(
        "選擇圖片", type=["jpg", "jpeg", "png", "webp", "bmp"], key="tagger_upload"
    )
    if uploaded_file is None:
        return

    image_bytes = uploaded_file.getvalue()
    image = Image.open(io.BytesIO(image_bytes))
    col1, col2 = st.columns([2, 1])
    with col1:
        st.image(image, caption=uploaded_file.name, use_container_width=True)
    with col2:
        st.write(f"尺寸: `{image.size[0]} x {image.size[1]}`")
        st.write(f"格式: `{image.format}`")
        st.write(f"大小: `{len(image_bytes) / 1024:.1f} KB`")
        if st.button("開始貼標", type="primary", use_container_width=True):
            with st.spinner("分析圖片中..."):
                result = tag_image(image_bytes, top_k, confidence_threshold)
            if "error" in result:
                st.error(result["error"])
            else:
                render_tag_results(result)


def render_rag_tab() -> None:
    st.header("RAG 資料管理")
    uploaded_file = st.file_uploader(
        "新增參考圖片", type=["jpg", "jpeg", "png", "webp", "bmp"], key="rag_upload"
    )
    tag_input = st.text_input("標籤", placeholder="用逗號分隔，例如: 貓娘, 女生制服")
    if uploaded_file is None:
        return

    image_bytes = uploaded_file.getvalue()
    image = Image.open(io.BytesIO(image_bytes))
    st.image(image, caption=uploaded_file.name, width=320)

    if st.button("加入 RAG", use_container_width=True):
        tags = [item.strip() for item in tag_input.split(",") if item.strip()]
        if not tags:
            st.error("請至少輸入一個標籤")
            return
        with st.spinner("寫入 RAG 中..."):
            result = add_to_rag(image_bytes, tags)
        if "error" in result:
            st.error(result["error"])
        else:
            st.success(result.get("message", "已加入 RAG"))
            if result.get("id"):
                st.code(result["id"])


def render_tags_tab() -> None:
    st.header("標籤瀏覽")
    tags = get_tags_list()
    if not tags:
        st.warning("目前無法取得標籤列表")
        return

    query = st.text_input("搜尋標籤")
    filtered = (
        [tag for tag in tags if query.lower() in tag.get("tag_name", "").lower()] if query else tags
    )

    st.write(f"共 {len(filtered)} 筆")
    for tag in filtered[:200]:
        title = tag.get("tag_name", "")
        desc = tag.get("description") or "無描述"
        with st.expander(title):
            st.write(desc)

    if len(filtered) > 200:
        st.caption("只顯示前 200 筆，請縮小搜尋範圍。")


def main() -> None:
    set_page_config()
    st.title("Manga Cover Auto-Tagger")
    st.caption("使用 FastAPI + VLM + RAG 的本機貼標工具")

    sidebar_values = render_sidebar()
    if sidebar_values is None:
        st.stop()

    top_k, confidence_threshold = sidebar_values
    tab1, tab2, tab3 = st.tabs(["貼標", "RAG", "標籤列表"])
    with tab1:
        render_tagger_tab(top_k, confidence_threshold)
    with tab2:
        render_rag_tab()
    with tab3:
        render_tags_tab()


if __name__ == "__main__":
    main()
