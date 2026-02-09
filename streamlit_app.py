#!/usr/bin/env python3
"""
Streamlit Frontend for Manga Tagger System
A web interface for uploading and tagging manga covers.
"""

import streamlit as st
import requests
import json
import time
from PIL import Image
import io
import base64
from typing import Dict, List, Any

# Configuration
API_BASE_URL = "http://localhost:8000"


def set_page_config():
    """Configure Streamlit page."""
    st.set_page_config(
        page_title="Manga Cover Auto-Tagger",
        page_icon="🎌",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def check_api_connection() -> bool:
    """Check if API server is running."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False


def get_api_health() -> Dict[str, Any]:
    """Get API health information."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return {}


def get_tags_list() -> List[Dict[str, Any]]:
    """Get list of available tags."""
    try:
        response = requests.get(f"{API_BASE_URL}/tags", timeout=5)
        if response.status_code == 200:
            data = response.json()
            return data.get("tags", [])
    except:
        pass
    return []


def get_rag_stats() -> Dict[str, Any]:
    """Get RAG database statistics."""
    try:
        response = requests.get(f"{API_BASE_URL}/rag/stats", timeout=5)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return {}


def tag_image(
    image_bytes: bytes, top_k: int = 5, confidence_threshold: float = 0.5
) -> Dict[str, Any]:
    """Tag an image using the API."""
    try:
        files = {"file": ("image.jpg", image_bytes, "image/jpeg")}
        data = {
            "top_k": top_k,
            "confidence_threshold": confidence_threshold,
            "include_metadata": True,
        }

        response = requests.post(
            f"{API_BASE_URL}/tag-cover", files=files, data=data, timeout=30
        )

        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API Error: {response.status_code} - {response.text}"}

    except Exception as e:
        return {"error": f"Connection Error: {str(e)}"}


def add_to_rag(image_bytes: bytes, tags: List[str]) -> Dict[str, Any]:
    """Add an image to RAG database."""
    try:
        files = {"file": ("image.jpg", image_bytes, "image/jpeg")}
        data = {
            "tags": json.dumps(tags, ensure_ascii=False),
            "metadata": json.dumps({"source": "web_interface"}, ensure_ascii=False),
        }

        response = requests.post(
            f"{API_BASE_URL}/rag/add", files=files, data=data, timeout=30
        )

        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API Error: {response.status_code}"}

    except Exception as e:
        return {"error": f"Connection Error: {str(e)}"}


def render_sidebar():
    """Render sidebar with controls and information."""
    with st.sidebar:
        st.header("⚙️ 設置")

        # API Status
        st.subheader("🔌 API 狀態")
        if check_api_connection():
            health = get_api_health()
            st.success("✅ API 連接正常")

            st.metric("版本", health.get("version", "Unknown"))
            models = health.get("models_loaded", {})

            if models.get("lm_studio_mode"):
                st.metric("LM Studio 模式", "已啟用")
                st.info("使用真實 LM Studio AI 模型")
                
                # Display specific model names
                if "vlm" in models:
                    st.write(f"**VLM 模型:** `{models['vlm']}`")
                if "llm" in models:
                    st.write(f"**LLM 模型:** `{models['llm']}`")
                if "rag" in models:
                    st.write(f"**RAG 模式:** `{models['rag']}`")
            else:
                st.metric("LM Studio 模式", "未啟用")

            # Display tag library info
            if "tag_library" in models:
                st.metric("標籤庫", f"{models['tag_library']} 個標籤")
        else:
            st.error("❌ API 連接失敗")
            st.code("請確保 API 服務器運行在 http://localhost:8000")
            return False

        st.divider()

        # Tagging Parameters
        st.subheader("🏷️ 標籤參數")
        top_k = st.slider("返回標籤數量", min_value=1, max_value=20, value=5, step=1)
        confidence_threshold = st.slider(
            "信心度閾值", min_value=0.0, max_value=1.0, value=0.5, step=0.1
        )

        st.divider()

        # Database Info
        st.subheader("📊 數據庫資訊")
        rag_stats = get_rag_stats()
        if rag_stats:
            st.metric("RAG 文檔數量", rag_stats.get("total_documents", 0))
            st.metric("嵌入模型", rag_stats.get("embedding_model", "Unknown"))

        return top_k, confidence_threshold


def render_main_interface(top_k: int, confidence_threshold: float):
    """Render main interface with tabs."""
    tab1, tab2, tab3 = st.tabs(["🖼️ 圖片標籤", "📚 RAG 管理", "📋 標籤瀏覽"])

    with tab1:
        render_tagger_tab(top_k, confidence_threshold)

    with tab2:
        render_rag_tab()

    with tab3:
        render_tags_browser_tab()


def render_tagger_tab(top_k: int, confidence_threshold: float):
    """Render image tagging interface."""
    st.header("🖼️ 圖片自動標籤")
    st.markdown("上傳漫畫封面圖片，AI 將自動分析並生成相關標籤。")

    # File Upload
    col1, col2 = st.columns([2, 1])

    with col1:
        uploaded_file = st.file_uploader(
            "選擇圖片文件",
            type=["jpg", "jpeg", "png", "webp", "bmp"],
            help="支援 JPG, PNG, WebP, BMP 格式",
        )

    with col2:
        st.info("💡 提示")
        st.write("• 清晰的封面圖片效果最佳")
        st.write("• 支援多種圖片格式")
        st.write("• 文件大小建議 < 10MB")

    if uploaded_file is not None:
        # Display uploaded image
        image_bytes = uploaded_file.read()
        image = Image.open(io.BytesIO(image_bytes))

        st.subheader("📸 預覽")
        col1, col2 = st.columns([2, 1])

        with col1:
            st.image(image, caption="上傳的圖片", use_column_width=True)

        with col2:
            # Image info
            st.write("**圖片資訊**")
            st.write(f"檔名: {uploaded_file.name}")
            st.write(f"尺寸: {image.size}")
            st.write(f"格式: {image.format}")

            # Tag button
            if st.button("🏷️ 開始標籤", type="primary", use_container_width=True):
                st.write(f"DEBUG: top_k={top_k}, confidence={confidence_threshold}")
                with st.spinner("AI 正在分析圖片..."):
                    result = tag_image(image_bytes, top_k, confidence_threshold)

                st.write(f"DEBUG: API response received, result keys: {result.keys()}")

                # Display results
                if "error" in result:
                    st.error(f"❌ 標籤失敗: {result['error']}")
                else:
                    tags_count = len(result.get("tags", []))
                    st.success(f"✅ 標籤完成！生成 {tags_count} 個標籤")
                    st.write(f"DEBUG: tags = {result.get('tags', [])}")
                    render_tagging_results(result, image)


def render_tagging_results(result: Dict[str, Any], image: Image.Image):
    """Render tagging results."""
    st.subheader("🏷️ 標籤結果")

    tags = result.get("tags", [])
    metadata = result.get("metadata", {})

    # Tags display
    col1, col2 = st.columns([3, 1])

    with col1:
        st.write("**生成標籤：**")
        for i, tag in enumerate(tags, 1):
            tag_name = tag.get("tag", "")
            confidence = tag.get("confidence", 0)
            source = tag.get("source", "")
            reason = tag.get("reason", "")

            # Create progress bar for confidence
            confidence_pct = int(confidence * 100)

            with st.expander(f"{i}. {tag_name} (信心度: {confidence_pct}%)"):
                st.write(f"**來源:** {source}")
                st.write(f"**信心度:** {confidence:.3f}")
                st.progress(confidence)
                if reason:
                    st.write(f"**理由:** {reason}")

    with col2:
        # Statistics
        st.write("**統計資訊：**")
        st.metric("標籤數量", len(tags))

        if tags:
            avg_confidence = sum(tag.get("confidence", 0) for tag in tags) / len(tags)
            st.metric("平均信心度", f"{avg_confidence:.3f}")

        if metadata.get("processing_time"):
            st.metric("處理時間", f"{metadata['processing_time']:.2f}s")

    # Metadata details
    if metadata:
        st.subheader("📋 詳細資訊")

        col1, col2 = st.columns([1, 1])

        with col1:
            if metadata.get("vlm_description"):
                st.write("**VLM 描述：**")
                st.info(metadata["vlm_description"])

            if metadata.get("rag_matches") and isinstance(metadata["rag_matches"], list):
                st.write("**RAG 匹配：**")
                for i, match in enumerate(metadata["rag_matches"][:3], 1):
                    if isinstance(match, dict):
                        score = match.get("score", 0)
                        tags = match.get("tags", [])
                        st.write(f"{i}. 相似度: {score:.3f}")
                        st.write(f"   標籤: {', '.join(tags)}")

        with col2:
            # New VLM Analysis Format
            if metadata.get("vlm_analysis") and isinstance(metadata["vlm_analysis"], dict):
                vlm_analysis = metadata["vlm_analysis"]
                st.write("**VLM 分析結果：**")

                if vlm_analysis.get("character_types") and isinstance(vlm_analysis["character_types"], list):
                    st.write(
                        f"角色類型: {', '.join(vlm_analysis['character_types'][:5])}"
                    )

                if vlm_analysis.get("clothing") and isinstance(vlm_analysis["clothing"], list):
                    st.write(f"服裝: {', '.join(vlm_analysis['clothing'][:5])}")

                if vlm_analysis.get("body_features") and isinstance(vlm_analysis["body_features"], list):
                    st.write(
                        f"身體特徵: {', '.join(vlm_analysis['body_features'][:5])}"
                    )

                if vlm_analysis.get("actions") and isinstance(vlm_analysis["actions"], list):
                    st.write(f"動作: {', '.join(vlm_analysis['actions'][:5])}")

                if vlm_analysis.get("themes") and isinstance(vlm_analysis["themes"], list):
                    st.write(f"主題: {', '.join(vlm_analysis['themes'][:5])}")

            # Legacy format support
            elif metadata.get("vlm_metadata"):
                vlm_meta = metadata["vlm_metadata"]
                st.write("**VLM 元數據：**")

                if vlm_meta.get("characters"):
                    st.write(f"角色: {', '.join(vlm_meta['characters'])}")

                if vlm_meta.get("themes"):
                    st.write(f"主題: {', '.join(vlm_meta['themes'])}")

                if vlm_meta.get("art_style"):
                    st.write(f"畫風: {vlm_meta['art_style']}")

                if vlm_meta.get("genre_indicators"):
                    st.write(f"類型: {', '.join(vlm_meta['genre_indicators'])}")

            # Display library info
            if metadata.get("library_tags_available"):
                st.write("**標籤庫：**")
                st.write(f"可用標籤: {metadata['library_tags_available']} 個")


def render_rag_tab():
    """Render RAG management interface."""
    st.header("📚 RAG 數據庫管理")
    st.markdown("管理參考圖片數據庫，添加新的參考圖片以改善標籤質量。")

    # Current stats
    rag_stats = get_rag_stats()
    if rag_stats:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("文檔總數", rag_stats.get("total_documents", 0))
        with col2:
            st.metric("集合名稱", rag_stats.get("collection_name", "Unknown"))
        with col3:
            embedding_mode = rag_stats.get("embedding_mode", "Unknown")
            st.metric("嵌入模式", embedding_mode)

    st.divider()

    # Add new image to RAG
    st.subheader("➕ 添加參考圖片")
    uploaded_file = st.file_uploader(
        "選擇參考圖片", type=["jpg", "jpeg", "png", "webp", "bmp"], key="rag_upload"
    )

    if uploaded_file is not None:
        # Preview
        image = Image.open(io.BytesIO(uploaded_file.read()))
        st.image(image, caption="參考圖片預覽", width=300)

        # Tag input
        tag_input = st.text_area(
            "標籤 (用逗號分隔)",
            placeholder="例如: 貓娘, 蘿莉, 校服",
            help="輸入此參考圖片包含的所有標籤",
        )

        if st.button("➕ 添加到 RAG", type="primary"):
            if tag_input.strip():
                tags = [tag.strip() for tag in tag_input.split(",") if tag.strip()]
                image_bytes = io.BytesIO()
                image.save(image_bytes, format="JPEG")
                image_bytes = image_bytes.getvalue()

                with st.spinner("正在添加到 RAG 數據庫..."):
                    result = add_to_rag(image_bytes, tags)

                if "error" in result:
                    st.error(f"❌ 添加失敗: {result['error']}")
                else:
                    st.success(f"✅ 成功添加！文檔 ID: {result.get('id', 'Unknown')}")
            else:
                st.error("❌ 請輸入至少一個標籤")


def render_tags_browser_tab():
    """Render tags browser interface."""
    st.header("📋 標籤瀏覽器")
    st.markdown("瀏覽系統中所有可用的標籤。")

    # Load tags
    tags = get_tags_list()

    if not tags:
        st.warning("⚠️ 無法載入標籤列表")
        return

    # Search/filter
    search_term = st.text_input("🔍 搜尋標籤", placeholder="輸入關鍵字搜尋...")

    # Filter tags
    if search_term:
        filtered_tags = [
            tag
            for tag in tags
            if search_term.lower() in tag.get("tag_name", "").lower()
        ]
    else:
        filtered_tags = tags

    st.write(f"找到 {len(filtered_tags)} 個標籤")

    # Display tags
    if filtered_tags:
        cols = st.columns(3)
        for i, tag in enumerate(filtered_tags):
            with cols[i % 3]:
                tag_name = tag.get("tag_name", "")
                description = tag.get("description", "")

                with st.expander(f"{tag_name}"):
                    if description:
                        st.write(description)
                    else:
                        st.write("無描述")
    else:
        st.info("沒有找到匹配的標籤")


def main():
    """Main application function."""
    set_page_config()

    # Title and description
    st.title("🎌 Manga Cover Auto-Tagger")
    st.markdown("""
    歡迎使用漫畫封面自動標籤系統！
    
    本系統使用 AI 技術自動分析漫畫封面並生成相關標籤。
    
    **功能特色：**
    - 🤖 AI 驅動的圖片分析
    - 🏷️ 自動標籤生成
    - 📚 RAG 技術支援
    - 🌐 Web 界面操作
    """)

    # Render sidebar and get parameters
    sidebar_result = render_sidebar()
    if not sidebar_result:
        st.stop()

    top_k, confidence_threshold = sidebar_result

    # Render main interface
    render_main_interface(top_k, confidence_threshold)

    # Footer
    st.divider()
    st.markdown(f"""
    ---
    **系統資訊：**
    - API 服務器: {API_BASE_URL}
    - 模式: LM Studio (真實 AI 模型)
    - API 文檔: {API_BASE_URL}/docs
    """)


if __name__ == "__main__":
    main()
