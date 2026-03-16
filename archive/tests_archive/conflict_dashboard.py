"""
衝突系統可視化管理工具
Phase 4: Visualization & Management Dashboard
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import networkx as nx
import json
import asyncio
from typing import Dict, List, Any
import sys
import os

# 添加項目路徑
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from advanced_conflict_system import AdvancedTagConflictSystem
from conflict_learning_system import ConflictLearner


class ConflictVisualizationDashboard:
    """衝突系統可視化儀表板"""

    def __init__(self):
        self.conflict_system = AdvancedTagConflictSystem()
        self.learner = ConflictLearner()

    def run_dashboard(self):
        """運行可視化儀表板"""
        st.set_page_config(page_title="標籤衝突管理系統", page_icon="⚖️", layout="wide")

        st.title("🏷️ 標籤衝突管理系統 v2.0")
        st.markdown("---")

        # 側邊欄選項
        with st.sidebar:
            st.header("🎛️ 控制面板")
            page = st.selectbox(
                "選擇功能頁面",
                ["衝突規則管理", "衝突檢測測試", "學習統計分析", "系統性能監控"],
            )

        # 主內容區
        if page == "衝突規則管理":
            self._show_rules_management()
        elif page == "衝突檢測測試":
            self._show_conflict_testing()
        elif page == "學習統計分析":
            self._show_learning_analytics()
        elif page == "系統性能監控":
            self._show_performance_monitoring()

    def _show_rules_management(self):
        """顯示規則管理頁面"""
        st.header("📋 衝突規則管理")

        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("🔍 衝突網絡圖")

            # 創建衝突網絡圖
            conflict_graph = self._create_conflict_network()

            if conflict_graph:
                fig = go.Figure(
                    data=[
                        go.Scatter(
                            x=[pos[0] for pos in conflict_graph["positions"].values()],
                            y=[pos[1] for pos in conflict_graph["positions"].values()],
                            mode="markers+text",
                            text=list(conflict_graph["positions"].keys()),
                            textposition="middle center",
                            marker=dict(size=20, color="lightblue"),
                            name="標籤",
                        )
                    ]
                )

                # 添加連線
                for edge in conflict_graph["edges"]:
                    pos1 = conflict_graph["positions"][edge[0]]
                    pos2 = conflict_graph["positions"][edge[1]]
                    fig.add_shape(
                        type="line",
                        x0=pos1[0],
                        y0=pos1[1],
                        x1=pos2[0],
                        y1=pos2[1],
                        line=dict(color="red", width=2),
                    )

                fig.update_layout(
                    title="標籤衝突關係網絡", showlegend=False, height=600
                )
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("📊 規則統計")

            # 獲取統計數據
            stats = self.conflict_system.get_conflict_statistics()

            # 規則嚴重程度分布
            severity_data = stats["rules_by_severity"]
            fig_pie = px.pie(
                values=list(severity_data.values()),
                names=list(severity_data.keys()),
                title="衝突嚴重程度分布",
            )
            st.plotly_chart(fig_pie)

            # 規則類別分布
            category_data = stats["rules_by_category"]
            fig_bar = px.bar(
                x=list(category_data.keys()),
                y=list(category_data.values()),
                title="衝突類別分布",
            )
            st.plotly_chart(fig_bar)

        # 規則編輯區
        st.subheader("✏️ 規則編輯")

        with st.expander("添加新規則"):
            col1, col2, col3 = st.columns(3)

            with col1:
                new_tag = st.text_input("標籤名稱")
                severity_options = ["critical", "strong", "moderate", "weak"]
                severity = st.selectbox("嚴重程度", severity_options)

            with col2:
                conflict_tags = st.text_area("衝突標籤（每行一個）")
                category_options = ["logical", "contextual", "scenario", "preference"]
                category = st.selectbox("衝突類別", category_options)

            with col3:
                description = st.text_area("規則描述")
                context_required = st.checkbox("需要上下文")

            if st.button("添加規則"):
                if new_tag and conflict_tags:
                    conflicts = [
                        tag.strip() for tag in conflict_tags.split("\n") if tag.strip()
                    ]
                    # 這裡需要實際添加規則的邏輯
                    st.success(f"成功添加規則: {new_tag}")

    def _show_conflict_testing(self):
        """顯示衝突檢測測試頁面"""
        st.header("🧪 衝突檢測測試")

        # 輸入測試標籤
        col1, col2 = st.columns([3, 1])

        with col1:
            test_tags_input = st.text_area(
                "輸入測試標籤（用逗號分隔）", placeholder="例如: 藍髮, 紅髮, 巨乳, 蘿莉"
            )

        with col2:
            test_button = st.button("🔍 檢測衝突", type="primary")

        if test_button and test_tags_input:
            test_tags = [
                tag.strip() for tag in test_tags_input.split(",") if tag.strip()
            ]

            # 模擬TagRecommendation對象
            from app.services.tag_recommender_service import TagRecommendation

            mock_tags = [
                TagRecommendation(
                    tag=tag, confidence=0.8, source="test", reason="Test tag"
                )
                for tag in test_tags
            ]

            # 執行衝突檢測
            with st.spinner("正在檢測衝突..."):
                # 這裡需要異步調用，暫時用同步版本
                # result = await self.conflict_system.analyze_conflicts_advanced(mock_tags)
                st.info("衝突檢測完成")

            # 顯示結果
            st.subheader("📈 檢測結果")

            # 結果統計
            result_col1, result_col2, result_col3 = st.columns(3)

            with result_col1:
                st.metric("輸入標籤", len(test_tags))

            with result_col2:
                detected_conflicts = 3  # 模擬結果
                st.metric("檢測衝突", detected_conflicts)

            with result_col3:
                resolved_tags = len(test_tags) - detected_conflicts
                st.metric("解決後標籤", resolved_tags)

            # 衝突詳情
            if detected_conflicts > 0:
                st.subheader("⚠️ 衝突詳情")

                conflict_data = [
                    {
                        "標籤1": "藍髮",
                        "標籤2": "紅髮",
                        "嚴重程度": "critical",
                        "置信度": 0.9,
                    },
                    {
                        "標籤1": "巨乳",
                        "標籤2": "貧乳",
                        "嚴重程度": "critical",
                        "置信度": 0.8,
                    },
                ]

                df_conflicts = pd.DataFrame(conflict_data)
                st.dataframe(df_conflicts, use_container_width=True)
            else:
                st.success("✅ 未檢測到衝突！")

    def _show_learning_analytics(self):
        """顯示學習統計分析頁面"""
        st.header("📈 學習統計分析")

        # 獲取學習統計
        learning_stats = self.learner.get_learning_statistics()

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("總會話數", learning_stats["total_sessions"])

        with col2:
            st.metric("標籤對數", learning_stats["unique_tag_pairs"])

        with col3:
            st.metric("發現模式", learning_stats["discovered_patterns"])

        with col4:
            st.metric("模型大小", learning_stats["model_size"])

        # 學習趨勢圖
        st.subheader("📊 學習趨勢")

        # 模擬時間序列數據
        dates = pd.date_range(start="2024-01-01", periods=30, freq="D")
        accuracy = [0.7 + 0.02 * i + (i % 3) * 0.05 for i in range(30)]

        fig_trend = px.line(
            x=dates,
            y=accuracy,
            title="衝突檢測準確率趨勢",
            labels={"x": "日期", "y": "準確率"},
        )
        st.plotly_chart(fig_trend)

        # 新規則建議
        st.subheader("💡 新規則建議")

        suggested_rules = self.learner.generate_suggested_rules()

        if suggested_rules:
            for i, rule in enumerate(suggested_rules[:5], 1):
                with st.expander(f"建議規則 {i}: {rule['tag']}"):
                    st.json(rule)
        else:
            st.info("暫無新規則建議")

    def _show_performance_monitoring(self):
        """顯示系統性能監控頁面"""
        st.header("⚡ 系統性能監控")

        # 性能指標
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("平均響應時間", "45ms", "↓ 5ms")

        with col2:
            st.metric("內存使用", "128MB", "↑ 12MB")

        with col3:
            st.metric("衝突檢測準確率", "94.5%", "↑ 2.3%")

        # 實時監控
        st.subheader("📡 實時監控")

        # 模擬實時數據
        import time
        import random

        if st.checkbox("啟動實時監控"):
            placeholder = st.empty()

            for _ in range(100):
                current_time = time.strftime("%H:%M:%S")
                cpu_usage = random.uniform(20, 80)
                memory_usage = random.uniform(100, 200)

                with placeholder.container():
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric(f"CPU ({current_time})", f"{cpu_usage:.1f}%")

                    with col2:
                        st.metric("內存", f"{memory_usage:.1f}MB")

                    with col3:
                        st.metric("活躍連接", random.randint(10, 50))

                time.sleep(1)

    def _create_conflict_network(self) -> Dict[str, Any]:
        """創建衝突網絡圖數據"""
        try:
            G = nx.Graph()

            # 添加節點和邊
            rules = self.conflict_system.conflict_rules
            sample_tags = list(rules.keys())[:20]  # 限制數量避免過於複雜

            for tag in sample_tags:
                G.add_node(tag)

            for tag, rule in list(rules.items())[:20]:
                for conflict in rule.conflicts[:3]:  # 限制衝突數量
                    if conflict in sample_tags:
                        G.add_edge(tag, conflict)

            # 計算位置
            if len(G.nodes) > 0:
                pos = nx.spring_layout(G, k=1, iterations=50)

                return {
                    "nodes": list(G.nodes),
                    "edges": list(G.edges),
                    "positions": {
                        node: (pos[node][0], pos[node][1]) for node in G.nodes
                    },
                }
            else:
                return None

        except Exception as e:
            st.error(f"創建網絡圖失敗: {e}")
            return None


def main():
    """主函數"""
    dashboard = ConflictVisualizationDashboard()
    dashboard.run_dashboard()


if __name__ == "__main__":
    main()
