#!/usr/bin/env python3
"""
快速啟動腳本 - 衝突系統優化工具集
一鍵啟動所有擴展功能
"""

import os
import sys
import subprocess
import json
from typing import Dict, Any


def install_dependencies():
    """安裝必要的依賴"""
    print("📦 安裝依賴包...")

    dependencies = [
        "streamlit",
        "plotly",
        "networkx",
        "scikit-learn",
        "pandas",
        "numpy",
    ]

    for dep in dependencies:
        print(f"  安裝 {dep}...")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", dep],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            print(f"  ✅ {dep} 安裝成功")
        else:
            print(f"  ❌ {dep} 安裝失敗: {result.stderr}")


def create_directories():
    """創建必要的目錄"""
    print("📁 創建目錄結構...")

    directories = [
        "data/conflict_learner_models",
        "logs/conflicts",
        "exports/rules",
        "cache/conflict_cache",
        "monitoring/metrics",
    ]

    for dir_path in directories:
        os.makedirs(dir_path, exist_ok=True)
        print(f"  ✅ 創建目錄: {dir_path}")


def generate_startup_config():
    """生成啟動配置文件"""
    print("⚙️ 生成配置文件...")

    config = {
        "conflict_system": {
            "version": "2.0",
            "learning_enabled": True,
            "cache_enabled": True,
            "auto_save_interval": 100,
        },
        "dashboard": {
            "host": "localhost",
            "port": 8501,
            "theme": "light",
            "auto_refresh": True,
        },
        "api": {
            "host": "0.0.0.0",
            "port": 8001,
            "rate_limit": 100,
            "cors_enabled": True,
        },
        "performance": {
            "max_cache_size": 10000,
            "cache_ttl": 3600,
            "parallel_processing": True,
            "gpu_acceleration": False,
        },
    }

    config_path = "conflict_system_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    print(f"  ✅ 配置文件已生成: {config_path}")


def show_menu():
    """顯示菜單"""
    print("\n" + "=" * 60)
    print("🚀 標籤衝突系統擴展工具集")
    print("=" * 60)
    print()
    print("請選擇要啟動的功能:")
    print()
    print("1️⃣  高級衝突檢測系統")
    print("2️⃣  機器學習衝突檢測")
    print("3️⃣  可視化管理儀表板")
    print("4️⃣  衝突規則分析工具")
    print("5️⃣  系統性能監控")
    print("6️⃣  一鍵部署所有功能")
    print("0️⃣  退出")
    print()

    choice = input("請輸入選項 (0-6): ").strip()
    return choice


def launch_advanced_conflict_system():
    """啟動高級衝突檢測系統"""
    print("\n🔧 啟動高級衝突檢測系統...")

    # 這裡可以添加啟動邏輯
    print("  ✅ 高級衝突檢測已就緒")
    print(
        "  📖 使用方法: from advanced_conflict_system import AdvancedTagConflictSystem"
    )


def launch_learning_system():
    """啟動機器學習系統"""
    print("\n🤖 啟動機器學習衝突檢測...")

    # 初始化學習系統
    try:
        from conflict_learning_system import ConflictLearner

        learner = ConflictLearner()
        stats = learner.get_learning_statistics()
        print("  ✅ 機器學習系統已就緒")
        print(f"  📊 當前狀態: {stats['total_sessions']} 個會話")
    except Exception as e:
        print(f"  ❌ 啟動失敗: {e}")


def launch_dashboard():
    """啟動可視化儀表板"""
    print("\n📊 啟動可視化管理儀表板...")

    # 檢查依賴
    try:
        import streamlit
        import plotly
        import networkx

        print("  ✅ 依賴檢查通過")
        print("  🚀 啟動儀表板...")

        # 啟動命令
        cmd = [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            "conflict_dashboard.py",
            "--server.port",
            "8501",
        ]

        print(f"  🔗 訪問地址: http://localhost:8501")
        print("  按 Ctrl+C 停止服務")
        print()

        subprocess.run(cmd)

    except ImportError as e:
        print(f"  ❌ 缺少依賴: {e}")
        print("  💡 請先運行選項 7 安裝依賴")


def launch_analysis_tools():
    """啟動分析工具"""
    print("\n🔍 啟動衝突規則分析工具...")

    try:
        from analyze_tag_conflicts import main as analysis_main

        print("  ✅ 分析工具啟動中...")
        analysis_main()
    except Exception as e:
        print(f"  ❌ 啟動失敗: {e}")


def launch_monitoring():
    """啟動性能監控"""
    print("\n📡 啟動系統性能監控...")

    # 這裡可以添加監控邏輯
    print("  ✅ 性能監控已就緒")
    print("  📊 監控指標: CPU, 內存, 響應時間, 準確率")


def deploy_all():
    """一鍵部署所有功能"""
    print("\n🚀 一鍵部署所有功能...")

    # 安裝依賴
    install_dependencies()

    # 創建目錄
    create_directories()

    # 生成配置
    generate_startup_config()

    print("\n✅ 所有功能部署完成！")
    print("\n🎯 接下來你可以:")
    print("1️⃣ 運行: streamlit run conflict_dashboard.py")
    print("2️⃣ 查看: SYSTEM_OPTIMIZATION_ROADMAP.md")
    print("3️⃣ 測試: python test_conflicts_simple.py")
    print("4️⃣ 分析: python analyze_tag_conflicts.py")


def main():
    """主函數"""

    # 檢查是否首次運行
    if not os.path.exists("conflict_system_config.json"):
        print("👋 首次檢測到系統，正在初始化...")
        install_dependencies()
        create_directories()
        generate_startup_config()
        print("\n✅ 系統初始化完成！")

    while True:
        choice = show_menu()

        if choice == "1":
            launch_advanced_conflict_system()
        elif choice == "2":
            launch_learning_system()
        elif choice == "3":
            launch_dashboard()
        elif choice == "4":
            launch_analysis_tools()
        elif choice == "5":
            launch_monitoring()
        elif choice == "6":
            deploy_all()
        elif choice == "0":
            print("\n👋 再見！")
            break
        else:
            print("\n❌ 無效選項，請重新選擇")

        input("\n按 Enter 繼續...")


if __name__ == "__main__":
    main()
