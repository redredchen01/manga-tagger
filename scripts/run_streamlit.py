#!/usr/bin/env python3
"""
Streamlit Frontend Launcher
Launches the Streamlit frontend for the Manga Tagger System.
"""

import subprocess
import sys
import os
import time
import requests


def check_api_server():
    """Check if API server is running."""
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        return response.status_code == 200
    except:
        return False


def main():
    """Main launcher function."""
    print("Manga Cover Auto-Tagger - Streamlit Frontend")
    print("=" * 50)

    # Check API server
    print("Checking API server status...")
    if not check_api_server():
        print("WARNING: API server is not running!")
        print("Please start API server first:")
        print("   python start_server.py")
        print("")
        print("Or run in separate terminal:")
        print("   Terminal 1: python start_server.py")
        print("   Terminal 2: python run_streamlit.py")
        sys.exit(1)

    print("API server is running normally")

    # Launch Streamlit
    print("Starting Streamlit frontend...")
    print("Frontend URL: http://localhost:8501")
    print("API Docs: http://localhost:8000/docs")
    print("")
    print("Press Ctrl+C to stop server")
    print("=" * 50)

    try:
        # Run streamlit
        subprocess.run(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                "streamlit_app.py",
                "--server.headless=false",
                "--server.port=8501",
                "--server.address=localhost",
                "--browser.gatherUsageStats=false",
            ],
            check=True,
        )
    except KeyboardInterrupt:
        print("\nStreamlit frontend stopped")
    except subprocess.CalledProcessError as e:
        print(f"Failed to start: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
