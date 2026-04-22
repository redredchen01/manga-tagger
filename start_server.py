#!/usr/bin/env python3
"""
Startup script for Manga Tagger API server.
"""

import subprocess
import sys
import time
from pathlib import Path

import requests

def check_server_ready():
    """Check if server is ready."""
    try:
        response = requests.get("http://localhost:8000/api/v1/health", timeout=5)
        return response.status_code == 200
    except requests.RequestException:
        return False


def main():
    """Start the server."""
    print("Starting Manga Tagger API Server...")
    print("=" * 50)

    # Check if we're in the right directory
    if not Path("app/main.py").exists():
        print("Error: app/main.py not found!")
        print("Please run this script from the project root directory.")
        sys.exit(1)

    # Start uvicorn in background
    print("Starting uvicorn server...")
    try:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "app.main:app",
                "--host",
                "0.0.0.0",
                "--port",
                "8000",
                "--reload",
            ]
        )

        # Wait for server to be ready — model loading (BAAI/bge-m3 etc.) can take ~45s
        max_wait = 180
        print(f"Waiting for server to start (up to {max_wait}s for model loading)...")
        for i in range(max_wait):
            if check_server_ready():
                print(f"Server is ready! (took {i+1} seconds)")
                print("\nAPI Documentation: http://localhost:8000/docs")
                print("Health Check: http://localhost:8000/api/v1/health")
                print("\nPress Ctrl+C to stop the server")
                try:
                    process.wait()
                except KeyboardInterrupt:
                    print("\nShutting down server...")
                    process.terminate()
                    process.wait()
                return

            if process.poll() is not None:
                print(f"Server process exited prematurely with code {process.returncode}")
                return

            time.sleep(1)
            if (i + 1) % 5 == 0:
                print(f"Waiting... {i+1}s")

        print(f"Server failed to start within {max_wait} seconds")
        process.terminate()
        process.wait(timeout=10)

    except Exception as e:
        print(f"Failed to start server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
