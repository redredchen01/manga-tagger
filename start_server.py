#!/usr/bin/env python3
"""
Startup script for Manga Tagger API server.
"""

import subprocess
import sys
import time
import requests
from pathlib import Path

def check_server_ready():
    """Check if server is ready."""
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        return response.status_code == 200
    except:
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
        process = subprocess.Popen([
            sys.executable, "-m", "uvicorn", "app.main:app",
            "--host", "0.0.0.0",
            "--port", "8000",
            "--reload"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        # Wait for server to be ready
        print("Waiting for server to start...")
        for i in range(30):  # Wait up to 30 seconds
            if check_server_ready():
                print(f"Server is ready! (took {i+1} seconds)")
                print("\nAPI Documentation: http://localhost:8000/docs")
                print("Health Check: http://localhost:8000/health")
                print("\nPress Ctrl+C to stop the server")
                try:
                    process.wait()
                except KeyboardInterrupt:
                    print("\nShutting down server...")
                    process.terminate()
                    process.wait()
                return
            
            time.sleep(1)
            print(f"Waiting... {i+1}")
        
        print("Server failed to start within 30 seconds")
        print("Error output:")
        _, stderr = process.communicate()
        if stderr:
            print(stderr)
        process.terminate()
        
    except Exception as e:
        print(f"Failed to start server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()