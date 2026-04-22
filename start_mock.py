import os

os.environ["USE_MOCK_SERVICES"] = "true"
os.environ["USE_LM_STUDIO"] = "false"

import subprocess
import sys
import time

print("Starting API server with mock mode...")

api_process = subprocess.Popen(
    [sys.executable, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "8000"],
    env={**os.environ, "USE_MOCK_SERVICES": "true", "USE_LM_STUDIO": "false"},
)

print(f"API server started (PID: {api_process.pid})")
print("Waiting for server to be ready...")

time.sleep(5)

# Test connection
import urllib.request

try:
    response = urllib.request.urlopen("http://127.0.0.1:8000/api/v1/health", timeout=5)
    print(f"Server is ready! Response: {response.read().decode()}")
except Exception as e:
    print(f"Health check failed: {e}")

print("\nServer running at http://localhost:8000")
print("Press Ctrl+C to stop")

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("\nStopping...")
    api_process.terminate()
