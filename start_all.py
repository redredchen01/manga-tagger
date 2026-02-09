import subprocess
import sys
import time

print("Starting services...")

# Start API server
print("[1] Starting API server on port 8000...")
api_process = subprocess.Popen(
    [
        sys.executable,
        "-m",
        "uvicorn",
        "app.main:app",
        "--host",
        "127.0.0.1",
        "--port",
        "8000",
    ],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
)

# Wait for API to start
time.sleep(3)

# Start Streamlit
print("[2] Starting Streamlit on port 8501...")
streamlit_process = subprocess.Popen(
    [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "streamlit_app.py",
        "--server.port",
        "8501",
        "--server.address",
        "127.0.0.1",
    ],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
)

print("\nServices started!")
print("API Server: http://127.0.0.1:8000")
print("Streamlit: http://127.0.0.1:8501")
print("\nPress Ctrl+C to stop...")

try:
    # Keep running
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("\n\nStopping services...")
    streamlit_process.terminate()
    api_process.terminate()
    print("Stopped")
