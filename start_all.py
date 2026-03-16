import subprocess
import sys
import time
import os

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
        "0.0.0.0",
        "--port",
        "8000",
    ],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
)

# Wait for API to start
time.sleep(3)

# Start Streamlit with environment variable for remote API access
print("[2] Starting Streamlit on port 8501...")
streamlit_env = os.environ.copy()
streamlit_env["STREAMLIT_API_URL"] = "http://25.8.89.95:8000/api/v1"  # Hamachi IP
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
        "0.0.0.0",  # Allow remote access to Streamlit
    ],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    env=streamlit_env,
)

print("\nServices started!")
print("API Server: http://0.0.0.0:8000 (accessible via Hamachi: http://25.8.89.95:8000)")
print("Streamlit: http://0.0.0.0:8501 (accessible via Hamachi: http://25.8.89.95:8501)")
print("\nNOTE: Both services are now accessible from remote machines!")
print("Press Ctrl+C to stop...")

try:
    # Keep running
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("\n\nStopping services...")
    streamlit_process.terminate()
    api_process.terminate()
    print("Stopped")
