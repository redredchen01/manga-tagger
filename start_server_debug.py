import subprocess
import sys
import time
import os

os.chdir(r"C:\tagger")

print("Starting API Server on port 8000...")
process = subprocess.Popen(
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
    stderr=subprocess.STDOUT,
    universal_newlines=True,
)

# Read output for 10 seconds
for i in range(20):
    line = process.stdout.readline()
    if line:
        print(line.strip())
    time.sleep(0.5)

print("\nServer should be running at http://127.0.0.1:8000")
print("Press Ctrl+C to stop...")

try:
    while True:
        line = process.stdout.readline()
        if line:
            print(line.strip())
except KeyboardInterrupt:
    print("\nStopping server...")
    process.terminate()
