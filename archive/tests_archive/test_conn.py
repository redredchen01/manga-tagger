import requests
import json
try:
    print("Testing http://127.0.0.1:8000/api/v1/health...")
    r = requests.get("http://127.0.0.1:8000/api/v1/health", timeout=5)
    print(f"Status: {r.status_code}")
    print(f"Response: {r.text}")
except Exception as e:
    print(f"Error: {e}")

try:
    print("\nTesting http://localhost:8000/api/v1/health...")
    r = requests.get("http://localhost:8000/api/v1/health", timeout=5)
    print(f"Status: {r.status_code}")
except Exception as e:
    print(f"Error: {e}")
