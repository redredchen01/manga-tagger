#!/usr/bin/env python3
"""
Simple test client for the Manga Tagger API.
"""

import requests
import json
from pathlib import Path

def test_health():
    """Test health endpoint."""
    response = requests.get("http://localhost:8000/health")
    print("Health Check:", response.status_code)
    print(json.dumps(response.json(), indent=2))
    print()

def test_tags_list():
    """Test tags list endpoint."""
    response = requests.get("http://localhost:8000/tags")
    print("Tags List:", response.status_code)
    data = response.json()
    print(f"Total tags: {data.get('total', 0)}")
    if data.get('tags'):
        print("First 5 tags:")
        for tag in data['tags'][:5]:
            print(f"  - {tag['tag_name']}: {tag.get('description', 'No description')}")
    print()

def test_tag_cover():
    """Test tag cover endpoint with mock data."""
    # Create a mock image file (just a small text file for testing)
    mock_image_data = b"fake_image_data_for_testing"
    
    files = {'file': ('test.jpg', mock_image_data, 'image/jpeg')}
    data = {
        'top_k': 5,
        'confidence_threshold': 0.5,
        'include_metadata': True
    }
    
    response = requests.post("http://localhost:8000/tag-cover", files=files, data=data)
    print("Tag Cover:", response.status_code)
    print(json.dumps(response.json(), indent=2))
    print()

def test_rag_stats():
    """Test RAG stats endpoint."""
    response = requests.get("http://localhost:8000/rag/stats")
    print("RAG Stats:", response.status_code)
    print(json.dumps(response.json(), indent=2))
    print()

def main():
    """Run all tests."""
    base_url = "http://localhost:8000"
    
    print("Testing Manga Tagger API")
    print("=" * 50)
    
    try:
        test_health()
        test_tags_list()
        test_rag_stats()
        test_tag_cover()
        
        print("All tests completed successfully!")
        
    except requests.exceptions.ConnectionError:
        print("Could not connect to API server.")
        print("   Make sure the server is running on http://localhost:8000")
        print("   Start with: python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload")
    except Exception as e:
        print(f"Test failed with error: {e}")

if __name__ == "__main__":
    main()