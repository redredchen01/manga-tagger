"""Example usage of the Manga Cover Auto-Tagger API."""

import requests
import json

BASE_URL = "http://localhost:8000"


def example_tag_cover():
    """Example: Tag a manga cover image."""
    image_path = "test_cover.jpg"  # Replace with actual image path
    
    with open(image_path, "rb") as f:
        files = {"file": ("cover.jpg", f, "image/jpeg")}
        data = {
            "top_k": 5,
            "confidence_threshold": 0.5,
            "include_metadata": True
        }
        
        response = requests.post(
            f"{BASE_URL}/tag-cover",
            files=files,
            data=data
        )
    
    if response.status_code == 200:
        result = response.json()
        print("Tags assigned:")
        for tag in result["tags"]:
            print(f"  - {tag['tag']}: {tag['confidence']:.2f} ({tag['source']})")
            print(f"    Reason: {tag.get('reason', 'N/A')}")
        
        if result.get("metadata"):
            print(f"\nProcessing time: {result['metadata']['processing_time']:.2f}s")
    else:
        print(f"Error: {response.status_code} - {response.text}")


def example_list_tags():
    """Example: List all available tags."""
    response = requests.get(f"{BASE_URL}/tags")
    
    if response.status_code == 200:
        result = response.json()
        print(f"Total tags available: {result['total']}")
        print("\nFirst 10 tags:")
        for tag in result["tags"][:10]:
            print(f"  - {tag['tag_name']}")
    else:
        print(f"Error: {response.status_code}")


def example_health_check():
    """Example: Check service health."""
    response = requests.get(f"{BASE_URL}/health")
    
    if response.status_code == 200:
        result = response.json()
        print(f"Status: {result['status']}")
        print("Models loaded:")
        for model, loaded in result["models_loaded"].items():
            print(f"  - {model}: {'✓' if loaded else '✗'}")
    else:
        print(f"Error: {response.status_code}")


def example_add_to_rag():
    """Example: Add image to RAG dataset."""
    image_path = "reference.jpg"  # Replace with actual image path
    tags = ["貓娘", "蘿莉", "校服"]
    
    with open(image_path, "rb") as f:
        files = {"file": ("reference.jpg", f, "image/jpeg")}
        data = {
            "tags": json.dumps(tags),
            "metadata": json.dumps({"source": "manual_upload"})
        }
        
        response = requests.post(
            f"{BASE_URL}/rag/add",
            files=files,
            data=data
        )
    
    if response.status_code == 200:
        result = response.json()
        print(f"Added successfully: {result['id']}")
        print(f"Message: {result['message']}")
    else:
        print(f"Error: {response.status_code} - {response.text}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python example_client.py <command>")
        print("Commands: tag, tags, health, add")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "tag":
        example_tag_cover()
    elif command == "tags":
        example_list_tags()
    elif command == "health":
        example_health_check()
    elif command == "add":
        example_add_to_rag()
    else:
        print(f"Unknown command: {command}")
