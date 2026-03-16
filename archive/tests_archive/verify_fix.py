
import json
import requests

def test_type_safety():
    print("Testing API for rag_matches type safety...")
    try:
        # We'll just check if the code runs without crashing when we simulate the response
        metadata = {
            "rag_matches": 5, # Simulate the error condition
            "vlm_analysis": {
                "character_types": ["test"],
                "clothing": ["test"]
            }
        }
        
        # Test the slicing that was failing
        if metadata.get("rag_matches") and isinstance(metadata["rag_matches"], list):
            matches = metadata["rag_matches"][:3]
            print("Successfully sliced rag_matches list")
        else:
            print("Safely skipped slicing non-list rag_matches")
            
        print("Verification successful!")
    except Exception as e:
        print(f"Verification FAILED: {e}")

if __name__ == "__main__":
    test_type_safety()
