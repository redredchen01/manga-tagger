
import asyncio
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.services.tag_mapper import get_tag_mapper
from app.services.tag_library_service import get_tag_library_service

def debug_mapping():
    mapper = get_tag_mapper()
    library = get_tag_library_service()
    
    term = "school uniform"
    mapped = mapper.to_chinese(term)
    print(f"'{term}' mapped to: '{mapped}'")
    
    if mapped:
        matches = library.match_tags_by_keywords([mapped], min_confidence=0.1)
        print(f"Library matches for '{mapped}': {matches}")
    else:
        # Try raw term
        matches = library.match_tags_by_keywords([term], min_confidence=0.1)
        print(f"Library matches for '{term}': {matches}")

if __name__ == "__main__":
    debug_mapping()
