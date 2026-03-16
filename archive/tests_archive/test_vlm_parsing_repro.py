
import re
from typing import List, Dict, Any

class MockVLMService:
    def _parse_response(self, text: str) -> Dict[str, Any]:
        result = {
            "description": text,
            "character_types": [],
            "clothing": [],
            "body_features": [],
            "actions": [],
            "themes": [],
            "settings": [],
            "raw_keywords": [],
        }

        try:
            # Clean up known junk phrases
            junk_phrases = [
                "the image shows", "this is a", "based on the", "analysis indicates",
                "i detected", "likely tags", "recommendations:", "also relevant",
                "here are some tags", "the following tags", "i suggest"
            ]
            text_clean = text.lower()
            for phrase in junk_phrases:
                text_clean = text_clean.replace(phrase, "")
            
            # Split by comma or newline OR full stop followed by space
            raw_tags = re.split(r'[,\n]|\.\s', text_clean)
            
            tags = []
            junk_words = {"also", "relevant", "detected", "suggested", "tags", "image", "manga", "cover"}
            
            for tag in raw_tags:
                # Remove content before colon (e.g., "detected: loli" -> "loli")
                if ":" in tag:
                    tag = tag.split(":")[-1]
                
                clean_tag = tag.strip().strip('.').lower()
                # Filter rules
                if (clean_tag and 
                    not clean_tag.isdigit() and 
                    len(clean_tag) > 1 and 
                    clean_tag not in junk_words and
                    not any(clean_tag == jw for jw in junk_words) and
                    len(clean_tag.split()) <= 4):
                    tags.append(clean_tag)

            result["raw_keywords"] = tags
            
            # Simplified categorization for test
            character_keywords = ["loli", "catgirl", "teen"]
            for tag in tags:
                if any(kw in tag for kw in character_keywords):
                    result["character_types"].append(tag)
                    
        except Exception as e:
            print(f"Error: {e}")
            
        return result

def test_parsing():
    service = MockVLMService()
    
    test_inputs = [
        "Also, relevant, 2",
        "loli, catgirl, school_uniform",
        "The following tags were detected: loli, catgirl. These are also relevant.",
        "1, 2, 3, a, b, c",
        "a very long sentence that should not be a tag at all",
        "loli. catgirl. school_uniform"
    ]
    
    for inp in test_inputs:
        res = service._parse_response(inp)
        print(f"Input: {inp}")
        print(f"Raw Keywords: {res['raw_keywords']}")
        print(f"Character Types: {res['character_types']}")
        print("-" * 20)

if __name__ == "__main__":
    test_parsing()
