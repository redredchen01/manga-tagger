
import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from app.services.lm_studio_vlm_service_v3 import LMStudioVLMService
from app.services.lm_studio_llm_service import LMStudioLLMService
from app.services.tag_recommender_service import TagRecommenderService
from app.models import VLMMetadata

async def verify_junk_filtering():
    print("Verification: Testing Junk Tag Filtering...")
    
    # 1. Simulate a "noisy" VLM response that previously caused issues
    noisy_vlm_content = """CHARACTER: Young girl, school uniform.
CLOTHING: Blue seifuku with white collar.
BODY: Pigtails, glasses.
ACTION: Standing and smiling.
THEME: School life.
INFO: This image was translated by ScanlateGroup. Page 2 of 50. High quality 2\" image."""

    vlm_service = LMStudioVLMService()
    parsed_metadata = vlm_service._parse_glm_response(noisy_vlm_content)
    
    print(f"Parsed Description: {parsed_metadata['description'][:50]}...")
    print(f"Raw Keywords extracted: {parsed_metadata['raw_keywords']}")
    
    # Check if 'translates' or '2\"' or 'scanlated' exists in keywords
    junk_found = any(junk in [k.lower() for k in parsed_metadata['raw_keywords']] 
                     for junk in ["translates", "scanlated", "page", "high"])
    
    if junk_found:
        print("❌ Warning: Junk keywords still found in raw extraction, but LLM might filter them.")
    else:
        print("✅ Success: Raw extraction is cleaner.")

    # 2. Test LLM Synthesis with the noisy description
    llm_service = LMStudioLLMService()
    vlm_meta_obj = VLMMetadata(
        description=parsed_metadata["description"],
        characters=parsed_metadata["character_types"],
        themes=parsed_metadata["themes"]
    )
    
    # Mock some RAG matches
    rag_matches = [
        {"score": 0.9, "tags": ["蘿莉", "校服"]}
    ]
    
    # We won't actually call the API here to avoid costs/timeouts in verification, 
    # but we will check the prompt construction.
    messages = llm_service._build_prompt(vlm_meta_obj, rag_matches, ["蘿莉", "校服", "眼鏡"])
    
    system_msg = messages[0]["content"]
    if "雜訊過濾" in system_msg and "translates" in system_msg:
        print("✅ Success: LLM prompt includes explicit junk filtering instructions.")
    else:
        print("❌ Failure: LLM prompt missing junk filtering instructions.")

    print("\nVerification Complete.")

if __name__ == "__main__":
    asyncio.run(verify_junk_filtering())
