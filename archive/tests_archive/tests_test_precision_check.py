import asyncio
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from app.services.lm_studio_vlm_service_v2 import LMStudioVLMService
from app.config import settings

async def test_sensitive_tag_verification():
    """
    Test that sensitive tag verification correctly rejects false positives.
    """
    print("Initializing VLM Service...")
    service = LMStudioVLMService()
    
    # Load a known safe image (assuming test_anime.jpg is a standard anime girl)
    image_path = Path("test_anime.jpg")
    if not image_path.exists():
        print(f"Error: {image_path} not found.")
        return

    with open(image_path, "rb") as f:
        image_bytes = f.read()
    
    # Test 1: Verify '肛交' (Anal) on a safe image
    print("\n--- Test 1: Verifying '肛交' (Anal) on safe image ---")
    print("Expected: False (Should retrieve NO from VLM)")
    # We pass the Chinese tag, which should be mapped to English question in VLM service
    result_anal = await service.verify_sensitive_tag(image_bytes, "肛交")
    print(f"Result: {result_anal}")
    
    if not result_anal:
        print("✅ PASS: '肛交' tag correctly rejected.")
    else:
        print("❌ FAIL: '肛交' tag was accepted (False Positive!).")

    # Test 2: Verify '蘿莉' (Loli) on a safe image 
    print("\n--- Test 2: Verifying '蘿莉' (Loli) on safe image ---")
    print("Expected: False (Should retrieve NO from VLM if character is teen/adult)")
    result_loli = await service.verify_sensitive_tag(image_bytes, "蘿莉")
    print(f"Result: {result_loli}")

    # Note: If test_anime.jpg IS a loli, this result being True is actually correct for the model,
    # but might mean we need a different test image to verify rejection of non-loli.
    # However, given the user complaint, we assume the system is being too aggressive.
    
    if not result_loli:
        print("✅ PASS: 'loli' tag rejected (or image identified as not-loli).")
    else:
        print("⚠️ NOTE: 'loli' tag was accepted. Check if image actually depicts a child.")


if __name__ == "__main__":
    if settings.USE_MOCK_SERVICES:
        print("Skipping test because USE_MOCK_SERVICES is True.")
    else:
        asyncio.run(test_sensitive_tag_verification())
