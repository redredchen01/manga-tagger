import asyncio
import sys
import io
from PIL import Image
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from app.services.lm_studio_vlm_service_v2 import LMStudioVLMService
from app.config import settings

async def debug_blue_square():
    print("Initializing VLM Service...")
    service = LMStudioVLMService()
    
    # Create blue square image exactly like test_api_v2.py
    print("Creating blue square image...")
    img = Image.new("RGB", (224, 224), color="blue")
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    image_bytes = buffer.getvalue()
    
    print("\n--- Testing '蘿莉' (Loli) on Blue Square ---")
    # We want to see the LOGS, effectively. 
    # Since I can't see the logs easily, I will rely on the return value first, 
    # but I also want to call the private _prepare_image and _encode... to emulate verify_sensitive_tag internals if needed.
    
    # Call verify_sensitive_tag
    result = await service.verify_sensitive_tag(image_bytes, "蘿莉")
    print(f"Result for '蘿莉': {result}")

    print("\n--- Testing '肛交' (Anal) on Blue Square ---")
    result_anal = await service.verify_sensitive_tag(image_bytes, "肛交")
    print(f"Result for '肛交': {result_anal}")

if __name__ == "__main__":
    if settings.USE_MOCK_SERVICES:
        print("Skipping test because USE_MOCK_SERVICES is True.")
    else:
        asyncio.run(debug_blue_square())
