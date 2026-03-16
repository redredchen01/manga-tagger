"""
GLM-4V Vision Client Module
Handles image analysis using GLM-4V model via OpenAI-compatible LM Studio endpoint
"""
import base64
import httpx
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
from PIL import Image
import io
import json


class GLM4VisionClient:
    """Client for GLM-4V vision model via OpenAI-compatible API"""
    
    def __init__(
        self,
        base_url: str = "http://localhost:1234/v1",
        api_key: str = "lm-studio",
        model: str = "glm-4v",
        timeout: float = 120.0,
        max_tokens: int = 2048,
        temperature: float = 0.7
    ):
        """
        Initialize GLM-4V client
        
        Args:
            base_url: LM Studio API endpoint URL
            api_key: API key (default for LM Studio)
            model: Model name
            timeout: Request timeout in seconds
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
        """
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.temperature = temperature
        
    def _encode_image_to_base64(self, image_path: Union[str, Path]) -> str:
        """
        Encode image file to base64 string
        
        Args:
            image_path: Path to image file
            
        Returns:
            Base64 encoded image string
        """
        with open(image_path, 'rb') as f:
            image_data = f.read()
        return base64.b64encode(image_data).decode('utf-8')
    
    def _resize_image_if_needed(
        self,
        image_path: Union[str, Path],
        max_size: tuple = (1024, 1024)
    ) -> bytes:
        """
        Resize image if it exceeds max dimensions
        
        Args:
            image_path: Path to image file
            max_size: Maximum (width, height)
            
        Returns:
            Resized image bytes
        """
        img = Image.open(image_path)
        
        # Convert to RGB if necessary
        if img.mode in ('RGBA', 'P'):
            img = img.convert('RGB')
        
        # Resize if needed
        if img.size[0] > max_size[0] or img.size[1] > max_size[1]:
            img.thumbnail(max_size, Image.Resampling.LANCZOS)
        
        # Save to bytes
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=85)
        return buffer.getvalue()
    
    def _get_mime_type(self, image_path: Union[str, Path]) -> str:
        """Get MIME type from image file extension"""
        ext = Path(image_path).suffix.lower()
        mime_types = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.webp': 'image/webp',
            '.bmp': 'image/bmp'
        }
        return mime_types.get(ext, 'image/jpeg')
    
    async def analyze_image(
        self,
        image_path: Union[str, Path],
        prompt: Optional[str] = None,
        system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze image using GLM-4V
        
        Args:
            image_path: Path to image file
            prompt: User prompt for image analysis
            system_prompt: System prompt for context
            
        Returns:
            Analysis result with description and metadata
        """
        # Default prompts for tagging task
        if prompt is None:
            prompt = """Please analyze this image and describe:
1. What characters or subjects are present
2. What actions or activities are happening
3. Any notable visual features, clothing, or attributes
4. The overall scene or setting

Provide a detailed description that would help identify relevant content tags."""
        
        if system_prompt is None:
            system_prompt = "You are an expert image analyzer specialized in content tagging and classification. Provide detailed, objective descriptions of visual content."
        
        # Prepare image
        image_bytes = self._resize_image_if_needed(image_path)
        image_b64 = base64.b64encode(image_bytes).decode('utf-8')
        mime_type = self._get_mime_type(image_path)
        
        # Build request payload
        messages = []
        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })
        
        messages.append({
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{image_b64}"
                    }
                }
            ]
        })
        
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature
        }
        
        # Make API request
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            result = response.json()
        
        # Extract and format response
        if 'choices' in result and len(result['choices']) > 0:
            content = result['choices'][0]['message']['content']
            return {
                'success': True,
                'description': content,
                'raw_response': result,
                'model': result.get('model', self.model),
                'usage': result.get('usage', {})
            }
        else:
            return {
                'success': False,
                'error': 'No response content from model',
                'raw_response': result
            }
    
    def analyze_image_sync(
        self,
        image_path: Union[str, Path],
        prompt: Optional[str] = None,
        system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """Synchronous version of analyze_image"""
        import asyncio
        return asyncio.run(self.analyze_image(image_path, prompt, system_prompt))
    
    async def batch_analyze(
        self,
        image_paths: List[Union[str, Path]],
        prompt: Optional[str] = None,
        system_prompt: Optional[str] = None,
        max_concurrent: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Analyze multiple images in batches
        
        Args:
            image_paths: List of image paths
            prompt: Analysis prompt
            system_prompt: System prompt
            max_concurrent: Maximum concurrent requests
            
        Returns:
            List of analysis results
        """
        import asyncio
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def analyze_with_limit(path):
            async with semaphore:
                return await self.analyze_image(path, prompt, system_prompt)
        
        tasks = [analyze_with_limit(path) for path in image_paths]
        return await asyncio.gather(*tasks)


class ImageTaggingPrompts:
    """Predefined prompts for image tagging tasks"""
    
    @staticmethod
    def get_tagging_prompt() -> str:
        """Get the standard tagging analysis prompt"""
        return """Please analyze this image carefully and provide a detailed description covering:

CHARACTERS & SUBJECTS:
- Number and types of characters (humans, humanoids, creatures, etc.)
- Physical features: age appearance, body type, skin color, hair style/color
- Distinguishing features: ears, tails, wings, horns, tattoos, piercings
- Clothing: type, style, amount (fully clothed, partially clothed, nude)

ACTIONS & ACTIVITIES:
- What are the characters doing?
- Any interactions between characters?
- Sexual activities (if any): type, positions, acts being performed

VISUAL ELEMENTS:
- Setting/environment
- Notable objects or props
- Special effects or visual styles
- Body modifications or transformations

Provide a comprehensive description that captures all relevant visual details for content classification."""

    @staticmethod
    def get_system_prompt() -> str:
        """Get the system prompt for tagging"""
        return "You are an expert content analyst specializing in detailed image description and tag identification. Provide objective, comprehensive visual analysis."
    
    @staticmethod
    def get_manga_classification_prompt() -> str:
        """Get the manga-specific classification prompt for generate_manga_description"""
        return """你是一個專業的漫畫分類員。請詳細描述這張圖片的視覺特徵，包括：

1. 角色特徵：
   - 角色數量與類型
   - 年齡外觀（蘿莉、人妻、少女等）
   - 體型特徵（巨乳、小胸部、豐滿等）
   - 種族/物種（人類、獸人、精靈、貓娘等）

2. 服裝與外觀：
   - 服裝類型（校服、兔女郎、泳裝、內衣等）
   - 特殊配件（眼鏡、貓耳、翅膀、尾巴等）
   - 頭髮顏色與髮型

3. 場景與動作：
   - 場景環境
   - 角色動作與姿態
   - 互動關係
   - 是否有性行為或親密場景

4. 風格與元素：
   - 畫風類型
   - 特殊效果
   - 標記或符號

請提供詳細的描述，以便進行準確的標籤分類與搜尋。"""

    @staticmethod
    def get_manga_system_prompt() -> str:
        """Get the system prompt for manga classification"""
        return "你是一個專業的漫畫內容分析師，擅長識別角色類型、服裝風格、場景類型和內容標籤。提供客觀、詳細的視覺分析，用於自動標籤系統。"


def generate_manga_description(
    image_path: str,
    base_url: str = "http://localhost:1234/v1",
    model: str = "glm-4v",
    timeout: float = 120.0
) -> Dict[str, Any]:
    """
    Generate a manga-specific description for an image.
    
    This function uses the specialized manga classification prompts to produce
    detailed descriptions optimized for RAG-based tag retrieval.
    
    Args:
        image_path: Path to the manga cover image
        base_url: LM Studio API endpoint URL
        model: Model name
        timeout: Request timeout in seconds
        
    Returns:
        Dictionary containing:
        - success: bool indicating if the request succeeded
        - description: Generated manga description
        - raw_response: Full API response
        - model: Model name used
        - usage: Token usage statistics
    """
    import asyncio
    
    client = GLM4VisionClient(
        base_url=base_url,
        model=model,
        timeout=timeout,
        max_tokens=2048,
        temperature=0.7
    )
    
    prompt = ImageTaggingPrompts.get_manga_classification_prompt()
    system_prompt = ImageTaggingPrompts.get_manga_system_prompt()
    
    try:
        result = asyncio.run(client.analyze_image(
            image_path=image_path,
            prompt=prompt,
            system_prompt=system_prompt
        ))
        return result
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'description': None,
            'raw_response': None
        }


# Convenience function
def create_vision_client(
    base_url: str = "http://localhost:1234/v1",
    model: str = "glm-4v"
) -> GLM4VisionClient:
    """
    Create a GLM-4V vision client
    
    Args:
        base_url: LM Studio API endpoint
        model: Model name
        
    Returns:
        Configured GLM4VisionClient instance
    """
    return GLM4VisionClient(base_url=base_url, model=model)


if __name__ == "__main__":
    # Test the client
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python glm4v_client.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    client = create_vision_client()
    
    print(f"Analyzing image: {image_path}")
    result = client.analyze_image_sync(image_path)
    
    if result['success']:
        print("\n--- Analysis Result ---")
        print(result['description'])
        print(f"\nTokens used: {result.get('usage', {})}")
    else:
        print(f"Error: {result.get('error')}")
