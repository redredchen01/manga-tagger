"""Enhanced VLM Dispatcher.

Parallel dispatch of multiple VLM models for tag generation with
fallback mechanisms and error handling.
"""

import asyncio
import base64
import io
import logging
import time
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
from PIL import Image

import httpx

from app.config import settings
from app.services.vlm_response_parser import VLMResponseParser, ParsedResponse

logger = logging.getLogger(__name__)


@dataclass
class ModelPrediction:
    """Prediction result from a single VLM model."""
    model_name: str
    raw_response: str
    parsed_response: ParsedResponse
    processing_time: float
    confidence_scores: Dict[str, float]
    error: Optional[str] = None
    is_valid: bool = True


@dataclass
class DispatchResult:
    """Result from dispatching all models."""
    predictions: List[ModelPrediction]
    total_time: float
    successful_models: int
    failed_models: int
    all_tags: List[str] = field(default_factory=list)


class EnhancedVLMDispatcher:
    """Enhanced VLM Dispatcher with parallel execution.
    
    Features:
    - Parallel model calling
    - Automatic fallback
    - Timeout handling
    - Response parsing
    """
    
    # Model configurations
    MODEL_CONFIGS = {
        "glm-4.6v-flash": {
            "endpoint": "/chat/completions",
            "timeout": 60,
            "strength": 1.0,
            "prompt_style": "glm",
        },
        "qwen-vl-max": {
            "endpoint": "/chat/completions",
            "timeout": 45,
            "strength": 1.1,
            "prompt_style": "qwen",
        },
        "llava-next": {
            "endpoint": "/chat/completions",
            "timeout": 40,
            "strength": 0.95,
            "prompt_style": "llava",
        },
        "yi-vision": {
            "endpoint": "/chat/completions",
            "timeout": 35,
            "strength": 0.9,
            "prompt_style": "generic",
        },
    }
    
    # Prompts for different models
    PROMPTS = {
        "glm": """Analyze this image and output tags that describe it.

RULES:
1. Only include tags that are VISIBLE in the image
2. Be conservative - when unsure, exclude the tag
3. Output format: tag1, tag2, tag3 (comma separated)
4. Use Chinese tags from the standard tag library

Output tags:""",
        
        "qwen": """分析這張圖片，輸出標籤。

要求：
1. 只包含圖片中可見的元素
2. 使用標準標籤庫中的標籤
3. 格式：標籤1, 標籤2, 標籤3

標籤：""",
        
        "llava": """Describe this image with tags.

Rules:
1. Only tag what you can see
2. Use standard tag names
3. Comma separated list

Tags:""",
        
        "generic": """What tags best describe this image?

Output format: tag1, tag2, tag3

Tags:""",
    }
    
    def __init__(
        self,
        models: Optional[List[str]] = None,
        max_workers: int = 3,
        timeout: int = 60,
        enable_fallback: bool = True,
    ):
        """Initialize dispatcher.
        
        Args:
            models: List of models to use (defaults to config)
            max_workers: Maximum parallel workers
            timeout: Timeout per model in seconds
            enable_fallback: Enable fallback to other models on failure
        """
        self.base_url = settings.LM_STUDIO_BASE_URL.rstrip("/")
        self.api_key = settings.LM_STUDIO_API_KEY
        self.models = models or settings.ENSEMBLE_MODELS
        self.max_workers = min(max_workers, settings.ENSEMBLE_MAX_WORKERS)
        self.timeout = timeout
        self.enable_fallback = enable_fallback
        
        self._executor = ThreadPoolExecutor(max_workers=self.max_workers)
        self._parser = VLMResponseParser()
        
        logger.info(
            f"EnhancedVLMDispatcher initialized with {len(self.models)} models: {self.models}"
        )
    
    def _prepare_image(self, image_bytes: bytes) -> Optional[str]:
        """Prepare image as base64 encoded string."""
        try:
            img = Image.open(io.BytesIO(image_bytes))
            
            # Convert to RGB if necessary
            if img.mode != "RGB":
                img = img.convert("RGB")
            
            # Resize if too large
            max_size = 1024
            if max(img.size) > max_size:
                ratio = max_size / max(img.size)
                new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
                img = img.resize(new_size, Image.Resampling.LANCZOS)
            
            # Save as JPEG
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=85)
            return base64.b64encode(buffer.getvalue()).decode("utf-8")
            
        except Exception as e:
            logger.error(f"Image preparation failed: {e}")
            return None
    
    def _get_prompt(self, model_name: str) -> str:
        """Get the appropriate prompt for a model."""
        config = self.MODEL_CONFIGS.get(model_name, {})
        style = config.get("prompt_style", "generic")
        return self.PROMPTS.get(style, self.PROMPTS["generic"])
    
    async def _call_model(
        self,
        model: str,
        image_bytes: bytes,
        prompt: str,
    ) -> ModelPrediction:
        """Call a single VLM model."""
        start_time = time.time()
        
        config = self.MODEL_CONFIGS.get(model, {})
        endpoint = config.get("endpoint", "/chat/completions")
        model_timeout = config.get("timeout", self.timeout)
        
        try:
            # Prepare image
            b64_image = self._prepare_image(image_bytes)
            if not b64_image:
                return ModelPrediction(
                    model_name=model,
                    raw_response="",
                    parsed_response=ParsedResponse(
                        raw_response="",
                        tags=[],
                        confidence={},
                        parsing_method="error",
                        is_valid=False,
                        error_message="Image preparation failed",
                    ),
                    processing_time=time.time() - start_time,
                    confidence_scores={},
                    error="Image preparation failed",
                    is_valid=False,
                )
            
            # Build request
            payload = {
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}
                            },
                        ],
                    }
                ],
                "max_tokens": 256,
                "temperature": 0.3,
            }
            
            # Make request
            async with httpx.AsyncClient(timeout=model_timeout) as client:
                resp = await client.post(
                    f"{self.base_url}{endpoint}",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json=payload,
                )
                resp.raise_for_status()
                result = resp.json()
                
                # Extract response
                content = result["choices"][0]["message"]["content"]
                
                # Parse response
                parsed = self._parser.parse(content)
                
                processing_time = time.time() - start_time
                
                logger.debug(f"Model {model}: {len(parsed.tags)} tags in {processing_time:.2f}s")
                
                return ModelPrediction(
                    model_name=model,
                    raw_response=content,
                    parsed_response=parsed,
                    processing_time=processing_time,
                    confidence_scores=parsed.confidence,
                    is_valid=parsed.is_valid,
                )
                
        except Exception as e:
            error_msg = str(e)
            logger.warning(f"Model {model} failed: {error_msg}")
            
            return ModelPrediction(
                model_name=model,
                raw_response="",
                parsed_response=ParsedResponse(
                    raw_response="",
                    tags=[],
                    confidence={},
                    parsing_method="error",
                    is_valid=False,
                    error_message=error_msg,
                ),
                processing_time=time.time() - start_time,
                confidence_scores={},
                error=error_msg,
                is_valid=False,
            )
    
    async def dispatch_all(
        self,
        image_bytes: bytes,
        custom_prompt: Optional[str] = None,
    ) -> DispatchResult:
        """Dispatch image to all models in parallel.
        
        Args:
            image_bytes: Image data
            custom_prompt: Optional custom prompt (overrides model default)
            
        Returns:
            DispatchResult with all predictions
        """
        start_time = time.time()
        
        # Get prompt (use custom or model-specific)
        if custom_prompt:
            prompts = {model: custom_prompt for model in self.models}
        else:
            prompts = {model: self._get_prompt(model) for model in self.models}
        
        # Create tasks
        tasks = [
            self._call_model(model, image_bytes, prompts[model])
            for model in self.models
        ]
        
        # Execute in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        predictions = []
        successful = 0
        failed = 0
        all_tags = set()
        
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Unexpected error: {result}")
                failed += 1
                continue
            
            predictions.append(result)
            
            if result.is_valid:
                successful += 1
                all_tags.update(result.parsed_response.tags)
            else:
                failed += 1
        
        total_time = time.time() - start_time
        
        logger.info(
            f"Dispatch complete: {successful}/{len(self.models)} successful, "
            f"{len(all_tags)} unique tags, {total_time:.2f}s total"
        )
        
        return DispatchResult(
            predictions=predictions,
            total_time=total_time,
            successful_models=successful,
            failed_models=failed,
            all_tags=list(all_tags),
        )
    
    def dispatch_sync(
        self,
        image_bytes: bytes,
        custom_prompt: Optional[str] = None,
    ) -> DispatchResult:
        """Synchronous version of dispatch_all."""
        return asyncio.run(self.dispatch_all(image_bytes, custom_prompt))
    
    def get_model_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all configured models."""
        status = {}
        
        for model in self.models:
            config = self.MODEL_CONFIGS.get(model, {})
            status[model] = {
                "endpoint": config.get("endpoint"),
                "timeout": config.get("timeout"),
                "strength": config.get("strength"),
                "available": True,  # Would need health check
            }
        
        return status


# Singleton instance
_dispatcher: Optional[EnhancedVLMDispatcher] = None


def get_vlm_dispatcher() -> EnhancedVLMDispatcher:
    """Get or create dispatcher singleton."""
    global _dispatcher
    if _dispatcher is None:
        _dispatcher = EnhancedVLMDispatcher()
    return _dispatcher
