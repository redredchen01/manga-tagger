"""Enhanced LM Studio VLM Service with GLM-4.6V specific fixes."""

import base64
import io
import logging
import re
from typing import Optional, Dict, Any, List

import httpx

from app.config import settings

logger = logging.getLogger(__name__)


class LMStudioVLMService:
    """LM Studio Vision-Language Model service with GLM-4.6V optimizations."""

    def __init__(self):
        """Initialize LM Studio VLM service."""
        self.base_url = settings.LM_STUDIO_BASE_URL.rstrip("/")
        self.api_key = settings.LM_STUDIO_API_KEY
        self.model = settings.LM_STUDIO_VISION_MODEL
        self.timeout = 30  # Reduced timeout for better responsiveness
        self.max_tokens = 1024  # Reduced for GLM-4.6V
        self.temperature = 0.3  # Lower temperature for more consistent results

        logger.info(f"LM Studio VLM service initialized with model: {self.model}")

    def _prepare_image(self, image_bytes: bytes) -> bytes:
        """Prepare image for LM Studio processing with GLM-4.6V optimization."""
        try:
            from PIL import Image

            image = Image.open(io.BytesIO(image_bytes))

            # Convert to RGB if necessary
            if image.mode != "RGB":
                image = image.convert("RGB")

            # GLM-4.6V works best with smaller images for faster processing
            max_size = 800  # Reduced from 1024 for GLM-4.6V
            if max(image.size) > max_size:
                ratio = max_size / max(image.size)
                new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
                image = image.resize(new_size, Image.Resampling.LANCZOS)

            # Use PNG format for better quality with GLM models
            buffer = io.BytesIO()
            image.save(buffer, format="PNG", quality=95)
            return buffer.getvalue()

        except Exception as e:
            logger.error(f"Image preparation failed: {e}")
            raise ValueError(f"Invalid image data: {e}")

    def _encode_image_to_base64(self, image_bytes: bytes) -> str:
        """Encode image bytes to base64 string."""
        return base64.b64encode(image_bytes).decode("utf-8")

    async def extract_metadata(self, image_bytes: bytes) -> Dict[str, Any]:
        """
        Extract visual metadata from manga cover using LM Studio GLM-4.6V.
        Returns structured features for tag matching.
        """
        # Check if mock mode is enabled for testing
        if settings.USE_MOCK_SERVICES:
            logger.info("Using mock VLM service for testing")
            return self._get_mock_metadata()

        try:
            prepared_bytes = self._prepare_image(image_bytes)
            base64_image = self._encode_image_to_base64(prepared_bytes)

            # Use GLM-4.6V optimized prompt format
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self._get_glm_optimized_prompt()},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            },
                        },
                    ],
                }
            ]

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }

            # GLM-4.6V specific payload configuration
            payload = {
                "model": self.model,
                "messages": messages,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "top_p": 0.8,  # Slightly reduced for more focused responses
                "stream": False,
                "frequency_penalty": 0.1,
                "presence_penalty": 0.1,
                # Disable GLM thinking mode which can cause empty responses
                "thinking": {"type": "disabled"},
            }

            logger.info(f"Sending GLM-4.6V request to {self.base_url}/chat/completions")
            logger.debug(f"Payload: {payload}")

            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions", headers=headers, json=payload
                )
                response.raise_for_status()
                result = response.json()

            logger.debug(f"LM Studio response: {result}")

            if "choices" in result and len(result["choices"]) > 0:
                content = result["choices"][0]["message"]["content"]
                logger.info(f"GLM-4.6V raw response: {content[:200]}...")

                # Special handling for GLM responses
                if not content or content.isspace():
                    logger.warning("GLM-4.6V returned empty content")
                    return self._get_fallback_metadata("Empty response from model")

                metadata = self._parse_glm_response(content)
                return metadata
            else:
                logger.error(f"GLM-4.6V analysis failed: {result}")
                return self._get_fallback_metadata("Invalid response format")

        except httpx.TimeoutException:
            logger.error("GLM-4.6V request timed out")
            return self._get_fallback_metadata("Request timeout")
        except httpx.HTTPStatusError as e:
            logger.error(
                f"GLM-4.6V HTTP error: {e.response.status_code} - {e.response.text}"
            )
            return self._get_fallback_metadata(f"HTTP error: {e.response.status_code}")
        except Exception as e:
            logger.error(f"GLM-4.6V extraction failed: {e}")
            return self._get_fallback_metadata(str(e))

    def _get_glm_optimized_prompt(self) -> str:
        """Get GLM-4.6V optimized prompt for detailed visual description."""
        return """Describe this manga/anime image in detail for a professional tagger.
Focus on:
1. CHARACTER: Age, gender, hair style/color, eye color, expression.
2. CLOTHING: Specific outfit names (e.g., seifuku, bikini, maid dress), accessories.
3. BODY: Build, chest size, specific features (glasses, tattoos, wings).
4. ACTION/POSE: What is happening? (e.g., sitting, standing, embracing).
5. THEME/SETTING: Background elements, mood, genre hints.

IMPORTANT: Do NOT include scanlation credits, website names, or page numbers in your description. 
Provide a clear, structured technical description in English."""

    def _parse_glm_response(self, text: str) -> Dict[str, Any]:
        """Parse GLM-4.6V response with enhanced token handling."""
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
            # Enhanced GLM token cleaning
            original_text = text
            text = text.strip()

            # Remove GLM special tokens with better regex
            text = re.sub(r"<\|[^|]+\|>", "", text)  # Remove <|token|> patterns
            text = re.sub(
                r"[^\x20-\x7E\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff]", "", text
            )  # Keep ASCII, Chinese, Japanese
            text = re.sub(r"\s+", " ", text).strip()  # Normalize whitespace

            logger.debug(f"Cleaned text after token removal: '{text}'")

            # Check if response is empty after cleaning
            if not text or len(text) < 3:
                logger.warning(
                    f"GLM-4.6V returned empty/short response: '{original_text}'"
                )
                return self._get_fallback_metadata("Empty response after cleaning")

            # Check if response contains only control characters
            if not any(c.isalnum() for c in text):
                logger.warning(
                    f"GLM-4.6V returned non-readable content: '{original_text[:100]}'"
                )
                return self._get_fallback_metadata("Unreadable content")

            # Extract tags/keywords from the description
            # Since the prompt now asks for a description, we treat the description as source for keywords
            
            # Simple keyword extraction from the English description
            # This is a fallback in case LLM synthesis isn't used
            words = re.findall(r'\b[a-zA-Z_]{3,}\b', text.lower())
            
            # Common technical manga terms we want to preserve
            tech_terms = {
                "loli", "shota", "mature", "milf", "catgirl", "nekomimi", 
                "seifuku", "maid", "bikini", "lingerie", "kimono", 
                "glasses", "stockings", "pantyhose", "tattoos",
                "vanilla", "ntr", "yuri", "yaoi", "harem"
            }
            
            tags = [w for w in words if w in tech_terms or len(w) > 4]
            result["raw_keywords"] = tags

            # Enhanced categorization with more comprehensive keywords
            character_keywords = [
                "loli",
                "shota",
                "teen",
                "mature",
                "milf",
                "catgirl",
                "doggirl",
                "foxgirl",
                "elf",
                "demon",
                "angel",
                "vampire",
                "monster_girl",
                "young",
                "girl",
                "boy",
                "bunny",
                "milf",
                "mother",
                "adult",
                "woman",
            ]

            clothing_keywords = [
                "school_uniform",
                "swimsuit",
                "bikini",
                "lingerie",
                "kimono",
                "maid_outfit",
                "nurse",
                "police_uniform",
                "bunny_suit",
                "dress",
                "shirt",
                "skirt",
                "panties",
                "bra",
                "uniform",
            ]

            body_keywords = [
                "flat_chest",
                "small_breasts",
                "large_breasts",
                "huge_breasts",
                "glasses",
                "stockings",
                "pantyhose",
                "knee_high_socks",
                "tattoo",
                "long_hair",
                "short_hair",
                "twintails",
                "blonde",
                "brunette",
            ]

            action_keywords = [
                "sex",
                "oral",
                "paizuri",
                "handjob",
                "anal",
                "bondage",
                "tentacles",
                "rape",
                "prostitution",
                "masturbation",
            ]

            theme_keywords = [
                "vanilla",
                "ntr",
                "netorare",
                "incest",
                "yuri",
                "yaoi",
                "harem",
                "school_life",
                "romance",
                "comedy",
                "drama",
                "fantasy",
            ]

            for tag in tags:
                tag_lower = tag.lower()

                # More flexible matching with partial matches
                if any(kw in tag_lower for kw in character_keywords):
                    result["character_types"].append(tag)
                elif any(kw in tag_lower for kw in clothing_keywords):
                    result["clothing"].append(tag)
                elif any(kw in tag_lower for kw in body_keywords):
                    result["body_features"].append(tag)
                elif any(kw in tag_lower for kw in action_keywords):
                    result["actions"].append(tag)
                elif any(kw in tag_lower for kw in theme_keywords):
                    result["themes"].append(tag)

            # Remove duplicates while preserving order
            for key in [
                "character_types",
                "clothing",
                "body_features",
                "actions",
                "themes",
            ]:
                seen = set()
                result[key] = [x for x in result[key] if not (x in seen or seen.add(x))]

            logger.info(f"GLM-4.6V extracted {len(tags)} tags: {tags}")
            logger.info(
                f"Categorized - chars:{len(result['character_types'])}, "
                f"clothing:{len(result['clothing'])}, "
                f"body:{len(result['body_features'])}, "
                f"actions:{len(result['actions'])}, "
                f"themes:{len(result['themes'])}"
            )

        except Exception as e:
            logger.warning(f"Failed to parse GLM response: {e}")
            return self._get_fallback_metadata(f"Parse error: {str(e)}")

        return result

    def _get_fallback_metadata(self, error_message: str) -> Dict[str, Any]:
        """Return fallback metadata when analysis fails."""
        return {
            "description": f"GLM-4.6V analysis failed: {error_message}",
            "character_types": [],
            "clothing": [],
            "body_features": [],
            "actions": [],
            "themes": [],
            "settings": [],
            "raw_keywords": [],
        }

    def _get_mock_metadata(self) -> Dict[str, Any]:
        """Return mock metadata for testing when VLM is not available."""
        import random

        # Sample tags for demonstration
        character_types = ["loli", "catgirl", "teen"]
        clothing = ["school_uniform", "swimsuit"]
        body_features = ["large_breasts", "glasses"]
        actions = []
        themes = ["vanilla", "romance"]

        # Randomly select some tags
        selected_chars = random.sample(character_types, min(2, len(character_types)))
        selected_clothing = random.sample(clothing, min(1, len(clothing)))
        selected_body = random.sample(body_features, min(1, len(body_features)))
        selected_themes = random.sample(themes, min(1, len(themes)))

        all_keywords = (
            selected_chars + selected_clothing + selected_body + selected_themes
        )

        logger.info(f"Mock GLM-4.6V generated tags: {all_keywords}")

        return {
            "description": f"Mock GLM-4.6V analysis: {', '.join(all_keywords)}",
            "character_types": selected_chars,
            "clothing": selected_clothing,
            "body_features": selected_body,
            "actions": [],
            "themes": selected_themes,
            "settings": [],
            "raw_keywords": all_keywords,
        }
