"""LM Studio VLM Service for Stage 1: Visual Metadata Extraction.

Uses LM Studio vision models directly for image analysis.
"""

import base64
import io
import logging
from typing import Optional

import httpx

from app.config import settings
from app.models import VLMMetadata

logger = logging.getLogger(__name__)


class LMStudioVLMService:
    """LM Studio Vision-Language Model service for extracting visual metadata."""

    def __init__(self):
        """Initialize LM Studio VLM service."""
        self.base_url = settings.LM_STUDIO_BASE_URL.rstrip("/")
        self.api_key = settings.LM_STUDIO_API_KEY
        self.model = settings.LM_STUDIO_VISION_MODEL
        self.timeout = settings.REQUEST_TIMEOUT
        self.max_tokens = settings.VLM_MAX_TOKENS
        self.temperature = settings.TEMPERATURE

        logger.info(f"LM Studio VLM service initialized with model: {self.model}")

    def _prepare_image(self, image_bytes: bytes) -> bytes:
        """Prepare image for LM Studio processing."""
        try:
            from PIL import Image

            # Validate it's a valid image
            image = Image.open(io.BytesIO(image_bytes))

            # Convert to RGB if necessary
            if image.mode != "RGB":
                image = image.convert("RGB")

            # Resize if too large (to prevent processing issues)
            max_size = 1024
            if max(image.size) > max_size:
                ratio = max_size / max(image.size)
                new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
                image = image.resize(new_size, Image.Resampling.LANCZOS)

            # Convert back to bytes
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=85)
            return buffer.getvalue()

        except Exception as e:
            logger.error(f"Image preparation failed: {e}")
            raise ValueError(f"Invalid image data: {e}")

    def _encode_image_to_base64(self, image_bytes: bytes) -> str:
        """Encode image bytes to base64 string."""
        return base64.b64encode(image_bytes).decode("utf-8")

    async def extract_metadata(self, image_bytes: bytes) -> VLMMetadata:
        """
        Extract visual metadata from manga cover using LM Studio.

        Args:
            image_bytes: Raw image bytes

        Returns:
            VLMMetadata with structured visual information
        """
        try:
            # Prepare image
            prepared_bytes = self._prepare_image(image_bytes)
            base64_image = self._encode_image_to_base64(prepared_bytes)

            # Build messages for vision analysis (LM Studio format)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self._get_manga_analysis_prompt()},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        },
                    ],
                }
            ]

            # Make request to LM Studio
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }

            payload = {
                "model": self.model,
                "messages": messages,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
            }

            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions", headers=headers, json=payload
                )
                response.raise_for_status()
                result = response.json()

            if "choices" in result and len(result["choices"]) > 0:
                content = result["choices"][0]["message"]["content"]
                metadata = self._parse_response(content)
                return metadata
            else:
                logger.error(f"LM Studio VLM analysis failed: {result}")
                return self._get_fallback_metadata("LM Studio analysis failed")

        except Exception as e:
            logger.error(f"VLM extraction failed: {e}")
            # Try one more time with fallback
            try:
                logger.info("Retrying VLM extraction with fallback...")
                return self._get_fallback_metadata(f"Retry after error: {str(e)[:100]}")
            except:
                return self._get_fallback_metadata(f"VLM error: {str(e)[:100]}")

    def _get_manga_analysis_prompt(self) -> str:
        """Get comprehensive manga analysis prompt."""
        return """Please analyze this manga cover image in detail and provide structured information in Chinese.

Analyze and categorize:
1. 角色特徵 (Character Features): Age, gender, special characteristics (cat ears, wings, etc.)
2. 服裝與外觀 (Clothing & Appearance): Uniform, casual wear, special outfits
3. 場景與動作 (Scene & Action): Setting, pose, action, background elements
4. 風格與元素 (Style & Elements): Art style, mood, visual effects
5. 主题與類型 (Themes & Genres): Overall themes and genre indicators

Format your response as detailed analysis covering these categories."""

    def _parse_response(self, text: str) -> VLMMetadata:
        """Parse LM Studio response into structured metadata."""
        try:
            # Handle GLM model special tokens
            # If text is just special tokens, try to extract content between them
            if any(
                token in text
                for token in [settings.GLM_BEGIN_TOKEN, settings.GLM_END_TOKEN]
            ):
                # Remove special tokens and clean up
                import re

                # Remove GLM special tokens
                cleaned_text = re.sub(r"<\|[^|]+\|>", "", text)
                # Clean up extra whitespace
                cleaned_text = re.sub(r"\n\s*\n", "\n\n", cleaned_text).strip()
                text = cleaned_text

            # If still empty or too short, try different patterns
            if not text or len(text) < 10:
                # Try to find content after last special token
                if settings.GLM_END_TOKEN in text:
                    parts = text.split(settings.GLM_END_TOKEN)
                    if len(parts) > 1:
                        text = parts[-1].strip()

            # If still problematic, return with description
            if not text or len(text) < 5:
                return VLMMetadata(
                    description="Unable to analyze image",
                    characters=[],
                    themes=[],
                    art_style=None,
                    genre_indicators=[],
                )

            # Try to extract structured information from the response
            lines = text.strip().split("\n")

            description = text  # Full text as description
            characters = []
            themes = []
            genre_indicators = []
            art_style = None

            current_section = None

            # Extract information based on Chinese/manga classification prompts
            for line in lines:
                line = line.strip()
                if not line:
                    continue

                lower_line = line.lower()

                # Detect sections (support both Chinese and English)
                if any(
                    keyword in lower_line
                    for keyword in ["角色特徵", "character", "人物", "年龄", "性别"]
                ):
                    current_section = "characters"
                    continue
                elif any(
                    keyword in lower_line
                    for keyword in ["服裝與外觀", "clothing", "服装"]
                ):
                    current_section = "clothing"
                    continue
                elif any(
                    keyword in lower_line for keyword in ["場景與動作", "scene", "动作"]
                ):
                    current_section = "scene"
                    continue
                elif any(
                    keyword in lower_line for keyword in ["風格與元素", "style", "风格"]
                ):
                    current_section = "style"
                    continue
                elif any(keyword in lower_line for keyword in ["主题", "theme"]):
                    current_section = "themes"
                    continue

                # Skip bullet points and numbered lists
                if line.startswith(("-", "•", "*", "1.", "2.", "3.", "4.", "5.")):
                    continue

                # Extract content based on section
                if current_section == "characters":
                    # Extract character types
                    for kw in [
                        "蘿莉",
                        "少女",
                        "人妻",
                        "貓娘",
                        "兽人",
                        "精靈",
                        "魔法少女",
                        "巫女",
                    ]:
                        if kw in line and kw not in characters:
                            characters.append(kw)
                elif current_section == "clothing":
                    for kw in ["校服", "兔女郎", "泳裝", "内衣", "和服"]:
                        if kw in line and kw not in themes:
                            themes.append(kw)
                elif current_section == "scene":
                    for kw in ["學校", "校園", "幻想", "動作", "戰鬥"]:
                        if kw in line and kw not in themes:
                            themes.append(kw)
                elif current_section == "style":
                    if not art_style:
                        art_style = line[:100] if len(line) > 100 else line
                elif current_section == "themes":
                    if len(line) > 2 and len(line) < 100:
                        if line not in themes:
                            themes.append(line[:50])

            # If parsing failed, use heuristics on the full text
            if not characters and not themes:
                characters, themes, genre_indicators = self._extract_keywords_heuristic(
                    text
                )

            # If still empty, add descriptive keywords from the analysis
            if not characters and not themes:
                # Extract any known tags from description
                known_tags = [
                    "蘿莉",
                    "少女",
                    "人妻",
                    "貓娘",
                    "兽人",
                    "精靈",
                    "魔法少女",
                    "校服",
                    "兔女郎",
                    "泳裝",
                    "内衣",
                    "和服",
                    "學校",
                    "校園",
                    "幻想",
                    "動作",
                    "戰鬥",
                    "戀愛",
                ]
                for tag in known_tags:
                    if tag in text:
                        if tag in ["校服", "兔女郎", "泳裝", "内衣", "和服"]:
                            if tag not in themes:
                                themes.append(tag)
                        else:
                            if tag not in characters:
                                characters.append(tag)

            return VLMMetadata(
                description=description[:1000],  # Limit description length
                characters=list(set(characters)),  # Remove duplicates
                themes=list(set(themes)),
                art_style=art_style,
                genre_indicators=list(set(genre_indicators)),
            )

        except Exception as e:
            logger.warning(f"Failed to parse VLM response: {e}")
            # Return metadata with the original text as description
            return VLMMetadata(
                description=text[:1000] if text else "Parse error",
                characters=[],
                themes=[],
                art_style=None,
                genre_indicators=[],
            )

    def _extract_keywords_heuristic(self, text: str) -> tuple:
        """Extract keywords heuristically from text."""
        # Common manga/anime character types (Chinese)
        char_keywords = [
            "蘿莉",
            "少女",
            "人妻",
            "貓娘",
            "兽人",
            "精靈",
            "巫女",
            "魔法少女",
        ]
        # Common themes
        theme_keywords = ["學校", "校園", "恋愛", "戰鬥", "幻想", "動作", "喜劇"]
        # Common genres
        genre_keywords = ["校服", "兔女郎", "泳裝", "内衣", "和服", "眼鏡", "雙馬尾"]

        text_lower = text.lower()

        characters = [k for k in char_keywords if k in text_lower]
        themes = [k for k in theme_keywords if k in text_lower]
        genres = [k for k in genre_keywords if k in text_lower]

        return characters, themes, genres

    def _get_fallback_metadata(self, error_message: str) -> VLMMetadata:
        """Return fallback metadata when analysis fails."""
        return VLMMetadata(
            description=f"Analysis failed: {error_message}",
            characters=[],
            themes=[],
            art_style=None,
            genre_indicators=[],
        )
