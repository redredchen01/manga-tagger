"""Enhanced LM Studio VLM Service with optimized tag extraction."""

import base64
import io
import logging
import re
from typing import Optional, Dict, Any, List

import httpx

from app.config import settings

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

            image = Image.open(io.BytesIO(image_bytes))

            if image.mode != "RGB":
                image = image.convert("RGB")

            # Resize if too large
            max_size = 1024
            if max(image.size) > max_size:
                ratio = max_size / max(image.size)
                new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
                image = image.resize(new_size, Image.Resampling.LANCZOS)

            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=85)
            return buffer.getvalue()

        except Exception as e:
            logger.error(f"Image preparation failed: {e}")
            raise ValueError(f"Invalid image data: {e}")

    def _encode_image_to_base64(self, image_bytes: bytes) -> str:
        """Encode image bytes to base64 string."""
        return base64.b64encode(image_bytes).decode("utf-8")

    async def extract_metadata(self, image_bytes: bytes) -> Dict[str, Any]:
        """
        Extract visual metadata from manga cover using LM Studio.
        Returns structured features for tag matching.
        """

        # Check if mock mode is enabled for testing
        if settings.USE_MOCK_SERVICES:
            logger.info("Using mock VLM service for testing")
            return self._get_mock_metadata()

        # Image preparation
        try:
            prepared_image = self._prepare_image(image_bytes)
            base64_image = self._encode_image_to_base64(prepared_image)
        except Exception as e:
            logger.error(f"Failed to prepare image for VLM: {e}")
            return self._get_fallback_metadata(f"Image preparation failed: {e}")

        # Try VLM but with very short timeout
        try:
            prompt = self._get_optimized_prompt()
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        },
                    ],
                }
            ]

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }

            payload = {
                "model": self.model,
                "messages": messages,
                "max_tokens": 150,
                "temperature": 0.3,
                "stream": False,
            }

            # Short timeout - don't block server
            async with httpx.AsyncClient(timeout=15) as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                )
                response.raise_for_status()
                result = response.json()

                if "choices" in result and len(result["choices"]) > 0:
                    message = result["choices"][0]["message"]
                    content = message.get("content", "")
                    reasoning = message.get("reasoning_content", "")

                    # GLM models put analysis in reasoning_content
                    effective_content = content if content else reasoning

                    if effective_content and len(effective_content.strip()) > 5:
                        logger.info(f"VLM succeeded: {effective_content[:100]}...")
                        
                        # Parse the response and clean up junk
                        parsed_metadata = self._parse_response(effective_content)
                        
                        # If reasoning content had interesting bullet points, blend them in
                        if reasoning and reasoning != content:
                            reasoning_tags = self._extract_tags_from_reasoning(reasoning)
                            if reasoning_tags:
                                # Merge with raw_keywords and deduplicate
                                current_tags = parsed_metadata.get("raw_keywords", [])
                                combined = list(set(current_tags + reasoning_tags))
                                parsed_metadata["raw_keywords"] = combined
                                logger.info(f"Enriched with {len(reasoning_tags)} tags from reasoning")
                        
                        return parsed_metadata

        except Exception as e:
            logger.debug(f"VLM call failed: {e}")

        # Return fallback - RAG will handle tagging
        logger.info("VLM unavailable, returning fallback. RAG will provide tags.")
        return self._get_fallback_metadata("VLM unavailable - using RAG fallback")

    def _get_optimized_prompt(self) -> str:
        """Get optimized prompt for tag extraction."""
        return """You are a highly precise vision-language model analyzing anime/manga images.
Your goal is to provide a detailed visual description AND extract specific tags.

Instructions:
1. **Detailed Description**: Write 2-3 sentences describing the characters, their clothing, hair, accessories, actions, and the environment.
2. **Tag List**: Extract a comma-separated list of concise tags based on the description.
3. **Strict Policy**: 
   - DO NOT include non-visual info like scanlation credits, chapter numbers, or URLs.
   - DO NOT include conversational filler like "The image shows".
   - **CRITICAL**: Do NOT use the tag 'loli' unless the character is CLEARLY a child (prepubescent). If in doubt, use 'teen' or 'young_girl'.
   - **CRITICAL**: Do NOT use the tag 'anal' unless there is EXPLICIT visual evidence of anal intercourse. Do not infer it from posture.

Format:
Description: [Your 2-3 sentence description]
Tags: [tag1, tag2, tag3...]

Response:"""

    def _parse_response(self, text: str) -> Dict[str, Any]:
        """Parse LM Studio response into structured metadata."""
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
            # Handle structured format if present
            content_to_parse = text.lower()
            description_text = text # Keep original for result
            
            if "tags:" in content_to_parse:
                parts = re.split(r"tags:", text, flags=re.IGNORECASE)
                if len(parts) >= 2:
                    description_text = parts[0].replace("Description:", "").strip()
                    tag_part = parts[1]
                else:
                    tag_part = text
            else:
                tag_part = text

            result["description"] = description_text
            
            # Clean up known junk phrases
            junk_phrases = [
                "the image shows", "this is a", "based on the", "analysis indicates",
                "i detected", "likely tags", "recommendations:", "also relevant",
                "here are some tags", "the following tags", "i suggest", "were detected",
                "extracted from", "vision analysis", "extracted tags", "tags detected",
                "description:", "tags:"
            ]
            text_clean = tag_part.lower()
            for phrase in junk_phrases:
                text_clean = text_clean.replace(phrase, "")
            
            # First try to extract tags from comma-separated list or newlines or dots
            # Split by comma, newline, or full stop followed by space
            raw_tags = re.split(r'[,\n]|\.\s', text_clean)
            
            tags = []
            junk_words = {
                "also", "relevant", "detected", "suggested", "tags", "image", "manga", 
                "cover", "vision", "analysis", "results", "showing", "with", "from",
                "translated", "scanlated", "credits", "chapter", "page", "raws", "version",
                "translates", "to", "of", "and", "the", "this", "that", "for", "in", "on",
                "at", "by", "with", "as", "is", "are", "was", "were", "be", "been", "being"
            }
            
            for tag in raw_tags:
                # Remove content before colon (e.g., "detected: loli" -> "loli")
                if ":" in tag:
                    tag = tag.split(":")[-1]
                
                clean_tag = tag.strip().strip('.').strip('"').strip("'").lower()
                # Filter rules: increase minimum length to 3 to avoid "to", "of", "2"" etc.
                if (clean_tag and 
                    not clean_tag.isdigit() and 
                    len(clean_tag) > 2 and 
                    clean_tag not in junk_words and
                    not any(jw in clean_tag.split() for jw in junk_words) and
                    len(clean_tag.split()) <= 4):
                    tags.append(clean_tag)

            # If no tags found this way, try extracting from descriptive text
            if len(tags) < 2:
                logger.info("No comma-separated tags found, extracting from keywords...")
                tags = self._extract_tags_from_description(text)

            result["raw_keywords"] = tags

            # Categorize tags
            character_keywords = ["loli", "shota", "teen", "mature", "milf", "catgirl", "doggirl", "foxgirl", "elf", "demon", "angel", "vampire"]
            clothing_keywords = ["school_uniform", "swimsuit", "bikini", "lingerie", "kimono", "maid_outfit", "nurse", "police_uniform", "bunny_suit", "dress", "shirt", "skirt", "panties", "bra"]
            body_keywords = ["flat_chest", "small_breasts", "large_breasts", "huge_breasts", "glasses", "stockings", "pantyhose", "knee_high_socks", "tattoo", "long_hair", "short_hair", "twintails"]
            action_keywords = ["sex", "oral", "paizuri", "handjob", "anal", "bondage", "tentacles", "rape", "prostitution", "masturbation"]
            theme_keywords = ["vanilla", "ntr", "netorare", "incest", "yuri", "yaoi", "harem", "school_life", "romance", "comedy", "drama"]

            for tag in tags:
                tag_lower = tag.lower()
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

        except Exception as e:
            logger.warning(f"Failed to parse response: {e}")

        return result

    def _extract_tags_from_reasoning(self, reasoning_text: str) -> List[str]:
        """Extract tags from GLM reasoning content."""
        tags = []
        import re
        bullet_pattern = r"[-]\s*([^\n]+)"
        bullets = re.findall(bullet_pattern, reasoning_text)

        for bullet in bullets:
            bullet_clean = bullet.strip().lower()
            if len(bullet_clean) < 50 and not any(x in bullet_clean for x in [".", "the ", "this ", "that "]):
                tag = bullet_clean.strip("\"' -")
                if tag and tag not in tags:
                    tags.append(tag)
        return tags

    def _extract_tags_from_description(self, text: str) -> List[str]:
        """Extract known tags from descriptive text using keyword matching."""
        text_lower = text.lower()
        found_tags = []
        
        # Simple keyword mapping for extraction
        keywords_map = {
            "loli": ["loli", "little girl", "child character"],  # Removed "young girl"
            "catgirl": ["catgirl", "cat ears"],
            "school_uniform": ["school uniform", "seifuku"],
            "swimsuit": ["swimsuit", "bikini"],
            "large_breasts": ["large breasts", "busty", "huge breasts"],
            "glasses": ["glasses"],
            "vanilla": ["vanilla", "pure love"],
            "ntr": ["ntr", "netorare"]
        }

        for tag, keywords in keywords_map.items():
            for kw in keywords:
                if kw in text_lower:
                    found_tags.append(tag)
                    break
        return found_tags

    def _get_fallback_metadata(self, error_message: str) -> Dict[str, Any]:
        """Return fallback metadata when analysis fails."""
        return {
            "description": f"Analysis failed: {error_message}",
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
        character_types = ["loli", "catgirl", "teen"]
        clothing = ["school_uniform", "swimsuit"]
        body_features = ["large_breasts", "glasses"]
        themes = ["vanilla", "romance"]

        selected_chars = random.sample(character_types, min(2, len(character_types)))
        selected_clothing = random.sample(clothing, min(1, len(clothing)))
        selected_body = random.sample(body_features, min(1, len(body_features)))
        selected_themes = random.sample(themes, min(1, len(themes)))

        all_keywords = selected_chars + selected_clothing + selected_body + selected_themes
        
        return {
            "description": f"Mock analysis: {', '.join(all_keywords)}",
            "character_types": selected_chars,
            "clothing": selected_clothing,
            "body_features": selected_body,
            "actions": [],
            "themes": selected_themes,
            "settings": [],
            "raw_keywords": all_keywords,
        }

    async def verify_sensitive_tag(self, image_bytes: bytes, tag: str) -> bool:
        """
        Verify a sensitive tag with a targeted VLM query.
        Returns True if the tag is verified, False otherwise.
        """
        try:
            # Skip verification if mock mode
            if settings.USE_MOCK_SERVICES:
                return True

            prepared_image = self._prepare_image(image_bytes)
            base64_image = self._encode_image_to_base64(prepared_image)

            # Specific prompts for sensitive tags
            # Map Chinese tags to English equivalents
            tag_lower = tag.lower()
            if tag_lower in ["loli", "蘿莉"]:
                question = "Is the character in this image explicitly a child (prepubescent)? Answer ONLY with YES or NO."
            elif tag_lower in ["anal", "肛交"]:
                question = "Does this image explicitly depict anal intercourse? Answer ONLY with YES or NO."
            elif tag_lower in ["rape", "強姦"]:
                 question = "Does this image explicitly depict rape or sexual assault? Answer ONLY with YES or NO."
            elif tag_lower in ["shota", "正太"]:
                 question = "Is the character in this image explicitly a young boy (child)? Answer ONLY with YES or NO."
            else:
                question = f"Does this image explicitly contain or depict '{tag}'? Answer ONLY with YES or NO."

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        },
                    ],
                }
            ]

            payload = {
                "model": self.model,
                "messages": messages,
                "max_tokens": 10,
                "temperature": 0.1,  # Low temperature for deterministic answer
                "stream": False,
            }

            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                )
                response.raise_for_status()
                result = response.json()

                if "choices" in result and len(result["choices"]) > 0:
                    content = result["choices"][0]["message"]["content"].strip().upper()
                    logger.info(f"Verification for '{tag}': {content}")
                    
                    # Strict check: Must start with YES or contain YES as a distinct word
                    # to avoid matching "I cannot say yes" or "Maybe yes"
                    if content.startswith("YES"):
                        return True
                    
                    # Also check for "YES." or "YES," or just "YES"
                    import re
                    if re.search(r'\bYES\b', content):
                        # But ensure no "NO" or "NOT" close to it if it's chatty
                        if "NO" in content or "NOT" in content:
                             # Ambiguous response, default to False for safety
                             return False
                        return True
                        
                    return False

            return False

        except Exception as e:
            logger.warning(f"Tag verification failed for '{tag}': {e}")
            # If verification fails, assume False to be safe (conservative approach)
            return False
