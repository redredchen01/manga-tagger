"""Ollama VLM Service with optimized tag extraction."""

import base64
import io
import logging
import time
from typing import Any, Dict, List, Optional

import httpx

from app.core.cache import cache_manager
from app.core.config import settings
from app.core.metrics import CACHE_HITS, CACHE_MISSES, VLM_LATENCY, VLM_REQUEST_COUNT
from app.core.http_client import get_http_client
from app.domain.prompts import get_optimized_prompt
from app.domain.tag.parser import (
    extract_tags_from_description,
    extract_tags_from_reasoning,
    get_fallback_metadata,
    get_mock_metadata,
    parse_response,
)

logger = logging.getLogger(__name__)


class OllamaVLMService:
    """Ollama Vision-Language Model service for extracting visual metadata."""

    def __init__(self):
        """Initialize Ollama VLM service."""
        self.base_url = settings.OLLAMA_BASE_URL.rstrip("/")
        self.model = settings.OLLAMA_VISION_MODEL
        self.timeout = settings.REQUEST_TIMEOUT
        self.max_tokens = settings.VLM_MAX_TOKENS
        self.temperature = settings.TEMPERATURE

        logger.info(f"Ollama VLM service initialized with model: {self.model}")

    def _prepare_image(self, image_bytes: bytes) -> bytes:
        """Prepare image for Ollama processing."""
        try:
            from PIL import Image

            image = Image.open(io.BytesIO(image_bytes))

            if image.mode != "RGB":
                image = image.convert("RGB")

            # Resize if too large (Ollama has more conservative limits)
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
            raise ValueError(f"Invalid image data: {e}") from None

    def _encode_image_to_base64(self, image_bytes: bytes) -> str:
        """Encode image bytes to base64 string."""
        return base64.b64encode(image_bytes).decode("utf-8")

    async def extract_metadata(self, image_bytes: bytes) -> Dict[str, Any]:
        """
        Extract visual metadata from manga cover using Ollama.
        Returns structured features for tag matching.
        """
        start_time = time.time()
        status = "success"

        # Check if mock mode is enabled for testing
        if settings.USE_MOCK_SERVICES:
            logger.info("Using mock VLM service for testing")
            return get_mock_metadata()

        # Check cache first
        cache_key = cache_manager._make_key("vlm", image_bytes.hex())
        cached = await cache_manager.get(cache_key)
        if cached:
            logger.debug("VLM cache hit")
            CACHE_HITS.labels(cache_type="vlm").inc()
            return cached
        CACHE_MISSES.labels(cache_type="vlm").inc()

        # Image preparation
        try:
            prepared_image = self._prepare_image(image_bytes)
            base64_image = self._encode_image_to_base64(prepared_image)
        except Exception as e:
            logger.error(f"Failed to prepare image for VLM: {e}")
            return get_fallback_metadata(f"Image preparation failed: {e}")

        # Try VLM but with short timeout
        try:
            prompt = get_optimized_prompt()

            # Ollama uses /api/generate with images array
            payload = {
                "model": self.model,
                "prompt": prompt,
                "images": [base64_image],
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens,
                },
            }

            headers = {
                "Content-Type": "application/json",
            }

            client = await get_http_client()
            response = await client.post(
                f"{self.base_url}/api/generate",
                headers=headers,
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            result = response.json()

            # Ollama returns { "response": "..." }
            content = result.get("response", "")

            if content and len(content.strip()) > 5:
                logger.info(f"VLM succeeded: {content[:100]}...")

                # Parse the response and clean up junk
                parsed_metadata = parse_response(content)

                # Cache successful result
                await cache_manager.set(cache_key, parsed_metadata)

                return parsed_metadata

        except httpx.TimeoutException:
            logger.warning(f"Ollama VLM call timed out after {self.timeout}s")
            status = "timeout"
            return get_fallback_metadata("VLM timeout - using RAG fallback")

        except httpx.HTTPStatusError as e:
            logger.warning(f"Ollama VLM HTTP error: {e.response.status_code} - {e}")
            status = "error"
            return get_fallback_metadata(
                f"VLM HTTP error {e.response.status_code} - using RAG fallback"
            )

        except Exception as e:
            logger.warning(f"Ollama VLM call failed ({type(e).__name__}): {e}")
            status = "error"
            # Return fallback - RAG will handle tagging
            logger.info("VLM unavailable, returning fallback. RAG will provide tags.")
            return get_fallback_metadata("VLM unavailable - using RAG fallback")

        finally:
            # Record metrics
            duration = time.time() - start_time
            VLM_REQUEST_COUNT.labels(status=status).inc()
            VLM_LATENCY.observe(duration)

        # This should not be reached but satisfies type checker
        return get_fallback_metadata("VLM returned empty response")

    def _normalize_sensitive_tag(self, tag: str) -> str:
        """Normalize localized tags to canonical English identifiers."""
        alias_map = {
            "蘿莉": "loli",
            "低含量蘿莉": "loli",
            "肛交": "anal",
            "高含量肛交": "anal",
            "亂倫": "incest",
            "低含量亂倫": "incest",
            "觸手": "tentacles",
            "獵奇": "guro",
            "食人": "vore",
            "人妻": "milf",
            "老太婆": "milf",
            "調教": "bdsm",
            "奴隸": "bdsm",
            "自慰": "masturbation",
            "乳交": "paizuri",
            "獸交": "bestiality",
        }
        return alias_map.get(tag, tag)

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

            # Specific prompts for sensitive tags - expanded list
            # Map Chinese tags to English equivalents with STRICT verification
            tag_lower = self._normalize_sensitive_tag(tag.lower())

            # Very sensitive: requires explicit visual evidence
            if tag_lower in ["loli", "蘿莉", "lolicon"]:
                question = "Is there EXPLICIT VISUAL EVIDENCE that the character is a child (prepubescent, child-like body)? Look for: small frame, child face, no sexual development. Answer ONLY with YES or NO."
            elif tag_lower in ["shota", "正太", "shotacon"]:
                question = "Is there EXPLICIT VISUAL EVIDENCE that the character is a young boy (child)? Look for: child face, child body, pre-pubescent. Answer ONLY with YES or NO."
            elif tag_lower in ["infant", "嬰兒", "baby"]:
                question = "Is there EXPLICIT VISUAL EVIDENCE of an infant or very young child (under 5 years)? Answer ONLY with YES or NO."
            elif tag_lower in ["anal", "肛交"]:
                question = "Does this image explicitly and graphically depict anal intercourse? Answer ONLY with YES or NO."
            elif tag_lower in ["rape", "強姦", "non-consensual", "強制"]:
                question = "Does this image explicitly depict sexual assault or non-consensual sex? Answer ONLY with YES or NO."
            elif tag_lower in ["incest", "亂倫", "近親"]:
                question = "Does this image explicitly depict incest (sexual act between family members)? Answer ONLY with YES or NO."
            elif tag_lower in ["bdsm", "sm", "調教"]:
                question = "Does this image contain EXPLICIT sexual content related to BDSM (bondage, dominance, sadism, masochism)? Answer ONLY with YES or NO."
            elif tag_lower in ["tentacles", "觸手", "触手"]:
                question = "Does this image contain EXPLICIT sexual acts involving tentacles or tentacle-like appendages? Answer ONLY with YES or NO."
            elif tag_lower in ["guro", "グロ", "血腥", "斷肢"]:
                question = "Does this image contain EXTREME GORE or graphic mutilation? Answer ONLY with YES or NO."
            elif tag_lower in ["vore", "吞食", "吃人"]:
                question = "Does this image contain graphic cannibalism or vore (swallowing whole)? Answer ONLY with YES or NO."
            elif tag_lower in ["ntr", "netorare", "綠帽"]:
                question = "Does this image explicitly depict cuckold/infidelity sexual content? Answer ONLY with YES or NO."
            elif tag_lower in ["milf", "熟女"]:
                question = "Is the character explicitly depicted as a sexually mature adult woman (20+)? Answer ONLY with YES or NO."
            elif tag_lower in ["少女", "少年"]:
                question = "Is the character clearly a TEENAGER (13-19) or ADULT (20+)? Look for: body development, face features. Answer: TEEN, ADULT, or UNCERTAIN."
            elif tag_lower in ["olderWoman", "大姐姐", "人妻"]:
                question = "Is there EXPLICIT VISUAL EVIDENCE that this character is a mature adult woman (25+)? Look for: adult body, mature face. Answer ONLY with YES or NO."
            elif tag_lower in ["voyeur", "偷窺", "exhibition", "露出"]:
                question = "Does this image explicitly show voyeurism or exhibitionism? Answer ONLY with YES or NO."
            elif tag_lower in ["foot", "戀足", "smell", "體味"]:
                question = "Does this image contain EXPLICIT foot fetish or smell fetish content? Answer ONLY with YES or NO."
            elif tag_lower in [
                "masturbation",
                "自慰",
                "handjob",
                "手淫",
                "paizuri",
                "乳交",
            ]:
                question = "Does this image explicitly depict masturbation or sexual activity? Answer ONLY with YES or NO."
            elif tag_lower in ["group", "多人", "3p", "gangbang", "輪姦"]:
                question = "Does this image explicitly depict group sexual activity (3+ people)? Answer ONLY with YES or NO."
            elif tag_lower in ["creampie", "中出", "bukkake", "顏射", "cum", "射精"]:
                question = "Does this image explicitly depict sexual acts with visible ejaculation? Answer ONLY with YES or NO."
            else:
                question = f"Does this image contain EXPLICIT sexual content depicting '{tag}'? Answer ONLY with YES or NO."

            # Ollama uses /api/generate endpoint
            payload = {
                "model": self.model,
                "prompt": question,
                "images": [base64_image],
                "stream": False,
                "options": {
                    "temperature": 0.1,  # Low temperature for deterministic answer
                    "num_predict": 10,
                },
            }

            client = await get_http_client()
            response = await client.post(
                f"{self.base_url}/api/generate",
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            result = response.json()

            content = result.get("response", "").strip().upper()
            logger.info(f"Verification for '{tag}': {content}")

            # Strict check: Must start with YES or contain YES as a distinct word
            # to avoid matching "I cannot say yes" or "Maybe yes"
            if content.startswith("YES"):
                return True

            # Also check for "YES." or "YES," or just "YES"
            import re

            if re.search(r"\bYES\b", content):
                # But ensure no "NO" or "NOT" close to it if it's chatty
                if "NO" in content or "NOT" in content:
                    # Ambiguous response, default to False for safety
                    return False
                return True

            return False

        except Exception as e:
            logger.warning(f"Tag verification failed for '{tag}': {e}")
            # If verification fails, assume False to be safe (conservative approach)
            return False
