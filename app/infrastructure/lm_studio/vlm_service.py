"""Enhanced LM Studio VLM Service with optimized tag extraction."""

import base64
import io
import json
import logging
import re
import time
from typing import Any, Dict, List, Optional

import httpx

from app.core.cache import cache_manager
from app.core.config import settings
from app.core.metrics import CACHE_HITS, CACHE_MISSES, VLM_LATENCY, VLM_REQUEST_COUNT
from app.core.http_client import get_http_client
from app.domain.prompts import get_structured_prompt
from app.domain.tag.parser import (
    get_fallback_metadata,
    get_mock_metadata,
)

logger = logging.getLogger(__name__)


_JSON_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*\n?(.*?)\n?```\s*$", re.DOTALL | re.IGNORECASE)
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_vlm_json(raw: str) -> dict | None:
    """Parse a VLM response into our canonical dict shape.

    Handles markdown fences and leading prose. Returns None if no valid
    JSON object can be extracted. Drops `tags` entries missing the `tag`
    field. Always returns at least {"description": ..., "tags": [...]}.
    """
    if not raw or not raw.strip():
        return None

    fenced = _JSON_FENCE_RE.match(raw)
    if fenced:
        raw = fenced.group(1)

    obj_match = _JSON_OBJECT_RE.search(raw)
    if not obj_match:
        return None

    try:
        data = json.loads(obj_match.group(0))
    except json.JSONDecodeError:
        return None

    if not isinstance(data, dict):
        return None

    description = data.get("description", "")
    if not isinstance(description, str):
        description = ""

    raw_tags = data.get("tags", [])
    if not isinstance(raw_tags, list):
        raw_tags = []

    cleaned_tags = []
    for t in raw_tags:
        if not isinstance(t, dict):
            continue
        tag_name = t.get("tag")
        if not isinstance(tag_name, str) or not tag_name.strip():
            continue
        cleaned_tags.append({
            "tag": tag_name.strip(),
            "category": t.get("category", ""),
            "confidence": float(t.get("confidence", 0.0)) if isinstance(t.get("confidence"), (int, float)) else 0.0,
            "evidence": t.get("evidence", ""),
        })

    return {"description": description, "tags": cleaned_tags}


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
            raise ValueError(f"Invalid image data: {e}") from None

    def _encode_image_to_base64(self, image_bytes: bytes) -> str:
        """Encode image bytes to base64 string."""
        return base64.b64encode(image_bytes).decode("utf-8")

    async def extract_metadata(self, image_bytes: bytes) -> Dict[str, Any]:
        """
        Extract visual metadata from manga cover using LM Studio.
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

        # Try VLM but with very short timeout
        try:
            from app.domain.tag.allowed_list import build_prompt_fragment
            from app.domain.tag.library import get_tag_library_service
            tag_lib = get_tag_library_service()
            allowed_fragment = build_prompt_fragment(tag_lib.tags)
            prompt = get_structured_prompt(allowed_fragment)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
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
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "stream": False,
            }

            # Short timeout - don't block server
            client = await get_http_client()
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
                # GLM-class models put analysis in reasoning_content if content is empty
                effective_content = content if content else reasoning

                parsed = parse_vlm_json(effective_content) if effective_content else None
                if parsed is not None:
                    logger.info(f"VLM succeeded: {len(parsed['tags'])} tags from JSON")
                    parsed["source"] = "vlm_json"
                    await cache_manager.set(cache_key, parsed)
                    return parsed

                # Parse failed — retry once with temperature=0.0 and a stricter system reminder
                logger.warning("VLM JSON parse failed; retrying once with temperature=0.0")
                payload["temperature"] = 0.0
                payload["messages"][0]["content"][0]["text"] += (
                    "\n\nIMPORTANT: previous attempt did not return valid JSON. "
                    "Output ONLY the JSON object, starting with { and ending with }. "
                    "No prose, no markdown."
                )
                response2 = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                )
                response2.raise_for_status()
                result2 = response2.json()
                if "choices" in result2 and result2["choices"]:
                    msg2 = result2["choices"][0]["message"]
                    content2 = msg2.get("content", "") or msg2.get("reasoning_content", "")
                    parsed2 = parse_vlm_json(content2)
                    if parsed2 is not None:
                        logger.info(f"VLM retry succeeded: {len(parsed2['tags'])} tags")
                        parsed2["source"] = "vlm_json_retry"
                        await cache_manager.set(cache_key, parsed2)
                        return parsed2

                logger.error("VLM JSON parse failed after retry; returning empty result")
                return {
                    "description": "",
                    "tags": [],
                    "source": "vlm_parse_fail",
                    "error": "VLM_PARSE_FAIL",
                }

        except Exception as e:
            logger.warning(f"VLM call failed ({type(e).__name__}): {e}")
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

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
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

            client = await get_http_client()
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

                if re.search(r"\bYES\b", content):
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
