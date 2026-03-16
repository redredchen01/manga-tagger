"""LM Studio LLM Service for Stage 3: Tag Synthesis.

Uses LM Studio text models for intelligent tag synthesis from VLM metadata and RAG matches.
"""

import json
import logging
from typing import List, Dict, Any, Optional

import httpx

from app.config import settings
from app.utils import safe_confidence
from app.models import VLMMetadata, TagResult

logger = logging.getLogger(__name__)


class LMStudioLLMService:
    """LM Studio Large Language Model service for tag synthesis."""

    def __init__(self):
        """Initialize LM Studio LLM service."""
        self.base_url = settings.LM_STUDIO_BASE_URL.rstrip("/")
        self.api_key = settings.LM_STUDIO_API_KEY
        self.model = settings.LM_STUDIO_TEXT_MODEL
        self.max_tokens = settings.LLM_MAX_TOKENS
        self.temperature = settings.TEMPERATURE
        self.top_p = settings.TOP_P

        logger.info(f"LM Studio LLM service initialized with model: {self.model}")

    async def _make_request(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """Make request to LM Studio chat completion API."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }

        async with httpx.AsyncClient(timeout=settings.REQUEST_TIMEOUT) as client:
            response = await client.post(
                f"{self.base_url}/chat/completions", headers=headers, json=payload
            )
            response.raise_for_status()
            return response.json()

    def _build_prompt(
        self,
        vlm_metadata: VLMMetadata,
        rag_matches: List[Dict[str, Any]],
        available_tags: List[str],
    ) -> List[Dict[str, str]]:
        """Build messages for tag synthesis."""

        # Format VLM metadata
        vlm_section = f"""
Vision Analysis Results:
- Description: {vlm_metadata.description}
- Characters: {", ".join(vlm_metadata.characters) if vlm_metadata.characters else "None detected"}
- Themes: {", ".join(vlm_metadata.themes) if vlm_metadata.themes else "None detected"}
- Art Style: {vlm_metadata.art_style or "Not specified"}
- Genre Indicators: {", ".join(vlm_metadata.genre_indicators) if vlm_metadata.genre_indicators else "None detected"}
"""

        # Format RAG matches
        rag_section = "Similar Images from Database:\n"
        if rag_matches:
            for i, match in enumerate(rag_matches[:5], 1):
                rag_section += f"{i}. Similarity Score: {match.get('score', 0):.2f} - Tags: {', '.join(match.get('tags', []))}\n"
        else:
            rag_section += "No similar images found in database.\n"

        # Format available tags with definitions for semantic matching
        available_tags_section = "Candidate Tags for Selection (MUST CHOOSE FROM THESE):\n"
        tag_definitions = getattr(vlm_metadata, 'tag_definitions', {})
        
        for tag in available_tags:
            definition = tag_definitions.get(tag, "")
            if definition:
                available_tags_section += f"- {tag}: {definition}\n"
            else:
                available_tags_section += f"- {tag}\n"

        # Build system prompt - PRECISION FOCUSED
        system_prompt = """You are a PRECISE tag synthesis engine.

CRITICAL RULES:
1. Only select tags with VISUAL EVIDENCE in the description
2. When unsure, prefer NO TAG over uncertain tag
3. Less accurate tags = worse results

STRICT SAFETY RULES:
- Tag 'loli' ONLY if character is OBVIOUSLY prepubescent (child body)
- Tag 'anal' ONLY if EXPLICIT visual act shown
- Do NOT infer tags from posture or implied context

Output format:
RESULT_JSON:
{"tags": [{"tag": "tag_name", "confidence": 0.95, "reason": "visual evidence found"}]}"""
        system_prompt = """You are a precise JSON tag synthesis engine.

Rules:
1. KEEP YOUR REASONING EXTREMELY BRIEF (max 100 words).
2. After analysis, you MUST output 'RESULT_JSON:' followed by the JSON block.
3. Select ALL tags from 'Candidate Tags' that match the 'Description'.
4. Use 'Tag Definitions' to ensure accuracy.
5. **STRICT SAFETY RULES**:
   - Tag 'loli' ONLY if character is visibly PREPUBESCENT (child). Do NOT apply to teenagers/students.
   - Tag 'anal' ONLY if there is EXPLICIT VISUAL EVIDENCE of anal intercourse. Do NOT infer it.
6. Format:
RESULT_JSON:
{"tags": [{"tag": "tag1", "confidence": 0.95, "reason": "..."}]}"""

        # Build user prompt
        user_prompt = f"""Synthesize high-precision tags based on:

{vlm_section}

{rag_section}

{available_tags_section}

Return ONLY the final JSON results verified against visual evidence."""

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def _load_available_tags(self) -> List[str]:
        """Load available tags from tag library."""
        try:
            from pathlib import Path

            # Try to load tag library
            tag_path = Path(settings.TAG_LIBRARY_PATH)
            if not tag_path.exists():
                # Fallback to 51标签库.json
                tag_path = Path("51標籤庫.json")

            if tag_path.exists():
                with open(tag_path, "r", encoding="utf-8") as f:
                    tags_data = json.load(f)

                return [
                    tag.get("tag_name", "") for tag in tags_data if tag.get("tag_name")
                ]

        except Exception as e:
            logger.warning(f"Could not load tag library: {e}")

        # Fallback: return common manga tags
        return [
            "蘿莉",
            "少女",
            "人妻",
            "貓娘",
            "兽人",
            "精靈",
            "巫女",
            "魔法少女",
            "校服",
            "兔女郎",
            "泳裝",
            "内衣",
            "和服",
            "眼鏡",
            "雙馬尾",
            "學校",
            "校園",
            "恋愛",
            "戰鬥",
            "幻想",
            "動作",
            "喜劇",
            "巨乳",
            "爆乳",
            "貧乳",
            "美乳",
            "普通",
            "輕度",
        ]

    async def synthesize_tags(
        self,
        vlm_metadata: VLMMetadata,
        rag_matches: List[Dict[str, Any]],
        candidate_tags: List[str] = None,
        top_k: int = 5,
        confidence_threshold: float = 0.6,  # Raised for better precision
    ) -> List[TagResult]:
        """
        Synthesize final tags from VLM metadata and RAG matches using LM Studio.
        """
        try:
            # Use candidate tags if provided, otherwise load from library
            available_tags = candidate_tags if candidate_tags else self._load_available_tags()

            # Build prompt messages
            messages = self._build_prompt(vlm_metadata, rag_matches, available_tags)

            # Make request to LM Studio
            response = await self._make_request(messages)
            logger.debug(f"Full LM Studio response: {json.dumps(response, ensure_ascii=False)[:500]}...")

            # Parse response
            if "choices" in response and len(response["choices"]) > 0:
                msg = response["choices"][0]["message"]
                content = msg.get("content", "")
                reasoning = msg.get("reasoning_content", "")
                
                # If content is empty but reasoning has content, use reasoning as fallback
                # (Some models put the final answer or thought process there)
                final_text = content if content else reasoning
                
                logger.debug(f"Raw LLM synthesis response length: {len(final_text)}")
                logger.debug(f"LLM response content: {final_text[:300]}...")
                if reasoning and not content:
                    logger.debug("Using reasoning_content as fallback")
                
                tags = self._parse_synthesis_response(final_text, confidence_threshold, top_k)

                logger.info(f"LM Studio LLM synthesis produced {len(tags)} tags")
                
                if tags:
                    return tags[:top_k]
                else:
                    # LLM produced no valid tags — fall back to candidate tags
                    logger.warning("LLM synthesis produced 0 tags after parsing, returning candidate-based fallback")
                    if candidate_tags:
                        return [
                            TagResult(
                                tag=t,
                                confidence=safe_confidence(0.6),
                                source="llm_fallback",
                                reason="LLM synthesis failed to parse, kept from initial matching",
                            )
                            for t in candidate_tags[:top_k]
                        ]
                    return self._fallback_tags(rag_matches, confidence_threshold)[:top_k]
            else:
                logger.error("No response content from LM Studio")
                return self._fallback_tags(rag_matches, confidence_threshold)[:top_k]

        except Exception as e:
            logger.error(f"Tag synthesis failed: {e}")
            # Return fallback tags from RAG
            return self._fallback_tags(rag_matches, confidence_threshold)[:top_k]

    def _parse_synthesis_response(
        self, response: str, confidence_threshold: float, max_tags: int
    ) -> List[TagResult]:
        """Parse LM Studio JSON response into TagResult objects."""
        tags = []

        try:
            # Extract JSON from response (handling markdown if present)
            json_str = self._extract_json(response)
            
            if json_str:
                # Clean up some common issues like trailing commas or special control characters
                # but keep valid escaped newlines
                logger.debug(f"Attempting to parse JSON: {json_str[:100]}...")
                
                # Use strict=False to allow unescaped newlines in strings
                data = json.loads(json_str, strict=False)

                for tag_data in data.get("tags", []):
                    tag_name = tag_data.get("tag", "")
                    # LLM sometimes returns tag name in English or variations
                    if not tag_name: continue
                    
                    confidence = safe_confidence(tag_data.get("confidence", 0.0))
                    if confidence >= confidence_threshold:
                        tags.append(
                            TagResult(
                                tag=tag_name,
                                confidence=confidence,
                                source="lm_studio_llm",
                                reason=tag_data.get("reason", ""),
                            )
                        )
                
                if not tags:
                    logger.warning(f"No tags passed confidence threshold {confidence_threshold}")

            # Sort by confidence
            tags.sort(key=lambda x: x.confidence, reverse=True)
            return tags[:max_tags]

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LM Studio response as JSON: {e}")
            # Try to extract tags heuristically
            tags = self._extract_tags_heuristic(response, confidence_threshold)

        return tags[:max_tags]

    def _extract_json(self, text: str) -> Optional[str]:
        """Exract JSON string from text, stripping markdown if necessary."""
        # 0. Look for our custom RESULT_JSON: marker
        if "RESULT_JSON:" in text:
            text = text.split("RESULT_JSON:")[-1].strip()

        # 1. Look for markdown code blocks
        import re
        json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
        if json_match:
            return json_match.group(1)
            
        # 2. Look for first { and last }
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            return text[start:end]
            
        logger.debug(f"JSON extraction failed. Text snippet: {text[:200]}...")
        return None

    def _extract_tags_heuristic(
        self, response: str, threshold: float
    ) -> List[TagResult]:
        """Extract tags heuristically from non-JSON or broken JSON response."""
        tags = []
        
        # Method 1: Regex to find all {"tag": "...", "confidence": ...} objects
        import re
        # More flexible regex for tag and confidence pairs
        obj_matches = re.finditer(r'[\{\s]*"tag"\s*:\s*"([^"]+)"\s*,\s*"confidence"\s*:\s*([0-9.]+)', response)
        for match in obj_matches:
            tag_name = match.group(1)
            try:
                conf = float(match.group(2))
                if conf >= threshold:
                    tags.append(TagResult(
                        tag=tag_name,
                        confidence=safe_confidence(conf),
                        source="lm_studio_llm_heuristic",
                        reason="從 JSON 片段中修復提取"
                    ))
            except (ValueError, IndexError):
                continue
        
        if tags:
            return tags

        # Method 2: Line-based extraction for plain text
        lines = response.strip().split("\n")
        for line in lines:
            line = line.strip()
            if ":" in line or "-" in line:
                parts = line.replace("-", ":").split(":")
                if len(parts) >= 2:
                    tag_name = parts[0].strip(' "*_')
                    try:
                        # Extract first float found in the value part
                        conf_match = re.search(r'([0-9.]+)', parts[1])
                        if conf_match:
                            confidence = float(conf_match.group(1))
                            if confidence >= threshold:
                                tags.append(
                                    TagResult(
                                        tag=tag_name,
                                        confidence=safe_confidence(confidence),
                                        source="lm_studio_llm_text",
                                        reason="從純文字描述中提取",
                                    )
                                )
                    except (ValueError, IndexError):
                        continue
                    except ValueError:
                        continue

        return tags

    def _merge_with_rag(
        self,
        llm_tags: List[TagResult],
        rag_matches: List[Dict[str, Any]],
        threshold: float,
    ) -> List[TagResult]:
        """Merge LLM tags with high-confidence RAG tags."""
        if not rag_matches:
            return llm_tags

        # Add high-confidence RAG tags
        for match in rag_matches:
            score = match.get("score", 0.0)
            if score >= 0.8:  # High confidence threshold
                rag_tags = match.get("tags", [])
                for rag_tag in rag_tags:
                    # Check if tag already exists
                    if not any(t.tag == rag_tag for t in llm_tags):
                        llm_tags.append(
                            TagResult(
                                tag=rag_tag,
                                confidence=safe_confidence(score),
                                source="vlm+rag",
                                reason=f"在 {len(rag_matches)} 張相似圖像中找到",
                            )
                        )
                        if len(llm_tags) >= 10:  # Limit total tags
                            break

        # Sort by confidence and limit
        llm_tags.sort(key=lambda x: x.confidence, reverse=True)
        return llm_tags

    def _fallback_tags(
        self, rag_matches: List[Dict[str, Any]], threshold: float
    ) -> List[TagResult]:
        """Generate fallback tags from RAG matches."""
        tags = []

        if rag_matches:
            # Sort by similarity score
            sorted_matches = sorted(
                rag_matches, key=lambda x: x.get("score", 0), reverse=True
            )

            for match in sorted_matches[:5]:
                score = match.get("score", 0.0)
                if score >= threshold:
                    rag_tags = match.get("tags", [])
                    for rag_tag in rag_tags[:2]:  # Max 2 tags per match
                        tags.append(
                            TagResult(
                                tag=rag_tag,
                                confidence=safe_confidence(score),
                                source="rag",
                                reason=f"相似度匹配: {score:.2f}",
                            )
                        )
                        if len(tags) >= 5:
                            break

        return tags
