"""Tag Recommender Service for intelligent tag suggestions.

Combines VLM analysis, tag library matching, RAG results, and LLM synthesis
to recommend the best tags from the 611-tag library.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from app.services.tag_library_service import get_tag_library_service
from app.services.tag_mapper import get_tag_mapper
from app.utils import safe_confidence
from app.config import settings
from app.models import VLMMetadata, TagResult

logger = logging.getLogger(__name__)


@dataclass
class TagRecommendation:
    """A tag recommendation with confidence and reasoning."""

    tag: str
    confidence: float
    source: str  # 'vlm', 'rag', 'llm', 'library_match'
    reason: str


class TagRecommenderService:
    """Service for recommending tags based on multiple sources."""

    def __init__(self):
        """Initialize the recommender service."""
        self.tag_library = get_tag_library_service()
        self.tag_mapper = get_tag_mapper()

    async def recommend_tags(
        self,
        vlm_analysis: Dict[str, Any],
        rag_matches: List[Dict[str, Any]],
        top_k: int = 5,
        confidence_threshold: float = 0.5,
        vlm_service: Any = None,
        image_bytes: bytes = None,
    ) -> List[TagRecommendation]:
        """
        Recommend tags based on VLM analysis and RAG matches.

        Args:
            vlm_analysis: VLM analysis results with features
            rag_matches: RAG similarity search results
            top_k: Number of tags to return
            confidence_threshold: Minimum confidence threshold

        Returns:
            List of tag recommendations
        """
        recommendations: List[TagRecommendation] = []

        # Check if VLM analysis is valid (not empty/fallback)
        vlm_keywords = self._extract_vlm_keywords(vlm_analysis)
        vlm_is_valid = self._is_vlm_analysis_valid(vlm_analysis)

        if vlm_is_valid:
            logger.info(
                f"VLM analysis valid. Extracted {len(vlm_keywords)} keywords: {vlm_keywords}"
            )
        else:
            logger.warning("VLM analysis invalid or empty. Using RAG-only mode.")
            # Use description to extract keywords if available
            description = vlm_analysis.get("description", "")
            if description and "failed" not in description.lower():
                # Try to extract keywords from description even if VLM failed
                vlm_keywords = self._extract_keywords_from_text(description)
                logger.info(f"Extracted {len(vlm_keywords)} keywords from description")

        # 1.5 Map English keywords to Chinese
        mapped_keywords = []
        for kw in vlm_keywords:
            cn_tag = self.tag_mapper.to_chinese(kw)
            if cn_tag:
                mapped_keywords.append(cn_tag)
                logger.info(f"Mapped '{kw}' -> '{cn_tag}'")
            else:
                mapped_keywords.append(kw)

        logger.info(f"Mapped keywords: {mapped_keywords}")

        # 2. Match keywords to tag library
        library_matches = self.tag_library.match_tags_by_keywords(
            mapped_keywords, min_confidence=0.5
        )
        logger.info(f"Matched {len(library_matches)} tags from library via string matching")

        for tag_name, confidence in library_matches:
            if confidence >= confidence_threshold:
                recommendations.append(
                    TagRecommendation(
                        tag=tag_name,
                        confidence=safe_confidence(confidence),
                        source="library_match",
                        reason=f"Matched from VLM analysis: {', '.join(vlm_keywords[:3])}",
                    )
                )

        # 2.5 Semantic Search Matching (New Optimization)
        from app.services.chinese_embedding_service import get_chinese_embedding_service
        embedding_service = get_chinese_embedding_service()
        
        if embedding_service.is_available() and len(recommendations) < top_k:
            logger.info("Performing semantic search matching for additional tags...")
            
            # Ensure tag library embeddings are cached (first run will be slow, then fast)
            if not embedding_service._tag_matrix_cache:
                all_tags = self.tag_library.get_all_tags()
                await embedding_service.cache_tag_embeddings(all_tags)
            
            for keyword in mapped_keywords:
                semantic_matches = await embedding_service.search_cached_tags(
                    keyword, top_k=2, threshold=settings.CHINESE_EMBEDDING_THRESHOLD
                )
                
                for s_match in semantic_matches:
                    tag_name = s_match["tag"]
                    similarity = s_match["similarity"]
                    
                    # Check if already in recommendations
                    if not any(r.tag == tag_name for r in recommendations):
                        recommendations.append(
                            TagRecommendation(
                                tag=tag_name,
                                confidence=safe_confidence(similarity * 0.95), # Slight penalty for semantic match
                                source="semantic_match",
                                reason=f"Semantically related to '{keyword}' (score: {similarity:.2f})",
                            )
                        )
                        logger.info(f"Added semantic match: {tag_name} for '{keyword}'")

        # 3. Extract tags from RAG matches (always do this, even if VLM fails)
        rag_tags = self._extract_rag_tags(rag_matches, confidence_threshold)

        # If VLM failed but we have RAG matches, boost RAG confidence
        if not vlm_is_valid and rag_matches:
            logger.info(
                f"VLM failed but found {len(rag_matches)} RAG matches. Using RAG-based tagging."
            )
            # Lower threshold for RAG-only mode
            confidence_threshold = max(0.3, confidence_threshold - 0.2)

        for tag_name, confidence, match_info in rag_tags:
            # Check if already in recommendations
            existing = next((r for r in recommendations if r.tag == tag_name), None)
            if existing:
                # Boost confidence if found in both
                existing.confidence = safe_confidence(
                    min(existing.confidence + 0.1, 1.0)
                )
                existing.source = "vlm+rag"
                existing.reason += f" | Also found in RAG: {match_info}"
            else:
                recommendations.append(
                    TagRecommendation(
                        tag=tag_name,
                        confidence=safe_confidence(confidence),
                        source="rag",
                        reason=f"Found in similar images: {match_info}",
                    )
                )

        # 4. Add VLM character type tags if not already present
        character_types = vlm_analysis.get("character_types", [])
        for char_type in character_types:
            if char_type not in [r.tag for r in recommendations]:
                # Map to Chinese first
                cn_char_type = self.tag_mapper.to_chinese(char_type) or char_type
                # Try to match character type to library
                matches = self.tag_library.match_tags_by_keywords(
                    [cn_char_type], min_confidence=0.5
                )
                for tag_name, confidence in matches[:1]:  # Take best match
                    if confidence >= confidence_threshold:
                        recommendations.append(
                            TagRecommendation(
                                tag=tag_name,
                                confidence=safe_confidence(
                                    confidence * 0.9
                                ),  # Slightly lower for direct VLM
                                source="vlm",
                                reason=f"Detected character type: {char_type} -> {cn_char_type}",
                            )
                        )

        # 5. Add clothing tags
        clothing_items = vlm_analysis.get("clothing", [])
        for item in clothing_items:
            if item not in [r.tag for r in recommendations]:
                # Map to Chinese first
                cn_item = self.tag_mapper.to_chinese(item) or item
                matches = self.tag_library.match_tags_by_keywords(
                    [cn_item], min_confidence=0.5
                )
                for tag_name, confidence in matches[:1]:
                    if confidence >= confidence_threshold:
                        recommendations.append(
                            TagRecommendation(
                                tag=tag_name,
                                confidence=safe_confidence(confidence * 0.9),
                                source="vlm",
                                reason=f"Detected clothing: {item} -> {cn_item}",
                            )
                        )

        # 6. Sort by confidence and limit results
        recommendations.sort(key=lambda x: x.confidence, reverse=True)

        # 7. Deduplicate AND apply confidence threshold
        seen_tags = set()
        unique_recommendations = []
        for rec in recommendations:
            # Apply confidence threshold BEFORE adding
            if rec.confidence < confidence_threshold:
                continue

            if rec.tag not in seen_tags:
                seen_tags.add(rec.tag)
                unique_recommendations.append(rec)
                if len(unique_recommendations) >= top_k:
                    break

        # 8. If we have LM Studio enabled, refine and synthesize with LLM
        if settings.USE_LM_STUDIO:
            logger.info("Refining tags with LM Studio LLM synthesis...")
            # Collect all candidate tags from VLM matching and RAG to pass to LLM
            candidate_tags = [r.tag for r in unique_recommendations]
            
            # Also add RAG tags that might not have made it to top_k yet but are candidates
            for tag, _, _ in rag_tags:
                if tag not in candidate_tags:
                    candidate_tags.append(tag)
                    
            unique_recommendations = await self.refine_tags_with_llm(
                unique_recommendations, vlm_analysis, rag_matches, candidate_tags, top_k
            )

        # 9. If we STILL don't have enough, suggest from related tags (only if threshold is low)
        if len(unique_recommendations) < top_k:
            existing_tags = [r.tag for r in unique_recommendations]
            related = self.tag_library.suggest_related_tags(
                existing_tags, limit=top_k - len(unique_recommendations)
            )
            for tag in related:
                # Apply confidence threshold to suggested tags too
                if safe_confidence(0.5) < confidence_threshold:
                    continue

                if tag not in seen_tags:
                    unique_recommendations.append(
                        TagRecommendation(
                            tag=tag,
                            confidence=safe_confidence(0.5),
                            source="suggested",
                            reason="Related to detected tags",
                        )
                    )
                    if len(unique_recommendations) >= top_k:
                        break

        # 10. Final Sensitivity Check (Verification)
        # Check specific sensitive tags that are prone to false positives
        # Include both English and Chinese variants
        sensitive_tags = ["loli", "anal", "rape", "shota", "蘿莉", "肛交", "強姦", "正太"]
        final_verified_recommendations = []
        
        for rec in unique_recommendations:
            # Check if tag is sensitive and we have the means to verify it
            if rec.tag in sensitive_tags and vlm_service and image_bytes:
                logger.info(f"Triggering strict verification for sensitive tag: {rec.tag}")
                try:
                    is_verified = await vlm_service.verify_sensitive_tag(image_bytes, rec.tag)
                    if not is_verified:
                        logger.warning(f"Sensitive tag '{rec.tag}' FAILED verification. Removing.")
                        continue # Skip adding this tag
                    else:
                        logger.info(f"Sensitive tag '{rec.tag}' PASSED verification.")
                        # Boost confidence/reason slightly to show it was verified?
                        rec.reason += " | Verified by strict check"
                except Exception as e:
                    logger.error(f"Error during tag verification: {e}")
                    # If verification errors out, what should we do? 
                    # User wants to avoid FALSE POSITIVES. So maybe remove it if unsafe?
                    # Or keep it if confidence is high? 
                    # Let's remove it to be safe as per user request "not allowed to label wrong".
                    continue
            
            final_verified_recommendations.append(rec)

        unique_recommendations = final_verified_recommendations

        logger.info(f"Returning {len(unique_recommendations)} tag recommendations")
        return unique_recommendations[:top_k]

    def _is_vlm_analysis_valid(self, vlm_analysis: Dict[str, Any]) -> bool:
        """Check if VLM analysis contains valid data (not fallback/empty)."""
        if not vlm_analysis:
            return False

        # Check if any category has data
        has_data = (
            vlm_analysis.get("character_types", [])
            or vlm_analysis.get("clothing", [])
            or vlm_analysis.get("body_features", [])
            or vlm_analysis.get("actions", [])
            or vlm_analysis.get("themes", [])
            or vlm_analysis.get("raw_keywords", [])
        )

        # Check description doesn't indicate failure
        description = vlm_analysis.get("description", "")
        is_failed = any(
            keyword in description.lower()
            for keyword in [
                "failed",
                "error",
                "unable to",
                "analysis failed",
                "empty response",
            ]
        )

        return has_data and not is_failed

    def _extract_keywords_from_text(self, text: str) -> List[str]:
        """Extract keywords from text description when VLM fails."""
        keywords = []
        text_lower = text.lower()

        # Character type keywords
        char_keywords = {
            "loli": ["loli", "young girl", "little girl", "child character"],
            "shota": ["shota", "young boy", "little boy"],
            "teen": ["teen", "teenager", "adolescent"],
            "mature": ["mature", "adult woman", "female character"],
            "catgirl": ["catgirl", "cat girl", "cat ears", "nekomimi"],
            "doggirl": ["doggirl", "dog girl", "dog ears"],
            "foxgirl": ["foxgirl", "fox girl", "fox ears", "kitsune"],
            "elf": ["elf", "elven", "pointed ears"],
            "demon": ["demon", "succubus", "devil", "horns"],
            "angel": ["angel", "angelic", "wings", "halo"],
            "vampire": ["vampire", "fangs"],
            "monster_girl": ["monster girl", "monstergirl"],
        }

        # Clothing keywords
        clothing_keywords = {
            "school_uniform": ["school uniform", "seifuku", "uniform"],
            "swimsuit": ["swimsuit", "swimwear", "bathing suit"],
            "bikini": ["bikini"],
            "lingerie": ["lingerie", "underwear"],
            "kimono": ["kimono", "yukata", "traditional"],
            "maid_outfit": ["maid", "maid outfit", "maid dress"],
            "nurse": ["nurse", "nurse outfit"],
            "police_uniform": ["police", "officer"],
            "bunny_suit": ["bunny suit", "bunny girl"],
        }

        # Body feature keywords
        body_keywords = {
            "flat_chest": ["flat chest", "small breasts"],
            "small_breasts": ["small breasts", "modest"],
            "large_breasts": ["large breasts", "big breasts", "busty"],
            "glasses": ["glasses", "eyewear", "spectacles"],
            "stockings": ["stockings", "thigh highs"],
            "pantyhose": ["pantyhose", "tights"],
            "knee_high_socks": ["knee high socks", "long socks"],
            "tattoo": ["tattoo", "ink"],
        }

        # Check each keyword map
        all_maps = [
            (char_keywords, "character_types"),
            (clothing_keywords, "clothing"),
            (body_keywords, "body_features"),
        ]

        for keyword_map, category in all_maps:
            for tag, patterns in keyword_map.items():
                for pattern in patterns:
                    if pattern in text_lower:
                        keywords.append(tag)
                        break

        return keywords

    def _extract_vlm_keywords(self, vlm_analysis: Dict[str, Any]) -> List[str]:
        """Extract keywords from VLM analysis with filtering."""
        keywords = []

        # 1. Collect structured keywords first (these are high confidence)
        structured_fields = [
            "character_types",
            "clothing",
            "body_features",
            "actions",
            "themes",
            "settings",
        ]

        found_structured = False
        for field in structured_fields:
            vals = vlm_analysis.get(field, [])
            if vals:
                keywords.extend(vals)
                found_structured = True

        # 2. Extract from description ONLY if structured info is sparse
        # or as a secondary source with strict filtering
        description = vlm_analysis.get("description", "")
        if description:
            # Common procedural or filler words to exclude
            STOP_WORDS = {
                "indicates", "also", "translates", "translated", "scanlated", 
                "scanlation", "credits", "page", "numbers", "chapter", "raws",
                "from", "the", "and", "this", "with", "that", "shows", 
                "presents", "wearing", "appears", "seems", "likely", "could", 
                "be", "is", "are", "has", "have", "contains", "analysis", 
                "results", "based", "on", "image", "manga", "cover", "showing", 
                "holding", "setting", "quality", "style", "art", "modern", 
                "clean", "which", "there", "these", "those", "for", "relevant", 
                "detected", "suggested", "including", "tags", "presents", "here", 
                "some", "following", "choice", "visible", "seen", "character",
                "clothing", "body", "action", "theme", "features", "details",
                "extracted", "background", "foreground", "looks", "likely",
                "possibly", "identified", "elements", "version", "update",
                "translates", "to", "of", "and", "the", "this", "that", "for"
            }
            
            # Split description into words
            import re
            words = re.findall(r'\w+', description.lower())
            
            for word in words:
                # Filter by length and stop words
                if len(word) > 2 and word not in STOP_WORDS:
                    # If we already have structured info, be extremely selective with description keywords
                    if found_structured:
                        # Only add if it's a long, likely descriptive word not in stop words
                        # and ONLY if we're not planning to use LLM for synthesis
                        if len(word) > 5 and not settings.USE_LM_STUDIO:
                            keywords.append(word)
                    else:
                        # Even without structured info, don't just add common junk
                        # and stay cautious if LLM will handle the synthesis later
                        min_len = 6 if settings.USE_LM_STUDIO else 4
                        if len(word) > min_len:
                            keywords.append(word)

        return list(set(keywords))  # Remove duplicates

    def _extract_rag_tags(
        self, rag_matches: List[Dict[str, Any]], min_confidence: float
    ) -> List[Tuple[str, float, str]]:
        """Extract tags from RAG matches.

        Returns list of (tag_name, confidence, match_info) tuples.
        """
        results = []

        for match in rag_matches:
            score = match.get("score", 0.0)
            tags = match.get("tags", [])

            # Only use high-confidence matches
            if score >= min_confidence:
                for tag in tags:
                    # Validate tag exists in library
                    if tag in self.tag_library.tag_names:
                        results.append((tag, score, f"similarity: {score:.2f}"))
                    else:
                        # Try to find similar tag in library
                        matches = self.tag_library.match_tags_by_keywords(
                            [tag], min_confidence=0.7
                        )
                        for lib_tag, conf in matches[:1]:
                            results.append(
                                (lib_tag, score * conf, f"mapped from '{tag}'")
                            )

        return results

    async def refine_tags_with_llm(
        self,
        initial_tags: List[TagRecommendation],
        vlm_analysis: Dict[str, Any],
        rag_matches: List[Dict[str, Any]],
        candidate_tags: List[str],
        top_k: int = 5,
    ) -> List[TagRecommendation]:
        """
        Refine tags using LM Studio LLM for final synthesis.
        """
        try:
            from app.services.lm_studio_llm_service import LMStudioLLMService

            llm_service = LMStudioLLMService()

            # Fetch tag definitions for lexical synthesis
            tag_definitions = self.tag_library.get_tag_definitions(candidate_tags)

            # Convert Dict VLM analysis to VLMMetadata object for the service
            vlm_metadata = VLMMetadata(
                description=vlm_analysis.get("description", ""),
                characters=vlm_analysis.get("character_types", []),
                themes=vlm_analysis.get("themes", []),
                art_style=vlm_analysis.get("art_style"),
                genre_indicators=vlm_analysis.get("genre_indicators", []),
                tag_definitions=tag_definitions
            )

            # Use real RAG matches for synthesis and pass the collected candidates
            refined_results = await llm_service.synthesize_tags(
                vlm_metadata=vlm_metadata,
                rag_matches=rag_matches,
                candidate_tags=candidate_tags,
                top_k=top_k,
            )

            if refined_results:
                # Convert TagResult objects back to TagRecommendation
                return [
                    TagRecommendation(
                        tag=res.tag,
                        confidence=res.confidence,
                        source=res.source,
                        reason=res.reason,
                    )
                    for res in refined_results
                ]

        except Exception as e:
            logger.error(f"LLM refinement failed: {e}")

        return initial_tags


# Singleton instance
_recommender_service: Optional[TagRecommenderService] = None


def get_tag_recommender_service() -> TagRecommenderService:
    """Get or create tag recommender service singleton."""
    global _recommender_service
    if _recommender_service is None:
        _recommender_service = TagRecommenderService()
    return _recommender_service
