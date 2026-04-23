"""Tag Recommender Service for intelligent tag suggestions.

Combines VLM analysis, tag library matching, RAG results, and LLM synthesis
to recommend the best tags from the 611-tag library.

Architecture:
1. _extract_keywords: Extract keywords from VLM analysis
2. _match_with_library: Match keywords to tag library
3. _search_semantic: Perform semantic search for additional tags
4. _extract_rag_tags: Extract tags from RAG matches
5. _add_vlm_categorized_tags: Add character/clothing tags from VLM
6. _verify_and_calibrate: Verify sensitive tags and apply frequency calibration
7. _refine_with_llm: Optionally refine with LLM synthesis
"""

import logging
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from app.core.config import settings
from app.core.metrics import TAG_RECOMMENDATION_LATENCY, TAG_RECOMMENDATION_COUNT
from app.domain.models import VLMMetadata
from app.domain.tag.library import get_tag_library_service
from app.domain.tag.mapper import get_tag_mapper
from app.domain.tag.parser import (
    BODY_KEYWORDS,
    BODY_KEYWORDS_DICT,
    CHAR_KEYWORDS,
    CLOTHING_KEYWORDS_DICT,
    STOP_WORDS,
)
from app.utils import safe_confidence

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
        image_bytes: Optional[bytes] = None,
    ) -> List[TagRecommendation]:
        """Recommend tags based on VLM analysis and RAG matches.

        This is the main entry point that orchestrates the multi-stage pipeline:
        1. Extract and validate VLM keywords
        2. Match keywords to tag library
        3. Search semantic similarity
        4. Extract RAG tags
        5. Add VLM categorized tags
        6. Verify sensitive tags and calibrate
        7. Optionally refine with LLM
        """
        start_time = time.time()
        status = "success"

        try:
            recommendations: List[TagRecommendation] = []

            # Stage 1: VLM JSON tags are now authoritative; do NOT fall back to
            # description-keyword extraction (that path was the source of hedge
            # contamination like '需要更多視覺證據' -> '卡在牆上').
            vlm_json_tags = vlm_analysis.get("tags", []) if isinstance(vlm_analysis, dict) else []
            mapped_keywords: List[str] = []
            vlm_is_valid = True
            used_vlm_json_path = False

            if vlm_json_tags and isinstance(vlm_json_tags, list) and any(
                isinstance(t, dict) and t.get("tag") for t in vlm_json_tags
            ):
                used_vlm_json_path = True
                # New path: use the tag names from the JSON contract directly
                logger.info(f"VLM JSON path: {len(vlm_json_tags)} tags from contract")
                for t in vlm_json_tags:
                    if not isinstance(t, dict):
                        continue
                    name = t.get("tag", "").strip() if isinstance(t.get("tag"), str) else ""
                    if not name or name not in self.tag_library.tag_names:
                        continue
                    confidence = float(t.get("confidence", 0.6)) if isinstance(t.get("confidence"), (int, float)) else 0.6
                    if confidence < 0.6:
                        continue
                    recommendations.append(
                        TagRecommendation(
                            tag=name,
                            confidence=safe_confidence(confidence),
                            source="vlm_json",
                            reason=t.get("evidence", "VLM JSON tag") or "VLM JSON tag",
                        )
                    )
                    mapped_keywords.append(name)  # used by semantic fallback only
            else:
                # Backward compat: legacy free-form analysis (mock services / old VLM)
                logger.warning("No VLM JSON tags; falling back to legacy keyword extraction")
                vlm_keywords = self._extract_vlm_keywords(vlm_analysis)
                vlm_is_valid = self._is_vlm_analysis_valid(vlm_analysis)
                if not vlm_is_valid:
                    logger.warning("VLM invalid. Using RAG-only mode")
                mapped_keywords = self._map_keywords_to_chinese(vlm_keywords)
                recommendations = self._match_with_library(mapped_keywords, confidence_threshold)

            # Stage 4: Semantic search (if available and needed)
            recommendations = await self._search_semantic(mapped_keywords, recommendations, top_k)

            # Stage 5: Extract tags from RAG matches
            rag_tags = self._extract_rag_tags(rag_matches, confidence_threshold)
            if not vlm_is_valid and rag_matches:
                confidence_threshold = max(0.3, confidence_threshold - 0.2)

            recommendations = self._merge_rag_tags(recommendations, rag_tags, confidence_threshold)

            # Stage 6: Add VLM categorized tags
            recommendations = self._add_vlm_categorized_tags(
                recommendations, vlm_analysis, confidence_threshold
            )

            # Stage 7: Deduplicate and apply threshold
            recommendations = self._deduplicate_and_filter(
                recommendations, top_k, confidence_threshold
            )

            # Stage 8: LLM refinement — Phase 1: skipped when VLM JSON path
            # was authoritative. Spec §3.4 + §3.7 treat VLM JSON as the truth;
            # re-synthesizing through an LLM was (a) the root of hedge-style
            # contamination re-entering results and (b) a 400-error vector
            # for qwen3.6 on the legacy synthesis prompt.
            if (
                not used_vlm_json_path
                and settings.USE_LM_STUDIO
                and not settings.USE_MOCK_SERVICES
            ):
                recommendations = await self._refine_with_llm(
                    recommendations, vlm_analysis, rag_matches, top_k
                )

            # Stage 9: Add related tags if needed
            recommendations = self._add_related_tags(recommendations, top_k, confidence_threshold)

            # Stage 10: Verify sensitive tags and apply calibration
            recommendations = await self._verify_and_calibrate(
                recommendations, vlm_service, image_bytes, rag_matches, vlm_analysis
            )

            logger.info(f"Returning {len(recommendations)} recommendations")
            return recommendations[:top_k]

        except Exception as e:
            logger.error(f"Tag recommendation failed: {e}")
            status = "error"
            return []
        finally:
            # Record metrics
            duration = time.time() - start_time
            TAG_RECOMMENDATION_COUNT.labels(status=status).inc()
            TAG_RECOMMENDATION_LATENCY.observe(duration)

    # =========================================================================
    # Private helper methods
    # =========================================================================

    def _is_vlm_analysis_valid(self, vlm_analysis: Dict[str, Any]) -> bool:
        """Check if VLM analysis contains valid data."""
        if not vlm_analysis:
            return False
        has_data = any(
            vlm_analysis.get(field)
            for field in [
                "character_types",
                "clothing",
                "body_features",
                "actions",
                "themes",
                "raw_keywords",
            ]
        )
        description = vlm_analysis.get("description", "")
        is_failed = any(
            kw in description.lower() for kw in ["failed", "error", "unable to", "empty response"]
        )
        return has_data and not is_failed

    def _map_keywords_to_chinese(self, keywords: List[str]) -> List[str]:
        """Map English keywords to Chinese tags."""
        mapped = []
        for kw in keywords:
            cn_tag = self.tag_mapper.to_chinese(kw)
            if cn_tag:
                mapped.append(cn_tag)
                logger.debug(f"Mapped '{kw}' -> '{cn_tag}'")
            else:
                mapped.append(kw)
        return mapped

    def _match_with_library(self, keywords: List[str], threshold: float) -> List[TagRecommendation]:
        """Match keywords against the tag library."""
        recommendations = []
        matches = self.tag_library.match_tags_by_keywords(keywords, min_confidence=0.5)

        for tag_name, confidence in matches:
            if confidence >= threshold:
                boost = (
                    settings.EXACT_MATCH_BOOST
                    if confidence >= 0.95
                    else settings.PARTIAL_MATCH_BOOST
                )
                recommendations.append(
                    TagRecommendation(
                        tag=tag_name,
                        confidence=safe_confidence(confidence * boost),
                        source="library_match",
                        reason=f"Matched from keywords: {', '.join(keywords[:3])}",
                    )
                )
        return recommendations

    async def _search_semantic(
        self,
        keywords: List[str],
        current_recs: List[TagRecommendation],
        top_k: int,
    ) -> List[TagRecommendation]:
        """Perform semantic search ONLY as a fallback when VLM under-delivered.

        Triggers when len(current_recs) < SEMANTIC_FALLBACK_TRIGGER_COUNT.
        Cap additions at SEMANTIC_FALLBACK_MAX_ADDITIONS.
        Filter results by CHINESE_EMBEDDING_THRESHOLD (0.75).
        """
        if settings.USE_MOCK_SERVICES:
            return current_recs
        if len(current_recs) >= settings.SEMANTIC_FALLBACK_TRIGGER_COUNT:
            return current_recs
        if len(current_recs) >= top_k:
            return current_recs

        try:
            from app.services.chinese_embedding_service import get_chinese_embedding_service

            embedding_service = get_chinese_embedding_service()
        except Exception:
            return current_recs

        if not embedding_service or not embedding_service.is_available():
            return current_recs

        logger.info("Performing semantic fallback (VLM under-delivered)")
        # ENCAPSULATION FIX: Use hasattr instead of direct private attribute access
        if (
            not hasattr(embedding_service, "_tag_matrix_cache")
            or embedding_service._tag_matrix_cache is None
        ):
            all_tags = self.tag_library.get_all_tags()
            await embedding_service.cache_tag_embeddings(all_tags)

        added_count = 0
        for keyword in keywords:
            if added_count >= settings.SEMANTIC_FALLBACK_MAX_ADDITIONS:
                break
            semantic_matches = await embedding_service.search_cached_tags(
                keyword, top_k=2, threshold=settings.CHINESE_EMBEDDING_THRESHOLD
            )
            for s_match in semantic_matches:
                if added_count >= settings.SEMANTIC_FALLBACK_MAX_ADDITIONS:
                    break
                tag_name = s_match["tag"]
                if any(r.tag == tag_name for r in current_recs):
                    continue
                current_recs.append(
                    TagRecommendation(
                        tag=tag_name,
                        confidence=safe_confidence(
                            s_match["similarity"] * settings.SEMANTIC_MATCH_PENALTY
                        ),
                        source="semantic_fallback",
                        reason=f"Semantic fallback for '{keyword}' (sim={s_match['similarity']:.2f})",
                    )
                )
                added_count += 1
        return current_recs

    def _extract_rag_tags(
        self, rag_matches: List[Dict[str, Any]], min_confidence: float
    ) -> List[Tuple[str, float, str]]:
        """Extract tags from RAG matches."""
        results = []
        for match in rag_matches:
            score = match.get("score", 0.0)
            if score < min_confidence:
                continue
            for tag in match.get("tags", []):
                if tag in self.tag_library.tag_names:
                    results.append((tag, score, f"similarity: {score:.2f}"))
                else:
                    mapped = self.tag_library.match_tags_by_keywords([tag], min_confidence=0.7)
                    for lib_tag, conf in mapped[:1]:
                        results.append((lib_tag, score * conf, f"mapped from '{tag}'"))
        return results

    def _merge_rag_tags(
        self,
        recommendations: List[TagRecommendation],
        rag_tags: List[Tuple[str, float, str]],
        threshold: float,
    ) -> List[TagRecommendation]:
        """Merge RAG tags into recommendations."""
        for tag_name, confidence, match_info in rag_tags:
            existing = next((r for r in recommendations if r.tag == tag_name), None)
            if existing:
                existing.confidence = safe_confidence(min(existing.confidence + 0.1, 1.0))
                existing.source = "vlm+rag"
                existing.reason += f" | RAG: {match_info}"
            else:
                recommendations.append(
                    TagRecommendation(
                        tag=tag_name,
                        confidence=safe_confidence(confidence),
                        source="rag",
                        reason=f"RAG match: {match_info}",
                    )
                )
        return recommendations

    def _add_vlm_categorized_tags(
        self,
        recommendations: List[TagRecommendation],
        vlm_analysis: Dict[str, Any],
        threshold: float,
    ) -> List[TagRecommendation]:
        """Add character type and clothing tags from VLM analysis."""
        existing_tags = {r.tag for r in recommendations}

        # Character types
        for char_type in vlm_analysis.get("character_types", []):
            if char_type not in existing_tags:
                cn_type = self.tag_mapper.to_chinese(char_type) or char_type
                matches = self.tag_library.match_tags_by_keywords([cn_type], min_confidence=0.5)
                for tag_name, conf in matches[:1]:
                    if conf >= threshold:
                        recommendations.append(
                            TagRecommendation(
                                tag=tag_name,
                                confidence=safe_confidence(conf * 0.9),
                                source="vlm",
                                reason=f"Character type: {char_type} -> {tag_name}",
                            )
                        )

        # Clothing
        for item in vlm_analysis.get("clothing", []):
            if item not in existing_tags:
                cn_item = self.tag_mapper.to_chinese(item) or item
                matches = self.tag_library.match_tags_by_keywords([cn_item], min_confidence=0.5)
                for tag_name, conf in matches[:1]:
                    if conf >= threshold:
                        recommendations.append(
                            TagRecommendation(
                                tag=tag_name,
                                confidence=safe_confidence(conf * 0.9),
                                source="vlm",
                                reason=f"Clothing: {item} -> {tag_name}",
                            )
                        )

        return recommendations

    def _deduplicate_and_filter(
        self,
        recommendations: List[TagRecommendation],
        top_k: int,
        threshold: float,
    ) -> List[TagRecommendation]:
        """Remove duplicates and filter by threshold."""
        recommendations.sort(key=lambda x: x.confidence, reverse=True)
        seen = set()
        unique = []
        for rec in recommendations:
            if rec.confidence < threshold:
                continue
            if rec.tag not in seen:
                seen.add(rec.tag)
                unique.append(rec)
                if len(unique) >= top_k:
                    break
        return unique

    async def _refine_with_llm(
        self,
        current_recs: List[TagRecommendation],
        vlm_analysis: Dict[str, Any],
        rag_matches: List[Dict[str, Any]],
        top_k: int,
    ) -> List[TagRecommendation]:
        """Refine tags using LLM synthesis."""
        logger.info("Refining with LLM")
        try:
            from app.services.lm_studio_llm_service import LMStudioLLMService

            llm_service = LMStudioLLMService()

            candidate_tags = [r.tag for r in current_recs]
            for tag, _, _ in self._extract_rag_tags(rag_matches, 0.3):
                if tag not in candidate_tags:
                    candidate_tags.append(tag)

            tag_defs = self.tag_library.get_tag_definitions(candidate_tags)
            vlm_metadata = VLMMetadata(
                description=vlm_analysis.get("description", ""),
                characters=vlm_analysis.get("character_types", []),
                themes=vlm_analysis.get("themes", []),
                art_style=vlm_analysis.get("art_style"),
                genre_indicators=vlm_analysis.get("genre_indicators", []),
                tag_definitions=tag_defs,
            )

            refined = await llm_service.synthesize_tags(
                vlm_metadata=vlm_metadata,
                rag_matches=rag_matches,
                candidate_tags=candidate_tags,
                top_k=top_k,
            )

            if refined:
                return [
                    TagRecommendation(
                        tag=r.tag,
                        confidence=r.confidence,
                        source=r.source,
                        reason=r.reason or "LLM synthesis",
                    )
                    for r in refined
                ]
        except Exception as e:
            logger.error(f"LLM refinement failed: {e}")
        return current_recs

    def _add_related_tags(
        self,
        recommendations: List[TagRecommendation],
        top_k: int,
        threshold: float,
    ) -> List[TagRecommendation]:
        """Add related tags if we don't have enough results."""
        if len(recommendations) >= top_k:
            return recommendations

        existing = [r.tag for r in recommendations]
        related = self.tag_library.suggest_related_tags(existing, top_k - len(recommendations))

        for tag in related:
            if safe_confidence(0.5) < threshold:
                continue
            recommendations.append(
                TagRecommendation(
                    tag=tag,
                    confidence=safe_confidence(0.5),
                    source="suggested",
                    reason="Related to detected tags",
                )
            )
            if len(recommendations) >= top_k:
                break
        return recommendations

    async def _verify_and_calibrate(
        self,
        recommendations: List[TagRecommendation],
        vlm_service: Any,
        image_bytes: Optional[bytes],
        rag_matches: List[Dict[str, Any]],
        vlm_analysis: Optional[Dict[str, Any]] = None,
    ) -> List[TagRecommendation]:
        """Verify sensitive tags and apply frequency calibration."""
        vlm_analysis = vlm_analysis or {}
        verified = []

        for rec in recommendations:
            is_sensitive = rec.tag in settings.SENSITIVE_TAGS

            if is_sensitive and vlm_service and image_bytes:
                is_verified = await vlm_service.verify_sensitive_tag(image_bytes, rec.tag)
                if not is_verified:
                    logger.warning(f"Sensitive tag '{rec.tag}' failed verification, removing")
                    continue
                rec.reason += " | Verified"

            elif is_sensitive and not (vlm_service and image_bytes):
                rec.confidence = safe_confidence(rec.confidence * 0.7)
                rec.reason += " | Unverified (penalized)"

            verified.append(rec)

        # Apply frequency calibration
        calibrated = []
        for rec in verified:
            if rec.tag in settings.TAG_FREQUENCY_CALIBRATION:
                calib = settings.TAG_FREQUENCY_CALIBRATION[rec.tag]
                rec.confidence = safe_confidence(rec.confidence * calib)
                direction = "penalty" if calib < 1.0 else "boost"
                rec.reason += f" (freq {direction}: {calib:.2f})"

            # Apply exact match penalty for common false positives
            if rec.tag in settings.EXACT_MATCH_PENALTY:
                penalty = settings.EXACT_MATCH_PENALTY[rec.tag]
                rec.confidence = safe_confidence(rec.confidence * penalty)
                rec.reason += f" (exact penalty: {penalty:.2f})"

            # Apply visual feature boost if we have visual support
            if rec.tag in settings.VISUAL_FEATURE_BOOST and rec.source in ("vlm", "library_match"):
                boost = settings.VISUAL_FEATURE_BOOST[rec.tag]
                rec.confidence = safe_confidence(rec.confidence * boost)
                rec.reason += f" (visual boost: {boost:.2f})"

            # Skip very low confidence after calibration
            if rec.confidence < settings.MIN_ACCEPTABLE_CONFIDENCE:
                logger.debug(f"Filtered low confidence: {rec.tag} ({rec.confidence:.2f})")
                continue

            calibrated.append(rec)

        # Apply RAG support adjustments
        rag_tags = {tag for m in rag_matches for tag in m.get("tags", [])}
        for rec in calibrated:
            if "rag" in rec.source.lower():
                rec.confidence = min(1.0, rec.confidence * settings.RAG_SUPPORT_BOOST)
                rec.reason += " (+RAG)"
            elif rec.source in ("vlm", "suggested"):
                rec.confidence = safe_confidence(rec.confidence * settings.RAG_SUPPORT_DECAY)
                rec.reason += " (no-RAG)"

            # Semantic siblings boost - improved logic
            if rec.source == "vlm" and rec.tag not in rag_tags:
                siblings = settings.SEMANTIC_SIBLINGS.get(rec.tag, set())
                sibling_intersection = siblings.intersection(rag_tags)
                if sibling_intersection:
                    rec.confidence = min(1.0, rec.confidence * 1.05)  # Increased from 1.03 to 1.05
                    rec.reason += f" (+semantic: {', '.join(sibling_intersection)})"

        calibrated.sort(key=lambda x: x.confidence, reverse=True)

        # Apply mutual exclusivity resolution
        calibrated = self._apply_mutual_exclusivity(calibrated)

        # Apply hierarchical boosting (specific > generic)
        calibrated = self._apply_hierarchical_boost(calibrated)

        # Apply cross-validation for age-related tags
        calibrated = self._validate_age_related_tags(calibrated, vlm_analysis)

        return calibrated

    def _apply_mutual_exclusivity(
        self, recommendations: List[TagRecommendation]
    ) -> List[TagRecommendation]:
        """Apply mutual exclusivity rules to filter conflicting tags."""
        if not recommendations:
            return recommendations

        # Sort by confidence descending
        sorted_recs = sorted(recommendations, key=lambda x: x.confidence, reverse=True)

        kept = []
        blocked = set()

        for rec in sorted_recs:
            if rec.tag in blocked:
                logger.debug(f"Filtered mutual exclusive: {rec.tag}")
                continue

            kept.append(rec)

            # Block conflicting tags
            conflicts = settings.MUTUAL_EXCLUSIVITY.get(rec.tag, set())
            for conflict in conflicts:
                blocked.add(conflict)

        return kept

    def _apply_hierarchical_boost(
        self, recommendations: List[TagRecommendation]
    ) -> List[TagRecommendation]:
        """Apply confidence boost for specific tags over generic ones."""
        if not recommendations:
            return recommendations

        tag_dict = {r.tag: r for r in recommendations}
        boosted = []
        processed_generics = set()

        for rec in recommendations:
            parent = settings.TAG_HIERARCHY.get(rec.tag)

            if parent and parent in tag_dict:
                # Boost specific tag, the generic will be handled separately
                specific_boost = 1.08
                rec.confidence = safe_confidence(rec.confidence * specific_boost)
                rec.reason += " (hierarchical specific)"
                boosted.append(rec)

                # Penalize generic but mark as processed
                if parent not in processed_generics:
                    generic_rec = tag_dict[parent]
                    generic_penalty = 0.80
                    generic_rec.confidence = safe_confidence(
                        generic_rec.confidence * generic_penalty
                    )
                    generic_rec.reason += " (hierarchical generic)"
                    boosted.append(generic_rec)
                    processed_generics.add(parent)
            elif rec.tag not in processed_generics:
                boosted.append(rec)

        # Re-sort by confidence
        boosted.sort(key=lambda x: x.confidence, reverse=True)
        return boosted

    def _validate_age_related_tags(
        self,
        recommendations: List[TagRecommendation],
        vlm_analysis: Dict[str, Any],
    ) -> List[TagRecommendation]:
        """Cross-validate age-related tags for consistency.

        Age-related tags (蘿莉, 少女, 人妻, etc.) should be mutually exclusive
        or at least have their confidence adjusted when conflicting.
        """
        # Define age group hierarchy: more specific -> less specific
        age_groups = {
            "蘿莉": 1,  # Youngest
            "少女": 2,
            "少年": 2,
            "正太": 1,
            "人妻": 4,
            "御姐": 3,
            "熟女": 4,
        }

        # Find age-related tags in recommendations
        age_tags = [(r, age_groups.get(r.tag, 99)) for r in recommendations if r.tag in age_groups]

        if len(age_tags) <= 1:
            return recommendations  # No conflict

        # Sort by specificity (lower number = more specific)
        age_tags.sort(key=lambda x: x[1])

        # Keep the most specific age tag, penalize others
        most_specific = age_tags[0][0]
        result = []

        for rec, _ in age_tags:
            if rec.tag == most_specific.tag:
                result.append(rec)  # Keep at full confidence
            else:
                # Penalize conflicting age tag
                rec.confidence = safe_confidence(rec.confidence * 0.6)
                rec.reason += " (age conflict penalty)"
                result.append(rec)

        # Add non-age tags unchanged
        non_age = [r for r in recommendations if r.tag not in age_groups]
        result.extend(non_age)

        # Re-sort
        result.sort(key=lambda x: x.confidence, reverse=True)
        return result

    def _extract_keywords_from_text(self, text: str) -> List[str]:
        """Extract keywords from text when VLM fails."""
        keywords = []
        text_lower = text.lower()

        all_maps = [
            (CHAR_KEYWORDS, "character_types"),
            (CLOTHING_KEYWORDS_DICT, "clothing"),
            (BODY_KEYWORDS_DICT, "body_features"),
        ]

        for keyword_map, _ in all_maps:
            for tag, patterns in keyword_map.items():
                if any(p in text_lower for p in patterns):
                    keywords.append(tag)
        return keywords

    def _extract_vlm_keywords(self, vlm_analysis: Dict[str, Any]) -> List[str]:
        """Extract keywords from VLM analysis."""
        keywords = []
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

        description = vlm_analysis.get("description", "")
        if description:
            words = re.findall(r"\w+", description.lower())
            for word in words:
                if len(word) > 2 and word not in STOP_WORDS:
                    if found_structured:
                        if len(word) > 5 and not settings.USE_LM_STUDIO:
                            keywords.append(word)
                    else:
                        min_len = 6 if settings.USE_LM_STUDIO else 4
                        if len(word) > min_len:
                            keywords.append(word)

        return list(set(keywords))


# Singleton instance
_recommender_service: Optional[TagRecommenderService] = None


def get_tag_recommender_service() -> TagRecommenderService:
    """Get or create tag recommender service singleton."""
    global _recommender_service
    if _recommender_service is None:
        _recommender_service = TagRecommenderService()
    return _recommender_service
