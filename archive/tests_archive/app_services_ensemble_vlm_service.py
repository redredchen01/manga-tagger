"""Ensemble VLM Service.

Unified service for multi-model VLM tag generation with voting.
Combines dispatcher, aggregator, and post-processing.
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import List, Dict, Optional, Set, Any
from pathlib import Path

from app.config import settings
from app.services.enhanced_vlm_dispatcher import (
    EnhancedVLMDispatcher,
    DispatchResult,
)
from app.services.ensemble_vote_aggregator import (
    EnsembleVoteAggregator,
    EnsembleResult,
)
from app.services.tag_library_service import TagLibraryService, get_tag_library_service
from app.services.tag_conflict_resolver import TagConflictResolver, get_conflict_resolver
from app.services.dynamic_threshold_service import DynamicThresholdService, get_dynamic_threshold_service
from app.services.confidence_calibrator import ConfidenceCalibrator, get_confidence_calibrator

logger = logging.getLogger(__name__)


@dataclass
class TagResult:
    """Final tag result with all metadata."""
    tag: str
    confidence: float
    source: str  # "ensemble", "single", "fallback"
    model_votes: int
    model_sources: List[str]
    is_consensus: bool
    warnings: List[str] = None


@dataclass
class EnsembleTagResult:
    """Complete result from ensemble tagging."""
    tags: List[TagResult]
    total_tags: int
    processing_time: float
    model_used: List[str]
    overall_confidence: float
    warnings: List[str]
    errors: List[str]


class EnsembleVLMService:
    """Unified service for ensemble VLM tag generation.
    
    Pipeline:
    1. Dispatch image to multiple VLMs
    2. Aggregate votes from all models
    3. Resolve conflicts
    4. Apply dynamic thresholds
    5. Calibrate confidence scores
    6. Validate against tag library
    """
    
    def __init__(
        self,
        enable_ensemble: bool = True,
        enable_conflict_resolution: bool = True,
        enable_threshold_filtering: bool = True,
    ):
        """Initialize service.
        
        Args:
            enable_ensemble: Use ensemble voting
            enable_conflict_resolution: Resolve tag conflicts
            enable_threshold_filtering: Apply dynamic thresholds
        """
        self.enable_ensemble = enable_ensemble
        self.enable_conflict_resolution = enable_conflict_resolution
        self.enable_threshold_filtering = enable_threshold_filtering
        
        # Initialize components
        self.dispatcher = EnhancedVLMDispatcher()
        self.aggregator = EnsembleVoteAggregator(
            vote_threshold=settings.ENSEMBLE_VOTE_THRESHOLD,
        )
        self.calibrator = get_confidence_calibrator()
        
        # Load tag library for validation
        try:
            self.tag_library = get_tag_library_service()
            self.known_tags: Set[str] = set(self.tag_library.get_all_tags())
        except Exception as e:
            logger.warning(f"Failed to load tag library: {e}")
            self.known_tags = set()
        
        # Initialize conflict resolver
        if enable_conflict_resolution:
            self.conflict_resolver = get_conflict_resolver()
        
        # Initialize threshold service
        if enable_threshold_filtering:
            self.threshold_service = get_dynamic_threshold_service()
        
        logger.info(
            f"EnsembleVLMService initialized: "
            f"ensemble={enable_ensemble}, "
            f"conflict_resolution={enable_conflict_resolution}, "
            f"threshold_filtering={enable_threshold_filtering}"
        )
    
    async def analyze_image(
        self,
        image_bytes: bytes,
        custom_prompt: Optional[str] = None,
        max_tags: int = 20,
    ) -> EnsembleTagResult:
        """Analyze an image and generate tags using ensemble voting.
        
        Args:
            image_bytes: Image data
            custom_prompt: Optional custom prompt
            max_tags: Maximum number of tags to return
            
        Returns:
            EnsembleTagResult with tags and metadata
        """
        warnings = []
        errors = []
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Step 1: Dispatch to multiple models
            dispatch_result = await self.dispatcher.dispatch_all(
                image_bytes,
                custom_prompt=custom_prompt,
            )
            
            if dispatch_result.failed_models == len(dispatch_result.predictions):
                errors.append("All models failed")
                return self._create_error_result(errors, start_time)
            
            if dispatch_result.successful_models == 0:
                errors.append("No successful predictions")
                return self._create_error_result(errors, start_time)
            
            # Step 2: Aggregate votes
            aggregate_result = self.aggregator.aggregate(
                dispatch_result,
                known_tags=self.known_tags,
            )
            
            if not aggregate_result.final_tags:
                warnings.append("No tags passed voting threshold")
            
            # Step 3: Resolve conflicts
            if self.enable_conflict_resolution:
                resolved_tags, resolved_scores = self.conflict_resolver.resolve(
                    aggregate_result.final_tags,
                    aggregate_result.tag_scores,
                    max_tags=max_tags,
                )
                aggregate_result.final_tags = resolved_tags
                aggregate_result.tag_scores = resolved_scores
            
            # Step 4: Apply dynamic thresholds
            if self.enable_threshold_filtering and aggregate_result.tag_scores:
                filtered_tags, filtered_scores = self.threshold_service.filter_by_threshold(
                    aggregate_result.final_tags,
                    aggregate_result.tag_scores,
                    base_threshold=settings.TAG_MATCH_THRESHOLD,
                )
                aggregate_result.final_tags = filtered_tags
                aggregate_result.tag_scores = filtered_scores
            
            # Step 5: Calibrate confidence scores
            if aggregate_result.tag_scores:
                calibrated = self.calibrator.calibrate_batch(
                    aggregate_result.tag_scores,
                )
                aggregate_result.tag_scores = {
                    tag: cs.calibrated_score
                    for tag, cs in calibrated.items()
                }
            
            # Step 6: Build final result
            final_tags = aggregate_result.final_tags[:max_tags]
            
            tag_results = []
            for tag in final_tags:
                vote_info = aggregate_result.tag_votes.get(tag)
                
                tag_results.append(TagResult(
                    tag=tag,
                    confidence=aggregate_result.tag_scores.get(tag, 0.5),
                    source="ensemble",
                    model_votes=vote_info.vote_count if vote_info else 1,
                    model_sources=vote_info.source_models if vote_info else [],
                    is_consensus=(
                        vote_info.model_agreement == 1.0
                        if vote_info else False
                    ),
                ))
            
            # Add warnings from aggregation
            warnings.extend(aggregate_result.rejected_tags)
            
            return EnsembleTagResult(
                tags=tag_results,
                total_tags=len(tag_results),
                processing_time=asyncio.get_event_loop().time() - start_time,
                model_used=[p.model_name for p in dispatch_result.predictions],
                overall_confidence=aggregate_result.overall_confidence,
                warnings=warnings,
                errors=errors,
            )
            
        except Exception as e:
            logger.exception(f"Ensemble analysis failed: {e}")
            errors.append(str(e))
            return self._create_error_result(errors, start_time)
    
    async def analyze_with_fallback(
        self,
        image_bytes: bytes,
        fallback_model: Optional[str] = None,
        max_tags: int = 20,
    ) -> EnsembleTagResult:
        """Analyze with fallback to single model if ensemble fails.
        
        Args:
            image_bytes: Image data
            fallback_model: Model to use if ensemble fails
            max_tags: Maximum number of tags
            
        Returns:
            EnsembleTagResult
        """
        # Try ensemble first
        if self.enable_ensemble:
            result = await self.analyze_image(image_bytes, max_tags=max_tags)
            
            if result.tags or not result.errors:
                return result
            
            logger.warning("Ensemble failed, falling back to single model")
        
        # Fallback to single model
        fallback_model = fallback_model or settings.LM_STUDIO_VISION_MODEL
        
        try:
            # Use dispatcher for single model
            dispatch_result = await self.dispatcher.dispatch_all(
                image_bytes,
            )
            
            # Get successful prediction
            successful_preds = [
                p for p in dispatch_result.predictions
                if p.is_valid
            ]
            
            if not successful_preds:
                return EnsembleTagResult(
                    tags=[],
                    total_tags=0,
                    processing_time=0,
                    model_used=[fallback_model],
                    overall_confidence=0.0,
                    warnings=[],
                    errors=["All models failed"],
                )
            
            pred = successful_preds[0]
            
            # Build result
            tag_results = [
                TagResult(
                    tag=tag,
                    confidence=pred.confidence_scores.get(tag, 0.5),
                    source="single",
                    model_votes=1,
                    model_sources=[pred.model_name],
                    is_consensus=True,
                )
                for tag in pred.parsed_response.tags[:max_tags]
            ]
            
            return EnsembleTagResult(
                tags=tag_results,
                total_tags=len(tag_results),
                processing_time=pred.processing_time,
                model_used=[pred.model_name],
                overall_confidence=sum(
                    t.confidence for t in tag_results
                ) / len(tag_results) if tag_results else 0.0,
                warnings=["Used fallback single model"],
                errors=[],
            )
            
        except Exception as e:
            logger.exception(f"Fallback analysis failed: {e}")
            return EnsembleTagResult(
                tags=[],
                total_tags=0,
                processing_time=0,
                model_used=[fallback_model],
                overall_confidence=0.0,
                warnings=[],
                errors=[str(e)],
            )
    
    def _create_error_result(
        self,
        errors: List[str],
        start_time: float,
    ) -> EnsembleTagResult:
        """Create an error result."""
        return EnsembleTagResult(
            tags=[],
            total_tags=0,
            processing_time=asyncio.get_event_loop().time() - start_time,
            model_used=[],
            overall_confidence=0.0,
            warnings=[],
            errors=errors,
        )
    
    def get_available_models(self) -> Dict[str, Dict[str, Any]]:
        """Get list of available models and their status."""
        return self.dispatcher.get_model_status()


# Singleton instance
_ensemble_service: Optional[EnsembleVLMService] = None


def get_ensemble_service() -> EnsembleVLMService:
    """Get or create ensemble service singleton."""
    global _ensemble_service
    if _ensemble_service is None:
        _ensemble_service = EnsembleVLMService()
    return _ensemble_service
