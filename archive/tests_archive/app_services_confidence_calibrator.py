"""Confidence Calibration Service.

Provides calibrated confidence scores for tag matching.
"""

import logging
import math
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CalibratedScore:
    """Calibrated confidence score with metadata."""
    raw_score: float
    calibrated_score: float
    lower_bound: float
    upper_bound: float
    calibration_factor: str


class ConfidenceCalibrator:
    """Service for calibrating confidence scores."""
    
    # Calibration factors for different matching methods
    CALIBRATION_METHODS = {
        "exact_match": 1.0,      # No calibration needed
        "alias_match": 0.95,      # Slight reduction
        "contains_match": 0.90,   # Moderate reduction
        "partial_match": 0.80,   # More reduction
        "vector_similarity": 0.85, # Vector-based matches need calibration
        "rag_search": 0.75,      # RAG matches need more calibration
        "hybrid_combined": 0.90,  # Combined scores need slight calibration
    }
    
    # Temperature scaling parameters
    TEMPERATURE = 0.1
    
    def __init__(self, temperature: float = 0.1):
        """Initialize calibrator.
        
        Args:
            temperature: Temperature for softmax-style scaling
        """
        self.temperature = temperature
        logger.info(f"ConfidenceCalibrator initialized with temperature={temperature}")
    
    def calibrate(
        self,
        raw_score: float,
        method: str,
        top_k: int = 1,
        num_total_tags: int = 611
    ) -> CalibratedScore:
        """Calibrate a raw confidence score.
        
        Args:
            raw_score: Raw confidence score
            method: Matching method used
            top_k: Position in top-k results
            num_total_tags: Total number of tags in library
            
        Returns:
            CalibratedScore with bounds
        """
        # Get base calibration factor
        base_factor = self.CALIBRATION_METHODS.get(method, 0.85)
        
        # Apply position-based calibration (lower ranks are less certain)
        position_factor = 1.0 - (top_k - 1) * 0.02
        position_factor = max(position_factor, 0.7)
        
        # Apply sparsity factor (rare tags need more calibration)
        sparsity_factor = 1.0
        if num_total_tags > 500:
            # Assume more tags = more sparse matches
            sparsity_factor = 0.95
        
        # Calculate calibrated score
        calibrated = raw_score * base_factor * position_factor * sparsity_factor
        
        # Apply temperature scaling to compress extreme scores
        if calibrated > 0.9:
            calibrated = 0.9 + (calibrated - 0.9) * 0.5
        elif calibrated < 0.1:
            calibrated = calibrated * 0.5
        
        # Calculate confidence bounds
        uncertainty = 0.05 + (top_k - 1) * 0.02
        lower = max(0.0, calibrated - uncertainty)
        upper = min(1.0, calibrated + uncertainty)
        
        return CalibratedScore(
            raw_score=raw_score,
            calibrated_score=round(calibrated, 4),
            lower_bound=round(lower, 4),
            upper_bound=round(upper, 4),
            calibration_factor=method
        )
    
    def calibrate_batch(
        self,
        scores: Dict[str, float],
        methods: Optional[Dict[str, str]] = None,
        top_k_start: int = 1
    ) -> Dict[str, CalibratedScore]:
        """Calibrate a batch of scores.
        
        Args:
            scores: Dict of tag -> raw score
            methods: Optional dict of tag -> matching method
            top_k_start: Starting rank for position calibration
            
        Returns:
            Dict of tag -> CalibratedScore
        """
        methods = methods or {}
        calibrated = {}
        
        # Sort by score for position tracking
        sorted_tags = sorted(scores.keys(), key=lambda t: scores[t], reverse=True)
        
        for rank, tag in enumerate(sorted_tags, start=top_k_start):
            raw = scores[tag]
            method = methods.get(tag, "unknown")
            num_tags = len(scores)
            
            calibrated[tag] = self.calibrate(
                raw_score=raw,
                method=method,
                top_k=rank,
                num_total_tags=num_tags
            )
        
        return calibrated
    
    def normalize_scores(self, scores: Dict[str, float]) -> Dict[str, float]:
        """Normalize scores to sum to 1.0 (softmax-like).
        
        Args:
            scores: Dict of tag -> raw score
            
        Returns:
            Normalized scores
        """
        if not scores:
            return {}
        
        # Apply temperature scaling
        scaled = {tag: score / self.temperature for tag, score in scores.items()}
        
        # Compute softmax
        max_scaled = max(scaled.values())
        exp_scores = {tag: math.exp(score - max_scaled) for tag, score in scaled.items()}
        sum_exp = sum(exp_scores.values())
        
        # Normalize
        normalized = {tag: exp_score / sum_exp for tag, exp_score in exp_scores.items()}
        
        return normalized
    
    def get_top_calibrated(
        self,
        scores: Dict[str, float],
        methods: Optional[Dict[str, str]] = None,
        top_k: int = 10,
        min_threshold: float = 0.0
    ) -> List[Tuple[str, float, str]]:
        """Get top-k calibrated scores.
        
        Args:
            scores: Dict of tag -> raw score
            methods: Optional dict of tag -> matching method
            top_k: Number of top tags to return
            min_threshold: Minimum calibrated score threshold
            
        Returns:
            List of (tag, calibrated_score, method) tuples
        """
        calibrated = self.calibrate_batch(scores, methods)
        
        # Filter and sort
        filtered = [
            (tag, cs.calibrated_score, cs.calibration_factor)
            for tag, cs in calibrated.items()
            if cs.calibrated_score >= min_threshold
        ]
        
        # Sort by calibrated score
        filtered.sort(key=lambda x: x[1], reverse=True)
        
        return filtered[:top_k]
    
    def merge_scores(
        self,
        score_sources: List[Dict[str, float]],
        weights: Optional[List[float]] = None
    ) -> Dict[str, float]:
        """Merge scores from multiple sources with weights.
        
        Args:
            score_sources: List of score dicts
            weights: Optional weights for each source (must sum to 1.0)
            
        Returns:
            Merged scores
        """
        if not score_sources:
            return {}
        
        weights = weights or [1.0 / len(score_sources)] * len(score_sources)
        
        if len(weights) != len(score_sources):
            logger.warning("Weights length doesn't match sources, using equal weights")
            weights = [1.0 / len(score_sources)] * len(score_sources)
        
        # Normalize weights
        weight_sum = sum(weights)
        weights = [w / weight_sum for w in weights]
        
        # Get all unique tags
        all_tags = set()
        for source in score_sources:
            all_tags.update(source.keys())
        
        # Weighted average
        merged = {}
        for tag in all_tags:
            weighted_sum = 0.0
            for source, weight in zip(score_sources, weights):
                if tag in source:
                    weighted_sum += source[tag] * weight
            merged[tag] = weighted_sum
        
        return merged
    
    def compute_uncertainty(
        self,
        scores: Dict[str, float],
        method: str = "entropy"
    ) -> float:
        """Compute uncertainty of score distribution.
        
        Args:
            scores: Dict of tag -> score
            method: Uncertainty method ('entropy' or 'variance')
            
        Returns:
            Uncertainty value (higher = more uncertain)
        """
        if not scores:
            return 0.0
        
        # Normalize to probability distribution
        total = sum(scores.values())
        probs = {tag: score / total for tag, score in scores.items()}
        
        if method == "entropy":
            # Shannon entropy
            entropy = 0.0
            for prob in probs.values():
                if prob > 0:
                    entropy -= prob * math.log2(prob)
            return min(entropy / math.log2(len(probs)), 1.0)  # Normalize
        
        elif method == "variance":
            # Variance of scores
            mean = sum(scores.values()) / len(scores)
            variance = sum((s - mean) ** 2 for s in scores.values()) / len(scores)
            return min(math.sqrt(variance), 1.0)
        
        return 0.0


# Singleton instance
_calibrator: Optional[ConfidenceCalibrator] = None


def get_confidence_calibrator() -> ConfidenceCalibrator:
    """Get or create ConfidenceCalibrator singleton."""
    global _calibrator
    if _calibrator is None:
        _calibrator = ConfidenceCalibrator()
    return _calibrator
