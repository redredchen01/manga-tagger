"""Ensemble Vote Aggregator.

Aggregates predictions from multiple VLM models using weighted voting,
conflict detection, and confidence calibration.
"""

import logging
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict, Counter
from enum import Enum

from app.config import settings
from app.services.enhanced_vlm_dispatcher import ModelPrediction, DispatchResult
from app.services.vlm_response_parser import ParsedResponse

logger = logging.getLogger(__name__)


class VoteStrategy(Enum):
    """Voting strategy options."""
    MAJORITY = "majority"  # Simple majority vote
    WEIGHTED = "weighted"  # Weight by model confidence
    CONFIDENCE = "confidence"  # Weight by prediction confidence
    COMBINED = "combined"  # Combine model and prediction confidence


@dataclass
class TagVote:
    """Vote information for a single tag."""
    tag: str
    vote_count: int
    total_models: int
    model_agreement: float
    avg_confidence: float
    weighted_score: float
    source_models: List[str]


@dataclass
class DisagreementInfo:
    """Information about model disagreement."""
    tag: str
    supporting_models: List[str]
    opposing_models: List[str]
    confidence_spread: float
    resolution: str  # "keep", "discard", "escalate"


@dataclass
class EnsembleResult:
    """Final result from ensemble voting."""
    final_tags: List[str]
    tag_scores: Dict[str, float]
    tag_votes: Dict[str, TagVote]
    disagreements: List[DisagreementInfo]
    model_agreements: Dict[str, int]
    overall_confidence: float
    processing_stats: Dict[str, any]
    rejected_tags: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class EnsembleVoteAggregator:
    """Aggregates predictions from multiple VLM models.
    
    Features:
    - Multiple voting strategies
    - Confidence-based weighting
    - Conflict detection and resolution
    - Model agreement analysis
    """
    
    # Model strength weights (can be tuned)
    MODEL_WEIGHTS = {
        "glm-4.6v-flash": 1.0,
        "qwen-vl-max": 1.1,
        "llava-next": 0.95,
        "yi-vision": 0.9,
    }
    
    # Tags that require consensus (higher threshold)
    CONSENSUS_TAGS = {
        "蘿莉", "正太", "幼女", "loli", "shota",
        "強姦", "rape", "anal", "肛交",
        "獸交", "bestiality",
    }
    
    def __init__(
        self,
        vote_threshold: float = 0.5,
        strategy: VoteStrategy = VoteStrategy.WEIGHTED,
        require_consensus: bool = False,
    ):
        """Initialize aggregator.
        
        Args:
            vote_threshold: Minimum vote ratio to include tag
            strategy: Voting strategy to use
            require_consensus: Require consensus for sensitive tags
        """
        self.vote_threshold = vote_threshold
        self.strategy = strategy
        self.require_consensus = require_consensus
        
        logger.info(
            f"EnsembleVoteAggregator initialized: "
            f"threshold={vote_threshold}, strategy={strategy.value}"
        )
    
    def aggregate(
        self,
        dispatch_result: DispatchResult,
        known_tags: Optional[Set[str]] = None,
    ) -> EnsembleResult:
        """Aggregate predictions from all models.
        
        Args:
            dispatch_result: Result from EnhancedVLMDispatcher
            known_tags: Set of valid tags from tag library
            
        Returns:
            EnsembleResult with final tags and metadata
        """
        if not dispatch_result.predictions:
            return EnsembleResult(
                final_tags=[],
                tag_scores={},
                tag_votes={},
                disagreements=[],
                model_agreements={},
                overall_confidence=0.0,
                processing_stats={
                    "total_models": 0,
                    "successful": 0,
                    "total_time": 0,
                },
                rejected_tags=[],
                warnings=["No predictions to aggregate"],
            )
        
        # Collect all predictions
        all_tags: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        model_tags: Dict[str, Set[str]] = defaultdict(set)
        
        for pred in dispatch_result.predictions:
            if not pred.is_valid:
                continue
            
            model_weight = self.MODEL_WEIGHTS.get(pred.model_name, 1.0)
            
            for tag in pred.parsed_response.tags:
                confidence = pred.confidence_scores.get(tag, 0.5)
                all_tags[tag][pred.model_name] = {
                    "confidence": confidence,
                    "weight": model_weight,
                }
                model_tags[pred.model_name].add(tag)
        
        # Calculate votes for each tag
        tag_votes: Dict[str, TagVote] = {}
        model_agreements: Dict[str, int] = defaultdict(int)
        
        num_successful = dispatch_result.successful_models
        total_models = len(dispatch_result.predictions)
        
        for tag, model_data in all_tags.items():
            supporting_models = list(model_data.keys())
            vote_count = len(supporting_models)
            vote_ratio = vote_count / num_successful if num_successful > 0 else 0
            
            # Calculate average confidence
            confidences = [d["confidence"] for d in model_data.values()]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            # Calculate weighted score
            if self.strategy == VoteStrategy.WEIGHTED:
                weighted_sum = sum(
                    d["confidence"] * d["weight"]
                    for d in model_data.values()
                )
                total_weight = sum(d["weight"] for d in model_data.values())
                weighted_score = weighted_sum / total_weight if total_weight > 0 else 0
            elif self.strategy == VoteStrategy.CONFIDENCE:
                weighted_score = avg_confidence
            else:
                weighted_score = vote_ratio
            
            # Update model agreements
            for model in supporting_models:
                model_agreements[model] += 1
            
            tag_votes[tag] = TagVote(
                tag=tag,
                vote_count=vote_count,
                total_models=num_successful,
                model_agreement=vote_ratio,
                avg_confidence=avg_confidence,
                weighted_score=weighted_score,
                source_models=supporting_models,
            )
        
        # Detect and resolve disagreements
        disagreements = self._detect_disagreements(tag_votes, num_successful)
        resolved_tags = self._resolve_disagreements(
            tag_votes, disagreements, known_tags
        )
        
        # Apply thresholds and select final tags
        final_tags = self._apply_thresholds(tag_votes, resolved_tags)
        
        # Calculate overall confidence
        if final_tags:
            avg_score = sum(
                tag_votes[tag].weighted_score
                for tag in final_tags
            ) / len(final_tags)
        else:
            avg_score = 0.0
        
        # Validate against known tags
        rejected_tags = []
        if known_tags:
            valid_tags = []
            for tag in final_tags:
                if tag in known_tags or any(
                    tag.lower() == kt.lower() for kt in known_tags
                ):
                    valid_tags.append(tag)
                else:
                    rejected_tags.append(tag)
            final_tags = valid_tags
        
        # Sort by weighted score
        final_tags.sort(
            key=lambda t: tag_votes.get(t, TagVote(t, 0, 0, 0, 0, 0, [])).weighted_score,
            reverse=True,
        )
        
        # Build result
        tag_scores = {
            tag: tag_votes[tag].weighted_score
            for tag in final_tags
            if tag in tag_votes
        }
        
        return EnsembleResult(
            final_tags=final_tags[:20],  # Max 20 tags
            tag_scores=tag_scores,
            tag_votes=tag_votes,
            disagreements=disagreements,
            model_agreements=dict(model_agreements),
            overall_confidence=avg_score,
            processing_stats={
                "total_models": total_models,
                "successful": num_successful,
                "total_time": dispatch_result.total_time,
                "unique_tags": len(all_tags),
            },
            rejected_tags=rejected_tags,
        )
    
    def _detect_disagreements(
        self,
        tag_votes: Dict[str, TagVote],
        num_models: int,
    ) -> List[DisagreementInfo]:
        """Detect tags where models disagree significantly."""
        disagreements = []
        
        for tag, vote in tag_votes.items():
            # Check if consensus is needed
            needs_consensus = (
                self.require_consensus or
                any(ct in tag for ct in self.CONSENSUS_TAGS)
            )
            
            if needs_consensus and vote.model_agreement < 0.8:
                # Disagreement on sensitive tag
                supporting = vote.source_models
                opposing = [
                    m for m in ["glm-4.6v-flash", "qwen-vl-max", "llava-next"]
                    if m not in supporting
                ]
                
                disagreements.append(DisagreementInfo(
                    tag=tag,
                    supporting_models=supporting,
                    opposing_models=opposing,
                    confidence_spread=1.0 - vote.model_agreement,
                    resolution="require_consensus",
                ))
            
            # Check for low agreement on any tag
            elif vote.model_agreement < 0.5:
                disagreements.append(DisagreementInfo(
                    tag=tag,
                    supporting_models=vote.source_models,
                    opposing_models=[],
                    confidence_spread=1.0 - vote.model_agreement,
                    resolution="review",
                ))
        
        return disagreements
    
    def _resolve_disagreements(
        self,
        tag_votes: Dict[str, TagVote],
        disagreements: List[DisagreementInfo],
        known_tags: Optional[Set[str]],
    ) -> Set[str]:
        """Resolve disagreements and return final tag set."""
        resolved = set()
        
        for vote in tag_votes.values():
            # Check if this tag had disagreement
            dis = next((d for d in disagreements if d.tag == vote.tag), None)
            
            if dis:
                if dis.resolution == "require_consensus":
                    # Require all models to agree
                    if vote.model_agreement >= 0.8:
                        resolved.add(vote.tag)
                    else:
                        logger.debug(f"Tag '{vote.tag}' rejected due to lack of consensus")
                elif dis.resolution == "review":
                    # Keep if confidence is high enough
                    if vote.avg_confidence >= 0.8:
                        resolved.add(vote.tag)
                    else:
                        logger.debug(f"Tag '{vote.tag}' rejected due to disagreement")
            else:
                # No disagreement, include if passes threshold
                resolved.add(vote.tag)
        
        return resolved
    
    def _apply_thresholds(
        self,
        tag_votes: Dict[str, TagVote],
        resolved_tags: Set[str],
    ) -> List[str]:
        """Apply voting thresholds to select final tags."""
        final = []
        
        for tag in resolved_tags:
            if tag not in tag_votes:
                continue
            
            vote = tag_votes[tag]
            
            # Apply vote threshold
            if vote.model_agreement < self.vote_threshold:
                logger.debug(f"Tag '{tag}' filtered: agreement={vote.model_agreement:.2%}")
                continue
            
            # Apply minimum vote count
            if vote.vote_count < 2 and len(tag_votes) > 3:
                logger.debug(f"Tag '{tag}' filtered: only {vote.vote_count} votes")
                continue
            
            final.append(tag)
        
        return final
    
    def get_model_performance(self, result: EnsembleResult) -> Dict[str, float]:
        """Get per-model performance metrics."""
        performance = {}
        
        total_tags = len(result.final_tags)
        if total_tags == 0:
            return performance
        
        for model, agreement_count in result.model_agreements.items():
            # Agreement rate (how often this model was in consensus)
            performance[model] = agreement_count / total_tags
        
        return performance
    
    def get_consensus_tags(self, result: EnsembleResult) -> List[str]:
        """Get tags that all models agreed on."""
        return [
            tag for tag in result.final_tags
            if result.tag_votes.get(tag, TagVote(tag, 0, 0, 0, 0, 0, [])).model_agreement == 1.0
        ]


# Singleton instance
_aggregator: Optional[EnsembleVoteAggregator] = None


def get_ensemble_aggregator() -> EnsembleVoteAggregator:
    """Get or create aggregator singleton."""
    global _aggregator
    if _aggregator is None:
        _aggregator = EnsembleVoteAggregator(
            vote_threshold=settings.ENSEMBLE_VOTE_THRESHOLD,
        )
    return _aggregator
