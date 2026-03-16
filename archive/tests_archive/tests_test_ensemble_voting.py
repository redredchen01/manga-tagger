"""Test suite for Ensemble VLM Voting System.

Tests the complete pipeline: dispatcher, aggregator, and service.
"""

import sys
import asyncio
import json
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
from typing import List, Dict

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.vlm_response_parser import VLMResponseParser, ParsedResponse
from app.services.enhanced_vlm_dispatcher import EnhancedVLMDispatcher, ModelPrediction, DispatchResult
from app.services.ensemble_vote_aggregator import EnsembleVoteAggregator, VoteStrategy, EnsembleResult
from app.services.ensemble_vlm_service import EnsembleVLMService


def test_vlm_response_parser():
    """Test VLM response parsing."""
    print("\n" + "=" * 60)
    print("Testing VLM Response Parser")
    print("=" * 60)
    
    parser = VLMResponseParser()
    
    # Test cases
    test_cases = [
        # JSON format
        (
            '["巨乳", "貓娘", "校服"]',
            ["巨乳", "貓娘", "校服"],
        ),
        # Comma-separated
        (
            "巨乳, 貓娘, 校服, 藍髮",
            ["巨乳", "貓娘", "校服", "藍髮"],
        ),
        # Numbered list
        (
            "1. 巨乳\n2. 貓娘\n3. 校服",
            ["巨乳", "貓娘", "校服"],
        ),
        # Bullet points
        (
            "- 巨乳\n- 貓娘\n- 校服",
            ["巨乳", "貓娘", "校服"],
        ),
    ]
    
    passed = 0
    for response, expected_tags in test_cases:
        result = parser.parse(response)
        
        if set(result.tags) == set(expected_tags):
            print(f"[PASS] Parsed: {result.tags[:3]}...")
            passed += 1
        else:
            print(f"[FAIL] Expected {expected_tags}, got {result.tags}")
    
    print(f"\nParser: {passed}/{len(test_cases)} tests passed")
    return passed == len(test_cases)


def test_dispatcher():
    """Test VLM dispatcher (mock)."""
    print("\n" + "=" * 60)
    print("Testing VLM Dispatcher")
    print("=" * 60)
    
    # Create mock predictions
    predictions = [
        ModelPrediction(
            model_name="glm-4.6v-flash",
            raw_response="巨乳, 貓娘, 校服",
            parsed_response=ParsedResponse(
                raw_response="巨乳, 貓娘, 校服",
                tags=["巨乳", "貓娘", "校服"],
                confidence={"巨乳": 0.9, "貓娘": 0.85, "校服": 0.8},
                parsing_method="comma_list",
                is_valid=True,
            ),
            processing_time=1.5,
            confidence_scores={"巨乳": 0.9, "貓娘": 0.85, "校服": 0.8},
            is_valid=True,
        ),
        ModelPrediction(
            model_name="qwen-vl-max",
            raw_response="貓娘, 巨乳, 藍髮",
            parsed_response=ParsedResponse(
                raw_response="貓娘, 巨乳, 藍髮",
                tags=["貓娘", "巨乳", "藍髮"],
                confidence={"貓娘": 0.88, "巨乳": 0.87, "藍髮": 0.75},
                parsing_method="comma_list",
                is_valid=True,
            ),
            processing_time=2.0,
            confidence_scores={"貓娘": 0.88, "巨乳": 0.87, "藍髮": 0.75},
            is_valid=True,
        ),
        ModelPrediction(
            model_name="llava-next",
            raw_response="貓娘, 校服",
            parsed_response=ParsedResponse(
                raw_response="貓娘, 校服",
                tags=["貓娘", "校服"],
                confidence={"貓娘": 0.82, "校服": 0.78},
                parsing_method="comma_list",
                is_valid=True,
            ),
            processing_time=1.8,
            confidence_scores={"貓娘": 0.82, "校服": 0.78},
            is_valid=True,
        ),
    ]
    
    # Create dispatch result
    dispatch_result = DispatchResult(
        predictions=predictions,
        total_time=2.5,
        successful_models=3,
        failed_models=0,
        all_tags=["巨乳", "貓娘", "校服", "藍髮"],
    )
    
    print(f"Total predictions: {len(predictions)}")
    print(f"Successful models: {dispatch_result.successful_models}")
    print(f"Unique tags: {dispatch_result.all_tags}")
    print(f"Total time: {dispatch_result.total_time:.2f}s")
    
    return True


def test_aggregator():
    """Test ensemble vote aggregator."""
    print("\n" + "=" * 60)
    print("Testing Ensemble Vote Aggregator")
    print("=" * 60)
    
    # Create mock dispatch result
    predictions = [
        ModelPrediction(
            model_name="glm-4.6v-flash",
            raw_response="",
            parsed_response=ParsedResponse(
                raw_response="",
                tags=["巨乳", "貓娘", "校服", "藍髮"],
                confidence={},
                parsing_method="test",
                is_valid=True,
            ),
            processing_time=1.5,
            confidence_scores={"巨乳": 0.9, "貓娘": 0.85, "校服": 0.8, "藍髮": 0.7},
            is_valid=True,
        ),
        ModelPrediction(
            model_name="qwen-vl-max",
            raw_response="",
            parsed_response=ParsedResponse(
                raw_response="",
                tags=["貓娘", "巨乳", "藍髮", "金髮"],
                confidence={},
                parsing_method="test",
                is_valid=True,
            ),
            processing_time=2.0,
            confidence_scores={"貓娘": 0.88, "巨乳": 0.87, "藍髮": 0.75, "金髮": 0.65},
            is_valid=True,
        ),
        ModelPrediction(
            model_name="llava-next",
            raw_response="",
            parsed_response=ParsedResponse(
                raw_response="",
                tags=["貓娘", "校服", "藍髮"],
                confidence={},
                parsing_method="test",
                is_valid=True,
            ),
            processing_time=1.8,
            confidence_scores={"貓娘": 0.82, "校服": 0.78, "藍髮": 0.72},
            is_valid=True,
        ),
    ]
    
    dispatch_result = DispatchResult(
        predictions=predictions,
        total_time=2.5,
        successful_models=3,
        failed_models=0,
        all_tags=[],
    )
    
    # Test aggregator
    aggregator = EnsembleVoteAggregator(
        vote_threshold=0.5,
        strategy=VoteStrategy.WEIGHTED,
    )
    
    known_tags = {"巨乳", "貓娘", "校服", "藍髮", "金髮"}
    result = aggregator.aggregate(dispatch_result, known_tags=known_tags)
    
    print(f"\nFinal tags ({len(result.final_tags)}):")
    for tag in result.final_tags[:10]:
        vote = result.tag_votes.get(tag)
        if vote:
            print(f"  - {tag}: score={vote.weighted_score:.3f}, "
                  f"votes={vote.vote_count}/{vote.total_models}, "
                  f"agreement={vote.model_agreement:.0%}")
    
    print(f"\nModel agreements:")
    for model, count in sorted(result.model_agreements.items()):
        print(f"  - {model}: {count} tags in common")
    
    print(f"\nOverall confidence: {result.overall_confidence:.3f}")
    
    # Test consensus tags
    consensus = aggregator.get_consensus_tags(result)
    print(f"Consensus tags: {consensus}")
    
    return True


def test_full_pipeline():
    """Test the full ensemble pipeline."""
    print("\n" + "=" * 60)
    print("Testing Full Ensemble Pipeline")
    print("=" * 60)
    
    # Create service with mock dispatcher
    service = EnsembleVLMService()
    
    # Mock the dispatcher
    predictions = [
        ModelPrediction(
            model_name="glm-4.6v-flash",
            raw_response="巨乳, 貓娘, 校服",
            parsed_response=ParsedResponse(
                raw_response="巨乳, 貓娘, 校服",
                tags=["巨乳", "貓娘", "校服"],
                confidence={"巨乳": 0.9, "貓娘": 0.85, "校服": 0.8},
                parsing_method="comma_list",
                is_valid=True,
            ),
            processing_time=1.5,
            confidence_scores={"巨乳": 0.9, "貓娘": 0.85, "校服": 0.8},
            is_valid=True,
        ),
        ModelPrediction(
            model_name="qwen-vl-max",
            raw_response="貓娘, 巨乳, 藍髮",
            parsed_response=ParsedResponse(
                raw_response="貓娘, 巨乳, 藍髮",
                tags=["貓娘", "巨乳", "藍髮"],
                confidence={"貓娘": 0.88, "巨乳": 0.87, "藍髮": 0.75},
                parsing_method="comma_list",
                is_valid=True,
            ),
            processing_time=2.0,
            confidence_scores={"貓娘": 0.88, "巨乳": 0.87, "藍髮": 0.75},
            is_valid=True,
        ),
        ModelPrediction(
            model_name="llava-next",
            raw_response="貓娘, 校服",
            parsed_response=ParsedResponse(
                raw_response="貓娘, 校服",
                tags=["貓娘", "校服"],
                confidence={"貓娘": 0.82, "校服": 0.78},
                parsing_method="comma_list",
                is_valid=True,
            ),
            processing_time=1.8,
            confidence_scores={"貓娘": 0.82, "校服": 0.78},
            is_valid=True,
        ),
    ]
    
    # Create mock dispatch result
    mock_dispatch = DispatchResult(
        predictions=predictions,
        total_time=2.5,
        successful_models=3,
        failed_models=0,
        all_tags=["巨乳", "貓娘", "校服", "藍髮"],
    )
    
    # Mock the dispatcher method
    service.dispatcher.dispatch_all = AsyncMock(return_value=mock_dispatch)
    
    # Run analysis
    async def run_test():
        result = await service.analyze_image(b"mock_image_data", max_tags=10)
        return result
    
    result = asyncio.run(run_test())
    
    print(f"\nPipeline Results:")
    print(f"  Total tags: {result.total_tags}")
    print(f"  Processing time: {result.processing_time:.2f}s")
    print(f"  Models used: {result.model_used}")
    print(f"  Overall confidence: {result.overall_confidence:.3f}")
    
    print(f"\nFinal tags:")
    for tag in result.tags:
        print(f"  - {tag.tag}: confidence={tag.confidence:.3f}, "
              f"votes={tag.model_votes}/{len(result.model_used)}")
    
    if result.warnings:
        print(f"\nWarnings: {result.warnings}")
    
    if result.errors:
        print(f"\nErrors: {result.errors}")
    
    return True


def test_voting_strategies():
    """Test different voting strategies."""
    print("\n" + "=" * 60)
    print("Testing Voting Strategies")
    print("=" * 60)
    
    predictions = [
        ModelPrediction(
            model_name="glm-4.6v-flash",
            raw_response="",
            parsed_response=ParsedResponse(
                raw_response="",
                tags=["A", "B", "C"],
                confidence={"A": 0.9, "B": 0.7, "C": 0.5},
                parsing_method="test",
                is_valid=True,
            ),
            processing_time=1.5,
            confidence_scores={"A": 0.9, "B": 0.7, "C": 0.5},
            is_valid=True,
        ),
        ModelPrediction(
            model_name="qwen-vl-max",
            raw_response="",
            parsed_response=ParsedResponse(
                raw_response="",
                tags=["A", "B", "D"],
                confidence={"A": 0.85, "B": 0.75, "D": 0.6},
                parsing_method="test",
                is_valid=True,
            ),
            processing_time=2.0,
            confidence_scores={"A": 0.85, "B": 0.75, "D": 0.6},
            is_valid=True,
        ),
        ModelPrediction(
            model_name="llava-next",
            raw_response="",
            parsed_response=ParsedResponse(
                raw_response="",
                tags=["A", "C", "D"],
                confidence={"A": 0.8, "C": 0.55, "D": 0.65},
                parsing_method="test",
                is_valid=True,
            ),
            processing_time=1.8,
            confidence_scores={"A": 0.8, "C": 0.55, "D": 0.65},
            is_valid=True,
        ),
    ]
    
    dispatch_result = DispatchResult(
        predictions=predictions,
        total_time=2.5,
        successful_models=3,
        failed_models=0,
        all_tags=[],
    )
    
    strategies = [
        VoteStrategy.MAJORITY,
        VoteStrategy.WEIGHTED,
        VoteStrategy.CONFIDENCE,
    ]
    
    for strategy in strategies:
        aggregator = EnsembleVoteAggregator(
            vote_threshold=0.5,
            strategy=strategy,
        )
        
        result = aggregator.aggregate(dispatch_result)
        
        print(f"\nStrategy: {strategy.value}")
        print(f"  Tags selected: {result.final_tags}")
        print(f"  Overall confidence: {result.overall_confidence:.3f}")
    
    return True


def main():
    """Run all tests."""
    print("\n" + "#" * 60)
    print("# Ensemble VLM Voting System Test Suite")
    print("#" * 60)
    
    results = []
    
    results.append(("VLM Parser", test_vlm_response_parser()))
    results.append(("VLM Dispatcher", test_dispatcher()))
    results.append(("Vote Aggregator", test_aggregator()))
    results.append(("Full Pipeline", test_full_pipeline()))
    results.append(("Voting Strategies", test_voting_strategies()))
    
    print("\n" + "#" * 60)
    print("# Test Summary")
    print("#" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False
    
    print("\n" + ("=" * 60))
    if all_passed:
        print("All tests passed!")
    else:
        print("Some tests failed!")
    print("=" * 60)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
