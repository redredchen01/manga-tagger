#!/usr/bin/env python3
"""Test script for Chinese embedding service integration.

This script tests:
1. Chinese embedding service initialization
2. Text encoding and similarity calculations
3. Tag search functionality
4. Hybrid search with RAG service
5. Backward compatibility
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add app directory to path
sys.path.insert(0, str(Path(__file__).parent / "app"))

from app.config import settings
from app.services.chinese_embedding_service import get_chinese_embedding_service
from app.services.rag_service import RAGService

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ChineseEmbeddingTester:
    """Test suite for Chinese embedding integration."""

    def __init__(self):
        self.chinese_service = None
        self.rag_service = None
        self.test_results = []

    def log_test(self, test_name: str, passed: bool, message: str = ""):
        """Log test result."""
        status = "PASS" if passed else "FAIL"
        logger.info(f"[{status}] {test_name}: {message}")
        self.test_results.append(
            {"test": test_name, "passed": passed, "message": message}
        )

    async def test_chinese_service_initialization(self):
        """Test Chinese embedding service initialization."""
        try:
            self.chinese_service = get_chinese_embedding_service()

            # Check if service is available
            is_available = self.chinese_service.is_available()
            stats = self.chinese_service.get_stats()

            self.log_test(
                "Chinese Service Initialization",
                True,
                f"Available: {is_available}, Model: {stats['model_name']}",
            )

            return True

        except Exception as e:
            self.log_test("Chinese Service Initialization", False, str(e))
            return False

    async def test_text_encoding(self):
        """Test basic text encoding functionality."""
        if not self.chinese_service or not self.chinese_service.is_available():
            self.log_test("Text Encoding", False, "Service not available")
            return False

        try:
            test_texts = ["猫娘", "萝莉", "白发", "蓝眼", "校服"]

            # Test batch encoding
            embeddings = await self.chinese_service.encode_batch(test_texts)

            # Verify embeddings shape
            expected_shape = (len(test_texts), self.chinese_service.target_dim)
            actual_shape = embeddings.shape

            shape_match = actual_shape == expected_shape

            self.log_test(
                "Text Encoding",
                shape_match,
                f"Shape: {actual_shape}, Expected: {expected_shape}",
            )

            return shape_match

        except Exception as e:
            self.log_test("Text Encoding", False, str(e))
            return False

    async def test_similarity_calculation(self):
        """Test similarity calculation between texts."""
        if not self.chinese_service or not self.chinese_service.is_available():
            self.log_test("Similarity Calculation", False, "Service not available")
            return False

        try:
            # Test similar texts
            similarity1 = await self.chinese_service.calculate_similarity(
                "猫娘", "猫耳少女"
            )
            similarity2 = await self.chinese_service.calculate_similarity(
                "猫娘", "机械"
            )

            # Similar texts should have higher similarity
            similar_higher = similarity1 > similarity2

            self.log_test(
                "Similarity Calculation",
                similar_higher,
                f"猫娘 vs 猫耳少女: {similarity1:.3f}, 猫娘 vs 机械: {similarity2:.3f}",
            )

            return similar_higher

        except Exception as e:
            self.log_test("Similarity Calculation", False, str(e))
            return False

    async def test_tag_search(self):
        """Test tag search functionality."""
        if not self.chinese_service or not self.chinese_service.is_available():
            self.log_test("Tag Search", False, "Service not available")
            return False

        try:
            query_text = "一个有猫耳朵和尾巴的女孩"
            tag_list = [
                "猫娘",
                "萝莉",
                "白发",
                "黑发",
                "蓝眼",
                "棕眼",
                "校服",
                "连衣裙",
                "长发",
                "短发",
                "猫耳",
                "尾巴",
                "机械",
                "科幻",
                "现代",
                "古代",
            ]

            results = await self.chinese_service.search_tags_by_text(
                query_text=query_text, tag_list=tag_list, top_k=5, threshold=0.1
            )

            # Check if relevant tags are found
            found_relevant = any(
                result["tag"] in ["猫娘", "猫耳", "尾巴"] for result in results
            )

            self.log_test(
                "Tag Search",
                found_relevant,
                f"Found {len(results)} results, Relevant tags found: {found_relevant}",
            )

            # Print results for manual inspection
            logger.info("Tag search results:")
            for result in results:
                logger.info(f"  {result['tag']}: {result['similarity']:.3f}")

            return True

        except Exception as e:
            self.log_test("Tag Search", False, str(e))
            return False

    async def test_rag_service_integration(self):
        """Test RAG service integration with Chinese embeddings."""
        try:
            self.rag_service = RAGService()

            # Check if Chinese embeddings are enabled in RAG service
            chinese_enabled = getattr(settings, "USE_CHINESE_EMBEDDINGS", True)
            chinese_available = (
                self.rag_service.chinese_embedding_service
                and self.rag_service.chinese_embedding_service.is_available()
            )

            self.log_test(
                "RAG Service Integration",
                chinese_enabled,
                f"Chinese embeddings enabled: {chinese_enabled}, Available: {chinese_available}",
            )

            return True

        except Exception as e:
            self.log_test("RAG Service Integration", False, str(e))
            return False

    async def test_hybrid_search(self):
        """Test hybrid search functionality."""
        if not self.rag_service:
            self.log_test("Hybrid Search", False, "RAG service not initialized")
            return False

        try:
            # Create a dummy image for testing
            import io
            from PIL import Image

            # Create a simple test image
            test_image = Image.new("RGB", (100, 100), color="red")
            buffer = io.BytesIO()
            test_image.save(buffer, format="JPEG")
            image_bytes = buffer.getvalue()

            query_text = "一个红色的图像"
            tag_list = ["红色", "蓝色", "绿色", "黄色", "紫色"]

            # Test hybrid search
            results = await self.rag_service.hybrid_search(
                image_bytes=image_bytes,
                query_text=query_text,
                tag_list=tag_list,
                image_top_k=3,
                text_top_k=5,
                text_threshold=0.1,
            )

            # Check if results are returned
            has_image_results = len(results.get("image_results", [])) > 0
            has_text_results = len(results.get("text_results", [])) > 0

            self.log_test(
                "Hybrid Search",
                has_image_results or has_text_results,
                f"Image results: {len(results.get('image_results', []))}, "
                f"Text results: {len(results.get('text_results', []))}",
            )

            return True

        except Exception as e:
            self.log_test("Hybrid Search", False, str(e))
            return False

    async def test_backward_compatibility(self):
        """Test backward compatibility with existing RAG functionality."""
        if not self.rag_service:
            self.log_test(
                "Backward Compatibility", False, "RAG service not initialized"
            )
            return False

        try:
            # Test existing image search functionality
            import io
            from PIL import Image

            # Create a simple test image
            test_image = Image.new("RGB", (100, 100), color="blue")
            buffer = io.BytesIO()
            test_image.save(buffer, format="JPEG")
            image_bytes = buffer.getvalue()

            # Test existing search_similar method
            results = await self.rag_service.search_similar(image_bytes, top_k=3)

            # Should return results (even if empty due to no dataset)
            method_works = isinstance(results, list)

            self.log_test(
                "Backward Compatibility",
                method_works,
                f"search_similar method works: {method_works}, Results: {len(results)}",
            )

            return True

        except Exception as e:
            self.log_test("Backward Compatibility", False, str(e))
            return False

    async def test_configuration(self):
        """Test configuration settings."""
        try:
            # Check configuration values
            chinese_model = getattr(settings, "CHINESE_EMBEDDING_MODEL", "")
            use_chinese = getattr(settings, "USE_CHINESE_EMBEDDINGS", False)
            threshold = getattr(settings, "CHINESE_EMBEDDING_THRESHOLD", 0.0)

            config_valid = (
                bool(chinese_model)
                and isinstance(use_chinese, bool)
                and isinstance(threshold, float)
            )

            self.log_test(
                "Configuration",
                config_valid,
                f"Model: {chinese_model}, Enabled: {use_chinese}, Threshold: {threshold}",
            )

            return config_valid

        except Exception as e:
            self.log_test("Configuration", False, str(e))
            return False

    async def run_all_tests(self):
        """Run all tests and generate report."""
        logger.info("Starting Chinese Embedding Integration Tests")
        logger.info("=" * 50)

        # Run all tests
        tests = [
            self.test_configuration(),
            self.test_chinese_service_initialization(),
            self.test_text_encoding(),
            self.test_similarity_calculation(),
            self.test_tag_search(),
            self.test_rag_service_integration(),
            self.test_hybrid_search(),
            self.test_backward_compatibility(),
        ]

        await asyncio.gather(*tests)

        # Generate report
        logger.info("=" * 50)
        logger.info("Test Results Summary")
        logger.info("=" * 50)

        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if result["passed"])

        for result in self.test_results:
            status = "✓" if result["passed"] else "✗"
            logger.info(f"{status} {result['test']}: {result['message']}")

        logger.info("=" * 50)
        logger.info(
            f"Total: {total_tests}, Passed: {passed_tests}, Failed: {total_tests - passed_tests}"
        )

        if passed_tests == total_tests:
            logger.info("🎉 All tests passed!")
        else:
            logger.warning(f"⚠️ {total_tests - passed_tests} test(s) failed")

        return passed_tests == total_tests


async def main():
    """Main test function."""
    tester = ChineseEmbeddingTester()
    success = await tester.run_all_tests()

    if success:
        logger.info("Chinese embedding integration test completed successfully")
        return 0
    else:
        logger.error("Chinese embedding integration test failed")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
