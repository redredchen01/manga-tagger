"""
TagMatcher Module
Matches manga descriptions to tags using RAG-based similarity search.

Enhanced with Adaptive Threshold Service for dynamic threshold calculation
based on image complexity and historical category performance.
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np


@dataclass
class TagMatch:
    """Represents a single tag match result"""

    tag_name: str
    description: str
    similarity: float
    matched_text: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        return {
            "tag_name": self.tag_name,
            "description": self.description,
            "similarity": round(self.similarity, 4),
            "matched_text": self.matched_text,
        }


class TagMatcher:
    """
    Hybrid tag matcher that combines exact keyword matching with RAG-based similarity search.

    Strategy:
    1. Exact Match: First check if any keywords exactly match tag names (confidence=1.0)
    2. Keyword Match: Then check for partial keyword matches (confidence=0.8-1.0)
    3. Vector Search: Finally use RAG similarity search as fallback (confidence=0.3-0.8)

    Usage:
        from tag_vector_store import init_tag_store
        from tag_matcher import TagMatcher

        store = init_tag_store("51標籤庫.json")
        matcher = TagMatcher(tag_store=store)

        description = "A cat girl with large ears and tail in a school uniform"
        matches = matcher.match(description, top_k=10, threshold=0.5)
    """

    def __init__(
        self,
        tag_store,
        tag_loader=None,
        tag_mapper=None,
        default_top_k: int = 10,
        default_threshold: float = 0.50,
        hybrid_alpha: float = 0.70,
        adaptive_threshold_service=None,
        enable_adaptive_threshold: bool = True,
        dynamic_weight_calculator=None,
        enable_dynamic_weights: bool = True,
    ):
        """
        Initialize the TagMatcher.

        Args:
            tag_store: Initialized TagVectorStore instance
            tag_loader: Optional TagLoader instance for exact matching
            tag_mapper: Optional TagMapper instance for English->Chinese mapping
            default_top_k: Default number of results to return
            default_threshold: Default similarity threshold (0-1)
            hybrid_alpha: Weight for lexical matching (0-1)
            adaptive_threshold_service: Optional AdaptiveThresholdService instance
            enable_adaptive_threshold: Whether to use adaptive threshold calculation
            dynamic_weight_calculator: Optional DynamicWeightCalculator instance
            enable_dynamic_weights: Whether to use dynamic weight calculation
        """
        self.tag_store = tag_store
        self.tag_loader = tag_loader
        self.tag_mapper = tag_mapper
        self.default_top_k = default_top_k
        self.default_threshold = default_threshold
        self.hybrid_alpha = hybrid_alpha

        # Initialize adaptive threshold service
        self.enable_adaptive_threshold = enable_adaptive_threshold
        if adaptive_threshold_service:
            self.adaptive_threshold_service = adaptive_threshold_service
        elif enable_adaptive_threshold:
            try:
                from app.services.adaptive_threshold_service import (
                    get_adaptive_threshold_service,
                )

                self.adaptive_threshold_service = get_adaptive_threshold_service()
            except ImportError:
                self.adaptive_threshold_service = None
                self.enable_adaptive_threshold = False
        else:
            self.adaptive_threshold_service = None

        # Initialize dynamic weight calculator
        self.enable_dynamic_weights = enable_dynamic_weights
        if dynamic_weight_calculator:
            self.weight_calculator = dynamic_weight_calculator
        elif enable_dynamic_weights:
            try:
                from app.services.dynamic_weight_calculator import (
                    get_dynamic_weight_calculator,
                )

                self.weight_calculator = get_dynamic_weight_calculator()
            except ImportError:
                self.weight_calculator = None
                self.enable_dynamic_weights = False
        else:
            self.weight_calculator = None

    def match(
        self,
        description: str,
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
        enable_hybrid: bool = True,
        image_features: Optional[Dict[str, Any]] = None,
        tag_category: str = "general",
    ) -> List[TagMatch]:
        """
        Match a manga description against the tag database using hybrid matching.

        Strategy:
        1. Exact Match: Check if description contains exact tag names (confidence=1.0)
        2. Keyword Match: Extract keywords and match against tags (confidence=0.8-1.0)
        3. Vector Search: Use RAG similarity search as fallback (confidence=0.3-0.8)

        Args:
            description: Manga description from VLM
            top_k: Number of results to return (uses default if None)
            similarity_threshold: Minimum similarity score (uses default if None)
            enable_hybrid: Enable hybrid matching (exact + vector)
            image_features: Optional image features for adaptive threshold calculation
            tag_category: Category of tags being matched (for adaptive threshold)

        Returns:
            List of TagMatch objects sorted by similarity (descending)
        """
        top_k = top_k or self.default_top_k

        # Calculate dynamic threshold if adaptive service is enabled
        if self.enable_adaptive_threshold and self.adaptive_threshold_service:
            threshold = self.adaptive_threshold_service.calculate_dynamic_threshold(
                tag_category=tag_category, image_features=image_features or {}
            )
        else:
            threshold = similarity_threshold or self.default_threshold

        matches = []
        matched_tags = set()

        # Step 1: Exact and keyword matching (if tag_loader available)
        if enable_hybrid and self.tag_loader:
            keyword_matches = self._exact_keyword_match(description, threshold)
            for match in keyword_matches:
                if match.tag_name not in matched_tags:
                    matches.append(match)
                    matched_tags.add(match.tag_name)

        # Step 2: Vector similarity search
        vector_results = self.tag_store.search(
            query=description, top_k=top_k, similarity_threshold=threshold
        )

        for result in vector_results:
            if result["tag_name"] not in matched_tags:
                match = TagMatch(
                    tag_name=result["tag_name"],
                    description=result.get("description", ""),
                    similarity=result["similarity"],
                    matched_text=result.get("full_text", ""),
                )
                matches.append(match)
                matched_tags.add(result["tag_name"])

        # Sort by similarity (descending)
        matches.sort(key=lambda x: x.similarity, reverse=True)

        # Limit results
        return matches[:top_k]

    def _exact_keyword_match(
        self, description: str, threshold: float = 0.3
    ) -> List[TagMatch]:
        """
        Perform exact and keyword-based matching on description.
        Uses tag_mapper to convert English keywords to Chinese before matching.

        Args:
            description: Text to match against tags
            threshold: Minimum confidence threshold

        Returns:
            List of TagMatch objects
        """
        if not self.tag_loader:
            return []

        matches = []
        description_lower = description.lower()

        # Get all tag names
        tag_names = self.tag_loader.get_tag_names()

        # If tag_mapper available, also get mapped keywords
        mapped_keywords = []
        if self.tag_mapper:
            # Extract words from description
            desc_words = set(description_lower.replace(",", " ").split())

            # Map each word to Chinese
            for word in desc_words:
                cn_tag = self.tag_mapper.to_chinese(word)
                if cn_tag:
                    mapped_keywords.append(cn_tag)

        # Create list of all keywords to match (original + mapped)
        all_keywords = [description]
        if mapped_keywords:
            all_keywords.extend(mapped_keywords)

        for keyword in all_keywords:
            keyword_lower = keyword.lower()

            for tag_name in tag_names:
                tag_lower = tag_name.lower()
                confidence = 0.0

                # Exact match (full text)
                if tag_lower == keyword_lower:
                    confidence = 1.0
                # Tag contained in keyword
                elif tag_lower in keyword_lower:
                    confidence = 0.95
                # Keyword contained in tag
                elif keyword_lower in tag_lower:
                    confidence = 0.9
                # Partial word match
                else:
                    kw_words = set(keyword_lower.split())
                    tag_words = set(tag_lower.split())
                    matching = kw_words & tag_words
                    if matching:
                        confidence = 0.6 + (len(matching) / len(tag_words)) * 0.3

                if confidence >= threshold:
                    # Avoid duplicate tags
                    if not any(m.tag_name == tag_name for m in matches):
                        doc = self.tag_loader.get_by_tag_name(tag_name)
                        match = TagMatch(
                            tag_name=tag_name,
                            description=doc.metadata.get("description", "")
                            if doc
                            else "",
                            similarity=confidence,
                            matched_text=description,
                        )
                        matches.append(match)

        return matches

    def similarity_search_with_score(
        self, description: str, top_k: int = 10
    ) -> List[tuple]:
        """
        Search for tags and return (tag, score) tuples.

        This method provides compatibility with LangChain's similarity_search_with_score.

        Args:
            description: Manga description to search
            top_k: Number of results to return

        Returns:
            List of tuples: (tag_dict, similarity_score)
        """
        results = self.tag_store.search(
            query=description,
            top_k=top_k,
            similarity_threshold=0.0,  # Return all results to get scores
        )

        # Convert to (dict, score) format for LangChain compatibility
        tagged_results = []
        for result in results:
            tag_dict = {
                "tag_name": result["tag_name"],
                "description": result.get("description", ""),
            }
            tagged_results.append((tag_dict, result["similarity"]))

        return tagged_results

    def get_top_tags(
        self, description: str, top_k: int = 10, similarity_threshold: float = 0.3
    ) -> List[str]:
        """
        Get only the tag names from a search.

        Args:
            description: Manga description to search
            top_k: Number of tags to return
            similarity_threshold: Minimum similarity score

        Returns:
            List of tag names
        """
        matches = self.match(
            description=description,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
        )

        return [match.tag_name for match in matches]

    def get_tags_with_scores(
        self, description: str, top_k: int = 10, similarity_threshold: float = 0.3
    ) -> Dict[str, float]:
        """
        Get tags as a dictionary with their similarity scores.

        Args:
            description: Manga description to search
            top_k: Number of tags to return
            similarity_threshold: Minimum similarity score

        Returns:
            Dictionary: {tag_name: similarity_score}
        """
        matches = self.match(
            description=description,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
        )

        return {match.tag_name: match.similarity for match in matches}


def create_tag_matcher(
    tag_json_path: str = "51標籤庫.json",
    persist_directory: str = "./chroma_db",
    force_reload: bool = False,
    default_top_k: int = 10,
    default_threshold: float = 0.50,
    hybrid_alpha: float = 0.70,
    enable_hybrid: bool = True,
    enable_adaptive_threshold: bool = True,
) -> TagMatcher:
    """
    Create a fully configured TagMatcher instance with hybrid matching.

    Args:
        tag_json_path: Path to tag definitions JSON
        persist_directory: ChromaDB persistence directory
        force_reload: Force reload tag embeddings
        default_top_k: Default number of results
        default_threshold: Default similarity threshold (0.50 for better precision)
        hybrid_alpha: Weight for lexical matching (0.70 for better precision)
        enable_hybrid: Enable hybrid matching (exact + vector)
        enable_adaptive_threshold: Enable adaptive threshold calculation

    Returns:
        Configured TagMatcher instance
    """
    from tag_vector_store import init_tag_store
    from tag_loader import TagLoader
    from app.services.tag_mapper import TagMapper
    from app.services.tag_alias_service import TagAliasService

    print("Initializing TagMatcher...")
    print(f"   Tag database: {tag_json_path}")
    print(f"   Storage: {persist_directory}")
    print(f"   Default top_k: {default_top_k}")
    print(f"   Default threshold: {default_threshold}")
    print(f"   Hybrid alpha: {hybrid_alpha}")
    print(f"   Hybrid matching: {'enabled' if enable_hybrid else 'disabled'}")
    print(
        f"   Adaptive threshold: {'enabled' if enable_adaptive_threshold else 'disabled'}"
    )

    # Initialize tag store
    store = init_tag_store(
        json_path=tag_json_path,
        persist_directory=persist_directory,
        force_reload=force_reload,
    )

    # Initialize tag loader for exact matching
    loader = TagLoader(tag_json_path)
    loader.load()

    # Initialize tag mapper for English->Chinese mapping
    mapper = TagMapper()

    # Initialize tag alias service
    alias_service = TagAliasService()

    # Initialize adaptive threshold service if enabled
    adaptive_service = None
    if enable_adaptive_threshold:
        try:
            from app.services.adaptive_threshold_service import (
                get_adaptive_threshold_service,
            )

            adaptive_service = get_adaptive_threshold_service()
            print("   Adaptive threshold service: initialized")
        except Exception as e:
            print(f"   Adaptive threshold service: failed to initialize ({e})")

    # Initialize dynamic weight calculator
    weight_calculator = None
    try:
        from app.services.dynamic_weight_calculator import (
            get_dynamic_weight_calculator,
        )

        weight_calculator = get_dynamic_weight_calculator()
        print("   Dynamic weight calculator: initialized")
    except Exception as e:
        print(f"   Dynamic weight calculator: failed to initialize ({e})")

    # Create matcher with hybrid support
    matcher = TagMatcher(
        tag_store=store,
        tag_loader=loader if enable_hybrid else None,
        tag_mapper=mapper if enable_hybrid else None,
        default_top_k=default_top_k,
        default_threshold=default_threshold,
        hybrid_alpha=hybrid_alpha,
        adaptive_threshold_service=adaptive_service,
        enable_adaptive_threshold=enable_adaptive_threshold,
        dynamic_weight_calculator=weight_calculator,
        enable_dynamic_weights=True,
    )

    print("TagMatcher ready!")
    print(f"   Precision mode: threshold={default_threshold}, alpha={hybrid_alpha}")

    return matcher


if __name__ == "__main__":
    # Test the TagMatcher
    import sys

    if len(sys.argv) < 2:
        print("Usage: python tag_matcher.py <description>")
        print("Example: python tag_matcher.py 'cat girl with large breasts'")
        sys.exit(1)

    description = sys.argv[1]

    # Create matcher
    matcher = create_tag_matcher()

    print(f"\n🔍 Searching for: '{description}'")
    print("=" * 60)

    # Test match method
    matches = matcher.match(description, top_k=5, similarity_threshold=0.2)

    print(f"\n📊 Results (match method):")
    for i, match in enumerate(matches, 1):
        bar = "█" * int(match.similarity * 20)
        print(f"{i:2d}. {match.tag_name:20s} [{bar:20s}] {match.similarity:.3f}")

    print("\n" + "=" * 60)

    # Test similarity_search_with_score
    results = matcher.similarity_search_with_score(description, top_k=5)

    print(f"\n📊 Results (similarity_search_with_score):")
    for tag, score in results:
        print(f"   {tag['tag_name']:20s} -> {score:.3f}")

    print("\n" + "=" * 60)

    # Test get_tags_with_scores
    tagged = matcher.get_tags_with_scores(
        description, top_k=5, similarity_threshold=0.2
    )

    #HK|    print(f"\n📊 Results (get_tags_with_scores):")
#XT|    for tag, score in tagged.items():
#XS|        print(f"   {tag:20s} -> {score:.3f}")

# ======= Cached singleton for API usage =======
_tag_matcher_instance: Optional['TagMatcher'] = None


def get_tag_matcher(
    tag_json_path: str = "51標籤庫.json",
    persist_directory: str = "./chroma_db",
    force_reload: bool = False,
    default_top_k: int = 10,
    default_threshold: float = 0.50,
    hybrid_alpha: float = 0.70,
    enable_hybrid: bool = True,
    enable_adaptive_threshold: bool = True,
) -> 'TagMatcher':
    """
    Get cached TagMatcher singleton instance.
    
    This avoids reloading the embedding model on every API request.
    First call will initialize the matcher, subsequent calls return the cached instance.
    """
    global _tag_matcher_instance
    
    if _tag_matcher_instance is None:
        _tag_matcher_instance = create_tag_matcher(
            tag_json_path=tag_json_path,
            persist_directory=persist_directory,
            force_reload=force_reload,
            default_top_k=default_top_k,
            default_threshold=default_threshold,
            hybrid_alpha=hybrid_alpha,
            enable_hybrid=enable_hybrid,
            enable_adaptive_threshold=enable_adaptive_threshold,
        )
        print("[get_tag_matcher] TagMatcher singleton initialized!")
    
    return _tag_matcher_instance

    for tag, score in tagged.items():
        print(f"   {tag:20s} -> {score:.3f}")
