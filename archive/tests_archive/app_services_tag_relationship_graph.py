"""
Tag Relationship Graph Module
Manages relationships between tags for validation and reasoning.

This module implements a graph-based system to track:
- Dependencies: Tags that commonly co-occur or imply each other
- Mutual Exclusions: Tags that cannot appear together
- Confidence Adjustments: Modify confidence based on tag relationships
"""

import json
import networkx as nx
from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class TagRelationship:
    """Represents a relationship between two tags."""

    source: str
    target: str
    relation_type: str  # "depends_on", "implies", "conflicts_with", "similar_to"
    confidence: float = 0.8  # Confidence in this relationship
    weight: float = 1.0  # Weight for reasoning

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "relation_type": self.relation_type,
            "confidence": self.confidence,
            "weight": self.weight,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TagRelationship":
        return cls(**data)


@dataclass
class ValidationResult:
    """Result of validating a tag combination."""

    is_valid: bool
    conflicts: List[Dict[str, Any]] = field(default_factory=list)
    recommendations: List[Dict[str, Any]] = field(default_factory=list)
    confidence_adjustments: Dict[str, float] = field(default_factory=dict)
    reasoning_chain: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "conflicts": self.conflicts,
            "recommendations": self.recommendations,
            "confidence_adjustments": self.confidence_adjustments,
            "reasoning_chain": self.reasoning_chain,
        }


class TagRelationshipGraph:
    """
    Graph-based system for managing tag relationships.

    Uses NetworkX to model relationships as a directed graph.
    Supports dependency tracking, conflict detection, and reasoning.
    """

    def __init__(self, persistence_path: Optional[str] = None):
        """
        Initialize the tag relationship graph.

        Args:
            persistence_path: Path to save/load relationship data
        """
        self.graph = nx.DiGraph()
        self.persistence_path = (
            Path(persistence_path)
            if persistence_path
            else Path("./data/tag_relationships.json")
        )

        # Default relationship definitions
        self._default_relationships = self._load_default_relationships()

        # Load existing relationships if available
        self._load_relationships()

        logger.info(
            f"TagRelationshipGraph initialized with {self.graph.number_of_nodes()} tags "
            f"and {self.graph.number_of_edges()} relationships"
        )

    def _load_default_relationships(self) -> Dict[str, List[Dict]]:
        """Load default tag relationship definitions."""
        return {
            "dependencies": [
                # Character dependencies
                {"source": "蘿莉", "target": "少女", "confidence": 0.9},
                {"source": "蘿莉", "target": "貧乳", "confidence": 0.8},
                {"source": "人妻", "target": "熟女", "confidence": 0.9},
                {"source": "貓娘", "target": "獸耳", "confidence": 0.95},
                {"source": "貓娘", "target": "尾巴", "confidence": 0.9},
                {"source": "女僕", "target": "絲襪", "confidence": 0.7},
                {"source": "女僕", "target": "洋裝", "confidence": 0.8},
                # Clothing dependencies
                {"source": "兔女郎", "target": "絲襪", "confidence": 0.9},
                {"source": "體操服", "target": "短褲", "confidence": 0.8},
                # Action dependencies
                {"source": "群交", "target": "多人", "confidence": 0.95},
                {"source": "3P", "target": "多人", "confidence": 0.9},
                {"source": "校服", "target": "學園", "confidence": 0.9},
                {"source": "泳裝", "target": "海邊", "confidence": 0.85},
            ],
            "conflicts": [
                # Age conflicts
                {"source": "蘿莉", "target": "人妻", "confidence": 1.0},
                {"source": "蘿莉", "target": "熟女", "confidence": 1.0},
                {"source": "少女", "target": "熟女", "confidence": 0.9},
                {"source": "正太", "target": "人妻", "confidence": 0.9},
                # Body feature conflicts
                {"source": "貧乳", "target": "巨乳", "confidence": 1.0},
                {"source": "貧乳", "target": "爆乳", "confidence": 1.0},
                # Theme conflicts
                {"source": "純愛", "target": "NTR", "confidence": 0.9},
                {"source": "純愛", "target": "強姦", "confidence": 0.8},
                {"source": "百合", "target": "耽美", "confidence": 0.9},
                # Clothing conflicts (partial)
                {"source": "和服", "target": "校服", "confidence": 0.7},
                {"source": "泳裝", "target": "校服", "confidence": 0.8},
                # Hair conflicts
                {"source": "單馬尾", "target": "短髮", "confidence": 1.0},
                {"source": "雙馬尾", "target": "短髮", "confidence": 1.0},
                {"source": "單馬尾", "target": "雙馬尾", "confidence": 1.0},
            ],
            "implies": [
                {"source": "貓娘", "target": "獸人", "confidence": 0.9},
                {"source": "狐娘", "target": "獸人", "confidence": 0.9},
                {"source": "犬娘", "target": "獸人", "confidence": 0.9},
                {"source": "蘿莉", "target": "未成年", "confidence": 0.8},
                {"source": "女僕", "target": "圍裙", "confidence": 0.85},
                {"source": "天使", "target": "翅膀", "confidence": 0.9},
                {"source": "惡魔", "target": "角", "confidence": 0.8},
                {"source": "惡魔", "target": "尾巴", "confidence": 0.7},
                {"source": "吸血鬼", "target": "獠牙", "confidence": 0.9},
            ],
            "similar_to": [
                {"source": "少女", "target": "年輕女性", "confidence": 0.9},
                {"source": "人妻", "target": "已婚女性", "confidence": 0.95},
                {"source": "貓娘", "target": "貓耳", "confidence": 0.8},
                {"source": "眼鏡", "target": "眼鏡娘", "confidence": 0.8},
            ],
        }

    def build_default_graph(self) -> None:
        """Build the graph with default relationships."""
        # Clear existing graph
        self.graph.clear()

        # Add dependency relationships
        for rel in self._default_relationships["dependencies"]:
            self.add_relationship(
                rel["source"], rel["target"], "depends_on", rel.get("confidence", 0.8)
            )

        # Add conflict relationships
        for rel in self._default_relationships["conflicts"]:
            self.add_relationship(
                rel["source"],
                rel["target"],
                "conflicts_with",
                rel.get("confidence", 1.0),
            )

        # Add implication relationships
        for rel in self._default_relationships["implies"]:
            self.add_relationship(
                rel["source"], rel["target"], "implies", rel.get("confidence", 0.8)
            )

        # Add similarity relationships
        for rel in self._default_relationships["similar_to"]:
            self.add_relationship(
                rel["source"], rel["target"], "similar_to", rel.get("confidence", 0.8)
            )

        self._save_relationships()
        logger.info(
            f"Built default graph with {self.graph.number_of_nodes()} nodes "
            f"and {self.graph.number_of_edges()} edges"
        )

    def add_relationship(
        self, source: str, target: str, relation_type: str, confidence: float = 0.8
    ) -> None:
        """
        Add a relationship between two tags.

        Args:
            source: Source tag
            target: Target tag
            relation_type: Type of relationship
            confidence: Confidence in this relationship (0-1)
        """
        # Add nodes if they don't exist
        if source not in self.graph:
            self.graph.add_node(source, tag=source)
        if target not in self.graph:
            self.graph.add_node(target, tag=target)

        # Add edge with attributes
        self.graph.add_edge(
            source, target, relation_type=relation_type, confidence=confidence
        )

    def remove_relationship(self, source: str, target: str) -> bool:
        """Remove a relationship between two tags."""
        if self.graph.has_edge(source, target):
            self.graph.remove_edge(source, target)
            return True
        return False

    def validate_tag_combination(
        self, tags: List[str], confidences: Optional[Dict[str, float]] = None
    ) -> ValidationResult:
        """
        Validate a combination of tags.

        Args:
            tags: List of tag names to validate
            confidences: Optional dictionary of confidence scores for each tag

        Returns:
            ValidationResult with conflicts, recommendations, and adjustments
        """
        result = ValidationResult(is_valid=True)
        confidences = confidences or {}

        # 1. Check for conflicts
        conflicts = self._find_conflicts(tags)
        if conflicts:
            result.is_valid = False
            result.conflicts = conflicts

        # 2. Find recommendations
        recommendations = self._suggest_related_tags(tags)
        result.recommendations = recommendations

        # 3. Calculate confidence adjustments
        adjustments = self._calculate_confidence_adjustments(tags, confidences)
        result.confidence_adjustments = adjustments

        # 4. Build reasoning chain
        reasoning = self._build_reasoning_chain(tags, conflicts, recommendations)
        result.reasoning_chain = reasoning

        return result

    def _find_conflicts(self, tags: List[str]) -> List[Dict[str, Any]]:
        """Find conflicts between tags."""
        conflicts = []
        tag_set = set(tags)

        for i, tag1 in enumerate(tags):
            for tag2 in tags[i + 1 :]:
                # Check direct conflict
                if self.graph.has_edge(tag1, tag2):
                    edge_data = self.graph.get_edge_data(tag1, tag2)
                    if edge_data.get("relation_type") == "conflicts_with":
                        conflicts.append(
                            {
                                "type": "direct_conflict",
                                "tags": [tag1, tag2],
                                "confidence": edge_data.get("confidence", 1.0),
                                "description": f"{tag1} 與 {tag2} 存在衝突關係",
                            }
                        )

                # Check reverse conflict
                if self.graph.has_edge(tag2, tag1):
                    edge_data = self.graph.get_edge_data(tag2, tag1)
                    if edge_data.get("relation_type") == "conflicts_with":
                        conflicts.append(
                            {
                                "type": "direct_conflict",
                                "tags": [tag2, tag1],
                                "confidence": edge_data.get("confidence", 1.0),
                                "description": f"{tag2} 與 {tag1} 存在衝突關係",
                            }
                        )

                # Check transitive conflicts through implication
                # If tag1 implies X and X conflicts with tag2
                for implied in self._get_implied_tags(tag1):
                    if self._has_conflict(implied, tag2):
                        conflicts.append(
                            {
                                "type": "transitive_conflict",
                                "tags": [tag1, tag2],
                                "through": implied,
                                "description": f"{tag1} 暗示 {implied}，與 {tag2} 衝突",
                            }
                        )

        return conflicts

    def _suggest_related_tags(self, tags: List[str]) -> List[Dict[str, Any]]:
        """Suggest related tags based on dependencies."""
        recommendations = []
        existing_tags = set(tags)

        for tag in tags:
            # Find tags that this tag depends on
            if tag in self.graph:
                for successor in self.graph.successors(tag):
                    edge_data = self.graph.get_edge_data(tag, successor)
                    if edge_data.get("relation_type") in ["depends_on", "implies"]:
                        if successor not in existing_tags:
                            recommendations.append(
                                {
                                    "tag": successor,
                                    "suggested_by": tag,
                                    "relation": edge_data.get("relation_type"),
                                    "confidence": edge_data.get("confidence", 0.8),
                                    "reason": f"{tag} 通常伴隨 {successor}",
                                }
                            )

                # Find tags that depend on this tag
                for predecessor in self.graph.predecessors(tag):
                    edge_data = self.graph.get_edge_data(predecessor, tag)
                    if edge_data.get("relation_type") == "depends_on":
                        if predecessor not in existing_tags:
                            recommendations.append(
                                {
                                    "tag": predecessor,
                                    "suggested_by": tag,
                                    "relation": "depended_by",
                                    "confidence": edge_data.get("confidence", 0.8),
                                    "reason": f"{predecessor} 通常需要 {tag}",
                                }
                            )

        # Sort by confidence
        recommendations.sort(key=lambda x: x["confidence"], reverse=True)

        # Remove duplicates
        seen_tags = set()
        unique_recommendations = []
        for rec in recommendations:
            if rec["tag"] not in seen_tags:
                seen_tags.add(rec["tag"])
                unique_recommendations.append(rec)

        return unique_recommendations[:10]  # Return top 10

    def _calculate_confidence_adjustments(
        self, tags: List[str], confidences: Dict[str, float]
    ) -> Dict[str, float]:
        """Calculate confidence adjustments based on tag relationships."""
        adjustments = {}

        for tag in tags:
            base_confidence = confidences.get(tag, 0.5)
            adjustment = 0.0

            # Check if tag has strong dependencies present
            if tag in self.graph:
                for successor in self.graph.successors(tag):
                    edge_data = self.graph.get_edge_data(tag, successor)
                    if edge_data.get("relation_type") == "depends_on":
                        if successor in tags:
                            # Dependency is satisfied, boost confidence
                            adjustment += 0.05 * edge_data.get("confidence", 0.8)
                        else:
                            # Dependency is missing, reduce confidence
                            adjustment -= 0.1 * edge_data.get("confidence", 0.8)

            # Check if tag conflicts with any present tag
            for other_tag in tags:
                if other_tag != tag and self._has_conflict(tag, other_tag):
                    # Strong penalty for conflicts
                    adjustment -= 0.2

            adjustments[tag] = max(-0.3, min(0.3, adjustment))  # Cap adjustments

        return adjustments

    def _build_reasoning_chain(
        self, tags: List[str], conflicts: List[Dict], recommendations: List[Dict]
    ) -> List[str]:
        """Build human-readable reasoning chain."""
        reasoning = []

        reasoning.append(f"正在驗證 {len(tags)} 個標籤的組合...")

        if conflicts:
            reasoning.append(f"發現 {len(conflicts)} 個衝突:")
            for conflict in conflicts:
                reasoning.append(f"  ⚠️  {conflict['description']}")
        else:
            reasoning.append("✅ 未發現標籤衝突")

        if recommendations:
            reasoning.append(f"💡 建議添加的相關標籤:")
            for rec in recommendations[:5]:  # Show top 5
                reasoning.append(
                    f"  • {rec['tag']} ({rec['reason']}, 信心度: {rec['confidence']:.2f})"
                )

        return reasoning

    def _get_implied_tags(self, tag: str) -> Set[str]:
        """Get all tags implied by a given tag."""
        implied = set()
        if tag in self.graph:
            for successor in self.graph.successors(tag):
                edge_data = self.graph.get_edge_data(tag, successor)
                if edge_data.get("relation_type") == "implies":
                    implied.add(successor)
        return implied

    def _has_conflict(self, tag1: str, tag2: str) -> bool:
        """Check if two tags have a conflict relationship."""
        if self.graph.has_edge(tag1, tag2):
            edge_data = self.graph.get_edge_data(tag1, tag2)
            if edge_data.get("relation_type") == "conflicts_with":
                return True
        if self.graph.has_edge(tag2, tag1):
            edge_data = self.graph.get_edge_data(tag2, tag1)
            if edge_data.get("relation_type") == "conflicts_with":
                return True
        return False

    def get_related_tags(
        self, tag: str, relation_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get all tags related to a given tag.

        Args:
            tag: Tag to find relationships for
            relation_type: Optional filter by relation type

        Returns:
            List of related tags with relationship info
        """
        related = []

        if tag not in self.graph:
            return related

        # Outgoing relationships
        for successor in self.graph.successors(tag):
            edge_data = self.graph.get_edge_data(tag, successor)
            if relation_type is None or edge_data.get("relation_type") == relation_type:
                related.append(
                    {
                        "tag": successor,
                        "relation": edge_data.get("relation_type"),
                        "confidence": edge_data.get("confidence", 0.8),
                        "direction": "outgoing",
                    }
                )

        # Incoming relationships
        for predecessor in self.graph.predecessors(tag):
            edge_data = self.graph.get_edge_data(predecessor, tag)
            if relation_type is None or edge_data.get("relation_type") == relation_type:
                related.append(
                    {
                        "tag": predecessor,
                        "relation": edge_data.get("relation_type"),
                        "confidence": edge_data.get("confidence", 0.8),
                        "direction": "incoming",
                    }
                )

        return related

    def export_graph(self) -> Dict[str, Any]:
        """Export the graph as a dictionary."""
        data = {"nodes": [], "edges": []}

        for node in self.graph.nodes():
            data["nodes"].append(
                {"tag": node, "attributes": dict(self.graph.nodes[node])}
            )

        for source, target, edge_data in self.graph.edges(data=True):
            data["edges"].append(
                {"source": source, "target": target, "attributes": dict(edge_data)}
            )

        return data

    def import_graph(self, data: Dict[str, Any]) -> None:
        """Import a graph from a dictionary."""
        self.graph.clear()

        for node_data in data.get("nodes", []):
            self.graph.add_node(node_data["tag"], **node_data.get("attributes", {}))

        for edge_data in data.get("edges", []):
            self.graph.add_edge(
                edge_data["source"],
                edge_data["target"],
                **edge_data.get("attributes", {}),
            )

    def _save_relationships(self) -> None:
        """Save relationships to disk."""
        try:
            self.persistence_path.parent.mkdir(parents=True, exist_ok=True)
            data = self.export_graph()

            with open(self.persistence_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

        except Exception as e:
            logger.error(f"Failed to save tag relationships: {e}")

    def _load_relationships(self) -> None:
        """Load relationships from disk."""
        if not self.persistence_path.exists():
            # Build default graph if no saved data
            self.build_default_graph()
            return

        try:
            with open(self.persistence_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.import_graph(data)
            logger.info(
                f"Loaded {self.graph.number_of_nodes()} tags and "
                f"{self.graph.number_of_edges()} relationships from disk"
            )

        except Exception as e:
            logger.error(f"Failed to load tag relationships: {e}")
            # Build default graph on error
            self.build_default_graph()


# Singleton instance
_relationship_graph: Optional[TagRelationshipGraph] = None


def get_tag_relationship_graph(
    persistence_path: Optional[str] = None,
) -> TagRelationshipGraph:
    """Get or create singleton instance of TagRelationshipGraph."""
    global _relationship_graph
    if _relationship_graph is None:
        _relationship_graph = TagRelationshipGraph(persistence_path)
    return _relationship_graph


def reset_tag_relationship_graph() -> None:
    """Reset the singleton instance."""
    global _relationship_graph
    _relationship_graph = None


if __name__ == "__main__":
    # Test the graph
    print("Testing TagRelationshipGraph...")
    print("=" * 60)

    graph = TagRelationshipGraph()

    # Test 1: Validate conflicting tags
    print("\n1. Testing conflict detection:")
    test_tags = ["蘿莉", "人妻", "貓娘"]
    result = graph.validate_tag_combination(test_tags)
    print(f"   Tags: {test_tags}")
    print(f"   Valid: {result.is_valid}")
    print(f"   Conflicts: {len(result.conflicts)}")
    for conflict in result.conflicts:
        print(f"     - {conflict['description']}")

    # Test 2: Validate compatible tags
    print("\n2. Testing compatible tags:")
    test_tags = ["蘿莉", "貧乳", "獸耳"]
    result = graph.validate_tag_combination(test_tags)
    print(f"   Tags: {test_tags}")
    print(f"   Valid: {result.is_valid}")
    print(f"   Recommendations: {len(result.recommendations)}")
    for rec in result.recommendations:
        print(f"     - {rec['tag']}: {rec['reason']}")

    # Test 3: Get related tags
    print("\n3. Testing related tags query:")
    related = graph.get_related_tags("貓娘")
    print(f"   Tags related to '貓娘':")
    for rel in related:
        print(f"     - {rel['tag']} ({rel['relation']}, {rel['direction']})")

    print("\n" + "=" * 60)
    print("✅ All tests passed!")
