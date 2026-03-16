"""Tag Conflict Resolution Service.

Handles mutual exclusion and consistency checking for tags.
"""

import logging
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ConflictResult:
    """Result of conflict resolution."""
    kept_tags: List[str]
    removed_tags: List[str]
    conflicts_found: List[Tuple[str, str, float]]
    warning_messages: List[str]


class TagConflictResolver:
    """Resolves tag conflicts and enforces mutual exclusion rules."""
    
    # Mutual exclusion rules: (tag1, tag2, priority_weight)
    # Higher weight means tag1 has priority over tag2
    MUTUAL_EXCLUSION: Dict[str, List[Tuple[str, float]]] = {
        # Age-related conflicts
        "蘿莉": [("熟女", 0.9), ("人妻", 0.9), ("御姐", 0.9), ("巨乳蘿莉", 0.5)],
        "正太": [("熟女", 0.9), ("人妻", 0.9)],
        "少女": [("熟女", 0.7), ("人妻", 0.7), ("老太婆", 1.0)],
        "人妻": [("蘿莉", 0.9), ("正太", 0.9)],
        "熟女": [("蘿莉", 0.9), ("正太", 0.8)],
        "老太婆": [("蘿莉", 1.0), ("少女", 1.0)],
        "年齡增長": [("年齡回溯", 1.0), ("返嬰癖", 1.0)],
        "年齡回溯": [("年齡增長", 1.0)],
        "返嬰癖": [("年齡增長", 1.0)],
        
        # Breast size conflicts
        "巨乳": [("貧乳", 1.0), ("平胸", 1.0), ("小胸部", 1.0)],
        "貧乳": [("巨乳", 1.0), ("大胸部", 1.0)],
        "平胸": [("巨乳", 1.0)],
        "小胸部": [("巨乳", 1.0)],
        
        # Theme conflicts
        "純愛": [("NTR", 1.0), ("凌辱", 0.8), ("調教", 0.7), ("強姦", 0.9), ("輪姦", 0.8)],
        "NTR": [("純愛", 1.0), ("綠帽", 0.5)],
        "凌辱": [("純愛", 0.8)],
        "調教": [("純愛", 0.7)],
        "強姦": [("純愛", 0.9)],
        "女性主導": [("男性主導", 1.0)],
        "男性主導": [("女性主導", 1.0)],
        
        # Gender conflicts
        "男": [("女", 0.8), ("女性", 0.8)],
        "女": [("男", 0.8), ("男性", 0.8)],
        "女性": [("男", 0.8)],
        "男性": [("女", 0.8)],
        "扶他": [("男", 0.5), ("女", 0.5)],
        "雙性人": [("男", 0.5), ("女", 0.5)],
        
        # Body size conflicts
        "大肌肉": [("纖細", 0.9), ("瘦", 0.9)],
        "肌肉": [("纖細", 0.7), ("瘦", 0.7)],
        "纖細": [("大肌肉", 0.9), ("肌肉", 0.7)],
        "瘦": [("大肌肉", 0.9), ("肌肉", 0.7)],
        
        # Clothing conflicts
        "內衣": [("全裸", 0.9), ("裸體", 0.9), ("上衣", 0.7)],
        "裸體": [("內衣", 0.9), ("穿衣", 0.9)],
        "全裸": [("內衣", 0.9), ("穿衣", 0.9)],
        
        # Hair color conflicts (keep the most prominent)
        "金髮": [("黑髮", 0.5), ("棕髮", 0.5), ("紅髮", 0.5)],
        "黑髮": [("金髮", 0.5), ("白髮", 0.5), ("紅髮", 0.5)],
        "白髮": [("黑髮", 0.5), ("金髮", 0.5)],
        "紅髮": [("金髮", 0.5), ("黑髮", 0.5)],
        
        # Race/species conflicts
        "貓娘": [("犬娘", 0.6), ("狐娘", 0.6)],
        "犬娘": [("貓娘", 0.6), ("狐娘", 0.6)],
        "狐娘": [("貓娘", 0.6), ("犬娘", 0.6)],
        "獸人": [("人類", 0.7)],
        "人類": [("獸人", 0.7), ("精靈", 0.5), ("魔物娘", 0.5)],
        
        # Action conflicts
        "口交": [("接吻", 0.3)],
        "肛交": [("陰道交", 0.4)],
        "自慰": [("做愛", 0.4)],
        "群交": [("做愛", 0.5)],
    }
    
    # Complementary tags (should not appear together)
    COMPLEMENTARY_EXCLUSION: List[Tuple[str, str]] = [
        ("近親", "無血緣"),
        ("人妻", "处女"),
        ("教師", "學生"),
    ]
    
    def __init__(self):
        """Initialize conflict resolver."""
        self.conflict_groups = []
        self.complementary_exclusion = []
        self._load_rules()
        logger.info(f"TagConflictResolver initialized with {len(self.conflict_groups)} conflict groups")

    def _load_rules(self):
        """Load conflict rules from JSON file and merge with hardcoded rules."""
        import json
        import os
        
        # 1. Convert hardcoded MUTUAL_EXCLUSION to list format
        hardcoded_groups = []
        for tag, conflicts in self.MUTUAL_EXCLUSION.items():
            # Check if we already have a group for this main tag
            # This is a simple conversion; assumes mutual exclusion
            # For a more robust merge, we might need to check IDs
            pass 
            
        # Actually, simpler: Use the JSON as base, and supplement with hardcoded
        # But wait, hardcoded structure is different: Dict[tag, List[Tuple[tag, score]]]
        # JSON structure is List[Dict[id, tags, description]]
        
        # Let's map hardcoded to JSON structure
        # We need to find groups. 
        # For valid conflict logic, we just need to ensure the rules are present.
        
        json_groups = []
        json_complementary = []
        
        # Default path
        file_path = os.path.join(os.path.dirname(__file__), "tag_conflicts.json")
        
        try:
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    json_groups = data.get("conflict_groups", [])
                    json_complementary = data.get("complementary_exclusions", [])
                    logger.info(f"Loaded {len(json_groups)} groups from {file_path}")
            else:
                logger.warning(f"Conflict rules file not found: {file_path}")
        except Exception as e:
            logger.error(f"Error loading conflict rules: {e}")

        # 2. Merge hardcoded rules into json_groups format isn't trivial because 
        # MUTUAL_EXCLUSION is directional (tag -> conflicting_tags).
        # JSON groups are sets of mutually exclusive tags.
        
        # However, check_conflicts uses self.conflict_groups.
        # Let's just create new groups for the hardcoded ones if they don't exist in JSON.
        
        # Actually, look at check_conflicts implementation (lines 154+).
        # It iterates over self.conflict_groups.
        
        # Allow enabling MUTUAL_EXCLUSION usage directly?
        # Let's just create a synthetic group for each entry in MUTUAL_EXCLUSION
        # This is inefficient (many small groups) but works.
        
        additional_groups = []
        for tag, conflicts in self.MUTUAL_EXCLUSION.items():
            for conflict_tag, _ in conflicts:
                # Create a mini group for this pair
                additional_groups.append({
                    "id": f"auto_{tag}_{conflict_tag}",
                    "tags": [tag, conflict_tag],
                    "description": "Hardcoded mutual exclusion"
                })
        
        # Combine
        self.conflict_groups = json_groups + additional_groups
        self.complementary_exclusion = json_complementary

    def check_conflicts(
        self, 
        tags: List[str], 
        scores: Optional[Dict[str, float]] = None
    ) -> ConflictResult:
        """Check and resolve conflicts in a list of tags.
        
        Args:
            tags: List of tag names
            scores: Optional dict of tag -> confidence score
            
        Returns:
            ConflictResult with resolved tags
        """
        if not tags:
            return ConflictResult(
                kept_tags=[],
                removed_tags=[],
                conflicts_found=[],
                warning_messages=[]
            )
        
        scores = scores or {tag: 0.5 for tag in tags}
        tags_set = set(tags)
        removed = set()
        conflicts = []
        warnings = []
        
        # 1. Handle Conflict Groups (Mutual Exclusion)
        # For each group, only one tag can survive - the one with the highest score
        for group in self.conflict_groups:
            group_tags = group.get("tags", [])
            present_tags = [t for t in group_tags if t in tags_set]
            
            if len(present_tags) > 1:
                # Find the tag with the highest score
                present_tags.sort(key=lambda t: scores.get(t, 0.0), reverse=True)
                winner = present_tags[0]
                losers = present_tags[1:]
                
                for loser in losers:
                    if loser not in removed:
                        removed.add(loser)
                        conflicts.append((winner, loser, 1.0)) # 1.0 means full priority
                        logger.info(f"Conflict Group '{group.get('id')}': Keeping '{winner}' ({scores.get(winner):.2f}), removing '{loser}' ({scores.get(loser):.2f})")
        
        # 2. Check Complementary Exclusions (Pairs that shouldn't coexist)
        for pair in self.complementary_exclusion:
            if len(pair) != 2:
                continue
            tag1, tag2 = pair
            if tag1 in tags_set and tag2 in tags_set:
                # Both found - determine which to keep based on score
                s1 = scores.get(tag1, 0.0)
                s2 = scores.get(tag2, 0.0)
                
                if s1 >= s2:
                    if tag2 not in removed:
                        removed.add(tag2)
                        conflicts.append((tag1, tag2, 1.0))
                else:
                    if tag1 not in removed:
                        removed.add(tag1)
                        conflicts.append((tag2, tag1, 1.0))
                
                warnings.append(f"Complementary tags conflict: '{tag1}' vs '{tag2}'")
        
        # Build result
        final_tags = [t for t in tags if t not in removed]
        
        return ConflictResult(
            kept_tags=final_tags,
            removed_tags=list(removed),
            conflicts_found=conflicts,
            warning_messages=warnings
        )

    
    def resolve(
        self,
        tags: List[str],
        scores: Optional[Dict[str, float]] = None,
        max_tags: int = 20
    ) -> Tuple[List[str], Dict[str, float]]:
        """Resolve tag conflicts and return cleaned tags.
        
        Args:
            tags: List of tag names
            scores: Optional dict of tag -> confidence score
            max_tags: Maximum number of tags to return
            
        Returns:
            Tuple of (resolved_tags, resolved_scores)
        """
        result = self.check_conflicts(tags, scores)
        
        # Sort by score (descending) and limit
        scored_tags = [(t, scores.get(t, 0.5)) for t in result.kept_tags]
        scored_tags.sort(key=lambda x: x[1], reverse=True)
        
        final_tags = [t for t, s in scored_tags[:max_tags]]
        final_scores = {t: s for t, s in scored_tags[:max_tags]}
        
        if result.removed_tags:
            logger.info(f"Resolved {len(result.removed_tags)} conflicts, removed: {result.removed_tags}")
        
        return final_tags, final_scores
    
    def get_category(self, tag: str) -> Optional[str]:
        """Get the category of a tag.
        
        Args:
            tag: Tag name
            
        Returns:
            Category string or None
        """
        # Character categories
        char_tags = {"蘿莉", "正太", "少女", "人妻", "熟女", "老太婆", "御姐"}
        if tag in char_tags:
            return "character"
        
        # Body categories
        body_tags = {"巨乳", "貧乳", "平胸", "大肌肉", "肌肉", "纖細", "瘦"}
        if tag in body_tags:
            return "body"
        
        # Theme categories
        theme_tags = {"純愛", "NTR", "凌辱", "調教", "強姦", "綠帽"}
        if tag in theme_tags:
            return "theme"
        
        # Clothing categories
        clothing_tags = {"內衣", "裸體", "全裸", "穿衣", "泳裝", "校服"}
        if tag in clothing_tags:
            return "clothing"
        
        return None
    
    def suggest_category_tags(
        self, 
        category: str, 
        exclude_tags: List[str],
        limit: int = 5
    ) -> List[str]:
        """Suggest tags from a category.
        
        Args:
            category: Category name
            exclude_tags: Tags to exclude
            limit: Maximum number of suggestions
            
        Returns:
            List of suggested tags
        """
        suggestions = []
        exclude_set = set(exclude_tags)
        
        if category == "character":
            candidates = ["蘿莉", "正太", "少女", "人妻", "熟女", "老太婆", "御姐"]
        elif category == "body":
            candidates = ["巨乳", "貧乳", "平胸", "大肌肉", "肌肉", "纖細", "瘦"]
        elif category == "theme":
            candidates = ["純愛", "NTR", "凌辱", "調教", "強姦"]
        elif category == "clothing":
            candidates = ["內衣", "裸體", "全裸", "穿衣", "泳裝", "校服", "女僕", "護士"]
        else:
            return []
        
        for tag in candidates:
            if tag not in exclude_set:
                suggestions.append(tag)
                if len(suggestions) >= limit:
                    break
        
        return suggestions


# Singleton instance
_conflict_resolver: Optional[TagConflictResolver] = None


def get_conflict_resolver() -> TagConflictResolver:
    """Get or create TagConflictResolver singleton."""
    global _conflict_resolver
    if _conflict_resolver is None:
        _conflict_resolver = TagConflictResolver()
    return _conflict_resolver
