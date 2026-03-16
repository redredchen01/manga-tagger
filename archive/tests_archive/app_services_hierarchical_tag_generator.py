"""Hierarchical Tag Generator.

Implements a two-stage tagging pipeline:
1. Category Identification - Identify major categories first
2. Fine-grained Tagging - Generate detailed tags per category
3. Consistency Validation - Ensure tag coherence
"""

import asyncio
import logging
from typing import List, Dict, Set, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict

from app.config import settings
from app.services.enhanced_vlm_dispatcher import EnhancedVLMDispatcher, ModelPrediction
from app.services.ensemble_vote_aggregator import EnsembleVoteAggregator
from app.services.tag_conflict_resolver import TagConflictResolver

logger = logging.getLogger(__name__)


class TagCategory(Enum):
    """Major tag categories."""
    CHARACTER = "character"      # 角色（蘿莉、正太、人妻等）
    BODY = "body"               # 身體特徵（巨乳、貧乳等）
    CLOTHING = "clothing"       # 服裝（校服、泳裝等）
    HAIR = "hair"              # 髮色髮型
    ACTION = "action"           # 動作（做愛、口交等）
    THEME = "theme"             # 主題（純愛、NTR等）
    SCENE = "scene"             # 場景（室內、戶外等）
    RELATIONSHIP = "relationship" # 關係（百合、耽美等）
    STYLE = "style"             # 畫風（寫實、卡通等）
    OTHER = "other"             # 其他


@dataclass
class CategoryResult:
    """Result from category identification."""
    category: TagCategory
    confidence: float
    evidence: List[str]
    tags: List[str] = field(default_factory=list)


@dataclass
class HierarchicalResult:
    """Result from hierarchical tagging."""
    categories: List[CategoryResult]
    all_tags: List[str]
    tags_by_category: Dict[TagCategory, List[str]]
    consistency_score: float
    warnings: List[str]
    rejected_tags: List[str]


class HierarchicalTagGenerator:
    """Hierarchical tag generator with category-aware processing.
    
    Pipeline:
    1. Identify major categories in the image
    2. For each category, generate fine-grained tags
    3. Validate consistency between categories
    4. Return coherent tag set
    """
    
    # Category keywords for identification
    CATEGORY_KEYWORDS = {
        TagCategory.CHARACTER: [
            "蘿莉", "正太", "少女", "人妻", "熟女", "御姐",
            "貓娘", "犬娘", "狐娘", "機娘", "天使", "惡魔",
            "loli", "shota", "catgirl", "maid", "nurse",
        ],
        TagCategory.BODY: [
            "巨乳", "貧乳", "平胸", "大胸部", "小胸部",
            "長腿", "美腿", "臀部", "屁股",
            "big_breasts", "small_breasts", "large_ass",
        ],
        TagCategory.CLOTHING: [
            "校服", "泳裝", "內衣", "女僕", "護士", "警察",
            "水手服", "體操服", "旗袍", "和服", "婚紗",
            "school_uniform", "swimsuit", "maid", "nurse",
        ],
        TagCategory.HAIR: [
            "金髮", "黑髮", "白髮", "紅髮", "藍髮", "綠髮",
            "雙馬尾", "馬尾", "短髮", "長髮", "捲髮",
            "blonde", "black_hair", "twintails",
        ],
        TagCategory.ACTION: [
            "做愛", "口交", "肛交", "手淫", "乳交", "足交",
            "顏射", "中出", "潮吹", "自慰", "69",
            "sex", "anal", "blowjob", "handjob",
        ],
        TagCategory.THEME: [
            "純愛", "NTR", "凌辱", "調教", "強姦", "綠帽",
            "亂倫", "百合", "耽美", "SM", "後宮",
            "love", "ntr", "rape", "yuri", "yaoi",
        ],
        TagCategory.SCENE: [
            "室內", "戶外", "學校", "臥室", "浴室", "客廳",
            "教室", "醫院", "餐廳",
            "indoor", "outdoor", "school", "bedroom",
        ],
        TagCategory.RELATIONSHIP: [
            "百合", "耽美", "母女", "母子", "姐妹", "師生",
            "yuri", "yaoi", "mother_daughter", "lesbian",
        ],
        TagCategory.STYLE: [
            "寫實", "卡通", "水彩", "寫意", "厚塗",
            "realistic", "cartoon", "watercolor",
        ],
    }
    
    # Category-specific prompts
    PROMPTS = {
        TagCategory.CHARACTER: """Identify the character types in this image.

Categories to look for:
- Age: 蘿莉 (loli), 正太 (shota), 少女 (young girl), 人妻 (married woman), 熟女 (milf)
- Type: 貓娘 (catgirl), 犬娘 (doggirl), 狐娘 (foxgirl), 機娘 (robot girl)
- Fantasy: 天使 (angel), 惡魔 (demon), 精靈 (elf)

Output format: character_type, character_type, ...

Examples:
- 蘿莉, 貓娘
- 人妻, 护士
- 正太, 少年

Characters:""",
        
        TagCategory.BODY: """Identify body features visible in this image.

Body features:
- Breast size: 巨乳 (large), 貧乳 (small), 平胸 (flat)
- Legs: 長腿 (long legs), 美腿 (beautiful legs)
- Other: 臀部 (buttocks), 腰 (waist)

Output format: feature, feature, ...

Features:""",
        
        TagCategory.CLOTHING: """Identify clothing/attire in this image.

Common clothing:
- Uniform: 校服 (school uniform), 水手服 (sailor), 體操服 (gym)
- Role: 女僕 (maid), 護士 (nurse), 警察 (police)
- Swimwear: 泳裝 (swimsuit), 比基尼 (bikini)
- Other: 內衣 (lingerie), 旗袍 (cheongsam)

Output format: clothing, clothing, ...

Clothing:""",
        
        TagCategory.HAIR: """Identify hair color and style in this image.

Hair colors:
- Light: 金髮 (blonde), 白髮 (white/silver), 銀髮
- Dark: 黑髮 (black), 棕髮 (brown)
- Bright: 紅髮 (red), 藍髮 (blue), 綠髮 (green), 粉紅髮 (pink)

Hair styles:
- 短髮 (short), 長髮 (long), 雙馬尾 (twintails), 馬尾 (ponytail)

Output format: color, style, ...

Hair:""",
        
        TagCategory.ACTION: """Identify visible actions or poses in this image.

Actions (only if clearly visible):
- Sexual: 做愛 (sex), 口交 (oral), 肛交 (anal), 手淫 (handjob)
- Non-sexual: 站立 (standing), 坐下 (sitting), 躺下 (lying)

IMPORTANT: Only include actions that are EXPLICITLY VISIBLE.

Output format: action, action, ...

Actions:""",
        
        TagCategory.THEME: """Identify the theme or genre of this image.

Themes:
- Romance: 純愛 (pure love), 後宮 (harem)
- Sensitive: NTR, 凌辱 (abuse), 調教 (training), 強姦 (rape)
- Relationship: 百合 (yuri), 耽美 (yaoi)

IMPORTANT: Only include if VISUALLY APPARENT from the image context.

Output format: theme, theme, ...

Themes:""",
        
        TagCategory.SCENE: """Identify the scene/location in this image.

Common locations:
- Indoor: 室內, 臥室, 浴室, 教室, 客廳
- Outdoor: 戶外, 公園, 學校, 海灘
- Venue: 醫院, 餐廳, 酒店

Output format: location, location, ...

Scene:""",
        
        TagCategory.RELATIONSHIP: """Identify the relationship between characters (if multiple people).

Relationships:
- 百合 (yuri - female x female)
- 耽美 (yaoi - male x male)
- 母女 (mother-daughter), 姐妹 (sisters)
- 師生 (teacher-student)

IMPORTANT: Only include if clearly visible.

Output format: relationship, relationship, ...

Relationships:""",
        
        TagCategory.STYLE: """Identify the art style of this image.

Styles:
- 寫實 (realistic), 卡通 (cartoon), 水彩 (watercolor)
- Japanese anime style
- Western comic style

Output format: style, style, ...

Style:""",
    }
    
    # Category hierarchy (parent -> children)
    CATEGORY_HIERARCHY = {
        TagCategory.CHARACTER: [TagCategory.BODY, TagCategory.CLOTHING, TagCategory.HAIR],
        TagCategory.THEME: [TagCategory.ACTION, TagCategory.RELATIONSHIP],
    }
    
    # Mutual exclusivity rules
    MUTUAL_EXCLUSIVE = {
        (TagCategory.CHARACTER, TagCategory.BODY): ["蘿莉", "正太"],
        (TagCategory.BODY, None): [("巨乳", "貧乳"), ("長髮", "短髮")],
    }
    
    def __init__(
        self,
        dispatcher: Optional[EnhancedVLMDispatcher] = None,
        aggregator: Optional[EnsembleVoteAggregator] = None,
        conflict_resolver: Optional[TagConflictResolver] = None,
    ):
        """Initialize hierarchical tag generator.
        
        Args:
            dispatcher: VLM dispatcher for model calls
            aggregator: Vote aggregator
            conflict_resolver: Tag conflict resolver
        """
        self.dispatcher = dispatcher or EnhancedVLMDispatcher()
        self.aggregator = aggregator or EnsembleVoteAggregator()
        self.conflict_resolver = conflict_resolver or TagConflictResolver()
        
        logger.info("HierarchicalTagGenerator initialized")
    
    async def generate(
        self,
        image_bytes: bytes,
        enable_consistency_check: bool = True,
    ) -> HierarchicalResult:
        """Generate tags hierarchically.
        
        Pipeline:
        1. Identify major categories
        2. Generate tags per category
        3. Validate consistency
        4. Return unified result
        
        Args:
            image_bytes: Image data
            enable_consistency_check: Enable consistency validation
            
        Returns:
            HierarchicalResult with tags organized by category
        """
        warnings = []
        rejected_tags = []
        
        # Step 1: Identify major categories
        categories = await self._identify_categories(image_bytes)
        
        if not categories:
            # Fallback: generate all categories
            categories = list(TagCategory)
            warnings.append("No specific categories identified, using all categories")
        
        # Step 2: Generate tags for each category
        category_tags: Dict[TagCategory, List[str]] = defaultdict(list)
        all_tags = set()
        
        for cat_result in categories:
            if cat_result.confidence < 0.3:
                # Skip low-confidence categories
                warnings.append(f"Low confidence for {cat_result.category.value}: {cat_result.confidence:.2f}")
                continue
            
            # Generate fine-grained tags for this category
            tags = await self._generate_category_tags(image_bytes, cat_result.category)
            
            if tags:
                category_tags[cat_result.category] = tags
                all_tags.update(tags)
                
                logger.debug(f"Category {cat_result.category.value}: {len(tags)} tags")
        
        # Step 3: Resolve conflicts
        if all_tags:
            tags_list = list(all_tags)
            scores = {tag: 0.8 for tag in tags_list}  # Default confidence
            
            resolved_tags, resolved_scores = self.conflict_resolver.resolve(
                tags_list, scores, max_tags=30
            )
            
            # Update all_tags
            rejected = set(tags_list) - set(resolved_tags)
            rejected_tags.extend(rejected)
            all_tags = set(resolved_tags)
            
            # Update category tags
            for cat in category_tags:
                category_tags[cat] = [t for t in category_tags[cat] if t in all_tags]
        
        # Step 4: Validate consistency
        consistency_score = 1.0
        if enable_consistency_check and all_tags:
            consistency_score = self._check_consistency(category_tags, all_tags)
            if consistency_score < 0.8:
                warnings.append(f"Low consistency score: {consistency_score:.2f}")
        
        # Build result
        result = HierarchicalResult(
            categories=categories,
            all_tags=list(all_tags),
            tags_by_category=dict(category_tags),
            consistency_score=consistency_score,
            warnings=warnings,
            rejected_tags=rejected_tags,
        )
        
        logger.info(
            f"Hierarchical tagging complete: {len(all_tags)} tags in {len(category_tags)} categories"
        )
        
        return result
    
    async def _identify_categories(
        self,
        image_bytes: bytes,
    ) -> List[CategoryResult]:
        """Identify major categories present in the image."""
        # Dispatch to multiple models for category identification
        prompt = """Analyze this image and identify which of these categories are present:

Categories:
- 角色 (character): 蘿莉, 正太, 貓娘, 人妻
- 身體 (body): 巨乳, 貧乳, 長腿
- 服裝 (clothing): 校服, 泳裝, 女僕, 內衣
- 動作 (action): 做愛, 口交, 站立, 坐下
- 主題 (theme): 純愛, NTR, 百合, 耽美
- 場景 (scene): 室內, 戶外, 學校
- 關係 (relationship): 百合, 耽美, 母女

Output format: category:confidence:evidence

Example output:
character:0.85:visible_loli_character
clothing:0.90:school_uniform_visible
action:0.30:not_clear

Only output categories that are VISIBLE. If not visible, don't include it.

Categories:"""
        
        try:
            dispatch_result = await self.dispatcher.dispatch_all(image_bytes, prompt)
            
            # Parse results
            categories = []
            for pred in dispatch_result.predictions:
                if pred.is_valid:
                    for line in pred.raw_response.split('\n'):
                        if ':' in line:
                            parts = line.split(':', 2)
                            if len(parts) >= 2:
                                cat_name = parts[0].strip().lower()
                                try:
                                    confidence = float(parts[1].strip())
                                    evidence = parts[2].strip() if len(parts) > 2 else ""
                                    
                                    # Match to enum
                                    for category in TagCategory:
                                        if category.value in cat_name:
                                            categories.append(CategoryResult(
                                                category=category,
                                                confidence=confidence,
                                                evidence=[evidence],
                                            ))
                                            break
                                except ValueError:
                                    continue
            
            # Sort by confidence and deduplicate
            categories.sort(key=lambda x: x.confidence, reverse=True)
            
            # Keep top categories (max 5)
            seen = set()
            unique = []
            for cat in categories:
                if cat.category not in seen:
                    seen.add(cat.category)
                    unique.append(cat)
            
            return unique[:5]
            
        except Exception as e:
            logger.error(f"Category identification failed: {e}")
            return []
    
    async def _generate_category_tags(
        self,
        image_bytes: bytes,
        category: TagCategory,
    ) -> List[str]:
        """Generate fine-grained tags for a specific category."""
        prompt = self.PROMPTS.get(category, "")
        
        if not prompt:
            return []
        
        try:
            # Dispatch to models
            dispatch_result = await self.dispatcher.dispatch_all(image_bytes, prompt)
            
            # Collect tags from all models
            all_tags = set()
            for pred in dispatch_result.predictions:
                if pred.is_valid:
                    # Parse comma-separated tags
                    raw_tags = pred.raw_response.split(',')
                    for tag in raw_tags:
                        tag = tag.strip()
                        # Clean tag
                        tag = self._clean_tag(tag)
                        if tag and len(tag) >= 2:
                            all_tags.add(tag)
            
            return list(all_tags)
            
        except Exception as e:
            logger.error(f"Category tag generation failed for {category.value}: {e}")
            return []
    
    def _clean_tag(self, tag: str) -> str:
        """Clean and normalize a tag."""
        # Remove common prefixes/suffixes
        tag = tag.strip()
        tag = tag.rstrip('.,，、')
        
        # Remove common phrases
        prefixes = ["tag:", "標籤:", "features:", "動作:", "服裝:"]
        for p in prefixes:
            if tag.lower().startswith(p.lower()):
                tag = tag[len(p):].strip()
        
        return tag
    
    def _check_consistency(
        self,
        category_tags: Dict[TagCategory, List[str]],
        all_tags: Set[str],
    ) -> float:
        """Check consistency between categories."""
        if not all_tags:
            return 1.0
        
        consistency_scores = []
        
        # Check for obvious conflicts
        conflicts = self.conflict_resolver.check_conflicts(list(all_tags))
        if conflicts.removed_tags:
            consistency_scores.append(0.7)
        
        # Check category-specific rules
        # e.g., "蘿莉" should have body-related tags consistent with age
        loli_tags = {"蘿莉", "loli", "正太", "shota"}
        if any(t in all_tags for t in loli_tags):
            # Check if mature body features are tagged
            mature_body = {"巨乳", "豐滿", "曲線"}
            if any(t in all_tags for t in mature_body):
                # This might be inconsistent
                consistency_scores.append(0.6)
        
        # Default score
        if not consistency_scores:
            return 1.0
        
        return sum(consistency_scores) / len(consistency_scores)
    
    def get_category_summary(self, result: HierarchicalResult) -> str:
        """Get a human-readable summary of hierarchical results."""
        lines = [
            f"Hierarchical Tagging Results",
            f"=" * 40,
            f"Total tags: {len(result.all_tags)}",
            f"Categories: {len(result.tags_by_category)}",
            f"Consistency: {result.consistency_score:.2%}",
            "",
            "Tags by Category:",
        ]
        
        for cat, tags in result.tags_by_category.items():
            if tags:
                lines.append(f"  {cat.value}: {', '.join(tags[:5])}")
        
        if result.warnings:
            lines.extend(["", "Warnings:"])
            for w in result.warnings:
                lines.append(f"  - {w}")
        
        return '\n'.join(lines)


# Singleton instance
_hierarchical_generator: Optional[HierarchicalTagGenerator] = None


def get_hierarchical_generator() -> HierarchicalTagGenerator:
    """Get or create hierarchical tag generator singleton."""
    global _hierarchical_generator
    if _hierarchical_generator is None:
        _hierarchical_generator = HierarchicalTagGenerator()
    return _hierarchical_generator
