"""
標籤衝突系統後續擴展優化方案
Phase 2-5: 高級智能衝突檢測與動態優化
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict, Counter
from enum import Enum

logger = logging.getLogger(__name__)


class ConflictSeverity(Enum):
    """衝突嚴重程度"""

    CRITICAL = "critical"  # 完全互斥，絕不能同時存在
    STRONG = "strong"  # 強衝突，通常不應同時存在
    MODERATE = "moderate"  # 中等衝突，特殊情況可共存
    WEAK = "weak"  # 輕微衝突，允許但需注意


class ConflictCategory(Enum):
    """衝突類別"""

    LOGICAL = "logical"  # 邏輯衝突（藍髮+紅髮）
    CONTEXTUAL = "contextual"  # 上下文衝突（純愛+強姦）
    SCENARIO = "scenario"  # 場景衝突（獨身+群交）
    PREFERENCE = "preference"  # 偏好衝突（可調整）


@dataclass
class AdvancedConflictRule:
    """高級衝突規則"""

    tag: str
    conflicts: List[str]
    severity: ConflictSeverity
    category: ConflictCategory
    description: str
    context_required: bool = False
    confidence_threshold: Optional[float] = None
    exceptions: List[str] = None


@dataclass
class ConflictAnalysis:
    """衝突分析結果"""

    conflict_pairs: List[Tuple[str, str, ConflictSeverity]]
    resolution_strategy: str
    recommended_tags: List[str]
    removed_tags: List[str]
    confidence_adjustments: Dict[str, float]


class AdvancedTagConflictSystem:
    """高級標籤衝突系統 v2.0"""

    def __init__(self):
        """初始化高級衝突系統"""
        self.conflict_rules: Dict[str, AdvancedConflictRule] = {}
        self.conflict_stats = defaultdict(int)
        self.tag_frequency = defaultdict(int)
        self.context_cache = {}
        self._load_advanced_rules()

    def _load_advanced_rules(self):
        """加載高級衝突規則"""

        # === 邏輯衝突（CRITICAL）===
        logical_conflicts = [
            AdvancedConflictRule(
                tag="藍髮",
                conflicts=[
                    "紅髮",
                    "金髮",
                    "黑髮",
                    "綠髮",
                    "紫髮",
                    "粉髮",
                    "白髮",
                    "棕髮",
                    "灰髮",
                    "銀髮",
                ],
                severity=ConflictSeverity.CRITICAL,
                category=ConflictCategory.LOGICAL,
                description="一個人只能有一種主要髮色",
            ),
            AdvancedConflictRule(
                tag="蘿莉",
                conflicts=["熟女", "人妻", "老太婆", "正太"],
                severity=ConflictSeverity.CRITICAL,
                category=ConflictCategory.LOGICAL,
                description="年齡衝突：不同年齡段互斥",
            ),
            AdvancedConflictRule(
                tag="巨乳",
                conflicts=["貧乳", "平胸", "爆乳"],
                severity=ConflictSeverity.CRITICAL,
                category=ConflictCategory.LOGICAL,
                description="身材類型衝突",
            ),
        ]

        # === 上下文衝突（STRONG）===
        contextual_conflicts = [
            AdvancedConflictRule(
                tag="純愛",
                conflicts=["強姦", "NTR", "調教", "凌辱"],
                severity=ConflictSeverity.STRONG,
                category=ConflictCategory.CONTEXTUAL,
                description="主題衝突：純愛與暴力內容不相容",
                context_required=True,
            ),
            AdvancedConflictRule(
                tag="學校",
                conflicts=["辦公室", "戰場", "未來"],
                severity=ConflictSeverity.MODERATE,
                category=ConflictCategory.CONTEXTUAL,
                description="場景衝突：特定場景互斥",
            ),
        ]

        # === 場景衝突（MODERATE）===
        scenario_conflicts = [
            AdvancedConflictRule(
                tag="獨身",
                conflicts=["群交", "多P"],
                severity=ConflictSeverity.MODERATE,
                category=ConflictCategory.SCENARIO,
                description="參與人數衝突",
            ),
            AdvancedConflictRule(
                tag="全裸",
                conflicts=["制服", "校服", "泳裝", "內衣"],
                severity=ConflictSeverity.STRONG,
                category=ConflictCategory.SCENARIO,
                description="服裝狀態衝突",
            ),
        ]

        # === 偏好衝突（WEAK）===
        preference_conflicts = [
            AdvancedConflictRule(
                tag="現代風格",
                conflicts=["古風", "和風", "蒸汽朋克"],
                severity=ConflictSeverity.WEAK,
                category=ConflictCategory.PREFERENCE,
                description="藝術風格衝突，可混合但影響一致性",
            ),
        ]

        # 合併所有規則
        all_conflicts = (
            logical_conflicts
            + contextual_conflicts
            + scenario_conflicts
            + preference_conflicts
        )

        for rule in all_conflicts:
            self.conflict_rules[rule.tag] = rule
            for conflict in rule.conflicts:
                if conflict not in self.conflict_rules:
                    # 自動生成反向規則（如果沒有定義）
                    reverse_rule = AdvancedConflictRule(
                        tag=conflict,
                        conflicts=[rule.tag],
                        severity=rule.severity,
                        category=rule.category,
                        description=f"反向衝突: {rule.description}",
                    )
                    self.conflict_rules[conflict] = reverse_rule

    async def analyze_conflicts_advanced(
        self, tags: List[Any], context: Optional[Dict[str, Any]] = None
    ) -> ConflictAnalysis:
        """
        高級衝突分析

        Args:
            tags: 標籤列表
            context: 上下文信息（場景、風格等）

        Returns:
            ConflictAnalysis: 詳細的衝突分析結果
        """

        # 統計標籤頻率
        tag_names = [self._extract_tag_name(tag) for tag in tags]
        for tag in tag_names:
            self.tag_frequency[tag] += 1

        # 識別所有衝突對
        conflict_pairs = []
        for i, tag1_name in enumerate(tag_names):
            for j, tag2_name in enumerate(tag_names[i + 1 :], i + 1):
                conflict_info = self._check_pair_conflict(tag1_name, tag2_name, context)
                if conflict_info:
                    severity = conflict_info["severity"]
                    conflict_pairs.append((tag1_name, tag2_name, severity))

        # 分析解決策略
        (
            resolution_strategy,
            recommended_tags,
            removed_tags,
        ) = await self._resolve_conflicts(tags, conflict_pairs, context)

        # 計算置信度調整
        confidence_adjustments = self._calculate_confidence_adjustments(
            conflict_pairs, recommended_tags, removed_tags
        )

        # 更新統計
        for pair in conflict_pairs:
            self.conflict_stats[f"{pair[0]}-{pair[1]}"] += 1

        return ConflictAnalysis(
            conflict_pairs=conflict_pairs,
            resolution_strategy=resolution_strategy,
            recommended_tags=recommended_tags,
            removed_tags=removed_tags,
            confidence_adjustments=confidence_adjustments,
        )

    def _check_pair_conflict(
        self, tag1: str, tag2: str, context: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """檢查兩個標籤是否衝突"""

        # 檢查直接規則
        rule1 = self.conflict_rules.get(tag1)
        rule2 = self.conflict_rules.get(tag2)

        if rule1 and tag2 in rule1.conflicts:
            return {
                "severity": rule1.severity,
                "category": rule1.category,
                "rule": rule1,
            }

        if rule2 and tag1 in rule2.conflicts:
            return {
                "severity": rule2.severity,
                "category": rule2.category,
                "rule": rule2,
            }

        # 檢查上下文衝突
        if context and rule1 and rule1.context_required:
            if self._check_contextual_conflict(tag1, tag2, context):
                return {
                    "severity": rule1.severity,
                    "category": rule1.category,
                    "rule": rule1,
                }

        return None

    async def _resolve_conflicts(
        self,
        tags: List[Any],
        conflict_pairs: List[Tuple[str, str, ConflictSeverity]],
        context: Optional[Dict[str, Any]],
    ) -> Tuple[str, List[str], List[str]]:
        """解決衝突，返回策略、保留標籤、移除標籤"""

        if not conflict_pairs:
            return "no_conflicts", self._extract_tag_names(tags), []

        # 按嚴重程度分組
        critical_conflicts = [
            p for p in conflict_pairs if p[2] == ConflictSeverity.CRITICAL
        ]
        strong_conflicts = [
            p for p in conflict_pairs if p[2] == ConflictSeverity.STRONG
        ]
        moderate_conflicts = [
            p for p in conflict_pairs if p[2] == ConflictSeverity.MODERATE
        ]

        tag_names = [self._extract_tag_name(tag) for tag in tags]
        tag_confidences = {
            self._extract_tag_name(tag): self._extract_confidence(tag) for tag in tags
        }

        # 處理CRITICAL衝突：必須解決
        removed_critical = []
        for tag1, tag2, _ in critical_conflicts:
            conf1 = tag_confidences.get(tag1, 0)
            conf2 = tag_confidences.get(tag2, 0)

            # 保留置信度高的，移除置信度低的
            if conf1 >= conf2 and tag2 not in removed_critical:
                removed_critical.append(tag2)
            elif conf2 > conf1 and tag1 not in removed_critical:
                removed_critical.append(tag1)

        # 處理STRONG衝突：通常解決，但可根據上下文調整
        removed_strong = []
        for tag1, tag2, _ in strong_conflicts:
            if context and self._should_allow_strong_conflict(tag1, tag2, context):
                continue  # 上下文允許，不處理

            conf1 = tag_confidences.get(tag1, 0)
            conf2 = tag_confidences.get(tag2, 0)

            if (
                conf1 >= conf2
                and tag2 not in removed_strong
                and tag2 not in removed_critical
            ):
                removed_strong.append(tag2)
            elif (
                conf2 > conf1
                and tag1 not in removed_strong
                and tag1 not in removed_critical
            ):
                removed_strong.append(tag1)

        # 處理MODERATE衝突：視情況處理
        removed_moderate = []
        for tag1, tag2, _ in moderate_conflicts:
            # MODERATE衝突可以共存，但降低置信度
            pass

        all_removed = removed_critical + removed_strong + removed_moderate
        recommended = [tag for tag in tag_names if tag not in all_removed]

        # 確定策略
        if critical_conflicts and strong_conflicts:
            strategy = "critical_strong_resolved"
        elif critical_conflicts:
            strategy = "critical_resolved"
        elif strong_conflicts:
            strategy = "strong_resolved"
        else:
            strategy = "moderate_resolved"

        return strategy, recommended, all_removed

    def _calculate_confidence_adjustments(
        self,
        conflict_pairs: List[Tuple[str, str, ConflictSeverity]],
        recommended_tags: List[str],
        removed_tags: List[str],
    ) -> Dict[str, float]:
        """計算置信度調整"""
        adjustments = {}

        # 被衝突影響的標籤可能需要降低置信度
        for tag1, tag2, severity in conflict_pairs:
            if tag1 in recommended_tags and tag2 in recommended_tags:
                # 兩個都保留了（MODERATE衝突），略微降低
                adjustment = -0.1 if severity == ConflictSeverity.MODERATE else -0.05
                adjustments[tag1] = max(adjustments.get(tag1, 0), adjustment)
                adjustments[tag2] = max(adjustments.get(tag2, 0), adjustment)

        return adjustments

    def _extract_tag_name(self, tag: Any) -> str:
        """提取標籤名稱"""
        if hasattr(tag, "tag"):
            return tag.tag
        elif isinstance(tag, str):
            return tag
        elif isinstance(tag, dict):
            return tag.get("tag", str(tag))
        return str(tag)

    def _extract_confidence(self, tag: Any) -> float:
        """提取置信度"""
        if hasattr(tag, "confidence"):
            return tag.confidence
        elif isinstance(tag, dict):
            return tag.get("confidence", 0.0)
        return 0.0

    def _check_contextual_conflict(
        self, tag1: str, tag2: str, context: Dict[str, Any]
    ) -> bool:
        """檢查上下文衝突"""
        # 實現上下文相關的衝突檢查邏輯
        scenario = context.get("scenario", "")
        style = context.get("style", "")

        # 示例：學校場景不應有辦公室主題
        if tag1 == "學校" and tag2 == "辦公室":
            return True

        return False

    def _should_allow_strong_conflict(
        self, tag1: str, tag2: str, context: Dict[str, Any]
    ) -> bool:
        """判斷是否應該允許STRONG衝突"""
        # 某些特殊情況下，STRONG衝突可以允許
        # 例如：藝術創作、惡搞、特殊情節
        content_type = context.get("content_type", "")

        if content_type in ["art", "parody", "fantasy"]:
            return True

        return False

    def get_conflict_statistics(self) -> Dict[str, Any]:
        """獲取衝突統計信息"""
        total_conflicts = sum(self.conflict_stats.values())
        most_common = Counter(self.conflict_stats).most_common(10)

        return {
            "total_conflicts": total_conflicts,
            "most_common_conflicts": most_common,
            "tag_frequencies": dict(self.tag_frequency),
            "rules_count": len(self.conflict_rules),
            "rules_by_severity": {
                severity.value: len(
                    [r for r in self.conflict_rules.values() if r.severity == severity]
                )
                for severity in ConflictSeverity
            },
            "rules_by_category": {
                category.value: len(
                    [r for r in self.conflict_rules.values() if r.category == category]
                )
                for category in ConflictCategory
            },
        }

    def export_rules(self, filepath: str):
        """導出規則到文件"""
        rules_data = []
        for rule in self.conflict_rules.values():
            rules_data.append(
                {
                    "tag": rule.tag,
                    "conflicts": rule.conflicts,
                    "severity": rule.severity.value,
                    "category": rule.category.value,
                    "description": rule.description,
                    "context_required": rule.context_required,
                    "confidence_threshold": rule.confidence_threshold,
                    "exceptions": rule.exceptions,
                }
            )

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(rules_data, f, ensure_ascii=False, indent=2)

        logger.info(f"Exported {len(rules_data)} conflict rules to {filepath}")
