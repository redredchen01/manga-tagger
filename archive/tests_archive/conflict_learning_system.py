"""
智能衝突學習系統 - 基於機器學習的自適應衝突檢測
Phase 3: Machine Learning & Adaptive Learning
"""

import json
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from collections import defaultdict, Counter
from dataclasses import dataclass
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans


@dataclass
class ConflictPattern:
    """衝突模式"""

    tag_combination: Tuple[str, ...]
    frequency: int
    resolution_type: str
    success_rate: float
    user_feedback: List[bool]


@dataclass
class CooccurrenceStats:
    """共現統計"""

    tag1: str
    tag2: str
    cooccurrence_count: int
    conflict_count: int
    compatibility_score: float


class ConflictLearner:
    """衝突學習系統"""

    def __init__(self, model_path: str = "data/conflict_learner_models"):
        self.model_path = model_path
        self.conflict_history: List[Dict[str, Any]] = []
        self.tag_cooccurrence: Dict[Tuple[str, str], int] = defaultdict(int)
        self.conflict_patterns: Dict[Tuple[str, ...], ConflictPattern] = {}
        self.vectorizer = TfidfVectorizer(max_features=1000)
        self.tag_embeddings: Dict[str, np.ndarray] = {}
        self.compatibility_matrix: Dict[str, Dict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )

        self._load_models()

    def learn_from_tagging_session(
        self, tags: List[str], conflicts_resolved: Dict[str, Any]
    ):
        """從標註會話中學習"""

        # 記錄標籤共現
        for i, tag1 in enumerate(tags):
            for tag2 in tags[i + 1 :]:
                self.tag_cooccurrence[(tag1, tag2)] += 1
                self.tag_cooccurrence[(tag2, tag1)] += 1

        # 記錄衝突解決方案
        session_data = {
            "timestamp": self._get_timestamp(),
            "input_tags": tags,
            "conflicts_detected": conflicts_resolved.get("conflicts", []),
            "resolution": conflicts_resolved.get("resolution", ""),
            "final_tags": conflicts_resolved.get("final_tags", []),
            "user_satisfaction": conflicts_resolved.get("satisfaction", None),
        }

        self.conflict_history.append(session_data)

        # 更新衝突模式
        self._update_conflict_patterns(tags, conflicts_resolved)

        # 定期保存模型
        if len(self.conflict_history) % 100 == 0:
            self._save_models()

    def _update_conflict_patterns(
        self, tags: List[str], conflicts_resolved: Dict[str, Any]
    ):
        """更新衝突模式"""
        conflicts = conflicts_resolved.get("conflicts", [])
        resolution = conflicts_resolved.get("resolution", "")

        if conflicts:
            # 創建衝突模式的鍵
            conflict_tags = tuple(
                sorted(set([c[0] for c in conflicts] + [c[1] for c in conflicts]))
            )

            if conflict_tags not in self.conflict_patterns:
                self.conflict_patterns[conflict_tags] = ConflictPattern(
                    tag_combination=conflict_tags,
                    frequency=0,
                    resolution_type=resolution,
                    success_rate=0.0,
                    user_feedback=[],
                )

            # 更新模式統計
            pattern = self.conflict_patterns[conflict_tags]
            pattern.frequency += 1

            # 更新成功率（基於用戶反饋）
            satisfaction = conflicts_resolved.get("satisfaction", 3)  # 1-5分制
            if satisfaction is not None:
                is_good = satisfaction >= 4
                pattern.user_feedback.append(is_good)
                pattern.success_rate = sum(pattern.user_feedback) / len(
                    pattern.user_feedback
                )

    def predict_conflicts(self, tags: List[str]) -> List[Tuple[str, str, float]]:
        """預測標籤衝突"""
        predicted_conflicts = []

        for i, tag1 in enumerate(tags):
            for j, tag2 in enumerate(tags[i + 1 :], i + 1):
                # 基於歷史數據預測
                historical_conflict_prob = self._get_historical_conflict_probability(
                    tag1, tag2
                )

                # 基於語義相似度預測
                semantic_conflict_prob = self._get_semantic_conflict_probability(
                    tag1, tag2
                )

                # 基於共現統計預測
                cooccurrence_conflict_prob = (
                    self._get_cooccurrence_conflict_probability(tag1, tag2)
                )

                # 綜合評分
                total_conflict_prob = (
                    0.4 * historical_conflict_prob
                    + 0.4 * semantic_conflict_prob
                    + 0.2 * cooccurrence_conflict_prob
                )

                if total_conflict_prob > 0.5:  # 閾值可調整
                    predicted_conflicts.append((tag1, tag2, total_conflict_prob))

        return predicted_conflicts

    def _get_historical_conflict_probability(self, tag1: str, tag2: str) -> float:
        """基於歷史數據的衝突概率"""
        # 檢查是否有衝突模式
        for pattern in self.conflict_patterns.values():
            if tag1 in pattern.tag_combination and tag2 in pattern.tag_combination:
                return pattern.frequency / max(len(self.conflict_history), 1)

        return 0.0

    def _get_semantic_conflict_probability(self, tag1: str, tag2: str) -> float:
        """基於語義相似度的衝突概率"""
        # 獲取標籤嵌入
        embed1 = self.tag_embeddings.get(tag1)
        embed2 = self.tag_embeddings.get(tag2)

        if embed1 is None or embed2 is None:
            return 0.0

        # 計算相似度
        similarity = cosine_similarity([embed1], [embed2])[0][0]

        # 高相似度但語義衝突的標籤組
        semantic_conflicts = {
            ("藍髮", "紅髮"): 0.9,
            ("金髮", "黑髮"): 0.9,
            ("巨乳", "貧乳"): 0.95,
            ("蘿莉", "人妻"): 0.9,
            ("純愛", "強姦"): 0.95,
        }

        # 檢查是否在已知衝突中
        if (tag1, tag2) in semantic_conflicts:
            return semantic_conflicts[(tag1, tag2)]
        elif (tag2, tag1) in semantic_conflicts:
            return semantic_conflicts[(tag2, tag1)]

        # 基於相似度推斷
        if similarity > 0.8:  # 高相似度可能表示衝突
            return 0.3

        return 0.1

    def _get_cooccurrence_conflict_probability(self, tag1: str, tag2: str) -> float:
        """基於共現統計的衝突概率"""
        total_cooccurrences = sum(self.tag_cooccurrence.values())

        if total_cooccurrences == 0:
            return 0.0

        cooccurrence_count = self.tag_cooccurrence.get((tag1, tag2), 0)

        # 如果經常共現，衝突概率低
        cooccurrence_ratio = cooccurrence_count / total_cooccurrences

        if cooccurrence_ratio > 0.01:  # 經常共現
            return 0.1
        elif cooccurrence_ratio < 0.001:  # 從不共現
            return 0.3
        else:
            return 0.2

    def discover_new_conflicts(self, min_frequency: int = 5) -> List[ConflictPattern]:
        """發現新的衝突模式"""
        new_conflicts = []

        # 分析標籤組合頻率
        combination_counts = defaultdict(int)
        for session in self.conflict_history:
            tags = session.get("input_tags", [])
            for i, tag1 in enumerate(tags):
                for tag2 in tags[i + 1 :]:
                    combination = tuple(sorted([tag1, tag2]))
                    combination_counts[combination] += 1

        # 尋找高頻但低解決滿意度的組合
        for combination, count in combination_counts.items():
            if count >= min_frequency:
                # 檢查是否已經在現有規則中
                if combination not in self.conflict_patterns:
                    # 分析是否應該成為衝突規則
                    satisfaction_scores = []
                    for session in self.conflict_history:
                        if set(combination).issubset(
                            set(session.get("input_tags", []))
                        ):
                            sat = session.get("satisfaction", 3)
                            if sat is not None:
                                satisfaction_scores.append(sat)

                    if satisfaction_scores:
                        avg_satisfaction = sum(satisfaction_scores) / len(
                            satisfaction_scores
                        )

                        # 如果平均滿意度低，可能是潛在衝突
                        if avg_satisfaction < 3.0:
                            new_conflicts.append(
                                ConflictPattern(
                                    tag_combination=combination,
                                    frequency=count,
                                    resolution_type="unknown",
                                    success_rate=0.0,
                                    user_feedback=[],
                                )
                            )

        return new_conflicts

    def generate_suggested_rules(self) -> List[Dict[str, Any]]:
        """生成建議的新規則"""
        new_conflicts = self.discover_new_conflicts()

        suggested_rules = []
        for conflict in new_conflicts:
            rule = {
                "tag": conflict.tag_combination[0],
                "conflicts": list(conflict.tag_combination[1:]),
                "evidence": {
                    "frequency": conflict.frequency,
                    "pattern_type": "learned",
                    "confidence": min(conflict.frequency / 10, 0.9),
                },
                "description": f"學習到的衝突規則: {conflict.tag_combination}",
            }
            suggested_rules.append(rule)

        return suggested_rules

    def _load_models(self):
        """加載訓練好的模型"""
        try:
            # 加載衝突歷史
            history_path = os.path.join(self.model_path, "conflict_history.json")
            if os.path.exists(history_path):
                with open(history_path, "r", encoding="utf-8") as f:
                    self.conflict_history = json.load(f)

            # 加載共現統計
            cooccurrence_path = os.path.join(self.model_path, "cooccurrence_stats.json")
            if os.path.exists(cooccurrence_path):
                with open(cooccurrence_path, "r", encoding="utf-8") as f:
                    cooccurrence_data = json.load(f)
                    self.tag_cooccurrence = defaultdict(int)
                    for pair_str, count in cooccurrence_data.items():
                        pair = tuple(json.loads(pair_str))
                        self.tag_cooccurrence[pair] = count

            # 加載標籤嵌入
            embedding_path = os.path.join(self.model_path, "tag_embeddings.pkl")
            if os.path.exists(embedding_path):
                with open(embedding_path, "rb") as f:
                    self.tag_embeddings = pickle.load(f)

            # 加載衝突模式
            pattern_path = os.path.join(self.model_path, "conflict_patterns.json")
            if os.path.exists(pattern_path):
                with open(pattern_path, "r", encoding="utf-8") as f:
                    patterns_data = json.load(f)
                    for pattern_str, pattern_info in patterns_data.items():
                        combination = tuple(json.loads(pattern_str))
                        self.conflict_patterns[combination] = ConflictPattern(
                            tag_combination=combination,
                            frequency=pattern_info["frequency"],
                            resolution_type=pattern_info["resolution_type"],
                            success_rate=pattern_info["success_rate"],
                            user_feedback=pattern_info["user_feedback"],
                        )

        except Exception as e:
            print(f"Error loading models: {e}")

    def _save_models(self):
        """保存訓練模型"""
        os.makedirs(self.model_path, exist_ok=True)

        # 保存衝突歷史
        history_path = os.path.join(self.model_path, "conflict_history.json")
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(self.conflict_history, f, ensure_ascii=False, indent=2)

        # 保存共現統計
        cooccurrence_path = os.path.join(self.model_path, "cooccurrence_stats.json")
        cooccurrence_data = {
            json.dumps(list(pair)): count
            for pair, count in self.tag_cooccurrence.items()
        }
        with open(cooccurrence_path, "w", encoding="utf-8") as f:
            json.dump(cooccurrence_data, f, ensure_ascii=False, indent=2)

        # 保存標籤嵌入
        embedding_path = os.path.join(self.model_path, "tag_embeddings.pkl")
        with open(embedding_path, "wb") as f:
            pickle.dump(self.tag_embeddings, f)

        # 保存衝突模式
        pattern_path = os.path.join(self.model_path, "conflict_patterns.json")
        patterns_data = {}
        for combination, pattern in self.conflict_patterns.items():
            patterns_data[json.dumps(list(combination))] = {
                "frequency": pattern.frequency,
                "resolution_type": pattern.resolution_type,
                "success_rate": pattern.success_rate,
                "user_feedback": pattern.user_feedback,
            }
        with open(pattern_path, "w", encoding="utf-8") as f:
            json.dump(patterns_data, f, ensure_ascii=False, indent=2)

    def _get_timestamp(self) -> str:
        """獲取時間戳"""
        import datetime

        return datetime.datetime.now().isoformat()

    def get_learning_statistics(self) -> Dict[str, Any]:
        """獲取學習統計信息"""
        return {
            "total_sessions": len(self.conflict_history),
            "unique_tag_pairs": len(self.tag_cooccurrence),
            "discovered_patterns": len(self.conflict_patterns),
            "model_size": len(self.tag_embeddings),
            "last_updated": self._get_timestamp(),
        }
