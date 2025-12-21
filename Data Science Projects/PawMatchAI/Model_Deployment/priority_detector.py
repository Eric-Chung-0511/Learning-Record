# %%writefile priority_detector.py
import re
import numpy as np
from typing import Dict, List, Tuple, Set, Optional, Any
from dataclasses import dataclass, field
import traceback


@dataclass
class PriorityDetectionResult:
    """優先級檢測結果"""
    dimension_priorities: Dict[str, float] = field(default_factory=dict)
    detected_emphases: Dict[str, List[float]] = field(default_factory=dict)
    detected_rankings: Dict[str, int] = field(default_factory=dict)
    detected_negatives: List[str] = field(default_factory=list)
    detection_confidence: float = 1.0


class PriorityDetector:
    """
    優先級檢測器
    檢測使用者輸入中的優先級表達，包括強調關鍵字、排序詞、負面約束
    """

    def __init__(self):
        """初始化優先級檢測器"""
        self.emphasis_keywords = self._initialize_emphasis_keywords()
        self.ranking_keywords = self._initialize_ranking_keywords()
        self.negative_keywords = self._initialize_negative_keywords()
        self.dimension_keywords = self._initialize_dimension_keywords()
        self.absolute_max_priority = 2.5

    def _initialize_emphasis_keywords(self) -> Dict[str, Dict[str, List[str]]]:
        """初始化強調關鍵字"""
        return {
            'strong_emphasis': {
                'en': [
                    'most important', 'most importantly', 'must have', 'absolutely need',
                    'critical', 'essential', 'top priority', 'crucial',
                    'absolutely', 'definitely', 'certainly', 'paramount',
                    'vital', 'indispensable', 'mandatory', 'imperative'
                ]
            },
            'medium_emphasis': {
                'en': [
                    'really want', 'prefer', 'hope for', 'would like',
                    'strongly prefer', 'important', 'significant',
                    'really need', 'very important', 'highly prefer'
                ]
            },
            'mild_emphasis': {
                'en': [
                    'nice to have', 'ideally', 'if possible', 'bonus if',
                    'preferably', 'would be nice', 'hopefully',
                    'optimally', 'wish for'
                ]
            }
        }

    def _initialize_ranking_keywords(self) -> Dict[str, List[str]]:
        """初始化排序關鍵字"""
        return {
            'en': [
                'first', 'second', 'third', 'fourth', 'fifth',
                '1st', '2nd', '3rd', '4th', '5th',
                'firstly', 'secondly', 'thirdly',
                'primary', 'secondary', 'tertiary'
            ]
        }

    def _initialize_negative_keywords(self) -> Dict[str, List[str]]:
        """初始化負面約束關鍵字"""
        return {
            'en': [
                'must not', 'cannot', "don't want", "don't need",
                'absolutely no', 'cannot tolerate', 'no way',
                'avoid', 'never', 'not', 'refuse',
                'unacceptable', 'won\'t accept'
            ]
        }

    def _initialize_dimension_keywords(self) -> Dict[str, List[str]]:
        """初始化維度關鍵字映射"""
        return {
            'noise': [
                'quiet', 'silent', 'not noisy', "doesn't bark", 'peaceful',
                'noise', 'barking', 'vocal', 'loud', 'sound'
            ],
            'size': [
                'small', 'medium', 'large', 'tiny', 'big', 'compact',
                'size', 'giant', 'toy', 'miniature'
            ],
            'grooming': [
                'low maintenance', 'easy care', 'minimal grooming', 'low-maintenance',
                'grooming', 'care', 'maintenance', 'brush', 'shed', 'shedding'
            ],
            'family': [
                'good with kids', 'child friendly', 'family dog',
                'children', 'kids', 'family', 'toddler', 'baby'
            ],
            'exercise': [
                'active', 'exercise', 'energy', 'activity',
                'lazy', 'calm', 'energetic', 'athletic', 'work full time'
            ],
            'experience': [
                'first time', 'first dog', 'beginner', 'new to dogs', 'inexperienced',
                'easy to train', 'trainable', 'obedient', 'never owned', 'never had'
            ],
            'health': [
                'healthy', 'health', 'lifespan', 'longevity',
                'medical', 'genetic issues'
            ]
        }

    def detect_priorities(self, user_input: str) -> PriorityDetectionResult:
        """
        檢測使用者輸入中的優先級

        Args:
            user_input: 使用者輸入文字

        Returns:
            PriorityDetectionResult: 優先級檢測結果
        """
        try:
            if not user_input or not user_input.strip():
                return PriorityDetectionResult()

            normalized_input = user_input.lower().strip()

            # Step 1: 檢測強調關鍵字
            detected_emphases = self._detect_emphasis_keywords(normalized_input)

            # Step 2: 檢測排序詞
            detected_rankings = self._detect_explicit_ranking(normalized_input)

            # Step 3: 檢測負面約束
            detected_negatives = self._detect_negative_constraints(normalized_input)

            # Step 4: 檢測所有提及的維度（即使沒有強調詞）
            mentioned_dimensions = self._detect_mentioned_dimensions(normalized_input)

            # Step 5: 計算疊加優先級（包括提及的維度）
            dimension_priorities = self._calculate_final_priorities(
                detected_emphases, detected_rankings, mentioned_dimensions
            )

            # Step 6: 計算信心度
            detection_confidence = self._calculate_detection_confidence(
                detected_emphases, detected_rankings, normalized_input
            )

            return PriorityDetectionResult(
                dimension_priorities=dimension_priorities,
                detected_emphases=detected_emphases,
                detected_rankings=detected_rankings,
                detected_negatives=detected_negatives,
                detection_confidence=detection_confidence
            )

        except Exception as e:
            print(f"Error detecting priorities: {str(e)}")
            print(traceback.format_exc())
            return PriorityDetectionResult()

    def _detect_mentioned_dimensions(self, text: str) -> Set[str]:
        """
        檢測文字中提及的所有維度（不需要強調詞）

        Args:
            text: 正規化後的輸入文字

        Returns:
            Set[str]: 提及的維度集合
        """
        mentioned = set()

        for dimension, keywords in self.dimension_keywords.items():
            for keyword in keywords:
                if keyword in text:
                    mentioned.add(dimension)
                    break  # 一個維度只需匹配一次

        return mentioned

    def _detect_emphasis_keywords(self, text: str) -> Dict[str, List[float]]:
        """檢測強調關鍵字"""
        detected = {}

        # 定義權重倍數
        emphasis_weights = {
            'strong_emphasis': 2.0,
            'medium_emphasis': 1.5,
            'mild_emphasis': 1.2
        }

        # 為每個強調級別檢測
        for emphasis_level, keywords_dict in self.emphasis_keywords.items():
            weight = emphasis_weights[emphasis_level]

            for lang, keywords in keywords_dict.items():
                for keyword in keywords:
                    if keyword in text:
                        # 找到關鍵字附近的維度詞
                        dimensions = self._extract_nearby_dimensions(text, keyword)
                        for dimension in dimensions:
                            if dimension not in detected:
                                detected[dimension] = []
                            detected[dimension].append(weight)

        return detected

    def _detect_explicit_ranking(self, text: str) -> Dict[str, int]:
        """檢測明確排序詞"""
        detected = {}

        # 排序詞到排名的映射
        ranking_map = {
            'first': 1, '1st': 1, 'firstly': 1, 'primary': 1,
            'second': 2, '2nd': 2, 'secondly': 2, 'secondary': 2,
            'third': 3, '3rd': 3, 'thirdly': 3, 'tertiary': 3,
            'fourth': 4, '4th': 4,
            'fifth': 5, '5th': 5
        }

        for keyword in self.ranking_keywords['en']:
            if keyword in text:
                rank = ranking_map.get(keyword, 0)
                if rank > 0:
                    # 找到排序詞附近的維度詞
                    dimensions = self._extract_nearby_dimensions(text, keyword)
                    for dimension in dimensions:
                        # 如果已經有排名，取較高優先級（較小的數字）
                        if dimension in detected:
                            detected[dimension] = min(detected[dimension], rank)
                        else:
                            detected[dimension] = rank

        return detected

    def _detect_negative_constraints(self, text: str) -> List[str]:
        """檢測負面約束"""
        detected = []

        for lang, keywords in self.negative_keywords.items():
            for keyword in keywords:
                if keyword in text:
                    # 找到負面關鍵字附近的維度詞
                    dimensions = self._extract_nearby_dimensions(text, keyword)
                    detected.extend(dimensions)

        return list(set(detected))

    def _extract_nearby_dimensions(self, text: str, keyword: str, window: int = 50) -> List[str]:
        """
        提取關鍵字附近的維度詞

        Args:
            text: 文字
            keyword: 關鍵字
            window: 搜尋窗口大小（字元數）

        Returns:
            List[str]: 檢測到的維度列表
        """
        detected_dimensions = []

        # 找到關鍵字位置
        keyword_positions = [m.start() for m in re.finditer(re.escape(keyword), text)]

        for pos in keyword_positions:
            # 定義搜尋窗口
            start = max(0, pos - window)
            end = min(len(text), pos + len(keyword) + window)
            window_text = text[start:end]

            # 在窗口中搜尋維度關鍵字
            for dimension, dimension_keywords in self.dimension_keywords.items():
                for dim_keyword in dimension_keywords:
                    if dim_keyword in window_text:
                        detected_dimensions.append(dimension)
                        break  # 找到一個就夠了，不重複添加

        return list(set(detected_dimensions))

    def _calculate_final_priorities(self,
                                   detected_emphases: Dict[str, List[float]],
                                   detected_rankings: Dict[str, int],
                                   mentioned_dimensions: Set[str] = None) -> Dict[str, float]:
        """
        計算最終優先級（疊加邏輯）

        Args:
            detected_emphases: 檢測到的強調 {dimension: [weights]}
            detected_rankings: 檢測到的排序 {dimension: rank}
            mentioned_dimensions: 被提及但沒有強調詞的維度

        Returns:
            Dict[str, float]: 最終優先級分數
        """
        final_priorities = {}

        if mentioned_dimensions is None:
            mentioned_dimensions = set()

        # 合併所有提及的維度（包括強調、排序、和一般提及）
        all_dimensions = set(detected_emphases.keys()) | set(detected_rankings.keys()) | mentioned_dimensions

        for dimension in all_dimensions:
            emphasis_scores = detected_emphases.get(dimension, [])
            ranking = detected_rankings.get(dimension, 0)
            is_mentioned = dimension in mentioned_dimensions

            # 計算疊加分數
            if emphasis_scores or ranking > 0:
                # 有強調詞或排序詞
                final_score = self._calculate_stacked_priority(emphasis_scores, ranking)
            elif is_mentioned:
                # 僅被提及（沒有強調詞），給予基本優先級提升
                final_score = 1.3  # 基本提升，讓系統知道這個維度是使用者關心的
            else:
                final_score = 1.0

            final_priorities[dimension] = final_score

        return final_priorities

    def _calculate_stacked_priority(self,
                                   emphases: List[float],
                                   ranking: int = 0) -> float:
        """
        計算疊加後的優先級分數

        邏輯:
        1. 取最高強調作為基礎
        2. 其他強調提供遞減加成
        3. 排序詞轉換為權重並疊加
        4. 確保不超過絕對上限 2.5

        Args:
            emphases: 強調權重列表
            ranking: 排序位置 (1=first, 2=second, etc.)

        Returns:
            float: 最終優先級分數
        """
        if not emphases and ranking == 0:
            return 1.0

        # 轉換排序為權重
        ranking_weights = {
            1: 2.0,   # first
            2: 1.7,   # second
            3: 1.4,   # third
            4: 1.2,   # fourth
            5: 1.1    # fifth
        }
        ranking_weight = ranking_weights.get(ranking, 0.0)

        # 合併所有權重
        all_weights = emphases.copy()
        if ranking_weight > 0:
            all_weights.append(ranking_weight)

        if not all_weights:
            return 1.0

        # 排序取最高作為基礎
        sorted_weights = sorted(all_weights, reverse=True)
        base_score = sorted_weights[0]

        # 額外權重提供遞減加成 (reduced stacking bonus)
        bonus = 0.0
        for i, weight in enumerate(sorted_weights[1:], start=1):
            # 遞減加成: 第2個給30%, 第3個給15%, 第4個給7.5% (reduced from 50/25/12.5)
            bonus += (weight - 1.0) * (0.3 / i)

        final_score = min(base_score + bonus, self.absolute_max_priority)
        return final_score

    def _calculate_detection_confidence(self,
                                       detected_emphases: Dict[str, List[float]],
                                       detected_rankings: Dict[str, int],
                                       text: str) -> float:
        """
        計算檢測信心度

        Args:
            detected_emphases: 檢測到的強調
            detected_rankings: 檢測到的排序
            text: 原始文字

        Returns:
            float: 信心度 (0-1)
        """
        confidence = 0.5  # 基礎信心度

        # 有明確強調 +0.3
        if detected_emphases:
            confidence += 0.3

        # 有明確排序 +0.2
        if detected_rankings:
            confidence += 0.2

        # 文字長度適中 +0.1
        word_count = len(text.split())
        if 10 <= word_count <= 100:
            confidence += 0.1

        return min(1.0, confidence)

    def get_detection_summary(self, result: PriorityDetectionResult) -> Dict[str, Any]:
        """
        獲取檢測摘要

        Args:
            result: 優先級檢測結果

        Returns:
            Dict[str, Any]: 檢測摘要
        """
        return {
            'total_dimensions_detected': len(result.dimension_priorities),
            'high_priority_dimensions': [
                dim for dim, score in result.dimension_priorities.items() if score >= 1.5
            ],
            'dimension_priorities': result.dimension_priorities,
            'emphases_detected': len(result.detected_emphases),
            'rankings_detected': len(result.detected_rankings),
            'negative_constraints': result.detected_negatives,
            'detection_confidence': result.detection_confidence
        }


def detect_user_priorities(user_input: str) -> PriorityDetectionResult:
    """
    便利函數: 檢測使用者優先級

    Args:
        user_input: 使用者輸入

    Returns:
        PriorityDetectionResult: 檢測結果
    """
    detector = PriorityDetector()
    return detector.detect_priorities(user_input)


def get_priority_summary(user_input: str) -> Dict[str, Any]:
    """
    便利函數: 獲取優先級摘要

    Args:
        user_input: 使用者輸入

    Returns:
        Dict[str, Any]: 優先級摘要
    """
    detector = PriorityDetector()
    result = detector.detect_priorities(user_input)
    return detector.get_detection_summary(result)
