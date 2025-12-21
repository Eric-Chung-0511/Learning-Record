# %%writefile adaptive_score_distribution.py
import numpy as np
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass, field
import traceback


@dataclass
class GradientAnalysis:
    """梯度分析結果"""
    top_score: float
    bottom_score: float
    score_range: float
    top5_std: float
    top5_range: float
    gradient_type: str  # 'steep', 'moderate', 'flat'
    score_distribution: List[float] = field(default_factory=list)


@dataclass
class ScenarioClassification:
    """情境分類結果"""
    scenario_type: str  # 'perfect_match', 'good_choices', 'moderate_fit', 'challenging'
    confidence: float
    reasoning: str


@dataclass
class DistributionResult:
    """分數分佈結果"""
    final_scores: List[Tuple[str, float]] = field(default_factory=list)
    gradient_analysis: Optional[GradientAnalysis] = None
    scenario_classification: Optional[ScenarioClassification] = None
    adjustment_applied: str = 'none'
    adjustment_notes: List[str] = field(default_factory=list)


class AdaptiveScoreDistribution:
    """
    自適應分數分佈系統
    根據情境梯度自然形成分數分佈，不強制固定範圍

    核心理念:
    - 完美匹配 → 自然高分 (90+)
    - 多個選擇 → 自然接近 (差距2-5分)
    - 不適合 → 自然偏低 (60-70)
    - 保證最低分 >= 60
    """

    def __init__(self):
        """初始化自適應分數分佈系統"""
        self.min_score = 0.60  # 全域最低分（觸底保護）
        self.no_intervention_threshold = 0.10
        self.gradient_thresholds = {
            'steep_std': 0.04,
            'steep_range': 0.12,
            'flat_std': 0.02,
            'flat_range': 0.05
        }

    def distribute_scores(self,
                         raw_scores: List[Tuple[str, float]]) -> DistributionResult:
        """
        自適應分數分佈

        Args:
            raw_scores: 原始分數列表 [(breed_name, score), ...]

        Returns:
            DistributionResult: 分佈結果
        """
        try:
            if not raw_scores:
                return DistributionResult()

            # Step 1: 分析梯度
            gradient_analysis = self._analyze_gradient(raw_scores)

            # Step 2: 判斷情境
            scenario = self._classify_scenario(gradient_analysis)

            # Step 3: 決定調整策略
            adjusted_scores, adjustment_type, notes = self._apply_adaptive_strategy(
                raw_scores, scenario, gradient_analysis
            )

            # Step 4: 應用最低分保護
            final_scores = self._apply_floor_protection(adjusted_scores)

            return DistributionResult(
                final_scores=final_scores,
                gradient_analysis=gradient_analysis,
                scenario_classification=scenario,
                adjustment_applied=adjustment_type,
                adjustment_notes=notes
            )

        except Exception as e:
            print(f"Error distributing scores: {str(e)}")
            print(traceback.format_exc())
            return DistributionResult(
                final_scores=raw_scores,
                adjustment_applied='error_fallback'
            )

    def _analyze_gradient(self,
                         scores: List[Tuple[str, float]]) -> GradientAnalysis:
        """
        分析分數梯度特徵

        Args:
            scores: 分數列表

        Returns:
            GradientAnalysis: 梯度分析結果
        """
        try:
            sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
            score_values = [s[1] for s in sorted_scores]

            top_score = score_values[0] if score_values else 0.5
            bottom_score = score_values[-1] if score_values else 0.5
            score_range = top_score - bottom_score

            # 前5名統計
            top5_scores = score_values[:min(5, len(score_values))]
            top5_std = float(np.std(top5_scores)) if len(top5_scores) > 1 else 0.0
            top5_range = top5_scores[0] - top5_scores[-1] if len(top5_scores) >= 2 else 0.0

            # 梯度類型判斷
            if top5_std > self.gradient_thresholds['steep_std'] or \
               top5_range > self.gradient_thresholds['steep_range']:
                gradient_type = 'steep'
            elif top5_std < self.gradient_thresholds['flat_std'] or \
                 top5_range < self.gradient_thresholds['flat_range']:
                gradient_type = 'flat'
            else:
                gradient_type = 'moderate'

            return GradientAnalysis(
                top_score=top_score,
                bottom_score=bottom_score,
                score_range=score_range,
                top5_std=top5_std,
                top5_range=top5_range,
                gradient_type=gradient_type,
                score_distribution=score_values
            )

        except Exception as e:
            print(f"Error analyzing gradient: {str(e)}")
            return GradientAnalysis(
                top_score=0.5,
                bottom_score=0.5,
                score_range=0.0,
                top5_std=0.0,
                top5_range=0.0,
                gradient_type='moderate',
                score_distribution=[]
            )

    def _classify_scenario(self,
                          gradient_analysis: GradientAnalysis) -> ScenarioClassification:
        """
        根據梯度分析分類情境

        情境類型:
        1. perfect_match: 完美匹配（第1名分數高且梯度陡峭）
        2. good_choices: 多個好選擇（前5名分數都高且梯度平坦）
        3. moderate_fit: 中等匹配（第1名分數中等）
        4. challenging: 挑戰情境（第1名分數偏低）

        Args:
            gradient_analysis: 梯度分析結果

        Returns:
            ScenarioClassification: 情境分類結果
        """
        top_score = gradient_analysis.top_score
        gradient_type = gradient_analysis.gradient_type

        if top_score >= 0.88 and gradient_type == 'steep':  # Increased from 0.85
            return ScenarioClassification(
                scenario_type='perfect_match',
                confidence=0.9,
                reasoning="High top score with clear differentiation indicates perfect match"
            )

        elif top_score >= 0.78 and gradient_type == 'flat':  # Increased from 0.75
            return ScenarioClassification(
                scenario_type='good_choices',
                confidence=0.85,
                reasoning="Multiple high-scoring breeds with similar fitness"
            )

        elif top_score >= 0.68:  # Reduced from 0.70 to be less inflating
            return ScenarioClassification(
                scenario_type='moderate_fit',
                confidence=0.75,
                reasoning="Moderate match quality with acceptable options"
            )

        else:
            return ScenarioClassification(
                scenario_type='challenging',
                confidence=0.65,
                reasoning="Lower overall match quality, may need requirement adjustment"
            )

    def _apply_adaptive_strategy(self,
                                raw_scores: List[Tuple[str, float]],
                                scenario: ScenarioClassification,
                                gradient_analysis: GradientAnalysis) -> Tuple[List[Tuple[str, float]], str, List[str]]:
        """
        根據情境類型應用不同的調整策略

        Args:
            raw_scores: 原始分數
            scenario: 情境分類
            gradient_analysis: 梯度分析

        Returns:
            Tuple: (調整後分數, 調整類型, 調整註記)
        """
        sorted_scores = sorted(raw_scores, key=lambda x: x[1], reverse=True)
        notes = []

        if scenario.scenario_type == 'perfect_match':
            # 完美匹配: 不調整，保持自然
            notes.append("Perfect match scenario: No adjustment needed")
            return sorted_scores, 'no_adjustment', notes

        elif scenario.scenario_type == 'good_choices':
            # 多個好選擇: 確保最小區分度
            adjusted, adjustment_notes = self._ensure_minimum_differentiation(
                sorted_scores, gradient_analysis
            )
            notes.extend(adjustment_notes)
            return adjusted, 'minimum_differentiation', notes

        elif scenario.scenario_type == 'moderate_fit':
            # 中等匹配: 溫和提升
            adjusted, adjustment_notes = self._gentle_uplift(
                sorted_scores, target_top=0.80
            )
            notes.extend(adjustment_notes)
            return adjusted, 'gentle_uplift', notes

        elif scenario.scenario_type == 'challenging':
            # 挑戰情境: 適度提升但不過度
            adjusted, adjustment_notes = self._moderate_uplift(
                sorted_scores, target_top=0.72
            )
            notes.extend(adjustment_notes)
            return adjusted, 'moderate_uplift', notes

        return sorted_scores, 'no_adjustment', notes

    def _ensure_minimum_differentiation(self,
                                       scores: List[Tuple[str, float]],
                                       gradient_analysis: GradientAnalysis) -> Tuple[List[Tuple[str, float]], List[str]]:
        """
        確保最小區分度（當分數過於接近時）

        Args:
            scores: 分數列表
            gradient_analysis: 梯度分析

        Returns:
            Tuple: (調整後分數, 註記)
        """
        notes = []
        top5_range = gradient_analysis.top5_range

        # 如果前5名差距 >= 5%，不需要調整
        if top5_range >= 0.05:
            notes.append(f"Differentiation sufficient (range: {top5_range:.3f})")
            return scores, notes

        # 需要擴展區分度
        top5 = scores[:5]
        rest = scores[5:]

        target_range = 0.05
        current_top = top5[0][1] if top5 else 0.5
        current_bottom = top5[-1][1] if len(top5) > 0 else 0.5

        adjusted_top5 = []
        for i, (breed, score) in enumerate(top5):
            if len(top5) > 1:
                position = i / (len(top5) - 1)
                new_score = current_top - (position * target_range)
            else:
                new_score = score
            adjusted_top5.append((breed, new_score))

        notes.append(f"Expanded top 5 differentiation to {target_range:.1%}")
        return adjusted_top5 + rest, notes

    def _gentle_uplift(self,
                      scores: List[Tuple[str, float]],
                      target_top: float = 0.75) -> Tuple[List[Tuple[str, float]], List[str]]:
        """
        溫和提升（保持分數分佈形狀）

        Args:
            scores: 分數列表
            target_top: 目標第1名分數 (reduced from 0.80 to 0.75)

        Returns:
            Tuple: (調整後分數, 註記)
        """
        notes = []

        if not scores:
            return scores, notes

        current_top = scores[0][1]

        if current_top >= target_top:
            notes.append(f"Top score already sufficient ({current_top:.3f})")
            return scores, notes

        # 計算提升量
        uplift = target_top - current_top

        # 所有品種統一提升
        adjusted = [(breed, min(1.0, score + uplift)) for breed, score in scores]

        notes.append(f"Applied gentle uplift: +{uplift:.3f} to all breeds")
        return adjusted, notes

    def _moderate_uplift(self,
                        scores: List[Tuple[str, float]],
                        target_top: float = 0.68) -> Tuple[List[Tuple[str, float]], List[str]]:
        """
        適度提升（挑戰情境）

        Args:
            scores: 分數列表
            target_top: 目標第1名分數 (reduced from 0.72 to 0.68)

        Returns:
            Tuple: (調整後分數, 註記)
        """
        notes = []

        if not scores:
            return scores, notes

        current_top = scores[0][1]
        current_bottom = scores[-1][1] if scores else 0.5

        adjusted = []
        for breed, score in scores:
            # 非線性提升: 分數越高提升越多
            if current_top > current_bottom:
                relative_position = (score - current_bottom) / (current_top - current_bottom + 0.001)
            else:
                relative_position = 1.0

            uplift_factor = 1.0 + (relative_position * 0.12)  # 最多提升12% (reduced from 15%)
            new_score = min(1.0, score * uplift_factor)
            adjusted.append((breed, new_score))

        notes.append("Applied moderate uplift with position-based scaling")
        return adjusted, notes

    def _apply_floor_protection(self,
                               scores: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        """
        應用最低分保護（確保沒有品種低於60分）

        Args:
            scores: 分數列表

        Returns:
            List[Tuple[str, float]]: 保護後分數
        """
        protected = []
        for breed, score in scores:
            protected_score = max(self.min_score, score)
            protected.append((breed, protected_score))

        return protected

    def get_distribution_summary(self, result: DistributionResult) -> Dict[str, Any]:
        """
        獲取分佈摘要

        Args:
            result: 分佈結果

        Returns:
            Dict[str, Any]: 分佈摘要
        """
        if not result.final_scores:
            return {'error': 'No scores to summarize'}

        score_values = [s[1] for s in result.final_scores]

        return {
            'scenario_type': result.scenario_classification.scenario_type if result.scenario_classification else 'unknown',
            'adjustment_applied': result.adjustment_applied,
            'score_statistics': {
                'top_score': max(score_values) if score_values else 0,
                'bottom_score': min(score_values) if score_values else 0,
                'mean_score': float(np.mean(score_values)) if score_values else 0,
                'std_score': float(np.std(score_values)) if score_values else 0,
                'range': max(score_values) - min(score_values) if score_values else 0
            },
            'gradient_info': {
                'type': result.gradient_analysis.gradient_type if result.gradient_analysis else 'unknown',
                'top5_std': result.gradient_analysis.top5_std if result.gradient_analysis else 0,
                'top5_range': result.gradient_analysis.top5_range if result.gradient_analysis else 0
            },
            'adjustment_notes': result.adjustment_notes,
            'top_3_breeds': result.final_scores[:3] if result.final_scores else []
        }


def distribute_breed_scores(raw_scores: List[Tuple[str, float]]) -> DistributionResult:
    """
    便利函數: 分佈品種分數

    Args:
        raw_scores: 原始分數列表

    Returns:
        DistributionResult: 分佈結果
    """
    distributor = AdaptiveScoreDistribution()
    return distributor.distribute_scores(raw_scores)


def get_distribution_summary(raw_scores: List[Tuple[str, float]]) -> Dict[str, Any]:
    """
    便利函數: 獲取分佈摘要

    Args:
        raw_scores: 原始分數列表

    Returns:
        Dict[str, Any]: 分佈摘要
    """
    distributor = AdaptiveScoreDistribution()
    result = distributor.distribute_scores(raw_scores)
    return distributor.get_distribution_summary(result)
