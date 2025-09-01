import numpy as np
from typing import List, Dict, Tuple, Any, Optional
from dataclasses import dataclass, field
import traceback
from scipy import stats

@dataclass
class CalibrationResult:
    """校準結果結構"""
    original_scores: List[float]
    calibrated_scores: List[float]
    score_mapping: Dict[str, float]  # breed -> calibrated_score
    calibration_method: str
    distribution_stats: Dict[str, float]
    quality_metrics: Dict[str, float] = field(default_factory=dict)

@dataclass
class ScoreDistribution:
    """分數分布統計"""
    mean: float
    std: float
    min_score: float
    max_score: float
    percentile_5: float
    percentile_95: float
    compression_ratio: float  # 分數壓縮比率
    effective_range: float    # 有效分數範圍

class ScoreCalibrator:
    """
    動態分數校準系統
    解決分數壓縮問題並保持相對排名
    """

    def __init__(self):
        """初始化校準器"""
        self.calibration_methods = {
            'dynamic_range_mapping': self._dynamic_range_mapping,
            'percentile_stretching': self._percentile_stretching,
            'gaussian_normalization': self._gaussian_normalization,
            'sigmoid_transformation': self._sigmoid_transformation
        }
        self.quality_thresholds = {
            'min_effective_range': 0.3,  # 最小有效分數範圍
            'max_compression_ratio': 0.2,  # 最大允許壓縮比率
            'target_distribution_range': (0.45, 0.95)  # 目標分布範圍
        }

    def calibrate_scores(self, breed_scores: List[Tuple[str, float]],
                        method: str = 'auto') -> CalibrationResult:
        """
        校準品種分數

        Args:
            breed_scores: (breed_name, score) 元組列表
            method: 校準方法 ('auto', 'dynamic_range_mapping', 'percentile_stretching', etc.)

        Returns:
            CalibrationResult: 校準結果
        """
        try:
            if not breed_scores:
                return CalibrationResult(
                    original_scores=[],
                    calibrated_scores=[],
                    score_mapping={},
                    calibration_method='none',
                    distribution_stats={}
                )

            # 提取分數和品種名稱
            breeds = [item[0] for item in breed_scores]
            original_scores = [item[1] for item in breed_scores]

            # 分析原始分數分布
            distribution = self._analyze_score_distribution(original_scores)

            # 選擇校準方法
            if method == 'auto':
                method = self._select_calibration_method(distribution)

            # 應用校準
            calibration_func = self.calibration_methods.get(method, self._dynamic_range_mapping)
            calibrated_scores = calibration_func(original_scores, distribution)

            # 保持排名一致性
            calibrated_scores = self._preserve_ranking(original_scores, calibrated_scores)

            # 建立分數映射
            score_mapping = dict(zip(breeds, calibrated_scores))

            # 計算品質指標
            quality_metrics = self._calculate_quality_metrics(
                original_scores, calibrated_scores, distribution
            )

            return CalibrationResult(
                original_scores=original_scores,
                calibrated_scores=calibrated_scores,
                score_mapping=score_mapping,
                calibration_method=method,
                distribution_stats=self._distribution_to_dict(distribution),
                quality_metrics=quality_metrics
            )

        except Exception as e:
            print(f"Error calibrating scores: {str(e)}")
            print(traceback.format_exc())
            # 回傳原始分數作為降級方案
            breeds = [item[0] for item in breed_scores]
            original_scores = [item[1] for item in breed_scores]
            return CalibrationResult(
                original_scores=original_scores,
                calibrated_scores=original_scores,
                score_mapping=dict(zip(breeds, original_scores)),
                calibration_method='fallback',
                distribution_stats={}
            )

    def _analyze_score_distribution(self, scores: List[float]) -> ScoreDistribution:
        """分析分數分布"""
        try:
            scores_array = np.array(scores)

            # 基本統計
            mean_score = np.mean(scores_array)
            std_score = np.std(scores_array)
            min_score = np.min(scores_array)
            max_score = np.max(scores_array)

            # 百分位數
            percentile_5 = np.percentile(scores_array, 5)
            percentile_95 = np.percentile(scores_array, 95)

            # 壓縮比率和有效範圍
            full_range = max_score - min_score
            effective_range = percentile_95 - percentile_5
            compression_ratio = 1.0 - (effective_range / 1.0) if full_range > 0 else 0.0

            return ScoreDistribution(
                mean=mean_score,
                std=std_score,
                min_score=min_score,
                max_score=max_score,
                percentile_5=percentile_5,
                percentile_95=percentile_95,
                compression_ratio=compression_ratio,
                effective_range=effective_range
            )

        except Exception as e:
            print(f"Error analyzing score distribution: {str(e)}")
            # 返回預設分布
            return ScoreDistribution(
                mean=0.5, std=0.1, min_score=0.0, max_score=1.0,
                percentile_5=0.4, percentile_95=0.6,
                compression_ratio=0.6, effective_range=0.2
            )

    def _select_calibration_method(self, distribution: ScoreDistribution) -> str:
        """根據分布特性選擇校準方法"""
        # 高度壓縮的分數需要強力展開
        if distribution.compression_ratio > 0.8:
            return 'percentile_stretching'

        # 中等壓縮使用動態範圍映射
        elif distribution.compression_ratio > 0.5:
            return 'dynamic_range_mapping'

        # 分數集中在中間使用 sigmoid 轉換
        elif 0.4 <= distribution.mean <= 0.6 and distribution.std < 0.1:
            return 'sigmoid_transformation'

        # 其他情況使用高斯正規化
        else:
            return 'gaussian_normalization'

    def _dynamic_range_mapping(self, scores: List[float],
                             distribution: ScoreDistribution) -> List[float]:
        """動態範圍映射校準"""
        try:
            scores_array = np.array(scores)

            # 使用5%和95%百分位數作為邊界
            lower_bound = distribution.percentile_5
            upper_bound = distribution.percentile_95

            # 避免除零
            if upper_bound - lower_bound < 0.001:
                upper_bound = distribution.max_score
                lower_bound = distribution.min_score
                if upper_bound - lower_bound < 0.001:
                    return scores  # 所有分數相同，無需校準

            # 映射到目標範圍 [0.45, 0.95]
            target_min, target_max = self.quality_thresholds['target_distribution_range']

            # 線性映射
            normalized = (scores_array - lower_bound) / (upper_bound - lower_bound)
            normalized = np.clip(normalized, 0, 1)  # 限制在 [0,1] 範圍
            calibrated = target_min + normalized * (target_max - target_min)

            return calibrated.tolist()

        except Exception as e:
            print(f"Error in dynamic range mapping: {str(e)}")
            return scores

    def _percentile_stretching(self, scores: List[float],
                             distribution: ScoreDistribution) -> List[float]:
        """百分位數拉伸校準"""
        try:
            scores_array = np.array(scores)

            # 計算百分位數排名
            percentile_ranks = stats.rankdata(scores_array, method='average') / len(scores_array)

            # 使用平方根轉換來增強差異
            stretched_ranks = np.sqrt(percentile_ranks)

            # 映射到目標範圍
            target_min, target_max = self.quality_thresholds['target_distribution_range']
            calibrated = target_min + stretched_ranks * (target_max - target_min)

            return calibrated.tolist()

        except Exception as e:
            print(f"Error in percentile stretching: {str(e)}")
            return self._dynamic_range_mapping(scores, distribution)

    def _gaussian_normalization(self, scores: List[float],
                              distribution: ScoreDistribution) -> List[float]:
        """高斯正規化校準"""
        try:
            scores_array = np.array(scores)

            # Z-score 正規化
            if distribution.std > 0:
                z_scores = (scores_array - distribution.mean) / distribution.std
                # 限制 Z-scores 在合理範圍內
                z_scores = np.clip(z_scores, -3, 3)
            else:
                z_scores = np.zeros_like(scores_array)

            # 轉換到目標範圍
            target_min, target_max = self.quality_thresholds['target_distribution_range']
            target_mean = (target_min + target_max) / 2
            target_std = (target_max - target_min) / 6  # 3-sigma 範圍

            calibrated = target_mean + z_scores * target_std
            calibrated = np.clip(calibrated, target_min, target_max)

            return calibrated.tolist()

        except Exception as e:
            print(f"Error in gaussian normalization: {str(e)}")
            return self._dynamic_range_mapping(scores, distribution)

    def _sigmoid_transformation(self, scores: List[float],
                              distribution: ScoreDistribution) -> List[float]:
        """Sigmoid 轉換校準"""
        try:
            scores_array = np.array(scores)

            # 中心化分數
            centered = scores_array - distribution.mean

            # Sigmoid 轉換 (增強中等分數的差異)
            sigmoid_factor = 10.0  # 控制 sigmoid 的陡峭程度
            transformed = 1 / (1 + np.exp(-sigmoid_factor * centered))

            # 映射到目標範圍
            target_min, target_max = self.quality_thresholds['target_distribution_range']
            calibrated = target_min + transformed * (target_max - target_min)

            return calibrated.tolist()

        except Exception as e:
            print(f"Error in sigmoid transformation: {str(e)}")
            return self._dynamic_range_mapping(scores, distribution)

    def _preserve_ranking(self, original_scores: List[float],
                         calibrated_scores: List[float]) -> List[float]:
        """確保校準後的分數保持原始排名"""
        try:
            # 獲取原始排名
            original_ranks = stats.rankdata([-score for score in original_scores], method='ordinal')

            # 獲取校準後的排名
            calibrated_with_ranks = list(zip(calibrated_scores, original_ranks))

            # 按原始排名排序校準後的分數
            calibrated_with_ranks.sort(key=lambda x: x[1])

            # 重新分配分數以保持排名但使用校準後的分布
            sorted_calibrated = sorted(calibrated_scores, reverse=True)

            # 建立新的分數列表
            preserved_scores = [0.0] * len(original_scores)
            for i, (_, original_rank) in enumerate(calibrated_with_ranks):
                # 找到原始位置
                original_index = original_ranks.tolist().index(original_rank)
                preserved_scores[original_index] = sorted_calibrated[i]

            return preserved_scores

        except Exception as e:
            print(f"Error preserving ranking: {str(e)}")
            return calibrated_scores

    def _calculate_quality_metrics(self, original_scores: List[float],
                                 calibrated_scores: List[float],
                                 distribution: ScoreDistribution) -> Dict[str, float]:
        """計算校準品質指標"""
        try:
            original_array = np.array(original_scores)
            calibrated_array = np.array(calibrated_scores)

            # 範圍改善
            original_range = np.max(original_array) - np.min(original_array)
            calibrated_range = np.max(calibrated_array) - np.min(calibrated_array)
            range_improvement = calibrated_range / max(0.001, original_range)

            # 分離度改善 (相鄰分數間的平均差異)
            original_sorted = np.sort(original_array)
            calibrated_sorted = np.sort(calibrated_array)

            original_separation = np.mean(np.diff(original_sorted)) if len(original_sorted) > 1 else 0
            calibrated_separation = np.mean(np.diff(calibrated_sorted)) if len(calibrated_sorted) > 1 else 0

            separation_improvement = (calibrated_separation / max(0.001, original_separation)
                                    if original_separation > 0 else 1.0)

            # 排名保持度 (Spearman 相關係數)
            if len(original_scores) > 1:
                rank_correlation, _ = stats.spearmanr(original_scores, calibrated_scores)
                rank_correlation = abs(rank_correlation) if not np.isnan(rank_correlation) else 1.0
            else:
                rank_correlation = 1.0

            # 分布品質
            calibrated_std = np.std(calibrated_array)
            distribution_quality = min(1.0, calibrated_std * 2)  # 標準差越大品質越好（在合理範圍內）

            return {
                'range_improvement': range_improvement,
                'separation_improvement': separation_improvement,
                'rank_preservation': rank_correlation,
                'distribution_quality': distribution_quality,
                'effective_range_achieved': calibrated_range,
                'compression_reduction': max(0, distribution.compression_ratio -
                                           (1.0 - calibrated_range))
            }

        except Exception as e:
            print(f"Error calculating quality metrics: {str(e)}")
            return {'error': str(e)}

    def _distribution_to_dict(self, distribution: ScoreDistribution) -> Dict[str, float]:
        """將分布統計轉換為字典"""
        return {
            'mean': distribution.mean,
            'std': distribution.std,
            'min_score': distribution.min_score,
            'max_score': distribution.max_score,
            'percentile_5': distribution.percentile_5,
            'percentile_95': distribution.percentile_95,
            'compression_ratio': distribution.compression_ratio,
            'effective_range': distribution.effective_range
        }

    def apply_tie_breaking(self, breed_scores: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        """應用確定性的打破平手機制"""
        try:
            # 按分數分組
            score_groups = {}
            for breed, score in breed_scores:
                rounded_score = round(score, 6)  # 避免浮點數精度問題
                if rounded_score not in score_groups:
                    score_groups[rounded_score] = []
                score_groups[rounded_score].append((breed, score))

            # 處理每個分數組
            result = []
            for rounded_score in sorted(score_groups.keys(), reverse=True):
                group = score_groups[rounded_score]

                if len(group) == 1:
                    result.extend(group)
                else:
                    # 按品種名稱字母順序打破平手
                    sorted_group = sorted(group, key=lambda x: x[0])

                    # 為平手的品種分配微小的分數差異
                    for i, (breed, original_score) in enumerate(sorted_group):
                        adjusted_score = original_score - (i * 0.0001)
                        result.append((breed, adjusted_score))

            return result

        except Exception as e:
            print(f"Error in tie breaking: {str(e)}")
            return breed_scores

    def get_calibration_summary(self, result: CalibrationResult) -> Dict[str, Any]:
        """獲取校準摘要資訊"""
        try:
            summary = {
                'method_used': result.calibration_method,
                'breeds_processed': len(result.original_scores),
                'score_range_before': {
                    'min': min(result.original_scores) if result.original_scores else 0,
                    'max': max(result.original_scores) if result.original_scores else 0,
                    'range': (max(result.original_scores) - min(result.original_scores))
                             if result.original_scores else 0
                },
                'score_range_after': {
                    'min': min(result.calibrated_scores) if result.calibrated_scores else 0,
                    'max': max(result.calibrated_scores) if result.calibrated_scores else 0,
                    'range': (max(result.calibrated_scores) - min(result.calibrated_scores))
                             if result.calibrated_scores else 0
                },
                'distribution_stats': result.distribution_stats,
                'quality_metrics': result.quality_metrics,
                'improvement_summary': {
                    'range_expanded': result.quality_metrics.get('range_improvement', 1.0) > 1.1,
                    'separation_improved': result.quality_metrics.get('separation_improvement', 1.0) > 1.1,
                    'ranking_preserved': result.quality_metrics.get('rank_preservation', 1.0) > 0.95
                }
            }

            return summary

        except Exception as e:
            print(f"Error generating calibration summary: {str(e)}")
            return {'error': str(e)}

def calibrate_breed_scores(breed_scores: List[Tuple[str, float]],
                          method: str = 'auto') -> CalibrationResult:
    """
    便利函數：校準品種分數

    Args:
        breed_scores: (breed_name, score) 元組列表
        method: 校準方法

    Returns:
        CalibrationResult: 校準結果
    """
    calibrator = ScoreCalibrator()
    return calibrator.calibrate_scores(breed_scores, method)

def get_calibrated_rankings(breed_scores: List[Tuple[str, float]],
                           method: str = 'auto') -> List[Tuple[str, float, int]]:
    """
    便利函數：獲取校準後的排名

    Args:
        breed_scores: (breed_name, score) 元組列表
        method: 校準方法

    Returns:
        List[Tuple[str, float, int]]: (breed_name, calibrated_score, rank) 列表
    """
    calibrator = ScoreCalibrator()
    result = calibrator.calibrate_scores(breed_scores, method)

    # 打破平手機制
    calibrated_with_breed = [(breed, result.score_mapping[breed]) for breed in result.score_mapping]
    calibrated_with_tie_breaking = calibrator.apply_tie_breaking(calibrated_with_breed)

    # 添加排名
    ranked_results = []
    for rank, (breed, score) in enumerate(calibrated_with_tie_breaking, 1):
        ranked_results.append((breed, score, rank))

    return ranked_results
