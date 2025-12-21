# %%writefile dynamic_weight_calculator.py
import numpy as np
from typing import Dict, List, Tuple, Set, Optional, Any
from dataclasses import dataclass, field
import traceback


@dataclass
class WeightAllocationResult:
    """權重分配結果"""
    dynamic_weights: Dict[str, float] = field(default_factory=dict)
    allocation_method: str = 'balanced'
    high_priority_count: int = 0
    mentioned_dimensions: Set[str] = field(default_factory=set)
    weight_sum: float = 1.0
    allocation_notes: List[str] = field(default_factory=list)


class DynamicWeightCalculator:
    """
    動態權重計算器
    根據使用者優先級動態調整維度權重

    策略:
    - 1個高優先級 → 固定預留 40%
    - 2個高優先級 → 固定預留 40% + 25%
    - 3個高優先級 → 固定預留 30% + 27% + 23%
    - 4+個高優先級 → 倍數正規化法
    """

    def __init__(self):
        """初始化動態權重計算器"""
        self.default_weights = self._initialize_default_weights()
        self.dimension_name_mapping = self._initialize_dimension_mapping()
        self.high_priority_threshold = 1.4  # Balanced threshold (not too high, not too low)
        self.min_weight_floor = 0.05
        self.contextual_weight_distribution = {
            'critical_dimensions_weight': 0.50,  # Moderate emphasis on critical dimensions
            'mentioned_dimensions_weight': 0.35,  # Good weight for mentioned dimensions
            'other_dimensions_weight': 0.15      # Reasonable baseline for other dimensions
        }

    def _initialize_default_weights(self) -> Dict[str, float]:
        """初始化預設權重（平衡配置）"""
        return {
            'activity_compatibility': 0.18,
            'noise_compatibility': 0.16,
            'spatial_compatibility': 0.13,
            'family_compatibility': 0.13,
            'maintenance_compatibility': 0.13,
            'experience_compatibility': 0.15,  # 新增獨立的experience維度
            'health_compatibility': 0.12
        }

    def _initialize_dimension_mapping(self) -> Dict[str, str]:
        """初始化維度名稱映射"""
        return {
            'noise': 'noise_compatibility',
            'size': 'spatial_compatibility',  # size更適合映射到spatial
            'exercise': 'activity_compatibility',
            'activity': 'activity_compatibility',
            'grooming': 'maintenance_compatibility',
            'maintenance': 'maintenance_compatibility',
            'family': 'family_compatibility',
            'experience': 'experience_compatibility',  # 獨立映射
            'health': 'health_compatibility',  # 獨立映射
            'spatial': 'spatial_compatibility',
            'space': 'spatial_compatibility'
        }

    def calculate_dynamic_weights(self,
                                 dimension_priorities: Dict[str, float],
                                 user_mentions: Optional[Set[str]] = None,
                                 use_contextual: bool = True) -> WeightAllocationResult:
        """
        計算動態權重

        Args:
            dimension_priorities: 維度優先級 {dimension: priority_score}
            user_mentions: 使用者明確提到的維度
            use_contextual: 是否使用情境相對評分（關鍵維度80%）

        Returns:
            WeightAllocationResult: 權重分配結果
        """
        try:
            if user_mentions is None:
                user_mentions = set()

            # Step 1: 標準化維度名稱
            normalized_priorities = self._normalize_dimension_names(dimension_priorities)

            # Step 2: 分類維度
            high_priority_dims = {
                dim: score for dim, score in normalized_priorities.items()
                if score >= self.high_priority_threshold
            }
            high_count = len(high_priority_dims)

            # Step 3: 根據高優先級數量選擇策略
            if high_count == 0:
                # 無優先級 → 使用預設權重
                result = self._allocate_default_weights(user_mentions)
                result.allocation_method = 'default_balanced'

            elif use_contextual:
                # 使用情境相對評分（關鍵維度80%）
                result = self._allocate_contextual_weights(
                    normalized_priorities, user_mentions
                )
                result.allocation_method = 'contextual_relative'

            elif high_count == 1:
                # 單一高優先級 → 固定預留40%
                result = self._allocate_single_priority(
                    normalized_priorities, user_mentions
                )
                result.allocation_method = 'single_fixed'

            elif high_count <= 3:
                # 2-3個高優先級 → 階梯固定預留法
                result = self._allocate_multiple_priorities_fixed(
                    normalized_priorities, user_mentions, high_count
                )
                result.allocation_method = f'multiple_fixed_{high_count}'

            else:
                # 4+個高優先級 → 倍數正規化法
                result = self._allocate_multiple_priorities_proportional(
                    normalized_priorities, user_mentions
                )
                result.allocation_method = 'proportional'

            # Step 4: 應用最低權重保護
            result.dynamic_weights = self._apply_weight_floor(result.dynamic_weights)

            # Step 5: 正規化確保總和為1.0
            result.dynamic_weights = self._normalize_weights(result.dynamic_weights)
            result.weight_sum = sum(result.dynamic_weights.values())

            result.high_priority_count = high_count
            result.mentioned_dimensions = user_mentions

            return result

        except Exception as e:
            print(f"Error calculating dynamic weights: {str(e)}")
            print(traceback.format_exc())
            return WeightAllocationResult(
                dynamic_weights=self.default_weights.copy(),
                allocation_method='fallback'
            )

    def _normalize_dimension_names(self,
                                   priorities: Dict[str, float]) -> Dict[str, float]:
        """標準化維度名稱"""
        normalized = {}
        for dim, score in priorities.items():
            mapped_dim = self.dimension_name_mapping.get(dim, dim)
            # 如果mapped_dim不在default_weights中，保留原維度名
            if mapped_dim not in self.default_weights:
                mapped_dim = dim
            normalized[mapped_dim] = max(normalized.get(mapped_dim, 1.0), score)
        return normalized

    def _allocate_default_weights(self,
                                 user_mentions: Set[str]) -> WeightAllocationResult:
        """分配預設平衡權重"""
        weights = self.default_weights.copy()
        notes = ["Using default balanced weights (no priorities detected)"]

        return WeightAllocationResult(
            dynamic_weights=weights,
            allocation_notes=notes
        )

    def _allocate_contextual_weights(self,
                                    priorities: Dict[str, float],
                                    user_mentions: Set[str]) -> WeightAllocationResult:
        """
        情境相對權重分配（關鍵維度50%）
        """
        weights = {}
        notes = []

        # 標準化user_mentions維度名稱
        normalized_mentions = set()
        for mention in user_mentions:
            normalized_name = self.dimension_name_mapping.get(mention, mention)
            if normalized_name in self.default_weights:
                normalized_mentions.add(normalized_name)

        # 分類維度
        critical_dims = [d for d, s in priorities.items() if s >= self.high_priority_threshold]
        mentioned_dims = [d for d in normalized_mentions if d not in critical_dims]
        other_dims = [d for d in self.default_weights.keys()
                     if d not in critical_dims and d not in mentioned_dims]

        # 權重分配
        total_critical = self.contextual_weight_distribution['critical_dimensions_weight']
        total_mentioned = self.contextual_weight_distribution['mentioned_dimensions_weight']
        total_other = self.contextual_weight_distribution['other_dimensions_weight']

        # 關鍵維度：按優先級比例分配50%
        if critical_dims:
            critical_priority_sum = sum(priorities.get(d, 1.0) for d in critical_dims)
            for dim in critical_dims:
                weight = (priorities.get(dim, 1.0) / critical_priority_sum) * total_critical
                weights[dim] = weight
            notes.append(f"Critical dimensions ({len(critical_dims)}): {total_critical:.0%} weight")

        # 提及維度：平均分配35%
        if mentioned_dims:
            for dim in mentioned_dims:
                weights[dim] = total_mentioned / len(mentioned_dims)
            notes.append(f"Mentioned dimensions ({len(mentioned_dims)}): {total_mentioned:.0%} weight")

        # 其他維度：平均分配15%
        if other_dims:
            for dim in other_dims:
                weights[dim] = total_other / len(other_dims)
            notes.append(f"Other dimensions ({len(other_dims)}): {total_other:.0%} weight")

        # 填充未覆蓋的維度
        for dim in self.default_weights.keys():
            if dim not in weights:
                weights[dim] = 0.05

        return WeightAllocationResult(
            dynamic_weights=weights,
            allocation_notes=notes
        )

    def _allocate_single_priority(self,
                                 priorities: Dict[str, float],
                                 user_mentions: Set[str]) -> WeightAllocationResult:
        """單一高優先級：固定預留40%"""
        weights = {}
        notes = []

        # 找到高優先級維度
        high_priority_dim = None
        max_priority = 0
        for dim, score in priorities.items():
            if score >= self.high_priority_threshold and score > max_priority:
                high_priority_dim = dim
                max_priority = score

        if high_priority_dim:
            # 高優先級維度：40%
            weights[high_priority_dim] = 0.40
            notes.append(f"{high_priority_dim}: 40% (high priority)")

            # 其他維度：平均分配剩餘60%
            other_dims = [d for d in self.default_weights.keys() if d != high_priority_dim]
            remaining_weight = 0.60
            for dim in other_dims:
                weights[dim] = remaining_weight / len(other_dims)
        else:
            weights = self.default_weights.copy()

        return WeightAllocationResult(
            dynamic_weights=weights,
            allocation_notes=notes
        )

    def _allocate_multiple_priorities_fixed(self,
                                          priorities: Dict[str, float],
                                          user_mentions: Set[str],
                                          high_count: int) -> WeightAllocationResult:
        """2-3個高優先級：階梯固定預留法"""
        weights = {}
        notes = []

        # 排序高優先級維度
        high_priority_dims = sorted(
            [(dim, score) for dim, score in priorities.items()
             if score >= self.high_priority_threshold],
            key=lambda x: x[1],
            reverse=True
        )

        # 根據數量分配固定權重
        if high_count == 2:
            # 2個高優先級：40% + 25%
            fixed_weights = [0.40, 0.25]
            remaining = 0.35
            notes.append("2 high priorities: 40% + 25%, others share 35%")
        elif high_count == 3:
            # 3個高優先級：30% + 27% + 23%
            fixed_weights = [0.30, 0.27, 0.23]
            remaining = 0.20
            notes.append("3 high priorities: 30% + 27% + 23%, others share 20%")
        else:
            # 降級處理
            return self._allocate_multiple_priorities_proportional(
                priorities, user_mentions
            )

        # 分配固定權重
        for i, (dim, score) in enumerate(high_priority_dims[:high_count]):
            weights[dim] = fixed_weights[i]

        # 其他維度：平均分配剩餘
        other_dims = [d for d in self.default_weights.keys()
                     if d not in [dim for dim, _ in high_priority_dims[:high_count]]]
        if other_dims:
            for dim in other_dims:
                weights[dim] = remaining / len(other_dims)

        return WeightAllocationResult(
            dynamic_weights=weights,
            allocation_notes=notes
        )

    def _allocate_multiple_priorities_proportional(self,
                                                  priorities: Dict[str, float],
                                                  user_mentions: Set[str]) -> WeightAllocationResult:
        """4+個高優先級：倍數正規化法"""
        weights = {}
        notes = []

        # 計算原始權重（基於優先級倍數）
        raw_weights = {}
        for dim in self.default_weights.keys():
            priority_score = priorities.get(dim, 1.0)
            raw_weights[dim] = 1.0 * priority_score

        # 正規化
        total_raw = sum(raw_weights.values())
        for dim, raw_weight in raw_weights.items():
            weights[dim] = raw_weight / total_raw

        notes.append(f"Proportional allocation for {len([d for d, s in priorities.items() if s >= self.high_priority_threshold])} high priorities")

        return WeightAllocationResult(
            dynamic_weights=weights,
            allocation_notes=notes
        )

    def _apply_weight_floor(self, weights: Dict[str, float]) -> Dict[str, float]:
        """應用最低權重保護"""
        protected_weights = {}
        for dim, weight in weights.items():
            protected_weights[dim] = max(self.min_weight_floor, weight)
        return protected_weights

    def _normalize_weights(self, weights: Dict[str, float]) -> Dict[str, float]:
        """正規化權重確保總和為1.0"""
        total = sum(weights.values())
        if total == 0:
            return self.default_weights.copy()

        normalized = {dim: weight / total for dim, weight in weights.items()}
        return normalized

    def get_weight_summary(self, result: WeightAllocationResult) -> Dict[str, Any]:
        """
        獲取權重分配摘要

        Args:
            result: 權重分配結果

        Returns:
            Dict[str, Any]: 權重摘要
        """
        return {
            'allocation_method': result.allocation_method,
            'high_priority_count': result.high_priority_count,
            'weight_sum': result.weight_sum,
            'weights': result.dynamic_weights,
            'top_3_dimensions': sorted(
                result.dynamic_weights.items(),
                key=lambda x: x[1],
                reverse=True
            )[:3],
            'allocation_notes': result.allocation_notes
        }


def calculate_weights_from_priorities(dimension_priorities: Dict[str, float],
                                     user_mentions: Optional[Set[str]] = None,
                                     use_contextual: bool = True) -> WeightAllocationResult:
    """
    便利函數: 從優先級計算權重

    Args:
        dimension_priorities: 維度優先級
        user_mentions: 使用者提及的維度
        use_contextual: 使用情境相對評分

    Returns:
        WeightAllocationResult: 權重分配結果
    """
    calculator = DynamicWeightCalculator()
    return calculator.calculate_dynamic_weights(
        dimension_priorities, user_mentions, use_contextual
    )


def get_weight_summary(dimension_priorities: Dict[str, float],
                      user_mentions: Optional[Set[str]] = None) -> Dict[str, Any]:
    """
    便利函數: 獲取權重摘要

    Args:
        dimension_priorities: 維度優先級
        user_mentions: 使用者提及的維度

    Returns:
        Dict[str, Any]: 權重摘要
    """
    calculator = DynamicWeightCalculator()
    result = calculator.calculate_dynamic_weights(dimension_priorities, user_mentions)
    return calculator.get_weight_summary(result)
