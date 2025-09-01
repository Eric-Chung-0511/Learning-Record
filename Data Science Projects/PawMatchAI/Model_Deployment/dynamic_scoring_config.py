from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import json
import os

@dataclass
class DimensionConfig:
    """維度配置"""
    name: str
    base_weight: float
    priority_multiplier: Dict[str, float]
    compatibility_matrix: Dict[str, Dict[str, float]]
    threshold_values: Dict[str, float]
    description: str


@dataclass
class ConstraintConfig:
    """約束配置"""
    name: str
    condition_keywords: List[str]
    elimination_threshold: float
    penalty_factors: Dict[str, float]
    exemption_conditions: List[str]
    description: str


@dataclass
class ScoringProfile:
    """評分配置檔"""
    profile_name: str
    dimensions: List[DimensionConfig]
    constraints: List[ConstraintConfig]
    normalization_method: str
    bias_correction_rules: Dict[str, Any]
    ui_preferences: Dict[str, Any]


class DynamicScoringConfig:
    """動態評分配置管理器"""

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or self._get_default_config_path()
        self.current_profile = self._load_default_profile()
        self.custom_profiles = {}

    def _get_default_config_path(self) -> str:
        """獲取默認配置路徑"""
        return os.path.join(os.path.dirname(__file__), 'scoring_configs')

    def _load_default_profile(self) -> ScoringProfile:
        """載入預設評分配置"""
        # 空間相容性維度配置
        space_dimension = DimensionConfig(
            name="space_compatibility",
            base_weight=0.30,
            priority_multiplier={
                "apartment_living": 1.5,
                "first_time_owner": 1.2,
                "limited_space": 1.4
            },
            compatibility_matrix={
                "apartment": {
                    "toy": 0.95, "small": 0.90, "medium": 0.50,
                    "large": 0.15, "giant": 0.05
                },
                "house_small": {
                    "toy": 0.85, "small": 0.90, "medium": 0.85,
                    "large": 0.60, "giant": 0.30
                },
                "house_medium": {
                    "toy": 0.80, "small": 0.85, "medium": 0.95,
                    "large": 0.85, "giant": 0.60
                },
                "house_large": {
                    "toy": 0.75, "small": 0.80, "medium": 0.90,
                    "large": 0.95, "giant": 0.95
                }
            },
            threshold_values={
                "elimination_threshold": 0.20,
                "warning_threshold": 0.40,
                "good_threshold": 0.70
            },
            description="Evaluates breed size compatibility with living space"
        )

        # 運動相容性維度配置
        exercise_dimension = DimensionConfig(
            name="exercise_compatibility",
            base_weight=0.25,
            priority_multiplier={
                "low_activity": 1.6,
                "high_activity": 1.3,
                "time_limited": 1.4
            },
            compatibility_matrix={
                "low_user": {
                    "low": 1.0, "moderate": 0.70, "high": 0.30, "very_high": 0.10
                },
                "moderate_user": {
                    "low": 0.80, "moderate": 1.0, "high": 0.80, "very_high": 0.50
                },
                "high_user": {
                    "low": 0.60, "moderate": 0.85, "high": 1.0, "very_high": 0.95
                }
            },
            threshold_values={
                "severe_mismatch": 0.25,
                "moderate_mismatch": 0.50,
                "good_match": 0.75
            },
            description="Matches user activity level with breed exercise needs"
        )

        # 噪音相容性維度配置
        noise_dimension = DimensionConfig(
            name="noise_compatibility",
            base_weight=0.15,
            priority_multiplier={
                "apartment_living": 1.8,
                "noise_sensitive": 2.0,
                "quiet_preference": 1.5
            },
            compatibility_matrix={
                "low_tolerance": {
                    "quiet": 1.0, "moderate": 0.60, "high": 0.20, "very_high": 0.05
                },
                "moderate_tolerance": {
                    "quiet": 0.90, "moderate": 1.0, "high": 0.70, "very_high": 0.40
                },
                "high_tolerance": {
                    "quiet": 0.80, "moderate": 0.90, "high": 1.0, "very_high": 0.85
                }
            },
            threshold_values={
                "unacceptable": 0.15,
                "concerning": 0.40,
                "acceptable": 0.70
            },
            description="Matches breed noise levels with user tolerance"
        )

        # 約束配置
        apartment_constraint = ConstraintConfig(
            name="apartment_size_constraint",
            condition_keywords=["apartment", "small space", "studio", "condo"],
            elimination_threshold=0.15,
            penalty_factors={
                "large_breed": 0.70,
                "giant_breed": 0.85,
                "high_exercise": 0.60
            },
            exemption_conditions=["experienced_owner", "large_apartment"],
            description="Eliminates breeds unsuitable for apartment living"
        )

        exercise_constraint = ConstraintConfig(
            name="exercise_mismatch_constraint",
            condition_keywords=["don't exercise", "low activity", "minimal exercise"],
            elimination_threshold=0.20,
            penalty_factors={
                "very_high_exercise": 0.80,
                "working_breed": 0.60,
                "high_energy": 0.70
            },
            exemption_conditions=["dog_park_access", "active_family"],
            description="Prevents high-energy breeds for low-activity users"
        )

        # 偏見修正規則
        bias_correction_rules = {
            "size_bias": {
                "enabled": True,
                "detection_threshold": 0.70,  # 70%以上大型犬觸發修正
                "correction_strength": 0.60,  # 修正強度
                "target_distribution": {
                    "toy": 0.10, "small": 0.25, "medium": 0.40,
                    "large": 0.20, "giant": 0.05
                }
            },
            "popularity_bias": {
                "enabled": True,
                "common_breeds_penalty": 0.05,
                "rare_breeds_bonus": 0.03
            }
        }

        # UI偏好設定
        ui_preferences = {
            "ranking_style": "gradient_badges",
            "score_display": "percentage_with_bars",
            "color_scheme": {
                "excellent": "#22C55E",
                "good": "#F59E0B",
                "moderate": "#6B7280",
                "poor": "#EF4444"
            },
            "animation_enabled": True,
            "detailed_breakdown": True
        }

        return ScoringProfile(
            profile_name="comprehensive_default",
            dimensions=[space_dimension, exercise_dimension, noise_dimension],
            constraints=[apartment_constraint, exercise_constraint],
            normalization_method="sigmoid_compression",
            bias_correction_rules=bias_correction_rules,
            ui_preferences=ui_preferences
        )

    def get_dimension_config(self, dimension_name: str) -> Optional[DimensionConfig]:
        """獲取維度配置"""
        for dim in self.current_profile.dimensions:
            if dim.name == dimension_name:
                return dim
        return None

    def get_constraint_config(self, constraint_name: str) -> Optional[ConstraintConfig]:
        """獲取約束配置"""
        for constraint in self.current_profile.constraints:
            if constraint.name == constraint_name:
                return constraint
        return None

    def calculate_dynamic_weights(self, user_context: Dict[str, Any]) -> Dict[str, float]:
        """根據用戶情境動態計算權重"""
        weights = {}
        total_weight = 0

        for dimension in self.current_profile.dimensions:
            base_weight = dimension.base_weight

            # 根據用戶情境調整權重
            for context_key, multiplier in dimension.priority_multiplier.items():
                if user_context.get(context_key, False):
                    base_weight *= multiplier

            weights[dimension.name] = base_weight
            total_weight += base_weight

        # 正規化權重
        return {k: v / total_weight for k, v in weights.items()}

    def get_compatibility_score(self, dimension_name: str,
                              user_category: str, breed_category: str) -> float:
        """獲取相容性分數"""
        dimension_config = self.get_dimension_config(dimension_name)
        if not dimension_config:
            return 0.5

        matrix = dimension_config.compatibility_matrix
        if user_category in matrix and breed_category in matrix[user_category]:
            return matrix[user_category][breed_category]

        return 0.5  # 預設值

    def should_eliminate_breed(self, constraint_name: str,
                              breed_info: Dict[str, Any],
                              user_input: str) -> tuple[bool, str]:
        """判斷是否應該淘汰品種"""
        constraint_config = self.get_constraint_config(constraint_name)
        if not constraint_config:
            return False, ""

        # 檢查觸發條件
        user_input_lower = user_input.lower()
        triggered = any(keyword in user_input_lower
                       for keyword in constraint_config.condition_keywords)

        if not triggered:
            return False, ""

        # 檢查豁免條件
        exempted = any(condition in user_input_lower
                      for condition in constraint_config.exemption_conditions)

        if exempted:
            return False, "Exempted due to special conditions"

        # 應用淘汰邏輯（具體實現取決於約束類型）
        return self._apply_elimination_logic(constraint_config, breed_info, user_input)

    def _apply_elimination_logic(self, constraint_config: ConstraintConfig,
                               breed_info: Dict[str, Any], user_input: str) -> tuple[bool, str]:
        """應用淘汰邏輯"""
        # 根據約束名稱決定具體邏輯
        if constraint_config.name == "apartment_size_constraint":
            breed_size = breed_info.get('Size', '').lower()
            if any(size in breed_size for size in ['large', 'giant']):
                return True, f"Breed size ({breed_size}) unsuitable for apartment"

        elif constraint_config.name == "exercise_mismatch_constraint":
            exercise_needs = breed_info.get('Exercise Needs', '').lower()
            if any(level in exercise_needs for level in ['very high', 'extreme']):
                return True, f"Exercise needs ({exercise_needs}) exceed user capacity"

        return False, ""

    def get_bias_correction_settings(self) -> Dict[str, Any]:
        """獲取偏見修正設定"""
        return self.current_profile.bias_correction_rules

    def get_ui_preferences(self) -> Dict[str, Any]:
        """獲取UI偏好設定"""
        return self.current_profile.ui_preferences

    def save_custom_profile(self, profile: ScoringProfile, filename: str):
        """保存自定義配置檔"""
        if not os.path.exists(self.config_path):
            os.makedirs(self.config_path)

        filepath = os.path.join(self.config_path, f"{filename}.json")

        # 將配置檔案轉換為JSON格式
        profile_dict = {
            "profile_name": profile.profile_name,
            "dimensions": [self._dimension_to_dict(dim) for dim in profile.dimensions],
            "constraints": [self._constraint_to_dict(cons) for cons in profile.constraints],
            "normalization_method": profile.normalization_method,
            "bias_correction_rules": profile.bias_correction_rules,
            "ui_preferences": profile.ui_preferences
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(profile_dict, f, indent=2, ensure_ascii=False)

    def load_custom_profile(self, filename: str) -> Optional[ScoringProfile]:
        """載入自定義配置檔"""
        filepath = os.path.join(self.config_path, f"{filename}.json")

        if not os.path.exists(filepath):
            return None

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                profile_dict = json.load(f)

            return self._dict_to_profile(profile_dict)
        except Exception as e:
            print(f"Error loading profile {filename}: {str(e)}")
            return None

    def _dimension_to_dict(self, dimension: DimensionConfig) -> Dict[str, Any]:
        """將維度配置轉換為字典"""
        return {
            "name": dimension.name,
            "base_weight": dimension.base_weight,
            "priority_multiplier": dimension.priority_multiplier,
            "compatibility_matrix": dimension.compatibility_matrix,
            "threshold_values": dimension.threshold_values,
            "description": dimension.description
        }

    def _constraint_to_dict(self, constraint: ConstraintConfig) -> Dict[str, Any]:
        """將約束配置轉換為字典"""
        return {
            "name": constraint.name,
            "condition_keywords": constraint.condition_keywords,
            "elimination_threshold": constraint.elimination_threshold,
            "penalty_factors": constraint.penalty_factors,
            "exemption_conditions": constraint.exemption_conditions,
            "description": constraint.description
        }

    def _dict_to_profile(self, profile_dict: Dict[str, Any]) -> ScoringProfile:
        """將字典轉換為評分配置檔"""
        dimensions = [self._dict_to_dimension(dim) for dim in profile_dict["dimensions"]]
        constraints = [self._dict_to_constraint(cons) for cons in profile_dict["constraints"]]

        return ScoringProfile(
            profile_name=profile_dict["profile_name"],
            dimensions=dimensions,
            constraints=constraints,
            normalization_method=profile_dict["normalization_method"],
            bias_correction_rules=profile_dict["bias_correction_rules"],
            ui_preferences=profile_dict["ui_preferences"]
        )

    def _dict_to_dimension(self, dim_dict: Dict[str, Any]) -> DimensionConfig:
        """將字典轉換為維度配置"""
        return DimensionConfig(
            name=dim_dict["name"],
            base_weight=dim_dict["base_weight"],
            priority_multiplier=dim_dict["priority_multiplier"],
            compatibility_matrix=dim_dict["compatibility_matrix"],
            threshold_values=dim_dict["threshold_values"],
            description=dim_dict["description"]
        )

    def _dict_to_constraint(self, cons_dict: Dict[str, Any]) -> ConstraintConfig:
        """將字典轉換為約束配置"""
        return ConstraintConfig(
            name=cons_dict["name"],
            condition_keywords=cons_dict["condition_keywords"],
            elimination_threshold=cons_dict["elimination_threshold"],
            penalty_factors=cons_dict["penalty_factors"],
            exemption_conditions=cons_dict["exemption_conditions"],
            description=cons_dict["description"]
        )

def get_scoring_config() -> DynamicScoringConfig:
    """獲取全局評分配置"""
    return scoring_config


def update_scoring_config(new_config: DynamicScoringConfig):
    """更新全局評分配置"""
    global scoring_config
    scoring_config = new_config
