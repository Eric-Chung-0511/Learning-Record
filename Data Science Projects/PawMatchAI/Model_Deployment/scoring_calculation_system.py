from dataclasses import dataclass
from typing import Dict, List, Any, Optional
import math
import random
import numpy as np
import traceback
from breed_health_info import breed_health_info
from breed_noise_info import breed_noise_info
from dog_database import get_dog_description
from dimension_score_calculator import DimensionScoreCalculator
from score_integration_manager import ScoreIntegrationManager, UserPreferences
from bonus_penalty_engine import BonusPenaltyEngine

@dataclass
class DimensionalScore:
    """維度分數結構"""
    dimension_name: str
    raw_score: float      # 原始計算分數 (0.0-1.0)
    weight: float         # 維度權重 (0.0-1.0)
    display_score: float  # 顯示分數 (0.0-1.0)
    explanation: str      # 評分說明


@dataclass
class UnifiedBreedScore:
    """統一品種評分結果"""
    breed_name: str
    overall_score: float           # 總體分數 (0.0-1.0)
    dimensional_scores: List[DimensionalScore]  # 各維度分數
    bonus_factors: Dict[str, float]    # 加分因素
    penalty_factors: Dict[str, float]  # 扣分因素
    confidence_level: float        # 推薦信心度 (0.0-1.0)
    match_explanation: str         # 匹配說明
    warnings: List[str]           # 警告訊息


# 初始化計算器實例
_dimension_calculator = DimensionScoreCalculator()
_score_manager = ScoreIntegrationManager()
_bonus_engine = BonusPenaltyEngine()


def apply_size_filter(breed_score: float, user_preference: str, breed_size: str) -> float:
    """
    強過濾機制，基於用戶的體型偏好過濾品種

    Parameters:
        breed_score (float): 原始品種評分
        user_preference (str): 用戶偏好的體型
        breed_size (str): 品種的實際體型

    Returns:
        float: 過濾後的評分，如果體型不符合會返回 0
    """
    return _score_manager.apply_size_filter(breed_score, user_preference, breed_size)


@staticmethod
def calculate_breed_bonus(breed_info: dict, user_prefs: 'UserPreferences') -> float:
    """計算品種額外加分"""
    return BonusPenaltyEngine.calculate_breed_bonus(breed_info, user_prefs)


@staticmethod
def calculate_additional_factors(breed_info: dict, user_prefs: 'UserPreferences') -> dict:
    """
    計算額外的評估因素，結合品種特性與使用者需求的全面評估系統
    """
    return BonusPenaltyEngine.calculate_additional_factors(breed_info, user_prefs)


def calculate_compatibility_score(breed_info: dict, user_prefs: UserPreferences) -> dict:
    """計算品種與使用者條件的相容性分數"""
    try:
        print(f"Processing breed: {breed_info.get('Breed', 'Unknown')}")
        print(f"Breed info keys: {breed_info.keys()}")

        if 'Size' not in breed_info:
            print("Missing Size information")
            raise KeyError("Size information missing")

        if user_prefs.size_preference != "no_preference":
            if breed_info['Size'].lower() != user_prefs.size_preference.lower():
                return {
                    'space': 0,
                    'exercise': 0,
                    'grooming': 0,
                    'experience': 0,
                    'health': 0,
                    'noise': 0,
                    'overall': 0,
                    'adaptability_bonus': 0
                }

        # 計算所有基礎分數並整合到字典中
        scores = {
            'space': _dimension_calculator.calculate_space_score(
                breed_info['Size'],
                user_prefs.living_space,
                user_prefs.yard_access != 'no_yard',
                breed_info.get('Exercise Needs', 'Moderate')
            ),
            'exercise': _dimension_calculator.calculate_exercise_score(
                breed_info.get('Exercise Needs', 'Moderate'),
                user_prefs.exercise_time,
                user_prefs.exercise_type,
                breed_info['Size'],
                user_prefs.living_space,
                breed_info
            ),
            'grooming': _dimension_calculator.calculate_grooming_score(
                breed_info.get('Grooming Needs', 'Moderate'),
                user_prefs.grooming_commitment.lower(),
                breed_info['Size']
            ),
            'experience': _dimension_calculator.calculate_experience_score(
                breed_info.get('Care Level', 'Moderate'),
                user_prefs.experience_level,
                breed_info.get('Temperament', '')
            ),
            'health': _dimension_calculator.calculate_health_score(
                breed_info.get('Breed', ''),
                user_prefs.health_sensitivity
            ),
            'noise': _dimension_calculator.calculate_noise_score(
                breed_info.get('Breed', ''),
                user_prefs.noise_tolerance,
                user_prefs.living_space,
                user_prefs.has_children,
                user_prefs.children_age
            )
        }

        final_score = _score_manager.calculate_breed_compatibility_score(
            scores=scores,
            user_prefs=user_prefs,
            breed_info=breed_info
        )

        # 計算環境適應性加成
        adaptability_bonus = _score_manager.calculate_environmental_fit(breed_info, user_prefs)

        if (breed_info.get('Exercise Needs') == "Very High" and
            user_prefs.living_space == "apartment" and
            user_prefs.exercise_time < 90):
            final_score *= 0.85  # 高運動需求但條件不足的懲罰

        # 整合最終分數和加成
        combined_score = (final_score * 0.9) + (adaptability_bonus * 0.1)

        # 體型過濾
        filtered_score = apply_size_filter(
            breed_score=combined_score,
            user_preference=user_prefs.size_preference,
            breed_size=breed_info['Size']
        )

        final_score = _bonus_engine.amplify_score_extreme(filtered_score)

        # 更新並返回完整的評分結果
        scores.update({
            'overall': final_score,
            'size': breed_info['Size'],
            'adaptability_bonus': adaptability_bonus
        })

        return scores

    except Exception as e:
        print(f"\n!!!!! Critical Error Occurred !!!!!")
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Message: {str(e)}")
        print(f"Full Error Traceback:")
        print(traceback.format_exc())
        return {k: 0.6 for k in ['space', 'exercise', 'grooming', 'experience', 'health', 'noise', 'overall']}


def calculate_environmental_fit(breed_info: dict, user_prefs: UserPreferences) -> float:
    """計算品種與環境的適應性加成"""
    return _score_manager.calculate_environmental_fit(breed_info, user_prefs)


def calculate_breed_compatibility_score(scores: dict, user_prefs: UserPreferences, breed_info: dict) -> float:
    """計算品種相容性總分"""
    return _score_manager.calculate_breed_compatibility_score(scores, user_prefs, breed_info)


def amplify_score_extreme(score: float) -> float:
    """
    優化分數分布，提供更有意義的評分範圍。
    純粹進行數學轉換，不依賴外部資訊。

    Parameters:
        score: 原始評分（0-1之間的浮點數）

    Returns:
        float: 調整後的評分（0-1之間的浮點數）
    """
    return _bonus_engine.amplify_score_extreme(score)


class UnifiedScoringSystem:
    """統一評分系統核心類"""

    def __init__(self):
        """初始化評分系統"""
        self.dimension_weights = {
            'space_compatibility': 0.30,      # Increased from 0.25
            'exercise_compatibility': 0.25,   # Increased from 0.20
            'grooming_compatibility': 0.10,   # Reduced from 0.15
            'experience_compatibility': 0.10,  # Reduced from 0.15
            'noise_compatibility': 0.15,     # Adjusted
            'family_compatibility': 0.10      # Added
        }
        random.seed(42)  # 確保一致性

    def calculate_space_compatibility(self, breed_info: Dict, user_prefs: UserPreferences) -> DimensionalScore:
        """計算空間適配性分數"""
        breed_size = breed_info.get('Size', 'Medium').lower()
        living_space = user_prefs.living_space
        yard_access = user_prefs.yard_access

        # 基礎空間評分邏輯
        space_score = 0.5  # 基礎分數
        explanation_parts = []

        # Enhanced size-space matrix with stricter penalties
        size_space_matrix = {
            'apartment': {
                'toy': 0.95, 'small': 0.90, 'medium': 0.50,  # Reduced medium score
                'large': 0.15, 'giant': 0.05  # Severe penalties for large/giant
            },
            'house_small': {
                'toy': 0.85, 'small': 0.90, 'medium': 0.85,
                'large': 0.60, 'giant': 0.30  # Still penalize giant breeds
            },
            'house_medium': {  # Added for medium houses
                'toy': 0.80, 'small': 0.85, 'medium': 0.95,
                'large': 0.85, 'giant': 0.60  # Giants still not ideal
            },
            'house_large': {
                'toy': 0.75, 'small': 0.80, 'medium': 0.90,
                'large': 0.95, 'giant': 0.95
            }
        }

        # Determine actual living space category
        if 'apartment' in living_space or 'small' in living_space:
            space_category = 'apartment'
        elif 'medium' in living_space:
            space_category = 'house_medium'
        elif 'large' in living_space:
            space_category = 'house_large'
        else:
            space_category = 'house_small'

        # Get base score from matrix
        base_score = size_space_matrix[space_category].get(
            self._normalize_size(breed_size), 0.5
        )

        # Apply additional penalties for exercise needs in small spaces
        if space_category == 'apartment':
            exercise_needs = breed_info.get('Exercise Needs', '').lower()
            if 'high' in exercise_needs:
                base_score *= 0.7  # 30% additional penalty
            if 'very high' in exercise_needs:
                base_score *= 0.5  # 50% additional penalty

        space_score = base_score
        explanation_parts = []
        if base_score < 0.3:
            explanation_parts.append(f"Poor match: {breed_size} dog in {space_category}")
        elif base_score < 0.7:
            explanation_parts.append(f"Moderate match: {breed_size} dog in {space_category}")
        else:
            explanation_parts.append(f"Good match: {breed_size} dog in {space_category}")

        # 院子需求調整
        if yard_access == 'private_yard':
            space_score = min(1.0, space_score + 0.1)
            explanation_parts.append("Private yard bonus")
        elif yard_access == 'no_yard' and breed_size in ['large', 'giant']:
            space_score *= 0.7
            explanation_parts.append("Large dog without yard penalty")

        # 運動需求考量
        exercise_needs = breed_info.get('Exercise Needs', 'Moderate').lower()
        if exercise_needs in ['high', 'very high'] and living_space == 'apartment':
            space_score *= 0.8
            explanation_parts.append("High exercise needs in apartment limitation")

        explanation = "; ".join(explanation_parts)

        return DimensionalScore(
            dimension_name='space_compatibility',
            raw_score=space_score,
            weight=self.dimension_weights['space_compatibility'],
            display_score=space_score,
            explanation=explanation
        )

    def calculate_exercise_compatibility(self, breed_info: Dict, user_prefs: UserPreferences) -> DimensionalScore:
        """計算運動適配性分數"""
        breed_exercise_needs = breed_info.get('Exercise Needs', 'Moderate').lower()
        user_exercise_time = user_prefs.exercise_time
        user_exercise_type = user_prefs.exercise_type

        # 運動需求映射
        exercise_requirements = {
            'low': {'min_time': 20, 'ideal_time': 30},
            'moderate': {'min_time': 45, 'ideal_time': 60},
            'high': {'min_time': 90, 'ideal_time': 120},
            'very high': {'min_time': 120, 'ideal_time': 180}
        }

        breed_req = exercise_requirements.get(breed_exercise_needs, exercise_requirements['moderate'])

        # 基礎時間匹配度
        if user_exercise_time >= breed_req['ideal_time']:
            time_score = 1.0
            time_explanation = "Sufficient exercise time"
        elif user_exercise_time >= breed_req['min_time']:
            time_score = 0.7 + 0.3 * (user_exercise_time - breed_req['min_time']) / (breed_req['ideal_time'] - breed_req['min_time'])
            time_explanation = "Exercise time meets basic requirements"
        else:
            time_score = 0.3 * user_exercise_time / breed_req['min_time']
            time_explanation = "Insufficient exercise time"

        # Enhanced compatibility matrix
        breed_level = self._parse_exercise_level(breed_exercise_needs)
        user_level = self._get_user_exercise_level(user_exercise_time)

        compatibility_matrix = {
            ('low', 'low'): 1.0,
            ('low', 'moderate'): 0.85,
            ('low', 'high'): 0.40,  # Stronger penalty
            ('low', 'very high'): 0.15,  # Severe penalty
            ('moderate', 'low'): 0.70,
            ('moderate', 'moderate'): 1.0,
            ('moderate', 'high'): 0.85,
            ('moderate', 'very high'): 0.60,
            ('high', 'low'): 0.20,  # Severe penalty
            ('high', 'moderate'): 0.65,
            ('high', 'high'): 1.0,
            ('high', 'very high'): 0.90,
        }

        base_score = compatibility_matrix.get((user_level, breed_level), 0.5)

        # Check for exercise type compatibility
        if hasattr(user_prefs, 'exercise_type'):
            exercise_type_bonus = self._calculate_exercise_type_match(
                breed_info, user_prefs.exercise_type
            )
            base_score = base_score * 0.8 + exercise_type_bonus * 0.2

        exercise_score = base_score

        explanation = f"{user_level} user with {breed_level} exercise breed"

        return DimensionalScore(
            dimension_name='exercise_compatibility',
            raw_score=exercise_score,
            weight=self.dimension_weights['exercise_compatibility'],
            display_score=exercise_score,
            explanation=explanation
        )

    def _normalize_size(self, breed_size: str) -> str:
        """Normalize breed size string"""
        breed_size = breed_size.lower()
        if 'giant' in breed_size:
            return 'giant'
        elif 'large' in breed_size:
            return 'large'
        elif 'medium' in breed_size:
            return 'medium'
        elif 'small' in breed_size:
            return 'small'
        elif 'toy' in breed_size or 'tiny' in breed_size:
            return 'toy'
        else:
            return 'medium'

    def _parse_exercise_level(self, exercise_description: str) -> str:
        """Parse exercise level from description"""
        exercise_lower = exercise_description.lower()
        if any(term in exercise_lower for term in ['very high', 'extremely high', 'intense']):
            return 'very high'
        elif 'high' in exercise_lower:
            return 'high'
        elif any(term in exercise_lower for term in ['low', 'minimal']):
            return 'low'
        else:
            return 'moderate'

    def _get_user_exercise_level(self, minutes: int) -> str:
        """Convert exercise minutes to level"""
        if minutes < 30:
            return 'low'
        elif minutes < 60:
            return 'moderate'
        else:
            return 'high'

    def _calculate_exercise_type_match(self, breed_info: Dict, user_type: str) -> float:
        """Calculate exercise type compatibility"""
        breed_description = str(breed_info.get('Exercise Needs', '')).lower()

        if user_type == 'active_training':
            if any(term in breed_description for term in ['agility', 'working', 'herding']):
                return 1.0
            elif 'sprint' in breed_description:
                return 0.6  # Afghan Hound case
        elif user_type == 'light_walks':
            if any(term in breed_description for term in ['gentle', 'moderate', 'light']):
                return 1.0
            elif any(term in breed_description for term in ['intense', 'vigorous']):
                return 0.3

        return 0.7  # Default moderate match

    def calculate_unified_breed_score(self, breed_name: str, user_prefs: UserPreferences) -> UnifiedBreedScore:
        """計算統一品種分數"""
        # 獲取品種資訊
        try:
            breed_info = get_dog_description(breed_name.replace(' ', '_'))
        except ImportError:
            breed_info = None

        if not breed_info:
            return self._get_default_breed_score(breed_name)

        breed_info['breed_name'] = breed_name

        # 計算各維度分數 (簡化版，包含主要維度)
        dimensional_scores = [
            self.calculate_space_compatibility(breed_info, user_prefs),
            self.calculate_exercise_compatibility(breed_info, user_prefs)
        ]

        # 計算加權總分
        weighted_sum = sum(score.raw_score * score.weight for score in dimensional_scores)
        total_weight = sum(score.weight for score in dimensional_scores)
        base_overall_score = weighted_sum / total_weight if total_weight > 0 else 0.5

        # 計算加分和扣分因素
        bonus_factors = {}
        penalty_factors = {}

        # 應用加分扣分
        overall_score = max(0.0, min(1.0, base_overall_score))

        return UnifiedBreedScore(
            breed_name=breed_name,
            overall_score=overall_score,
            dimensional_scores=dimensional_scores,
            bonus_factors=bonus_factors,
            penalty_factors=penalty_factors,
            confidence_level=0.8,
            match_explanation=f"Breed assessment for {breed_name} based on unified scoring system",
            warnings=[]
        )

    def _get_default_breed_score(self, breed_name: str) -> UnifiedBreedScore:
        """獲取預設品種分數"""
        default_dimensional_scores = [
            DimensionalScore('space_compatibility', 0.6, 0.25, 0.6, 'Insufficient information'),
            DimensionalScore('exercise_compatibility', 0.6, 0.20, 0.6, 'Insufficient information')
        ]

        return UnifiedBreedScore(
            breed_name=breed_name,
            overall_score=0.6,
            dimensional_scores=default_dimensional_scores,
            bonus_factors={},
            penalty_factors={},
            confidence_level=0.3,
            match_explanation="Insufficient data available, recommend further research on this breed",
            warnings=["Incomplete breed information, scores are for reference only"]
        )


def calculate_unified_breed_scores(breed_list: List[str], user_prefs: UserPreferences) -> List[UnifiedBreedScore]:
    """計算多個品種的統一分數"""
    scoring_system = UnifiedScoringSystem()
    scores = []

    for breed in breed_list:
        breed_score = scoring_system.calculate_unified_breed_score(breed, user_prefs)
        scores.append(breed_score)

    # 按總分排序
    scores.sort(key=lambda x: x.overall_score, reverse=True)

    return scores
