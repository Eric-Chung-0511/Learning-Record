import math
from typing import Dict, Any
from dataclasses import dataclass

@dataclass
class UserPreferences:
    """使用者偏好設定的資料結構"""
    living_space: str  # "apartment", "house_small", "house_large"
    yard_access: str  # "no_yard", "shared_yard", "private_yard"
    exercise_time: int  # minutes per day
    exercise_type: str  # "light_walks", "moderate_activity", "active_training"
    grooming_commitment: str  # "low", "medium", "high"
    experience_level: str  # "beginner", "intermediate", "advanced"
    time_availability: str  # "limited", "moderate", "flexible"
    has_children: bool
    children_age: str  # "toddler", "school_age", "teenager"
    noise_tolerance: str  # "low", "medium", "high"
    space_for_play: bool
    other_pets: bool
    climate: str  # "cold", "moderate", "hot"
    health_sensitivity: str = "medium"
    barking_acceptance: str = None
    size_preference: str = "no_preference"  # "no_preference", "small", "medium", "large", "giant"
    training_commitment: str = "medium"  # "low", "medium", "high" - 訓練投入程度
    living_environment: str = "ground_floor"  # "ground_floor", "with_elevator", "walk_up" - 居住環境細節

    def __post_init__(self):
        if self.barking_acceptance is None:
            self.barking_acceptance = self.noise_tolerance


class BonusPenaltyEngine:
    """
    加分扣分引擎類別
    負責處理所有品種加分機制、額外評估因素和分數分布優化
    """
    
    def __init__(self):
        """初始化加分扣分引擎"""
        pass
    
    @staticmethod
    def calculate_breed_bonus(breed_info: dict, user_prefs: 'UserPreferences') -> float:
        """
        計算品種額外加分
        
        Args:
            breed_info: 品種資訊字典
            user_prefs: 使用者偏好設定
            
        Returns:
            float: 品種加分 (-0.25 到 0.5 之間)
        """
        bonus = 0.0
        temperament = breed_info.get('Temperament', '').lower()

        # 1. 壽命加分（最高0.05）
        try:
            lifespan = breed_info.get('Lifespan', '10-12 years')
            years = [int(x) for x in lifespan.split('-')[0].split()[0:1]]
            longevity_bonus = min(0.05, (max(years) - 10) * 0.01)
            bonus += longevity_bonus
        except:
            pass

        # 2. 性格特徵加分（最高0.15）
        positive_traits = {
            'friendly': 0.05,
            'gentle': 0.05,
            'patient': 0.05,
            'intelligent': 0.04,
            'adaptable': 0.04,
            'affectionate': 0.04,
            'easy-going': 0.03,
            'calm': 0.03
        }

        negative_traits = {
            'aggressive': -0.08,
            'stubborn': -0.06,
            'dominant': -0.06,
            'aloof': -0.04,
            'nervous': -0.05,
            'protective': -0.04
        }

        personality_score = sum(value for trait, value in positive_traits.items() if trait in temperament)
        personality_score += sum(value for trait, value in negative_traits.items() if trait in temperament)
        bonus += max(-0.15, min(0.15, personality_score))

        # 3. 適應性加分（最高0.1）
        adaptability_bonus = 0.0
        if breed_info.get('Size') == "Small" and user_prefs.living_space == "apartment":
            adaptability_bonus += 0.05
        if 'adaptable' in temperament or 'versatile' in temperament:
            adaptability_bonus += 0.05
        bonus += min(0.1, adaptability_bonus)

        # 4. 家庭相容性（最高0.15）
        if user_prefs.has_children:
            family_traits = {
                'good with children': 0.06,
                'patient': 0.05,
                'gentle': 0.05,
                'tolerant': 0.04,
                'playful': 0.03
            }
            unfriendly_traits = {
                'aggressive': -0.08,
                'nervous': -0.07,
                'protective': -0.06,
                'territorial': -0.05
            }

            # 年齡評估
            age_adjustments = {
                'toddler': {'bonus_mult': 0.7, 'penalty_mult': 1.3},
                'school_age': {'bonus_mult': 1.0, 'penalty_mult': 1.0},
                'teenager': {'bonus_mult': 1.2, 'penalty_mult': 0.8}
            }

            adj = age_adjustments.get(user_prefs.children_age,
                                    {'bonus_mult': 1.0, 'penalty_mult': 1.0})

            family_bonus = sum(value for trait, value in family_traits.items()
                              if trait in temperament) * adj['bonus_mult']
            family_penalty = sum(value for trait, value in unfriendly_traits.items()
                               if trait in temperament) * adj['penalty_mult']

            bonus += min(0.15, max(-0.2, family_bonus + family_penalty))

        # 5. 專門技能加分（最高0.1）
        skill_bonus = 0.0
        special_abilities = {
            'working': 0.03,
            'herding': 0.03,
            'hunting': 0.03,
            'tracking': 0.03,
            'agility': 0.02
        }
        for ability, value in special_abilities.items():
            if ability in temperament.lower():
                skill_bonus += value
        bonus += min(0.1, skill_bonus)

        # 6. 適應性評估（增強版）
        adaptability_bonus = 0.0
        if breed_info.get('Size') == "Small" and user_prefs.living_space == "apartment":
            adaptability_bonus += 0.08  # 小型犬更適合公寓

        # 環境適應性評估
        if 'adaptable' in temperament or 'versatile' in temperament:
            if user_prefs.living_space == "apartment":
                adaptability_bonus += 0.10  # 適應性在公寓環境更重要
            else:
                adaptability_bonus += 0.05  # 其他環境仍有加分

        # 氣候適應性
        description = breed_info.get('Description', '').lower()
        climate = user_prefs.climate
        if climate == 'hot':
            if 'heat tolerant' in description or 'warm climate' in description:
                adaptability_bonus += 0.08
            elif 'thick coat' in description or 'cold climate' in description:
                adaptability_bonus -= 0.10
        elif climate == 'cold':
            if 'thick coat' in description or 'cold climate' in description:
                adaptability_bonus += 0.08
            elif 'heat tolerant' in description or 'short coat' in description:
                adaptability_bonus -= 0.10

        bonus += min(0.15, adaptability_bonus)

        return min(0.5, max(-0.25, bonus))

    @staticmethod
    def calculate_additional_factors(breed_info: dict, user_prefs: 'UserPreferences') -> dict:
        """
        計算額外的評估因素，結合品種特性與使用者需求的全面評估系統

        1. 多功能性評估 - 品種的多樣化能力
        2. 訓練性評估 - 學習和服從能力
        3. 能量水平評估 - 活力和運動需求
        4. 美容需求評估 - 護理和維護需求
        5. 社交需求評估 - 與人互動的需求程度
        6. 氣候適應性 - 對環境的適應能力
        7. 運動類型匹配 - 與使用者運動習慣的契合度
        8. 生活方式適配 - 與使用者日常生活的匹配度
        """
        factors = {
            'versatility': 0.0,        # 多功能性
            'trainability': 0.0,       # 可訓練度
            'energy_level': 0.0,       # 能量水平
            'grooming_needs': 0.0,     # 美容需求
            'social_needs': 0.0,       # 社交需求
            'weather_adaptability': 0.0,# 氣候適應性
            'exercise_match': 0.0,     # 運動匹配度
            'lifestyle_fit': 0.0       # 生活方式適配度
        }

        temperament = breed_info.get('Temperament', '').lower()
        description = breed_info.get('Description', '').lower()
        size = breed_info.get('Size', 'Medium')

        # 1. 多功能性評估 - 加強品種用途評估
        versatile_traits = {
            'intelligent': 0.25,
            'adaptable': 0.25,
            'trainable': 0.20,
            'athletic': 0.15,
            'versatile': 0.15
        }

        working_roles = {
            'working': 0.20,
            'herding': 0.15,
            'hunting': 0.15,
            'sporting': 0.15,
            'companion': 0.10
        }

        # 計算特質分數
        trait_score = sum(value for trait, value in versatile_traits.items()
                         if trait in temperament)

        # 計算角色分數
        role_score = sum(value for role, value in working_roles.items()
                        if role in description)

        # 根據使用者需求調整多功能性評分
        purpose_traits = {
            'light_walks': ['calm', 'gentle', 'easy-going'],
            'moderate_activity': ['adaptable', 'balanced', 'versatile'],
            'active_training': ['intelligent', 'trainable', 'working']
        }

        if user_prefs.exercise_type in purpose_traits:
            matching_traits = sum(1 for trait in purpose_traits[user_prefs.exercise_type]
                                if trait in temperament)
            trait_score += matching_traits * 0.15

        factors['versatility'] = min(1.0, trait_score + role_score)

        # 2. 訓練性評估
        trainable_traits = {
            'intelligent': 0.3,
            'eager to please': 0.3,
            'trainable': 0.2,
            'quick learner': 0.2,
            'obedient': 0.2
        }

        base_trainability = sum(value for trait, value in trainable_traits.items()
                              if trait in temperament)

        # 根據使用者經驗調整訓練性評分
        experience_multipliers = {
            'beginner': 1.2,    # 新手更需要容易訓練的狗
            'intermediate': 1.0,
            'advanced': 0.8     # 專家能處理較難訓練的狗
        }

        factors['trainability'] = min(1.0, base_trainability *
                                    experience_multipliers.get(user_prefs.experience_level, 1.0))

        # 3. 能量水平評估
        exercise_needs = breed_info.get('Exercise Needs', 'MODERATE').upper()
        energy_levels = {
            'VERY HIGH': {
                'score': 1.0,
                'min_exercise': 120,
                'ideal_exercise': 150
            },
            'HIGH': {
                'score': 0.8,
                'min_exercise': 90,
                'ideal_exercise': 120
            },
            'MODERATE': {
                'score': 0.6,
                'min_exercise': 60,
                'ideal_exercise': 90
            },
            'LOW': {
                'score': 0.4,
                'min_exercise': 30,
                'ideal_exercise': 60
            }
        }

        breed_energy = energy_levels.get(exercise_needs, energy_levels['MODERATE'])

        # 計算運動時間匹配度
        if user_prefs.exercise_time >= breed_energy['ideal_exercise']:
            energy_score = breed_energy['score']
        else:
            # 如果運動時間不足，按比例降低分數
            deficit_ratio = max(0.4, user_prefs.exercise_time / breed_energy['ideal_exercise'])
            energy_score = breed_energy['score'] * deficit_ratio

        factors['energy_level'] = energy_score

        # 4. 美容需求評估
        grooming_needs = breed_info.get('Grooming Needs', 'MODERATE').upper()
        grooming_levels = {
            'HIGH': 1.0,
            'MODERATE': 0.6,
            'LOW': 0.3
        }

        # 特殊毛髮類型評估
        coat_adjustments = 0
        if 'long coat' in description:
            coat_adjustments += 0.2
        if 'double coat' in description:
            coat_adjustments += 0.15
        if 'curly' in description:
            coat_adjustments += 0.15

        # 根據使用者承諾度調整
        commitment_multipliers = {
            'low': 1.5,     # 低承諾度時加重美容需求的影響
            'medium': 1.0,
            'high': 0.8     # 高承諾度時降低美容需求的影響
        }

        base_grooming = grooming_levels.get(grooming_needs, 0.6) + coat_adjustments
        factors['grooming_needs'] = min(1.0, base_grooming *
                                      commitment_multipliers.get(user_prefs.grooming_commitment, 1.0))

        # 5. 社交需求評估
        social_traits = {
            'friendly': 0.25,
            'social': 0.25,
            'affectionate': 0.20,
            'people-oriented': 0.20
        }

        antisocial_traits = {
            'independent': -0.20,
            'aloof': -0.20,
            'reserved': -0.15
        }

        social_score = sum(value for trait, value in social_traits.items()
                          if trait in temperament)
        antisocial_score = sum(value for trait, value in antisocial_traits.items()
                              if trait in temperament)

        # 家庭情況調整
        if user_prefs.has_children:
            child_friendly_bonus = 0.2 if 'good with children' in temperament else 0
            social_score += child_friendly_bonus

        factors['social_needs'] = min(1.0, max(0.0, social_score + antisocial_score))

        # 6. 氣候適應性評估 - 更細緻的環境適應評估
        climate_traits = {
            'cold': {
                'positive': ['thick coat', 'winter', 'cold climate'],
                'negative': ['short coat', 'heat sensitive']
            },
            'hot': {
                'positive': ['short coat', 'heat tolerant', 'warm climate'],
                'negative': ['thick coat', 'cold climate']
            },
            'moderate': {
                'positive': ['adaptable', 'all climate'],
                'negative': []
            }
        }

        climate_score = 0.4  # 基礎分數
        if user_prefs.climate in climate_traits:
            # 正面特質加分
            climate_score += sum(0.2 for term in climate_traits[user_prefs.climate]['positive']
                               if term in description)
            # 負面特質減分
            climate_score -= sum(0.2 for term in climate_traits[user_prefs.climate]['negative']
                               if term in description)

        factors['weather_adaptability'] = min(1.0, max(0.0, climate_score))

        # 7. 運動類型匹配評估
        exercise_type_traits = {
            'light_walks': ['calm', 'gentle'],
            'moderate_activity': ['adaptable', 'balanced'],
            'active_training': ['athletic', 'energetic']
        }

        if user_prefs.exercise_type in exercise_type_traits:
            match_score = sum(0.25 for trait in exercise_type_traits[user_prefs.exercise_type]
                             if trait in temperament)
            factors['exercise_match'] = min(1.0, match_score + 0.5)  # 基礎分0.5

        # 8. 生活方式適配評估
        lifestyle_score = 0.5  # 基礎分數

        # 空間適配
        if user_prefs.living_space == 'apartment':
            if size == 'Small':
                lifestyle_score += 0.2
            elif size == 'Large':
                lifestyle_score -= 0.2
        elif user_prefs.living_space == 'house_large':
            if size in ['Large', 'Giant']:
                lifestyle_score += 0.2

        # 時間可用性適配
        time_availability_bonus = {
            'limited': -0.1,
            'moderate': 0,
            'flexible': 0.1
        }
        lifestyle_score += time_availability_bonus.get(user_prefs.time_availability, 0)

        factors['lifestyle_fit'] = min(1.0, max(0.0, lifestyle_score))

        return factors

    def amplify_score_extreme(self, score: float) -> float:
        """
        優化分數分布，提供更有意義的評分範圍。
        純粹進行數學轉換，不依賴外部資訊。

        Parameters:
            score: 原始評分（0-1之間的浮點數）

        Returns:
            float: 調整後的評分（0-1之間的浮點數）
        """
        def smooth_curve(x: float, steepness: float = 12) -> float:
            """創建平滑的S型曲線用於分數轉換"""
            return 1 / (1 + math.exp(-steepness * (x - 0.5)))

        # 90-100分的轉換（極佳匹配）
        if score >= 0.90:
            position = (score - 0.90) / 0.10
            return 0.96 + (position * 0.04)

        # 80-90分的轉換（優秀匹配）
        elif score >= 0.80:
            position = (score - 0.80) / 0.10
            return 0.90 + (position * 0.06)

        # 70-80分的轉換（良好匹配）
        elif score >= 0.70:
            position = (score - 0.70) / 0.10
            return 0.82 + (position * 0.08)

        # 50-70分的轉換（可接受匹配）
        elif score >= 0.50:
            position = (score - 0.50) / 0.20
            return 0.75 + (smooth_curve(position) * 0.07)

        # 50分以下的轉換（較差匹配）
        else:
            position = score / 0.50
            return 0.70 + (smooth_curve(position) * 0.05)

    def apply_special_case_adjustments(self, score: float, user_prefs: UserPreferences, breed_info: dict) -> float:
        """
        處理特殊情況和極端案例的評分調整。這個函數特別關注：
        1. 條件組合的協同效應
        2. 品種特性的獨特需求
        3. 極端情況的合理處理

        這個函數就像是一個細心的裁判，會考慮到各種特殊情況，
        並根據具體場景做出合理的評分調整。

        Parameters:
            score: 初始評分
            user_prefs: 使用者偏好
            breed_info: 品種資訊
        Returns:
            float: 調整後的評分（0.2-1.0之間）
        """
        severity_multiplier = 1.0

        def evaluate_spatial_exercise_combination() -> float:
            """
            評估空間與運動需求的組合效應。

            這個函數不再過分懲罰大型犬，而是更多地考慮品種的實際特性。
            就像評估一個運動員是否適合在特定場地訓練一樣，我們需要考慮
            場地大小和運動需求的整體匹配度。
            """
            multiplier = 1.0

            if user_prefs.living_space == 'apartment':
                temperament = breed_info.get('Temperament', '').lower()
                description = breed_info.get('Description', '').lower()

                # 檢查品種是否有利於公寓生活的特徵
                apartment_friendly = any(trait in temperament or trait in description
                                      for trait in ['calm', 'adaptable', 'quiet'])

                # 大型犬的特殊處理
                if breed_info['Size'] in ['Large', 'Giant']:
                    if apartment_friendly:
                        multiplier *= 0.85  # 從0.7提升到0.85，降低懲罰
                    else:
                        multiplier *= 0.75  # 從0.5提升到0.75

                # 檢查運動需求的匹配度
                exercise_needs = breed_info.get('Exercise Needs', 'MODERATE').upper()
                exercise_time = user_prefs.exercise_time

                if exercise_needs in ['HIGH', 'VERY HIGH']:
                    if exercise_time >= 120:  # 高運動量可以部分補償空間限制
                        multiplier *= 1.1

            return multiplier

        def evaluate_experience_combination() -> float:
            """
            評估經驗需求的複合影響。

            這個函數就像是評估一個工作崗位與應聘者經驗的匹配度，
            需要綜合考慮工作難度和應聘者能力。
            """
            multiplier = 1.0
            temperament = breed_info.get('Temperament', '').lower()
            care_level = breed_info.get('Care Level', 'MODERATE')

            # 新手飼主的特殊考慮，更寬容的評估標準
            if user_prefs.experience_level == 'beginner':
                if care_level == 'HIGH':
                    if user_prefs.has_children:
                        multiplier *= 0.7  # 從0.5提升到0.7
                    else:
                        multiplier *= 0.8  # 從0.6提升到0.8

                # 性格特徵影響，降低懲罰程度
                challenging_traits = {
                    'stubborn': -0.10,      # 從-0.15降低
                    'independent': -0.08,    # 從-0.12降低
                    'dominant': -0.08,       # 從-0.12降低
                    'protective': -0.06,     # 從-0.10降低
                    'aggressive': -0.15      # 保持較高懲罰因安全考慮
                }

                for trait, penalty in challenging_traits.items():
                    if trait in temperament:
                        multiplier *= (1 + penalty)

            return multiplier

        def evaluate_breed_specific_requirements() -> float:
            """
            評估品種特定需求。

            這個函數就像是為每個品種量身定制評估標準，
            考慮其獨特的特性和需求。
            """
            multiplier = 1.0
            exercise_time = user_prefs.exercise_time
            exercise_type = user_prefs.exercise_type

            # 檢查品種特性
            temperament = breed_info.get('Temperament', '').lower()
            description = breed_info.get('Description', '').lower()
            exercise_needs = breed_info.get('Exercise Needs', 'MODERATE').upper()

            # 運動需求匹配度評估，更合理的標準
            if exercise_needs == 'LOW':
                if exercise_time > 120:
                    multiplier *= 0.85  # 從0.5提升到0.85
            elif exercise_needs == 'VERY HIGH':
                if exercise_time < 60:
                    multiplier *= 0.7   # 從0.5提升到0.7

            # 特殊品種類型的考慮
            if 'sprint' in temperament:
                if exercise_time > 120 and exercise_type != 'active_training':
                    multiplier *= 0.85  # 從0.7提升到0.85

            if any(trait in temperament for trait in ['working', 'herding']):
                if exercise_time < 90 or exercise_type == 'light_walks':
                    multiplier *= 0.8   # 從0.7提升到0.8

            return multiplier

        # 計算各項調整
        space_exercise_mult = evaluate_spatial_exercise_combination()
        experience_mult = evaluate_experience_combination()
        breed_specific_mult = evaluate_breed_specific_requirements()

        # 整合所有調整因素
        severity_multiplier *= space_exercise_mult
        severity_multiplier *= experience_mult
        severity_multiplier *= breed_specific_mult

        # 應用最終調整，確保分數在合理範圍內
        final_score = score * severity_multiplier
        return max(0.2, min(1.0, final_score))
