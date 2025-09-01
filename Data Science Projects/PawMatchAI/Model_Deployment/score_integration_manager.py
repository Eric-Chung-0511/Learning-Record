import math
import traceback
from typing import Dict, Any, List
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


class ScoreIntegrationManager:
    """
    評分整合管理器類別
    負責動態權重計算、評分整合和條件互動評估
    """
    
    def __init__(self):
        """初始化評分整合管理器"""
        pass
    
    def apply_size_filter(self, breed_score: float, user_preference: str, breed_size: str) -> float:
        """
        強過濾機制，基於用戶的體型偏好過濾品種

        Parameters:
            breed_score (float): 原始品種評分
            user_preference (str): 用戶偏好的體型
            breed_size (str): 品種的實際體型

        Returns:
            float: 過濾後的評分，如果體型不符合會返回 0
        """
        if user_preference == "no_preference":
            return breed_score

        # 標準化 size 字串以進行比較
        breed_size = breed_size.lower().strip()
        user_preference = user_preference.lower().strip()

        # 特殊處理 "varies" 的情況
        if breed_size == "varies":
            return breed_score * 0.5  # 給予一個折扣係數，因為不確定性

        # 如果用戶有明確體型偏好但品種不符合，返回 0
        if user_preference != breed_size:
            return 0

        return breed_score

    def calculate_breed_compatibility_score(self, scores: dict, user_prefs: UserPreferences, breed_info: dict) -> float:
        """
        計算品種相容性總分，完整實現原始版本的複雜邏輯：
        1. 運動類型與時間的精確匹配
        2. 進階使用者的專業需求
        3. 空間利用的實際效果
        4. 條件組合的嚴格評估
        """
        def evaluate_perfect_conditions():
            """
            評估條件匹配度，特別強化：
            1. 運動類型與時間的綜合評估
            2. 專業技能需求評估
            3. 品種特性評估
            """
            perfect_matches = {
                'size_match': 0,
                'exercise_match': 0,
                'experience_match': 0,
                'living_condition_match': 0,
                'breed_trait_match': 0  # 新增品種特性匹配度
            }

            # 第一部分：運動需求評估
            def evaluate_exercise_compatibility():
                """
                評估運動需求的匹配度，特別關注：
                1. 時間與強度的合理搭配
                2. 不同品種的運動特性
                3. 運動類型的適配性

                這個函數就像是一個體育教練，需要根據每個"運動員"(狗品種)的特點，
                為他們制定合適的訓練計劃。
                """
                exercise_needs = breed_info.get('Exercise Needs', 'MODERATE').upper()
                exercise_time = user_prefs.exercise_time
                exercise_type = user_prefs.exercise_type
                temperament = breed_info.get('Temperament', '').lower()
                description = breed_info.get('Description', '').lower()

                # 定義更精確的品種運動特性
                breed_exercise_patterns = {
                    'sprint_type': {  # 短跑型犬種，如 Whippet, Saluki
                        'identifiers': ['fast', 'speed', 'sprint', 'racing', 'coursing', 'sight hound'],
                        'ideal_exercise': {
                            'active_training': 1.0,     # 完美匹配高強度訓練
                            'moderate_activity': 0.5,    # 持續運動不是最佳選擇
                            'light_walks': 0.3          # 輕度運動效果很差
                        },
                        'time_ranges': {
                            'ideal': (30, 60),          # 最適合的運動時間範圍
                            'acceptable': (20, 90),      # 可以接受的時間範圍
                            'penalty_start': 90         # 開始給予懲罰的時間點
                        },
                        'penalty_rate': 0.8            # 超出範圍時的懲罰係數
                    },
                    'endurance_type': {  # 耐力型犬種，如 Border Collie
                        'identifiers': ['herding', 'working', 'tireless', 'energetic', 'stamina', 'athletic'],
                        'ideal_exercise': {
                            'active_training': 0.9,     # 高強度訓練很好
                            'moderate_activity': 1.0,    # 持續運動是最佳選擇
                            'light_walks': 0.4          # 輕度運動不足
                        },
                        'time_ranges': {
                            'ideal': (90, 180),         # 需要較長的運動時間
                            'acceptable': (60, 180),
                            'penalty_start': 60         # 運動時間過短會受罰
                        },
                        'penalty_rate': 0.7
                    },
                    'moderate_type': {  # 一般活動型犬種，如 Labrador
                        'identifiers': ['friendly', 'playful', 'adaptable', 'versatile', 'companion'],
                        'ideal_exercise': {
                            'active_training': 0.8,
                            'moderate_activity': 1.0,
                            'light_walks': 0.6
                        },
                        'time_ranges': {
                            'ideal': (60, 120),
                            'acceptable': (45, 150),
                            'penalty_start': 150
                        },
                        'penalty_rate': 0.6
                    }
                }

                def determine_breed_type():
                    """改進品種運動類型的判斷，更精確識別工作犬"""
                    # 優先檢查特殊運動類型的標識符
                    for breed_type, pattern in breed_exercise_patterns.items():
                        if any(identifier in temperament or identifier in description
                              for identifier in pattern['identifiers']):
                            return breed_type

                    # 改進：根據運動需求和工作犬特徵進行更細緻的判斷
                    if (exercise_needs in ['VERY HIGH', 'HIGH'] or
                        any(trait in temperament.lower() for trait in
                            ['herding', 'working', 'intelligent', 'athletic', 'tireless'])):
                        if user_prefs.experience_level == 'advanced':
                            return 'endurance_type'  # 優先判定為耐力型
                    elif exercise_needs == 'LOW':
                        return 'moderate_type'

                    return 'moderate_type'

                def calculate_time_match(pattern):
                    """
                    計算運動時間的匹配度。
                    這就像在判斷運動時間是否符合訓練計劃。
                    """
                    ideal_min, ideal_max = pattern['time_ranges']['ideal']
                    accept_min, accept_max = pattern['time_ranges']['acceptable']
                    penalty_start = pattern['time_ranges']['penalty_start']

                    # 在理想範圍內
                    if ideal_min <= exercise_time <= ideal_max:
                        return 1.0

                    # 超出可接受範圍的嚴格懲罰
                    elif exercise_time < accept_min:
                        deficit = accept_min - exercise_time
                        return max(0.2, 1 - (deficit / accept_min) * 1.2)
                    elif exercise_time > accept_max:
                        excess = exercise_time - penalty_start
                        penalty = min(0.8, (excess / penalty_start) * pattern['penalty_rate'])
                        return max(0.2, 1 - penalty)

                    # 在可接受範圍但不在理想範圍
                    else:
                        if exercise_time < ideal_min:
                            progress = (exercise_time - accept_min) / (ideal_min - accept_min)
                            return 0.6 + (0.4 * progress)
                        else:
                            remaining = (accept_max - exercise_time) / (accept_max - ideal_max)
                            return 0.6 + (0.4 * remaining)

                def apply_special_adjustments(time_score, type_score, breed_type, pattern):
                    """
                    處理特殊情況，確保運動方式真正符合品種需求。
                    特別加強：
                    1. 短跑型犬種的長時間運動懲罰
                    2. 耐力型犬種的獎勵機制
                    3. 運動類型匹配的重要性
                    """
                    # 短跑型品種的特殊處理
                    if breed_type == 'sprint_type':
                        if exercise_time > pattern['time_ranges']['penalty_start']:
                            # 加重長時間運動的懲罰
                            penalty_factor = min(0.8, (exercise_time - pattern['time_ranges']['penalty_start']) / 60)
                            time_score *= max(0.3, 1 - penalty_factor)  # 最低降到0.3
                            # 運動類型不適合時的額外懲罰
                            if exercise_type != 'active_training':
                                type_score *= 0.3  # 更嚴重的懲罰

                    # 耐力型品種的特殊處理
                    elif breed_type == 'endurance_type':
                        if exercise_time < pattern['time_ranges']['penalty_start']:
                            time_score *= 0.5  # 維持運動不足的懲罰
                        elif exercise_time >= 150:  # 新增：高運動量獎勵
                            if exercise_type in ['active_training', 'moderate_activity']:
                                time_bonus = min(0.3, (exercise_time - 150) / 150)
                                time_score = min(1.0, time_score * (1 + time_bonus))
                                type_score = min(1.0, type_score * 1.2)

                        # 運動強度不足的懲罰
                        if exercise_type == 'light_walks':
                            if exercise_time > 90:
                                type_score *= 0.4  # 加重懲罰
                            else:
                                type_score *= 0.5

                    return time_score, type_score

                # 執行評估流程
                breed_type = determine_breed_type()
                pattern = breed_exercise_patterns[breed_type]

                # 計算基礎分數
                time_score = calculate_time_match(pattern)
                type_score = pattern['ideal_exercise'].get(exercise_type, 0.5)

                # 應用特殊調整
                time_score, type_score = apply_special_adjustments(time_score, type_score, breed_type, pattern)

                # 根據品種類型決定最終權重
                if breed_type == 'sprint_type':
                    if exercise_time > pattern['time_ranges']['penalty_start']:
                        # 超時時更重視運動類型的匹配度
                        return (time_score * 0.3) + (type_score * 0.7)
                    else:
                        return (time_score * 0.5) + (type_score * 0.5)
                elif breed_type == 'endurance_type':
                    if exercise_time < pattern['time_ranges']['penalty_start']:
                        # 時間不足時更重視時間因素
                        return (time_score * 0.7) + (type_score * 0.3)
                    else:
                        return (time_score * 0.6) + (type_score * 0.4)
                else:
                    return (time_score * 0.5) + (type_score * 0.5)

            # 第二部分：專業技能需求評估
            def evaluate_expertise_requirements():
                care_level = breed_info.get('Care Level', 'MODERATE').upper()
                temperament = breed_info.get('Temperament', '').lower()

                # 定義專業技能要求
                expertise_requirements = {
                    'training_complexity': {
                        'VERY HIGH': {'beginner': 0.2, 'intermediate': 0.5, 'advanced': 0.9},
                        'HIGH': {'beginner': 0.3, 'intermediate': 0.7, 'advanced': 1.0},
                        'MODERATE': {'beginner': 0.6, 'intermediate': 0.9, 'advanced': 1.0},
                        'LOW': {'beginner': 0.9, 'intermediate': 0.95, 'advanced': 0.9}
                    },
                    'special_traits': {
                        'working': 0.2,    # 工作犬需要額外技能
                        'herding': 0.2,    # 牧羊犬需要特殊訓練
                        'intelligent': 0.15,# 高智商犬種需要心智刺激
                        'independent': 0.15,# 獨立性強的需要特殊處理
                        'protective': 0.1   # 護衛犬需要適當訓練
                    }
                }

                # 基礎分數
                base_score = expertise_requirements['training_complexity'][care_level][user_prefs.experience_level]

                # 特殊特徵評估
                trait_penalty = 0
                for trait, penalty in expertise_requirements['special_traits'].items():
                    if trait in temperament:
                        if user_prefs.experience_level == 'beginner':
                            trait_penalty += penalty
                        elif user_prefs.experience_level == 'advanced':
                            trait_penalty -= penalty * 0.5  # 專家反而因應對特殊特徵而加分

                return max(0.2, min(1.0, base_score - trait_penalty))

            def evaluate_living_conditions() -> float:
                """
                評估生活環境適配性，特別加強：
                1. 降低對大型犬的過度懲罰
                2. 增加品種特性評估
                3. 提升對適應性的重視度
                """
                size = breed_info['Size']
                exercise_needs = breed_info.get('Exercise Needs', 'MODERATE').upper()
                temperament = breed_info.get('Temperament', '').lower()
                description = breed_info.get('Description', '').lower()

                # 重新定義空間需求矩陣，降低對大型犬的懲罰
                space_requirements = {
                    'apartment': {
                        'Small': 1.0,
                        'Medium': 0.8,
                        'Large': 0.7,
                        'Giant': 0.6
                    },
                    'house_small': {
                        'Small': 0.9,
                        'Medium': 1.0,
                        'Large': 0.8,
                        'Giant': 0.7
                    },
                    'house_large': {
                        'Small': 0.8,
                        'Medium': 0.9,
                        'Large': 1.0,
                        'Giant': 1.0
                    }
                }

                # 基礎空間分數
                space_score = space_requirements.get(
                    user_prefs.living_space,
                    space_requirements['house_small']
                )[size]

                # 品種適應性評估
                adaptability_bonus = 0
                adaptable_traits = ['adaptable', 'calm', 'quiet', 'gentle', 'laid-back']
                challenging_traits = ['hyperactive', 'restless', 'requires space']

                # 計算適應性加分
                if user_prefs.living_space == 'apartment':
                    for trait in adaptable_traits:
                        if trait in temperament or trait in description:
                            adaptability_bonus += 0.1

                    # 特別處理大型犬的適應性
                    if size in ['Large', 'Giant']:
                        apartment_friendly_traits = ['calm', 'gentle', 'quiet']
                        matched_traits = sum(1 for trait in apartment_friendly_traits
                                           if trait in temperament or trait in description)
                        if matched_traits > 0:
                            adaptability_bonus += 0.15 * matched_traits

                # 活動空間需求調整，更寬容的評估
                if exercise_needs in ['HIGH', 'VERY HIGH']:
                    if user_prefs.living_space != 'house_large':
                        space_score *= 0.9  # 從0.8提升到0.9，降低懲罰

                # 院子可用性評估，提供更合理的獎勵
                yard_scores = {
                    'no_yard': 0.85,      # 從0.7提升到0.85
                    'shared_yard': 0.92,  # 從0.85提升到0.92
                    'private_yard': 1.0
                }
                yard_multiplier = yard_scores.get(user_prefs.yard_access, 0.85)

                # 根據體型調整院子重要性
                if size in ['Large', 'Giant']:
                    yard_importance = 1.2
                elif size == 'Medium':
                    yard_importance = 1.1
                else:
                    yard_importance = 1.0

                # 計算最終分數
                final_score = space_score * (1 + adaptability_bonus)

                # 應用院子影響
                if user_prefs.yard_access != 'no_yard':
                    yard_bonus = (yard_multiplier - 1) * yard_importance
                    final_score = min(1.0, final_score + yard_bonus)

                # 確保分數在合理範圍內，但提供更高的基礎分數
                return max(0.4, min(1.0, final_score))

            # 第四部分：品種特性評估
            def evaluate_breed_traits():
                temperament = breed_info.get('Temperament', '').lower()
                description = breed_info.get('Description', '').lower()

                trait_scores = []

                # 評估性格特徵
                if user_prefs.has_children:
                    if 'good with children' in description:
                        trait_scores.append(1.0)
                    elif 'patient' in temperament or 'gentle' in temperament:
                        trait_scores.append(0.8)
                    else:
                        trait_scores.append(0.5)

                # 評估適應性
                adaptability_keywords = ['adaptable', 'versatile', 'flexible']
                if any(keyword in temperament for keyword in adaptability_keywords):
                    trait_scores.append(1.0)
                else:
                    trait_scores.append(0.7)

                return sum(trait_scores) / len(trait_scores) if trait_scores else 0.7

            # 計算各項匹配分數
            perfect_matches['exercise_match'] = evaluate_exercise_compatibility()
            perfect_matches['experience_match'] = evaluate_expertise_requirements()
            perfect_matches['living_condition_match'] = evaluate_living_conditions()
            perfect_matches['size_match'] = evaluate_living_conditions()  # 共用生活環境評估
            perfect_matches['breed_trait_match'] = evaluate_breed_traits()

            return perfect_matches

        def calculate_weights() -> dict:
            """
            動態計算評分權重，特別關注：
            1. 極端情況的權重調整
            2. 使用者條件的協同效應
            3. 品種特性的影響

            Returns:
                dict: 包含各評分項目權重的字典
            """
            # 定義基礎權重 - 提供更合理的起始分配
            base_weights = {
                'space': 0.25,        # 提升空間權重，因為這是最基本的需求
                'exercise': 0.25,     # 運動需求同樣重要
                'experience': 0.20,   # 保持經驗的重要性
                'grooming': 0.10,     # 稍微降低美容需求的權重
                'noise': 0.10,        # 維持噪音評估的權重
                'health': 0.10        # 維持健康評估的權重
            }

            def analyze_condition_extremity() -> dict:
                """
                評估使用者條件的極端程度，這影響權重的動態調整。
                根據條件的極端程度返回相應的調整建議。
                """
                extremities = {}

                # 運動時間評估 - 更細緻的分級
                if user_prefs.exercise_time <= 30:
                    extremities['exercise'] = ('extremely_low', 0.8)
                elif user_prefs.exercise_time <= 60:
                    extremities['exercise'] = ('low', 0.6)
                elif user_prefs.exercise_time >= 180:
                    extremities['exercise'] = ('extremely_high', 0.8)
                elif user_prefs.exercise_time >= 120:
                    extremities['exercise'] = ('high', 0.6)
                else:
                    extremities['exercise'] = ('moderate', 0.3)

                # 空間限制評估 - 更合理的空間評估
                space_extremity = {
                    'apartment': ('restricted', 0.7),    # 從0.9降低到0.7，減少限制
                    'house_small': ('moderate', 0.5),
                    'house_large': ('spacious', 0.3)
                }
                extremities['space'] = space_extremity.get(user_prefs.living_space, ('moderate', 0.5))

                # 經驗水平評估 - 保持原有的評估邏輯
                experience_extremity = {
                    'beginner': ('low', 0.7),
                    'intermediate': ('moderate', 0.4),
                    'advanced': ('high', 0.6)
                }
                extremities['experience'] = experience_extremity.get(user_prefs.experience_level, ('moderate', 0.5))

                return extremities

            def calculate_weight_adjustments(extremities: dict) -> dict:
                """
                根據極端程度計算權重調整，特別注意條件組合的影響。
                """
                adjustments = {}
                temperament = breed_info.get('Temperament', '').lower()
                is_working_dog = any(trait in temperament
                                   for trait in ['herding', 'working', 'intelligent', 'tireless'])

                # 空間權重調整
                if extremities['space'][0] == 'restricted':
                    if extremities['exercise'][0] in ['high', 'extremely_high']:
                        adjustments['space'] = 1.3
                        adjustments['exercise'] = 2.3
                    else:
                        adjustments['space'] = 1.6
                        adjustments['noise'] = 1.5

                # 運動需求權重調整
                if extremities['exercise'][0] in ['extremely_high', 'extremely_low']:
                    base_adjustment = 2.0
                    if extremities['exercise'][0] == 'extremely_high':
                        if is_working_dog:
                            base_adjustment = 2.3
                    adjustments['exercise'] = base_adjustment

                # 經驗需求權重調整
                if extremities['experience'][0] == 'low':
                    adjustments['experience'] = 1.8
                    if breed_info.get('Care Level') == 'HIGH':
                        adjustments['experience'] = 2.0
                elif extremities['experience'][0] == 'high':
                    if is_working_dog:
                        adjustments['experience'] = 1.8  # 從2.5降低到1.8

                # 特殊組合的處理
                def adjust_for_combinations():
                    if (extremities['space'][0] == 'restricted' and
                        extremities['exercise'][0] in ['high', 'extremely_high']):
                        # 適度降低極端組合的影響
                        adjustments['space'] = adjustments.get('space', 1.0) * 1.2
                        adjustments['exercise'] = adjustments.get('exercise', 1.0) * 1.2

                    # 理想組合的獎勵
                    if (extremities['experience'][0] == 'high' and
                        extremities['space'][0] == 'spacious' and
                        extremities['exercise'][0] in ['high', 'extremely_high'] and
                        is_working_dog):
                        adjustments['exercise'] = adjustments.get('exercise', 1.0) * 1.3
                        adjustments['experience'] = adjustments.get('experience', 1.0) * 1.3

                adjust_for_combinations()
                return adjustments

            # 獲取條件極端度
            extremities = analyze_condition_extremity()

            # 計算權重調整
            weight_adjustments = calculate_weight_adjustments(extremities)

            # 應用權重調整，確保總和為1
            final_weights = base_weights.copy()
            for key, adjustment in weight_adjustments.items():
                if key in final_weights:
                    final_weights[key] *= adjustment

            # 正規化權重
            total_weight = sum(final_weights.values())
            normalized_weights = {k: v/total_weight for k, v in final_weights.items()}

            return normalized_weights

        def calculate_base_score(scores: dict, weights: dict) -> float:
            """
            計算基礎評分分數，採用更靈活的評分機制。

            這個函數使用了改進後的評分邏輯，主要關注：
            1. 降低關鍵指標的最低門檻，使系統更包容
            2. 引入非線性評分曲線，讓分數分布更合理
            3. 優化多重條件失敗的處理方式
            4. 加強對品種特性的考慮

            Parameters:
                scores: 包含各項評分的字典
                weights: 包含各項權重的字典

            Returns:
                float: 0.2到1.0之間的基礎分數
            """
            # 重新定義關鍵指標閾值，提供更寬容的評分標準
            critical_thresholds = {
                'space': 0.35,
                'exercise': 0.35,
                'experience': 0.5,
                'noise': 0.5
            }

            # 評估關鍵指標失敗情況
            def evaluate_critical_failures() -> list:
                """
                評估關鍵指標的失敗情況，但採用更寬容的標準。
                根據品種特性動態調整失敗判定。
                """
                failures = []
                temperament = breed_info.get('Temperament', '').lower()

                for metric, threshold in critical_thresholds.items():
                    if scores[metric] < threshold:
                        # 特殊情況處理：適應性強的品種可以有更低的空間要求
                        if metric == 'space' and any(trait in temperament
                           for trait in ['adaptable', 'calm', 'apartment']):
                            if scores[metric] >= threshold - 0.1:
                                continue

                        # 運動需求的特殊處理
                        elif metric == 'exercise':
                            exercise_needs = breed_info.get('Exercise Needs', 'MODERATE').upper()
                            if exercise_needs == 'LOW' and scores[metric] >= threshold - 0.1:
                                continue

                        failures.append((metric, scores[metric]))

                return failures

            # 計算基礎分數
            def calculate_weighted_score() -> float:
                """
                計算加權分數，使用非線性函數使分數分布更合理。
                """
                weighted_scores = []
                for key, score in scores.items():
                    if key in weights:
                        # 使用sigmoid函數使分數曲線更平滑
                        adjusted_score = 1 / (1 + math.exp(-10 * (score - 0.5)))
                        weighted_scores.append(adjusted_score * weights[key])

                return sum(weighted_scores)

            # 處理臨界失敗情況
            critical_failures = evaluate_critical_failures()
            base_score = calculate_weighted_score()

            if critical_failures:
                # 分離空間和運動相關的懲罰
                space_exercise_penalty = 0
                other_penalty = 0

                for metric, score in critical_failures:
                    if metric in ['space', 'exercise']:
                        # 降低空間和運動失敗的懲罰程度
                        penalty = (critical_thresholds[metric] - score) * 0.08
                        space_exercise_penalty += penalty
                    else:
                        # 其他失敗的懲罰保持較高
                        penalty = (critical_thresholds[metric] - score) * 0.20
                        other_penalty += penalty

                # 計算總懲罰，但使用更溫和的方式
                total_penalty = (space_exercise_penalty + other_penalty) / 2
                base_score *= (1 - total_penalty)

                # 多重失敗的處理更寬容
                if len(critical_failures) > 1:
                    # 從0.98提升到0.99，降低多重失敗的疊加懲罰
                    base_score *= (0.99 ** (len(critical_failures) - 1))

            # 品種特性加分
            def apply_breed_bonus() -> float:
                """
                根據品種特性提供額外加分，
                特別是對於在特定環境下表現良好的品種。
                """
                bonus = 0
                temperament = breed_info.get('Temperament', '').lower()
                description = breed_info.get('Description', '').lower()

                # 適應性加分
                adaptability_traits = ['adaptable', 'versatile', 'easy-going']
                if any(trait in temperament for trait in adaptability_traits):
                    bonus += 0.05

                # 公寓適應性加分
                if user_prefs.living_space == 'apartment':
                    apartment_traits = ['calm', 'quiet', 'good for apartments']
                    if any(trait in temperament or trait in description for trait in apartment_traits):
                        bonus += 0.05

                return min(0.1, bonus)  # 限制最大加分

            # 應用品種特性加分
            breed_bonus = apply_breed_bonus()
            base_score = min(1.0, base_score * (1 + breed_bonus))

            # 確保最終分數在合理範圍內
            return max(0.2, min(1.0, base_score))

        def evaluate_condition_interactions(scores: dict) -> float:
            """評估不同條件間的相互影響，更寬容地處理極端組合"""
            interaction_penalty = 1.0

            # 只保留最基本的經驗相關評估
            if user_prefs.experience_level == 'beginner':
                if breed_info.get('Care Level') == 'HIGH':
                    interaction_penalty *= 0.95

            # 運動時間與類型的基本互動也降低懲罰程度
            exercise_needs = breed_info.get('Exercise Needs', 'MODERATE').upper()
            if exercise_needs == 'VERY HIGH' and user_prefs.exercise_type == 'light_walks':
                interaction_penalty *= 0.95

            return interaction_penalty

        def calculate_adjusted_perfect_bonus(perfect_conditions: dict) -> float:
            """計算完美匹配獎勵，但更注重條件的整體表現"""
            bonus = 1.0

            # 降低單項獎勵的影響力
            bonus += 0.06 * perfect_conditions['size_match']
            bonus += 0.06 * perfect_conditions['exercise_match']
            bonus += 0.06 * perfect_conditions['experience_match']
            bonus += 0.03 * perfect_conditions['living_condition_match']

            # 如果有任何條件表現不佳，降低整體獎勵
            low_scores = [score for score in perfect_conditions.values() if score < 0.6]
            if low_scores:
                bonus *= (0.85 ** len(low_scores))

            # 確保獎勵不會過高
            return min(1.25, bonus)

        def apply_breed_specific_adjustments(score: float) -> float:
            """根據品種特性進行最終調整"""
            # 檢查是否存在極端不匹配的情況
            exercise_mismatch = False
            size_mismatch = False
            experience_mismatch = False

            # 運動需求極端不匹配
            if breed_info.get('Exercise Needs', 'MODERATE').upper() == 'VERY HIGH':
                if user_prefs.exercise_time < 90 or user_prefs.exercise_type == 'light_walks':
                    exercise_mismatch = True

            # 體型與空間極端不匹配
            if user_prefs.living_space == 'apartment' and breed_info['Size'] in ['Large', 'Giant']:
                size_mismatch = True

            # 經驗需求極端不匹配
            if user_prefs.experience_level == 'beginner' and breed_info.get('Care Level') == 'HIGH':
                experience_mismatch = True

            # 根據不匹配的數量進行懲罰
            mismatch_count = sum([exercise_mismatch, size_mismatch, experience_mismatch])
            if mismatch_count > 0:
                score *= (0.8 ** mismatch_count)

            return score

        # 計算動態權重
        weights = calculate_weights()

        # 正規化權重
        total_weight = sum(weights.values())
        normalized_weights = {k: v/total_weight for k, v in weights.items()}

        # 計算基礎分數
        base_score = calculate_base_score(scores, normalized_weights)

        # 評估條件互動
        interaction_multiplier = evaluate_condition_interactions(scores)

        # 計算完美匹配獎勵
        perfect_conditions = evaluate_perfect_conditions()
        perfect_bonus = calculate_adjusted_perfect_bonus(perfect_conditions)

        # 計算初步分數
        preliminary_score = base_score * interaction_multiplier * perfect_bonus

        # 應用品種特定調整
        final_score = apply_breed_specific_adjustments(preliminary_score)

        # 確保分數在合理範圍內，並降低最高可能分數
        max_possible_score = 0.96  # 降低最高可能分數
        min_possible_score = 0.3

        return min(max_possible_score, max(min_possible_score, final_score))

    def calculate_environmental_fit(self, breed_info: dict, user_prefs: UserPreferences) -> float:
        """
        計算品種與環境的適應性加成
        
        Args:
            breed_info: 品種資訊
            user_prefs: 使用者偏好
            
        Returns:
            float: 環境適應性加成分數
        """
        adaptability_score = 0.0
        description = breed_info.get('Description', '').lower()
        temperament = breed_info.get('Temperament', '').lower()

        # 環境適應性評估
        if user_prefs.living_space == 'apartment':
            if 'adaptable' in temperament or 'apartment' in description:
                adaptability_score += 0.1
            if breed_info.get('Size') == 'Small':
                adaptability_score += 0.05
        elif user_prefs.living_space == 'house_large':
            if 'active' in temperament or 'energetic' in description:
                adaptability_score += 0.1

        # 氣候適應性
        if user_prefs.climate in description or user_prefs.climate in temperament:
            adaptability_score += 0.05

        return min(0.2, adaptability_score)
