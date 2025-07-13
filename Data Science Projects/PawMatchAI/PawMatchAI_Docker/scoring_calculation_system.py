from dataclasses import dataclass
from breed_health_info import breed_health_info
from breed_noise_info import breed_noise_info
import traceback
import math

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


@staticmethod
def calculate_breed_bonus(breed_info: dict, user_prefs: 'UserPreferences') -> float:
    """計算品種額外加分"""
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

    # 4. 家庭相容性（最高0.1）
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


    # 6. 適應性評估 
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

        def calculate_space_score(size: str, living_space: str, has_yard: bool, exercise_needs: str) -> float:
            """
            1. 動態的基礎分數矩陣
            2. 強化空間品質評估
            3. 增加極端情況處理
            4. 考慮不同空間組合的協同效應
            """
            def get_base_score():
                # 基礎分數矩陣 - 更極端的分數分配
                base_matrix = {
                    "Small": {
                        "apartment": {
                            "no_yard": 0.85,      # 小型犬在公寓仍然適合
                            "shared_yard": 0.90,   # 共享院子提供額外活動空間
                            "private_yard": 0.95   # 私人院子最理想
                        },
                        "house_small": {
                            "no_yard": 0.80,
                            "shared_yard": 0.85,
                            "private_yard": 0.90
                        },
                        "house_large": {
                            "no_yard": 0.75,
                            "shared_yard": 0.80,
                            "private_yard": 0.85
                        }
                    },
                    "Medium": {
                        "apartment": {
                            "no_yard": 0.75,      
                            "shared_yard": 0.85,
                            "private_yard": 0.90
                        },
                        "house_small": {
                            "no_yard": 0.80,
                            "shared_yard": 0.90,
                            "private_yard": 0.90
                        },
                        "house_large": {
                            "no_yard": 0.85,
                            "shared_yard": 0.90,
                            "private_yard": 0.95
                        }
                    },
                    "Large": {
                        "apartment": {
                            "no_yard": 0.70,      
                            "shared_yard": 0.80,
                            "private_yard": 0.85
                        },
                        "house_small": {
                            "no_yard": 0.75,
                            "shared_yard": 0.85,
                            "private_yard": 0.90
                        },
                        "house_large": {
                            "no_yard": 0.85,
                            "shared_yard": 0.90,
                            "private_yard": 1.0
                        }
                    },
                    "Giant": {
                        "apartment": {
                            "no_yard": 0.65,      
                            "shared_yard": 0.75,
                            "private_yard": 0.80
                        },
                        "house_small": {
                            "no_yard": 0.70,
                            "shared_yard": 0.80,
                            "private_yard": 0.85
                        },
                        "house_large": {
                            "no_yard": 0.80,
                            "shared_yard": 0.90,
                            "private_yard": 1.0
                        }
                    }
                }
                
                yard_type = "private_yard" if has_yard else "no_yard"
                return base_matrix.get(size, base_matrix["Medium"])[living_space][yard_type]
        
            def calculate_exercise_adjustment():
                # 運動需求對空間評分的影響
                exercise_impact = {
                    "Very High": {
                        "apartment": -0.10,    
                        "house_small": -0.05,
                        "house_large": 0
                    },
                    "High": {
                        "apartment": -0.08,
                        "house_small": -0.05,
                        "house_large": 0
                    },
                    "Moderate": {
                        "apartment": -0.5,
                        "house_small": -0.02,
                        "house_large": 0
                    },
                    "Low": {
                        "apartment": 0.10,     
                        "house_small": 0.05,
                        "house_large": 0
                    }
                }
                
                return exercise_impact.get(exercise_needs, exercise_impact["Moderate"])[living_space]
        
            def calculate_yard_bonus():
                # 院子效益評估更加細緻
                if not has_yard:
                    return 0
                    
                yard_benefits = {
                    "Giant": {
                        "Very High": 0.25,
                        "High": 0.20,
                        "Moderate": 0.15,
                        "Low": 0.10
                    },
                    "Large": {
                        "Very High": 0.20,
                        "High": 0.15,
                        "Moderate": 0.10,
                        "Low": 0.05
                    },
                    "Medium": {
                        "Very High": 0.15,
                        "High": 0.10,
                        "Moderate": 0.08,
                        "Low": 0.05
                    },
                    "Small": {
                        "Very High": 0.10,
                        "High": 0.08,
                        "Moderate": 0.05,
                        "Low": 0.03
                    }
                }
                
                size_benefits = yard_benefits.get(size, yard_benefits["Medium"])
                return size_benefits.get(exercise_needs, size_benefits["Moderate"])
        
            def apply_extreme_case_adjustments(score):
                # 處理極端情況
                if size == "Giant" and living_space == "apartment":
                    return score * 0.85  
                
                if size == "Large" and living_space == "apartment" and exercise_needs == "Very High":
                    return score * 0.85  
                    
                if size == "Small" and living_space == "house_large" and exercise_needs == "Low":
                    return score * 0.9  # 低運動需求的小型犬在大房子可能過於寬敞
                    
                return score
        
            # 計算最終分數
            base_score = get_base_score()
            exercise_adj = calculate_exercise_adjustment()
            yard_bonus = calculate_yard_bonus()
            
            # 整合所有評分因素
            initial_score = base_score + exercise_adj + yard_bonus
            
            # 應用極端情況調整
            final_score = apply_extreme_case_adjustments(initial_score)
            
            # 確保分數在有效範圍內，但允許更極端的結果
            return max(0.05, min(1.0, final_score))


        def calculate_exercise_score(breed_needs: str, exercise_time: int, exercise_type: str, breed_size: str, living_space: str) -> float:
            """
            計算品種運動需求與使用者運動條件的匹配度。此函數特別著重：
            1. 不同品種的運動耐受度差異
            2. 運動時間與類型的匹配度
            3. 極端運動量的嚴格限制
            
            Parameters:
            breed_needs: 品種的運動需求等級
            exercise_time: 使用者計劃的運動時間（分鐘）
            exercise_type: 運動類型（輕度/中度/高度）
            
            Returns:
            float: 0.1到1.0之間的匹配分數
            """
            # 定義每個運動需求等級的具體參數
            exercise_levels = {
                'VERY HIGH': {
                    'min': 120,          # 最低需求
                    'ideal': 150,        # 理想運動量
                    'max': 180,          # 最大建議量
                    'type_weights': {    # 不同運動類型的權重
                        'active_training': 1.0,
                        'moderate_activity': 0.6,
                        'light_walks': 0.3
                    }
                },
                'HIGH': {
                    'min': 90,
                    'ideal': 120,
                    'max': 150,
                    'type_weights': {
                        'active_training': 0.9,
                        'moderate_activity': 0.8,
                        'light_walks': 0.4
                    }
                },
                'MODERATE': {
                    'min': 45,
                    'ideal': 60,
                    'max': 90,
                    'type_weights': {
                        'active_training': 0.7,
                        'moderate_activity': 1.0,
                        'light_walks': 0.8
                    }
                },
                'LOW': {
                    'min': 15,
                    'ideal': 30,
                    'max': 45,
                    'type_weights': {
                        'active_training': 0.5,
                        'moderate_activity': 0.8,
                        'light_walks': 1.0
                    }
                }
            }
        
            # 獲取品種的運動參數
            breed_level = exercise_levels.get(breed_needs.upper(), exercise_levels['MODERATE'])
            
            # 計算時間匹配度
            def calculate_time_score():
                """計算運動時間的匹配度，特別處理過度運動的情況"""
                if exercise_time < breed_level['min']:
                    # 運動不足的嚴格懲罰
                    deficit_ratio = exercise_time / breed_level['min']
                    return max(0.1, deficit_ratio * 0.4)
                
                elif exercise_time <= breed_level['ideal']:
                    # 理想範圍內的漸進提升
                    progress = (exercise_time - breed_level['min']) / (breed_level['ideal'] - breed_level['min'])
                    return 0.6 + (progress * 0.4)
                
                elif exercise_time <= breed_level['max']:
                    # 理想到最大範圍的平緩下降
                    excess_ratio = (exercise_time - breed_level['ideal']) / (breed_level['max'] - breed_level['ideal'])
                    return 1.0 - (excess_ratio * 0.2)
                
                else:
                    # 過度運動的顯著懲罰
                    excess = (exercise_time - breed_level['max']) / breed_level['max']
                    # 低運動需求品種的過度運動懲罰更嚴重
                    penalty_factor = 1.5 if breed_needs.upper() == 'LOW' else 1.0
                    return max(0.1, 0.8 - (excess * 0.5 * penalty_factor))
        
            # 計算運動類型匹配度
            def calculate_type_score():
                """評估運動類型的適合度，考慮品種特性"""
                base_type_score = breed_level['type_weights'].get(exercise_type, 0.5)
                
                # 特殊情況處理
                if breed_needs.upper() == 'LOW' and exercise_type == 'active_training':
                    # 低運動需求品種不適合高強度運動
                    base_type_score *= 0.5
                elif breed_needs.upper() == 'VERY HIGH' and exercise_type == 'light_walks':
                    # 高運動需求品種需要更多強度
                    base_type_score *= 0.6
                    
                return base_type_score
        
            # 計算最終分數
            time_score = calculate_time_score()
            type_score = calculate_type_score()
            
            # 根據運動需求等級調整權重
            if breed_needs.upper() == 'LOW':
                # 低運動需求品種更重視運動類型的合適性
                final_score = (time_score * 0.6) + (type_score * 0.4)
            elif breed_needs.upper() == 'VERY HIGH':
                # 高運動需求品種更重視運動時間的充足性
                final_score = (time_score * 0.7) + (type_score * 0.3)
            else:
                final_score = (time_score * 0.65) + (type_score * 0.35)

            if breed_info['Size'] in ['Large', 'Giant'] and user_prefs.living_space == 'apartment':
                if exercise_time >= 120:
                    final_score = min(1.0, final_score * 1.2)  
        
            # 極端情況的最終調整
            if breed_needs.upper() == 'LOW' and exercise_time > breed_level['max'] * 2:
                # 低運動需求品種的過度運動顯著降分
                final_score *= 0.6
            elif breed_needs.upper() == 'VERY HIGH' and exercise_time < breed_level['min'] * 0.5:
                # 高運動需求品種運動嚴重不足降分
                final_score *= 0.5
        
            return max(0.1, min(1.0, final_score))


        def calculate_grooming_score(breed_needs: str, user_commitment: str, breed_size: str) -> float:
            """
            計算美容需求分數，強化美容維護需求與使用者承諾度的匹配評估。
            這個函數特別注意品種大小對美容工作的影響，以及不同程度的美容需求對時間投入的要求。
            """
            # 重新設計基礎分數矩陣，讓美容需求的差異更加明顯
            base_scores = {
                "High": {
                    "low": 0.20,      # 高需求對低承諾極不合適，顯著降低初始分數
                    "medium": 0.65,   # 中等承諾仍有挑戰
                    "high": 1.0       # 高承諾最適合
                },
                "Moderate": {
                    "low": 0.45,      # 中等需求對低承諾有困難
                    "medium": 0.85,   # 較好的匹配
                    "high": 0.95      # 高承諾會有餘力
                },
                "Low": {
                    "low": 0.90,      # 低需求對低承諾很合適
                    "medium": 0.85,   # 略微降低以反映可能過度投入
                    "high": 0.80      # 可能造成資源浪費
                }
            }
        
            # 取得基礎分數
            base_score = base_scores.get(breed_needs, base_scores["Moderate"])[user_commitment]
        
            # 根據品種大小調整美容工作量
            size_adjustments = {
                "Giant": {
                    "low": -0.20,     # 大型犬的美容工作量顯著增加
                    "medium": -0.10,
                    "high": -0.05
                },
                "Large": {
                    "low": -0.15,
                    "medium": -0.05,
                    "high": 0
                },
                "Medium": {
                    "low": -0.10,
                    "medium": -0.05,
                    "high": 0
                },
                "Small": {
                    "low": -0.05,
                    "medium": 0,
                    "high": 0
                }
            }
        
            # 應用體型調整
            size_adjustment = size_adjustments.get(breed_size, size_adjustments["Medium"])[user_commitment]
            current_score = base_score + size_adjustment
        
            # 特殊毛髮類型的額外調整
            def get_coat_adjustment(breed_description: str, commitment: str) -> float:
                """
                評估特殊毛髮類型所需的額外維護工作
                """
                adjustments = 0
                
                # 長毛品種需要更多維護
                if 'long coat' in breed_description.lower():
                    coat_penalties = {
                        'low': -0.20,
                        'medium': -0.15,
                        'high': -0.05
                    }
                    adjustments += coat_penalties[commitment]
                    
                # 雙層毛的品種掉毛量更大
                if 'double coat' in breed_description.lower():
                    double_coat_penalties = {
                        'low': -0.15,
                        'medium': -0.10,
                        'high': -0.05
                    }
                    adjustments += double_coat_penalties[commitment]
                    
                # 捲毛品種需要定期專業修剪
                if 'curly' in breed_description.lower():
                    curly_penalties = {
                        'low': -0.15,
                        'medium': -0.10,
                        'high': -0.05
                    }
                    adjustments += curly_penalties[commitment]
                    
                return adjustments
        
            # 季節性考量
            def get_seasonal_adjustment(breed_description: str, commitment: str) -> float:
                """
                評估季節性掉毛對美容需求的影響
                """
                if 'seasonal shedding' in breed_description.lower():
                    seasonal_penalties = {
                        'low': -0.15,
                        'medium': -0.10,
                        'high': -0.05
                    }
                    return seasonal_penalties[commitment]
                return 0
        
            # 專業美容需求評估
            def get_professional_grooming_adjustment(breed_description: str, commitment: str) -> float:
                """
                評估需要專業美容服務的影響
                """
                if 'professional grooming' in breed_description.lower():
                    grooming_penalties = {
                        'low': -0.20,
                        'medium': -0.15,
                        'high': -0.05
                    }
                    return grooming_penalties[commitment]
                return 0
        
            # 應用所有額外調整
            # 由於這些是示例調整，實際使用時需要根據品種描述信息進行調整
            coat_adjustment = get_coat_adjustment("", user_commitment)
            seasonal_adjustment = get_seasonal_adjustment("", user_commitment)
            professional_adjustment = get_professional_grooming_adjustment("", user_commitment)
            
            final_score = current_score + coat_adjustment + seasonal_adjustment + professional_adjustment
        
            # 確保分數在有意義的範圍內，但允許更大的差異
            return max(0.1, min(1.0, final_score))


        def calculate_experience_score(care_level: str, user_experience: str, temperament: str) -> float:
            """
            計算使用者經驗與品種需求的匹配分數，更平衡的經驗等級影響
            
            改進重點：
            1. 提高初學者的基礎分數
            2. 縮小經驗等級間的差距
            3. 保持適度的區分度
            """
            # 基礎分數矩陣 - 更合理的分數分配
            base_scores = {
                "High": {
                    "beginner": 0.55,      # 提高起始分，讓新手也有機會
                    "intermediate": 0.80,   # 中級玩家有不錯的勝任能力
                    "advanced": 0.95        # 資深者幾乎完全勝任
                },
                "Moderate": {
                    "beginner": 0.65,      # 適中難度對新手更友善
                    "intermediate": 0.85,   # 中級玩家相當適合
                    "advanced": 0.90        # 資深者完全勝任
                },
                "Low": {
                    "beginner": 0.85,      # 新手友善品種維持高分
                    "intermediate": 0.90,   # 中級玩家幾乎完全勝任
                    "advanced": 0.90        # 資深者完全勝任
                }
            }
            
            # 取得基礎分數
            score = base_scores.get(care_level, base_scores["Moderate"])[user_experience]
            
            # 性格評估的權重也需要調整
            temperament_lower = temperament.lower()
            temperament_adjustments = 0.0
            
            # 根據經驗等級設定不同的特徵評估標準，降低懲罰程度
            if user_experience == "beginner":
                difficult_traits = {
                    'stubborn': -0.15,        # 降低懲罰程度
                    'independent': -0.12,
                    'dominant': -0.12,
                    'strong-willed': -0.10,
                    'protective': -0.10,
                    'aloof': -0.08,
                    'energetic': -0.08,
                    'aggressive': -0.20        # 保持較高懲罰，因為安全考慮
                }
                
                easy_traits = {
                    'gentle': 0.08,           # 提高獎勵以平衡
                    'friendly': 0.08,
                    'eager to please': 0.10,
                    'patient': 0.08,
                    'adaptable': 0.08,
                    'calm': 0.08
                }
                
                # 計算特徵調整
                for trait, penalty in difficult_traits.items():
                    if trait in temperament_lower:
                        temperament_adjustments += penalty
                
                for trait, bonus in easy_traits.items():
                    if trait in temperament_lower:
                        temperament_adjustments += bonus
                        
                # 品種類型特殊評估，降低懲罰程度
                if 'terrier' in temperament_lower:
                    temperament_adjustments -= 0.10  # 降低懲罰
                elif 'working' in temperament_lower:
                    temperament_adjustments -= 0.12
                elif 'guard' in temperament_lower:
                    temperament_adjustments -= 0.12
                    
            # 中級和高級玩家的調整保持不變...
            elif user_experience == "intermediate":
                moderate_traits = {
                    'stubborn': -0.08,
                    'independent': -0.05,
                    'intelligent': 0.10,
                    'athletic': 0.08,
                    'versatile': 0.08,
                    'protective': -0.05
                }
                
                for trait, adjustment in moderate_traits.items():
                    if trait in temperament_lower:
                        temperament_adjustments += adjustment
                        
            else:  # advanced
                advanced_traits = {
                    'stubborn': 0.05,
                    'independent': 0.05,
                    'intelligent': 0.10,
                    'protective': 0.05,
                    'strong-willed': 0.05
                }
                
                for trait, bonus in advanced_traits.items():
                    if trait in temperament_lower:
                        temperament_adjustments += bonus
            
            # 確保最終分數範圍合理
            final_score = max(0.15, min(1.0, score + temperament_adjustments))
            
            return final_score

        def calculate_health_score(breed_name: str, user_prefs: UserPreferences) -> float:
            """
            計算品種健康分數，加強健康問題的影響力和與使用者敏感度的連結
  
            1. 根據使用者的健康敏感度調整分數
            2. 更嚴格的健康問題評估
            3. 考慮多重健康問題的累積效應
            4. 加入遺傳疾病的特別考量
            """
            if breed_name not in breed_health_info:
                return 0.5
        
            health_notes = breed_health_info[breed_name]['health_notes'].lower()
            
            # 嚴重健康問題 - 加重扣分
            severe_conditions = {
                'hip dysplasia': -0.20,           # 髖關節發育不良，影響生活品質
                'heart disease': -0.15,           # 心臟疾病，需要長期治療
                'progressive retinal atrophy': -0.15,  # 進行性視網膜萎縮，導致失明
                'bloat': -0.18,                   # 胃扭轉，致命風險
                'epilepsy': -0.15,                # 癲癇，需要長期藥物控制
                'degenerative myelopathy': -0.15,  # 脊髓退化，影響行動能力
                'von willebrand disease': -0.12    # 血液凝固障礙
            }
            
            # 中度健康問題 - 適度扣分
            moderate_conditions = {
                'allergies': -0.12,               # 過敏問題，需要持續關注
                'eye problems': -0.15,            # 眼睛問題，可能需要手術
                'joint problems': -0.15,          # 關節問題，影響運動能力
                'hypothyroidism': -0.12,          # 甲狀腺功能低下，需要藥物治療
                'ear infections': -0.10,          # 耳道感染，需要定期清理
                'skin issues': -0.12              # 皮膚問題，需要特殊護理
            }
            
            # 輕微健康問題 - 輕微扣分
            minor_conditions = {
                'dental issues': -0.08,           # 牙齒問題，需要定期護理
                'weight gain tendency': -0.08,    # 易胖體質，需要控制飲食
                'minor allergies': -0.06,         # 輕微過敏，可控制
                'seasonal allergies': -0.06       # 季節性過敏
            }
        
            # 計算基礎健康分數
            health_score = 1.0
            
            # 健康問題累積效應計算
            condition_counts = {
                'severe': 0,
                'moderate': 0,
                'minor': 0
            }
            
            # 計算各等級健康問題的數量和影響
            for condition, penalty in severe_conditions.items():
                if condition in health_notes:
                    health_score += penalty
                    condition_counts['severe'] += 1
                    
            for condition, penalty in moderate_conditions.items():
                if condition in health_notes:
                    health_score += penalty
                    condition_counts['moderate'] += 1
                    
            for condition, penalty in minor_conditions.items():
                if condition in health_notes:
                    health_score += penalty
                    condition_counts['minor'] += 1
            
            # 多重問題的額外懲罰（累積效應）
            if condition_counts['severe'] > 1:
                health_score *= (0.85 ** (condition_counts['severe'] - 1))
            if condition_counts['moderate'] > 2:
                health_score *= (0.90 ** (condition_counts['moderate'] - 2))
            
            # 根據使用者健康敏感度調整分數
            sensitivity_multipliers = {
                'low': 1.1,      # 較不在意健康問題
                'medium': 1.0,   # 標準評估
                'high': 0.85     # 非常注重健康問題
            }
            
            health_score *= sensitivity_multipliers.get(user_prefs.health_sensitivity, 1.0)
        
            # 壽命影響評估
            try:
                lifespan = breed_health_info[breed_name].get('average_lifespan', '10-12')
                years = float(lifespan.split('-')[0])
                if years < 8:
                    health_score *= 0.85   # 短壽命顯著降低分數
                elif years < 10:
                    health_score *= 0.92   # 較短壽命輕微降低分數
                elif years > 13:
                    health_score *= 1.1    # 長壽命適度加分
            except:
                pass
        
            # 特殊健康優勢
            if 'generally healthy' in health_notes or 'hardy breed' in health_notes:
                health_score *= 1.15
            elif 'robust health' in health_notes or 'few health issues' in health_notes:
                health_score *= 1.1
        
            # 確保分數在合理範圍內，但允許更大的分數差異
            return max(0.1, min(1.0, health_score))
            

        def calculate_noise_score(breed_name: str, user_prefs: UserPreferences) -> float:
            """
            計算品種噪音分數，特別加強噪音程度與生活環境的關聯性評估，很多人棄養就是因為叫聲
            """
            if breed_name not in breed_noise_info:
                return 0.5
        
            noise_info = breed_noise_info[breed_name]
            noise_level = noise_info['noise_level'].lower()
            noise_notes = noise_info['noise_notes'].lower()
        
            # 重新設計基礎噪音分數矩陣，考慮不同情境下的接受度
            base_scores = {
                'low': {
                    'low': 1.0,       # 安靜的狗對低容忍完美匹配
                    'medium': 0.95,   # 安靜的狗對一般容忍很好
                    'high': 0.90      # 安靜的狗對高容忍當然可以
                },
                'medium': {
                    'low': 0.60,      # 一般吠叫對低容忍較困難
                    'medium': 0.90,   # 一般吠叫對一般容忍可接受
                    'high': 0.95      # 一般吠叫對高容忍很好
                },
                'high': {
                    'low': 0.25,      # 愛叫的狗對低容忍極不適合
                    'medium': 0.65,   # 愛叫的狗對一般容忍有挑戰
                    'high': 0.90      # 愛叫的狗對高容忍可以接受
                },
                'varies': {
                    'low': 0.50,      # 不確定的情況對低容忍風險較大
                    'medium': 0.75,   # 不確定的情況對一般容忍可嘗試
                    'high': 0.85      # 不確定的情況對高容忍問題較小
                }
            }
        
            # 取得基礎分數
            base_score = base_scores.get(noise_level, {'low': 0.6, 'medium': 0.75, 'high': 0.85})[user_prefs.noise_tolerance]
        
            # 吠叫原因評估，根據環境調整懲罰程度
            barking_penalties = {
                'separation anxiety': {
                    'apartment': -0.30,    # 在公寓對鄰居影響更大
                    'house_small': -0.25,
                    'house_large': -0.20
                },
                'excessive barking': {
                    'apartment': -0.25,
                    'house_small': -0.20,
                    'house_large': -0.15
                },
                'territorial': {
                    'apartment': -0.20,    # 在公寓更容易被觸發
                    'house_small': -0.15,
                    'house_large': -0.10
                },
                'alert barking': {
                    'apartment': -0.15,    # 公寓環境刺激較多
                    'house_small': -0.10,
                    'house_large': -0.08
                },
                'attention seeking': {
                    'apartment': -0.15,
                    'house_small': -0.12,
                    'house_large': -0.10
                }
            }
        
            # 計算環境相關的吠叫懲罰
            living_space = user_prefs.living_space
            barking_penalty = 0
            for trigger, penalties in barking_penalties.items():
                if trigger in noise_notes:
                    barking_penalty += penalties.get(living_space, -0.15)
        
            # 特殊情況評估
            special_adjustments = 0
            if user_prefs.has_children:
                # 孩童年齡相關調整
                child_age_adjustments = {
                    'toddler': {
                        'high': -0.20,     # 幼童對吵鬧更敏感
                        'medium': -0.15,
                        'low': -0.05
                    },
                    'school_age': {
                        'high': -0.15,
                        'medium': -0.10,
                        'low': -0.05
                    },
                    'teenager': {
                        'high': -0.10,
                        'medium': -0.05,
                        'low': -0.02
                    }
                }
                
                # 根據孩童年齡和噪音等級調整
                age_adj = child_age_adjustments.get(user_prefs.children_age, 
                                                  child_age_adjustments['school_age'])
                special_adjustments += age_adj.get(noise_level, -0.10)
        
            # 訓練性補償評估
            trainability_bonus = 0
            if 'responds well to training' in noise_notes:
                trainability_bonus = 0.12
            elif 'can be trained' in noise_notes:
                trainability_bonus = 0.08
            elif 'difficult to train' in noise_notes:
                trainability_bonus = 0.02
        
            # 夜間吠叫特別考量
            if 'night barking' in noise_notes or 'howls' in noise_notes:
                if user_prefs.living_space == 'apartment':
                    special_adjustments -= 0.15
                elif user_prefs.living_space == 'house_small':
                    special_adjustments -= 0.10
                else:
                    special_adjustments -= 0.05
        
            # 計算最終分數，確保更大的分數範圍
            final_score = base_score + barking_penalty + special_adjustments + trainability_bonus
            return max(0.1, min(1.0, final_score))
            

        # 1. 計算基礎分數
        print("\n=== 開始計算品種相容性分數 ===")
        print(f"處理品種: {breed_info.get('Breed', 'Unknown')}")
        print(f"品種信息: {breed_info}")
        print(f"使用者偏好: {vars(user_prefs)}")

        # 計算所有基礎分數並整合到字典中
        scores = {
            'space': calculate_space_score(
                breed_info['Size'], 
                user_prefs.living_space,
                user_prefs.yard_access != 'no_yard',
                breed_info.get('Exercise Needs', 'Moderate')
            ),
            'exercise': calculate_exercise_score(
                breed_info.get('Exercise Needs', 'Moderate'),
                user_prefs.exercise_time,
                user_prefs.exercise_type,
                breed_info['Size'],
                user_prefs.living_space
            ),
            'grooming': calculate_grooming_score(
                breed_info.get('Grooming Needs', 'Moderate'),
                user_prefs.grooming_commitment.lower(),
                breed_info['Size']
            ),
            'experience': calculate_experience_score(
                breed_info.get('Care Level', 'Moderate'),
                user_prefs.experience_level,
                breed_info.get('Temperament', '')
            ),
            'health': calculate_health_score(
                breed_info.get('Breed', ''),
                user_prefs
            ),
            'noise': calculate_noise_score(
                breed_info.get('Breed', ''),
                user_prefs
            )
        }

        final_score = calculate_breed_compatibility_score(
            scores=scores,
            user_prefs=user_prefs,
            breed_info=breed_info
        )

        # 計算環境適應性加成
        adaptability_bonus = calculate_environmental_fit(breed_info, user_prefs)
        
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
        
        final_score = amplify_score_extreme(filtered_score)

        # 更新並返回完整的評分結果
        scores.update({
            'overall': final_score,
            'size': breed_info['Size'],
            'adaptability_bonus': adaptability_bonus
        })

        return scores

    except Exception as e:
        print(f"\n!!!!! 發生嚴重錯誤 !!!!!")
        print(f"錯誤類型: {type(e).__name__}")
        print(f"錯誤訊息: {str(e)}")
        print(f"完整錯誤追蹤:")
        print(traceback.format_exc())
        return {k: 0.6 for k in ['space', 'exercise', 'grooming', 'experience', 'health', 'noise', 'overall']}


def calculate_environmental_fit(breed_info: dict, user_prefs: UserPreferences) -> float:
    """計算品種與環境的適應性加成"""
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
    

def calculate_breed_compatibility_score(scores: dict, user_prefs: UserPreferences, breed_info: dict) -> float:
    """
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
    
        def calculate_weight_adjustments(extremities):
            """
            1. 高運動量時對耐力型犬種的偏好
            2. 專家級別對工作犬種的偏好
            3. 條件組合的整體評估
            """
            adjustments = {}
            temperament = breed_info.get('Temperament', '').lower()
            is_working_dog = any(trait in temperament 
                                for trait in ['herding', 'working', 'intelligent', 'tireless'])
            
            # 空間權重調整邏輯保持不變
            if extremities['space'][0] == 'highly_restricted':
                if extremities['exercise'][0] in ['high', 'extremely_high']:
                    adjustments['space'] = 1.8  # 降低空間限制的權重
                    adjustments['exercise'] = 2.5  # 提高運動能力的權重
                else:
                    adjustments['space'] = 2.5
                    adjustments['noise'] = 2.0
            elif extremities['space'][0] == 'restricted':
                adjustments['space'] = 1.8
                adjustments['noise'] = 1.5
            elif extremities['space'][0] == 'spacious':
                adjustments['space'] = 0.8
                adjustments['exercise'] = 1.4
            
            # 改進運動需求權重調整
            if extremities['exercise'][0] in ['high', 'extremely_high']:
                # 提高運動量高時的基礎分數
                base_exercise_adjustment = 2.2
                if user_prefs.living_space == 'apartment':
                    base_exercise_adjustment = 2.5  # 特別獎勵公寓住戶的高運動量
                adjustments['exercise'] = base_exercise_adjustment
            if extremities['exercise'][0] in ['extremely_low', 'extremely_high']:
                base_adjustment = 2.5
                if extremities['exercise'][0] == 'extremely_high':
                    if is_working_dog:
                        base_adjustment = 3.0  # 工作犬在高運動量時獲得更高權重
                adjustments['exercise'] = base_adjustment
            elif extremities['exercise'][0] in ['low', 'high']:
                adjustments['exercise'] = 1.8
            
            # 改進經驗需求權重調整
            if extremities['experience'][0] == 'low':
                adjustments['experience'] = 2.2
                if breed_info.get('Care Level') == 'HIGH':
                    adjustments['experience'] = 2.5
            elif extremities['experience'][0] == 'high':
                if is_working_dog:
                    adjustments['experience'] = 2.5  # 提高專家對工作犬的權重
                    if extremities['exercise'][0] in ['high', 'extremely_high']:
                        adjustments['experience'] = 2.8  # 特別強化高運動量工作犬
                else:
                    adjustments['experience'] = 1.8
            
            # 綜合條件影響
            def adjust_for_combinations():
                # 保持原有的基礎邏輯
                if (extremities['space'][0] == 'highly_restricted' and 
                    extremities['exercise'][0] in ['high', 'extremely_high']):
                    adjustments['space'] = adjustments.get('space', 1.0) * 1.3
                    adjustments['exercise'] = adjustments.get('exercise', 1.0) * 1.3
                
                # 新增：專家 + 大空間 + 高運動量 + 工作犬的組合
                if (extremities['experience'][0] == 'high' and 
                    extremities['space'][0] == 'spacious' and
                    extremities['exercise'][0] in ['high', 'extremely_high'] and
                    is_working_dog):
                    adjustments['exercise'] = adjustments.get('exercise', 1.0) * 1.4
                    adjustments['experience'] = adjustments.get('experience', 1.0) * 1.4
                
                if extremities['space'][0] == 'spacious':
                    for key in ['grooming', 'health', 'noise']:
                        if key not in adjustments:
                            adjustments[key] = 1.2

            def ensure_minimum_score(score):
                if all([
                    extremities['exercise'][0] in ['high', 'extremely_high'],
                    breed_matches_exercise_needs(),  # 檢查品種是否適合該運動量
                    score < 0.85
                ]):
                    return 0.85
                return score
            
            adjust_for_combinations()
            return adjustments
    
        # 獲取條件極端度
        extremities = analyze_condition_extremity()
        
        # 計算權重調整
        weight_adjustments = calculate_weight_adjustments(extremities)
        
        # 應用權重調整
        final_weights = base_weights.copy()
        for key, adjustment in weight_adjustments.items():
            if key in final_weights:
                final_weights[key] *= adjustment
                
        return final_weights

    def apply_special_case_adjustments(score: float) -> float:
        """
        處理特殊情況和極端案例的評分調整。這個函數特別關注：
        1. 條件組合的協同效應
        2. 品種特性的獨特需求
        3. 極端情況的合理處理
        
        這個函數就像是一個細心的裁判，會考慮到各種特殊情況，
        並根據具體場景做出合理的評分調整。
        
        Parameters:
            score: 初始評分
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
        """
        評估不同條件間的相互影響，更寬容地處理極端組合
        """
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
        """
        計算完美匹配獎勵，但更注重條件的整體表現。
        """
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
        """
        根據品種特性進行最終調整。
        考慮品種的特殊性質和限制因素。
        """
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


def amplify_score_extreme(score: float) -> float:
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
        import math
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

    return round(min(1.0, max(0.0, score)), 4)