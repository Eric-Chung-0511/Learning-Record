import traceback
from typing import Dict, Any
from breed_health_info import breed_health_info
from breed_noise_info import breed_noise_info

class DimensionScoreCalculator:
    """
    維度評分計算器類別
    負責計算各個維度的具體評分，包含空間、運動、美容、經驗、健康和噪音等維度
    """
    
    def __init__(self):
        """初始化維度評分計算器"""
        pass
    
    def calculate_space_score(self, size: str, living_space: str, has_yard: bool, exercise_needs: str) -> float:
        """
        計算空間適配性評分
        
        完整實現原始版本的空間計算邏輯，包含：
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
                    "apartment": -0.05,
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

    def calculate_exercise_score(self, breed_needs: str, exercise_time: int, exercise_type: str, breed_size: str, living_space: str, breed_info: dict = None) -> float:
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

        if breed_size in ['Large', 'Giant'] and living_space == 'apartment':
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

    def calculate_grooming_score(self, breed_needs: str, user_commitment: str, breed_size: str) -> float:
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
            """評估特殊毛髮類型所需的額外維護工作"""
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
            """評估季節性掉毛對美容需求的影響"""
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
            """評估需要專業美容服務的影響"""
            if 'professional grooming' in breed_description.lower():
                grooming_penalties = {
                    'low': -0.20,
                    'medium': -0.15,
                    'high': -0.05
                }
                return grooming_penalties[commitment]
            return 0

        # 應用所有額外調整
        coat_adjustment = get_coat_adjustment("", user_commitment)
        seasonal_adjustment = get_seasonal_adjustment("", user_commitment)
        professional_adjustment = get_professional_grooming_adjustment("", user_commitment)

        final_score = current_score + coat_adjustment + seasonal_adjustment + professional_adjustment

        # 確保分數在有意義的範圍內，但允許更大的差異
        return max(0.1, min(1.0, final_score))

    def calculate_experience_score(self, care_level: str, user_experience: str, temperament: str) -> float:
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

    def calculate_health_score(self, breed_name: str, health_sensitivity: str) -> float:
        """
        計算品種健康分數，加強健康問題的影響力和與使用者敏感度的連結

        1. 根據使用者的健康敏感度調整分數
        2. 更嚴格的健康問題評估
        3. 考慮多重健康問題的累積效應
        4. 加入遺傳疾病的特別考量
        """
        try:          
            if breed_name not in breed_health_info:
                return 0.5
        except ImportError:
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

        health_score *= sensitivity_multipliers.get(health_sensitivity, 1.0)

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

    def calculate_noise_score(self, breed_name: str, noise_tolerance: str, living_space: str, has_children: bool, children_age: str) -> float:
        """
        計算品種噪音分數，特別加強噪音程度與生活環境的關聯性評估，很多人棄養就是因為叫聲
        """
        try:           
            if breed_name not in breed_noise_info:
                return 0.5
        except ImportError:
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
        base_score = base_scores.get(noise_level, {'low': 0.6, 'medium': 0.75, 'high': 0.85})[noise_tolerance]

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
        barking_penalty = 0
        for trigger, penalties in barking_penalties.items():
            if trigger in noise_notes:
                barking_penalty += penalties.get(living_space, -0.15)

        # 特殊情況評估
        special_adjustments = 0
        if has_children:
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
            age_adj = child_age_adjustments.get(children_age,
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
            if living_space == 'apartment':
                special_adjustments -= 0.15
            elif living_space == 'house_small':
                special_adjustments -= 0.10
            else:
                special_adjustments -= 0.05

        # 計算最終分數，確保更大的分數範圍
        final_score = base_score + barking_penalty + special_adjustments + trainability_bonus
        return max(0.1, min(1.0, final_score))
