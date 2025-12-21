# %%writefile matching_score_calculator.py
import random
import hashlib
import numpy as np
import sqlite3
import re
import traceback
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
import torch
from sklearn.metrics.pairwise import cosine_similarity
from dog_database import get_dog_description
from breed_health_info import breed_health_info
from breed_noise_info import breed_noise_info
from scoring_calculation_system import UserPreferences, calculate_compatibility_score, UnifiedScoringSystem, calculate_unified_breed_scores
from query_understanding import QueryUnderstandingEngine, analyze_user_query
from constraint_manager import ConstraintManager, apply_breed_constraints
from multi_head_scorer import MultiHeadScorer, score_breed_candidates, BreedScore
from score_calibrator import ScoreCalibrator, calibrate_breed_scores
from config_manager import get_config_manager, get_standardized_breed_data

class MatchingScoreCalculator:
    """
    匹配評分計算器
    處理多維度匹配計算、約束條件過濾和評分校準
    """

    def __init__(self, breed_list: List[str]):
        """初始化匹配評分計算器"""
        self.breed_list = breed_list

    def apply_size_distribution_correction(self, recommendations: List[Dict]) -> List[Dict]:
        """應用尺寸分佈修正以防止大型品種偏差"""
        if len(recommendations) < 10:
            return recommendations

        # 分析尺寸分佈
        size_counts = {'toy': 0, 'small': 0, 'medium': 0, 'large': 0, 'giant': 0}

        for rec in recommendations:
            breed_info = get_dog_description(rec['breed'])
            if breed_info:
                size = self._normalize_breed_size(breed_info.get('Size', 'Medium'))
                size_counts[size] += 1

        total_recs = len(recommendations)
        large_giant_ratio = (size_counts['large'] + size_counts['giant']) / total_recs

        # 如果超過 70% 是大型/巨型品種，應用修正
        if large_giant_ratio > 0.7:
            corrected_recommendations = []
            size_quotas = {'toy': 2, 'small': 4, 'medium': 6, 'large': 2, 'giant': 1}
            current_counts = {'toy': 0, 'small': 0, 'medium': 0, 'large': 0, 'giant': 0}

            # 第一輪：在配額內添加品種
            for rec in recommendations:
                breed_info = get_dog_description(rec['breed'])
                if breed_info:
                    size = self._normalize_breed_size(breed_info.get('Size', 'Medium'))
                    if current_counts[size] < size_quotas[size]:
                        corrected_recommendations.append(rec)
                        current_counts[size] += 1

            # 第二輪：用最佳剩餘候選品種填滿剩餘位置
            remaining_slots = 15 - len(corrected_recommendations)
            remaining_breeds = [rec for rec in recommendations if rec not in corrected_recommendations]

            corrected_recommendations.extend(remaining_breeds[:remaining_slots])
            return corrected_recommendations

        return recommendations

    def _normalize_breed_size(self, size: str) -> str:
        """標準化品種尺寸到標準分類"""
        if not isinstance(size, str):
            return 'medium'

        size_lower = size.lower()
        if any(term in size_lower for term in ['toy', 'tiny']):
            return 'toy'
        elif 'small' in size_lower:
            return 'small'
        elif 'medium' in size_lower:
            return 'medium'
        elif 'large' in size_lower:
            return 'large'
        elif any(term in size_lower for term in ['giant', 'extra large']):
            return 'giant'
        else:
            return 'medium'

    def apply_hard_constraints(self, breed: str, user_input: str, breed_characteristics: Dict[str, Any]) -> float:
        """增強硬約束，具有更嚴格的懲罰"""
        penalty = 0.0
        user_text_lower = user_input.lower()

        # 獲取品種信息
        breed_info = get_dog_description(breed)
        if not breed_info:
            return 0.0

        breed_size = breed_info.get('Size', '').lower()
        exercise_needs = breed_info.get('Exercise Needs', '').lower()

        # 公寓居住約束 - 更嚴格
        if any(term in user_text_lower for term in ['apartment', 'flat', 'studio', 'small space']):
            if 'giant' in breed_size:
                return -2.0  # 完全淘汰
            elif 'large' in breed_size:
                if any(term in exercise_needs for term in ['high', 'very high']):
                    return -2.0  # 完全淘汰
                else:
                    penalty -= 0.5  # 仍有顯著懲罰
            elif 'medium' in breed_size and 'very high' in exercise_needs:
                penalty -= 0.6

        # 運動不匹配約束
        if "don't exercise much" in user_text_lower or "low exercise" in user_text_lower:
            if any(term in exercise_needs for term in ['very high', 'extreme', 'intense']):
                return -2.0  # 完全淘汰
            elif 'high' in exercise_needs:
                penalty -= 0.8

        # 中等生活方式檢測
        if any(term in user_text_lower for term in ['moderate', 'balanced', '30 minutes', 'half hour']):
            # 懲罰極端情況
            if 'giant' in breed_size:
                penalty -= 0.7  # 對巨型犬的強懲罰
            elif 'very high' in exercise_needs:
                penalty -= 0.5

        # 兒童安全（現有邏輯保持但增強）
        if any(term in user_text_lower for term in ['child', 'kids', 'family', 'baby']):
            good_with_children = breed_info.get('Good with Children', '').lower()
            if good_with_children == 'no':
                return -2.0  # 為了安全完全淘汰

        return penalty

    def calculate_lifestyle_bonus(self, breed_characteristics: Dict[str, Any],
                                 lifestyle_keywords: Dict[str, List[str]]) -> float:
        """增強生活方式匹配獎勵計算"""
        bonus = 0.0
        penalties = 0.0

        # 增強尺寸匹配
        breed_size = breed_characteristics.get('size', '').lower()
        size_prefs = lifestyle_keywords.get('size_preference', [])
        for pref in size_prefs:
            if pref in breed_size:
                bonus += 0.25  # 尺寸匹配的強獎勵
            elif (pref == 'small' and 'large' in breed_size) or \
                 (pref == 'large' and 'small' in breed_size):
                penalties += 0.15  # 尺寸不匹配的懲罰

        # 增強活動水平匹配
        breed_exercise = breed_characteristics.get('exercise_needs', '').lower()
        activity_prefs = lifestyle_keywords.get('activity_level', [])

        if 'high' in activity_prefs:
            if 'high' in breed_exercise or 'very high' in breed_exercise:
                bonus += 0.2
            elif 'low' in breed_exercise:
                penalties += 0.2
        elif 'low' in activity_prefs:
            if 'low' in breed_exercise:
                bonus += 0.2
            elif 'high' in breed_exercise or 'very high' in breed_exercise:
                penalties += 0.25
        elif 'moderate' in activity_prefs:
            if 'moderate' in breed_exercise:
                bonus += 0.15

        # 增強家庭情況匹配
        good_with_children = breed_characteristics.get('good_with_children', 'Yes')
        family_prefs = lifestyle_keywords.get('family_situation', [])

        if 'children' in family_prefs:
            if good_with_children == 'Yes':
                bonus += 0.15
            else:
                penalties += 0.3  # 對非兒童友好品種的強懲罰

        # 增強居住空間匹配
        living_prefs = lifestyle_keywords.get('living_space', [])
        if 'apartment' in living_prefs:
            if 'small' in breed_size:
                bonus += 0.2
            elif 'medium' in breed_size and 'low' in breed_exercise:
                bonus += 0.1
            elif 'large' in breed_size or 'giant' in breed_size:
                penalties += 0.2  # 公寓中大型犬的懲罰

        # 噪音偏好匹配
        noise_prefs = lifestyle_keywords.get('noise_preference', [])
        temperament = breed_characteristics.get('temperament', '').lower()

        if 'low' in noise_prefs:
            # 獎勵安靜品種
            if any(term in temperament for term in ['gentle', 'calm', 'quiet']):
                bonus += 0.1

        # 照護水平匹配
        grooming_needs = breed_characteristics.get('grooming_needs', '').lower()
        care_prefs = lifestyle_keywords.get('care_level', [])

        if 'low' in care_prefs and 'low' in grooming_needs:
            bonus += 0.1
        elif 'high' in care_prefs and 'high' in grooming_needs:
            bonus += 0.1
        elif 'low' in care_prefs and 'high' in grooming_needs:
            penalties += 0.15

        # 特殊需求匹配
        special_needs = lifestyle_keywords.get('special_needs', [])

        if 'guard' in special_needs:
            if any(term in temperament for term in ['protective', 'alert', 'watchful']):
                bonus += 0.1
        elif 'companion' in special_needs:
            if any(term in temperament for term in ['affectionate', 'gentle', 'loyal']):
                bonus += 0.1

        # 計算包含懲罰的最終獎勵
        final_bonus = bonus - penalties
        return max(-0.3, min(0.5, final_bonus))  # 允許負獎勵但限制範圍

    def apply_intelligent_trait_matching(self, recommendations: List[Dict], user_input: str) -> List[Dict]:
        """基於增強關鍵字提取和數據庫挖掘應用智能特徵匹配"""
        try:
            # 從用戶輸入提取增強關鍵字
            extracted_keywords = self._extract_enhanced_lifestyle_keywords(user_input)

            # 對每個推薦應用智能特徵匹配
            enhanced_recommendations = []

            for rec in recommendations:
                breed_name = rec['breed'].replace(' ', '_')

                # 獲取品種數據庫信息
                breed_info = get_dog_description(breed_name) or {}

                # 計算智能特徵獎勵
                intelligence_bonus = 0.0
                trait_match_details = {}

                # 1. 智力匹配
                if extracted_keywords.get('intelligence_preference'):
                    intelligence_pref = extracted_keywords['intelligence_preference'][0]
                    breed_desc = breed_info.get('Description', '').lower()

                    if intelligence_pref == 'high':
                        if any(word in breed_desc for word in ['intelligent', 'smart', 'clever', 'quick to learn', 'trainable']):
                            intelligence_bonus += 0.05
                            trait_match_details['intelligence_match'] = 'High intelligence match detected'
                        elif any(word in breed_desc for word in ['stubborn', 'independent', 'difficult']):
                            intelligence_bonus -= 0.02
                            trait_match_details['intelligence_warning'] = 'May be challenging to train'

                    elif intelligence_pref == 'independent':
                        if any(word in breed_desc for word in ['independent', 'stubborn', 'strong-willed']):
                            intelligence_bonus += 0.03
                            trait_match_details['independence_match'] = 'Independent nature match'

                # 2. 美容偏好匹配
                if extracted_keywords.get('grooming_preference'):
                    grooming_pref = extracted_keywords['grooming_preference'][0]
                    breed_grooming = breed_info.get('Grooming Needs', '').lower()

                    if grooming_pref == 'low' and 'low' in breed_grooming:
                        intelligence_bonus += 0.03
                        trait_match_details['grooming_match'] = 'Low maintenance grooming match'
                    elif grooming_pref == 'high' and 'high' in breed_grooming:
                        intelligence_bonus += 0.03
                        trait_match_details['grooming_match'] = 'High maintenance grooming match'
                    elif grooming_pref == 'low' and 'high' in breed_grooming:
                        intelligence_bonus -= 0.04
                        trait_match_details['grooming_mismatch'] = 'High grooming needs may not suit preferences'

                # 3. 氣質偏好匹配
                if extracted_keywords.get('temperament_preference'):
                    temp_prefs = extracted_keywords['temperament_preference']
                    breed_temperament = breed_info.get('Temperament', '').lower()
                    breed_desc = breed_info.get('Description', '').lower()

                    temp_text = (breed_temperament + ' ' + breed_desc).lower()

                    for temp_pref in temp_prefs:
                        if temp_pref == 'gentle' and any(word in temp_text for word in ['gentle', 'calm', 'peaceful', 'mild']):
                            intelligence_bonus += 0.04
                            trait_match_details['temperament_match'] = f'Gentle temperament match: {temp_pref}'
                        elif temp_pref == 'playful' and any(word in temp_text for word in ['playful', 'energetic', 'lively', 'fun']):
                            intelligence_bonus += 0.04
                            trait_match_details['temperament_match'] = f'Playful temperament match: {temp_pref}'
                        elif temp_pref == 'protective' and any(word in temp_text for word in ['protective', 'guard', 'alert', 'watchful']):
                            intelligence_bonus += 0.04
                            trait_match_details['temperament_match'] = f'Protective temperament match: {temp_pref}'
                        elif temp_pref == 'friendly' and any(word in temp_text for word in ['friendly', 'social', 'outgoing', 'people']):
                            intelligence_bonus += 0.04
                            trait_match_details['temperament_match'] = f'Friendly temperament match: {temp_pref}'

                # 4. 經驗水平匹配
                if extracted_keywords.get('experience_level'):
                    exp_level = extracted_keywords['experience_level'][0]
                    breed_desc = breed_info.get('Description', '').lower()

                    if exp_level == 'beginner':
                        # 為初學者偏愛易於處理的品種
                        if any(word in breed_desc for word in ['easy', 'gentle', 'good for beginners', 'family', 'calm']):
                            intelligence_bonus += 0.06
                            trait_match_details['beginner_friendly'] = 'Good choice for first-time owners'
                        elif any(word in breed_desc for word in ['challenging', 'dominant', 'requires experience', 'strong-willed']):
                            intelligence_bonus -= 0.08
                            trait_match_details['experience_warning'] = 'May be challenging for first-time owners'

                    elif exp_level == 'advanced':
                        # 高級用戶可以處理更具挑戰性的品種
                        if any(word in breed_desc for word in ['working', 'requires experience', 'intelligent', 'strong']):
                            intelligence_bonus += 0.03
                            trait_match_details['advanced_suitable'] = 'Good match for experienced owners'

                # 5. 壽命偏好匹配
                if extracted_keywords.get('lifespan_preference'):
                    lifespan_pref = extracted_keywords['lifespan_preference'][0]
                    breed_lifespan = breed_info.get('Lifespan', '10-12 years')

                    try:
                        import re
                        years = re.findall(r'\d+', breed_lifespan)
                        if years:
                            avg_years = sum(int(y) for y in years) / len(years)
                            if lifespan_pref == 'long' and avg_years >= 13:
                                intelligence_bonus += 0.02
                                trait_match_details['longevity_match'] = f'Long lifespan match: {breed_lifespan}'
                            elif lifespan_pref == 'healthy' and avg_years >= 12:
                                intelligence_bonus += 0.02
                                trait_match_details['health_match'] = f'Healthy lifespan: {breed_lifespan}'
                    except:
                        pass

                # 將智力獎勵應用到總分
                original_score = rec['overall_score']
                enhanced_score = min(1.0, original_score + intelligence_bonus)

                # 創建包含特徵匹配詳細信息的增強推薦
                enhanced_rec = rec.copy()
                enhanced_rec['overall_score'] = enhanced_score
                enhanced_rec['intelligence_bonus'] = intelligence_bonus
                enhanced_rec['trait_match_details'] = trait_match_details

                # 如果發生顯著增強，添加詳細說明
                if abs(intelligence_bonus) > 0.02:
                    enhancement_explanation = []
                    for detail_key, detail_value in trait_match_details.items():
                        enhancement_explanation.append(detail_value)

                    if enhancement_explanation:
                        current_explanation = enhanced_rec.get('explanation', '')
                        enhanced_explanation = current_explanation + f" Enhanced matching: {'; '.join(enhancement_explanation)}"
                        enhanced_rec['explanation'] = enhanced_explanation

                enhanced_recommendations.append(enhanced_rec)

            # 按增強總分重新排序
            enhanced_recommendations.sort(key=lambda x: x['overall_score'], reverse=True)

            # 更新排名
            for i, rec in enumerate(enhanced_recommendations):
                rec['rank'] = i + 1

            print(f"Applied intelligent trait matching with average bonus: {sum(r['intelligence_bonus'] for r in enhanced_recommendations) / len(enhanced_recommendations):.3f}")

            return enhanced_recommendations

        except Exception as e:
            print(f"Error in intelligent trait matching: {str(e)}")
            # 如果特徵匹配失敗，返回原始推薦
            return recommendations

    def _extract_enhanced_lifestyle_keywords(self, user_input: str) -> Dict[str, List[str]]:
        """提取增強的生活方式關鍵字（用於智能特徵匹配）"""
        keywords = {
            'intelligence_preference': [],
            'grooming_preference': [],
            'temperament_preference': [],
            'experience_level': [],
            'lifespan_preference': []
        }

        text = user_input.lower()

        # 智力偏好檢測
        smart_terms = ['smart', 'intelligent', 'clever', 'bright', 'quick learner', 'easy to train', 'trainable', 'genius', 'brilliant']
        independent_terms = ['independent', 'stubborn', 'strong-willed', 'less trainable', 'thinks for themselves']

        if any(term in text for term in smart_terms):
            keywords['intelligence_preference'].append('high')
        if any(term in text for term in independent_terms):
            keywords['intelligence_preference'].append('independent')

        # 美容偏好檢測
        low_grooming_terms = ['low grooming', 'minimal grooming', 'easy care', 'wash and wear', 'no grooming', 'simple coat']
        high_grooming_terms = ['high grooming', 'professional grooming', 'lots of care', 'high maintenance coat', 'daily brushing', 'regular grooming']

        if any(term in text for term in low_grooming_terms):
            keywords['grooming_preference'].append('low')
        if any(term in text for term in high_grooming_terms):
            keywords['grooming_preference'].append('high')

        # 氣質偏好檢測
        gentle_terms = ['gentle', 'calm', 'peaceful', 'laid back', 'chill', 'mellow', 'docile']
        playful_terms = ['playful', 'energetic', 'fun', 'active personality', 'lively', 'spirited', 'bouncy']
        protective_terms = ['protective', 'guard', 'watchdog', 'alert', 'vigilant', 'defensive']
        friendly_terms = ['friendly', 'social', 'outgoing', 'loves people', 'sociable', 'gregarious']

        if any(term in text for term in gentle_terms):
            keywords['temperament_preference'].append('gentle')
        if any(term in text for term in playful_terms):
            keywords['temperament_preference'].append('playful')
        if any(term in text for term in protective_terms):
            keywords['temperament_preference'].append('protective')
        if any(term in text for term in friendly_terms):
            keywords['temperament_preference'].append('friendly')

        # 經驗水平檢測
        beginner_terms = ['first time', 'beginner', 'new to dogs', 'never had', 'novice', 'inexperienced']
        advanced_terms = ['experienced', 'advanced', 'dog expert', 'many dogs before', 'professional', 'seasoned']

        if any(term in text for term in beginner_terms):
            keywords['experience_level'].append('beginner')
        if any(term in text for term in advanced_terms):
            keywords['experience_level'].append('advanced')

        # 壽命偏好檢測
        long_lived_terms = ['long lived', 'long lifespan', 'live long', 'many years', '15+ years', 'longevity']
        healthy_terms = ['healthy breed', 'few health issues', 'robust', 'hardy', 'strong constitution']

        if any(term in text for term in long_lived_terms):
            keywords['lifespan_preference'].append('long')
        if any(term in text for term in healthy_terms):
            keywords['lifespan_preference'].append('healthy')

        return keywords

    def calculate_enhanced_matching_score(self, breed: str, breed_info: dict, user_description: str, base_similarity: float) -> dict:
        """計算增強的匹配分數，基於用戶描述和品種特性"""
        try:
            user_desc = user_description.lower()

            # 分析用戶需求
            space_requirements = self._analyze_space_requirements(user_desc)
            exercise_requirements = self._analyze_exercise_requirements(user_desc)
            noise_requirements = self._analyze_noise_requirements(user_desc)
            size_requirements = self._analyze_size_requirements(user_desc)
            family_requirements = self._analyze_family_requirements(user_desc)

            # 獲取品種特性
            breed_size = breed_info.get('Size', '').lower()
            breed_exercise = breed_info.get('Exercise Needs', '').lower()
            breed_noise = breed_noise_info.get(breed, {}).get('noise_level', 'moderate').lower()
            breed_temperament = breed_info.get('Temperament', '').lower()
            breed_good_with_children = breed_info.get('Good with Children', '').lower()

            # 計算各維度匹配分數
            dimension_scores = {}

            # 空間匹配 (30% 權重)
            space_score = self._calculate_space_compatibility(space_requirements, breed_size, breed_exercise)
            dimension_scores['space'] = space_score

            # 運動需求匹配 (25% 權重)
            exercise_score = self._calculate_exercise_compatibility(exercise_requirements, breed_exercise)
            dimension_scores['exercise'] = exercise_score

            # 噪音匹配 (20% 權重)
            noise_score = self._calculate_noise_compatibility(noise_requirements, breed_noise)
            dimension_scores['noise'] = noise_score

            # 體型匹配 (15% 權重)
            size_score = self._calculate_size_compatibility(size_requirements, breed_size)
            dimension_scores['grooming'] = min(0.9, base_similarity + 0.1)  # 美容需求基於語意相似度

            # 家庭相容性 (10% 權重)
            family_score = self._calculate_family_compatibility(family_requirements, breed_good_with_children, breed_temperament)
            dimension_scores['family'] = family_score

            # 經驗相容性 - 使用真實品種特性計算
            experience_requirements = self._analyze_experience_requirements(user_desc)
            experience_score = self._calculate_experience_compatibility(
                experience_requirements, breed_info, breed_temperament
            )
            dimension_scores['experience'] = experience_score

            # 應用硬約束過濾
            constraint_penalty = self._apply_hard_constraints_enhanced(user_desc, breed_info)

            # 計算加權總分 - 精確化維度權重配置
            # 根據指導建議重新平衡維度權重
            weighted_score = (
                space_score * 0.30 +      # 空間相容性（降低5%）
                exercise_score * 0.28 +   # 運動需求匹配（降低2%）
                noise_score * 0.18 +      # 噪音控制（提升3%）
                family_score * 0.12 +     # 家庭相容性（提升2%）
                size_score * 0.08 +       # 體型匹配（降低2%）
                min(0.9, base_similarity + 0.1) * 0.04  # 護理需求（新增獨立權重）
            )

            # 優化完美匹配獎勵機制 - 降低觸發門檻並增加層次
            perfect_match_bonus = 0.0
            if space_score >= 0.88 and exercise_score >= 0.88 and noise_score >= 0.85:
                perfect_match_bonus = 0.08  # 卓越匹配獎勵
            elif space_score >= 0.82 and exercise_score >= 0.82 and noise_score >= 0.75:
                perfect_match_bonus = 0.04  # 優秀匹配獎勵
            elif space_score >= 0.75 and exercise_score >= 0.75:
                perfect_match_bonus = 0.02  # 良好匹配獎勵

            # 結合語意相似度與維度匹配 - 調整為75%維度匹配 25%語義相似度
            base_combined_score = (weighted_score * 0.75 + base_similarity * 0.25) + perfect_match_bonus

            # 應用漸進式約束懲罰，但確保基礎分數保障
            raw_final_score = base_combined_score + constraint_penalty

            # 實施動態分數保障機制 - 提升至40-42%基礎分數
            # 根據品種特性動態調整基礎分數
            base_guaranteed_score = 0.42  # 提升基礎保障分數

            # 特殊品種基礎分數調整
            high_adaptability_breeds = ['French_Bulldog', 'Pug', 'Golden_Retriever', 'Labrador_Retriever']
            if any(breed in breed for breed in high_adaptability_breeds):
                base_guaranteed_score = 0.45  # 高適應性品種更高基礎分數

            # 動態分數分佈優化
            if raw_final_score >= base_guaranteed_score:
                # 對於高分品種，實施適度壓縮避免過度集中
                if raw_final_score > 0.85:
                    compression_factor = 0.92  # 輕度壓縮高分
                    final_score = 0.85 + (raw_final_score - 0.85) * compression_factor
                else:
                    final_score = raw_final_score
                final_score = min(0.93, final_score)  # 降低最高分數限制
            else:
                # 對於低分品種，使用改進的保障機制
                normalized_raw_score = max(0.15, raw_final_score)
                # 基礎保障75% + 實際計算25%，保持一定區分度
                final_score = base_guaranteed_score * 0.75 + normalized_raw_score * 0.25
                final_score = max(base_guaranteed_score, min(0.93, final_score))

            lifestyle_bonus = max(0.0, weighted_score - base_similarity)

            return {
                'final_score': final_score,
                'weighted_score': weighted_score,
                'lifestyle_bonus': lifestyle_bonus,
                'dimension_scores': dimension_scores,
                'constraint_penalty': constraint_penalty
            }

        except Exception as e:
            print(f"Error in enhanced matching calculation for {breed}: {str(e)}")
            return {
                'final_score': base_similarity,
                'weighted_score': base_similarity,
                'lifestyle_bonus': 0.0,
                'dimension_scores': {
                    'space': base_similarity * 0.9,
                    'exercise': base_similarity * 0.85,
                    'grooming': base_similarity * 0.8,
                    'experience': base_similarity * 0.75,
                    'noise': base_similarity * 0.7,
                    'family': base_similarity * 0.65
                },
                'constraint_penalty': 0.0
            }

    def _analyze_space_requirements(self, user_desc: str) -> dict:
        """分析空間需求 - 增強中等活動量識別"""
        requirements = {'type': 'unknown', 'size': 'medium', 'importance': 0.5}

        if any(word in user_desc for word in ['apartment', 'small apartment', 'small space', 'condo', 'flat']):
            requirements['type'] = 'apartment'
            requirements['size'] = 'small'
            requirements['importance'] = 0.95  # 提高重要性
        elif any(word in user_desc for word in ['medium-sized house', 'medium house', 'townhouse']):
            requirements['type'] = 'medium_house'
            requirements['size'] = 'medium'
            requirements['importance'] = 0.8  # 中等活動量用戶的特殊標記
        elif any(word in user_desc for word in ['large house', 'big house', 'yard', 'garden', 'large space', 'backyard']):
            requirements['type'] = 'house'
            requirements['size'] = 'large'
            requirements['importance'] = 0.7

        return requirements

    def _analyze_exercise_requirements(self, user_desc: str) -> dict:
        """分析運動需求 - 增強中等活動量識別"""
        requirements = {'level': 'moderate', 'importance': 0.5}

        # 低運動量識別
        if any(word in user_desc for word in ["don't exercise", "don't exercise much", "low exercise", "minimal", "lazy", "not active"]):
            requirements['level'] = 'low'
            requirements['importance'] = 0.95
        # 中等運動量的精確識別
        elif any(phrase in user_desc for phrase in ['30 minutes', 'half hour', 'moderate', 'balanced', 'walk about']):
            if 'walk' in user_desc or 'daily' in user_desc:
                requirements['level'] = 'moderate'
                requirements['importance'] = 0.85  # 中等活動量的特殊標記
        # 高運動量識別
        elif any(word in user_desc for word in ['active', 'hiking', 'outdoor activities', 'running', 'outdoors', 'love hiking']):
            requirements['level'] = 'high'
            requirements['importance'] = 0.9

        return requirements

    def _analyze_noise_requirements(self, user_desc: str) -> dict:
        """分析噪音需求"""
        requirements = {'tolerance': 'medium', 'importance': 0.5}

        if any(word in user_desc for word in ['quiet', 'no bark', "won't bark", "doesn't bark", 'silent', 'peaceful']):
            requirements['tolerance'] = 'low'
            requirements['importance'] = 0.9
        elif any(word in user_desc for word in ['loud', 'barking ok', 'noise ok']):
            requirements['tolerance'] = 'high'
            requirements['importance'] = 0.7

        return requirements

    def _analyze_size_requirements(self, user_desc: str) -> dict:
        """分析體型需求"""
        requirements = {'preferred': 'any', 'importance': 0.5}

        if any(word in user_desc for word in ['small', 'tiny', 'little', 'lap dog', 'compact']):
            requirements['preferred'] = 'small'
            requirements['importance'] = 0.8
        elif any(word in user_desc for word in ['large', 'big', 'giant']):
            requirements['preferred'] = 'large'
            requirements['importance'] = 0.8

        return requirements

    def _analyze_family_requirements(self, user_desc: str) -> dict:
        """分析家庭需求"""
        requirements = {'children': False, 'importance': 0.3}

        if any(word in user_desc for word in ['children', 'kids', 'family', 'child']):
            requirements['children'] = True
            requirements['importance'] = 0.8

        return requirements

    def _calculate_space_compatibility(self, space_req: dict, breed_size: str, breed_exercise: str) -> float:
        """計算空間相容性分數 - 增強中等活動量處理"""
        if space_req['type'] == 'apartment':
            if 'small' in breed_size or 'toy' in breed_size:
                base_score = 0.95
            elif 'medium' in breed_size:
                if 'low' in breed_exercise:
                    base_score = 0.75
                else:
                    base_score = 0.45  # 降低中型犬在公寓的分數
            elif 'large' in breed_size:
                base_score = 0.05  # 大型犬極度不適合公寓
            elif 'giant' in breed_size:
                base_score = 0.01  # 超大型犬完全不適合公寓
            else:
                base_score = 0.7
        elif space_req['type'] == 'medium_house':
            # 中型房屋的特殊處理 - 適合中等活動量用戶
            if 'small' in breed_size or 'toy' in breed_size:
                base_score = 0.9
            elif 'medium' in breed_size:
                base_score = 0.95  # 中型犬在中型房屋很適合
            elif 'large' in breed_size:
                if 'moderate' in breed_exercise or 'low' in breed_exercise:
                    base_score = 0.8  # 低運動量大型犬還可以
                else:
                    base_score = 0.6  # 高運動量大型犬不太適合
            elif 'giant' in breed_size:
                base_score = 0.3  # 超大型犬在中型房屋不太適合
            else:
                base_score = 0.85
        else:
            # 大型房屋的情況
            if 'small' in breed_size or 'toy' in breed_size:
                base_score = 0.85
            elif 'medium' in breed_size:
                base_score = 0.9
            elif 'large' in breed_size or 'giant' in breed_size:
                base_score = 0.95
            else:
                base_score = 0.8

        return min(0.95, base_score)

    def _calculate_exercise_compatibility(self, exercise_req: dict, breed_exercise: str) -> float:
        """計算運動需求相容性分數 - 增強中等活動量處理"""
        if exercise_req['level'] == 'low':
            if 'low' in breed_exercise or 'minimal' in breed_exercise:
                return 0.95
            elif 'moderate' in breed_exercise:
                return 0.5  # 降低不匹配分數
            elif 'high' in breed_exercise:
                return 0.1  # 進一步降低高運動需求的匹配
            else:
                return 0.7
        elif exercise_req['level'] == 'high':
            if 'high' in breed_exercise:
                return 0.95
            elif 'moderate' in breed_exercise:
                return 0.8
            elif 'low' in breed_exercise:
                return 0.6
            else:
                return 0.7
        else:  # moderate - 中等活動量的精確處理
            if 'moderate' in breed_exercise:
                return 0.95  # 完美匹配
            elif 'low' in breed_exercise:
                return 0.85  # 低運動需求的品種對中等活動量用戶也不錯
            elif 'high' in breed_exercise:
                return 0.5  # 中等活動量用戶不太適合高運動需求品種
            else:
                return 0.75

        return 0.6

    def _calculate_noise_compatibility(self, noise_req: dict, breed_noise: str) -> float:
        """計算噪音相容性分數，更好處理複合等級"""
        breed_noise_lower = breed_noise.lower()

        if noise_req['tolerance'] == 'low':
            if 'low' in breed_noise_lower and 'moderate' not in breed_noise_lower:
                return 0.95  # 純低噪音
            elif 'low-moderate' in breed_noise_lower or 'low to moderate' in breed_noise_lower:
                return 0.8   # 低到中等噪音，還可接受
            elif breed_noise_lower in ['moderate']:
                return 0.4   # 中等噪音有些問題
            elif 'high' in breed_noise_lower:
                return 0.1   # 高噪音不適合
            else:
                return 0.6   # 未知噪音水平，保守估計
        elif noise_req['tolerance'] == 'high':
            if 'high' in breed_noise_lower:
                return 0.9
            elif 'moderate' in breed_noise_lower:
                return 0.85
            elif 'low' in breed_noise_lower:
                return 0.8   # 安靜犬對高容忍度的人也很好
            else:
                return 0.8
        else:  # moderate tolerance
            if 'moderate' in breed_noise_lower:
                return 0.9
            elif 'low' in breed_noise_lower:
                return 0.85
            elif 'high' in breed_noise_lower:
                return 0.6
            else:
                return 0.75

        return 0.7

    def _calculate_size_compatibility(self, size_req: dict, breed_size: str) -> float:
        """計算體型相容性分數"""
        if size_req['preferred'] == 'small':
            if any(word in breed_size for word in ['small', 'toy', 'tiny']):
                return 0.9
            elif 'medium' in breed_size:
                return 0.6
            else:
                return 0.3
        elif size_req['preferred'] == 'large':
            if any(word in breed_size for word in ['large', 'giant']):
                return 0.9
            elif 'medium' in breed_size:
                return 0.7
            else:
                return 0.4

        return 0.7  # 無特別偏好

    def _calculate_family_compatibility(self, family_req: dict, good_with_children: str, temperament: str) -> float:
        """計算家庭相容性分數"""
        if family_req['children']:
            if 'yes' in good_with_children.lower():
                return 0.9
            elif any(word in temperament for word in ['gentle', 'patient', 'friendly']):
                return 0.8
            elif 'no' in good_with_children.lower():
                return 0.2
            else:
                return 0.6

        return 0.7

    def _analyze_experience_requirements(self, user_desc: str) -> dict:
        """分析用戶經驗水平需求"""
        requirements = {'level': 'intermediate', 'importance': 0.5}

        # 新手識別 - 關鍵詞匹配
        beginner_terms = ['first dog', 'first time', 'beginner', 'new to dogs', 'inexperienced',
                         'never owned', 'never had a dog', 'first-time owner', 'my first']
        if any(term in user_desc for term in beginner_terms):
            requirements['level'] = 'beginner'
            requirements['importance'] = 0.95  # 對新手非常重要

        # 高級用戶識別
        advanced_terms = ['experienced', 'advanced', 'expert', 'breeder', 'trainer', 'many dogs']
        if any(term in user_desc for term in advanced_terms):
            requirements['level'] = 'advanced'
            requirements['importance'] = 0.6

        # 易於訓練需求
        if any(term in user_desc for term in ['easy to train', 'trainable', 'obedient', 'well-behaved']):
            requirements['needs_easy_training'] = True
            requirements['importance'] = max(requirements['importance'], 0.85)

        # 低維護需求通常暗示需要更易處理的品種
        if any(term in user_desc for term in ['low maintenance', 'low-maintenance', 'easy care']):
            requirements['needs_easy_care'] = True
            if requirements['level'] == 'intermediate':
                requirements['level'] = 'beginner'  # 低維護需求暗示初學者
                requirements['importance'] = 0.85

        return requirements

    def _calculate_experience_compatibility(self, experience_req: dict, breed_info: dict, temperament: str) -> float:
        """
        計算經驗相容性分數 - 基於品種特性和用戶經驗水平

        這是修復的關鍵函數！確保敏感/難以處理的品種對新手有低分數
        """
        care_level = breed_info.get('Care Level', 'Moderate').lower()
        temperament_lower = temperament.lower()
        user_level = experience_req.get('level', 'intermediate')

        # 基礎分數矩陣
        base_scores = {
            'high': {
                'beginner': 0.45,      # 高照護品種對新手困難
                'intermediate': 0.75,
                'advanced': 0.90
            },
            'moderate': {
                'beginner': 0.65,
                'intermediate': 0.85,
                'advanced': 0.90
            },
            'low': {
                'beginner': 0.85,      # 低照護品種對新手友善
                'intermediate': 0.90,
                'advanced': 0.90
            }
        }

        # 獲取基礎分數
        score = base_scores.get(care_level, base_scores['moderate']).get(user_level, 0.70)

        # 性格特徵調整 - 對新手特別重要
        if user_level == 'beginner':
            # 困難性格懲罰
            difficult_traits = {
                'sensitive': -0.20,      # 敏感品種對新手非常困難（需要細心處理）
                'stubborn': -0.15,
                'independent': -0.12,
                'dominant': -0.15,
                'aggressive': -0.25,
                'nervous': -0.15,
                'alert': -0.08,          # 過度警覺可能導致吠叫問題
                'shy': -0.12,
                'timid': -0.12,
                'strong-willed': -0.12,
                'protective': -0.10
            }

            for trait, penalty in difficult_traits.items():
                if trait in temperament_lower:
                    score += penalty

            # 友善性格獎勵
            easy_traits = {
                'gentle': 0.10,
                'friendly': 0.12,
                'eager to please': 0.15,
                'patient': 0.10,
                'calm': 0.10,
                'outgoing': 0.08,
                'affectionate': 0.08,
                'playful': 0.05,       # 輕微加分（可能太活潑）
                'loyal': 0.05
            }

            for trait, bonus in easy_traits.items():
                if trait in temperament_lower:
                    score += bonus

            # 易於訓練需求額外懲罰/獎勵
            if experience_req.get('needs_easy_training'):
                if any(term in temperament_lower for term in ['stubborn', 'independent', 'strong-willed']):
                    score -= 0.12
                elif any(term in temperament_lower for term in ['eager to please', 'intelligent', 'trainable']):
                    score += 0.10

            # Good with Children = No 對新手也是警示
            good_with_children = breed_info.get('Good with Children', 'Yes')
            if good_with_children == 'No':
                score -= 0.08  # 額外扣分：不適合兒童的狗通常對新手也更具挑戰

        elif user_level == 'intermediate':
            # 中級用戶的適度調整
            if 'stubborn' in temperament_lower:
                score -= 0.05
            if 'independent' in temperament_lower:
                score -= 0.03

        elif user_level == 'advanced':
            # 高級用戶可以處理具挑戰性的品種
            if any(term in temperament_lower for term in ['intelligent', 'working', 'athletic']):
                score += 0.05

        # 確保分數在合理範圍內
        return max(0.15, min(0.95, score))

    def _apply_hard_constraints_enhanced(self, user_desc: str, breed_info: dict) -> float:
        """應用品種特性感知的動態懲罰機制"""
        penalty = 0.0

        # 建立懲罰衰減係數和補償機制
        penalty_decay_factor = 0.7
        breed_adaptability_bonus = 0.0
        breed_size = breed_info.get('Size', '').lower()
        breed_exercise = breed_info.get('Exercise Needs', '').lower()
        breed_name = breed_info.get('Breed', '').replace(' ', '_')

        # 公寓空間約束 - 品種特性感知懲罰機制
        if 'apartment' in user_desc or 'small apartment' in user_desc:
            if 'giant' in breed_size:
                base_penalty = -0.35  # 減少基礎懲罰
                # 特定品種適應性補償
                adaptable_giants = ['Mastiff', 'Great Dane']  # 相對安靜的巨型犬
                if any(adapt_breed in breed_name for adapt_breed in adaptable_giants):
                    breed_adaptability_bonus += 0.08
                penalty += base_penalty * penalty_decay_factor
            elif 'large' in breed_size:
                base_penalty = -0.25  # 減少大型犬懲罰
                # 適合公寓的大型犬補償
                apartment_friendly_large = ['Greyhound', 'Great_Dane']
                if any(apt_breed in breed_name for apt_breed in apartment_friendly_large):
                    breed_adaptability_bonus += 0.06
                penalty += base_penalty * penalty_decay_factor
            elif 'medium' in breed_size and 'high' in breed_exercise:
                penalty += -0.15 * penalty_decay_factor  # 進一步減少懲罰

        # 運動需求不匹配 - 品種特性感知懲罰機制
        if any(phrase in user_desc for phrase in ["don't exercise", "not active", "low exercise", "don't exercise much"]):
            if 'high' in breed_exercise:
                base_penalty = -0.28  # 減少基礎懲罰
                # 低維護高運動犬種補償
                adaptable_high_energy = ['Greyhound', 'Whippet']  # 運動爆發型，平時安靜
                if any(adapt_breed in breed_name for adapt_breed in adaptable_high_energy):
                    breed_adaptability_bonus += 0.10
                penalty += base_penalty * penalty_decay_factor
            elif 'moderate' in breed_exercise:
                penalty += -0.08 * penalty_decay_factor  # 進一步減少懲罰

        # 噪音控制需求不匹配 - 品種特性感知懲罰機制
        if any(phrase in user_desc for phrase in ['quiet', "won't bark", "doesn't bark", "silent"]):
            breed_noise = breed_noise_info.get(breed_name, {}).get('noise_level', 'moderate').lower()
            if 'high' in breed_noise:
                base_penalty = -0.18  # 減少基礎懲罰
                # 訓練性良好的高噪音品種補償
                trainable_vocal_breeds = ['German_Shepherd', 'Golden_Retriever']
                if any(train_breed in breed_name for train_breed in trainable_vocal_breeds):
                    breed_adaptability_bonus += 0.05
                penalty += base_penalty * penalty_decay_factor
            elif 'moderate' in breed_noise and 'low' not in breed_noise:
                penalty += -0.05 * penalty_decay_factor

        # 體型偏好不匹配 - 漸進式懲罰
        if any(phrase in user_desc for phrase in ['small', 'tiny', 'little']):
            if 'giant' in breed_size:
                penalty -= 0.35  # 超大型犬懲罰
            elif 'large' in breed_size:
                penalty -= 0.20  # 大型犬懲罰

        # 中等活動量用戶的特殊約束處理 - 漸進式懲罰
        moderate_activity_terms = ['30 minutes', 'half hour', 'moderate', 'balanced', 'medium-sized house']
        if any(term in user_desc for term in moderate_activity_terms):
            # 超大型犬對中等活動量用戶的適度懲罰
            giant_breeds = ['Saint Bernard', 'Tibetan Mastiff', 'Great Dane', 'Mastiff', 'Newfoundland']
            if any(giant in breed_name for giant in giant_breeds) or 'giant' in breed_size:
                penalty -= 0.35  # 適度懲罰，不完全排除

            # 中型房屋 + 超大型犬的額外考量
            if 'medium-sized house' in user_desc and any(giant in breed_name for giant in giant_breeds):
                if not any(high_activity in user_desc for high_activity in ['hiking', 'running', 'active', 'outdoor activities']):
                    penalty -= 0.15  # 輕度額外懲罰

        # 30分鐘散步對極高運動需求品種的懲罰
        if any(term in user_desc for term in ['30 minutes', 'half hour']) and 'walk' in user_desc:
            high_energy_breeds = ['Siberian Husky', 'Border Collie', 'Jack Russell Terrier', 'Weimaraner']
            if any(he_breed in breed_name for he_breed in high_energy_breeds) and 'high' in breed_exercise:
                penalty -= 0.25  # 適度懲罰極高運動需求品種

        # 添加特殊品種適應性補償機制
        # 對於邊界適配品種，給予適度補償
        # 注意：僅補償真正對新手友善的品種
        boundary_adaptable_breeds = {
            # 'Italian_Greyhound' 已移除：Sensitive 性格對新手不友好
            'Boston_Bull': 0.06,        # 適應性強的小型犬
            'Havanese': 0.05,           # 友好適應的小型犬
            'Silky_terrier': 0.04,      # 安靜的玩具犬
            'Basset': 0.07,             # 低能量但友好的中型犬
            'Cavalier_King_Charles_Spaniel': 0.08,  # 溫和友善，適合新手
            'Bichon_Frise': 0.06        # 友善易訓練
        }

        if breed_name in boundary_adaptable_breeds:
            breed_adaptability_bonus += boundary_adaptable_breeds[breed_name]

        # 應用品種適應性補償並設置懲罰上限
        final_penalty = penalty + breed_adaptability_bonus
        # 限制最大懲罰，避免單一約束主導評分
        final_penalty = max(-0.4, final_penalty)

        return final_penalty

    def get_breed_characteristics_enhanced(self, breed: str) -> Dict[str, Any]:
        """獲取品種特徵"""
        breed_info = get_dog_description(breed)
        if not breed_info:
            return {}

        characteristics = {
            'size': breed_info.get('Size', 'Unknown'),
            'temperament': breed_info.get('Temperament', ''),
            'exercise_needs': breed_info.get('Exercise Needs', 'Moderate'),
            'grooming_needs': breed_info.get('Grooming Needs', 'Moderate'),
            'good_with_children': breed_info.get('Good with Children', 'Unknown'),
            'lifespan': breed_info.get('Lifespan', '10-12 years'),
            'description': breed_info.get('Description', '')
        }

        # 添加噪音資訊
        noise_info = breed_noise_info.get(breed, {})
        characteristics['noise_level'] = noise_info.get('noise_level', 'moderate')

        return characteristics

    def get_breed_info_from_standardized(self, standardized_info) -> Dict[str, Any]:
        """將標準化品種信息轉換為字典格式"""
        try:
            size_map = {1: 'Tiny', 2: 'Small', 3: 'Medium', 4: 'Large', 5: 'Giant'}
            exercise_map = {1: 'Low', 2: 'Moderate', 3: 'High', 4: 'Very High'}
            care_map = {1: 'Low', 2: 'Moderate', 3: 'High'}

            return {
                'Size': size_map.get(standardized_info.size_category, 'Medium'),
                'Exercise Needs': exercise_map.get(standardized_info.exercise_level, 'Moderate'),
                'Grooming Needs': care_map.get(standardized_info.care_complexity, 'Moderate'),
                'Good with Children': 'Yes' if standardized_info.child_compatibility >= 0.8 else
                                     'No' if standardized_info.child_compatibility <= 0.2 else 'Unknown',
                'Temperament': 'Varies by individual',
                'Lifespan': '10-12 years',
                'Description': f'A {size_map.get(standardized_info.size_category, "medium")} sized breed'
            }
        except Exception as e:
            print(f"Error converting standardized info: {str(e)}")
            return {}

    def get_fallback_recommendations(self, top_k: int = 15) -> List[Dict[str, Any]]:
        """當增強系統失敗時獲取備用推薦"""
        try:
            safe_breeds = [
                ('Labrador Retriever', 0.85),
                ('Golden Retriever', 0.82),
                ('Cavalier King Charles Spaniel', 0.80),
                ('French Bulldog', 0.78),
                ('Boston Terrier', 0.76),
                ('Bichon Frise', 0.74),
                ('Pug', 0.72),
                ('Cocker Spaniel', 0.70)
            ]

            recommendations = []
            for i, (breed, score) in enumerate(safe_breeds[:top_k]):
                breed_info = get_dog_description(breed.replace(' ', '_')) or {}

                recommendation = {
                    'breed': breed,
                    'rank': i + 1,
                    'overall_score': score,
                    'final_score': score,
                    'semantic_score': score * 0.8,
                    'comparative_bonus': 0.0,
                    'lifestyle_bonus': 0.0,
                    'size': breed_info.get('Size', 'Unknown'),
                    'temperament': breed_info.get('Temperament', ''),
                    'exercise_needs': breed_info.get('Exercise Needs', 'Moderate'),
                    'grooming_needs': breed_info.get('Grooming Needs', 'Moderate'),
                    'good_with_children': breed_info.get('Good with Children', 'Yes'),
                    'lifespan': breed_info.get('Lifespan', '10-12 years'),
                    'description': breed_info.get('Description', ''),
                    'search_type': 'fallback'
                }
                recommendations.append(recommendation)

            return recommendations

        except Exception as e:
            print(f"Error generating fallback recommendations: {str(e)}")
            return []
