# %%writefile semantic_breed_recommender.py
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
from semantic_vector_manager import SemanticVectorManager, BreedDescriptionVector
from user_query_analyzer import UserQueryAnalyzer
from matching_score_calculator import MatchingScoreCalculator
from smart_breed_filter import apply_smart_filtering

class SemanticBreedRecommender:
    """
    增強的基於 SBERT 的語義品種推薦系統 (Facade Pattern)
    為狗品種推薦提供多維度自然語言理解
    """

    def __init__(self):
        """初始化語義品種推薦器"""
        # 初始化語義向量管理器
        self.vector_manager = SemanticVectorManager()

        # 初始化用戶查詢分析器
        self.query_analyzer = UserQueryAnalyzer(self.vector_manager.get_breed_list())

        # 初始化匹配評分計算器
        self.score_calculator = MatchingScoreCalculator(self.vector_manager.get_breed_list())

        # 保留原有屬性以維持向後兼容性
        self.model_name = self.vector_manager.model_name
        self.sbert_model = self.vector_manager.get_sbert_model()
        self.breed_vectors = self.vector_manager.get_breed_vectors()
        self.breed_list = self.vector_manager.get_breed_list()
        self.comparative_keywords = self.query_analyzer.comparative_keywords

        # 初始化增強系統組件（如果可用）
        try:
            self.query_engine = QueryUnderstandingEngine()
            print("QueryUnderstandingEngine initialized")
            self.constraint_manager = ConstraintManager()
            print("ConstraintManager initialized")
            self.multi_head_scorer = None
            self.score_calibrator = ScoreCalibrator()
            print("ScoreCalibrator initialized")
            self.config_manager = get_config_manager()

            # 如果 SBERT 模型可用，初始化多頭評分器
            if self.sbert_model:
                self.multi_head_scorer = MultiHeadScorer(self.sbert_model)
                print("Multi-head scorer initialized with SBERT model")
            else:
                print("WARNING: SBERT model not available, multi_head_scorer will be None")
        except Exception as e:
            print(f"Error initializing enhanced system components: {str(e)}")
            print(traceback.format_exc())
            self.query_engine = None
            self.constraint_manager = None
            self.multi_head_scorer = None
            self.score_calibrator = None
            self.config_manager = None

    def _parse_comparative_preferences(self, user_input: str) -> Dict[str, float]:
        """解析比較性偏好表達"""
        return self.query_analyzer.parse_comparative_preferences(user_input)

    def _extract_lifestyle_keywords(self, user_input: str) -> Dict[str, List[str]]:
        """增強的生活方式關鍵字提取，具有更好的模式匹配"""
        return self.query_analyzer.extract_lifestyle_keywords(user_input)

    def _apply_size_distribution_correction(self, recommendations: List[Dict]) -> List[Dict]:
        """應用尺寸分佈修正以防止大型品種偏差"""
        return self.score_calculator.apply_size_distribution_correction(recommendations)

    def _normalize_breed_size(self, size: str) -> str:
        """標準化品種尺寸到標準分類"""
        return self.score_calculator._normalize_breed_size(size)

    def _parse_user_requirements(self, user_input: str) -> Dict[str, Any]:
        """更準確地解析用戶需求"""
        return self.query_analyzer.parse_user_requirements(user_input)

    def _apply_hard_constraints(self, breed: str, user_input: str, breed_characteristics: Dict[str, Any]) -> float:
        """增強硬約束，具有更嚴格的懲罰"""
        return self.score_calculator.apply_hard_constraints(breed, user_input, breed_characteristics)

    def _calculate_lifestyle_bonus(self, breed_characteristics: Dict[str, Any],
                                 lifestyle_keywords: Dict[str, List[str]]) -> float:
        """增強生活方式匹配獎勵計算"""
        return self.score_calculator.calculate_lifestyle_bonus(breed_characteristics, lifestyle_keywords)

    def _apply_intelligent_trait_matching(self, recommendations: List[Dict], user_input: str) -> List[Dict]:
        """基於增強關鍵字提取和數據庫挖掘應用智能特徵匹配"""
        return self.score_calculator.apply_intelligent_trait_matching(recommendations, user_input)

    def _get_breed_info_from_standardized(self, standardized_info) -> Dict[str, Any]:
        """將標準化品種信息轉換為字典格式"""
        return self.score_calculator.get_breed_info_from_standardized(standardized_info)

    def _get_fallback_recommendations(self, top_k: int = 15) -> List[Dict[str, Any]]:
        """當增強系統失敗時獲取備用推薦"""
        return self.score_calculator.get_fallback_recommendations(top_k)

    def _get_fallback_scoring_with_constraints(self, user_input: str,
                                               passed_breeds: set,
                                               dimensions: 'QueryDimensions',
                                               top_k: int = 15) -> List[Dict[str, Any]]:
        """
        當 multi_head_scorer 不可用時的回退評分方法
        關鍵：仍然尊重 constraint_manager 的過濾結果，並產生自然分佈的分數
        """
        print(f"Fallback scoring for {len(passed_breeds)} filtered breeds")

        recommendations = []
        user_text = user_input.lower()

        # 提取用戶需求關鍵詞
        lifestyle_keywords = self._extract_lifestyle_keywords(user_input)

        for breed in passed_breeds:
            breed_info = get_dog_description(breed.replace(' ', '_')) or {}
            if not breed_info:
                continue

            # 計算多維度匹配分數
            dimension_scores = self._calculate_comprehensive_dimension_scores(
                breed, breed_info, user_text, dimensions, lifestyle_keywords
            )

            # 基於維度分數計算加權總分
            weights = self._get_dimension_weights_from_query(user_text, dimensions)
            weighted_sum = sum(dimension_scores.get(dim, 0.7) * weights.get(dim, 1.0)
                             for dim in dimension_scores)
            total_weight = sum(weights.get(dim, 1.0) for dim in dimension_scores)
            final_score = weighted_sum / total_weight if total_weight > 0 else 0.7

            # 確保分數在合理範圍內（允許高分，非常契合的品種可超過 90%）
            final_score = max(0.45, min(0.98, final_score))
            dimension_scores['overall'] = final_score

            recommendation = {
                'breed': breed.replace('_', ' '),
                'rank': 0,
                'overall_score': final_score,
                'final_score': final_score,
                'scores': dimension_scores,
                'size': breed_info.get('Size', 'Unknown'),
                'temperament': breed_info.get('Temperament', ''),
                'exercise_needs': breed_info.get('Exercise Needs', 'Moderate'),
                'grooming_needs': breed_info.get('Grooming Needs', 'Moderate'),
                'good_with_children': breed_info.get('Good with Children', 'Yes'),
                'lifespan': breed_info.get('Lifespan', '10-12 years'),
                'description': breed_info.get('Description', ''),
                'search_type': 'fallback_with_constraints',
            }

            recommendations.append(recommendation)

        # 按分數排序
        recommendations.sort(key=lambda x: -x['final_score'])

        # 更新排名
        for i, rec in enumerate(recommendations[:top_k]):
            rec['rank'] = i + 1

        print(f"Generated {len(recommendations[:top_k])} fallback recommendations")
        return recommendations[:top_k]

    def _calculate_comprehensive_dimension_scores(self, breed: str, breed_info: Dict,
                                                   user_text: str, dimensions,
                                                   lifestyle_keywords: Dict) -> Dict[str, float]:
        """
        計算全面的維度分數，產生自然分佈的評分
        """
        scores = {}
        temperament = breed_info.get('Temperament', '').lower()
        size = breed_info.get('Size', 'Medium').lower()
        exercise_needs = breed_info.get('Exercise Needs', 'Moderate').lower()
        grooming_needs = breed_info.get('Grooming Needs', 'Moderate').lower()
        good_with_children = breed_info.get('Good with Children', 'Yes')
        care_level = breed_info.get('Care Level', 'Moderate').lower()
        description = breed_info.get('Description', '').lower()

        # 1. 空間相容性
        space_score = 0.7
        if 'apartment' in user_text or 'small space' in user_text:
            if 'small' in size or 'toy' in size:
                space_score = 0.96
            elif 'medium' in size:
                space_score = 0.78
            elif 'large' in size:
                space_score = 0.52
            else:
                space_score = 0.45
        elif 'house' in user_text or 'yard' in user_text:
            if 'large' in size:
                space_score = 0.92
            elif 'medium' in size:
                space_score = 0.88
            else:
                space_score = 0.82
        scores['space'] = space_score

        # 2. 運動相容性
        exercise_score = 0.7
        user_wants_high = any(w in user_text for w in ['energetic', 'active', 'running', 'hiking', 'athletic'])
        user_wants_low = any(w in user_text for w in ['low maintenance', 'relaxed', 'calm', 'couch'])

        if user_wants_high:
            if 'very high' in exercise_needs:
                exercise_score = 0.98
            elif 'high' in exercise_needs:
                exercise_score = 0.92
            elif 'moderate' in exercise_needs:
                exercise_score = 0.68
            else:
                exercise_score = 0.48
        elif user_wants_low:
            if 'low' in exercise_needs:
                exercise_score = 0.96
            elif 'moderate' in exercise_needs:
                exercise_score = 0.78
            elif 'high' in exercise_needs:
                exercise_score = 0.52
            else:
                exercise_score = 0.42
        else:
            # 中等運動需求
            if 'moderate' in exercise_needs:
                exercise_score = 0.88
            elif 'low' in exercise_needs or 'high' in exercise_needs:
                exercise_score = 0.72
            else:
                exercise_score = 0.65
        scores['exercise'] = exercise_score

        # 3. 美容需求相容性
        grooming_score = 0.7
        user_wants_low_maintenance = any(w in user_text for w in ['low maintenance', 'easy care', 'minimal grooming'])

        if user_wants_low_maintenance:
            if 'low' in grooming_needs or 'minimal' in grooming_needs:
                grooming_score = 0.96
            elif 'moderate' in grooming_needs:
                grooming_score = 0.75
            else:
                grooming_score = 0.50
        else:
            if 'low' in grooming_needs:
                grooming_score = 0.85
            elif 'moderate' in grooming_needs:
                grooming_score = 0.78
            else:
                grooming_score = 0.70
        scores['grooming'] = grooming_score

        # 4. 噪音相容性
        noise_score = 0.7
        user_wants_quiet = any(w in user_text for w in ['quiet', 'silent', 'noise', 'bark', 'neighbors'])

        if user_wants_quiet:
            # 從 breed_noise_info 獲取噪音資訊
            noise_info = breed_noise_info.get(breed.replace(' ', '_'), {})
            noise_level = noise_info.get('noise_level', 'Moderate').lower()

            if 'low' in noise_level or 'quiet' in noise_level:
                noise_score = 0.97
            elif 'moderate' in noise_level:
                noise_score = 0.72
            elif 'high' in noise_level:
                noise_score = 0.45
            else:
                # 根據性格推斷
                if any(w in temperament for w in ['calm', 'quiet', 'gentle', 'reserved']):
                    noise_score = 0.88
                elif any(w in temperament for w in ['alert', 'vocal', 'energetic']):
                    noise_score = 0.55
                else:
                    noise_score = 0.70
        scores['noise'] = noise_score

        # 5. 家庭相容性
        family_score = 0.7
        has_family_context = any(w in user_text for w in ['kids', 'children', 'family', 'child'])

        if has_family_context:
            if good_with_children == 'Yes':
                family_score = 0.94
                # 額外加分：溫和性格
                if any(w in temperament for w in ['gentle', 'friendly', 'patient', 'loving']):
                    family_score = min(0.98, family_score + 0.04)
            elif good_with_children == 'No':
                family_score = 0.32
            else:
                family_score = 0.62
        else:
            family_score = 0.76 if good_with_children == 'Yes' else 0.70
        scores['family'] = family_score

        # 6. 經驗相容性
        experience_score = 0.7
        is_beginner = any(w in user_text for w in ['first dog', 'first time', 'beginner', 'new owner', 'never had'])

        if is_beginner:
            # 評估品種對新手的友好程度
            if 'low' in care_level or 'easy' in care_level:
                experience_score = 0.94
            elif 'moderate' in care_level:
                experience_score = 0.78
            else:
                experience_score = 0.52

            # 性格調整
            if any(w in temperament for w in ['eager to please', 'trainable', 'intelligent', 'friendly']):
                experience_score = min(0.98, experience_score + 0.08)
            if any(w in temperament for w in ['stubborn', 'independent', 'strong-willed']):
                experience_score = max(0.38, experience_score - 0.18)
        else:
            experience_score = 0.80
        scores['experience'] = experience_score

        # 7. 健康分數（基於壽命和品種特性）
        health_score = 0.75
        lifespan = breed_info.get('Lifespan', '10-12 years')
        try:
            # 解析壽命
            years = [int(y) for y in lifespan.replace(' years', '').split('-') if y.strip().isdigit()]
            if years:
                avg_lifespan = sum(years) / len(years)
                if avg_lifespan >= 14:
                    health_score = 0.94
                elif avg_lifespan >= 12:
                    health_score = 0.85
                elif avg_lifespan >= 10:
                    health_score = 0.75
                else:
                    health_score = 0.62
        except:
            pass
        scores['health'] = health_score

        return scores

    def _get_dimension_weights_from_query(self, user_text: str, dimensions) -> Dict[str, float]:
        """
        根據用戶查詢動態計算維度權重
        """
        weights = {
            'space': 1.0,
            'exercise': 1.0,
            'grooming': 1.0,
            'noise': 1.0,
            'family': 1.0,
            'experience': 1.0,
            'health': 0.8
        }

        # 根據 dimensions 的 priority 調整權重
        if hasattr(dimensions, 'dimension_priorities'):
            priority_map = getattr(dimensions, 'dimension_priorities', {})
            for dim, priority in priority_map.items():
                if dim in weights:
                    weights[dim] = priority
                # 映射不同名稱
                if dim == 'size':
                    weights['space'] = max(weights['space'], priority)
                if dim == 'family':
                    weights['family'] = max(weights['family'], priority)

        # 根據關鍵詞強化權重
        if any(w in user_text for w in ['quiet', 'noise', 'bark', 'neighbors', 'thin walls']):
            weights['noise'] = max(weights['noise'], 2.2)
        if any(w in user_text for w in ['kids', 'children', 'family', 'child']):
            weights['family'] = max(weights['family'], 2.0)
        if any(w in user_text for w in ['first', 'beginner', 'new owner']):
            weights['experience'] = max(weights['experience'], 2.0)
        if any(w in user_text for w in ['apartment', 'small space', 'studio']):
            weights['space'] = max(weights['space'], 1.8)
        if any(w in user_text for w in ['energetic', 'active', 'running', 'hiking']):
            weights['exercise'] = max(weights['exercise'], 2.0)
        if any(w in user_text for w in ['low maintenance', 'easy care']):
            weights['grooming'] = max(weights['grooming'], 1.8)

        return weights

    def _calculate_real_dimension_scores(self, breed: str, breed_info: Dict,
                                        user_input: str, overall_score: float) -> Dict[str, float]:
        """
        計算真實的維度分數（基於品種特性和用戶需求）
        這個方法取代了假分數生成器，提供真實的評分

        Args:
            breed: 品種名稱
            breed_info: 品種資訊字典
            user_input: 用戶輸入文字
            overall_score: 總體分數

        Returns:
            Dict[str, float]: 維度分數字典
        """
        if not breed_info:
            breed_info = {}

        user_text = user_input.lower()
        temperament = breed_info.get('Temperament', '').lower()
        size = breed_info.get('Size', 'Medium').lower()
        exercise_needs = breed_info.get('Exercise Needs', 'Moderate').lower()
        grooming_needs = breed_info.get('Grooming Needs', 'Moderate').lower()
        good_with_children = breed_info.get('Good with Children', 'Yes')
        care_level = breed_info.get('Care Level', 'Moderate').lower()

        scores = {}

        # 1. Space Compatibility (空間相容性)
        space_score = 0.7
        if 'apartment' in user_text or 'small' in user_text:
            if 'small' in size:
                space_score = 0.9
            elif 'medium' in size:
                space_score = 0.7
            elif 'large' in size:
                space_score = 0.5
            elif 'giant' in size:
                space_score = 0.3
        elif 'house' in user_text or 'yard' in user_text:
            if 'large' in size or 'giant' in size:
                space_score = 0.85
            else:
                space_score = 0.8
        scores['space'] = space_score

        # 2. Exercise Compatibility (運動相容性)
        exercise_score = 0.7
        if 'low' in exercise_needs or 'minimal' in exercise_needs:
            if any(term in user_text for term in ['work full time', 'busy', 'low exercise', 'not much exercise']):
                exercise_score = 0.9
            else:
                exercise_score = 0.75
        elif 'high' in exercise_needs or 'very high' in exercise_needs:
            if any(term in user_text for term in ['active', 'running', 'hiking', 'exercise']):
                exercise_score = 0.9
            elif any(term in user_text for term in ['work full time', 'busy']):
                exercise_score = 0.5
            else:
                exercise_score = 0.65
        else:  # moderate
            exercise_score = 0.75
        scores['exercise'] = exercise_score

        # 3. Grooming/Maintenance Compatibility (美容/維護相容性)
        grooming_score = 0.7
        if 'low' in grooming_needs:
            if any(term in user_text for term in ['low maintenance', 'low-maintenance', 'easy care', 'minimal grooming']):
                grooming_score = 0.9
            else:
                grooming_score = 0.8
        elif 'high' in grooming_needs:
            if any(term in user_text for term in ['low maintenance', 'low-maintenance', 'easy care']):
                grooming_score = 0.4
            else:
                grooming_score = 0.6

        # 敏感品種需要額外照顧
        if 'sensitive' in temperament:
            grooming_score -= 0.1
        # 特殊品種需要額外護理
        breed_lower = breed.lower()
        if any(term in breed_lower for term in ['italian', 'greyhound', 'whippet', 'hairless']):
            if any(term in user_text for term in ['low maintenance', 'low-maintenance', 'easy']):
                grooming_score -= 0.15
        scores['grooming'] = max(0.2, grooming_score)

        # 4. Experience Compatibility (經驗相容性) - 關鍵維度！
        experience_score = 0.7
        is_beginner = any(term in user_text for term in ['first dog', 'first time', 'beginner', 'new to dogs', 'never owned', 'never had'])

        if is_beginner:
            # 新手評估
            if 'low' in care_level:
                experience_score = 0.85
            elif 'moderate' in care_level:
                experience_score = 0.65
            elif 'high' in care_level:
                experience_score = 0.45

            # 性格懲罰 - 對新手很重要
            difficult_traits = ['sensitive', 'stubborn', 'independent', 'dominant', 'aggressive', 'nervous', 'shy', 'timid', 'alert']
            for trait in difficult_traits:
                if trait in temperament:
                    if trait == 'sensitive':
                        experience_score -= 0.15  # 敏感性格對新手很具挑戰
                    elif trait == 'aggressive':
                        experience_score -= 0.25
                    elif trait in ['stubborn', 'independent', 'dominant']:
                        experience_score -= 0.12
                    else:
                        experience_score -= 0.08

            # 友善性格獎勵
            easy_traits = ['friendly', 'gentle', 'eager to please', 'patient', 'calm', 'outgoing']
            for trait in easy_traits:
                if trait in temperament:
                    experience_score += 0.08

            # 易於訓練的加分
            if any(term in user_text for term in ['easy to train', 'trainable']):
                if any(term in temperament for term in ['eager to please', 'intelligent', 'trainable']):
                    experience_score += 0.1
                elif any(term in temperament for term in ['stubborn', 'independent']):
                    experience_score -= 0.1
        else:
            # 有經驗的飼主
            experience_score = 0.8

        scores['experience'] = max(0.2, min(0.95, experience_score))

        # 5. Noise Compatibility (噪音相容性)
        noise_score = 0.75
        if any(term in user_text for term in ['quiet', 'apartment', 'neighbors']):
            if any(term in temperament for term in ['quiet', 'calm', 'gentle']):
                noise_score = 0.9
            elif any(term in temperament for term in ['alert', 'vocal', 'barking']):
                noise_score = 0.5
        scores['noise'] = noise_score

        # 6. Family Compatibility (家庭相容性)
        family_score = 0.7
        if any(term in user_text for term in ['children', 'kids', 'family']):
            if good_with_children == 'Yes' or good_with_children == True:
                family_score = 0.9
                if any(term in temperament for term in ['gentle', 'patient', 'friendly']):
                    family_score = 0.95
            else:
                family_score = 0.35
        scores['family'] = family_score

        # 7. Overall
        scores['overall'] = overall_score

        return scores

    def get_enhanced_semantic_recommendations(self, user_input: str, top_k: int = 15) -> List[Dict[str, Any]]:
        """
        增強的多維度語義品種推薦

        Args:
            user_input: 用戶的自然語言描述
            top_k: 返回的推薦數量

        Returns:
            增強評分的推薦品種列表
        """
        try:
            # 階段 1: 查詢理解
            if self.query_engine:
                dimensions = self.query_engine.analyze_query(user_input)
                print(f"Query dimensions detected: {len(dimensions.spatial_constraints + dimensions.activity_level + dimensions.noise_preferences + dimensions.size_preferences + dimensions.family_context + dimensions.maintenance_level + dimensions.special_requirements)} total dimensions")
            else:
                print("Query engine not available, using basic analysis")
                return self.get_semantic_recommendations(user_input, top_k)

            # 階段 2: 應用約束
            if self.constraint_manager:
                filter_result = self.constraint_manager.apply_constraints(dimensions, min_candidates=max(8, top_k))
                print(f"Constraint filtering: {len(self.breed_list)} -> {len(filter_result.passed_breeds)} candidates")

                if not filter_result.passed_breeds:
                    error_msg = f"No dog breeds match your requirements after applying constraints. Applied constraints: {filter_result.applied_constraints}. Consider relaxing some requirements."
                    print(f"ERROR: {error_msg}")
                    raise ValueError(error_msg)
            else:
                print("Constraint manager not available, using all breeds")
                filter_result = type('FilterResult', (), {
                    'passed_breeds': self.breed_list,
                    'applied_constraints': [],
                    'relaxed_constraints': [],
                    'warnings': []
                })()

            # 階段 3: 多頭評分
            if self.multi_head_scorer:
                breed_scores = self.multi_head_scorer.score_breeds(filter_result.passed_breeds, dimensions)
                print(f"Multi-head scoring completed for {len(breed_scores)} breeds")
                # Debug: 顯示前5名的分數和維度breakdown
                for bs in breed_scores[:5]:
                    print(f"  {bs.breed_name}: final={bs.final_score:.3f}, breakdown={bs.dimensional_breakdown}")
            else:
                # 使用回退評分，但仍然尊重 constraint 過濾結果
                print("Multi-head scorer not available, using fallback scoring with constraint filtering")
                fallback_results = self._get_fallback_scoring_with_constraints(
                    user_input, filter_result.passed_breeds, dimensions, top_k
                )
                return fallback_results

            # 階段 4: 分數校準
            if self.score_calibrator:
                breed_score_tuples = [(score.breed_name, score.final_score) for score in breed_scores]
                calibration_result = self.score_calibrator.calibrate_scores(breed_score_tuples)
                print(f"Score calibration: method={calibration_result.calibration_method}")
            else:
                print("Score calibrator not available, using raw scores")
                calibration_result = type('CalibrationResult', (), {
                    'score_mapping': {score.breed_name: score.final_score for score in breed_scores},
                    'calibration_method': 'none'
                })()

            # 階段 5: 生成最終推薦
            final_recommendations = []

            for i, breed_score in enumerate(breed_scores[:top_k]):
                breed_name = breed_score.breed_name

                # 獲取校準後的分數
                calibrated_score = calibration_result.score_mapping.get(breed_name, breed_score.final_score)

                # 獲取標準化品種信息
                if self.config_manager:
                    standardized_info = get_standardized_breed_data(breed_name.replace(' ', '_'))
                    if standardized_info:
                        breed_info = self._get_breed_info_from_standardized(standardized_info)
                    else:
                        breed_info = get_dog_description(breed_name.replace(' ', '_')) or {}
                else:
                    breed_info = get_dog_description(breed_name.replace(' ', '_')) or {}

                # 將 dimensional_breakdown 轉換為 UI 需要的 scores 格式
                breakdown = breed_score.dimensional_breakdown or {}
                ui_scores = {
                    'space': breakdown.get('spatial_compatibility', 0.7),
                    'exercise': breakdown.get('activity_compatibility', 0.7),
                    'grooming': breakdown.get('maintenance_compatibility', 0.7),
                    'experience': breakdown.get('experience_compatibility', 0.7),
                    'noise': breakdown.get('noise_compatibility', 0.7),
                    'family': breakdown.get('family_compatibility', 0.7),
                    'health': breakdown.get('health_compatibility', 0.7),
                    'overall': calibrated_score
                }

                recommendation = {
                    'breed': breed_name,
                    'rank': i + 1,
                    'overall_score': calibrated_score,
                    'final_score': calibrated_score,
                    'semantic_score': breed_score.semantic_component,
                    'attribute_score': breed_score.attribute_component,
                    'bidirectional_bonus': breed_score.bidirectional_bonus,
                    'confidence_score': breed_score.confidence_score,
                    'dimensional_breakdown': breed_score.dimensional_breakdown,
                    'scores': ui_scores,  # UI 需要的格式
                    'explanation': breed_score.explanation,
                    'size': breed_info.get('Size', 'Unknown'),
                    'temperament': breed_info.get('Temperament', ''),
                    'exercise_needs': breed_info.get('Exercise Needs', 'Moderate'),
                    'grooming_needs': breed_info.get('Grooming Needs', 'Moderate'),
                    'good_with_children': breed_info.get('Good with Children', 'Yes'),
                    'lifespan': breed_info.get('Lifespan', '10-12 years'),
                    'description': breed_info.get('Description', ''),
                    'search_type': 'enhanced_description',
                    'calibration_method': calibration_result.calibration_method,
                    'applied_constraints': filter_result.applied_constraints,
                    'relaxed_constraints': filter_result.relaxed_constraints,
                    'warnings': filter_result.warnings
                }

                final_recommendations.append(recommendation)

            # 應用尺寸分佈修正
            corrected_recommendations = self._apply_size_distribution_correction(final_recommendations)

            # 階段 6: 應用智能特徵匹配增強
            intelligence_enhanced_recommendations = self._apply_intelligent_trait_matching(corrected_recommendations, user_input)

            print(f"Generated {len(intelligence_enhanced_recommendations)} enhanced semantic recommendations with intelligent trait matching")
            return intelligence_enhanced_recommendations

        except Exception as e:
            print(f"Error in enhanced semantic recommendations: {str(e)}")
            print(traceback.format_exc())
            # 回退到原始方法
            return self.get_semantic_recommendations(user_input, top_k)

    def get_semantic_recommendations(self, user_input: str, top_k: int = 15) -> List[Dict[str, Any]]:
        """
        基於自然語言描述獲取品種推薦

        Args:
            user_input: 用戶的自然語言描述
            top_k: 返回的推薦數量

        Returns:
            推薦品種列表
        """
        try:
            print(f"Processing user input: {user_input}")

            # 檢查模型是否可用 - 如果不可用，則報錯
            if self.sbert_model is None:
                error_msg = "SBERT model not available. This could be due to:\n• Model download failed\n• Insufficient memory\n• Network connectivity issues\n\nPlease check your environment and try again."
                print(f"ERROR: {error_msg}")
                raise RuntimeError(error_msg)

            # 生成用戶輸入嵌入
            user_embedding = self.vector_manager.encode_text(user_input)

            # 解析比較性偏好
            comparative_prefs = self._parse_comparative_preferences(user_input)

            # 提取生活方式關鍵字
            lifestyle_keywords = self._extract_lifestyle_keywords(user_input)

            # 計算與所有品種的相似度並應用約束
            similarities = []

            for breed, breed_vector in self.breed_vectors.items():
                # 首先應用硬約束
                constraint_penalty = self._apply_hard_constraints(breed, user_input, breed_vector.characteristics)

                # 跳過違反關鍵約束的品種
                if constraint_penalty <= -1.0:  # 完全取消資格
                    continue

                # 基本語義相似度
                semantic_score = cosine_similarity(
                    [user_embedding],
                    [breed_vector.embedding]
                )[0][0]

                # 比較性偏好加權
                comparative_bonus = comparative_prefs.get(breed, 0.0)

                # 生活方式匹配獎勵
                lifestyle_bonus = self._calculate_lifestyle_bonus(
                    breed_vector.characteristics,
                    lifestyle_keywords
                )

                # 應用約束懲罰
                lifestyle_bonus += constraint_penalty

                # 更好分佈的增強組合分數
                # 應用指數縮放以創建更自然的分數分佈
                base_semantic = semantic_score ** 0.8  # 輕微壓縮高分
                enhanced_lifestyle = lifestyle_bonus * 2.0  # 放大生活方式匹配
                enhanced_comparative = comparative_bonus * 1.5  # 放大品種偏好

                final_score = (
                    base_semantic * 0.55 +
                    enhanced_comparative * 0.30 +
                    enhanced_lifestyle * 0.15
                )

                # 添加小的隨機變化以自然地打破平局
                random.seed(hash(breed))  # 對相同品種保持一致
                final_score += random.uniform(-0.03, 0.03)

                # 確保最終分數不超過 1.0
                final_score = min(1.0, final_score)

                similarities.append({
                    'breed': breed,
                    'score': final_score,
                    'semantic_score': semantic_score,
                    'comparative_bonus': comparative_bonus,
                    'lifestyle_bonus': lifestyle_bonus
                })

            # 計算平衡分佈的標準化顯示分數
            breed_display_scores = []

            # 首先，收集所有語義分數以進行標準化
            all_semantic_scores = [breed_data['semantic_score'] for breed_data in similarities]
            semantic_mean = np.mean(all_semantic_scores)
            semantic_std = np.std(all_semantic_scores) if len(all_semantic_scores) > 1 else 1.0

            for breed_data in similarities:
                breed = breed_data['breed']
                base_semantic = breed_data['semantic_score']

                # 標準化語義分數以防止極端異常值
                if semantic_std > 0:
                    normalized_semantic = (base_semantic - semantic_mean) / semantic_std
                    normalized_semantic = max(-2.0, min(2.0, normalized_semantic))  # 限制在 2 個標準差
                    scaled_semantic = 0.5 + (normalized_semantic * 0.1)  # 映射到 0.3-0.7 範圍
                else:
                    scaled_semantic = 0.5

                # 獲取品種特徵
                breed_info = get_dog_description(breed) if breed != 'Unknown' else {}
                breed_size = breed_info.get('Size', '').lower() if breed_info else ''
                exercise_needs = breed_info.get('Exercise Needs', '').lower() if breed_info else ''

                # 計算特徵匹配分數（比純語義相似度更重要）
                feature_score = 0.0
                user_text = user_input.lower()

                # 尺寸和空間需求（高權重）
                if any(term in user_text for term in ['apartment', 'small', 'limited space']):
                    if 'small' in breed_size:
                        feature_score += 0.25
                    elif 'medium' in breed_size:
                        feature_score += 0.05
                    elif 'large' in breed_size or 'giant' in breed_size:
                        feature_score -= 0.30

                # 運動需求（高權重）
                if any(term in user_text for term in ['low exercise', 'minimal exercise', "doesn't need", 'not much']):
                    if 'low' in exercise_needs or 'minimal' in exercise_needs:
                        feature_score += 0.20
                    elif 'high' in exercise_needs or 'very high' in exercise_needs:
                        feature_score -= 0.25
                elif any(term in user_text for term in ['active', 'high exercise', 'running', 'hiking']):
                    if 'high' in exercise_needs:
                        feature_score += 0.20
                    elif 'low' in exercise_needs:
                        feature_score -= 0.15

                # 家庭相容性
                if any(term in user_text for term in ['children', 'kids', 'family']):
                    good_with_children = breed_info.get('Good with Children', '') if breed_info else ''
                    if good_with_children == 'Yes':
                        feature_score += 0.10
                    elif good_with_children == 'No':
                        feature_score -= 0.20

                # 平衡權重組合分數
                final_score = (
                    scaled_semantic * 0.35 +  # 降低語義權重
                    feature_score * 0.45 +    # 增加特徵匹配權重
                    breed_data['lifestyle_bonus'] * 0.15 +
                    breed_data['comparative_bonus'] * 0.05
                )

                # 計算基本相容性分數
                base_compatibility = final_score

                # 應用自然分佈的動態評分
                if base_compatibility >= 0.9:  # 例外匹配
                    score_range = (0.92, 0.98)
                    position = (base_compatibility - 0.9) / 0.1
                elif base_compatibility >= 0.75:  # 優秀匹配
                    score_range = (0.85, 0.91)
                    position = (base_compatibility - 0.75) / 0.15
                elif base_compatibility >= 0.6:  # 良好匹配
                    score_range = (0.75, 0.84)
                    position = (base_compatibility - 0.6) / 0.15
                elif base_compatibility >= 0.45:  # 公平匹配
                    score_range = (0.65, 0.74)
                    position = (base_compatibility - 0.45) / 0.15
                elif base_compatibility >= 0.3:  # 較差匹配
                    score_range = (0.55, 0.64)
                    position = (base_compatibility - 0.3) / 0.15
                else:  # 非常差的匹配
                    score_range = (0.45, 0.54)
                    position = max(0, base_compatibility / 0.3)

                # 計算帶自然變化的最終分數
                score_span = score_range[1] - score_range[0]
                base_score = score_range[0] + (position * score_span)

                # 添加控制的隨機變化以進行自然排名
                random.seed(hash(breed + user_input[:15]))
                variation = random.uniform(-0.015, 0.015)
                display_score = round(max(0.45, min(0.98, base_score + variation)), 3)

                breed_display_scores.append({
                    'breed': breed,
                    'display_score': display_score,
                    'semantic_score': base_semantic,
                    'comparative_bonus': breed_data['comparative_bonus'],
                    'lifestyle_bonus': breed_data['lifestyle_bonus']
                })

            # 計算真實維度分數並整合到排序中
            for breed_data in breed_display_scores:
                breed = breed_data['breed']
                breed_info = get_dog_description(breed)
                real_scores = self._calculate_real_dimension_scores(
                    breed, breed_info, user_input, breed_data['display_score']
                )
                breed_data['real_scores'] = real_scores

                # 計算加權的最終分數（考慮維度分數）
                # 原始顯示分數權重 50%，維度分數平均權重 50%
                dim_scores = [real_scores.get('space', 0.7), real_scores.get('exercise', 0.7),
                             real_scores.get('grooming', 0.7), real_scores.get('experience', 0.7),
                             real_scores.get('noise', 0.7)]
                avg_dim_score = sum(dim_scores) / len(dim_scores)

                # 對低維度分數施加懲罰
                min_dim_score = min(dim_scores)
                penalty = 0
                if min_dim_score < 0.5:
                    penalty = (0.5 - min_dim_score) * 0.3  # 最低分數懲罰

                # 最終排序分數
                breed_data['adjusted_score'] = (
                    breed_data['display_score'] * 0.5 +
                    avg_dim_score * 0.5 -
                    penalty
                )

            # 按調整後的分數排序
            breed_display_scores.sort(key=lambda x: x['adjusted_score'], reverse=True)
            top_breeds = breed_display_scores[:top_k]

            # 轉換為標準推薦格式
            recommendations = []
            for i, breed_data in enumerate(top_breeds):
                breed = breed_data['breed']
                adjusted_score = breed_data['adjusted_score']
                real_scores = breed_data['real_scores']

                # 獲取詳細信息
                breed_info = get_dog_description(breed)

                recommendation = {
                    'breed': breed.replace('_', ' '),
                    'rank': i + 1,
                    'overall_score': adjusted_score,  # 使用調整後的分數
                    'final_score': adjusted_score,    # 確保 final_score 與 overall_score 匹配
                    'semantic_score': breed_data['semantic_score'],
                    'comparative_bonus': breed_data['comparative_bonus'],
                    'lifestyle_bonus': breed_data['lifestyle_bonus'],
                    'size': breed_info.get('Size', 'Unknown') if breed_info else 'Unknown',
                    'temperament': breed_info.get('Temperament', '') if breed_info else '',
                    'exercise_needs': breed_info.get('Exercise Needs', 'Moderate') if breed_info else 'Moderate',
                    'grooming_needs': breed_info.get('Grooming Needs', 'Moderate') if breed_info else 'Moderate',
                    'good_with_children': breed_info.get('Good with Children', 'Yes') if breed_info else 'Yes',
                    'lifespan': breed_info.get('Lifespan', '10-12 years') if breed_info else '10-12 years',
                    'description': breed_info.get('Description', '') if breed_info else '',
                    'search_type': 'description',
                    'scores': real_scores  # 添加真實的維度分數
                }

                recommendations.append(recommendation)

            print(f"Generated {len(recommendations)} semantic recommendations")
            return recommendations

        except Exception as e:
            print(f"Failed to generate semantic recommendations: {str(e)}")
            print(traceback.format_exc())
            return []

    def get_enhanced_recommendations_with_unified_scoring(self, user_input: str, top_k: int = 15) -> List[Dict[str, Any]]:
        """
        增強推薦方法 - 使用完整的多頭評分系統

        這個方法使用:
        - QueryUnderstandingEngine: 解析用戶意圖
        - PriorityDetector: 檢測維度優先級
        - MultiHeadScorer: 多維度評分
        - DynamicWeightCalculator: 動態權重分配
        """
        try:
            print(f"Processing enhanced recommendation with multi-head scoring: {user_input[:50]}...")

            # 使用完整的增強語義推薦系統（包含 multi_head_scorer）
            return self.get_enhanced_semantic_recommendations(user_input, top_k)

        except Exception as e:
            error_msg = f"Enhanced recommendation error: {str(e)}. Please check your description."
            print(f"ERROR: {error_msg}")
            print(traceback.format_exc())
            raise RuntimeError(error_msg) from e

    def _analyze_user_description_enhanced(self, user_description: str) -> Dict[str, Any]:
        """增強用戶描述分析"""
        return self.query_analyzer.analyze_user_description_enhanced(user_description)

    def _create_user_preferences_from_analysis_enhanced(self, analysis: Dict[str, Any]) -> UserPreferences:
        """從分析結果創建用戶偏好物件"""
        return self.query_analyzer.create_user_preferences_from_analysis_enhanced(analysis)

    def _get_candidate_breeds_enhanced(self, analysis: Dict[str, Any]) -> List[str]:
        """獲取候選品種列表"""
        return self.query_analyzer.get_candidate_breeds_enhanced(analysis)

    def _apply_constraint_filtering_enhanced(self, breed: str, analysis: Dict[str, Any]) -> float:
        """應用約束過濾，返回調整分數"""
        # 這個方法需要從 score_calculator 調用適當的方法
        # 但原始實現中沒有這個具體方法，所以我們提供基本實現
        constraint_penalty = 0.0

        breed_info = get_dog_description(breed)
        if not breed_info:
            return constraint_penalty

        # 低噪音要求
        if 'low_noise' in analysis['constraint_requirements']:
            noise_info = breed_noise_info.get(breed, {})
            noise_level = noise_info.get('noise_level', 'moderate').lower()
            if 'high' in noise_level:
                constraint_penalty -= 0.3  # 嚴重扣分
            elif 'low' in noise_level:
                constraint_penalty += 0.1  # 輕微加分

        # 公寓適合性
        if 'apartment_suitable' in analysis['constraint_requirements']:
            size = breed_info.get('Size', '').lower()
            exercise_needs = breed_info.get('Exercise Needs', '').lower()

            if size in ['large', 'giant']:
                constraint_penalty -= 0.2
            elif size in ['small', 'tiny']:
                constraint_penalty += 0.1

            if 'high' in exercise_needs:
                constraint_penalty -= 0.15

        # 兒童友善性
        if 'child_friendly' in analysis['constraint_requirements']:
            good_with_children = breed_info.get('Good with Children', 'Unknown')
            if good_with_children == 'Yes':
                constraint_penalty += 0.15
            elif good_with_children == 'No':
                constraint_penalty -= 0.4  # 嚴重扣分

        return constraint_penalty

    def _get_breed_characteristics_enhanced(self, breed: str) -> Dict[str, Any]:
        """獲取品種特徵"""
        return self.score_calculator.get_breed_characteristics_enhanced(breed)

    def get_hybrid_recommendations(self, user_description: str,
                                 user_preferences: Optional[Any] = None,
                                 top_k: int = 15) -> List[Dict[str, Any]]:
        """
        混合推薦：結合語義匹配與傳統評分

        Args:
            user_description: 用戶的自然語言描述
            user_preferences: 可選的結構化偏好設置
            top_k: 返回的推薦數量

        Returns:
            混合推薦結果
        """
        try:
            # 獲取語義推薦
            semantic_recommendations = self.get_semantic_recommendations(user_description, top_k * 2)

            if not user_preferences:
                return semantic_recommendations[:top_k]

            # 與傳統評分結合
            hybrid_results = []

            for semantic_rec in semantic_recommendations:
                breed_name = semantic_rec['breed'].replace(' ', '_')

                # 計算傳統相容性分數
                traditional_score = calculate_compatibility_score(user_preferences, breed_name)

                # 混合分數（語義 40% + 傳統 60%）
                hybrid_score = (
                    semantic_rec['overall_score'] * 0.4 +
                    traditional_score * 0.6
                )

                semantic_rec['hybrid_score'] = hybrid_score
                semantic_rec['traditional_score'] = traditional_score
                hybrid_results.append(semantic_rec)

            # 按混合分數重新排序
            hybrid_results.sort(key=lambda x: x['hybrid_score'], reverse=True)

            # 更新排名
            for i, result in enumerate(hybrid_results[:top_k]):
                result['rank'] = i + 1
                result['overall_score'] = result['hybrid_score']

            return hybrid_results[:top_k]

        except Exception as e:
            print(f"Hybrid recommendation failed: {str(e)}")
            print(traceback.format_exc())
            return self.get_semantic_recommendations(user_description, top_k)


def get_breed_recommendations_by_description(user_description: str,
                                           user_preferences: Optional[Any] = None,
                                           top_k: int = 15) -> List[Dict[str, Any]]:
    """基於描述獲取品種推薦的主要介面函數"""
    try:
        print("Initializing Enhanced SemanticBreedRecommender...")
        recommender = SemanticBreedRecommender()

        # 優先使用整合統一評分系統的增強推薦
        print("Using enhanced recommendation system with unified scoring")
        results = recommender.get_enhanced_recommendations_with_unified_scoring(user_description, top_k)

        if results and len(results) > 0:
            print(f"Generated {len(results)} enhanced recommendations successfully")
            return results
        else:
            # 如果增強系統無結果，嘗試原有增強系統
            print("Enhanced unified system returned no results, trying original enhanced system")
            results = recommender.get_enhanced_semantic_recommendations(user_description, top_k)

            if results and len(results) > 0:
                return results
            else:
                # 最後回退到標準系統
                print("All enhanced systems failed, using standard system")
                if user_preferences:
                    results = recommender.get_hybrid_recommendations(user_description, user_preferences, top_k)
                else:
                    results = recommender.get_semantic_recommendations(user_description, top_k)

                if not results:
                    error_msg = f"All recommendation systems failed to generate results. Please check your input description and try again. Error details may be in the console."
                    print(f"ERROR: {error_msg}")
                    raise RuntimeError(error_msg)
                return results

    except Exception as e:
        error_msg = f"Critical error in recommendation system: {str(e)}. Please check your input and system configuration."
        print(f"ERROR: {error_msg}")
        print(traceback.format_exc())
        raise RuntimeError(error_msg) from e


def get_enhanced_recommendations_with_unified_scoring(user_description: str, top_k: int = 15) -> List[Dict[str, Any]]:
    """
    模組層級便利函數 - 使用完整的多頭評分系統

    這個函數呼叫 SemanticBreedRecommender 的增強推薦方法，使用:
    - QueryUnderstandingEngine: 解析用戶意圖
    - PriorityDetector: 檢測維度優先級
    - MultiHeadScorer: 多維度評分
    - DynamicWeightCalculator: 動態權重分配
    - SmartBreedFilter: 智慧風險過濾（只對真正危害用戶的情況干預）

    如果增強系統失敗，會自動回退到基本語義推薦
    """
    try:
        print(f"Processing description-based recommendation with multi-head scoring: {user_description[:50]}...")

        # 創建推薦器實例
        recommender = SemanticBreedRecommender()

        # 檢查 SBERT 模型是否可用
        if not recommender.vector_manager.is_model_available():
            print("SBERT model not available, using basic text matching...")
            results = _get_basic_text_matching_recommendations(user_description, top_k, recommender)
            # 應用智慧過濾
            results = apply_smart_filtering(results, user_description)
            return results

        # 嘗試使用完整的增強語義推薦系統
        try:
            results = recommender.get_enhanced_semantic_recommendations(user_description, top_k)
            if results:
                # 應用智慧過濾
                results = apply_smart_filtering(results, user_description)
                return results
            else:
                print("Enhanced recommendations returned empty, falling back to basic semantic...")
        except Exception as enhanced_error:
            print(f"Enhanced recommendation failed: {str(enhanced_error)}, falling back to basic semantic...")
            print(traceback.format_exc())

        # 回退到基本語義推薦
        try:
            results = recommender.get_semantic_recommendations(user_description, top_k)
            if results:
                # 應用智慧過濾
                results = apply_smart_filtering(results, user_description)
                return results
        except Exception as semantic_error:
            print(f"Basic semantic recommendation also failed: {str(semantic_error)}")

        # 最後回退到基本文字匹配
        print("All semantic methods failed, using basic text matching as last resort...")
        results = _get_basic_text_matching_recommendations(user_description, top_k, recommender)
        # 應用智慧過濾
        results = apply_smart_filtering(results, user_description)
        return results

    except Exception as e:
        error_msg = f"Error in semantic recommendation system: {str(e)}. Please check your input and try again."
        print(f"ERROR: {error_msg}")
        print(traceback.format_exc())
        raise RuntimeError(error_msg) from e

def _get_basic_text_matching_recommendations(user_description: str, top_k: int = 15, recommender=None) -> List[Dict[str, Any]]:
    """基本文字匹配推薦（SBERT 不可用時的後備方案）"""
    try:
        print("Using basic text matching as fallback...")

        # 如果沒有提供 recommender，創建一個新的
        if recommender is None:
            recommender = SemanticBreedRecommender()

        # 基本關鍵字匹配
        keywords = user_description.lower().split()
        breed_scores = []

        # 從數據庫獲取品種清單或使用預設清單
        try:
            conn = sqlite3.connect('animal_detector.db')
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT Breed FROM AnimalCatalog LIMIT 50")
            basic_breeds = [row[0] for row in cursor.fetchall()]
            cursor.close()
            conn.close()
            # 過濾掉野生動物品種
            basic_breeds = [breed for breed in basic_breeds if breed != 'Dhole']
        except Exception as e:
            print(f"Could not load breed list from database: {str(e)}")
            # 後備品種清單
            basic_breeds = [
                'Labrador_Retriever', 'Golden_Retriever', 'German_Shepherd', 'French_Bulldog',
                'Border_Collie', 'Poodle', 'Beagle', 'Rottweiler', 'Yorkshire_Terrier',
                'Dachshund', 'Boxer', 'Siberian_Husky', 'Great_Dane', 'Pomeranian', 'Shih_Tzu',
                'Maltese_Dog', 'Chihuahua', 'Cavalier_King_Charles_Spaniel', 'Boston_Terrier',
                'Japanese_Spaniel', 'Toy_Terrier', 'Affenpinscher', 'Pekingese', 'Lhasa'
            ]

        # 應用約束過濾 - 關鍵修復！
        try:
            from constraint_manager import ConstraintManager
            from query_understanding import QueryUnderstandingEngine

            query_engine = QueryUnderstandingEngine()
            dimensions = query_engine.analyze_query(user_description)
            constraint_manager = ConstraintManager()
            filter_result = constraint_manager.apply_constraints(dimensions)

            # 只保留通過約束的品種
            allowed_breeds = filter_result.passed_breeds
            filtered_count = len(basic_breeds)
            basic_breeds = [b for b in basic_breeds if b in allowed_breeds]
            print(f"Constraint filtering: {filtered_count} -> {len(basic_breeds)} breeds")

            # 記錄被過濾的原因（用於調試）
            for breed, reason in filter_result.filtered_breeds.items():
                if breed in ['Italian_Greyhound', 'Rottweiler', 'Malinois']:
                    print(f"  Filtered {breed}: {reason}")
        except Exception as e:
            print(f"Warning: Could not apply constraints: {str(e)}")

        for breed in basic_breeds:
            breed_info = get_dog_description(breed) or {}
            breed_text = f"{breed} {breed_info.get('Temperament', '')} {breed_info.get('Size', '')} {breed_info.get('Description', '')}".lower()

            # 計算關鍵字匹配分數
            matches = sum(1 for keyword in keywords if keyword in breed_text)
            base_score = min(0.95, 0.3 + (matches / len(keywords)) * 0.6)

            # 應用增強匹配邏輯
            enhanced_score = recommender.score_calculator.calculate_enhanced_matching_score(
                breed, breed_info, user_description, base_score
            )

            breed_scores.append((breed, enhanced_score['final_score'], breed_info, enhanced_score))

        # 按分數排序
        breed_scores.sort(key=lambda x: x[1], reverse=True)

        recommendations = []
        for i, (breed, final_score, breed_info, enhanced_score) in enumerate(breed_scores[:top_k]):
            recommendation = {
                'breed': breed.replace('_', ' '),
                'rank': i + 1,
                'overall_score': final_score,
                'final_score': final_score,
                'semantic_score': enhanced_score.get('weighted_score', final_score),
                'comparative_bonus': enhanced_score.get('lifestyle_bonus', 0.0),
                'lifestyle_bonus': enhanced_score.get('lifestyle_bonus', 0.0),
                'size': breed_info.get('Size', 'Unknown'),
                'temperament': breed_info.get('Temperament', 'Unknown'),
                'exercise_needs': breed_info.get('Exercise Needs', 'Moderate'),
                'grooming_needs': breed_info.get('Grooming Needs', 'Moderate'),
                'good_with_children': breed_info.get('Good with Children', 'Unknown'),
                'lifespan': breed_info.get('Lifespan', '10-12 years'),
                'description': breed_info.get('Description', 'No description available'),
                'search_type': 'description',
                'scores': enhanced_score.get('dimension_scores', {
                    'space': final_score * 0.9,
                    'exercise': final_score * 0.85,
                    'grooming': final_score * 0.8,
                    'experience': final_score * 0.75,
                    'noise': final_score * 0.7,
                    'family': final_score * 0.65
                })
            }
            recommendations.append(recommendation)

        return recommendations

    except Exception as e:
        error_msg = f"Error in basic text matching: {str(e)}"
        print(f"ERROR: {error_msg}")
        raise RuntimeError(error_msg) from e
