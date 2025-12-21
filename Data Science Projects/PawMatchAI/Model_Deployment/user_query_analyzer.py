import random
import hashlib
import numpy as np
import sqlite3
import re
import traceback
from typing import List, Dict, Tuple, Optional, Any, Set
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
from priority_detector import PriorityDetector, PriorityDetectionResult
from inference_engine import BreedRecommendationInferenceEngine, InferenceResult

class UserQueryAnalyzer:
    """
    用戶查詢分析器
    專門處理用戶輸入分析、生活方式關鍵字提取和偏好解析
    """

    def __init__(self, breed_list: List[str]):
        """初始化用戶查詢分析器"""
        self.breed_list = breed_list
        self.priority_detector = PriorityDetector()
        self.inference_engine = BreedRecommendationInferenceEngine()
        self.comparative_keywords = {
            'most': 1.0, 'love': 1.0, 'prefer': 0.9, 'like': 0.8,
            'then': 0.7, 'second': 0.7, 'followed': 0.6,
            'third': 0.5, 'least': 0.3, 'dislike': 0.2
        }
        self.stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            'from', 'up', 'about', 'into', 'through', 'during', 'before', 'after', 'above', 'below',
            'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
            'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'i', 'me', 'my', 'myself',
            'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he',
            'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they',
            'them', 'their', 'theirs', 'themselves'
        }

    def parse_comparative_preferences(self, user_input: str) -> Dict[str, float]:
        """解析比較性偏好表達"""
        breed_scores = {}

        # 標準化輸入
        text = user_input.lower()

        # 找到品種名稱和偏好關鍵字
        for breed in self.breed_list:
            breed_display = breed.replace('_', ' ').lower()
            breed_words = breed_display.split()

            # 檢查是否提到此品種
            breed_mentioned = False
            for word in breed_words:
                if word in text:
                    breed_mentioned = True
                    break

            if breed_mentioned:
                # 在附近找到偏好關鍵字
                breed_score = 0.5  # 預設分數

                # 在品種名稱 50 字符內尋找關鍵字
                breed_pos = text.find(breed_words[0])
                if breed_pos != -1:
                    # 檢查背景中的關鍵字
                    context_start = max(0, breed_pos - 50)
                    context_end = min(len(text), breed_pos + 50)
                    context = text[context_start:context_end]

                    for keyword, score in self.comparative_keywords.items():
                        if keyword in context:
                            breed_score = max(breed_score, score)

                breed_scores[breed] = breed_score

        return breed_scores

    def _merge_priorities(self,
                         explicit_priorities: Dict[str, float],
                         implicit_priorities: Dict[str, float]) -> Dict[str, float]:
        """
        合併顯式和隱式優先級

        規則:
        1. 明確提及的維度，使用明確優先級
        2. 未明確提及但推斷出的維度，使用隱含優先級
        3. 隱含優先級不會覆蓋明確優先級

        Args:
            explicit_priorities: 明確優先級
            implicit_priorities: 隱含優先級

        Returns:
            Dict[str, float]: 合併後的優先級
        """
        merged = explicit_priorities.copy()

        for dim, implicit_score in implicit_priorities.items():
            if dim not in merged:
                # 只添加未明確提及的隱含優先級
                merged[dim] = implicit_score
            # 如果已有明確優先級，保持不變

        return merged

    def extract_lifestyle_keywords(self, user_input: str) -> Dict[str, List[str]]:
        """增強的生活方式關鍵字提取，具有更好的模式匹配"""
        keywords = {
            'living_space': [],
            'activity_level': [],
            'family_situation': [],
            'noise_preference': [],
            'size_preference': [],
            'care_level': [],
            'special_needs': [],
            'intelligence_preference': [],
            'grooming_preference': [],
            'lifespan_preference': [],
            'temperament_preference': [],
            'experience_level': []
        }

        text = user_input.lower()

        # 增強居住空間檢測
        apartment_terms = ['apartment', 'flat', 'condo', 'small space', 'city living', 'urban', 'no yard', 'indoor']
        house_terms = ['house', 'yard', 'garden', 'backyard', 'large space', 'suburban', 'rural', 'farm']

        if any(term in text for term in apartment_terms):
            keywords['living_space'].append('apartment')
        if any(term in text for term in house_terms):
            keywords['living_space'].append('house')

        # 增強活動水平檢測
        high_activity = ['active', 'energetic', 'exercise', 'hiking', 'running', 'outdoor', 'sports', 'jogging',
                        'athletic', 'adventure', 'vigorous', 'high energy', 'workout']
        low_activity = ['calm', 'lazy', 'indoor', 'low energy', 'couch', 'sedentary', 'relaxed',
                       'peaceful', 'quiet lifestyle', 'minimal exercise']
        moderate_activity = ['moderate', 'walk', 'daily walks', 'light exercise']

        if any(term in text for term in high_activity):
            keywords['activity_level'].append('high')
        if any(term in text for term in low_activity):
            keywords['activity_level'].append('low')
        if any(term in text for term in moderate_activity):
            keywords['activity_level'].append('moderate')

        # 增強家庭情況檢測
        children_terms = ['children', 'kids', 'family', 'child', 'toddler', 'baby', 'teenage', 'school age']
        elderly_terms = ['elderly', 'senior', 'old', 'retirement', 'aged', 'mature']
        single_terms = ['single', 'alone', 'individual', 'solo', 'myself']

        if any(term in text for term in children_terms):
            keywords['family_situation'].append('children')
        if any(term in text for term in elderly_terms):
            keywords['family_situation'].append('elderly')
        if any(term in text for term in single_terms):
            keywords['family_situation'].append('single')

        # 增強噪音偏好檢測
        quiet_terms = ['quiet', 'silent', 'noise-sensitive', 'peaceful', 'no barking', 'minimal noise',
                      'soft-spoken', 'calm', 'tranquil']
        noise_ok_terms = ['loud', 'barking ok', 'noise tolerant', 'vocal', 'doesn\'t matter']

        if any(term in text for term in quiet_terms):
            keywords['noise_preference'].append('low')
        if any(term in text for term in noise_ok_terms):
            keywords['noise_preference'].append('high')

        # 增強體型偏好檢測
        small_terms = ['small', 'tiny', 'little', 'compact', 'miniature', 'toy', 'lap dog']
        large_terms = ['large', 'big', 'giant', 'huge', 'massive', 'great']
        medium_terms = ['medium', 'moderate size', 'average', 'mid-sized']

        if any(term in text for term in small_terms):
            keywords['size_preference'].append('small')
        if any(term in text for term in large_terms):
            keywords['size_preference'].append('large')
        if any(term in text for term in medium_terms):
            keywords['size_preference'].append('medium')

        # 增強照護水平檢測
        low_care = ['low maintenance', 'easy care', 'simple', 'minimal grooming', 'wash and go']
        high_care = ['high maintenance', 'grooming', 'care intensive', 'professional grooming', 'daily brushing']

        if any(term in text for term in low_care):
            keywords['care_level'].append('low')
        if any(term in text for term in high_care):
            keywords['care_level'].append('high')

        # 智力偏好檢測（新增）
        smart_terms = ['smart', 'intelligent', 'clever', 'bright', 'quick learner', 'easy to train', 'trainable', 'genius', 'brilliant']
        independent_terms = ['independent', 'stubborn', 'strong-willed', 'less trainable', 'thinks for themselves']

        if any(term in text for term in smart_terms):
            keywords['intelligence_preference'].append('high')
        if any(term in text for term in independent_terms):
            keywords['intelligence_preference'].append('independent')

        # 美容偏好檢測（新增）
        low_grooming_terms = ['low grooming', 'minimal grooming', 'easy care', 'wash and wear', 'no grooming', 'simple coat']
        high_grooming_terms = ['high grooming', 'professional grooming', 'lots of care', 'high maintenance coat', 'daily brushing', 'regular grooming']

        if any(term in text for term in low_grooming_terms):
            keywords['grooming_preference'].append('low')
        if any(term in text for term in high_grooming_terms):
            keywords['grooming_preference'].append('high')

        # 壽命偏好檢測（新增）
        long_lived_terms = ['long lived', 'long lifespan', 'live long', 'many years', '15+ years', 'longevity']
        healthy_terms = ['healthy breed', 'few health issues', 'robust', 'hardy', 'strong constitution']

        if any(term in text for term in long_lived_terms):
            keywords['lifespan_preference'].append('long')
        if any(term in text for term in healthy_terms):
            keywords['lifespan_preference'].append('healthy')

        # 氣質偏好檢測（新增）
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

        # 經驗水平檢測（新增）
        beginner_terms = ['first time', 'beginner', 'new to dogs', 'never had', 'novice', 'inexperienced']
        advanced_terms = ['experienced', 'advanced', 'dog expert', 'many dogs before', 'professional', 'seasoned']

        if any(term in text for term in beginner_terms):
            keywords['experience_level'].append('beginner')
        if any(term in text for term in advanced_terms):
            keywords['experience_level'].append('advanced')

        # 增強特殊需求檢測
        guard_terms = ['guard', 'protection', 'security', 'watchdog', 'protective', 'defender']
        companion_terms = ['therapy', 'emotional support', 'companion', 'comfort', 'lap dog', 'cuddly']
        hypoallergenic_terms = ['hypoallergenic', 'allergies', 'non-shedding', 'allergy-friendly', 'no shed']
        multi_pet_terms = ['good with cats', 'cat friendly', 'multi-pet', 'other animals']

        if any(term in text for term in guard_terms):
            keywords['special_needs'].append('guard')
        if any(term in text for term in companion_terms):
            keywords['special_needs'].append('companion')
        if any(term in text for term in hypoallergenic_terms):
            keywords['special_needs'].append('hypoallergenic')
        if any(term in text for term in multi_pet_terms):
            keywords['special_needs'].append('multi_pet')

        return keywords

    def preprocess_text(self, text: str) -> str:
        """預處理文本"""
        # 轉換為小寫
        text = text.lower()

        # 移除特殊字符，保留字母、數字和基本標點
        text = re.sub(r'[^\w\s\-\']', ' ', text)

        # 標準化空格
        text = ' '.join(text.split())

        return text

    def generate_search_keywords(self, text: str) -> List[str]:
        """
        為語義搜索生成關鍵字

        Args:
            text: 輸入文本

        Returns:
            關鍵字列表
        """
        text = self.preprocess_text(text)
        keywords = []

        try:
            # 分詞並過濾停用詞
            words = text.split()
            for word in words:
                if len(word) > 2 and word not in self.stop_words:
                    keywords.append(word)

            # 提取重要短語
            phrases = self._extract_phrases(text)
            keywords.extend(phrases)

            # 移除重複項
            keywords = list(set(keywords))

            return keywords

        except Exception as e:
            print(f"Error generating search keywords: {str(e)}")
            return []

    def _extract_phrases(self, text: str) -> List[str]:
        """
        提取重要短語

        Args:
            text: 輸入文本

        Returns:
            短語列表
        """
        phrases = []

        # 定義重要短語模式
        phrase_patterns = [
            r'good with \w+',
            r'apartment \w+',
            r'family \w+',
            r'exercise \w+',
            r'grooming \w+',
            r'noise \w+',
            r'training \w+',
            r'health \w+',
            r'\w+ friendly',
            r'\w+ tolerant',
            r'\w+ maintenance',
            r'\w+ energy',
            r'\w+ barking',
            r'\w+ shedding'
        ]

        for pattern in phrase_patterns:
            matches = re.findall(pattern, text)
            phrases.extend(matches)

        return phrases

    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """
        分析文本情感

        Args:
            text: 輸入文本

        Returns:
            情感分析結果
        """
        # 簡化的情感分析實現
        positive_words = [
            'love', 'like', 'prefer', 'enjoy', 'want', 'need', 'looking for',
            'good', 'great', 'excellent', 'perfect', 'wonderful', 'amazing'
        ]

        negative_words = [
            'hate', 'dislike', 'avoid', 'don\'t want', 'no', 'not',
            'bad', 'terrible', 'awful', 'horrible', 'worst', 'never'
        ]

        words = text.lower().split()
        positive_count = sum(1 for word in words if word in positive_words)
        negative_count = sum(1 for word in words if word in negative_words)
        total_words = len(words)

        if total_words == 0:
            return {'positive': 0.5, 'negative': 0.5, 'neutral': 0.0}

        positive_score = positive_count / total_words
        negative_score = negative_count / total_words
        neutral_score = max(0, 1 - positive_score - negative_score)

        return {
            'positive': positive_score,
            'negative': negative_score,
            'neutral': neutral_score
        }

    def parse_user_requirements(self, user_input: str) -> Dict[str, Any]:
        """更準確地解析用戶需求"""
        requirements = {
            'living_space': None,
            'exercise_level': None,
            'preferred_size': None,
            'noise_tolerance': None
        }

        input_lower = user_input.lower()

        # 居住空間檢測
        if 'apartment' in input_lower or 'small' in input_lower:
            requirements['living_space'] = 'apartment'
        elif 'large house' in input_lower or 'big' in input_lower:
            requirements['living_space'] = 'large_house'
        elif 'medium' in input_lower:
            requirements['living_space'] = 'medium_house'

        # 運動水平檢測
        if "don't exercise" in input_lower or 'low exercise' in input_lower:
            requirements['exercise_level'] = 'low'
        elif any(term in input_lower for term in ['hiking', 'running', 'active']):
            requirements['exercise_level'] = 'high'
        elif '30 minutes' in input_lower or 'moderate' in input_lower:
            requirements['exercise_level'] = 'moderate'

        # 體型偏好檢測
        if any(term in input_lower for term in ['small dog', 'tiny', 'toy']):
            requirements['preferred_size'] = 'small'
        elif any(term in input_lower for term in ['large dog', 'big dog']):
            requirements['preferred_size'] = 'large'
        elif 'medium' in input_lower:
            requirements['preferred_size'] = 'medium'

        return requirements

    def analyze_user_description_enhanced(self, user_description: str) -> Dict[str, Any]:
        """增強用戶描述分析"""
        text = user_description.lower()
        analysis = {
            'mentioned_breeds': [],
            'lifestyle_keywords': {},
            'preference_strength': {},
            'constraint_requirements': [],
            'user_context': {}
        }

        # 提取提及的品種
        for breed in self.breed_list:
            breed_display = breed.replace('_', ' ').lower()
            if breed_display in text or any(word in text for word in breed_display.split()):
                analysis['mentioned_breeds'].append(breed)
                # 簡單偏好強度分析
                if any(word in text for word in ['love', 'prefer', 'like', '喜歡', '最愛']):
                    analysis['preference_strength'][breed] = 0.8
                else:
                    analysis['preference_strength'][breed] = 0.5

        # 提取約束要求
        if any(word in text for word in ['quiet', 'silent', 'no barking', '安靜']):
            analysis['constraint_requirements'].append('low_noise')
        if any(word in text for word in ['apartment', 'small space', '公寓']):
            analysis['constraint_requirements'].append('apartment_suitable')
        if any(word in text for word in ['children', 'kids', 'family', '小孩']):
            analysis['constraint_requirements'].append('child_friendly')

        # 提取用戶背景
        analysis['user_context'] = {
            'has_children': any(word in text for word in ['children', 'kids', '小孩']),
            'living_space': 'apartment' if any(word in text for word in ['apartment', '公寓']) else 'house',
            'activity_level': 'high' if any(word in text for word in ['active', 'energetic', '活躍']) else 'moderate',
            'noise_sensitive': any(word in text for word in ['quiet', 'silent', '安靜']),
            'experience_level': 'beginner' if any(word in text for word in ['first time', 'beginner', '新手']) else 'intermediate'
        }

        # 優先級檢測與推理
        try:
            # Step 1: 檢測明確優先級
            priority_result = self.priority_detector.detect_priorities(user_description)
            explicit_priorities = priority_result.dimension_priorities

            # Step 2: 推斷隱含優先級
            inference_result = self.inference_engine.infer_implicit_priorities(
                user_description,
                analysis['user_context']
            )
            implicit_priorities = inference_result.implicit_priorities

            # Step 3: 合併優先級（明確優先級 > 隱含優先級）
            final_priorities = self._merge_priorities(explicit_priorities, implicit_priorities)

            # 添加到分析結果
            analysis['dimension_priorities'] = final_priorities
            analysis['explicit_priorities'] = explicit_priorities
            analysis['implicit_priorities'] = implicit_priorities
            analysis['priority_detection_confidence'] = priority_result.detection_confidence
            analysis['inference_confidence'] = inference_result.confidence

        except Exception as e:
            print(f"Error in priority detection/inference: {str(e)}")
            analysis['dimension_priorities'] = {}
            analysis['explicit_priorities'] = {}
            analysis['implicit_priorities'] = {}

        return analysis

    def create_user_preferences_from_analysis_enhanced(self, analysis: Dict[str, Any]) -> 'UserPreferences':
        """從分析結果創建用戶偏好物件"""
        context = analysis['user_context']

        # 推斷居住空間類型
        living_space = 'apartment' if context.get('living_space') == 'apartment' else 'house_small'

        # 推斷院子權限
        yard_access = 'no_yard' if living_space == 'apartment' else 'shared_yard'

        # 推斷運動時間
        activity_level = context.get('activity_level', 'moderate')
        exercise_time_map = {'high': 120, 'moderate': 60, 'low': 30}
        exercise_time = exercise_time_map.get(activity_level, 60)

        # 推斷運動類型
        exercise_type_map = {'high': 'active_training', 'moderate': 'moderate_activity', 'low': 'light_walks'}
        exercise_type = exercise_type_map.get(activity_level, 'moderate_activity')

        # 推斷噪音容忍度
        noise_tolerance = 'low' if context.get('noise_sensitive', False) else 'medium'

        return UserPreferences(
            living_space=living_space,
            yard_access=yard_access,
            exercise_time=exercise_time,
            exercise_type=exercise_type,
            grooming_commitment='medium',
            experience_level=context.get('experience_level', 'intermediate'),
            time_availability='moderate',
            has_children=context.get('has_children', False),
            children_age='school_age' if context.get('has_children', False) else None,
            noise_tolerance=noise_tolerance,
            space_for_play=(living_space != 'apartment'),
            other_pets=False,
            climate='moderate',
            health_sensitivity='medium',
            barking_acceptance=noise_tolerance,
            size_preference='no_preference'
        )

    def get_candidate_breeds_enhanced(self, analysis: Dict[str, Any]) -> List[str]:
        """獲取候選品種列表"""
        candidate_breeds = set()

        # 如果提及特定品種，優先包含
        if analysis['mentioned_breeds']:
            candidate_breeds.update(analysis['mentioned_breeds'])

        # 根據約束要求過濾品種
        if 'apartment_suitable' in analysis['constraint_requirements']:
            apartment_suitable = [
                'French_Bulldog', 'Cavalier_King_Charles_Spaniel', 'Boston_Terrier',
                'Pug', 'Bichon_Frise', 'Cocker_Spaniel', 'Yorkshire_Terrier', 'Shih_Tzu'
            ]
            candidate_breeds.update(breed for breed in apartment_suitable if breed in self.breed_list)

        if 'child_friendly' in analysis['constraint_requirements']:
            child_friendly = [
                'Labrador_Retriever', 'Golden_Retriever', 'Beagle', 'Cavalier_King_Charles_Spaniel',
                'Bichon_Frise', 'Poodle', 'Cocker_Spaniel'
            ]
            candidate_breeds.update(breed for breed in child_friendly if breed in self.breed_list)

        # 如果候選品種不足，添加更多通用品種
        if len(candidate_breeds) < 20:
            general_breeds = [
                'Labrador_Retriever', 'German_Shepherd', 'Golden_Retriever', 'French_Bulldog',
                'Bulldog', 'Poodle', 'Beagle', 'Rottweiler', 'Yorkshire_Terrier', 'Boston_Terrier',
                'Border_Collie', 'Siberian_Husky', 'Cavalier_King_Charles_Spaniel', 'Boxer',
                'Bichon_Frise', 'Cocker_Spaniel', 'Shih_Tzu', 'Pug', 'Chihuahua'
            ]
            candidate_breeds.update(breed for breed in general_breeds if breed in self.breed_list)

        return list(candidate_breeds)[:30]  # 限制候選數量以提高效率
