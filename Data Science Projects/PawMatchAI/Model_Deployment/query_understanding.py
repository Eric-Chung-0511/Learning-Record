# %%writefile query_understanding.py
import re
import json
import numpy as np
import sqlite3
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import traceback
from sentence_transformers import SentenceTransformer
from dog_database import get_dog_description
from breed_health_info import breed_health_info
from breed_noise_info import breed_noise_info
from priority_detector import PriorityDetector

@dataclass
class QueryDimensions:
    """Structured query intent data structure"""
    spatial_constraints: List[str] = field(default_factory=list)
    activity_level: List[str] = field(default_factory=list)
    noise_preferences: List[str] = field(default_factory=list)
    size_preferences: List[str] = field(default_factory=list)
    family_context: List[str] = field(default_factory=list)
    maintenance_level: List[str] = field(default_factory=list)
    experience_level: List[str] = field(default_factory=list)  # 用戶經驗等級
    special_requirements: List[str] = field(default_factory=list)
    breed_mentions: List[str] = field(default_factory=list)
    confidence_scores: Dict[str, float] = field(default_factory=dict)
    dimension_priorities: Dict[str, float] = field(default_factory=dict)

@dataclass
class DimensionalSynonyms:
    """Dimensional synonyms dictionary structure"""
    spatial: Dict[str, List[str]] = field(default_factory=dict)
    activity: Dict[str, List[str]] = field(default_factory=dict)
    noise: Dict[str, List[str]] = field(default_factory=dict)
    size: Dict[str, List[str]] = field(default_factory=dict)
    family: Dict[str, List[str]] = field(default_factory=dict)
    maintenance: Dict[str, List[str]] = field(default_factory=dict)
    special: Dict[str, List[str]] = field(default_factory=dict)

class QueryUnderstandingEngine:
    """
    多維度語義查詢理解引擎
    支援中英文自然語言理解並轉換為結構化品種推薦查詢
    """

    def __init__(self):
        """初始化查詢理解引擎"""
        self.sbert_model = None
        self._sbert_loading_attempted = False
        self.breed_list = self._load_breed_list()
        self.synonyms = self._initialize_synonyms()
        self.semantic_templates = {}
        self.priority_detector = PriorityDetector()  # 初始化優先級檢測器
        # 延遲SBERT載入直到需要時才在GPU環境中進行
        print("QueryUnderstandingEngine initialized (SBERT loading deferred)")

    def _load_breed_list(self) -> List[str]:
        """載入品種清單"""
        try:
            conn = sqlite3.connect('animal_detector.db')
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT Breed FROM AnimalCatalog")
            breeds = [row[0] for row in cursor.fetchall()]
            cursor.close()
            conn.close()
            return breeds
        except Exception as e:
            print(f"Error loading breed list: {str(e)}")
            # 備用品種清單
            return ['Labrador_Retriever', 'German_Shepherd', 'Golden_Retriever',
                   'Bulldog', 'Poodle', 'Beagle', 'Border_Collie', 'Yorkshire_Terrier']

    def _initialize_sbert_model(self):
        """初始化 SBERT 模型 - 延遲載入以避免ZeroGPU CUDA初始化問題"""
        if self.sbert_model is not None or getattr(self, '_sbert_loading_attempted', False):
            return self.sbert_model
            
        try:
            print("Loading SBERT model for query understanding in GPU context...")
            model_options = ['all-MiniLM-L6-v2', 'all-mpnet-base-v2', 'all-MiniLM-L12-v2']

            for model_name in model_options:
                try:
                    import torch
                    device = 'cuda' if torch.cuda.is_available() else 'cpu'
                    self.sbert_model = SentenceTransformer(model_name, device=device)
                    print(f"SBERT model {model_name} loaded successfully for query understanding on {device}")
                    return self.sbert_model
                except Exception as e:
                    print(f"Failed to load {model_name}: {str(e)}")
                    continue

            print("All SBERT models failed to load. Using keyword-only analysis.")
            self.sbert_model = None
            return None

        except Exception as e:
            print(f"Failed to initialize SBERT model: {str(e)}")
            self.sbert_model = None
            return None
        finally:
            self._sbert_loading_attempted = True

    def _initialize_synonyms(self) -> DimensionalSynonyms:
        """初始化多維度同義詞字典"""
        return DimensionalSynonyms(
            spatial={
                'apartment': ['apartment', 'flat', 'condo', 'small space', 'city living',
                             'urban', 'no yard', 'indoor'],
                'house': ['house', 'home', 'yard', 'garden', 'backyard', 'large space',
                         'suburban', 'rural', 'farm']
            },
            activity={
                'low': ['low activity', 'sedentary', 'couch potato', 'minimal exercise',
                       'indoor lifestyle', 'lazy', 'calm'],
                'moderate': ['moderate activity', 'daily walks', 'light exercise',
                           'regular walks'],
                'high': ['high activity', 'energetic', 'active', 'exercise', 'hiking',
                        'running', 'jogging', 'outdoor sports']
            },
            noise={
                'low': ['quiet', 'silent', 'no barking', 'peaceful', 'low noise',
                       'rarely barks', 'soft-spoken'],
                'moderate': ['moderate barking', 'occasional barking'],
                'high': ['loud', 'barking', 'vocal', 'noisy', 'frequent barking',
                        'alert dog']
            },
            size={
                'small': ['small', 'tiny', 'little', 'compact', 'miniature', 'toy',
                         'lap dog'],
                'medium': ['medium', 'moderate size', 'average', 'mid-sized'],
                'large': ['large', 'big', 'giant', 'huge', 'massive', 'great']
            },
            family={
                'children': ['children', 'kids', 'family', 'child-friendly', 'toddler',
                           'baby', 'school age', 'young kids', 'young children',
                           'aged 1', 'aged 2', 'aged 3', 'aged 4', 'aged 5',
                           '1 year', '2 year', '3 year', '4 year', '5 year',
                           'infant', 'preschool'],
                'elderly': ['elderly', 'senior', 'old people', 'retirement', 'aged', 'retired'],
                'single': ['single', 'alone', 'individual', 'solo', 'myself', 'living alone']
            },
            maintenance={
                'low': ['low maintenance', 'easy care', 'simple', 'minimal grooming',
                       'wash and go'],
                'moderate': ['moderate maintenance', 'regular grooming'],
                'high': ['high maintenance', 'professional grooming', 'daily brushing',
                        'care intensive']
            },
            special={
                'guard': ['guard dog', 'protection', 'security', 'watchdog',
                         'protective', 'defender'],
                'companion': ['companion', 'therapy', 'emotional support', 'comfort',
                            'cuddly', 'lap dog'],
                'hypoallergenic': ['hypoallergenic', 'allergies', 'non-shedding',
                                 'allergy-friendly', 'no shed'],
                'first_time': ['first time', 'beginner', 'new to dogs', 'inexperienced',
                              'never owned'],
                'senior': ['senior', 'elderly', 'retired', 'older person', 'old age',
                          'aging', 'older adult', 'golden years', 'retirement']
            }
        )

    def _build_semantic_templates(self):
        """建立語義模板向量（僅在 SBERT 可用時）"""
        # Initialize SBERT model if needed
        if self.sbert_model is None:
            self._initialize_sbert_model()
            
        if not self.sbert_model:
            return

        try:
            # 為每個維度建立模板句子
            templates = {
                'spatial_apartment': "I live in an apartment with limited space and no yard",
                'spatial_house': "I live in a house with a large yard and outdoor space",
                'activity_low': "I prefer a calm, low-energy dog that doesn't need much exercise",
                'activity_high': "I want an active, energetic dog for hiking and outdoor activities",
                'noise_low': "I need a quiet dog that rarely barks and won't disturb neighbors",
                'noise_high': "I don't mind a vocal dog that barks and makes noise",
                'size_small': "I prefer small, compact dogs that are easy to handle",
                'size_large': "I want a large, impressive dog with strong presence",
                'family_children': "I have young children and need a child-friendly dog",
                'family_elderly': "I'm looking for a calm companion dog for elderly person",
                'maintenance_low': "I want a low-maintenance dog that's easy to care for",
                'maintenance_high': "I don't mind high-maintenance dogs requiring professional grooming"
            }

            # 生成模板向量
            for key, template in templates.items():
                embedding = self.sbert_model.encode(template, convert_to_tensor=False)
                self.semantic_templates[key] = embedding

            print(f"Built {len(self.semantic_templates)} semantic templates")

        except Exception as e:
            print(f"Error building semantic templates: {str(e)}")
            self.semantic_templates = {}

    def analyze_query(self, user_input: str) -> QueryDimensions:
        """
        分析使用者查詢並提取多維度意圖

        Args:
            user_input: 使用者的自然語言查詢

        Returns:
            QueryDimensions: 結構化的查詢維度
        """
        try:
            # 正規化輸入文字
            normalized_input = user_input.lower().strip()

            # 基於關鍵字的維度分析
            dimensions = self._extract_keyword_dimensions(normalized_input)

            # 如果 SBERT 可用，進行語義分析增強
            if self.sbert_model is None:
                self._initialize_sbert_model()

            if self.sbert_model:
                semantic_dimensions = self._extract_semantic_dimensions(user_input)
                dimensions = self._merge_dimensions(dimensions, semantic_dimensions)

            # 提取品種提及
            dimensions.breed_mentions = self._extract_breed_mentions(normalized_input)

            # 計算信心分數
            dimensions.confidence_scores = self._calculate_confidence_scores(dimensions, user_input)

            # **關鍵修復：使用 PriorityDetector 檢測維度優先級**
            priority_result = self.priority_detector.detect_priorities(user_input)
            dimensions.dimension_priorities = priority_result.dimension_priorities

            # Debug 輸出
            print(f"=== Query Analysis Debug ===")
            print(f"  experience_level: {dimensions.experience_level}")
            print(f"  maintenance_level: {dimensions.maintenance_level}")
            print(f"  spatial_constraints: {dimensions.spatial_constraints}")
            print(f"  dimension_priorities: {dimensions.dimension_priorities}")
            print(f"============================")

            return dimensions

        except Exception as e:
            print(f"Error analyzing query: {str(e)}")
            print(traceback.format_exc())
            # 回傳空的維度結構
            return QueryDimensions()

    def _extract_keyword_dimensions(self, text: str) -> QueryDimensions:
        """基於關鍵字提取維度"""
        dimensions = QueryDimensions()

        # 空間限制分析
        for category, keywords in self.synonyms.spatial.items():
            if any(keyword in text for keyword in keywords):
                dimensions.spatial_constraints.append(category)

        # 活動水平分析
        for level, keywords in self.synonyms.activity.items():
            if any(keyword in text for keyword in keywords):
                dimensions.activity_level.append(level)

        # 噪音偏好分析
        for level, keywords in self.synonyms.noise.items():
            if any(keyword in text for keyword in keywords):
                dimensions.noise_preferences.append(level)

        # 尺寸偏好分析
        for size, keywords in self.synonyms.size.items():
            if any(keyword in text for keyword in keywords):
                dimensions.size_preferences.append(size)

        # 家庭情況分析
        for context, keywords in self.synonyms.family.items():
            if any(keyword in text for keyword in keywords):
                dimensions.family_context.append(context)

        # 維護水平分析
        for level, keywords in self.synonyms.maintenance.items():
            if any(keyword in text for keyword in keywords):
                dimensions.maintenance_level.append(level)

        # 特殊需求分析
        for requirement, keywords in self.synonyms.special.items():
            if any(keyword in text for keyword in keywords):
                dimensions.special_requirements.append(requirement)
                # 如果檢測到first_time，同時設置experience_level
                if requirement == 'first_time':
                    dimensions.experience_level.append('beginner')

        # 額外的經驗等級檢測
        experience_keywords = {
            'beginner': ['first dog', 'first time', 'beginner', 'new to dogs', 'inexperienced',
                        'never owned', 'never had a dog', 'first-time owner', 'first-time dog owner',
                        'first time owner', 'first time dog owner', 'new dog owner', 'new owner'],
            'intermediate': ['some experience', 'had dogs before', 'owned dogs'],
            'advanced': ['experienced', 'expert', 'professional', 'breeder', 'dog trainer']
        }
        for level, keywords in experience_keywords.items():
            if any(keyword in text for keyword in keywords):
                if level not in dimensions.experience_level:
                    dimensions.experience_level.append(level)

        return dimensions

    def _extract_semantic_dimensions(self, text: str) -> QueryDimensions:
        """基於語義相似度提取維度（需要 SBERT）"""
        if not self.sbert_model or not self.semantic_templates:
            return QueryDimensions()

        try:
            # 生成查詢向量
            query_embedding = self.sbert_model.encode(text, convert_to_tensor=False)

            dimensions = QueryDimensions()

            # 計算與各個模板的相似度
            similarities = {}
            for template_key, template_embedding in self.semantic_templates.items():
                similarity = np.dot(query_embedding, template_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(template_embedding)
                )
                similarities[template_key] = similarity

            # 設定相似度閾值
            threshold = 0.5

            # 根據相似度提取維度
            for template_key, similarity in similarities.items():
                if similarity > threshold:
                    if template_key.startswith('spatial_'):
                        category = template_key.replace('spatial_', '')
                        if category not in dimensions.spatial_constraints:
                            dimensions.spatial_constraints.append(category)
                    elif template_key.startswith('activity_'):
                        level = template_key.replace('activity_', '')
                        if level not in dimensions.activity_level:
                            dimensions.activity_level.append(level)
                    elif template_key.startswith('noise_'):
                        level = template_key.replace('noise_', '')
                        if level not in dimensions.noise_preferences:
                            dimensions.noise_preferences.append(level)
                    elif template_key.startswith('size_'):
                        size = template_key.replace('size_', '')
                        if size not in dimensions.size_preferences:
                            dimensions.size_preferences.append(size)
                    elif template_key.startswith('family_'):
                        context = template_key.replace('family_', '')
                        if context not in dimensions.family_context:
                            dimensions.family_context.append(context)
                    elif template_key.startswith('maintenance_'):
                        level = template_key.replace('maintenance_', '')
                        if level not in dimensions.maintenance_level:
                            dimensions.maintenance_level.append(level)

            return dimensions

        except Exception as e:
            print(f"Error in semantic dimension extraction: {str(e)}")
            return QueryDimensions()

    def _extract_breed_mentions(self, text: str) -> List[str]:
        """提取品種提及"""
        mentioned_breeds = []

        for breed in self.breed_list:
            # 將品種名稱轉換為顯示格式
            breed_display = breed.replace('_', ' ').lower()
            breed_words = breed_display.split()

            # 檢查品種名稱是否在文字中
            breed_found = False

            # 完整品種名稱匹配
            if breed_display in text:
                breed_found = True
            else:
                # 部分匹配（至少匹配品種名稱的主要部分）
                main_word = breed_words[0] if breed_words else ""
                if len(main_word) > 3 and main_word in text:
                    breed_found = True

            if breed_found:
                mentioned_breeds.append(breed)

        return mentioned_breeds

    def _merge_dimensions(self, keyword_dims: QueryDimensions,
                         semantic_dims: QueryDimensions) -> QueryDimensions:
        """合併關鍵字和語義維度"""
        merged = QueryDimensions()

        # 合併各個維度的結果（去重）
        merged.spatial_constraints = list(set(
            keyword_dims.spatial_constraints + semantic_dims.spatial_constraints
        ))
        merged.activity_level = list(set(
            keyword_dims.activity_level + semantic_dims.activity_level
        ))
        merged.noise_preferences = list(set(
            keyword_dims.noise_preferences + semantic_dims.noise_preferences
        ))
        merged.size_preferences = list(set(
            keyword_dims.size_preferences + semantic_dims.size_preferences
        ))
        merged.family_context = list(set(
            keyword_dims.family_context + semantic_dims.family_context
        ))
        merged.maintenance_level = list(set(
            keyword_dims.maintenance_level + semantic_dims.maintenance_level
        ))
        merged.experience_level = list(set(
            keyword_dims.experience_level + semantic_dims.experience_level
        ))
        merged.special_requirements = list(set(
            keyword_dims.special_requirements + semantic_dims.special_requirements
        ))

        return merged

    def _calculate_confidence_scores(self, dimensions: QueryDimensions,
                                   original_text: str) -> Dict[str, float]:
        """計算各維度的信心分數"""
        confidence_scores = {}

        # 基於匹配的關鍵字數量計算信心分數
        text_length = len(original_text.split())

        # 空間限制信心分數
        spatial_matches = len(dimensions.spatial_constraints)
        confidence_scores['spatial'] = min(1.0, spatial_matches * 0.5)

        # 活動水平信心分數
        activity_matches = len(dimensions.activity_level)
        confidence_scores['activity'] = min(1.0, activity_matches * 0.5)

        # 噪音偏好信心分數
        noise_matches = len(dimensions.noise_preferences)
        confidence_scores['noise'] = min(1.0, noise_matches * 0.5)

        # 尺寸偏好信心分數
        size_matches = len(dimensions.size_preferences)
        confidence_scores['size'] = min(1.0, size_matches * 0.5)

        # 家庭情況信心分數
        family_matches = len(dimensions.family_context)
        confidence_scores['family'] = min(1.0, family_matches * 0.5)

        # 維護水平信心分數
        maintenance_matches = len(dimensions.maintenance_level)
        confidence_scores['maintenance'] = min(1.0, maintenance_matches * 0.5)

        # 特殊需求信心分數
        special_matches = len(dimensions.special_requirements)
        confidence_scores['special'] = min(1.0, special_matches * 0.5)

        # 品種提及信心分數
        breed_matches = len(dimensions.breed_mentions)
        confidence_scores['breeds'] = min(1.0, breed_matches * 0.3)

        # 整體信心分數（基於總匹配數量和文字長度）
        total_matches = sum([
            spatial_matches, activity_matches, noise_matches, size_matches,
            family_matches, maintenance_matches, special_matches, breed_matches
        ])
        confidence_scores['overall'] = min(1.0, total_matches / max(1, text_length * 0.1))

        return confidence_scores

    def get_dimension_summary(self, dimensions: QueryDimensions) -> Dict[str, Any]:
        """獲取維度摘要信息"""
        return {
            'spatial_constraints': dimensions.spatial_constraints,
            'activity_level': dimensions.activity_level,
            'noise_preferences': dimensions.noise_preferences,
            'size_preferences': dimensions.size_preferences,
            'family_context': dimensions.family_context,
            'maintenance_level': dimensions.maintenance_level,
            'special_requirements': dimensions.special_requirements,
            'breed_mentions': [breed.replace('_', ' ') for breed in dimensions.breed_mentions],
            'confidence_scores': dimensions.confidence_scores,
            'total_dimensions_detected': sum([
                len(dimensions.spatial_constraints),
                len(dimensions.activity_level),
                len(dimensions.noise_preferences),
                len(dimensions.size_preferences),
                len(dimensions.family_context),
                len(dimensions.maintenance_level),
                len(dimensions.special_requirements)
            ])
        }

# 便利函數
def analyze_user_query(user_input: str) -> QueryDimensions:
    """
    便利函數：分析使用者查詢

    Args:
        user_input: 使用者的自然語言查詢

    Returns:
        QueryDimensions: 結構化的查詢維度
    """
    engine = QueryUnderstandingEngine()
    return engine.analyze_query(user_input)

def get_query_summary(user_input: str) -> Dict[str, Any]:
    """
    便利函數：獲取查詢摘要

    Args:
        user_input: 使用者的自然語言查詢

    Returns:
        Dict: 查詢維度摘要
    """
    engine = QueryUnderstandingEngine()
    dimensions = engine.analyze_query(user_input)
    return engine.get_dimension_summary(dimensions)
