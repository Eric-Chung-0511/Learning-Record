import numpy as np
import json
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import traceback
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from dog_database import get_dog_description
from breed_health_info import breed_health_info
from breed_noise_info import breed_noise_info
from query_understanding import QueryDimensions
from constraint_manager import FilterResult

@dataclass
class DimensionalScores:
    """多維度評分結果"""
    semantic_scores: Dict[str, float] = field(default_factory=dict)
    attribute_scores: Dict[str, float] = field(default_factory=dict)
    fused_scores: Dict[str, float] = field(default_factory=dict)
    bidirectional_scores: Dict[str, float] = field(default_factory=dict)
    confidence_weights: Dict[str, float] = field(default_factory=dict)

@dataclass
class BreedScore:
    """品種總體評分結果"""
    breed_name: str
    final_score: float
    dimensional_breakdown: Dict[str, float] = field(default_factory=dict)
    semantic_component: float = 0.0
    attribute_component: float = 0.0
    bidirectional_bonus: float = 0.0
    confidence_score: float = 1.0
    explanation: Dict[str, Any] = field(default_factory=dict)

class ScoringHead(ABC):
    """抽象評分頭基類"""

    @abstractmethod
    def score_dimension(self, breed_info: Dict[str, Any],
                       dimensions: QueryDimensions,
                       dimension_type: str) -> float:
        """為特定維度評分"""
        pass

class SemanticScoringHead(ScoringHead):
    """語義評分頭"""

    def __init__(self, sbert_model: Optional[SentenceTransformer] = None):
        self.sbert_model = sbert_model
        self.dimension_embeddings = {}
        if self.sbert_model:
            self._build_dimension_embeddings()

    def _build_dimension_embeddings(self):
        """建立維度模板嵌入"""
        dimension_templates = {
            'spatial_apartment': "small apartment living, limited space, no yard, urban environment",
            'spatial_house': "house with yard, outdoor space, suburban living, large property",
            'activity_low': "low energy, minimal exercise needs, calm lifestyle, indoor activities",
            'activity_moderate': "moderate exercise, daily walks, balanced activity level",
            'activity_high': "high energy, vigorous exercise, outdoor sports, active lifestyle",
            'noise_low': "quiet, rarely barks, peaceful, suitable for noise-sensitive environments",
            'noise_moderate': "moderate barking, occasional vocalizations, average noise level",
            'noise_high': "vocal, frequent barking, alert dog, comfortable with noise",
            'size_small': "small compact breed, easy to handle, portable size",
            'size_medium': "medium sized dog, balanced proportions, moderate size",
            'size_large': "large impressive dog, substantial presence, bigger breed",
            'family_children': "child-friendly, gentle with kids, family-oriented, safe around children",
            'family_elderly': "calm companion, gentle nature, suitable for seniors, low maintenance",
            'maintenance_low': "low grooming needs, minimal care requirements, easy maintenance",
            'maintenance_moderate': "regular grooming, moderate care needs, standard maintenance",
            'maintenance_high': "high grooming requirements, professional care, intensive maintenance"
        }

        for key, template in dimension_templates.items():
            if self.sbert_model:
                embedding = self.sbert_model.encode(template, convert_to_tensor=False)
                self.dimension_embeddings[key] = embedding

    def score_dimension(self, breed_info: Dict[str, Any],
                       dimensions: QueryDimensions,
                       dimension_type: str) -> float:
        """語義維度評分"""
        if not self.sbert_model or dimension_type not in self.dimension_embeddings:
            return 0.5  # 預設中性分數

        try:
            # 建立品種描述
            breed_description = self._create_breed_description(breed_info, dimension_type)

            # 生成嵌入
            breed_embedding = self.sbert_model.encode(breed_description, convert_to_tensor=False)
            dimension_embedding = self.dimension_embeddings[dimension_type]

            # 計算相似度
            similarity = cosine_similarity([breed_embedding], [dimension_embedding])[0][0]

            # 正規化到 0-1 範圍
            normalized_score = (similarity + 1) / 2  # 從 [-1,1] 轉換到 [0,1]

            return max(0.0, min(1.0, normalized_score))

        except Exception as e:
            print(f"Error in semantic scoring for {dimension_type}: {str(e)}")
            return 0.5

    def _create_breed_description(self, breed_info: Dict[str, Any],
                                dimension_type: str) -> str:
        """為特定維度創建品種描述"""
        breed_name = breed_info.get('display_name', breed_info.get('breed_name', ''))

        if dimension_type.startswith('spatial_'):
            size = breed_info.get('size', 'medium')
            exercise = breed_info.get('exercise_needs', 'moderate')
            return f"{breed_name} is a {size} dog with {exercise} exercise needs"

        elif dimension_type.startswith('activity_'):
            exercise = breed_info.get('exercise_needs', 'moderate')
            temperament = breed_info.get('temperament', '')
            return f"{breed_name} has {exercise} exercise requirements and {temperament} temperament"

        elif dimension_type.startswith('noise_'):
            noise_level = breed_info.get('noise_level', 'moderate')
            temperament = breed_info.get('temperament', '')
            return f"{breed_name} has {noise_level} noise level and {temperament} nature"

        elif dimension_type.startswith('size_'):
            size = breed_info.get('size', 'medium')
            return f"{breed_name} is a {size} sized dog breed"

        elif dimension_type.startswith('family_'):
            children = breed_info.get('good_with_children', 'Yes')
            temperament = breed_info.get('temperament', '')
            return f"{breed_name} is {children} with children and has {temperament} temperament"

        elif dimension_type.startswith('maintenance_'):
            grooming = breed_info.get('grooming_needs', 'moderate')
            care_level = breed_info.get('care_level', 'moderate')
            return f"{breed_name} requires {grooming} grooming and {care_level} care level"

        return f"{breed_name} is a dog breed with various characteristics"

class AttributeScoringHead(ScoringHead):
    """屬性評分頭"""

    def __init__(self):
        self.scoring_matrices = self._initialize_scoring_matrices()

    def _initialize_scoring_matrices(self) -> Dict[str, Dict[str, float]]:
        """初始化評分矩陣"""
        return {
            'spatial_scoring': {
                # (user_preference, breed_attribute) -> score
                ('apartment', 'small'): 1.0,
                ('apartment', 'medium'): 0.6,
                ('apartment', 'large'): 0.2,
                ('apartment', 'giant'): 0.0,
                ('house', 'small'): 0.7,
                ('house', 'medium'): 0.9,
                ('house', 'large'): 1.0,
                ('house', 'giant'): 1.0,
            },
            'activity_scoring': {
                ('low', 'low'): 1.0,
                ('low', 'moderate'): 0.7,
                ('low', 'high'): 0.2,
                ('low', 'very high'): 0.0,
                ('moderate', 'low'): 0.8,
                ('moderate', 'moderate'): 1.0,
                ('moderate', 'high'): 0.8,
                ('high', 'moderate'): 0.7,
                ('high', 'high'): 1.0,
                ('high', 'very high'): 1.0,
            },
            'noise_scoring': {
                ('low', 'low'): 1.0,
                ('low', 'moderate'): 0.6,
                ('low', 'high'): 0.1,
                ('moderate', 'low'): 0.8,
                ('moderate', 'moderate'): 1.0,
                ('moderate', 'high'): 0.7,
                ('high', 'low'): 0.7,
                ('high', 'moderate'): 0.9,
                ('high', 'high'): 1.0,
            },
            'size_scoring': {
                ('small', 'small'): 1.0,
                ('small', 'medium'): 0.5,
                ('small', 'large'): 0.2,
                ('medium', 'small'): 0.6,
                ('medium', 'medium'): 1.0,
                ('medium', 'large'): 0.6,
                ('large', 'medium'): 0.7,
                ('large', 'large'): 1.0,
                ('large', 'giant'): 0.9,
            },
            'maintenance_scoring': {
                ('low', 'low'): 1.0,
                ('low', 'moderate'): 0.6,
                ('low', 'high'): 0.2,
                ('moderate', 'low'): 0.8,
                ('moderate', 'moderate'): 1.0,
                ('moderate', 'high'): 0.7,
                ('high', 'low'): 0.6,
                ('high', 'moderate'): 0.8,
                ('high', 'high'): 1.0,
            }
        }

    def score_dimension(self, breed_info: Dict[str, Any],
                       dimensions: QueryDimensions,
                       dimension_type: str) -> float:
        """屬性維度評分"""
        try:
            if dimension_type.startswith('spatial_'):
                return self._score_spatial_compatibility(breed_info, dimensions)
            elif dimension_type.startswith('activity_'):
                return self._score_activity_compatibility(breed_info, dimensions)
            elif dimension_type.startswith('noise_'):
                return self._score_noise_compatibility(breed_info, dimensions)
            elif dimension_type.startswith('size_'):
                return self._score_size_compatibility(breed_info, dimensions)
            elif dimension_type.startswith('family_'):
                return self._score_family_compatibility(breed_info, dimensions)
            elif dimension_type.startswith('maintenance_'):
                return self._score_maintenance_compatibility(breed_info, dimensions)
            else:
                return 0.5  # 預設中性分數

        except Exception as e:
            print(f"Error in attribute scoring for {dimension_type}: {str(e)}")
            return 0.5

    def _score_spatial_compatibility(self, breed_info: Dict[str, Any],
                                   dimensions: QueryDimensions) -> float:
        """空間相容性評分"""
        if not dimensions.spatial_constraints:
            return 0.5

        breed_size = breed_info.get('size', 'medium').lower()
        total_score = 0.0

        for spatial_constraint in dimensions.spatial_constraints:
            key = (spatial_constraint, breed_size)
            score = self.scoring_matrices['spatial_scoring'].get(key, 0.5)
            total_score += score

        return total_score / len(dimensions.spatial_constraints)

    def _score_activity_compatibility(self, breed_info: Dict[str, Any],
                                    dimensions: QueryDimensions) -> float:
        """活動相容性評分"""
        if not dimensions.activity_level:
            return 0.5

        breed_exercise = breed_info.get('exercise_needs', 'moderate').lower()
        # 清理品種運動需求字串
        if 'very high' in breed_exercise:
            breed_exercise = 'very high'
        elif 'high' in breed_exercise:
            breed_exercise = 'high'
        elif 'low' in breed_exercise:
            breed_exercise = 'low'
        else:
            breed_exercise = 'moderate'

        total_score = 0.0
        for activity_level in dimensions.activity_level:
            key = (activity_level, breed_exercise)
            score = self.scoring_matrices['activity_scoring'].get(key, 0.5)
            total_score += score

        return total_score / len(dimensions.activity_level)

    def _score_noise_compatibility(self, breed_info: Dict[str, Any],
                                 dimensions: QueryDimensions) -> float:
        """噪音相容性評分"""
        if not dimensions.noise_preferences:
            return 0.5

        breed_noise = breed_info.get('noise_level', 'moderate').lower()
        total_score = 0.0

        for noise_pref in dimensions.noise_preferences:
            key = (noise_pref, breed_noise)
            score = self.scoring_matrices['noise_scoring'].get(key, 0.5)
            total_score += score

        return total_score / len(dimensions.noise_preferences)

    def _score_size_compatibility(self, breed_info: Dict[str, Any],
                                dimensions: QueryDimensions) -> float:
        """尺寸相容性評分"""
        if not dimensions.size_preferences:
            return 0.5

        breed_size = breed_info.get('size', 'medium').lower()
        total_score = 0.0

        for size_pref in dimensions.size_preferences:
            key = (size_pref, breed_size)
            score = self.scoring_matrices['size_scoring'].get(key, 0.5)
            total_score += score

        return total_score / len(dimensions.size_preferences)

    def _score_family_compatibility(self, breed_info: Dict[str, Any],
                                  dimensions: QueryDimensions) -> float:
        """家庭相容性評分"""
        if not dimensions.family_context:
            return 0.5

        good_with_children = breed_info.get('good_with_children', 'Yes')
        temperament = breed_info.get('temperament', '').lower()

        total_score = 0.0
        score_count = 0

        for family_context in dimensions.family_context:
            if family_context == 'children':
                if good_with_children == 'Yes':
                    total_score += 1.0
                elif good_with_children == 'No':
                    total_score += 0.1
                else:
                    total_score += 0.6
                score_count += 1
            elif family_context == 'elderly':
                # 溫和、冷靜的品種適合老年人
                if any(trait in temperament for trait in ['gentle', 'calm', 'docile']):
                    total_score += 1.0
                elif any(trait in temperament for trait in ['energetic', 'hyperactive']):
                    total_score += 0.3
                else:
                    total_score += 0.7
                score_count += 1
            elif family_context == 'single':
                # 大多數品種都適合單身人士
                total_score += 0.8
                score_count += 1

        return total_score / max(1, score_count)

    def _score_maintenance_compatibility(self, breed_info: Dict[str, Any],
                                       dimensions: QueryDimensions) -> float:
        """維護相容性評分"""
        if not dimensions.maintenance_level:
            return 0.5

        breed_grooming = breed_info.get('grooming_needs', 'moderate').lower()
        total_score = 0.0

        for maintenance_level in dimensions.maintenance_level:
            key = (maintenance_level, breed_grooming)
            score = self.scoring_matrices['maintenance_scoring'].get(key, 0.5)
            total_score += score

        return total_score / len(dimensions.maintenance_level)

class MultiHeadScorer:
    """
    多頭評分系統
    結合語義和屬性評分，提供雙向相容性評估
    """

    def __init__(self, sbert_model: Optional[SentenceTransformer] = None):
        self.sbert_model = sbert_model
        self.semantic_head = SemanticScoringHead(sbert_model)
        self.attribute_head = AttributeScoringHead()
        self.dimension_weights = self._initialize_dimension_weights()
        self.head_fusion_weights = self._initialize_head_fusion_weights()

    def _initialize_dimension_weights(self) -> Dict[str, float]:
        """初始化維度權重"""
        return {
            'activity_compatibility': 0.35,    # 最高優先級：生活方式匹配
            'noise_compatibility': 0.25,       # 關鍵：居住和諧
            'spatial_compatibility': 0.15,     # 基本：物理約束
            'family_compatibility': 0.10,      # 重要：社交相容性
            'maintenance_compatibility': 0.10,  # 實際：持續護理評估
            'size_compatibility': 0.05         # 基本：偏好匹配
        }

    def _initialize_head_fusion_weights(self) -> Dict[str, Dict[str, float]]:
        """初始化頭融合權重"""
        return {
            'activity_compatibility': {'semantic': 0.4, 'attribute': 0.6},
            'noise_compatibility': {'semantic': 0.3, 'attribute': 0.7},
            'spatial_compatibility': {'semantic': 0.3, 'attribute': 0.7},
            'family_compatibility': {'semantic': 0.5, 'attribute': 0.5},
            'maintenance_compatibility': {'semantic': 0.4, 'attribute': 0.6},
            'size_compatibility': {'semantic': 0.2, 'attribute': 0.8}
        }

    def score_breeds(self, candidate_breeds: Set[str],
                    dimensions: QueryDimensions) -> List[BreedScore]:
        """
        為候選品種評分

        Args:
            candidate_breeds: 通過約束篩選的候選品種
            dimensions: 查詢維度

        Returns:
            List[BreedScore]: 品種評分結果列表
        """
        try:
            breed_scores = []

            # 為每個品種計算分數
            for breed in candidate_breeds:
                breed_info = self._get_breed_info(breed)
                score_result = self._score_single_breed(breed_info, dimensions)
                breed_scores.append(score_result)

            # 按最終分數排序
            breed_scores.sort(key=lambda x: x.final_score, reverse=True)

            return breed_scores

        except Exception as e:
            print(f"Error scoring breeds: {str(e)}")
            print(traceback.format_exc())
            return []

    def _get_breed_info(self, breed: str) -> Dict[str, Any]:
        """獲取品種資訊"""
        try:
            # 基本品種資訊
            breed_info = get_dog_description(breed) or {}

            # 健康資訊
            health_info = breed_health_info.get(breed, {})

            # 噪音資訊
            noise_info = breed_noise_info.get(breed, {})

            # 整合資訊
            return {
                'breed_name': breed,
                'display_name': breed.replace('_', ' '),
                'size': breed_info.get('Size', '').lower(),
                'exercise_needs': breed_info.get('Exercise Needs', '').lower(),
                'grooming_needs': breed_info.get('Grooming Needs', '').lower(),
                'temperament': breed_info.get('Temperament', '').lower(),
                'good_with_children': breed_info.get('Good with Children', 'Yes'),
                'care_level': breed_info.get('Care Level', '').lower(),
                'lifespan': breed_info.get('Lifespan', '10-12 years'),
                'noise_level': noise_info.get('noise_level', 'moderate').lower(),
                'description': breed_info.get('Description', ''),
                'raw_breed_info': breed_info,
                'raw_health_info': health_info,
                'raw_noise_info': noise_info
            }
        except Exception as e:
            print(f"Error getting breed info for {breed}: {str(e)}")
            return {
                'breed_name': breed,
                'display_name': breed.replace('_', ' ')
            }

    def _score_single_breed(self, breed_info: Dict[str, Any],
                          dimensions: QueryDimensions) -> BreedScore:
        """為單一品種評分"""
        try:
            dimensional_scores = {}
            semantic_total = 0.0
            attribute_total = 0.0

            # 動態權重分配（基於用戶表達的維度）
            active_dimensions = self._get_active_dimensions(dimensions)
            adjusted_weights = self._adjust_dimension_weights(active_dimensions)

            # 為每個活躍維度評分
            for dimension, weight in adjusted_weights.items():
                # 語義評分
                semantic_score = self.semantic_head.score_dimension(
                    breed_info, dimensions, dimension
                )

                # 屬性評分
                attribute_score = self.attribute_head.score_dimension(
                    breed_info, dimensions, dimension
                )

                # 頭融合
                fusion_weights = self.head_fusion_weights.get(
                    dimension, {'semantic': 0.5, 'attribute': 0.5}
                )

                fused_score = (semantic_score * fusion_weights['semantic'] +
                              attribute_score * fusion_weights['attribute'])

                dimensional_scores[dimension] = fused_score
                semantic_total += semantic_score * weight
                attribute_total += attribute_score * weight

            # 雙向相容性評估
            bidirectional_bonus = self._calculate_bidirectional_bonus(
                breed_info, dimensions
            )

            # Apply size bias correction
            bias_correction = self._calculate_size_bias_correction(breed_info, dimensions)

            # 計算最終分數
            base_score = sum(score * adjusted_weights[dim]
                           for dim, score in dimensional_scores.items())

            # Apply corrections
            final_score = max(0.0, min(1.0, base_score + bidirectional_bonus + bias_correction))

            # 信心度評估
            confidence_score = self._calculate_confidence(dimensions)

            return BreedScore(
                breed_name=breed_info.get('display_name', breed_info['breed_name']),
                final_score=final_score,
                dimensional_breakdown=dimensional_scores,
                semantic_component=semantic_total,
                attribute_component=attribute_total,
                bidirectional_bonus=bidirectional_bonus,
                confidence_score=confidence_score,
                explanation=self._generate_explanation(breed_info, dimensions, dimensional_scores)
            )

        except Exception as e:
            print(f"Error scoring breed {breed_info.get('breed_name', 'unknown')}: {str(e)}")
            return BreedScore(
                breed_name=breed_info.get('display_name', breed_info.get('breed_name', 'Unknown')),
                final_score=0.5,
                confidence_score=0.0
            )

    def _get_active_dimensions(self, dimensions: QueryDimensions) -> Set[str]:
        """獲取活躍的維度"""
        active = set()

        if dimensions.spatial_constraints:
            active.add('spatial_compatibility')
        if dimensions.activity_level:
            active.add('activity_compatibility')
        if dimensions.noise_preferences:
            active.add('noise_compatibility')
        if dimensions.size_preferences:
            active.add('size_compatibility')
        if dimensions.family_context:
            active.add('family_compatibility')
        if dimensions.maintenance_level:
            active.add('maintenance_compatibility')

        return active

    def _adjust_dimension_weights(self, active_dimensions: Set[str]) -> Dict[str, float]:
        """調整維度權重"""
        if not active_dimensions:
            return self.dimension_weights

        # 只為活躍維度分配權重
        active_weights = {dim: weight for dim, weight in self.dimension_weights.items()
                         if dim in active_dimensions}

        # 正規化權重總和為 1.0
        total_weight = sum(active_weights.values())
        if total_weight > 0:
            active_weights = {dim: weight / total_weight
                            for dim, weight in active_weights.items()}

        return active_weights

    def _calculate_bidirectional_bonus(self, breed_info: Dict[str, Any],
                                     dimensions: QueryDimensions) -> float:
        """計算雙向相容性獎勵"""
        try:
            bonus = 0.0

            # 正向相容性：品種滿足用戶需求
            forward_compatibility = self._assess_forward_compatibility(breed_info, dimensions)

            # 反向相容性：用戶生活方式適合品種需求
            reverse_compatibility = self._assess_reverse_compatibility(breed_info, dimensions)

            # 雙向獎勵（較為保守）
            bonus = min(0.1, (forward_compatibility + reverse_compatibility) * 0.05)

            return bonus

        except Exception as e:
            print(f"Error calculating bidirectional bonus: {str(e)}")
            return 0.0

    def _assess_forward_compatibility(self, breed_info: Dict[str, Any],
                                    dimensions: QueryDimensions) -> float:
        """評估正向相容性"""
        compatibility = 0.0

        # 空間需求匹配
        if 'apartment' in dimensions.spatial_constraints:
            size = breed_info.get('size', '')
            if 'small' in size:
                compatibility += 0.3
            elif 'medium' in size:
                compatibility += 0.1

        # 活動需求匹配
        if 'low' in dimensions.activity_level:
            exercise = breed_info.get('exercise_needs', '')
            if 'low' in exercise:
                compatibility += 0.3
            elif 'moderate' in exercise:
                compatibility += 0.1

        return compatibility

    def _assess_reverse_compatibility(self, breed_info: Dict[str, Any],
                                    dimensions: QueryDimensions) -> float:
        """評估反向相容性"""
        compatibility = 0.0

        # 品種是否能在用戶環境中茁壯成長
        exercise_needs = breed_info.get('exercise_needs', '')

        if 'high' in exercise_needs:
            # 高運動需求品種需要確認用戶能提供足夠運動
            if ('high' in dimensions.activity_level or
                'house' in dimensions.spatial_constraints):
                compatibility += 0.2
            else:
                compatibility -= 0.2

        # 品種護理需求是否與用戶能力匹配
        grooming_needs = breed_info.get('grooming_needs', '')
        if 'high' in grooming_needs:
            if 'high' in dimensions.maintenance_level:
                compatibility += 0.1
            elif 'low' in dimensions.maintenance_level:
                compatibility -= 0.1

        return compatibility

    def _calculate_size_bias_correction(self, breed_info: Dict,
                                       dimensions: QueryDimensions) -> float:
        """Correct systematic bias toward larger breeds"""
        breed_size = breed_info.get('size', '').lower()

        # Default no bias correction
        correction = 0.0

        # Detect if user specified moderate/balanced preferences
        if any(term in dimensions.activity_level for term in ['moderate', 'balanced', 'average']):
            # Penalize extremes
            if breed_size in ['giant', 'toy']:
                correction = -0.1
            elif breed_size in ['large']:
                correction = -0.05

        # Boost medium breeds for moderate requirements
        if 'medium' in breed_size and 'balanced' in str(dimensions.activity_level):
            correction = 0.1

        return correction

    def _calculate_confidence(self, dimensions: QueryDimensions) -> float:
        """計算推薦信心度"""
        # 基於維度覆蓋率和信心分數計算
        dimension_count = sum([
            len(dimensions.spatial_constraints),
            len(dimensions.activity_level),
            len(dimensions.noise_preferences),
            len(dimensions.size_preferences),
            len(dimensions.family_context),
            len(dimensions.maintenance_level),
            len(dimensions.special_requirements)
        ])

        # 基礎信心度
        base_confidence = min(1.0, dimension_count * 0.15)

        # 品種提及獎勵
        breed_bonus = min(0.2, len(dimensions.breed_mentions) * 0.1)

        # 整體信心分數
        overall_confidence = dimensions.confidence_scores.get('overall', 0.5)

        return min(1.0, base_confidence + breed_bonus + overall_confidence * 0.3)

    def _generate_explanation(self, breed_info: Dict[str, Any],
                            dimensions: QueryDimensions,
                            dimensional_scores: Dict[str, float]) -> Dict[str, Any]:
        """生成評分解釋"""
        try:
            explanation = {
                'strengths': [],
                'considerations': [],
                'match_highlights': [],
                'score_breakdown': dimensional_scores
            }

            breed_name = breed_info.get('display_name', '')

            # 分析各維度表現
            for dimension, score in dimensional_scores.items():
                if score >= 0.8:
                    explanation['strengths'].append(self._get_strength_text(dimension, breed_info))
                elif score <= 0.3:
                    explanation['considerations'].append(self._get_consideration_text(dimension, breed_info))
                else:
                    explanation['match_highlights'].append(f"{dimension}: {score:.2f}")

            return explanation

        except Exception as e:
            print(f"Error generating explanation: {str(e)}")
            return {'strengths': [], 'considerations': [], 'match_highlights': []}

    def _get_strength_text(self, dimension: str, breed_info: Dict[str, Any]) -> str:
        """Get strength description"""
        breed_name = breed_info.get('display_name', '')

        if dimension == 'activity_compatibility':
            return f"{breed_name} has an activity level that matches your lifestyle very well"
        elif dimension == 'noise_compatibility':
            return f"{breed_name} has noise characteristics that fit your environment"
        elif dimension == 'spatial_compatibility':
            return f"{breed_name} is very suitable for your living space"
        elif dimension == 'family_compatibility':
            return f"{breed_name} performs well in a family environment"
        elif dimension == 'maintenance_compatibility':
            return f"{breed_name} has grooming needs that match your willingness to commit"
        else:
            return f"{breed_name} shows strong performance in {dimension}"

    def _get_consideration_text(self, dimension: str, breed_info: Dict[str, Any]) -> str:
        """Get consideration description"""
        breed_name = breed_info.get('display_name', '')

        if dimension == 'activity_compatibility':
            return f"{breed_name} may have exercise needs that differ from your lifestyle"
        elif dimension == 'noise_compatibility':
            return f"{breed_name} has noise characteristics that require special consideration"
        elif dimension == 'maintenance_compatibility':
            return f"{breed_name} has relatively high grooming requirements"
        else:
            return f"{breed_name} requires extra consideration in {dimension}"


def score_breed_candidates(candidate_breeds: Set[str],
                         dimensions: QueryDimensions,
                         sbert_model: Optional[SentenceTransformer] = None) -> List[BreedScore]:
    """
    便利函數：為候選品種評分

    Args:
        candidate_breeds: 候選品種集合
        dimensions: 查詢維度
        sbert_model: 可選的SBERT模型

    Returns:
        List[BreedScore]: 評分結果列表
    """
    scorer = MultiHeadScorer(sbert_model)
    return scorer.score_breeds(candidate_breeds, dimensions)
