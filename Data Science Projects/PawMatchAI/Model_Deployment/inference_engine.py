# %%writefile inference_engine.py
import re
import numpy as np
from typing import Dict, List, Tuple, Set, Optional, Any, Callable
from dataclasses import dataclass, field
import traceback


@dataclass
class InferenceRule:
    """推理規則結構"""
    name: str
    condition: Callable[[str, Dict[str, Any]], bool]
    imply: Callable[[str, Dict[str, Any]], Dict[str, float]]
    reasoning: str
    priority: int = 1  # 規則優先級


@dataclass
class InferenceResult:
    """推理結果結構"""
    implicit_priorities: Dict[str, float] = field(default_factory=dict)
    triggered_rules: List[str] = field(default_factory=list)
    reasoning_chains: List[str] = field(default_factory=list)
    confidence: float = 1.0


class BreedRecommendationInferenceEngine:
    """
    品種推薦推理引擎
    從使用者明確輸入中推斷隱含需求，補充優先級設定

    核心邏輯:
    1. 居住環境推理 (公寓 → 安靜、中小型)
    2. 家庭情況推理 (有小孩 → 溫和、耐心)
    3. 經驗程度推理 (新手 → 易照顧)
    4. 生活方式推理 (忙碌 → 低維護)
    """

    def __init__(self):
        """初始化推理引擎"""
        self.inference_rules = self._build_inference_rules()
        self.spatial_keywords = self._initialize_spatial_keywords()
        self.lifestyle_keywords = self._initialize_lifestyle_keywords()

    def _initialize_spatial_keywords(self) -> Dict[str, List[str]]:
        """初始化空間相關關鍵字"""
        return {
            'apartment': [
                'apartment', 'flat', 'condo', 'studio',
                'small space', 'limited space', 'city living', 'urban'
            ],
            'small_house': [
                'small house', 'townhouse', 'small home'
            ],
            'large_house': [
                'large house', 'big house', 'spacious home',
                'yard', 'garden', 'backyard', 'outdoor space'
            ]
        }

    def _initialize_lifestyle_keywords(self) -> Dict[str, List[str]]:
        """初始化生活方式關鍵字"""
        return {
            'has_children': [
                'kids', 'children', 'toddler', 'baby', 'school age',
                'child', 'son', 'daughter', 'family with kids'
            ],
            'beginner': [
                'first dog', 'first time', 'beginner', 'never had',
                'new to dogs', 'inexperienced', 'no experience'
            ],
            'busy': [
                'busy', 'limited time', 'work full time', 'not much time',
                'long hours', 'busy schedule', 'hectic lifestyle'
            ],
            'active': [
                'active', 'athletic', 'outdoor', 'hiking', 'running',
                'jogging', 'sports', 'exercise enthusiast'
            ]
        }

    def _build_inference_rules(self) -> List[InferenceRule]:
        """構建推理規則庫"""
        return [
            # 規則1: 公寓居住推理
            InferenceRule(
                name="apartment_living",
                condition=lambda input_text, ctx: self._check_apartment(input_text, ctx),
                imply=lambda input_text, ctx: {
                    'noise': 1.3,      # 公寓暗示需要安靜 (reduced from 1.4)
                    'size': 1.2,       # 公寓暗示偏好中小型 (reduced from 1.3)
                    'exercise': 1.15   # 公寓暗示可能運動空間有限 (reduced from 1.2)
                },
                reasoning="Apartment living typically requires quieter, smaller dogs with moderate exercise needs",
                priority=1
            ),

            # 規則2: 有小孩推理
            InferenceRule(
                name="has_children",
                condition=lambda input_text, ctx: self._check_children(input_text, ctx),
                imply=lambda input_text, ctx: {
                    'family': 1.4,     # 明確需要家庭友善 (reduced from 1.5)
                    'experience': 1.15, # 暗示希望容易訓練 (reduced from 1.2)
                    'noise': 1.15      # 有小孩通常希望狗較安靜 (reduced from 1.2)
                },
                reasoning="Families with children need gentle, patient, child-safe breeds",
                priority=1
            ),

            # 規則3: 新手飼主推理
            InferenceRule(
                name="beginner_owner",
                condition=lambda input_text, ctx: self._check_beginner(input_text, ctx),
                imply=lambda input_text, ctx: {
                    'experience': 1.3,  # 需要容易照顧 (reduced from 1.4)
                    'grooming': 1.25,   # 偏好低維護 (reduced from 1.3)
                    'health': 1.15      # 希望健康問題少 (reduced from 1.2)
                },
                reasoning="Beginners benefit from easier-to-care-for, low-maintenance breeds",
                priority=1
            ),

            # 規則4: 忙碌生活方式推理
            InferenceRule(
                name="busy_lifestyle",
                condition=lambda input_text, ctx: self._check_busy(input_text, ctx),
                imply=lambda input_text, ctx: {
                    'grooming': 1.3,    # 需要低維護 (reduced from 1.4)
                    'exercise': 1.25,   # 不能需要太多運動 (reduced from 1.3)
                    'experience': 1.15  # 希望獨立性強 (reduced from 1.2)
                },
                reasoning="Busy owners need lower-maintenance breeds with moderate exercise needs",
                priority=1
            ),

            # 規則5: 大型住宅推理
            InferenceRule(
                name="large_house",
                condition=lambda input_text, ctx: self._check_large_house(input_text, ctx),
                imply=lambda input_text, ctx: {
                    'size': 1.15,      # 可以接受大型犬 (reduced from 1.2)
                    'exercise': 1.2    # 可能有更多運動空間 (reduced from 1.3)
                },
                reasoning="Large homes can accommodate more active, larger breeds",
                priority=2
            ),

            # 規則6: 有院子推理
            InferenceRule(
                name="has_yard",
                condition=lambda input_text, ctx: self._check_yard(input_text, ctx),
                imply=lambda input_text, ctx: {
                    'exercise': 1.2,   # 有院子可以支持更活躍的品種 (reduced from 1.3)
                    'size': 1.15       # 可以考慮較大的品種 (reduced from 1.2)
                },
                reasoning="Yards provide exercise space for more active breeds",
                priority=2
            ),

            # 規則7: 噪音敏感環境推理
            InferenceRule(
                name="noise_sensitive",
                condition=lambda input_text, ctx: self._check_noise_sensitive(input_text, ctx),
                imply=lambda input_text, ctx: {
                    'noise': 1.5       # 強調需要安靜 (reduced from 1.6)
                },
                reasoning="Noise-sensitive environments require quieter breeds",
                priority=1
            ),

            # 規則8: 過敏體質推理
            InferenceRule(
                name="allergy_concerns",
                condition=lambda input_text, ctx: self._check_allergies(input_text, ctx),
                imply=lambda input_text, ctx: {
                    'grooming': 1.4,   # 需要低掉毛品種 (reduced from 1.5)
                    'health': 1.25     # 關注健康問題 (reduced from 1.3)
                },
                reasoning="Allergy concerns require hypoallergenic, low-shedding breeds",
                priority=1
            ),

            # 規則9: 活躍生活方式推理
            InferenceRule(
                name="active_lifestyle",
                condition=lambda input_text, ctx: self._check_active(input_text, ctx),
                imply=lambda input_text, ctx: {
                    'exercise': 1.3,   # 需要高運動量品種 (reduced from 1.4)
                    'size': 1.15       # 可能偏好中大型犬 (reduced from 1.2)
                },
                reasoning="Active lifestyle matches well with energetic, athletic breeds",
                priority=1
            ),

            # 規則10: 小型空間推理
            InferenceRule(
                name="small_space",
                condition=lambda input_text, ctx: self._check_small_space(input_text, ctx),
                imply=lambda input_text, ctx: {
                    'size': 1.3,       # 強調需要小型犬 (reduced from 1.4)
                    'noise': 1.25,     # 小空間需要安靜 (reduced from 1.3)
                    'exercise': 1.15   # 運動需求不宜過高 (reduced from 1.2)
                },
                reasoning="Small spaces require compact, quiet dogs with moderate energy",
                priority=1
            )
        ]

    def _check_apartment(self, input_text: str, ctx: Dict[str, Any]) -> bool:
        """檢查是否提到公寓"""
        text_lower = input_text.lower()
        return (any(keyword in text_lower for keyword in self.spatial_keywords['apartment']) or
                ctx.get('living_space') == 'apartment')

    def _check_children(self, input_text: str, ctx: Dict[str, Any]) -> bool:
        """檢查是否有小孩"""
        text_lower = input_text.lower()
        return (any(keyword in text_lower for keyword in self.lifestyle_keywords['has_children']) or
                ctx.get('has_children') == True)

    def _check_beginner(self, input_text: str, ctx: Dict[str, Any]) -> bool:
        """檢查是否為新手"""
        text_lower = input_text.lower()
        return (any(keyword in text_lower for keyword in self.lifestyle_keywords['beginner']) or
                ctx.get('experience_level') == 'beginner')

    def _check_busy(self, input_text: str, ctx: Dict[str, Any]) -> bool:
        """檢查是否為忙碌生活方式"""
        text_lower = input_text.lower()
        return (any(keyword in text_lower for keyword in self.lifestyle_keywords['busy']) or
                ctx.get('time_availability') == 'limited')

    def _check_large_house(self, input_text: str, ctx: Dict[str, Any]) -> bool:
        """檢查是否有大房子"""
        text_lower = input_text.lower()
        return (any(keyword in text_lower for keyword in self.spatial_keywords['large_house']) or
                ctx.get('living_space') in ['house_large', 'house'])

    def _check_yard(self, input_text: str, ctx: Dict[str, Any]) -> bool:
        """檢查是否有院子"""
        text_lower = input_text.lower()
        return (any(keyword in text_lower for keyword in ['yard', 'garden', 'backyard', 'outdoor space']) or
                ctx.get('yard_access') in ['shared_yard', 'private_yard'])

    def _check_noise_sensitive(self, input_text: str, ctx: Dict[str, Any]) -> bool:
        """檢查是否為噪音敏感環境"""
        text_lower = input_text.lower()
        noise_keywords = ['noise sensitive', 'thin walls', 'neighbors close', 'townhouse', 'condo']
        return any(keyword in text_lower for keyword in noise_keywords)

    def _check_allergies(self, input_text: str, ctx: Dict[str, Any]) -> bool:
        """檢查是否有過敏體質"""
        text_lower = input_text.lower()
        allergy_keywords = ['allergies', 'hypoallergenic', 'sensitive to fur', 'asthma', 'allergy']
        return (any(keyword in text_lower for keyword in allergy_keywords) or
                ctx.get('has_allergies') == True)

    def _check_active(self, input_text: str, ctx: Dict[str, Any]) -> bool:
        """檢查是否為活躍生活方式"""
        text_lower = input_text.lower()
        return (any(keyword in text_lower for keyword in self.lifestyle_keywords['active']) or
                ctx.get('activity_level') == 'high')

    def _check_small_space(self, input_text: str, ctx: Dict[str, Any]) -> bool:
        """檢查是否為小型空間"""
        text_lower = input_text.lower()
        small_space_keywords = ['small space', 'limited space', 'tiny', 'compact', 'studio']
        return any(keyword in text_lower for keyword in small_space_keywords)

    def infer_implicit_priorities(self,
                                  explicit_input: str,
                                  user_context: Optional[Dict[str, Any]] = None) -> InferenceResult:
        """
        從明確輸入和使用者上下文推斷隱含優先級

        Args:
            explicit_input: 使用者明確輸入
            user_context: 使用者上下文資訊

        Returns:
            InferenceResult: 推理結果
        """
        try:
            if user_context is None:
                user_context = {}

            implicit_priorities = {}
            triggered_rules = []
            reasoning_chains = []

            # 按優先級排序規則
            sorted_rules = sorted(self.inference_rules, key=lambda r: r.priority)

            # 應用推理規則
            for rule in sorted_rules:
                try:
                    if rule.condition(explicit_input, user_context):
                        # 規則觸發
                        implied = rule.imply(explicit_input, user_context)
                        triggered_rules.append(rule.name)
                        reasoning_chains.append(rule.reasoning)

                        # 合併隱含優先級（取最大值）
                        for dim, score in implied.items():
                            implicit_priorities[dim] = max(
                                implicit_priorities.get(dim, 1.0),
                                score
                            )
                except Exception as e:
                    print(f"Error applying rule {rule.name}: {str(e)}")
                    continue

            # 計算信心度
            confidence = self._calculate_inference_confidence(
                triggered_rules, explicit_input
            )

            return InferenceResult(
                implicit_priorities=implicit_priorities,
                triggered_rules=triggered_rules,
                reasoning_chains=reasoning_chains,
                confidence=confidence
            )

        except Exception as e:
            print(f"Error inferring implicit priorities: {str(e)}")
            print(traceback.format_exc())
            return InferenceResult()

    def _calculate_inference_confidence(self,
                                       triggered_rules: List[str],
                                       input_text: str) -> float:
        """
        計算推理信心度

        Args:
            triggered_rules: 觸發的規則列表
            input_text: 輸入文字

        Returns:
            float: 信心度 (0-1)
        """
        base_confidence = 0.6

        # 觸發的規則越多，信心度越高
        rule_bonus = min(0.3, len(triggered_rules) * 0.1)

        # 輸入文字越詳細，信心度越高
        word_count = len(input_text.split())
        detail_bonus = min(0.1, word_count / 100)

        return min(1.0, base_confidence + rule_bonus + detail_bonus)

    def get_inference_summary(self, result: InferenceResult) -> Dict[str, Any]:
        """
        獲取推理摘要

        Args:
            result: 推理結果

        Returns:
            Dict[str, Any]: 推理摘要
        """
        return {
            'total_implicit_priorities': len(result.implicit_priorities),
            'implicit_priorities': result.implicit_priorities,
            'triggered_rules': result.triggered_rules,
            'reasoning_chains': result.reasoning_chains,
            'inference_confidence': result.confidence,
            'high_confidence_inferences': [
                dim for dim, score in result.implicit_priorities.items() if score >= 1.4
            ]
        }


def infer_user_priorities(user_input: str,
                         user_context: Optional[Dict[str, Any]] = None) -> InferenceResult:
    """
    便利函數: 推斷使用者隱含優先級

    Args:
        user_input: 使用者輸入
        user_context: 使用者上下文

    Returns:
        InferenceResult: 推理結果
    """
    engine = BreedRecommendationInferenceEngine()
    return engine.infer_implicit_priorities(user_input, user_context)


def get_inference_summary(user_input: str,
                         user_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    便利函數: 獲取推理摘要

    Args:
        user_input: 使用者輸入
        user_context: 使用者上下文

    Returns:
        Dict[str, Any]: 推理摘要
    """
    engine = BreedRecommendationInferenceEngine()
    result = engine.infer_implicit_priorities(user_input, user_context)
    return engine.get_inference_summary(result)
