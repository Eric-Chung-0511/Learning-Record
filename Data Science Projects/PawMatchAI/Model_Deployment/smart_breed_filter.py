# %%writefile smart_breed_filter.py
"""
Smart Breed Filter - 智慧品種過濾系統

設計原則：
1. 只對「真正危害用戶」的情況進行干預
2. 無傷大雅的偏好差異維持原有評分邏輯
3. 所有規則基於通用性設計，不針對特定品種硬編碼

危害類型：
- 安全風險：幼童 + 高風險行為特徵
- 生活品質嚴重影響：噪音零容忍 + 焦慮/警戒吠叫品種
"""

from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from breed_noise_info import breed_noise_info


@dataclass
class UserPriorityContext:
    """用戶優先級上下文"""
    noise_intolerance: bool = False      # 噪音零容忍
    has_young_children: bool = False     # 有幼童
    is_beginner: bool = False            # 新手
    is_senior: bool = False              # 老年人
    priority_dimensions: Dict[str, str] = None  # 各維度優先級

    def __post_init__(self):
        if self.priority_dimensions is None:
            self.priority_dimensions = {}


class PriorityParser:
    """
    優先級語意解析器

    識別用戶是否對某些維度有「絕對需求」vs「一般偏好」
    只在用戶明確強調時才觸發嚴格約束
    """

    # 絕對需求信號詞
    ABSOLUTE_SIGNALS = [
        'most importantly', 'absolutely need', 'must have', 'essential',
        'critical', 'cannot', "can't", 'no way', 'zero tolerance',
        'very noise sensitive', 'neighbors complain', 'thin walls'
    ]

    # 主要需求信號詞
    PRIMARY_SIGNALS = [
        'first', 'primarily', 'main priority', 'most important',
        'first priority', 'number one'
    ]

    # 維度關鍵詞
    DIMENSION_KEYWORDS = {
        'noise': ['quiet', 'noise', 'bark', 'silent', 'neighbors',
                  'thin walls', 'apartment noise', 'loud', 'vocal'],
        'children': ['kids', 'children', 'child', 'toddler', 'baby',
                    'infant', 'young kids', 'aged 1', 'aged 2', 'aged 3',
                    'aged 4', 'aged 5', 'preschool'],
        'exercise': ['active', 'exercise', 'running', 'hiking', 'energetic',
                    'athletic', 'jogging', 'outdoor activities'],
        'grooming': ['maintenance', 'grooming', 'shedding', 'brush', 'coat',
                    'low maintenance', 'easy care'],
    }

    def parse(self, user_input: str) -> UserPriorityContext:
        """解析用戶輸入，提取優先級上下文"""
        text = user_input.lower()
        context = UserPriorityContext()

        # 檢測噪音零容忍
        context.noise_intolerance = self._detect_noise_intolerance(text)

        # 檢測是否有幼童
        context.has_young_children = self._detect_young_children(text)

        # 檢測各維度優先級
        context.priority_dimensions = self._detect_dimension_priorities(text)

        return context

    def _detect_noise_intolerance(self, text: str) -> bool:
        """
        檢測噪音零容忍

        只有當用戶明確表達噪音是嚴重問題時才觸發
        例如：thin walls, neighbors complain, noise sensitive neighbors
        """
        # 強烈噪音敏感信號
        strong_signals = [
            'thin walls', 'noise sensitive', 'neighbors complain',
            'zero tolerance', 'cannot bark', "can't bark",
            'absolutely quiet', 'must be quiet', 'noise restriction'
        ]

        # 需要同時出現「噪音相關詞」+「強調詞」
        noise_words = ['quiet', 'noise', 'bark', 'silent', 'loud']
        emphasis_words = ['most importantly', 'absolutely', 'must', 'essential',
                         'critical', 'very', 'extremely', 'cannot', "can't"]

        # 檢查強烈信號
        if any(signal in text for signal in strong_signals):
            return True

        # 檢查組合：噪音詞 + 強調詞
        has_noise_word = any(w in text for w in noise_words)
        has_emphasis = any(w in text for w in emphasis_words)

        return has_noise_word and has_emphasis

    def _detect_young_children(self, text: str) -> bool:
        """
        檢測是否有幼童或一般兒童

        對於兒童安全，我們採取保守策略：
        - 明確提到 kids/children 就視為有兒童風險需要考慮
        - 因為牧羊本能的 nipping 對任何年齡兒童都有風險
        """
        # 任何提到兒童的情況都需要考慮安全
        child_signals = [
            'kids', 'children', 'child', 'toddler', 'baby', 'infant',
            'young kids', 'young children',
            'aged 1', 'aged 2', 'aged 3', 'aged 4', 'aged 5',
            '1 year', '2 year', '3 year', '4 year', '5 year',
            'preschool', 'newborn', 'family with'
        ]
        return any(signal in text for signal in child_signals)

    def _detect_dimension_priorities(self, text: str) -> Dict[str, str]:
        """檢測各維度的優先級"""
        priorities = {}

        for dimension, keywords in self.DIMENSION_KEYWORDS.items():
            if any(kw in text for kw in keywords):
                # 檢查是否有絕對需求信號
                if any(signal in text for signal in self.ABSOLUTE_SIGNALS):
                    # 檢查信號是否與該維度相關（在附近）
                    for signal in self.ABSOLUTE_SIGNALS:
                        if signal in text:
                            signal_pos = text.find(signal)
                            for kw in keywords:
                                if kw in text:
                                    kw_pos = text.find(kw)
                                    # 如果信號詞和維度關鍵詞距離在50字符內
                                    if abs(signal_pos - kw_pos) < 80:
                                        priorities[dimension] = 'ABSOLUTE'
                                        break
                            if dimension in priorities:
                                break

                # 檢查是否有主要需求信號
                if dimension not in priorities:
                    if any(signal in text for signal in self.PRIMARY_SIGNALS):
                        priorities[dimension] = 'PRIMARY'
                    else:
                        priorities[dimension] = 'PREFERENCE'

        return priorities


class BreedRiskAnalyzer:
    """
    品種風險分析器

    只分析「真正的危害風險」，不對一般偏好差異進行干預
    """

    # 焦慮相關觸發詞（會導致持續吠叫的真正問題）
    ANXIETY_TRIGGERS = ['anxiety', 'separation anxiety', 'loneliness']

    # 高警戒觸發詞（會導致頻繁吠叫）
    HIGH_ALERT_TRIGGERS = ['stranger alerts', 'strangers approaching',
                           'suspicious activity', 'territorial defense',
                           'protecting territory']

    # 牧羊/追逐本能（對幼童有 nipping 風險）
    HERDING_INDICATORS = ['herding instincts', 'herding', 'nipping']

    # 獵物驅動（可能追逐小孩）
    PREY_DRIVE_INDICATORS = ['prey drive', 'prey sighting', 'chase']

    def analyze_noise_risk(self, breed_info: Dict, noise_info: Dict) -> Dict:
        """
        分析品種的噪音風險

        只標記「真正會造成問題」的品種：
        - 有焦慮吠叫傾向（持續性問題）
        - 高度警戒吠叫（頻繁問題）

        不標記：
        - 偶爾興奮吠叫（正常狗行為）
        - 打招呼吠叫（短暫且可控）
        """
        noise_notes = noise_info.get('noise_notes', '').lower()
        noise_level = noise_info.get('noise_level', 'Moderate').lower()
        temperament = breed_info.get('Temperament', '').lower()

        risk_factors = []

        # 1. 焦慮觸發 - 這是真正的問題（持續性吠叫）
        has_anxiety = any(t in noise_notes for t in self.ANXIETY_TRIGGERS)
        if has_anxiety:
            risk_factors.append('anxiety_barking')

        # 2. 高度警戒 - 頻繁吠叫風險
        has_high_alert = any(t in noise_notes for t in self.HIGH_ALERT_TRIGGERS)
        if has_high_alert:
            risk_factors.append('high_alert_barking')

        # 3. 敏感性格 + 焦慮觸發的組合（更嚴重）
        is_sensitive = 'sensitive' in temperament
        if is_sensitive and has_anxiety:
            risk_factors.append('sensitive_anxiety_combo')

        # 4. 基礎噪音等級高
        if noise_level in ['high', 'moderate-high', 'moderate to high']:
            risk_factors.append('high_base_noise')

        # 計算風險等級
        # 只有真正問題的組合才是 HIGH
        if 'sensitive_anxiety_combo' in risk_factors:
            risk_level = 'HIGH'
        elif 'anxiety_barking' in risk_factors and 'high_alert_barking' in risk_factors:
            risk_level = 'HIGH'
        elif 'anxiety_barking' in risk_factors or len(risk_factors) >= 2:
            risk_level = 'MODERATE'
        elif len(risk_factors) >= 1:
            risk_level = 'LOW'
        else:
            risk_level = 'NONE'

        return {
            'risk_level': risk_level,
            'risk_factors': risk_factors
        }

    def analyze_child_safety_risk(self, breed_info: Dict, noise_info: Dict) -> Dict:
        """
        分析品種對幼童的安全風險

        只標記「真正的安全風險」：
        - 牧羊本能（nipping 風險）
        - 高獵物驅動 + 大體型（追逐風險）
        - Good with Children = No 且有其他風險因素

        不標記：
        - 只是體型大但性格溫和
        - 活力高但無追逐/牧羊本能
        """
        temperament = breed_info.get('Temperament', '').lower()
        description = breed_info.get('Description', '').lower()
        noise_notes = noise_info.get('noise_notes', '').lower()
        size = breed_info.get('Size', '').lower()
        good_with_children = breed_info.get('Good with Children', 'Yes')
        exercise = breed_info.get('Exercise Needs', '').lower()

        risk_factors = []

        # 1. 牧羊本能 - 真正的 nipping 風險
        has_herding = any(ind in noise_notes or ind in description
                         for ind in self.HERDING_INDICATORS)
        if has_herding:
            risk_factors.append('herding_instinct')

        # 2. 獵物驅動 - 追逐風險
        has_prey_drive = any(ind in noise_notes or ind in description
                            for ind in self.PREY_DRIVE_INDICATORS)
        if has_prey_drive:
            risk_factors.append('prey_drive')

        # 3. Good with Children = No 是強烈信號
        if good_with_children == 'No':
            risk_factors.append('not_child_friendly')

        # 4. 大體型 + 高驅動 + 牧羊/獵物本能的組合才是風險
        is_large = size in ['large', 'giant']
        is_very_high_energy = 'very high' in exercise

        if is_large and (has_herding or has_prey_drive) and is_very_high_energy:
            risk_factors.append('large_high_drive_instinct')

        # 計算風險等級
        # 只有真正危險的組合才是 HIGH
        if 'not_child_friendly' in risk_factors and len(risk_factors) >= 2:
            risk_level = 'HIGH'
        elif 'large_high_drive_instinct' in risk_factors:
            risk_level = 'HIGH'
        elif 'herding_instinct' in risk_factors and is_very_high_energy:
            # 牧羊本能 + 高能量 = 對兒童的真正風險（nipping + 控制不住）
            risk_level = 'HIGH'
        elif 'herding_instinct' in risk_factors or 'prey_drive' in risk_factors:
            # 單獨的牧羊或獵物本能仍是中等風險
            risk_level = 'MODERATE'
        elif 'not_child_friendly' in risk_factors:
            risk_level = 'MODERATE'
        elif len(risk_factors) >= 1:
            risk_level = 'LOW'
        else:
            risk_level = 'NONE'

        return {
            'risk_level': risk_level,
            'risk_factors': risk_factors
        }


class SmartBreedFilter:
    """
    智慧品種過濾器

    整合優先級解析和風險分析，只對真正危害用戶的情況進行干預
    """

    def __init__(self):
        self.priority_parser = PriorityParser()
        self.risk_analyzer = BreedRiskAnalyzer()

    def analyze_user_context(self, user_input: str) -> UserPriorityContext:
        """分析用戶輸入，提取優先級上下文"""
        return self.priority_parser.parse(user_input)

    def should_exclude_breed(self, breed_info: Dict, noise_info: Dict,
                            user_context: UserPriorityContext) -> Tuple[bool, str]:
        """
        判斷是否應該排除該品種

        返回: (是否排除, 排除原因)
        """
        # 1. 噪音零容忍 + 高噪音風險
        if user_context.noise_intolerance:
            noise_risk = self.risk_analyzer.analyze_noise_risk(breed_info, noise_info)
            if noise_risk['risk_level'] == 'HIGH':
                return True, f"High noise risk ({', '.join(noise_risk['risk_factors'])}) conflicts with noise intolerance"

        # 2. 有幼童 + 高兒童安全風險
        if user_context.has_young_children:
            child_risk = self.risk_analyzer.analyze_child_safety_risk(breed_info, noise_info)
            if child_risk['risk_level'] == 'HIGH':
                return True, f"Child safety risk ({', '.join(child_risk['risk_factors'])}) with young children"

        return False, ""

    def calculate_risk_penalty(self, breed_info: Dict, noise_info: Dict,
                               user_context: UserPriorityContext) -> float:
        """
        計算風險懲罰分數

        只對中等風險進行輕微降權，不排除
        返回: 懲罰係數 (0.0 - 0.3)
        """
        penalty = 0.0

        # 噪音相關懲罰（只在用戶關注噪音時）
        if 'noise' in user_context.priority_dimensions:
            noise_risk = self.risk_analyzer.analyze_noise_risk(breed_info, noise_info)
            if noise_risk['risk_level'] == 'MODERATE':
                penalty += 0.1
            elif noise_risk['risk_level'] == 'HIGH' and not user_context.noise_intolerance:
                penalty += 0.15

        # 兒童安全相關懲罰（只在用戶有孩子時）
        if 'children' in user_context.priority_dimensions or user_context.has_young_children:
            child_risk = self.risk_analyzer.analyze_child_safety_risk(breed_info, noise_info)
            if child_risk['risk_level'] == 'MODERATE':
                penalty += 0.1
            elif child_risk['risk_level'] == 'HIGH' and not user_context.has_young_children:
                penalty += 0.15

        return min(penalty, 0.3)  # 最大懲罰 30%

    def filter_and_adjust_recommendations(self, recommendations: List[Dict],
                                          user_input: str) -> List[Dict]:
        """
        過濾並調整推薦結果

        這是主要入口函數，整合所有過濾和調整邏輯
        """
        user_context = self.analyze_user_context(user_input)

        filtered_recommendations = []

        for rec in recommendations:
            breed = rec.get('breed', '')

            # 智能獲取品種資訊：優先從 info 欄位，否則從 rec 本身，最後從資料庫
            breed_info = rec.get('info')
            if not breed_info:
                # 嘗試從 rec 中構建標準化的 breed_info（處理大小寫差異）
                breed_info = {
                    'Temperament': rec.get('Temperament', rec.get('temperament', '')),
                    'Description': rec.get('Description', rec.get('description', '')),
                    'Size': rec.get('Size', rec.get('size', '')),
                    'Exercise Needs': rec.get('Exercise Needs', rec.get('exercise_needs', '')),
                    'Good with Children': rec.get('Good with Children', rec.get('good_with_children', 'Yes')),
                    'Care Level': rec.get('Care Level', rec.get('care_level', '')),
                }
                # 如果關鍵資訊缺失，從資料庫獲取
                if not breed_info['Temperament'] and not breed_info['Description']:
                    from dog_database import get_dog_description
                    db_info = get_dog_description(breed.replace(' ', '_'))
                    if db_info:
                        breed_info = db_info

            # 獲取噪音資訊（嘗試兩種品種名稱格式）
            noise_info = breed_noise_info.get(breed) or breed_noise_info.get(breed.replace(' ', '_'), {
                'noise_notes': '',
                'noise_level': 'Moderate'
            })

            # 檢查是否應該排除
            should_exclude, reason = self.should_exclude_breed(
                breed_info, noise_info, user_context
            )

            if should_exclude:
                print(f"  [SmartFilter] Excluded {breed}: {reason}")
                continue

            # 計算風險懲罰
            penalty = self.calculate_risk_penalty(breed_info, noise_info, user_context)

            if penalty > 0:
                original_score = rec.get('final_score', rec.get('overall_score', 0.8))
                adjusted_score = original_score * (1 - penalty)
                rec['final_score'] = adjusted_score
                rec['risk_penalty'] = penalty

            filtered_recommendations.append(rec)

        # 重新排序
        filtered_recommendations.sort(key=lambda x: -x.get('final_score', 0))

        # 更新排名
        for i, rec in enumerate(filtered_recommendations):
            rec['rank'] = i + 1

        return filtered_recommendations


# 模組級便捷函數
_smart_filter = None

def get_smart_filter() -> SmartBreedFilter:
    """獲取單例過濾器"""
    global _smart_filter
    if _smart_filter is None:
        _smart_filter = SmartBreedFilter()
    return _smart_filter

def apply_smart_filtering(recommendations: List[Dict], user_input: str) -> List[Dict]:
    """便捷函數：應用智慧過濾"""
    return get_smart_filter().filter_and_adjust_recommendations(recommendations, user_input)
