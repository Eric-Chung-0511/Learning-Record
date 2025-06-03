import re
import logging
import traceback
from typing import Dict, List, Any, Optional, Set, Tuple


class TextQualityValidator:
    """
    負責驗證和確保生成文本的品質和事實準確性。
    包含事實檢查、視角一致性、場景類型一致性等驗證功能。
    """

    def __init__(self):
        """初始化文本品質驗證器"""
        # 設置專屬logger
        self.logger = logging.getLogger(self.__class__.__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

        # 初始化驗證規則
        self._initialize_validation_rules()
        self.logger.info("TextQualityValidator initialized successfully")

    def _initialize_validation_rules(self):
        """初始化各種驗證規則和詞彙庫"""
        try:
            # 地點和文化詞彙列表
            self.location_terms = ["plaza", "square", "market", "mall", "avenue", "boulevard"]
            self.cultural_terms = ["european", "asian", "american", "african", "western", "eastern"]

            # 視角詞彙對應表
            self.perspective_terms = {
                "aerial": ["aerial", "bird's-eye", "overhead", "top-down", "above", "looking down"],
                "ground": ["street-level", "ground level", "eye-level", "standing"],
                "indoor": ["inside", "interior", "indoor", "within"],
                "close-up": ["close-up", "detailed view", "close shot"]
            }

            # 視角前綴對應表
            self.perspective_prefixes = {
                "aerial": "From an aerial perspective, ",
                "ground": "From street level, ",
                "indoor": "In this indoor setting, ",
                "close-up": "In this close-up view, "
            }

            # 數值檢測模式
            self.number_patterns = [
                (r'(\d+)\s+(people|person|pedestrians|individuals)', r'\1', r'\2'),
                (r'(\d+)\s+(cars|vehicles|automobiles)', r'\1', r'\2'),
                (r'(\d+)\s+(buildings|structures)', r'\1', r'\2'),
                (r'(\d+)\s+(plants|potted plants|flowers)', r'\1', r'\2'),
                (r'(\d+)\s+(beds|furniture|tables|chairs)', r'\1', r'\2')
            ]

            # 禁用場景詞列表
            self.prohibited_scene_words = ["plaza", "square", "european", "asian", "american"]

            self.logger.info("Validation rules initialized successfully")

        except Exception as e:
            error_msg = f"Failed to initialize validation rules: {str(e)}"
            self.logger.error(error_msg)
            self.logger.error(traceback.format_exc())
            raise Exception(error_msg) from e

    def verify_factual_accuracy(self,
                               original_desc: str,
                               generated_desc: str,
                               object_list: str) -> str:
        """
        驗證生成描述的事實準確性

        Args:
            original_desc: 原始場景描述
            generated_desc: 生成的描述
            object_list: 檢測到的物件列表

        Returns:
            str: 驗證並可能修正後的描述
        """
        try:
            self.logger.debug("Starting factual accuracy verification")

            # 將原始描述和物體列表合併為授權詞彙源
            authorized_content = original_desc.lower() + " " + object_list.lower()

            # 檢查和替換未授權的地點和文化詞彙
            verified_desc = self._check_unauthorized_terms(generated_desc, authorized_content)

            # 檢查重複用詞問題
            verified_desc = self._detect_repetitive_patterns(verified_desc)

            self.logger.debug("Factual accuracy verification completed")
            return verified_desc

        except Exception as e:
            error_msg = f"Factual accuracy verification failed: {str(e)}"
            self.logger.error(error_msg)
            self.logger.error(traceback.format_exc())
            return generated_desc  # 發生錯誤時返回原始生成描述

    def _check_unauthorized_terms(self, generated_desc: str, authorized_content: str) -> str:
        """檢查並替換未授權的詞彙"""
        # 檢查生成文本中的每個詞
        for term in self.location_terms + self.cultural_terms:
            # 僅當該詞出現在生成文本但不在授權內容中時進行替換
            if term in generated_desc.lower() and term not in authorized_content:
                # 根據詞語類型選擇適當的替換詞
                if term in self.location_terms:
                    replacement = "area"
                else:
                    replacement = "scene"

                # 使用正則表達式進行完整詞匹配替換
                pattern = re.compile(r'\b' + term + r'\b', re.IGNORECASE)
                generated_desc = pattern.sub(replacement, generated_desc)

        return generated_desc

    def _detect_repetitive_patterns(self, generated_desc: str) -> str:
        """檢測並處理重複用詞問題"""
        repetitive_patterns = [
            (r'\b(visible)\b.*?\b(visible)\b', 'Multiple uses of "visible" detected'),
            (r'\b(positioned)\b.*?\b(positioned)\b', 'Multiple uses of "positioned" detected'),
            (r'\b(located)\b.*?\b(located)\b', 'Multiple uses of "located" detected'),
            (r'\b(situated)\b.*?\b(situated)\b', 'Multiple uses of "situated" detected'),
            (r'\b(appears)\b.*?\b(appears)\b', 'Multiple uses of "appears" detected'),
            (r'\b(features)\b.*?\b(features)\b', 'Multiple uses of "features" detected'),
            (r'\bThis\s+(\w+)\s+.*?\bThis\s+\1\b', 'Repetitive sentence structure detected')
        ]

        # 替換詞典
        replacement_dict = {
            'visible': ['present', 'evident', 'apparent', 'observable'],
            'positioned': ['arranged', 'placed', 'set', 'organized'],
            'located': ['found', 'placed', 'situated', 'established'],
            'situated': ['placed', 'positioned', 'arranged', 'set'],
            'appears': ['seems', 'looks', 'presents', 'exhibits'],
            'features': ['includes', 'contains', 'displays', 'showcases']
        }

        for pattern, issue in repetitive_patterns:
            matches = list(re.finditer(pattern, generated_desc, re.IGNORECASE | re.DOTALL))
            if matches:
                self.logger.warning(f"Text quality issue detected: {issue}")

                # 針對特定重複詞彙進行替換
                for word in replacement_dict.keys():
                    if word in issue.lower():
                        word_pattern = re.compile(r'\b' + word + r'\b', re.IGNORECASE)
                        word_matches = list(word_pattern.finditer(generated_desc))

                        # 保留第一次出現，替換後續出現
                        for i, match in enumerate(word_matches[1:], 1):
                            if i <= len(replacement_dict[word]):
                                replacement = replacement_dict[word][(i-1) % len(replacement_dict[word])]

                                # 保持原始大小寫格式
                                if match.group().isupper():
                                    replacement = replacement.upper()
                                elif match.group().istitle():
                                    replacement = replacement.capitalize()

                                # 執行替換
                                generated_desc = generated_desc[:match.start()] + replacement + generated_desc[match.end():]
                                # 重新計算後續匹配位置
                                word_matches = list(word_pattern.finditer(generated_desc))
                        break

        return generated_desc

    def fact_check_description(self,
                             original_desc: str,
                             enhanced_desc: str,
                             scene_type: str,
                             detected_objects: List[str]) -> str:
        """
        對增強後的描述進行全面的事實檢查

        Args:
            original_desc: 原始場景描述
            enhanced_desc: 增強後的描述
            scene_type: 場景類型
            detected_objects: 檢測到的物體名稱列表

        Returns:
            str: 經過事實檢查的描述
        """
        try:
            self.logger.debug("Starting comprehensive fact checking")

            # 如果增強描述為空或太短，返回原始描述
            if not enhanced_desc or len(enhanced_desc) < 30:
                return original_desc

            # 1. 檢查數值一致性
            enhanced_desc = self._check_numerical_consistency(original_desc, enhanced_desc)

            # 2. 檢查視角一致性
            enhanced_desc = self._check_perspective_consistency(original_desc, enhanced_desc)

            # 3. 檢查場景類型一致性
            enhanced_desc = self._check_scene_type_consistency(enhanced_desc, scene_type)

            # 4. 確保文字長度適當
            enhanced_desc = self._ensure_appropriate_length(enhanced_desc)

            self.logger.debug("Comprehensive fact checking completed")
            return enhanced_desc

        except Exception as e:
            error_msg = f"Fact checking failed: {str(e)}"
            self.logger.error(error_msg)
            self.logger.error(traceback.format_exc())
            return enhanced_desc  # 發生錯誤時返回增強描述

    def _check_numerical_consistency(self, original_desc: str, enhanced_desc: str) -> str:
        """檢查數值一致性"""
        # 檢查原始描述中的每個數字
        for pattern, num_group, word_group in self.number_patterns:
            original_matches = re.finditer(pattern, original_desc, re.IGNORECASE)
            for match in original_matches:
                number = match.group(1)
                noun = match.group(2)

                # 檢查增強描述中是否保留了這個數字
                enhanced_pattern = r'(\d+)\s+(' + re.escape(noun) + r'|' + re.escape(noun.rstrip('s')) + r'|' + re.escape(noun + 's') + r')'
                enhanced_matches = list(re.finditer(enhanced_pattern, enhanced_desc, re.IGNORECASE))

                if not enhanced_matches:
                    # 數字+名詞未在增強描述中找到
                    plural_form = noun if noun.endswith('s') or number == '1' else noun + 's'
                    if enhanced_desc.startswith("This") or enhanced_desc.startswith("The"):
                        enhanced_desc = enhanced_desc.replace("This ", f"This scene with {number} {plural_form} ", 1)
                        enhanced_desc = enhanced_desc.replace("The ", f"The scene with {number} {plural_form} ", 1)
                    else:
                        enhanced_desc = f"The scene includes {number} {plural_form}. " + enhanced_desc
                elif enhanced_matches and enhanced_matches[0].group(1) != number:
                    # 存在但數字不一致，需要更正數字
                    for ematch in enhanced_matches:
                        wrong_number = ematch.group(1)
                        enhanced_desc = enhanced_desc.replace(f"{wrong_number} {ematch.group(2)}", f"{number} {ematch.group(2)}")

        return enhanced_desc

    def _check_perspective_consistency(self, original_desc: str, enhanced_desc: str) -> str:
        """檢查視角一致性"""
        # 確定原始視角
        original_perspective = None
        for persp, terms in self.perspective_terms.items():
            if any(term in original_desc.lower() for term in terms):
                original_perspective = persp
                break

        # 檢查是否保留了視角
        if original_perspective:
            enhanced_has_perspective = any(term in enhanced_desc.lower() for term in self.perspective_terms[original_perspective])

            if not enhanced_has_perspective:
                # 添加缺失的視角
                prefix = self.perspective_prefixes.get(original_perspective, "")
                if prefix:
                    if enhanced_desc[0].isupper():
                        enhanced_desc = prefix + enhanced_desc[0].lower() + enhanced_desc[1:]
                    else:
                        enhanced_desc = prefix + enhanced_desc

        return enhanced_desc

    def _check_scene_type_consistency(self, enhanced_desc: str, scene_type: str) -> str:
        """檢查場景類型一致性"""
        if scene_type and scene_type.lower() != "unknown" and scene_type.lower() not in enhanced_desc.lower():
            # 添加場景類型
            if enhanced_desc.startswith("This ") or enhanced_desc.startswith("The "):
                # 避免產生重複
                if "scene" in enhanced_desc[:15].lower():
                    fixed_type = scene_type.lower()
                    enhanced_desc = enhanced_desc.replace("scene", fixed_type, 1)
                else:
                    enhanced_desc = enhanced_desc.replace("This ", f"This {scene_type} ", 1)
                    enhanced_desc = enhanced_desc.replace("The ", f"The {scene_type} ", 1)
            else:
                enhanced_desc = f"This {scene_type} " + enhanced_desc

        return enhanced_desc

    def _ensure_appropriate_length(self, enhanced_desc: str) -> str:
        """確保文字長度適當"""
        words = enhanced_desc.split()
        if len(words) > 200:
            # 找尋接近字數限制的句子結束處
            truncated = ' '.join(words[:200])
            last_period = max(truncated.rfind('.'), truncated.rfind('!'), truncated.rfind('?'))

            if last_period > 0:
                enhanced_desc = truncated[:last_period+1]
            else:
                enhanced_desc = truncated + '.'

        return enhanced_desc

    def ensure_scene_type_consistency(self,
                                    description: str,
                                    scene_type: str,
                                    original_desc: str) -> str:
        """
        確保描述中的場景類型與指定的場景類型一致

        Args:
            description: 待檢查的描述
            scene_type: 指定的場景類型
            original_desc: 原始描述（用於參考）

        Returns:
            str: 場景類型一致的描述
        """
        try:
            self.logger.debug("Ensuring scene type consistency")
            scene_type = scene_type.replace('_', ' ')
            # 檢查是否包含禁止的場景詞
            for word in self.prohibited_scene_words:
                if word in description.lower() and word not in original_desc.lower() and word not in scene_type.lower():
                    # 替換錯誤場景詞為正確場景類型
                    pattern = re.compile(r'\b' + word + r'\b', re.IGNORECASE)
                    description = pattern.sub(scene_type, description)

            # 確保場景類型在描述中被提及
            if scene_type.lower() not in description.lower():
                # 尋找通用場景詞並替換
                for general_term in ["scene", "area", "place", "location"]:
                    if general_term in description.lower():
                        pattern = re.compile(r'\b' + general_term + r'\b', re.IGNORECASE)
                        description = pattern.sub(scene_type, description, count=1)
                        break
                else:
                    # 如果沒有找到通用詞，在開頭添加場景類型
                    if description.startswith("The "):
                        description = description.replace("The ", f"The {scene_type} ", 1)
                    elif description.startswith("This "):
                        description = description.replace("This ", f"This {scene_type} ", 1)
                    else:
                        description = f"This {scene_type} " + description

            self.logger.debug("Scene type consistency ensured")
            return description

        except Exception as e:
            error_msg = f"Scene type consistency check failed: {str(e)}"
            self.logger.error(error_msg)
            self.logger.error(traceback.format_exc())
            return description

    def extract_perspective_from_description(self, description: str) -> str:
        """
        從原始描述中提取視角信息

        Args:
            description: 原始場景描述

        Returns:
            str: 提取到的視角描述，如果沒有則返回空字符串
        """
        try:
            for persp_type, terms in self.perspective_terms.items():
                for term in terms:
                    if term.lower() in description.lower():
                        self.logger.debug(f"Perspective detected: {term}")
                        return term

            return ""

        except Exception as e:
            self.logger.error(f"Perspective extraction failed: {str(e)}")
            return ""

    def extract_objects_from_description(self, description: str) -> List[str]:
        """
        從原始描述中提取物件提及

        Args:
            description: 原始場景描述

        Returns:
            List[str]: 提取到的物件列表
        """
        try:
            extracted_objects = []

            for pattern in self.number_patterns:
                matches = re.finditer(pattern[0], description, re.IGNORECASE)
                for match in matches:
                    number = match.group(1)
                    object_type = match.group(2)
                    extracted_objects.append(f"{number} {object_type}")

            self.logger.debug(f"Extracted {len(extracted_objects)} objects from description")
            return extracted_objects

        except Exception as e:
            self.logger.error(f"Object extraction failed: {str(e)}")
            return []

    def validate_response_completeness(self, response: str) -> Tuple[bool, str]:
        """
        驗證回應的完整性

        Args:
            response: 待驗證的回應

        Returns:
            Tuple[bool, str]: (是否完整, 問題描述)
        """
        try:
            # 檢查回應長度
            if len(response) < 100:
                return False, "Response too short"

            # 檢查句子結尾
            if len(response) < 200 and "." not in response[-30:]:
                return False, "No proper sentence ending"

            # 檢查不完整短語
            incomplete_phrases = ["in the", "with the", "and the"]
            if any(response.endswith(phrase) for phrase in incomplete_phrases):
                return False, "Ends with incomplete phrase"

            return True, "Response is complete"

        except Exception as e:
            self.logger.error(f"Response completeness validation failed: {str(e)}")
            return False, "Validation error"

    def get_validator_info(self) -> Dict[str, Any]:
        """
        獲取驗證器信息

        Returns:
            Dict[str, Any]: 包含驗證器狀態和配置的信息
        """
        return {
            "location_terms_count": len(self.location_terms),
            "cultural_terms_count": len(self.cultural_terms),
            "perspective_types_count": len(self.perspective_terms),
            "number_patterns_count": len(self.number_patterns),
            "prohibited_words_count": len(self.prohibited_scene_words),
            "initialization_status": "success"
        }
