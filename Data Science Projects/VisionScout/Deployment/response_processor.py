import re
import logging
import traceback
from typing import Dict, List, Any, Optional, Set


class ResponseProcessingError(Exception):
    """回應處理相關錯誤的自定義異常"""
    pass


class ResponseProcessor:
    """
    負責處理和清理LLM模型輸出的回應。
    包含格式清理、重複內容檢測、語法完整性確保等功能。
    """

    def __init__(self):
        """初始化回應處理器"""
        # set the logger
        self.logger = logging.getLogger(self.__class__.__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

        # 初始化清理規則和替換字典
        self._initialize_cleaning_rules()
        self.logger.info("ResponseProcessor initialized successfully")


    def _initialize_cleaning_rules(self):
        """初始化各種清理規則和替換字典，把常見有問題情況優化"""
        try:
            # 設置重複詞彙的替換字典
            self.replacement_alternatives = {
                'visible': ['present', 'evident', 'apparent', 'observable'],
                'positioned': ['arranged', 'placed', 'set', 'organized'],
                'located': ['found', 'placed', 'situated', 'established'],
                'situated': ['placed', 'positioned', 'arranged', 'set'],
                'appears': ['seems', 'looks', 'presents', 'exhibits'],
                'features': ['includes', 'contains', 'displays', 'showcases'],
                'shows': ['reveals', 'presents', 'exhibits', 'demonstrates'],
                'displays': ['presents', 'exhibits', 'shows', 'reveals']
            }

            # 設置需要移除的前綴短語
            self.prefixes_to_remove = [
                "Here's the enhanced description:",
                "Enhanced description:",
                "Here is the enhanced scene description:",
                "I've enhanced the description while preserving all factual details:",
                "Enhanced Description:",
                "Scene Description:",
                "Description:",
                "Here is the enhanced description:",
                "Here's the enhanced description:",
                "Here is a rewritten scene description that adheres to the provided critical rules:",
                "Here is the rewritten scene description:",
                "Here's a rewritten scene description:",
                "The rewritten scene description is as follows:"
            ]

            # 設置需要移除的後綴短語
            self.suffixes_to_remove = [
                "I've maintained all the key factual elements",
                "I've preserved all the factual details",
                "All factual elements have been maintained"
            ]

            # 設置重複檢測模式
            self.repetitive_patterns = [
                (r'\b(visible)\b.*?\b(visible)\b', 'Multiple uses of "visible" detected'),
                (r'\b(positioned)\b.*?\b(positioned)\b', 'Multiple uses of "positioned" detected'),
                (r'\b(located)\b.*?\b(located)\b', 'Multiple uses of "located" detected'),
                (r'\b(situated)\b.*?\b(situated)\b', 'Multiple uses of "situated" detected'),
                (r'\b(appears)\b.*?\b(appears)\b', 'Multiple uses of "appears" detected'),
                (r'\b(features)\b.*?\b(features)\b', 'Multiple uses of "features" detected'),
                (r'\bThis\s+(\w+)\s+.*?\bThis\s+\1\b', 'Repetitive sentence structure detected')
            ]

            # 斜線組合的形容詞替換字典(有時會有斜線格式問題)
            self.slash_replacements = {
                'sunrise/sunset': 'warm lighting',
                'sunset/sunrise': 'warm lighting',
                'day/night': 'ambient lighting',
                'night/day': 'ambient lighting',
                'morning/evening': 'soft lighting',
                'evening/morning': 'soft lighting',
                'dawn/dusk': 'gentle lighting',
                'dusk/dawn': 'gentle lighting',
                'sunny/cloudy': 'natural lighting',
                'cloudy/sunny': 'natural lighting',
                'bright/dark': 'varied lighting',
                'dark/bright': 'varied lighting',
                'light/shadow': 'contrasting illumination',
                'shadow/light': 'contrasting illumination',
                'indoor/outdoor': 'mixed environment',
                'outdoor/indoor': 'mixed environment',
                'inside/outside': 'transitional space',
                'outside/inside': 'transitional space',
                'urban/rural': 'diverse landscape',
                'rural/urban': 'diverse landscape',
                'modern/traditional': 'architectural blend',
                'traditional/modern': 'architectural blend',
                'old/new': 'varied architecture',
                'new/old': 'varied architecture',
                'busy/quiet': 'dynamic atmosphere',
                'quiet/busy': 'dynamic atmosphere',
                'crowded/empty': 'varying occupancy',
                'empty/crowded': 'varying occupancy',
                'hot/cold': 'comfortable temperature',
                'cold/hot': 'comfortable temperature',
                'wet/dry': 'mixed conditions',
                'dry/wet': 'mixed conditions',
                'summer/winter': 'seasonal atmosphere',
                'winter/summer': 'seasonal atmosphere',
                'spring/autumn': 'transitional season',
                'autumn/spring': 'transitional season',
                'left/right': 'balanced composition',
                'right/left': 'balanced composition',
                'near/far': 'layered perspective',
                'far/near': 'layered perspective',
                'high/low': 'varied elevation',
                'low/high': 'varied elevation',
                'big/small': 'diverse scale',
                'small/big': 'diverse scale',
                'wide/narrow': 'varied width',
                'narrow/wide': 'varied width',
                'open/closed': 'flexible space',
                'closed/open': 'flexible space',
                'public/private': 'community space',
                'private/public': 'community space',
                'formal/informal': 'relaxed setting',
                'informal/formal': 'relaxed setting',
                'commercial/residential': 'mixed-use area',
                'residential/commercial': 'mixed-use area'
            }

            # 新增：擴展的底線替換字典
            self.underscore_replacements = {
                'urban_intersection': 'urban intersection',
                'tourist_landmark': 'tourist landmark',
                'historical_site': 'historical site',
                'religious_building': 'religious building',
                'natural_landmark': 'natural landmark',
                'commercial_area': 'commercial area',
                'residential_area': 'residential area',
                'public_space': 'public space',
                'outdoor_scene': 'outdoor scene',
                'indoor_scene': 'indoor scene',
                'street_scene': 'street scene',
                'city_center': 'city center',
                'shopping_district': 'shopping district',
                'business_district': 'business district',
                'traffic_light': 'traffic light',
                'street_lamp': 'street lamp',
                'parking_meter': 'parking meter',
                'fire_hydrant': 'fire hydrant',
                'bus_stop': 'bus stop',
                'train_station': 'train station',
                'police_car': 'police car',
                'fire_truck': 'fire truck',
                'school_bus': 'school bus',
                'time_of_day': 'time of day',
                'weather_condition': 'weather condition',
                'lighting_condition': 'lighting condition',
                'atmospheric_condition': 'atmospheric condition',
                'human_activity': 'human activity',
                'pedestrian_traffic': 'pedestrian traffic',
                'vehicle_traffic': 'vehicle traffic',
                'social_gathering': 'social gathering',
                'object_detection': 'object detection',
                'scene_analysis': 'scene analysis',
                'image_classification': 'image classification',
                'computer_vision': 'computer vision'
            }

            self.logger.info("Cleaning rules initialized successfully")

        except Exception as e:
            error_msg = f"Failed to initialize cleaning rules: {str(e)}"
            self.logger.error(error_msg)
            self.logger.error(traceback.format_exc())
            raise ResponseProcessingError(error_msg) from e

    def clean_response(self, response: str, model_type: str = "general") -> str:
        """
        清理LLM回應

        Args:
            response: 原始LLM回應
            model_type: 模型類型（用於特定清理規則）

        Returns:
            str: 清理後的回應

        Raises:
            ResponseProcessingError: 當回應處理失敗時
        """
        if not response:
            raise ResponseProcessingError("Empty response provided for cleaning")

        try:
            self.logger.debug(f"Starting response cleaning (original length: {len(response)})")

            # 保存原始回應作為備份
            original_response = response

            # 根據模型類型選擇清理策略
            if "llama" in model_type.lower():
                cleaned_response = self._clean_llama_response(response)
            else:
                cleaned_response = self._clean_general_response(response)

            # 如果清理後內容過短，嘗試從原始回應中恢復
            if len(cleaned_response.strip()) < 40:
                self.logger.warning("Cleaned response too short, attempting recovery")
                cleaned_response = self._recover_from_overcleaning(original_response)

            # 最終驗證
            self._validate_cleaned_response(cleaned_response)

            self.logger.debug(f"Response cleaning completed (final length: {len(cleaned_response)})")
            return cleaned_response

        except Exception as e:
            error_msg = f"Response cleaning failed: {str(e)}"
            self.logger.error(error_msg)
            self.logger.error(traceback.format_exc())
            raise ResponseProcessingError(error_msg) from e

    def _clean_llama_response(self, response: str) -> str:
        """
        專門處理Llama模型的回應清理

        Args:
            response: 原始Llama回應

        Returns:
            str: 清理後的回應
        """
        # 首先應用通用清理
        response = self._clean_general_response(response)

        # Llama特有的前綴清理
        llama_prefixes = [
            "Here's the enhanced description:",
            "Enhanced description:",
            "Here is the enhanced scene description:",
            "I've enhanced the description while preserving all factual details:"
        ]

        for prefix in llama_prefixes:
            if response.lower().startswith(prefix.lower()):
                response = response[len(prefix):].strip()

        # Llama特有的後綴清理
        llama_suffixes = [
            "I've maintained all the key factual elements",
            "I've preserved all the factual details",
            "All factual elements have been maintained"
        ]

        for suffix in llama_suffixes:
            if response.lower().endswith(suffix.lower()):
                response = response[:response.rfind(suffix)].strip()

        return response

    def _clean_general_response(self, response: str) -> str:
        """
        通用回應清理方法

        Args:
            response: 原始回應

        Returns:
            str: 清理後的回應
        """
        response = self._critical_format_preprocess(response)

        # 1. 移除系統remark
        response = self._remove_system_markers(response)

        # 2. 移除介紹性prefix
        response = self._remove_introduction_prefixes(response)

        # 3. 移除格式標記和上下文標籤
        response = self._remove_format_markers(response)

        # 4. 清理場景類型引用
        response = self._clean_scene_type_references(response)

        # 5. 標準化標點符號
        response = self._normalize_punctuation(response)

        # 6. 移除重複句子
        response = self._remove_duplicate_sentences(response)

        # 7. 處理重複詞彙
        response = self._handle_repetitive_vocabulary(response)

        # 8. ensure completement
        response = self._ensure_grammatical_completeness(response)

        # 9. 控制字數長度
        response = self._control_word_length(response)

        # 10. 最終格式化
        response = self._final_formatting(response)

        return response


    def _critical_format_preprocess(self, response: str) -> str:
        """
        關鍵格式預處理，處理最常見的格式問題

        Args:
            response: 原始回應

        Returns:
            str: 預處理後的回應
        """
        if not response:
            return response

        try:
            import re

            # 第一優先級：處理斜線問題
            # 首先處理已知的斜線組合，使用形容詞替換
            for slash_combo, replacement in self.slash_replacements.items():
                if slash_combo.lower() in response.lower():
                    # 保持原始大小寫格式
                    if slash_combo.upper() in response:
                        replacement_formatted = replacement.upper()
                    elif slash_combo.title() in response:
                        replacement_formatted = replacement.title()
                    else:
                        replacement_formatted = replacement

                    # 執行替換（不區分大小寫）
                    response = re.sub(re.escape(slash_combo), replacement_formatted, response, flags=re.IGNORECASE)
                    self.logger.debug(f"Replaced slash pattern '{slash_combo}' with '{replacement_formatted}'")

            # 處理其他未預定義的斜線模式
            # 標準斜線模式：word/word
            slash_pattern = r'\b([a-zA-Z]+)/([a-zA-Z]+)\b'
            matches = list(re.finditer(slash_pattern, response))
            for match in reversed(matches):  # 從後往前處理避免位置偏移
                word1, word2 = match.groups()
                # 選擇較短或更常見的詞作為替換
                if len(word1) <= len(word2):
                    replacement = word1
                else:
                    replacement = word2
                response = response[:match.start()] + replacement + response[match.end():]
                self.logger.debug(f"Replaced general slash pattern '{match.group(0)}' with '{replacement}'")

            # 第二優先級：處理底線格式
            # 首先處理已知的底線組合
            for underscore_combo, replacement in self.underscore_replacements.items():
                if underscore_combo in response:
                    response = response.replace(underscore_combo, replacement)
                    self.logger.debug(f"Replaced underscore pattern '{underscore_combo}' with '{replacement}'")

            # 處理三個詞的底線組合：word_word_word → word word word
            response = re.sub(r'\b([a-z]+)_([a-z]+)_([a-z]+)\b', r'\1 \2 \3', response)

            # 處理任何剩餘的底線模式：word_word → word word
            response = re.sub(r'\b([a-zA-Z]+)_([a-zA-Z]+)\b', r'\1 \2', response)

            # 第三優先級：修正不完整句子
            incomplete_sentence_fixes = [
                (r'\bIn\s*,\s*', 'Throughout the area, '),
                (r'\bOverall,\s+exudes\b', 'Overall, the scene exudes'),
                (r'\bThe overall atmosphere of\s+is\b', 'The overall atmosphere'),
                (r'\bwith its lights turned illuminating\b', 'with its lights illuminating'),
                (r'\bwhere it stands as\b', 'where it stands as'),
            ]

            for pattern, replacement in incomplete_sentence_fixes:
                response = re.sub(pattern, replacement, response, flags=re.IGNORECASE)

            # 第四優先級：語法修正處理(像是person and people)
            grammar_fixes = [
                (r'\b(\d+)\s+persons\b', r'\1 people'),
                (r'\bone\s+persons\b', 'one person'),
                (r'\btwo\s+persons\b', 'two people'),
                (r'\bthree\s+persons\b', 'three people'),
                (r'\bfour\s+persons\b', 'four people'),
                (r'\bfive\s+persons\b', 'five people'),
                (r'\bsix\s+persons\b', 'six people'),
                (r'\bseven\s+persons\b', 'seven people'),
                (r'\beight\s+persons\b', 'eight people'),
                (r'\bnine\s+persons\b', 'nine people'),
                (r'\bten\s+persons\b', 'ten people'),
                (r'\bmultiple\s+persons\b', 'multiple people'),
                (r'\bseveral\s+persons\b', 'several people'),
                (r'\bmany\s+persons\b', 'many people'),
                (r'\ba\s+few\s+persons\b', 'a few people'),
                (r'\bsome\s+persons\b', 'some people')
            ]

            for pattern, replacement in grammar_fixes:
                response = re.sub(pattern, replacement, response, flags=re.IGNORECASE)

            return response

        except Exception as e:
            self.logger.warning(f"Error in critical format preprocessing: {str(e)}")
            return response

    def _remove_system_markers(self, response: str) -> str:
        """移除系統樣式標記"""
        # 移除對話remark
        response = re.sub(r'<\|.*?\|>', '', response)

        # 移除輸出remark
        output_start = response.find("[OUTPUT_START]")
        output_end = response.find("[OUTPUT_END]")
        if output_start != -1 and output_end != -1 and output_end > output_start:
            response = response[output_start + len("[OUTPUT_START]"):output_end].strip()

        # 移除其他remark
        section_markers = [
            r'\[.*?\]',
            r'OUTPUT_START\s*:|OUTPUT_END\s*:',
            r'ENHANCED DESCRIPTION\s*:',
            r'Scene Type\s*:.*?(?=\n|$)',
            r'Original Description\s*:.*?(?=\n|$)',
            r'GOOD\s*:|BAD\s*:',
            r'PROBLEM\s*:.*?(?=\n|$)',
            r'</?\|(?:assistant|system|user)\|>',
            r'\(Note:.*?\)',
            r'\(.*?I\'ve.*?\)',
            r'\(.*?as per your request.*?\)'
        ]

        for marker in section_markers:
            response = re.sub(marker, '', response, flags=re.IGNORECASE)

        return response

    def _remove_introduction_prefixes(self, response: str) -> str:
        """移除介紹性前綴"""
        # 處理 "Here is..." 類型的prefix
        intro_prefixes = [
            r'^Here\s+is\s+(?:a\s+|the\s+)?(?:rewritten\s+|enhanced\s+)?scene\s+description.*?:\s*',
            r'^The\s+(?:rewritten\s+|enhanced\s+)?(?:scene\s+)?description\s+is.*?:\s*',
            r'^Here\'s\s+(?:a\s+|the\s+)?(?:rewritten\s+|enhanced\s+)?description.*?:\s*'
        ]

        for prefix_pattern in intro_prefixes:
            response = re.sub(prefix_pattern, '', response, flags=re.IGNORECASE)

        # 處理固定prefix
        for prefix in self.prefixes_to_remove:
            if response.lower().startswith(prefix.lower()):
                response = response[len(prefix):].strip()

        return response

    def _remove_format_markers(self, response: str) -> str:
        """移除格式標記和上下文標籤（保留括號內的地理與細節資訊）"""
        # 移除上下文相關remark
        response = re.sub(r'<\s*Context:.*?>', '', response)
        response = re.sub(r'Context:.*?(?=\n|$)', '', response)
        response = re.sub(r'Note:.*?(?=\n|$)', '', response, flags=re.IGNORECASE)

        # 移除Markdown格式
        response = re.sub(r'\*\*|\*|__|\|', '', response)

        # 移除任何剩餘的特殊標記 (避開括號內容，以免剔除地理位置等有用資訊)
        response = re.sub(r'</?\|.*?\|>', '', response)
        # ※ 以下移除「刪除整個括號及其內文」的方式已註解，以保留地理位置資訊
        # response = re.sub(r'\(.*?\)', '', response)

        return response


    def _clean_scene_type_references(self, response: str) -> str:
        """清理不當的場景類型引用"""
        scene_type_pattern = r'This ([a-zA-Z_]+) (features|shows|displays|contains)'
        match = re.search(scene_type_pattern, response)
        if match and '_' in match.group(1):
            fixed_text = f"This scene {match.group(2)}"
            response = re.sub(scene_type_pattern, fixed_text, response)

        return response

    def _normalize_punctuation(self, response: str) -> str:
        """標準化標點符號"""
        # 減少破折號使用
        response = re.sub(r'—', ', ', response)
        response = re.sub(r' - ', ', ', response)

        # 處理連續標點符號
        response = re.sub(r'([.,;:!?])\1+', r'\1', response)

        # 修復不完整句子的標點
        response = re.sub(r',\s*$', '.', response)

        # 修復句號後缺少空格的問題
        response = re.sub(r'([.!?])([A-Z])', r'\1 \2', response)

        # 清理多餘空格和換行
        response = response.replace('\r', ' ')
        response = re.sub(r'\n+', ' ', response)
        response = re.sub(r'\s{2,}', ' ', response)

        return response


    def _remove_duplicate_sentences(self, response: str, similarity_threshold: float = 0.85) -> str:
        """
        移除重複或高度相似的句子，使用 Jaccard 相似度進行比較。
        Args:
            response: 原始回應文本。
            similarity_threshold: 認定句子重複的相似度閾值 (0.0 到 1.0)。
                                  較高的閾值表示句子需要非常相似才會被移除。
        Returns:
            str: 移除重複句子後的文本。
        """
        try:
            if not response or not response.strip():
                return ""

            # (?<=[.!?]) 會保留分隔符在句尾, \s+ 會消耗句尾的空格
            # 這樣用 ' ' join 回去時, 標點和下個句子間剛好一個空格
            sentences = re.split(r'(?<=[.!?])\s+', response.strip())

            unique_sentences_data = [] # Store tuples of (original_sentence, simplified_word_set)

            min_sentence_len_for_check = 8 # 簡化後詞彙數少於此值，除非完全相同否則不輕易判斷為重複

            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue

                # 創建簡化版本用於比較 (小寫，移除標點，分割為詞彙集合)
                # 保留數字，因為數字可能是關鍵資訊
                simplified_text = re.sub(r'[^\w\s\d]', '', sentence.lower())
                current_sentence_words = set(simplified_text.split())

                if not current_sentence_words: # 如果處理後是空集合，跳過
                    continue

                is_duplicate = False
                # 與已保留的唯一句子比較
                for i, (kept_sentence_text, kept_sentence_words) in enumerate(unique_sentences_data):
                    # Jaccard Index
                    intersection_len = len(current_sentence_words.intersection(kept_sentence_words))
                    union_len = len(current_sentence_words.union(kept_sentence_words))

                    if union_len == 0: # 兩個都是空集合，代表相同句子
                        jaccard_similarity = 1.0
                    else:
                        jaccard_similarity = intersection_len / union_len

                    # 用Jaccard 相似度超過閾值，不是兩個都非常短的句子 (避免 "Yes." 和 "No." 被錯誤合併)
                    # 新句子完全被舊句子包含 (且舊句子更長)
                    # 舊句子完全被新句子包含 (且新句子更長) -> 這種情況就需要替換
                    if jaccard_similarity >= similarity_threshold:
                        # 如果當前句子比已保留的句子短，且高度相似，則認為是重複
                        if len(current_sentence_words) < len(kept_sentence_words):
                            is_duplicate = True
                            self.logger.debug(f"Sentence \"{sentence[:30]}...\" marked duplicate (shorter, similar to \"{kept_sentence_text[:30]}...\") Jaccard: {jaccard_similarity:.2f}")
                            break
                        # 如果當前句子比已保留的句子長，且高度相似，則替換掉已保留的
                        elif len(current_sentence_words) > len(kept_sentence_words):
                            self.logger.debug(f"Sentence \"{kept_sentence_text[:30]}...\" replaced by longer similar sentence \"{sentence[:30]}...\" Jaccard: {jaccard_similarity:.2f}")
                            unique_sentences_data.pop(i) # 移除舊的、較短的句子

                        # 如果長度差不多，但相似度高，保留第一個出現的
                        elif current_sentence_words != kept_sentence_words : # 避免完全相同的句子被錯誤地跳過替換邏輯
                             is_duplicate = True # 保留先出現的
                             self.logger.debug(f"Sentence \"{sentence[:30]}...\" marked duplicate (similar length, similar to \"{kept_sentence_text[:30]}...\") Jaccard: {jaccard_similarity:.2f}")
                             break

                if not is_duplicate:
                    unique_sentences_data.append((sentence, current_sentence_words))

            # 重組唯一句子
            final_sentences = [s_data[0] for s_data in unique_sentences_data]

            # 確保每個句子以標點結尾 (因為 split 可能會產生沒有標點的最後一個片段)
            reconstructed_response = ""
            for i, s in enumerate(final_sentences):
                s = s.strip()
                if not s: continue
                if not s[-1] in ".!?":
                    s += "."
                reconstructed_response += s
                if i < len(final_sentences) - 1:
                     reconstructed_response += " " # 在句子間添加空格

            return reconstructed_response.strip()

        except Exception as e:
            self.logger.error(f"Error in _remove_duplicate_sentences: {str(e)}")
            self.logger.error(traceback.format_exc())
            return response # 發生錯誤時返回原始回應

    def _handle_repetitive_vocabulary(self, response: str) -> str:
        """處理重複詞彙，使用 re.sub 和可呼叫的替換函數以提高效率和準確性。"""
        try:
            # 檢測重複模式 (僅警告)
            if hasattr(self, 'repetitive_patterns'):
                for pattern, issue in self.repetitive_patterns:
                    if re.search(pattern, response, re.IGNORECASE | re.DOTALL):
                        self.logger.warning(f"Text quality issue detected: {issue} in response: \"{response[:100]}...\"")

            if not hasattr(self, 'replacement_alternatives') or not self.replacement_alternatives:
                return response

            processed_response = response

            for word_to_replace, alternatives in self.replacement_alternatives.items():
                if not alternatives:  # 如果沒有可用的替代詞，則跳過
                    continue

                # 為每個詞創建一個獨立的計數器和替代索引
                # 使用閉包或一個小類來封裝狀態
                class WordReplacer:
                    def __init__(self, alternatives_list):
                        self.count = 0
                        self.alternative_idx = 0
                        self.alternatives_list = alternatives_list

                    def __call__(self, match_obj):
                        self.count += 1
                        original_word = match_obj.group(0)
                        if self.count > 1:  # 從第二次出現開始替換
                            replacement = self.alternatives_list[self.alternative_idx % len(self.alternatives_list)]
                            self.alternative_idx += 1
                            # 保持原始大小寫格式
                            if original_word.isupper():
                                return replacement.upper()
                            elif original_word.istitle():
                                return replacement.capitalize()
                            return replacement
                        return original_word # 因為第一次出現, 就不用替換

                replacer_instance = WordReplacer(alternatives)
                # 使用 \b 確保匹配的是整個單詞
                pattern = re.compile(r'\b' + re.escape(word_to_replace) + r'\b', re.IGNORECASE)
                processed_response = pattern.sub(replacer_instance, processed_response)

            # 移除 identical 等重複性描述詞彙
            identical_cleanup_patterns = [
                (r'\b(\d+)\s+identical\s+([a-zA-Z\s]+)', r'\1 \2'),
                (r'\b(two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+identical\s+([a-zA-Z\s]+)', r'\1 \2'),
                (r'\bidentical\s+([a-zA-Z\s]+)', r'\1'),
                (r'\bcomprehensive arrangement of\b', 'arrangement of'),
                (r'\bcomprehensive view featuring\b', 'scene featuring'),
                (r'\bcomprehensive display of\b', 'display of'),
            ]

            for pattern, replacement in identical_cleanup_patterns:
                processed_response = re.sub(pattern, replacement, processed_response, flags=re.IGNORECASE)

            # 數字到文字
            number_conversions = {
                '2': 'two', '3': 'three', '4': 'four', '5': 'five', '6': 'six',
                '7': 'seven', '8': 'eight', '9': 'nine', '10': 'ten', 
                '11': 'eleven', '12': 'twelve'
            }

            # 處理各種語法結構中的數字
            for digit, word in number_conversions.items():
                # 模式1: 數字 + 單一複數詞 (如 "7 chairs")
                pattern1 = rf'\b{digit}\s+([a-zA-Z]+s)\b'
                processed_response = re.sub(pattern1, rf'{word} \1', processed_response)
                
                # 模式2: 數字 + 修飾詞 + 複數詞 (如 "7 more chairs")
                pattern2 = rf'\b{digit}\s+(more|additional|other|identical)\s+([a-zA-Z]+s)\b'
                processed_response = re.sub(pattern2, rf'{word} \1 \2', processed_response, flags=re.IGNORECASE)
                
                # 模式3: 數字 + 形容詞 + 複數詞 (如 "2 dining tables")
                pattern3 = rf'\b{digit}\s+([a-zA-Z]+)\s+([a-zA-Z]+s)\b'
                processed_response = re.sub(pattern3, rf'{word} \1 \2', processed_response)
                
                # 模式4: 介詞片語中的數字 (如 "around 2 tables")
                pattern4 = rf'\b(around|approximately|about)\s+{digit}\s+([a-zA-Z]+s)\b'
                processed_response = re.sub(pattern4, rf'\1 {word} \2', processed_response, flags=re.IGNORECASE)

            return processed_response

        except Exception as e:
            self.logger.error(f"Error in _handle_repetitive_vocabulary: {str(e)}")
            self.logger.error(traceback.format_exc())
            return response # 發生錯誤時返回原始回應

    def _ensure_grammatical_completeness(self, response: str) -> str:
        """
        確保語法完整性，處理不完整句子和格式問題

        Args:
            response: 待檢查的回應文本

        Returns:
            str: 語法完整的回應文本
        """
        try:
            if not response or not response.strip():
                return response

            # 第一階段：檢查並修正不完整的句子模式
            incomplete_patterns = [
                # 介詞後直接結束的問題（針對 "over ." 等情況）
                (r'\b(over|under|through|across|along|beneath|beyond|throughout)\s*\.', 'incomplete_preposition'),
                (r'\b(with|without|against|towards|beside|between|among)\s*\.', 'incomplete_preposition'),
                (r'\b(into|onto|upon|within|behind|below|above)\s*\.', 'incomplete_preposition'),

                # 處理 "In ," 這類缺失詞彙的問題
                (r'\bIn\s*,', 'incomplete_location'),
                (r'\bAt\s*,', 'incomplete_location'),
                (r'\bOn\s*,', 'incomplete_location'),
                (r'\bWith\s*,', 'incomplete_context'),

                # 不完整的描述模式
                (r'\b(fine|the)\s+(the\s+)?(?:urban|area|scene)\b(?!\s+\w)', 'incomplete_description'),

                # 連詞或介詞後直接標點的問題
                (r'\b(and|or|but|with|from|in|at|on|by|for|to)\s*[.!?]', 'incomplete_conjunction'),

                # 重複詞彙
                (r'\b(\w+)\s+\1\b', 'word_repetition'),

                # 不完整的場景類型引用（如 "urban_intersection" 格式問題）
                (r'\b(\w+)_(\w+)\b', 'underscore_format'),

                # 地標場景特有問題
                (r'\btourist_landmark\b', 'underscore_format'),
                (r'\burban_intersection\b', 'underscore_format'),
                (r'\bIn\s*,\s*(?=\w)', 'incomplete_prepositional'),
                (r'\bOverall,\s+(?=exudes|shows|displays)(?!\s+(?:the|this|it))', 'missing_subject'),
                (r'\batmosphere of\s+is one of\b', 'redundant_structure'),
                (r'\bwith.*?turned\s+illuminating\b', 'redundant_participle')
            ]

            for pattern, issue_type in incomplete_patterns:
                try:
                    matches = list(re.finditer(pattern, response, re.IGNORECASE))

                    for match in matches:
                        if issue_type == 'incomplete_preposition':
                            # 處理介詞後直接結束的情況
                            response = self._fix_incomplete_preposition(response, match)

                        elif issue_type == 'underscore_format':
                            # 將下劃線格式轉換為空格分隔
                            original = match.group(0)
                            replacement = original.replace('_', ' ')
                            response = response.replace(original, replacement)

                        elif issue_type == 'word_repetition':
                            # 移除重複的詞彙
                            repeated_word = match.group(1)
                            response = response.replace(f"{repeated_word} {repeated_word}", repeated_word)

                        elif issue_type == 'incomplete_location' or issue_type == 'incomplete_context':
                            # 移除不完整的位置或上下文引用
                            response = response.replace(match.group(0), '')

                        elif issue_type == 'incomplete_prepositional':
                            # 處理不完整的介詞短語
                            response = re.sub(r'\bIn\s*,\s*', 'Throughout the scene, ', response)

                        elif issue_type == 'missing_subject':
                            # 為Overall句子添加主語
                            response = re.sub(r'\bOverall,\s+(?=exudes)', 'Overall, the scene ', response)

                        elif issue_type == 'redundant_structure':
                            # 簡化冗餘結構
                            response = re.sub(r'\batmosphere of\s+is one of\b', 'atmosphere is one of', response)

                        elif issue_type == 'redundant_participle':
                            # 清理冗餘分詞
                            response = re.sub(r'turned\s+illuminating', 'illuminating', response)

                        else:
                            # 對於其他不完整模式，直接移除
                            response = response.replace(match.group(0), '')

                    # 清理多餘空格
                    response = re.sub(r'\s{2,}', ' ', response).strip()

                except re.error as e:
                    self.logger.warning(f"Regular expression pattern error for {issue_type}: {pattern} - {str(e)}")
                    continue

            # 第二階段：處理物件類別格式問題
            response = self._clean_object_class_references(response)

            # 第三階段：確保句子正確結束
            response = self._ensure_proper_sentence_ending(response)

            # 第四階段：最終語法檢查
            response = self._final_grammar_check(response)

            return response.strip()

        except Exception as e:
            self.logger.error(f"Error in _ensure_grammatical_completeness: {str(e)}")
            return response

    def _fix_incomplete_preposition(self, response: str, match) -> str:
        """
        修正不完整的介詞短語

        Args:
            response: 回應文本
            match: 正則匹配對象

        Returns:
            str: 修正後的回應
        """
        preposition = match.group(1)
        match_start = match.start()

        # 找到句子的開始位置
        sentence_start = response.rfind('.', 0, match_start)
        sentence_start = sentence_start + 1 if sentence_start != -1 else 0

        # 提取句子片段
        sentence_fragment = response[sentence_start:match_start].strip()

        # 如果句子片段有意義，嘗試移除不完整的介詞部分
        if len(sentence_fragment) > 10:
            # 移除介詞及其後的內容，添加適當的句號
            response = response[:match_start].rstrip() + '.'
        else:
            # 如果句子片段太短，移除整個不完整的句子
            response = response[:sentence_start] + response[match.end():]

        return response

    def _clean_object_class_references(self, response: str) -> str:
        """
        清理物件類別引用中的格式問題

        Args:
            response: 回應文本

        Returns:
            str: 清理後的回應
        """
        # 移除類別ID引用（如 "unknown-class 2", "Class 0" 等）
        class_id_patterns = [
            r'\bunknown[- ]?class\s*\d+\s*objects?',
            r'\bclass[- ]?\d+\s*objects?',
            r'\b[Cc]lass\s*\d+\s*objects?',
            r'\bunknown[- ][Cc]lass\s*\d+\s*objects?'
        ]

        for pattern in class_id_patterns:
            try:
                # 替換為更自然的描述
                response = re.sub(pattern, 'objects', response, flags=re.IGNORECASE)
            except re.error as e:
                self.logger.warning(f"Error cleaning class reference pattern {pattern}: {str(e)}")
                continue

        # 處理數量描述中的問題
        response = re.sub(r'\b(\w+)\s+unknown[- ]?\w*\s*objects?', r'\1 objects', response, flags=re.IGNORECASE)

        return response

    def _ensure_proper_sentence_ending(self, response: str) -> str:
        """
        確保句子有適當的結尾

        Args:
            response: 回應文本

        Returns:
            str: 具有適當結尾的回應
        """
        if not response or not response.strip():
            return response

        response = response.strip()

        # 檢查是否以標點符號結尾
        if response and response[-1] not in ['.', '!', '?']:

            # 常見介詞和連詞列表
            problematic_endings = [
                "into", "onto", "about", "above", "across", "after", "along", "around",
                "at", "before", "behind", "below", "beneath", "beside", "between",
                "beyond", "by", "down", "during", "except", "for", "from", "in",
                "inside", "near", "of", "off", "on", "over", "through", "to",
                "toward", "under", "up", "upon", "with", "within", "and", "or", "but"
            ]

            words = response.split()
            if words:
                last_word = words[-1].lower().rstrip('.,!?')

                if last_word in problematic_endings:
                    # 找到最後完整的句子
                    last_period_pos = max(
                        response.rfind('.'),
                        response.rfind('!'),
                        response.rfind('?')
                    )

                    if last_period_pos > len(response) // 2:  # 如果有較近的完整句子
                        response = response[:last_period_pos + 1]
                    else:
                        # 移除問題詞彙並添加句號
                        if len(words) > 1:
                            response = " ".join(words[:-1]) + "."
                        else:
                            response = "The scene displays various elements."
                else:
                    # 正常情況下添加句號
                    response += "."

        return response

    def _final_grammar_check(self, response: str) -> str:
        """
        最終語法檢查和清理

        Args:
            response: 回應文本

        Returns:
            str: 最終清理後的回應
        """
        if not response:
            return response

        # 修正連續標點符號
        response = re.sub(r'([.!?]){2,}', r'\1', response)

        # 修正句號前的空格
        response = re.sub(r'\s+([.!?])', r'\1', response)

        # 修正句號後缺少空格的問題
        response = re.sub(r'([.!?])([A-Z])', r'\1 \2', response)

        # 確保首字母大寫
        if response and response[0].islower():
            response = response[0].upper() + response[1:]

        # 移除多餘的空格
        response = re.sub(r'\s{2,}', ' ', response)

        # 處理空句子或過短的回應
        if len(response.strip()) < 20:
            return "The scene contains various visual elements."

        return response.strip()

    def _control_word_length(self, response: str) -> str:
        """控制文字長度在合理範圍內"""
        words = response.split()
        if len(words) > 200:
            # 找到接近字數限制的句子結束處
            truncated = ' '.join(words[:200])
            last_period = max(truncated.rfind('.'), truncated.rfind('!'), truncated.rfind('?'))

            if last_period > 0:
                response = truncated[:last_period+1]
            else:
                response = truncated + "."

        return response

    def _final_formatting(self, response: str) -> str:
        """最終格式化處理"""
        # 確保首字母大寫
        if response and response[0].islower():
            response = response[0].upper() + response[1:]

        # 統一格式為單一段落
        response = re.sub(r'\s*\n\s*', ' ', response)
        response = ' '.join(response.split())

        return response.strip()

    def _recover_from_overcleaning(self, original_response: str) -> str:
        """從過度清理中恢復內容"""
        try:
            # 嘗試從原始回應中找到最佳段落
            paragraphs = [p for p in original_response.split('\n\n') if p.strip()]
            if paragraphs:
                # 選擇最長的段落作為主要描述
                best_para = max(paragraphs, key=len)
                # 使用基本清理規則
                best_para = re.sub(r'\[.*?\]', '', best_para)
                best_para = re.sub(r'\s{2,}', ' ', best_para).strip()

                if len(best_para) >= 40:
                    return best_para

            return "Unable to generate a valid enhanced description."

        except Exception as e:
            self.logger.error(f"Recovery from overcleaning failed: {str(e)}")
            return "Description generation error."

    def _validate_cleaned_response(self, response: str):
        """驗證清理後的回應"""
        if not response:
            raise ResponseProcessingError("Response is empty after cleaning")

        if len(response.strip()) < 20:
            raise ResponseProcessingError("Response is too short after cleaning")

        # 檢查是否包含基本的句子結構
        if not re.search(r'[.!?]', response):
            raise ResponseProcessingError("Response lacks proper sentence structure")

    def remove_explanatory_notes(self, response: str) -> str:
        """
        移除解釋性注釋和說明

        Args:
            response: 包含可能注釋的回應

        Returns:
            str: 移除注釋後的回應
        """
        try:
            # 識別常見的注釋和解釋模式
            note_patterns = [
                r'(?:^|\n)Note:.*?(?:\n|$)',
                r'(?:^|\n)I have (?:followed|adhered to|ensured).*?(?:\n|$)',
                r'(?:^|\n)This description (?:follows|adheres to|maintains).*?(?:\n|$)',
                r'(?:^|\n)The enhanced description (?:maintains|preserves).*?(?:\n|$)'
            ]

            # 尋找段落
            paragraphs = [p.strip() for p in response.split('\n\n') if p.strip()]

            # 如果只有一個段落，檢查並清理它
            if len(paragraphs) == 1:
                for pattern in note_patterns:
                    paragraphs[0] = re.sub(pattern, '', paragraphs[0], flags=re.IGNORECASE)
                return paragraphs[0].strip()

            # 如果有多個段落，移除注釋段落
            content_paragraphs = []
            for paragraph in paragraphs:
                is_note = False
                for pattern in note_patterns:
                    if re.search(pattern, paragraph, flags=re.IGNORECASE):
                        is_note = True
                        break

                # 檢查段落是否以常見的注釋詞開頭
                if paragraph.lower().startswith(('note:', 'please note:', 'remember:')):
                    is_note = True

                if not is_note:
                    content_paragraphs.append(paragraph)

            return '\n\n'.join(content_paragraphs).strip()

        except Exception as e:
            self.logger.error(f"Failed to remove explanatory notes: {str(e)}")
            return response

    def get_processor_info(self) -> Dict[str, Any]:
        """
        獲取處理器信息

        Returns:
            Dict[str, Any]: 包含處理器狀態和配置的信息
        """
        return {
            "replacement_alternatives_count": len(self.replacement_alternatives),
            "prefixes_to_remove_count": len(self.prefixes_to_remove),
            "suffixes_to_remove_count": len(self.suffixes_to_remove),
            "repetitive_patterns_count": len(self.repetitive_patterns),
            "initialization_status": "success"
        }
