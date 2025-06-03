import logging
import traceback
import re
from typing import Dict, List, Optional

from landmark_data import ALL_LANDMARKS

class TextFormattingError(Exception):
    """文本格式化過程中的自定義異常"""
    pass


class TextFormatter:
    """
    文本格式化器 - 負責文本拼接、格式化和最終輸出優化

    該類別處理所有與文本格式化相關的邏輯，包括智能文本拼接、
    標點符號處理、大小寫規範化以及地標引用的過濾功能。
    """

    def __init__(self):
        """
        初始化文本格式化器
        """
        self.logger = logging.getLogger(self.__class__.__name__)

        try:
            # 載入地標數據用於引用過濾
            self.landmark_data = self._load_landmark_data()

            self.logger.info("TextFormatter initialized successfully")

        except Exception as e:
            error_msg = f"Failed to initialize TextFormatter: {str(e)}"
            self.logger.error(f"{error_msg}\n{traceback.format_exc()}")
            raise TextFormattingError(error_msg) from e

    def _load_landmark_data(self) -> Dict:
        """
        載入地標數據

        Returns:
            Dict: 地標數據字典
        """
        try:
            return ALL_LANDMARKS
        except ImportError:
            self.logger.warning("Failed to import landmark data, landmark filtering will be disabled")
            return {}
        except Exception as e:
            self.logger.warning(f"Error loading landmark data: {str(e)}")
            return {}

    def smart_append(self, current_text: str, new_fragment: str) -> str:
        """
        將新文本片段附加到現有文本，處理標點符號和大小寫

        Args:
            current_text: 要加到的現有文本
            new_fragment: 要加的新文本片段

        Returns:
            str: 合併後的文本，具有適當的格式化
        """
        try:
            # 處理空值情況
            if not new_fragment:
                return current_text

            if not current_text:
                # 確保第一個字符大寫
                return new_fragment[0].upper() + new_fragment[1:] if new_fragment else ""

            # 清理現有文本
            current_text = current_text.rstrip()

            # 檢查結尾標點符號
            ends_with_sentence = current_text.endswith(('.', '!', '?'))
            ends_with_comma = current_text.endswith(',')

            # 特別處理 "A xxx A yyy" 模式
            if (current_text.startswith("A ") or current_text.startswith("An ")) and \
               (new_fragment.startswith("A ") or new_fragment.startswith("An ")):
                return current_text + ". " + new_fragment

            # 檢查新片段是否包含地標名稱（通常為專有名詞）
            has_landmark_name = any(word[0].isupper() for word in new_fragment.split()
                                  if len(word) > 2 and not word.startswith(("A ", "An ", "The ")))

            # 決定如何連接文本
            if ends_with_sentence:
                # 句子後，以大寫開始並添加適當間距
                joined_text = current_text + " " + (new_fragment[0].upper() + new_fragment[1:])
            elif ends_with_comma:
                # 逗號後，要保持流暢性，除非是專有名詞或特殊情況
                if new_fragment.startswith(('I ', 'I\'', 'A ', 'An ', 'The ')) or new_fragment[0].isupper() or has_landmark_name:
                    joined_text = current_text + " " + new_fragment
                else:
                    joined_text = current_text + " " + new_fragment[0].lower() + new_fragment[1:]
            elif "scene is" in new_fragment.lower() or "scene includes" in new_fragment.lower():
                # 加關於場景的新句子時，使用句號
                joined_text = current_text + ". " + new_fragment
            else:
                # 其他情況，根據內容決定
                if self._is_related_phrases(current_text, new_fragment):
                    if new_fragment.startswith(('I ', 'I\'', 'A ', 'An ', 'The ')) or new_fragment[0].isupper() or has_landmark_name:
                        joined_text = current_text + ", " + new_fragment
                    else:
                        joined_text = current_text + ", " + new_fragment[0].lower() + new_fragment[1:]
                else:
                    # 對不相關的短語使用句號
                    joined_text = current_text + ". " + (new_fragment[0].upper() + new_fragment[1:])

            return joined_text

        except Exception as e:
            self.logger.warning(f"Error in smart_append: {str(e)}")
            # 備用簡單拼接
            return f"{current_text} {new_fragment}" if current_text else new_fragment

    def _is_related_phrases(self, text1: str, text2: str) -> bool:
        """
        判斷兩個短語是否相關，應該用逗號

        Args:
            text1: 第一個文本片段
            text2: 要加的第二個文本片段

        Returns:
            bool: 短語是否相關
        """
        try:
            # 檢查兩個短語是否都以 "A" 或 "An" 開始 - 這些是獨立的描述
            if (text1.startswith("A ") or text1.startswith("An ")) and \
               (text2.startswith("A ") or text2.startswith("An ")):
                return False  # 這些是獨立的描述，不是相關短語

            # 檢查第二個短語是否以連接詞開始
            connecting_words = ["which", "where", "who", "whom", "whose", "with", "without",
                              "this", "these", "that", "those", "and", "or", "but"]

            first_word = text2.split()[0].lower() if text2 else ""
            if first_word in connecting_words:
                return True

            # 檢查第一個短語是否以暗示連續性的內容結尾
            ending_patterns = ["such as", "including", "like", "especially", "particularly",
                             "for example", "for instance", "namely", "specifically"]

            for pattern in ending_patterns:
                if text1.lower().endswith(pattern):
                    return True

            # 檢查兩個短語是否都關於場景
            if "scene" in text1.lower() and "scene" in text2.lower():
                return False  # 關於場景的獨立陳述應該是分開的句子

            return False

        except Exception as e:
            self.logger.warning(f"Error checking phrase relationship: {str(e)}")
            return False

    def format_final_description(self, text: str) -> str:
        """
        格式化最終描述文本，確保正確的標點符號、大小寫和間距

        Args:
            text: 要格式化的文本

        Returns:
            str: 格式化後的文本
        """
        try:
            if not text or not text.strip():
                return ""

            # 首先修剪前導/尾隨空白
            text = text.strip()

            # 1. 處理連續的 "A/An" 段落（可能將它們分成句子）
            text = re.sub(r'(A\s+[^.!?]+?[\w\.])\s+(A\s+)', r'\1. \2', text, flags=re.IGNORECASE)
            text = re.sub(r'(An\s+[^.!?]+?[\w\.])\s+(An?\s+)', r'\1. \2', text, flags=re.IGNORECASE)

            # 2. 確保整個文本的第一個字符大寫
            if text:
                text = text[0].upper() + text[1:]

            # 3. 規範化空白：多個空格變為一個
            text = re.sub(r'\s{2,}', ' ', text)

            # 4. 句子結尾標點符號後大寫
            def capitalize_after_punctuation(match):
                return match.group(1) + match.group(2).upper()
            text = re.sub(r'([.!?]\s+)([a-z])', capitalize_after_punctuation, text)

            # 5. 處理逗號後的大小寫
            def fix_capitalization_after_comma(match):
                leading_comma_space = match.group(1)  # (,\s+)
                word_after_comma = match.group(2)     # ([A-Z][a-zA-Z]*)

                proper_nouns_exceptions = ["I", "I'm", "I've", "I'd", "I'll",
                                         "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
                                         "January", "February", "March", "April", "May", "June", "July",
                                         "August", "September", "October", "November", "December"]

                if word_after_comma in proper_nouns_exceptions:
                    return match.group(0)

                # 如果詞看起來像專有名詞（已經大寫且不是常用詞），保持不變
                if len(word_after_comma) > 2 and word_after_comma[0].isupper() and word_after_comma.lower() not in ["this", "that", "these", "those", "they", "their", "then", "thus"]:
                    return match.group(0)  # 如果看起來已經是專有名詞則保持不變

                return leading_comma_space + word_after_comma[0].lower() + word_after_comma[1:]
            text = re.sub(r'(,\s+)([A-Z][a-zA-Z\'\-]+)', fix_capitalization_after_comma, text)

            # 6. 修正標點符號周圍的間距
            text = re.sub(r'\s*([.,;:!?])\s*', r'\1 ', text)  # 確保標點符號後有一個空格，前面沒有
            text = text.replace(' .', '.').replace(' ,', ',')  # 清理標點符號前可能的空格

            # 7. 合併多個句子結尾標點符號
            text = re.sub(r'[.!?]{2,}', '.', text)  # 將多個轉換為單個句號
            text = re.sub(r',+', ',', text)  # 多個逗號變為一個

            # 8. 確保文本以單個句子結尾標點符號結尾
            text = text.strip()  # 檢查最後一個字符前移除尾隨空白
            if text and not text[-1] in '.!?':
                text += '.'

            # 9. 處理空的佔位符和前導標點符號
            text = re.sub(r'\bIn\s*,\s*', 'In this scene, ', text)  # 修復 "In , " 問題
            text = re.sub(r'\s*,\s*([A-Z])', r'. \1', text)  # 修復逗號後直接跟大寫字母的問題
            text = re.sub(r'^[.,;:!?\s]+', '', text)  # 移除前導標點符號

            # 10. 第一個字母大寫的最終檢查
            if text:
                text = text[0].upper() + text[1:]

            # 11. 移除最終標點符號前的空格（如果規則7意外添加）
            text = re.sub(r'\s+([.!?])$', r'\1', text)

            return text.strip()  # 最終修剪

        except Exception as e:
            self.logger.warning(f"Error formatting final description: {str(e)}")
            # 備用基本格式化
            if text:
                text = text.strip()
                if text and not text.endswith(('.', '!', '?')):
                    text += '.'
                if text:
                    text = text[0].upper() + text[1:]
                return text
            return ""

    def filter_landmark_references(self, text: str, enable_landmark: bool = True) -> str:
        """
        動態過濾文本中的地標引用

        Args:
            text: 需要過濾的文本
            enable_landmark: 是否啟用地標功能

        Returns:
            str: 過濾後的文本
        """
        try:
            if enable_landmark or not text:
                return text

            # 動態收集所有地標名稱和位置
            landmark_names = []
            locations = []

            for landmark_id, info in self.landmark_data.items():
                # 收集地標名稱及其別名
                landmark_names.append(info["name"])
                landmark_names.extend(info.get("aliases", []))

                # 收集地理位置
                if "location" in info:
                    location = info["location"]
                    locations.append(location)

                    # 處理分離的城市和國家名稱
                    parts = location.split(",")
                    if len(parts) >= 1:
                        locations.append(parts[0].strip())
                    if len(parts) >= 2:
                        locations.append(parts[1].strip())

            # 替換所有地標名稱
            for name in landmark_names:
                if name and len(name) > 2:  # 避免過短的名稱
                    text = re.sub(r'\b' + re.escape(name) + r'\b', "tall structure", text, flags=re.IGNORECASE)

            # 動態替換所有位置引用
            for location in locations:
                if location and len(location) > 2:
                    # 替換常見位置表述模式
                    text = re.sub(r'in ' + re.escape(location), "in the urban area", text, flags=re.IGNORECASE)
                    text = re.sub(r'of ' + re.escape(location), "of the urban area", text, flags=re.IGNORECASE)
                    text = re.sub(r'\b' + re.escape(location) + r'\b', "the urban area", text, flags=re.IGNORECASE)

            # 通用地標描述模式替換
            landmark_patterns = [
                (r'a (tourist|popular|famous) landmark', r'an urban structure'),
                (r'an iconic structure in ([A-Z][a-zA-Z\s,]+)', r'an urban structure in the area'),
                (r'a famous (monument|tower|landmark) in ([A-Z][a-zA-Z\s,]+)', r'an urban structure in the area'),
                (r'(centered|built|located|positioned) around the ([A-Z][a-zA-Z\s]+? (Tower|Monument|Landmark))', r'located in this area'),
                (r'(sightseeing|guided tours|cultural tourism) (at|around|near) (this landmark|the [A-Z][a-zA-Z\s]+)', r'\1 in this area'),
                (r'this (famous|iconic|historic|well-known) (landmark|monument|tower|structure)', r'this urban structure'),
                (r'([A-Z][a-zA-Z\s]+) Tower', r'tall structure'),
                (r'a (tower|structure) in ([A-Z][a-zA-Z\s,]+)', r'a \1 in the area'),
                (r'landmark scene', r'urban scene'),
                (r'tourist destination', r'urban area'),
                (r'tourist attraction', r'urban area')
            ]

            for pattern, replacement in landmark_patterns:
                text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

            return text

        except Exception as e:
            self.logger.warning(f"Error filtering landmark references: {str(e)}")
            return text

    def optimize_text_flow(self, text: str) -> str:
        """
        優化文本流暢性，減少重複和改善可讀性

        Args:
            text: 要優化的文本

        Returns:
            str: 優化後的文本
        """
        try:
            if not text:
                return text

            # 移除重複的短語
            text = self._remove_duplicate_phrases(text)

            # 優化連接詞使用
            text = self._optimize_connectors(text)

            # 平衡句子長度
            text = self._balance_sentence_length(text)

            return text

        except Exception as e:
            self.logger.warning(f"Error optimizing text flow: {str(e)}")
            return text

    def _remove_duplicate_phrases(self, text: str) -> str:
        """
        移除文本中的重複短語

        Args:
            text: 輸入文本

        Returns:
            str: 移除重複後的文本
        """
        try:
            # 分割成句子
            sentences = re.split(r'[.!?]+', text)
            unique_sentences = []
            seen_content = set()

            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue

                # 規範化以進行比較（移除額外空白和標點符號）
                normalized = re.sub(r'\s+', ' ', sentence.lower().strip())

                # 檢查是否實質相似
                is_duplicate = False
                for seen in seen_content:
                    if self._sentences_similar(normalized, seen):
                        is_duplicate = True
                        break

                if not is_duplicate:
                    unique_sentences.append(sentence)
                    seen_content.add(normalized)

            return '. '.join(unique_sentences) + '.' if unique_sentences else ""

        except Exception as e:
            self.logger.warning(f"Error removing duplicate phrases: {str(e)}")
            return text

    def _sentences_similar(self, sent1: str, sent2: str) -> bool:
        """
        檢查兩個句子是否相似

        Args:
            sent1: 第一個句子
            sent2: 第二個句子

        Returns:
            bool: 句子是否相似
        """
        try:
            # 簡單的相似性檢查：如果80%的詞彙重疊
            words1 = set(sent1.split())
            words2 = set(sent2.split())

            if not words1 or not words2:
                return False

            intersection = len(words1 & words2)
            union = len(words1 | words2)

            similarity = intersection / union if union > 0 else 0
            return similarity > 0.8

        except Exception as e:
            self.logger.warning(f"Error checking sentence similarity: {str(e)}")
            return False

    def _optimize_connectors(self, text: str) -> str:
        """
        優化連接詞的使用

        Args:
            text: 輸入文本

        Returns:
            str: 優化連接詞後的文本
        """
        try:
            # 替換重複的連接詞
            text = re.sub(r'\band\s+and\b', 'and', text, flags=re.IGNORECASE)
            text = re.sub(r'\bwith\s+with\b', 'with', text, flags=re.IGNORECASE)

            # 改善過度使用 "and" 的情況
            text = re.sub(r'(\w+),\s+and\s+(\w+),\s+and\s+(\w+)', r'\1, \2, and \3', text)

            return text

        except Exception as e:
            self.logger.warning(f"Error optimizing connectors: {str(e)}")
            return text

    def _balance_sentence_length(self, text: str) -> str:
        """
        平衡句子長度，分割過長的句子

        Args:
            text: 輸入文本

        Returns:
            str: 平衡句子長度後的文本
        """
        try:
            sentences = re.split(r'([.!?]+)', text)
            balanced_text = ""

            for i in range(0, len(sentences), 2):
                if i + 1 < len(sentences):
                    sentence = sentences[i]
                    punctuation = sentences[i + 1]

                    # 如果句子太長（超過150個字符），嘗試在適當位置分割
                    if len(sentence) > 150:
                        # 在逗號或連接詞處分割
                        split_points = [m.start() for m in re.finditer(r',\s+(?:and|but|or|while|when|where)', sentence)]
                        if split_points:
                            mid_point = split_points[len(split_points) // 2]
                            first_part = sentence[:mid_point].strip()
                            second_part = sentence[mid_point + 1:].strip()
                            if second_part and not second_part[0].isupper():
                                second_part = second_part[0].upper() + second_part[1:]
                            balanced_text += first_part + ". " + second_part + punctuation + " "
                        else:
                            balanced_text += sentence + punctuation + " "
                    else:
                        balanced_text += sentence + punctuation + " "

            return balanced_text.strip()

        except Exception as e:
            self.logger.warning(f"Error balancing sentence length: {str(e)}")
            return text

    def validate_text_quality(self, text: str) -> Dict[str, bool]:
        """
        驗證文本質量

        Args:
            text: 要驗證的文本

        Returns:
            Dict[str, bool]: 質量檢查結果
        """
        try:
            quality_checks = {
                "has_content": bool(text and text.strip()),
                "proper_capitalization": bool(text and text[0].isupper()) if text else False,
                "ends_with_punctuation": bool(text and text.strip()[-1] in '.!?') if text else False,
                "no_double_spaces": "  " not in text if text else True,
                "no_leading_punctuation": not bool(re.match(r'^[.,;:!?]', text.strip())) if text else True,
                "reasonable_length": 20 <= len(text) <= 1000 if text else False
            }

            return quality_checks

        except Exception as e:
            self.logger.warning(f"Error validating text quality: {str(e)}")
            return {"error": True}

    def get_text_statistics(self, text: str) -> Dict[str, int]:
        """
        獲取文本統計信息

        Args:
            text: 要分析的文本

        Returns:
            Dict[str, int]: 文本統計信息
        """
        try:
            if not text:
                return {"characters": 0, "words": 0, "sentences": 0}

            characters = len(text)
            words = len(text.split())
            sentences = len(re.findall(r'[.!?]+', text))

            return {
                "characters": characters,
                "words": words,
                "sentences": sentences
            }

        except Exception as e:
            self.logger.warning(f"Error getting text statistics: {str(e)}")
            return {"characters": 0, "words": 0, "sentences": 0}
