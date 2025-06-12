import re
import logging
from typing import Dict, List, Optional, Any, Tuple

class TextOptimizer:
    """
    文本優化器 - 專門處理文本格式化、清理和優化
    負責物件列表格式化、重複移除、複數形式處理以及描述文本的優化
    """

    def __init__(self):
        """初始化文本優化器"""
        self.logger = logging.getLogger(self.__class__.__name__)

    def format_object_list_for_description(self,
                                          objects: List[Dict],
                                          use_indefinite_article_for_one: bool = False,
                                          count_threshold_for_generalization: int = -1,
                                          max_types_to_list: int = 5) -> str:
        """
        將物件列表格式化為人類可讀的字符串，包含總計數字

        Args:
            objects: 物件字典列表，每個應包含 'class_name'
            use_indefinite_article_for_one: 單個物件是否使用 "a/an"，否則使用 "one"
            count_threshold_for_generalization: 超過此計數時使用通用術語，-1表示精確計數
            max_types_to_list: 列表中包含的不同物件類型最大數量

        Returns:
            str: 格式化的物件描述字符串
        """
        try:
            if not objects:
                return "no specific objects clearly identified"

            counts: Dict[str, int] = {}
            for obj in objects:
                name = obj.get("class_name", "unknown object")
                if name == "unknown object" or not name:
                    continue
                counts[name] = counts.get(name, 0) + 1

            if not counts:
                return "no specific objects clearly identified"

            descriptions = []
            # 按計數降序然後按名稱升序排序，限制物件類型數量
            sorted_counts = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:max_types_to_list]

            for name, count in sorted_counts:
                if count == 1:
                    if use_indefinite_article_for_one:
                        if name[0].lower() in 'aeiou':
                            descriptions.append(f"an {name}")
                        else:
                            descriptions.append(f"a {name}")
                    else:
                        descriptions.append(f"one {name}")
                else:
                    # 處理複數形式
                    plural_name = self._get_plural_form(name)

                    if count_threshold_for_generalization != -1 and count > count_threshold_for_generalization:
                        if count <= count_threshold_for_generalization + 3:
                            descriptions.append(f"several {plural_name}")
                        else:
                            descriptions.append(f"many {plural_name}")
                    else:
                        descriptions.append(f"{count} {plural_name}")

            if not descriptions:
                return "no specific objects clearly identified"

            if len(descriptions) == 1:
                return descriptions[0]
            elif len(descriptions) == 2:
                return f"{descriptions[0]} and {descriptions[1]}"
            else:
                # 使用牛津逗號格式
                return ", ".join(descriptions[:-1]) + f", and {descriptions[-1]}"

        except Exception as e:
            self.logger.warning(f"Error formatting object list: {str(e)}")
            return "various objects"

    def optimize_object_description(self, description: str) -> str:
        """
        優化物件描述文本，消除多餘重複並改善表達流暢度

        這個函數是後處理階段的關鍵組件，負責清理和精簡自然語言生成系統
        產出的描述文字。它專門處理常見的重複問題，如相同物件的重複
        列舉和冗餘的空間描述，讓最終的描述更簡潔自然。

        Args:
            description: 原始的場景描述文本，可能包含重複或冗餘的表達

        Returns:
            str: 經過優化清理的描述文本，如果處理失敗則返回原始文本
        """
        try:
            # 1. 處理多餘的空間限定表達
            # 使用通用模式來識別和移除不必要的空間描述
            description = self._remove_redundant_spatial_qualifiers(description)

            # 2. 辨識並處理物件列表的重複問題
            # 尋找形如 "with X, Y, Z" 或 "with X and Y" 的物件列表
            object_lists = re.findall(r'with ([^.]+?)(?=\.|$)', description)

            # 遍歷每個找到的物件列表進行重複檢測和優化
            for obj_list in object_lists:
                # 3. 解析單個物件列表中的項目
                all_items = self._parse_object_list_items(obj_list)

                # 4. 統計物件出現頻率
                item_counts = self._count_object_items(all_items)

                # 5. 生成優化後的物件列表
                if item_counts:
                    new_items = self._generate_optimized_item_list(item_counts)
                    new_list = self._format_item_list(new_items)
                    description = description.replace(obj_list, new_list)

            return description

        except Exception as e:
            self.logger.warning(f"Error optimizing object description: {str(e)}")
            return description

    def remove_repetitive_descriptors(self, description: str) -> str:
        """
        移除描述中的重複性和不適當的描述詞彙，特別是 "identical" 等詞彙

        Args:
            description: 原始描述文本

        Returns:
            str: 清理後的描述文本
        """
        try:
            # 定義需要移除或替換的模式
            cleanup_patterns = [
                # 移除 "identical" 描述模式
                (r'\b(\d+)\s+identical\s+([a-zA-Z\s]+)', r'\1 \2'),
                (r'\b(two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+identical\s+([a-zA-Z\s]+)', r'\1 \2'),
                (r'\bidentical\s+([a-zA-Z\s]+)', r'\1'),

                # 改善 "comprehensive arrangement" 等過於技術性的表達
                (r'\bcomprehensive arrangement of\b', 'arrangement of'),
                (r'\bcomprehensive view featuring\b', 'scene featuring'),
                (r'\bcomprehensive display of\b', 'display of'),

                # 簡化過度描述性的短語
                (r'\bpositioning around\s+(\d+)\s+identical\b', r'positioning around \1'),
                (r'\barranged around\s+(\d+)\s+identical\b', r'arranged around \1'),
            ]

            processed_description = description
            for pattern, replacement in cleanup_patterns:
                processed_description = re.sub(pattern, replacement, processed_description, flags=re.IGNORECASE)

            # 進一步清理可能的多餘空格
            processed_description = re.sub(r'\s+', ' ', processed_description).strip()

            self.logger.debug(f"Cleaned description: removed repetitive descriptors")
            return processed_description

        except Exception as e:
            self.logger.warning(f"Error removing repetitive descriptors: {str(e)}")
            return description

    def format_object_count_description(self, class_name: str, count: int,
                                      scene_type: Optional[str] = None,
                                      detected_objects: Optional[List[Dict]] = None,
                                      avg_confidence: float = 0.0) -> str:
        """
        格式化物件數量描述的核心方法，整合空間排列、材質推斷和場景語境

        Args:
            class_name: 標準化後的類別名稱
            count: 物件數量
            scene_type: 場景類型，用於語境化描述
            detected_objects: 該類型的所有檢測物件，用於空間分析
            avg_confidence: 平均檢測置信度，影響材質推斷的可信度

        Returns:
            str: 完整的格式化數量描述
        """
        try:
            if count <= 0:
                return ""

            # 獲取基礎的複數形式
            plural_form = self._get_plural_form(class_name)

            # 單數情況的處理
            if count == 1:
                return self._format_single_object_description(class_name, scene_type,
                                                            detected_objects, avg_confidence)

            # 複數情況的處理
            return self._format_multiple_objects_description(class_name, count, plural_form,
                                                           scene_type, detected_objects, avg_confidence)

        except Exception as e:
            self.logger.warning(f"Error formatting object count for '{class_name}': {str(e)}")
            return f"{count} {class_name}s" if count > 1 else class_name

    def normalize_object_class_name(self, class_name: str) -> str:
        """
        標準化物件類別名稱，確保輸出自然語言格式

        Args:
            class_name: 原始類別名稱

        Returns:
            str: 標準化後的類別名稱
        """
        try:
            if not class_name or not isinstance(class_name, str):
                return "object"

            # 移除可能的技術性前綴或後綴
            normalized = re.sub(r'^(class_|id_|type_)', '', class_name.lower())
            normalized = re.sub(r'(_class|_id|_type)$', '', normalized)

            # 將下劃線和連字符替換為空格
            normalized = normalized.replace('_', ' ').replace('-', ' ')

            # 移除多餘空格
            normalized = ' '.join(normalized.split())

            # 特殊類別名稱的標準化映射
            class_name_mapping = {
                'traffic light': 'traffic light',
                'stop sign': 'stop sign',
                'fire hydrant': 'fire hydrant',
                'dining table': 'dining table',
                'potted plant': 'potted plant',
                'tv monitor': 'television',
                'cell phone': 'mobile phone',
                'wine glass': 'wine glass',
                'hot dog': 'hot dog',
                'teddy bear': 'teddy bear',
                'hair drier': 'hair dryer',
                'toothbrush': 'toothbrush'
            }

            return class_name_mapping.get(normalized, normalized)

        except Exception as e:
            self.logger.warning(f"Error normalizing class name '{class_name}': {str(e)}")
            return class_name if isinstance(class_name, str) else "object"

    def _remove_redundant_spatial_qualifiers(self, description: str) -> str:
        """
        移除描述中冗餘的空間限定詞

        Args:
            description: 包含可能多餘空間描述的文本

        Returns:
            str: 移除多餘空間限定詞後的文本
        """
        # 定義常見的多餘空間表達模式
        redundant_patterns = [
            # 室內物件的多餘房間描述
            (r'\b(bed|sofa|couch|chair|table|desk|dresser|nightstand)\s+in\s+the\s+(room|bedroom|living\s+room)', r'\1'),
            # 廚房物件的多餘描述
            (r'\b(refrigerator|stove|oven|sink|microwave)\s+in\s+the\s+kitchen', r'\1'),
            # 浴室物件的多餘描述
            (r'\b(toilet|shower|bathtub|sink)\s+in\s+the\s+(bathroom|restroom)', r'\1'),
            # 一般性的多餘表達：「在場景中」、「在圖片中」等
            (r'\b([\w\s]+)\s+in\s+the\s+(scene|image|picture|frame)', r'\1'),
        ]

        for pattern, replacement in redundant_patterns:
            description = re.sub(pattern, replacement, description, flags=re.IGNORECASE)

        return description

    def _parse_object_list_items(self, obj_list: str) -> List[str]:
        """
        解析物件列表中的項目

        Args:
            obj_list: 物件列表字符串

        Returns:
            List[str]: 解析後的項目列表
        """
        # 先處理逗號格式 "A, B, and C"
        if ", and " in obj_list:
            before_last_and = obj_list.rsplit(", and ", 1)[0]
            last_item = obj_list.rsplit(", and ", 1)[1]
            front_items = [item.strip() for item in before_last_and.split(",")]
            all_items = front_items + [last_item.strip()]
        elif " and " in obj_list:
            all_items = [item.strip() for item in obj_list.split(" and ")]
        else:
            all_items = [item.strip() for item in obj_list.split(",")]

        return all_items

    def _count_object_items(self, all_items: List[str]) -> Dict[str, int]:
        """
        統計物件項目的出現次數

        Args:
            all_items: 所有項目列表

        Returns:
            Dict[str, int]: 項目計數字典
        """
        item_counts = {}

        for item in all_items:
            item = item.strip()
            if item and item not in ["and", "with", ""]:
                clean_item = self._normalize_item_for_counting(item)
                if clean_item not in item_counts:
                    item_counts[clean_item] = 0
                item_counts[clean_item] += 1

        return item_counts

    def _generate_optimized_item_list(self, item_counts: Dict[str, int]) -> List[str]:
        """
        生成優化後的項目列表

        Args:
            item_counts: 項目計數字典

        Returns:
            List[str]: 優化後的項目列表
        """
        new_items = []

        for item, count in item_counts.items():
            if count > 1:
                plural_item = self._make_plural(item)
                new_items.append(f"{count} {plural_item}")
            else:
                new_items.append(item)

        return new_items

    def _format_item_list(self, new_items: List[str]) -> str:
        """
        格式化項目列表為字符串

        Args:
            new_items: 新項目列表

        Returns:
            str: 格式化後的字符串
        """
        if len(new_items) == 1:
            return new_items[0]
        elif len(new_items) == 2:
            return f"{new_items[0]} and {new_items[1]}"
        else:
            return ", ".join(new_items[:-1]) + f", and {new_items[-1]}"

    def _normalize_item_for_counting(self, item: str) -> str:
        """
        正規化物件項目以便準確計數

        Args:
            item: 原始物件項目字串

        Returns:
            str: 正規化後的物件項目
        """
        item = re.sub(r'^(a|an|the)\s+', '', item.lower())
        return item.strip()

    def _make_plural(self, item: str) -> str:
        """
        將單數名詞轉換為複數形式

        Args:
            item: 單數形式的名詞

        Returns:
            str: 複數形式的名詞
        """
        if item.endswith("y") and len(item) > 1 and item[-2].lower() not in 'aeiou':
            return item[:-1] + "ies"
        elif item.endswith(("s", "sh", "ch", "x", "z")):
            return item + "es"
        elif not item.endswith("s"):
            return item + "s"
        else:
            return item

    def _get_plural_form(self, word: str) -> str:
        """
        獲取詞彙的複數形式

        Args:
            word: 單數詞彙

        Returns:
            str: 複數形式
        """
        try:
            # 特殊複數形式
            irregular_plurals = {
                'person': 'people',
                'child': 'children',
                'foot': 'feet',
                'tooth': 'teeth',
                'mouse': 'mice',
                'man': 'men',
                'woman': 'women'
            }

            if word.lower() in irregular_plurals:
                return irregular_plurals[word.lower()]

            # 規則複數形式
            if word.endswith(('s', 'sh', 'ch', 'x', 'z')):
                return word + 'es'
            elif word.endswith('y') and word[-2] not in 'aeiou':
                return word[:-1] + 'ies'
            elif word.endswith('f'):
                return word[:-1] + 'ves'
            elif word.endswith('fe'):
                return word[:-2] + 'ves'
            else:
                return word + 's'

        except Exception as e:
            self.logger.warning(f"Error getting plural form for '{word}': {str(e)}")
            return word + 's'

    def _format_single_object_description(self, class_name: str, scene_type: Optional[str],
                                        detected_objects: Optional[List[Dict]],
                                        avg_confidence: float) -> str:
        """
        處理單個物件的描述生成

        Args:
            class_name: 物件類別名稱
            scene_type: 場景類型
            detected_objects: 檢測物件列表
            avg_confidence: 平均置信度

        Returns:
            str: 單個物件的完整描述
        """
        article = "an" if class_name[0].lower() in 'aeiou' else "a"

        # 獲取材質描述符
        material_descriptor = self._get_material_descriptor(class_name, scene_type, avg_confidence)

        # 獲取位置或特徵描述符
        feature_descriptor = self._get_single_object_feature(class_name, scene_type, detected_objects)

        # 組合描述
        descriptors = []
        if material_descriptor:
            descriptors.append(material_descriptor)
        if feature_descriptor:
            descriptors.append(feature_descriptor)

        if descriptors:
            return f"{article} {' '.join(descriptors)} {class_name}"
        else:
            return f"{article} {class_name}"

    def _format_multiple_objects_description(self, class_name: str, count: int, plural_form: str,
                                           scene_type: Optional[str], detected_objects: Optional[List[Dict]],
                                           avg_confidence: float) -> str:
        """
        處理多個物件的描述生成

        Args:
            class_name: 物件類別名稱
            count: 物件數量
            plural_form: 複數形式
            scene_type: 場景類型
            detected_objects: 檢測物件列表
            avg_confidence: 平均置信度

        Returns:
            str: 多個物件的完整描述
        """
        # 數字到文字的轉換映射
        number_words = {
            2: "two", 3: "three", 4: "four", 5: "five", 6: "six",
            7: "seven", 8: "eight", 9: "nine", 10: "ten",
            11: "eleven", 12: "twelve"
        }

        # 確定基礎數量表達
        if count in number_words:
            count_expression = number_words[count]
        elif count <= 20:
            count_expression = "several"
        else:
            count_expression = "numerous"

        # 獲取材質或功能描述符
        material_descriptor = self._get_material_descriptor(class_name, scene_type, avg_confidence)

        # 構建基礎描述
        descriptors = []
        if material_descriptor:
            descriptors.append(material_descriptor)

        base_description = f"{count_expression} {' '.join(descriptors)} {plural_form}".strip()
        return base_description

    def _get_material_descriptor(self, class_name: str, scene_type: Optional[str],
                               avg_confidence: float) -> Optional[str]:
        """
        基於場景語境和置信度進行材質推斷

        Args:
            class_name: 物件類別名稱
            scene_type: 場景類型
            avg_confidence: 檢測置信度

        Returns:
            Optional[str]: 材質描述符
        """
        # 只有在置信度足夠高時才進行材質推斷
        if avg_confidence < 0.5:
            return None

        # 餐廳和用餐相關場景
        if scene_type and scene_type in ["dining_area", "restaurant", "upscale_dining", "cafe"]:
            material_mapping = {
                "chair": "wooden" if avg_confidence > 0.7 else None,
                "dining table": "wooden",
                "couch": "upholstered",
                "vase": "decorative"
            }
            return material_mapping.get(class_name)

        # 辦公場景
        elif scene_type and scene_type in ["office_workspace", "meeting_room", "conference_room"]:
            material_mapping = {
                "chair": "office",
                "dining table": "conference",
                "laptop": "modern",
                "book": "reference"
            }
            return material_mapping.get(class_name)

        # 客廳場景
        elif scene_type and scene_type in ["living_room"]:
            material_mapping = {
                "couch": "comfortable",
                "chair": "accent",
                "tv": "large",
                "vase": "decorative"
            }
            return material_mapping.get(class_name)

        # 室外場景
        elif scene_type and scene_type in ["city_street", "park_area", "parking_lot"]:
            material_mapping = {
                "car": "parked",
                "person": "walking",
                "bicycle": "stationed"
            }
            return material_mapping.get(class_name)

        # 如果沒有特定的場景映射，返回通用描述符
        generic_mapping = {
            "chair": "comfortable",
            "dining table": "sturdy",
            "car": "parked",
            "person": "present"
        }

        return generic_mapping.get(class_name)

    def _get_single_object_feature(self, class_name: str, scene_type: Optional[str],
                                 detected_objects: Optional[List[Dict]]) -> Optional[str]:
        """
        為單個物件生成特徵描述符

        Args:
            class_name: 物件類別名稱
            scene_type: 場景類型
            detected_objects: 檢測物件

        Returns:
            Optional[str]: 特徵描述符
        """
        if not detected_objects or len(detected_objects) != 1:
            return None

        obj = detected_objects[0]
        region = obj.get("region", "").lower()

        # 基於位置的描述
        if "center" in region:
            if class_name == "dining table":
                return "central"
            elif class_name == "chair":
                return "centrally placed"
        elif "corner" in region or "left" in region or "right" in region:
            return "positioned"

        # 基於場景的功能描述
        if scene_type and scene_type in ["dining_area", "restaurant"]:
            if class_name == "chair":
                return "dining"
            elif class_name == "vase":
                return "decorative"

        return None
