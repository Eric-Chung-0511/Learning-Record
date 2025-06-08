import logging
import traceback
from typing import Dict, List, Tuple, Optional, Any
import numpy as np

class ObjectDescriptionError(Exception):
    """物件描述生成過程中的自定義異常"""
    pass


class ObjectDescriptionGenerator:
    """
    物件描述生成器 - 負責將檢測到的物件轉換為自然語言描述

    該類別處理物件相關的所有描述生成邏輯，包括重要物件的識別、
    空間位置描述、物件列表格式化以及描述文本的優化。
    """

    def __init__(self,
                 min_prominence_score: float = 0.1,
                 max_categories_to_return: int = 5,
                 max_total_objects: int = 7,
                 confidence_threshold_for_description: float = 0.25,
                 region_analyzer: Optional[Any] = None):
        """
        初始化物件描述生成器

        Args:
            min_prominence_score: 物件顯著性的最低分數閾值
            max_categories_to_return: 返回的物件類別最大數量
            max_total_objects: 返回的物件總數上限
            confidence_threshold_for_description: 用於描述的置信度閾值
        """
        self.logger = logging.getLogger(self.__class__.__name__)

        self.min_prominence_score = min_prominence_score
        self.max_categories_to_return = max_categories_to_return
        self.max_total_objects = max_total_objects
        self.confidence_threshold_for_description = confidence_threshold_for_description
        self.region_analyzer = region_analyzer

        self.logger.info("ObjectDescriptionGenerator initialized with prominence_score=%.2f, "
                        "max_categories=%d, max_objects=%d, confidence_threshold=%.2f",
                        min_prominence_score, max_categories_to_return,
                        max_total_objects, confidence_threshold_for_description)

    def get_prominent_objects(self, detected_objects: List[Dict],
                          min_prominence_score: float = 0.5,
                          max_categories_to_return: Optional[int] = None) -> List[Dict]:
        """
        獲取最重要的物件，基於置信度、大小和位置計算重要性評分

        Args:
            detected_objects: 檢測到的物件列表
            min_prominence_score: 最小重要性分數閾值，範圍 0.0-1.0
            max_categories_to_return: 可選的最大返回類別數量限制

        Returns:
            List[Dict]: 按重要性排序的物件列表
        """
        try:
            if not detected_objects:
                return []

            prominent_objects = []

            for obj in detected_objects:
                # 計算重要性評分
                prominence_score = self._calculate_prominence_score(obj)

                # 只保留超過閾值的物件
                if prominence_score >= min_prominence_score:
                    obj_copy = obj.copy()
                    obj_copy['prominence_score'] = prominence_score
                    prominent_objects.append(obj_copy)

            # 按重要性評分排序（從高到低）
            prominent_objects.sort(key=lambda x: x.get('prominence_score', 0), reverse=True)

            # 如果指定了最大類別數量限制，進行過濾
            if max_categories_to_return is not None and max_categories_to_return > 0:
                categories_seen = set()
                filtered_objects = []

                for obj in prominent_objects:
                    class_name = obj.get("class_name", "unknown")

                    # 如果是新類別且未達到限制
                    if class_name not in categories_seen:
                        if len(categories_seen) < max_categories_to_return:
                            categories_seen.add(class_name)
                            filtered_objects.append(obj)
                    else:
                        # 已見過的類別，直接添加
                        filtered_objects.append(obj)

                return filtered_objects

            return prominent_objects

        except Exception as e:
            self.logger.error(f"Error calculating prominent objects: {str(e)}")
            return []

    def set_region_analyzer(self, region_analyzer: Any) -> None:
        """
        設置RegionAnalyzer，用於標準化空間描述生成

        Args:
            region_analyzer: RegionAnalyzer實例
        """
        try:
            self.region_analyzer = region_analyzer
            self.logger.info("RegionAnalyzer instance set for ObjectDescriptionGenerator")
        except Exception as e:
            self.logger.warning(f"Error setting RegionAnalyzer: {str(e)}")

    def _get_standardized_spatial_description(self, obj: Dict) -> str:
        """
        使用RegionAnalyzer生成標準化空間描述的內部方法

        Args:
            obj: 物件字典

        Returns:
            str: 標準化空間描述，失敗時返回空字串
        """
        try:
            if hasattr(self, 'region_analyzer') and self.region_analyzer:
                region = obj.get("region", "")
                object_type = obj.get("class_name", "")

                if hasattr(self.region_analyzer, 'get_contextual_spatial_description'):
                    return self.region_analyzer.get_contextual_spatial_description(region, object_type)
                elif hasattr(self.region_analyzer, 'get_spatial_description_phrase'):
                    return self.region_analyzer.get_spatial_description_phrase(region)

            return ""

        except Exception as e:
            self.logger.warning(f"Error getting standardized spatial description: {str(e)}")
            if object_type:
                return f"visible in the scene"
            return "present in the view"

    def _calculate_prominence_score(self, obj: Dict) -> float:
        """
        計算物件的重要性評分

        Args:
            obj: 物件字典，包含檢測信息

        Returns:
            float: 重要性評分 (0.0-1.0)
        """
        try:
            # 基礎置信度評分 (權重: 40%)
            confidence = obj.get("confidence", 0.5)
            confidence_score = confidence * 0.4

            # 大小評分 (權重: 30%)
            normalized_area = obj.get("normalized_area", 0.1)
            # 使用對數縮放避免過大物件主導評分
            size_score = min(np.log(normalized_area * 10 + 1) / np.log(11), 1.0) * 0.3

            # 位置評分 (權重: 20%)
            # 中心區域的物件通常更重要
            center_x, center_y = obj.get("normalized_center", [0.5, 0.5])
            distance_from_center = np.sqrt((center_x - 0.5)**2 + (center_y - 0.5)**2)
            position_score = (1 - min(distance_from_center * 2, 1.0)) * 0.2

            # 類別重要性評分 (權重: 10%)
            class_importance = self._get_class_importance(obj.get("class_name", "unknown"))
            class_score = class_importance * 0.1

            total_score = confidence_score + size_score + position_score + class_score

            # 確保評分在有效範圍內
            return max(0.0, min(1.0, total_score))

        except Exception as e:
            self.logger.warning(f"Error calculating prominence score for object: {str(e)}")
            return 0.5  # 返回中等評分作為備用

    def _get_class_importance(self, class_name: str) -> float:
        """
        根據物件類別返回重要性係數

        Args:
            class_name: 物件類別名稱

        Returns:
            float: 類別重要性係數 (0.0-1.0)
        """
        # 高重要性物件（人、車輛、建築）
        high_importance = ["person", "car", "truck", "bus", "motorcycle", "bicycle", "building"]

        # 中等重要性物件（家具、電器）
        medium_importance = ["chair", "couch", "tv", "laptop", "refrigerator", "dining table", "bed"]

        # 低重要性物件（小物品、配件）
        low_importance = ["handbag", "backpack", "umbrella", "cell phone", "remote", "mouse"]

        class_name_lower = class_name.lower()

        if any(item in class_name_lower for item in high_importance):
            return 1.0
        elif any(item in class_name_lower for item in medium_importance):
            return 0.7
        elif any(item in class_name_lower for item in low_importance):
            return 0.4
        else:
            return 0.6  # 預設中等重要性

    def format_object_list_for_description(self,
                                          objects: List[Dict],
                                          use_indefinite_article_for_one: bool = False,
                                          count_threshold_for_generalization: int = -1,
                                          max_types_to_list: int = 5) -> str:
        """
        將物件列表格式化為人類可讀的字符串，包含計數信息

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
                    plural_name = name
                    if name.endswith("y") and not name.lower().endswith(("ay", "ey", "iy", "oy", "uy")):
                        plural_name = name[:-1] + "ies"
                    elif name.endswith(("s", "sh", "ch", "x", "z")):
                        plural_name = name + "es"
                    elif not name.endswith("s"):
                        plural_name = name + "s"

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

    def get_spatial_description(self, obj: Dict, image_width: Optional[int] = None,
                           image_height: Optional[int] = None,
                           region_analyzer: Optional[Any] = None) -> str:
        """
        為物件生成空間位置描述

        Args:
            obj: 物件字典
            image_width: 可選的圖像寬度
            image_height: 可選的圖像高度
            region_analyzer: 可選的RegionAnalyzer實例，用於生成標準化描述

        Returns:
            str: 空間描述字符串，空值region時返回空字串
        """
        try:
            region = obj.get("region") or ""

            # 處理空值或無效region，直接返回空字串避免不完整描述
            if not region.strip() or region == "unknown":
                # 根據物件類型提供合適的預設位置描述
                if object_type and any(vehicle in object_type.lower() for vehicle in ["car", "truck", "bus"]):
                    return "positioned in the scene"
                elif object_type and "person" in object_type.lower():
                    return "present in the area"
                else:
                    return "located in the scene"

            # 如果提供了RegionAnalyzer實例，使用其標準化方法
            if region_analyzer and hasattr(region_analyzer, 'get_spatial_description_phrase'):
                object_type = obj.get("class_name", "")
                if hasattr(region_analyzer, 'get_contextual_spatial_description'):
                    spatial_desc = region_analyzer.get_contextual_spatial_description(region, object_type)
                else:
                    spatial_desc = region_analyzer.get_spatial_description_phrase(region)

                if spatial_desc:
                    return spatial_desc

            # 備用邏輯：使用改進的內建映射
            clean_region = region.replace('_', ' ').strip().lower()

            region_map = {
                "top left": "in the upper left area",
                "top center": "in the upper area",
                "top right": "in the upper right area",
                "middle left": "on the left side",
                "middle center": "in the center",
                "center": "in the center",
                "middle right": "on the right side",
                "bottom left": "in the lower left area",
                "bottom center": "in the lower area",
                "bottom right": "in the lower right area"
            }

            # 直接映射匹配
            if clean_region in region_map:
                return region_map[clean_region]

            # 模糊匹配處理
            if "top" in clean_region and "left" in clean_region:
                return "in the upper left area"
            elif "top" in clean_region and "right" in clean_region:
                return "in the upper right area"
            elif "bottom" in clean_region and "left" in clean_region:
                return "in the lower left area"
            elif "bottom" in clean_region and "right" in clean_region:
                return "in the lower right area"
            elif "top" in clean_region:
                return "in the upper area"
            elif "bottom" in clean_region:
                return "in the lower area"
            elif "left" in clean_region:
                return "on the left side"
            elif "right" in clean_region:
                return "on the right side"
            elif "center" in clean_region or "middle" in clean_region:
                return "in the center"

            # 如果region無法識別，使用normalized_center作為最後備用
            norm_center = obj.get("normalized_center")
            if norm_center and image_width and image_height:
                x_norm, y_norm = norm_center
                h_pos = "left" if x_norm < 0.4 else "right" if x_norm > 0.6 else "center"
                v_pos = "upper" if y_norm < 0.4 else "lower" if y_norm > 0.6 else "center"

                if h_pos == "center" and v_pos == "center":
                    return "in the center"
                return f"in the {v_pos} {h_pos} area"

            # 如果所有方法都失敗，返回空字串
            return ""

        except Exception as e:
            self.logger.warning(f"Error generating spatial description: {str(e)}")
            return ""

    def optimize_object_description(self, description: str) -> str:
        """
        優化物件描述，避免重複列舉相同物件

        Args:
            description: 原始描述文本

        Returns:
            str: 優化後的描述文本
        """
        try:
            import re

            # 處理床鋪重複描述
            if "bed in the room" in description:
                description = description.replace("a bed in the room", "a bed")

            # 處理重複的物件列表
            object_lists = re.findall(r'with ([^\.]+?)(?:\.|\band\b)', description)

            for obj_list in object_lists:
                # 計算每個物件出現次數
                items = re.findall(r'([a-zA-Z\s]+)(?:,|\band\b|$)', obj_list)
                item_counts = {}

                for item in items:
                    item = item.strip()
                    if item and item not in ["and", "with"]:
                        if item not in item_counts:
                            item_counts[item] = 0
                        item_counts[item] += 1

                # 生成優化後的物件列表
                if item_counts:
                    new_items = []
                    for item, count in item_counts.items():
                        if count > 1:
                            new_items.append(f"{count} {item}s")
                        else:
                            new_items.append(item)

                    # 格式化新列表
                    if len(new_items) == 1:
                        new_list = new_items[0]
                    elif len(new_items) == 2:
                        new_list = f"{new_items[0]} and {new_items[1]}"
                    else:
                        new_list = ", ".join(new_items[:-1]) + f", and {new_items[-1]}"

                    # 替換原始列表
                    description = description.replace(obj_list, new_list)

            return description

        except Exception as e:
            self.logger.warning(f"Error optimizing object description: {str(e)}")
            return description

    def generate_dynamic_everyday_description(self,
                                            detected_objects: List[Dict],
                                            lighting_info: Optional[Dict] = None,
                                            viewpoint: str = "eye_level",
                                            spatial_analysis: Optional[Dict] = None,
                                            image_dimensions: Optional[Tuple[int, int]] = None,
                                            places365_info: Optional[Dict] = None,
                                            object_statistics: Optional[Dict] = None) -> str:
        """
        為日常場景動態生成描述，基於所有相關的檢測物件、計數和上下文

        Args:
            detected_objects: 檢測到的物件列表
            lighting_info: 照明信息
            viewpoint: 視角類型
            spatial_analysis: 空間分析結果
            image_dimensions: 圖像尺寸
            places365_info: Places365場景分類信息
            object_statistics: 物件統計信息

        Returns:
            str: 動態生成的場景描述
        """
        try:
            description_segments = []
            image_width, image_height = image_dimensions if image_dimensions else (None, None)

            self.logger.debug(f"Generating dynamic description for {len(detected_objects)} objects, "
                            f"viewpoint: {viewpoint}, lighting: {lighting_info is not None}")

            # 1. 整體氛圍（照明和視角）
            ambiance_parts = []
            if lighting_info:
                time_of_day = lighting_info.get("time_of_day", "unknown lighting")
                is_indoor = lighting_info.get("is_indoor")
                ambiance_statement = "This is"
                if is_indoor is True:
                    ambiance_statement += " an indoor scene"
                elif is_indoor is False:
                    ambiance_statement += " an outdoor scene"
                else:
                    ambiance_statement += " a scene"

                # remove underline
                readable_lighting = f"with {time_of_day.replace('_', ' ')} lighting conditions"
                ambiance_statement += f", likely {readable_lighting}."
                ambiance_parts.append(ambiance_statement)

            if viewpoint and viewpoint != "eye_level":
                if not ambiance_parts:
                    ambiance_parts.append(f"From {viewpoint.replace('_', ' ')}, the general layout of the scene is observed.")
                else:
                    ambiance_parts[-1] = ambiance_parts[-1].rstrip('.') + f", viewed from {viewpoint.replace('_', ' ')}."

            if ambiance_parts:
                description_segments.append(" ".join(ambiance_parts))

            # 2. 描述所有檢測到的物件，按類別分組，使用準確計數和位置
            if not detected_objects:
                if not description_segments:
                    description_segments.append("A general scene is visible, but no specific objects were clearly identified.")
                else:
                    description_segments.append("Within this setting, no specific objects were clearly identified.")
            else:
                objects_by_class: Dict[str, List[Dict]] = {}

                # 使用置信度過濾
                confident_objects = [obj for obj in detected_objects
                                   if obj.get("confidence", 0) >= self.confidence_threshold_for_description]
                print(f"DEBUG: After confidence filtering (threshold={self.confidence_threshold_for_description}):")
                for class_name in ["car", "traffic light", "person", "handbag"]:
                    class_objects = [obj for obj in confident_objects if obj.get("class_name") == class_name]
                    print(f"DEBUG: {class_name}: {len(class_objects)} confident objects")

                if not confident_objects:
                    no_confident_obj_msg = "While some elements might be present, no objects were identified with sufficient confidence for a detailed description."
                    if not description_segments:
                        description_segments.append(no_confident_obj_msg)
                    else:
                        description_segments.append(no_confident_obj_msg.lower().capitalize())
                else:
                    if object_statistics:
                        # 使用預計算的統計信息，採用動態的信心度
                        for class_name, stats in object_statistics.items():
                            count = stats.get("count", 0)
                            avg_confidence = stats.get("avg_confidence", 0)

                            # 動態調整置信度閾值
                            dynamic_threshold = self.confidence_threshold_for_description
                            if class_name in ["potted plant", "vase", "clock", "book"]:
                                dynamic_threshold = max(0.15, self.confidence_threshold_for_description * 0.6)
                            elif count >= 3:
                                dynamic_threshold = max(0.2, self.confidence_threshold_for_description * 0.8)

                            if count > 0 and avg_confidence >= dynamic_threshold:
                                matching_objects = [obj for obj in confident_objects if obj.get("class_name") == class_name]
                                if not matching_objects:
                                    matching_objects = [obj for obj in detected_objects
                                                      if obj.get("class_name") == class_name and obj.get("confidence", 0) >= dynamic_threshold]

                                if matching_objects:
                                    actual_count = min(stats["count"], len(matching_objects))
                                    objects_by_class[class_name] = matching_objects[:actual_count]
                    else:
                        # 備用邏輯，同樣使用動態閾值
                        for obj in confident_objects:
                            name = obj.get("class_name", "unknown object")
                            if name == "unknown object" or not name:
                                continue
                            if name not in objects_by_class:
                                objects_by_class[name] = []
                            objects_by_class[name].append(obj)

                            print(f"DEBUG: Before spatial deduplication:")
                            for class_name in ["car", "traffic light", "person", "handbag"]:
                                if class_name in objects_by_class:
                                    print(f"DEBUG: {class_name}: {len(objects_by_class[class_name])} objects before dedup")

                    if not objects_by_class:
                        description_segments.append("No common objects were confidently identified for detailed description.")
                    else:
                        # 物件組排序函數
                        def sort_key_object_groups(item_tuple: Tuple[str, List[Dict]]):
                            class_name_key, obj_group_list = item_tuple
                            priority = 3
                            count = len(obj_group_list)

                            # 確保類別名稱已標準化
                            normalized_class_name = self._normalize_object_class_name(class_name_key)

                            # 動態優先級
                            if normalized_class_name == "person":
                                priority = 0
                            elif normalized_class_name in ["dining table", "chair", "sofa", "bed"]:
                                priority = 1
                            elif normalized_class_name in ["car", "bus", "truck", "traffic light"]:
                                priority = 2
                            elif count >= 3:
                                priority = max(1, priority - 1)
                            elif normalized_class_name in ["potted plant", "vase", "clock", "book"] and count >= 2:
                                priority = 2

                            avg_area = sum(o.get("normalized_area", 0.0) for o in obj_group_list) / len(obj_group_list) if obj_group_list else 0
                            quantity_bonus = min(count / 5.0, 1.0)

                            return (priority, -len(obj_group_list), -avg_area, -quantity_bonus)

                        # remove duplicate
                        deduplicated_objects_by_class = {}
                        processed_positions = []

                        for class_name, group_of_objects in objects_by_class.items():
                            unique_objects = []

                            for obj in group_of_objects:
                                obj_position = obj.get("normalized_center", [0.5, 0.5])
                                is_duplicate = False

                                for processed_pos in processed_positions:
                                    position_distance = abs(obj_position[0] - processed_pos[0]) + abs(obj_position[1] - processed_pos[1])
                                    if position_distance < 0.15:
                                        is_duplicate = True
                                        break

                                if not is_duplicate:
                                    unique_objects.append(obj)
                                    processed_positions.append(obj_position)

                            if unique_objects:
                                deduplicated_objects_by_class[class_name] = unique_objects

                        objects_by_class = deduplicated_objects_by_class
                        print(f"DEBUG: After spatial deduplication:")
                        for class_name in ["car", "traffic light", "person", "handbag"]:
                            if class_name in objects_by_class:
                                print(f"DEBUG: {class_name}: {len(objects_by_class[class_name])} objects after dedup")

                        sorted_object_groups = sorted(objects_by_class.items(), key=sort_key_object_groups)

                        object_clauses = []

                        for class_name, group_of_objects in sorted_object_groups:
                            count = len(group_of_objects)
                            if class_name in ["car", "traffic light", "person", "handbag"]:
                                print(f"DEBUG: Final count for {class_name}: {count}")
                            if count == 0:
                                continue

                            # 標準化class name
                            normalized_class_name = self._normalize_object_class_name(class_name)

                            # 使用統計信息確保準確的數量描述
                            if object_statistics and class_name in object_statistics:
                                actual_count = object_statistics[class_name]["count"]
                                formatted_name_with_exact_count = self._format_object_count_description(
                                    normalized_class_name, 
                                    actual_count,
                                    scene_type=scene_type  
                                )
                            else:
                                formatted_name_with_exact_count = self._format_object_count_description(
                                    normalized_class_name, 
                                    count,
                                    scene_type=scene_type  
                                )

                            if formatted_name_with_exact_count == "no specific objects clearly identified" or not formatted_name_with_exact_count:
                                continue

                            # 確定群組的集體位置
                            location_description_suffix = ""
                            if count == 1:
                                spatial_desc = self.get_spatial_description(group_of_objects[0], image_width, image_height, self.region_analyzer)
                                if spatial_desc:
                                    location_description_suffix = f"is {spatial_desc}"
                                else:
                                    distinct_regions = sorted(list(set(obj.get("region", "") for obj in group_of_objects if obj.get("region"))))
                                    valid_regions = [r for r in distinct_regions if r and r != "unknown" and r.strip()]
                                    if not valid_regions:
                                        location_description_suffix = "is positioned in the scene"
                                    elif len(valid_regions) == 1:
                                        spatial_desc = self.get_spatial_description_phrase(valid_regions[0])
                                        location_description_suffix = f"is primarily {spatial_desc}" if spatial_desc else "is positioned in the scene"
                                    elif len(valid_regions) == 2:
                                        clean_region1 = valid_regions[0].replace('_', ' ')
                                        clean_region2 = valid_regions[1].replace('_', ' ')
                                        location_description_suffix = f"is mainly across the {clean_region1} and {clean_region2} areas"
                                    else:
                                        location_description_suffix = "is distributed in various parts of the scene"
                            else:
                                distinct_regions = sorted(list(set(obj.get("region", "") for obj in group_of_objects if obj.get("region"))))
                                valid_regions = [r for r in distinct_regions if r and r != "unknown" and r.strip()]
                                if not valid_regions:
                                    location_description_suffix = "are visible in the scene"
                                elif len(valid_regions) == 1:
                                    clean_region = valid_regions[0].replace('_', ' ')
                                    location_description_suffix = f"are primarily in the {clean_region} area"
                                elif len(valid_regions) == 2:
                                    clean_region1 = valid_regions[0].replace('_', ' ')
                                    clean_region2 = valid_regions[1].replace('_', ' ')
                                    location_description_suffix = f"are mainly across the {clean_region1} and {clean_region2} areas"
                                else:
                                    location_description_suffix = "are distributed in various parts of the scene"

                            # 首字母大寫
                            formatted_name_capitalized = formatted_name_with_exact_count[0].upper() + formatted_name_with_exact_count[1:]
                            object_clauses.append(f"{formatted_name_capitalized} {location_description_suffix}")

                        if object_clauses:
                            if not description_segments:
                                if object_clauses:
                                    first_clause = object_clauses.pop(0)
                                    description_segments.append(first_clause + ".")
                            else:
                                if object_clauses:
                                    description_segments.append("The scene features:")

                            if object_clauses:
                                joined_object_clauses = ". ".join(object_clauses)
                                if joined_object_clauses and not joined_object_clauses.endswith("."):
                                    joined_object_clauses += "."
                                description_segments.append(joined_object_clauses)

                        elif not description_segments:
                            return "The image depicts a scene, but specific objects could not be described with confidence or detail."

            # 最終組裝和格式化
            raw_description = ""
            for i, segment in enumerate(filter(None, description_segments)):
                segment = segment.strip()
                if not segment:
                    continue

                if not raw_description:
                    raw_description = segment
                else:
                    if not raw_description.endswith(('.', '!', '?')):
                        raw_description += "."
                    raw_description += " " + (segment[0].upper() + segment[1:] if len(segment) > 1 else segment.upper())

            if raw_description and not raw_description.endswith(('.', '!', '?')):
                raw_description += "."

            # 移除重複性和不適當的描述詞彙
            raw_description = self._remove_repetitive_descriptors(raw_description)

            if not raw_description or len(raw_description.strip()) < 20:
                if 'confident_objects' in locals() and confident_objects:
                    return "The scene contains several detected objects, but a detailed textual description could not be fully constructed."
                else:
                    return "A general scene is depicted with no objects identified with high confidence."

            return raw_description

        except Exception as e:
            error_msg = f"Error generating dynamic everyday description: {str(e)}"
            self.logger.error(f"{error_msg}\n{traceback.format_exc()}")
            raise ObjectDescriptionError(error_msg) from e

    def _remove_repetitive_descriptors(self, description: str) -> str:
        """
        移除描述中的重複性和不適當的描述詞彙，特別是 "identical" 等詞彙
        
        Args:
            description: 原始描述文本
            
        Returns:
            str: 清理後的描述文本
        """
        try:
            import re
            
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

    def _format_object_count_description(self, class_name: str, count: int, 
                                    scene_type: Optional[str] = None,
                                    detected_objects: Optional[List[Dict]] = None,
                                    avg_confidence: float = 0.0) -> str:
        """
        格式化物件數量描述的核心方法，整合空間排列、材質推斷和場景語境
        
        這個方法是整個物件描述系統的核心，它將多個子功能整合在一起：
        1. 數字到文字的轉換（避免阿拉伯數字）
        2. 基於場景的材質推斷
        3. 空間排列模式的描述
        4. 語境化的物件描述
        
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

    def _format_single_object_description(self, class_name: str, scene_type: Optional[str],
                                        detected_objects: Optional[List[Dict]], 
                                        avg_confidence: float) -> str:
        """
        處理單個物件的描述生成
        
        對於單個物件，我們重點在於通過材質推斷和位置描述來豐富描述內容，
        避免簡單的 "a chair" 這樣的描述，而是生成 "a wooden dining chair" 這樣的表達
        
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
        
        對於多個物件，我們的重點是：
        1. 將數字轉換為文字表達
        2. 分析空間排列模式
        3. 添加適當的材質或功能描述
        4. 生成自然流暢的描述
        
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
        
        # 獲取空間排列描述
        spatial_descriptor = self._get_spatial_arrangement_descriptor(class_name, scene_type, 
                                                                    detected_objects, count)
        
        # 組合最終描述
        descriptors = []
        if material_descriptor:
            descriptors.append(material_descriptor)
        
        # 構建基礎描述
        base_description = f"{count_expression} {' '.join(descriptors)} {plural_form}".strip()
        
        # 添加空間排列信息
        if spatial_descriptor:
            return f"{base_description} {spatial_descriptor}"
        else:
            return base_description

    def _get_material_descriptor(self, class_name: str, scene_type: Optional[str], 
                            avg_confidence: float) -> Optional[str]:
        """
        基於場景語境和置信度進行材質推斷
        
        這個方法實現了智能的材質推斷，它不依賴複雜的圖像分析，
        而是基於常識和場景邏輯來推斷最可能的材質描述
        
        Args:
            class_name: 物件類別名稱
            scene_type: 場景類型
            avg_confidence: 檢測置信度，影響推斷的保守程度
            
        Returns:
            Optional[str]: 材質描述符，如果無法推斷則返回None
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
                "dining table": "conference",  # 在辦公環境中，餐桌通常是會議桌
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

    def _get_spatial_arrangement_descriptor(self, class_name: str, scene_type: Optional[str],
                                        detected_objects: Optional[List[Dict]], 
                                        count: int) -> Optional[str]:
        """
        分析物件的空間排列模式並生成相應描述
        
        這個方法通過分析物件的位置分布來判斷排列模式，
        然後根據物件類型和場景生成適當的空間描述
        
        Args:
            class_name: 物件類別名稱
            scene_type: 場景類型
            detected_objects: 該類型的所有檢測物件
            count: 物件數量
            
        Returns:
            Optional[str]: 空間排列描述，如果無法分析則返回None
        """
        if not detected_objects or len(detected_objects) < 2:
            return None
        
        try:
            # 提取物件的標準化位置
            positions = []
            for obj in detected_objects:
                center = obj.get("normalized_center", [0.5, 0.5])
                if isinstance(center, (list, tuple)) and len(center) >= 2:
                    positions.append(center)
            
            if len(positions) < 2:
                return None
            
            # 分析排列模式
            arrangement_pattern = self._analyze_arrangement_pattern(positions)
            
            # 根據物件類型和場景生成描述
            return self._generate_arrangement_description(class_name, scene_type, 
                                                        arrangement_pattern, count)
        
        except Exception as e:
            self.logger.warning(f"Error analyzing spatial arrangement: {str(e)}")
            return None

    def _analyze_arrangement_pattern(self, positions: List[List[float]]) -> str:
        """
        分析位置點的排列模式
        
        這個方法使用簡單的幾何分析來判斷物件的排列類型，
        幫助我們理解物件在空間中的組織方式
        
        Args:
            positions: 標準化的位置座標列表
            
        Returns:
            str: 排列模式類型（linear, clustered, scattered, circular等）
        """
        import numpy as np
        
        if len(positions) < 2:
            return "single"
        
        # 轉換為numpy陣列便於計算
        pos_array = np.array(positions)
        
        # 計算位置的分布特徵
        x_coords = pos_array[:, 0]
        y_coords = pos_array[:, 1]
        
        # 分析x和y方向的變異程度
        x_variance = np.var(x_coords)
        y_variance = np.var(y_coords)
        
        # 計算物件間的平均距離
        distances = []
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                dist = np.sqrt((positions[i][0] - positions[j][0])**2 + 
                            (positions[i][1] - positions[j][1])**2)
                distances.append(dist)
        
        avg_distance = np.mean(distances) if distances else 0
        distance_variance = np.var(distances) if distances else 0
        
        # 判斷排列模式
        if len(positions) >= 4 and self._is_circular_pattern(positions):
            return "circular"
        elif x_variance < 0.05 or y_variance < 0.05:  # 一個方向變異很小
            return "linear"
        elif avg_distance < 0.3 and distance_variance < 0.02:  # 物件聚集且距離相近
            return "clustered"
        elif avg_distance > 0.6:  # 物件分散
            return "scattered"
        elif distance_variance < 0.03:  # 距離一致，可能是規則排列
            return "regular"
        else:
            return "distributed"

    def _is_circular_pattern(self, positions: List[List[float]]) -> bool:
        """
        檢查位置是否形成圓形或環形排列
        
        Args:
            positions: 位置座標列表
            
        Returns:
            bool: 是否為圓形排列
        """
        import numpy as np
        
        if len(positions) < 4:
            return False
        
        try:
            pos_array = np.array(positions)
            
            # 計算中心點
            center_x = np.mean(pos_array[:, 0])
            center_y = np.mean(pos_array[:, 1])
            
            # 計算每個點到中心的距離
            distances_to_center = []
            for pos in positions:
                dist = np.sqrt((pos[0] - center_x)**2 + (pos[1] - center_y)**2)
                distances_to_center.append(dist)
            
            # 如果所有距離都相近，可能是圓形排列
            distance_variance = np.var(distances_to_center)
            return distance_variance < 0.05 and np.mean(distances_to_center) > 0.2
        
        except:
            return False

    def _generate_arrangement_description(self, class_name: str, scene_type: Optional[str],
                                        arrangement_pattern: str, count: int) -> Optional[str]:
        """
        根據物件類型、場景和排列模式生成空間描述
        
        這個方法將抽象的排列模式轉換為自然語言描述，
        並根據具體的物件類型和場景語境進行定制
        
        Args:
            class_name: 物件類別名稱
            scene_type: 場景類型
            arrangement_pattern: 排列模式
            count: 物件數量
            
        Returns:
            Optional[str]: 生成的空間排列描述
        """
        # 基於物件類型的描述模板
        arrangement_templates = {
            "chair": {
                "linear": "arranged in a row",
                "clustered": "grouped together for conversation",
                "circular": "arranged around the table",
                "scattered": "positioned throughout the space",
                "regular": "evenly spaced",
                "distributed": "thoughtfully positioned"
            },
            "dining table": {
                "linear": "aligned to create a unified dining space",
                "clustered": "grouped to form intimate dining areas",
                "scattered": "distributed to optimize space flow",
                "regular": "systematically positioned",
                "distributed": "strategically placed"
            },
            "car": {
                "linear": "parked in sequence",
                "clustered": "grouped in the parking area",
                "scattered": "distributed throughout the lot",
                "regular": "neatly parked",
                "distributed": "positioned across the area"
            },
            "person": {
                "linear": "moving in a line",
                "clustered": "gathered together",
                "circular": "forming a circle",
                "scattered": "spread across the area",
                "distributed": "positioned throughout the scene"
            }
        }
        
        # 獲取對應的描述模板
        if class_name in arrangement_templates:
            template_dict = arrangement_templates[class_name]
            base_description = template_dict.get(arrangement_pattern, "positioned in the scene")
        else:
            # 通用的排列描述
            generic_templates = {
                "linear": "arranged in a line",
                "clustered": "grouped together",
                "circular": "arranged in a circular pattern",
                "scattered": "distributed across the space",
                "regular": "evenly positioned",
                "distributed": "thoughtfully placed"
            }
            base_description = generic_templates.get(arrangement_pattern, "positioned in the scene")
        
        return base_description

    def _get_single_object_feature(self, class_name: str, scene_type: Optional[str],
                                detected_objects: Optional[List[Dict]]) -> Optional[str]:
        """
        為單個物件生成特徵描述符
        
        當只有一個物件時，我們可以提供更具體的位置或功能描述
        
        Args:
            class_name: 物件類別名稱
            scene_type: 場景類型
            detected_objects: 檢測物件（單個）
            
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

    def _normalize_object_class_name(self, class_name: str) -> str:
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
            import re
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

    def generate_basic_details(self, scene_type: str, detected_objects: List[Dict]) -> str:
        """
        當模板不可用時生成基本詳細信息

        Args:
            scene_type: 識別的場景類型
            detected_objects: 檢測到的物件列表

        Returns:
            str: 基本場景詳細信息
        """
        try:
            # 處理特定場景類型的自定義邏輯
            if scene_type == "living_room":
                tv_objs = [obj for obj in detected_objects if obj.get("class_id") == 62]  # TV
                sofa_objs = [obj for obj in detected_objects if obj.get("class_id") == 57]  # Sofa

                if tv_objs and sofa_objs:
                    tv_region = tv_objs[0].get("region", "center")
                    sofa_region = sofa_objs[0].get("region", "center")

                    arrangement = f"The TV is in the {tv_region.replace('_', ' ')} of the image, "
                    arrangement += f"while the sofa is in the {sofa_region.replace('_', ' ')}. "

                    return f"{arrangement}This appears to be a space designed for relaxation and entertainment."

            elif scene_type == "bedroom":
                bed_objs = [obj for obj in detected_objects if obj.get("class_id") == 59]  # Bed

                if bed_objs:
                    bed_region = bed_objs[0].get("region", "center")
                    extra_items = []

                    for obj in detected_objects:
                        if obj.get("class_id") == 74:  # Clock
                            extra_items.append("clock")
                        elif obj.get("class_id") == 73:  # Book
                            extra_items.append("book")

                    extras = ""
                    if extra_items:
                        extras = f" There is also a {' and a '.join(extra_items)} visible."

                    return f"The bed is located in the {bed_region.replace('_', ' ')} of the image.{extras}"

            elif scene_type in ["dining_area", "kitchen"]:
                # 計算食物和餐飲相關物品
                food_items = []
                for obj in detected_objects:
                    if obj.get("class_id") in [39, 41, 42, 43, 44, 45]:  # 廚房物品
                        food_items.append(obj.get("class_name", "kitchen item"))

                food_str = ""
                if food_items:
                    unique_items = list(set(food_items))
                    if len(unique_items) <= 3:
                        food_str = f" with {', '.join(unique_items)}"
                    else:
                        food_str = f" with {', '.join(unique_items[:3])} and other items"

                return f"{food_str}."

            elif scene_type == "city_street":
                # 計算人員和車輛
                people_count = len([obj for obj in detected_objects if obj.get("class_id") == 0])
                vehicle_count = len([obj for obj in detected_objects
                                   if obj.get("class_id") in [1, 2, 3, 5, 7]])  # Bicycle, car, motorbike, bus, truck

                traffic_desc = ""
                if people_count > 0 and vehicle_count > 0:
                    traffic_desc = f" with {people_count} {'people' if people_count > 1 else 'person'} and "
                    traffic_desc += f"{vehicle_count} {'vehicles' if vehicle_count > 1 else 'vehicle'}"
                elif people_count > 0:
                    traffic_desc = f" with {people_count} {'people' if people_count > 1 else 'person'}"
                elif vehicle_count > 0:
                    traffic_desc = f" with {vehicle_count} {'vehicles' if vehicle_count > 1 else 'vehicle'}"

                return f"{traffic_desc}."

            elif scene_type == "asian_commercial_street":
                # 尋找關鍵城市元素
                people_count = len([obj for obj in detected_objects if obj.get("class_id") == 0])
                vehicle_count = len([obj for obj in detected_objects if obj.get("class_id") in [1, 2, 3]])

                # 分析行人分布
                people_positions = []
                for obj in detected_objects:
                    if obj.get("class_id") == 0:  # Person
                        people_positions.append(obj.get("normalized_center", (0.5, 0.5)))

                # 檢查人員是否沿線分布（表示步行路徑）
                structured_path = False
                if len(people_positions) >= 3:
                    # 簡化檢查 - 查看多個人員的y坐標是否相似
                    y_coords = [pos[1] for pos in people_positions]
                    y_mean = sum(y_coords) / len(y_coords)
                    y_variance = sum((y - y_mean)**2 for y in y_coords) / len(y_coords)
                    if y_variance < 0.05:  # 低變異數表示線性排列
                        structured_path = True

                street_desc = "A commercial street with "
                if people_count > 0:
                    street_desc += f"{people_count} {'pedestrians' if people_count > 1 else 'pedestrian'}"
                    if vehicle_count > 0:
                        street_desc += f" and {vehicle_count} {'vehicles' if vehicle_count > 1 else 'vehicle'}"
                elif vehicle_count > 0:
                    street_desc += f"{vehicle_count} {'vehicles' if vehicle_count > 1 else 'vehicle'}"
                else:
                    street_desc += "various commercial elements"

                if structured_path:
                    street_desc += ". The pedestrians appear to be following a defined walking path"

                # 添加文化元素
                street_desc += ". The signage and architectural elements suggest an Asian urban setting."

                return street_desc

            # 默認通用描述
            return "The scene contains various elements characteristic of this environment."

        except Exception as e:
            self.logger.warning(f"Error generating basic details for scene_type '{scene_type}': {str(e)}")
            return "The scene contains various elements characteristic of this environment."

    def generate_placeholder_content(self, placeholder: str, detected_objects: List[Dict], scene_type: str) -> str:
        """
        為模板佔位符生成內容

        Args:
            placeholder: 模板佔位符
            detected_objects: 檢測到的物件列表
            scene_type: 場景類型

        Returns:
            str: 生成的佔位符內容
        """
        try:
            # 處理不同類型的佔位符與自定義邏輯
            if placeholder == "furniture":
                # 提取家具物品
                furniture_ids = [56, 57, 58, 59, 60, 61]  # 家具類別ID示例
                furniture_objects = [obj for obj in detected_objects if obj.get("class_id") in furniture_ids]

                if furniture_objects:
                    furniture_names = []
                    for obj in furniture_objects[:3]:
                        raw_name = obj.get("class_name", "furniture")
                        normalized_name = self._normalize_object_class_name(raw_name)
                        furniture_names.append(normalized_name)

                    unique_names = list(set(furniture_names))
                    if len(unique_names) == 1:
                        return unique_names[0]
                    elif len(unique_names) == 2:
                        return f"{unique_names[0]} and {unique_names[1]}"
                    else:
                        return ", ".join(unique_names[:-1]) + f", and {unique_names[-1]}"
                return "various furniture items"

            elif placeholder == "electronics":
                # 提取電子物品
                electronics_ids = [62, 63, 64, 65, 66, 67, 68, 69, 70]  # 電子設備類別ID示例
                electronics_objects = [obj for obj in detected_objects if obj.get("class_id") in electronics_ids]

                if electronics_objects:
                    electronics_names = [obj.get("class_name", "electronic device") for obj in electronics_objects[:3]]
                    return ", ".join(set(electronics_names))
                return "electronic devices"

            elif placeholder == "people_count":
                # 計算人數
                people_count = len([obj for obj in detected_objects if obj.get("class_id") == 0])

                if people_count == 0:
                    return "no people"
                elif people_count == 1:
                    return "one person"
                elif people_count < 5:
                    return f"{people_count} people"
                else:
                    return "several people"

            elif placeholder == "seating":
                # 提取座位物品
                seating_ids = [56, 57]  # chair, sofa
                seating_objects = [obj for obj in detected_objects if obj.get("class_id") in seating_ids]

                if seating_objects:
                    seating_names = [obj.get("class_name", "seating") for obj in seating_objects[:2]]
                    return ", ".join(set(seating_names))
                return "seating arrangements"

            # 默認情況 - 空字符串
            return ""

        except Exception as e:
            self.logger.warning(f"Error generating placeholder content for '{placeholder}': {str(e)}")
            return ""

    def describe_functional_zones(self, functional_zones: Dict) -> str:
        """
        生成場景功能區域的描述，優化處理行人區域、人數統計和物品重複問題

        Args:
            functional_zones: 識別出的功能區域字典

        Returns:
            str: 功能區域描述
        """
        try:
            if not functional_zones:
                return ""

            # 處理不同類型的 functional_zones 參數
            if isinstance(functional_zones, list):
                # 如果是列表，轉換為字典格式
                zones_dict = {}
                for i, zone in enumerate(functional_zones):
                    if isinstance(zone, dict) and 'name' in zone:
                        zone_name = self._normalize_zone_name(zone['name'])
                    else:
                        zone_name = f"functional area {i+1}"
                    zones_dict[zone_name] = zone if isinstance(zone, dict) else {"description": str(zone)}
                functional_zones = zones_dict
            elif not isinstance(functional_zones, dict):
                return ""

            # 標準化所有區域鍵名，移除內部標識符格式
            normalized_zones = {}
            for zone_key, zone_data in functional_zones.items():
                normalized_key = self._normalize_zone_name(zone_key)
                normalized_zones[normalized_key] = zone_data
            functional_zones = normalized_zones

            # 計算場景中的總人數
            total_people_count = 0
            people_by_zone = {}

            # 計算每個區域的人數並累計總人數
            for zone_name, zone_info in functional_zones.items():
                if "objects" in zone_info:
                    zone_people_count = zone_info["objects"].count("person")
                    people_by_zone[zone_name] = zone_people_count
                    total_people_count += zone_people_count

            # 分類區域為行人區域和其他區域
            pedestrian_zones = []
            other_zones = []

            for zone_name, zone_info in functional_zones.items():
                # 檢查是否是行人相關區域
                if any(keyword in zone_name.lower() for keyword in ["pedestrian", "crossing", "people"]):
                    pedestrian_zones.append((zone_name, zone_info))
                else:
                    other_zones.append((zone_name, zone_info))

            # 獲取最重要的行人區域和其他區域
            main_pedestrian_zones = sorted(pedestrian_zones,
                                        key=lambda z: people_by_zone.get(z[0], 0),
                                        reverse=True)[:1]  # 最多1個主要行人區域

            top_other_zones = sorted(other_zones,
                                key=lambda z: len(z[1].get("objects", [])),
                                reverse=True)[:2]  # 最多2個其他區域

            # 合併區域
            top_zones = main_pedestrian_zones + top_other_zones

            if not top_zones:
                return ""

            # 生成匯總描述
            summary = ""
            max_mentioned_people = 0  # 追蹤已經提到的最大人數

            # 如果總人數顯著且還沒在主描述中提到，添加總人數描述
            if total_people_count > 5:
                summary = f"The scene contains a significant number of pedestrians ({total_people_count} people). "
                max_mentioned_people = total_people_count  # 更新已提到的最大人數

            # 處理每個區域的描述，確保人數信息的一致性
            processed_zones = []

            for zone_name, zone_info in top_zones:
                zone_desc = zone_info.get("description", "a functional zone")
                zone_people_count = people_by_zone.get(zone_name, 0)

                # 檢查描述中是否包含人數資訊
                contains_people_info = "with" in zone_desc and ("person" in zone_desc.lower() or "people" in zone_desc.lower())

                # 如果描述包含人數信息，且人數較小（小於已提到的最大人數），則修改描述
                if contains_people_info and zone_people_count < max_mentioned_people:
                    parts = zone_desc.split("with")
                    if len(parts) > 1:
                        # 移除人數部分
                        zone_desc = parts[0].strip() + " area"

                processed_zones.append((zone_name, {"description": zone_desc}))

            # 根據處理後的區域數量生成最終描述
            final_desc = ""

            if len(processed_zones) == 1:
                _, zone_info = processed_zones[0]
                zone_desc = zone_info["description"]
                final_desc = summary + f"The scene includes {zone_desc}."
            elif len(processed_zones) == 2:
                _, zone1_info = processed_zones[0]
                _, zone2_info = processed_zones[1]
                zone1_desc = zone1_info["description"]
                zone2_desc = zone2_info["description"]
                final_desc = summary + f"The scene is divided into two main areas: {zone1_desc} and {zone2_desc}."
            else:
                zones_desc = ["The scene contains multiple functional areas including"]
                zone_descriptions = [z[1]["description"] for z in processed_zones]

                # 格式化最終的多區域描述
                if len(zone_descriptions) == 3:
                    formatted_desc = f"{zone_descriptions[0]}, {zone_descriptions[1]}, and {zone_descriptions[2]}"
                else:
                    formatted_desc = ", ".join(zone_descriptions[:-1]) + f", and {zone_descriptions[-1]}"

                final_desc = summary + f"{zones_desc[0]} {formatted_desc}."

            return self.optimize_object_description(final_desc)

        except Exception as e:
            self.logger.warning(f"Error describing functional zones: {str(e)}")
            return ""

    def _normalize_zone_name(self, zone_name: str) -> str:
        """
        將內部區域鍵名標準化為自然語言描述

        Args:
            zone_name: 原始區域名稱

        Returns:
            str: 標準化後的區域名稱
        """
        try:
            if not zone_name or not isinstance(zone_name, str):
                return "functional area"

            # 移除數字後綴（如 crossing_zone_1 -> crossing_zone）
            import re
            base_name = re.sub(r'_\d+$', '', zone_name)

            # 將下劃線替換為空格
            normalized = base_name.replace('_', ' ')

            # 標準化常見的區域類型名稱
            zone_type_mapping = {
                'crossing zone': 'pedestrian crossing area',
                'vehicle zone': 'vehicle movement area',
                'pedestrian zone': 'pedestrian activity area',
                'traffic zone': 'traffic flow area',
                'waiting zone': 'waiting area',
                'seating zone': 'seating area',
                'dining zone': 'dining area',
                'furniture zone': 'furniture arrangement area',
                'electronics zone': 'electronics area',
                'people zone': 'social activity area',
                'functional area': 'activity area'
            }

            # 檢查是否有對應的標準化名稱
            for pattern, replacement in zone_type_mapping.items():
                if pattern in normalized.lower():
                    return replacement

            # 如果沒有特定映射，使用通用格式
            if 'zone' in normalized.lower():
                normalized = normalized.replace('zone', 'area')
            elif not any(keyword in normalized.lower() for keyword in ['area', 'space', 'region']):
                normalized += ' area'

            return normalized.strip()

        except Exception as e:
            self.logger.warning(f"Error normalizing zone name '{zone_name}': {str(e)}")
            return "activity area"

    def get_configuration(self) -> Dict[str, Any]:
        """
        獲取當前配置參數

        Returns:
            Dict[str, Any]: 配置參數字典
        """
        return {
            "min_prominence_score": self.min_prominence_score,
            "max_categories_to_return": self.max_categories_to_return,
            "max_total_objects": self.max_total_objects,
            "confidence_threshold_for_description": self.confidence_threshold_for_description
        }

    def update_configuration(self, **kwargs):
        """
        更新配置參數

        Args:
            **kwargs: 要更新的配置參數
        """
        try:
            for key, value in kwargs.items():
                if hasattr(self, key):
                    old_value = getattr(self, key)
                    setattr(self, key, value)
                    self.logger.info(f"Updated {key}: {old_value} -> {value}")
                else:
                    self.logger.warning(f"Unknown configuration parameter: {key}")

        except Exception as e:
            self.logger.error(f"Error updating configuration: {str(e)}")
            raise ObjectDescriptionError(f"Failed to update configuration: {str(e)}") from e
