import logging
from typing import Dict, List, Tuple, Optional, Any

class ObjectGroupProcessor:
    """
    物件組處理器 - 專門處理物件分組、排序和子句生成的邏輯
    負責物件按類別分組、重複物件檢測移除、物件組優先級排序以及描述子句的生成
    """

    def __init__(self, confidence_threshold_for_description: float = 0.25,
                 spatial_handler: Optional[Any] = None,
                 text_optimizer: Optional[Any] = None):
        """
        初始化物件組處理器

        Args:
            confidence_threshold_for_description: 用於描述的置信度閾值
            spatial_handler: 空間位置處理器實例
            text_optimizer: 文本優化器實例
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.confidence_threshold_for_description = confidence_threshold_for_description
        self.spatial_handler = spatial_handler
        self.text_optimizer = text_optimizer

    def group_objects_by_class(self, confident_objects: List[Dict],
                              object_statistics: Optional[Dict]) -> Dict[str, List[Dict]]:
        """
        按類別分組物件

        Args:
            confident_objects: 置信度過濾後的物件
            object_statistics: 物件統計信息

        Returns:
            Dict[str, List[Dict]]: 按類別分組的物件
        """
        objects_by_class = {}

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
                        matching_objects = [obj for obj in confident_objects
                                          if obj.get("class_name") == class_name and obj.get("confidence", 0) >= dynamic_threshold]

                    if matching_objects:
                        actual_count = min(stats["count"], len(matching_objects))
                        objects_by_class[class_name] = matching_objects[:actual_count]

                        # Debug logging for specific classes
                        if class_name in ["car", "traffic light", "person", "handbag"]:
                            print(f"DEBUG: Before spatial deduplication:")
                            print(f"DEBUG: {class_name}: {len(objects_by_class[class_name])} objects before dedup")
        else:
            # 備用邏輯，同樣使用動態閾值
            for obj in confident_objects:
                name = obj.get("class_name", "unknown object")
                if name == "unknown object" or not name:
                    continue
                if name not in objects_by_class:
                    objects_by_class[name] = []
                objects_by_class[name].append(obj)

        return objects_by_class

    def remove_duplicate_objects(self, objects_by_class: Dict[str, List[Dict]]) -> Dict[str, List[Dict]]:
        """
        移除重複物件

        Args:
            objects_by_class: 按類別分組的物件

        Returns:
            Dict[str, List[Dict]]: 去重後的物件
        """
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

        # Debug logging after deduplication
        for class_name in ["car", "traffic light", "person", "handbag"]:
            if class_name in deduplicated_objects_by_class:
                print(f"DEBUG: After spatial deduplication:")
                print(f"DEBUG: {class_name}: {len(deduplicated_objects_by_class[class_name])} objects after dedup")

        return deduplicated_objects_by_class

    def sort_object_groups(self, objects_by_class: Dict[str, List[Dict]]) -> List[Tuple[str, List[Dict]]]:
        """
        排序物件組

        Args:
            objects_by_class: 按類別分組的物件

        Returns:
            List[Tuple[str, List[Dict]]]: 排序後的物件組
        """
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

        return sorted(objects_by_class.items(), key=sort_key_object_groups)

    def generate_object_clauses(self, sorted_object_groups: List[Tuple[str, List[Dict]]],
                               object_statistics: Optional[Dict],
                               scene_type: str,
                               image_width: Optional[int],
                               image_height: Optional[int],
                               region_analyzer: Optional[Any] = None) -> List[str]:
        """
        生成物件描述子句

        Args:
            sorted_object_groups: 排序後的物件組
            object_statistics: 物件統計信息
            scene_type: 場景類型
            image_width: 圖像寬度
            image_height: 圖像高度
            region_analyzer: 區域分析器實例

        Returns:
            List[str]: 物件描述子句列表
        """
        object_clauses = []

        for class_name, group_of_objects in sorted_object_groups:
            count = len(group_of_objects)

            # Debug logging for final count
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
            location_description_suffix = self._generate_location_description(
                group_of_objects, count, image_width, image_height, region_analyzer
            )

            # 首字母大寫
            formatted_name_capitalized = formatted_name_with_exact_count[0].upper() + formatted_name_with_exact_count[1:]
            object_clauses.append(f"{formatted_name_capitalized} {location_description_suffix}")

        return object_clauses

    def format_object_clauses(self, object_clauses: List[str]) -> str:
        """
        格式化物件描述子句

        Args:
            object_clauses: 物件描述子句列表

        Returns:
            str: 格式化後的描述
        """
        if not object_clauses:
            return "No common objects were confidently identified for detailed description."

        # 處理第一個子句
        first_clause = object_clauses.pop(0)
        result = first_clause + "."

        # 處理剩餘子句
        if object_clauses:
            result += " The scene features:"
            joined_object_clauses = ". ".join(object_clauses)
            if joined_object_clauses and not joined_object_clauses.endswith("."):
                joined_object_clauses += "."
            result += " " + joined_object_clauses

        return result

    def _generate_location_description(self, group_of_objects: List[Dict], count: int,
                                     image_width: Optional[int], image_height: Optional[int],
                                     region_analyzer: Optional[Any] = None) -> str:
        """
        生成位置描述

        Args:
            group_of_objects: 物件組
            count: 物件數量
            image_width: 圖像寬度
            image_height: 圖像高度
            region_analyzer: 區域分析器實例

        Returns:
            str: 位置描述
        """
        if count == 1:
            if self.spatial_handler:
                spatial_desc = self.spatial_handler.generate_spatial_description(
                    group_of_objects[0], image_width, image_height, region_analyzer
                )
            else:
                spatial_desc = self._get_spatial_description_phrase(group_of_objects[0].get("region", ""))

            if spatial_desc:
                return f"is {spatial_desc}"
            else:
                distinct_regions = sorted(list(set(obj.get("region", "") for obj in group_of_objects if obj.get("region"))))
                valid_regions = [r for r in distinct_regions if r and r != "unknown" and r.strip()]
                if not valid_regions:
                    return "is positioned in the scene"
                elif len(valid_regions) == 1:
                    spatial_desc = self._get_spatial_description_phrase(valid_regions[0])
                    return f"is primarily {spatial_desc}" if spatial_desc else "is positioned in the scene"
                elif len(valid_regions) == 2:
                    clean_region1 = valid_regions[0].replace('_', ' ')
                    clean_region2 = valid_regions[1].replace('_', ' ')
                    return f"is mainly across the {clean_region1} and {clean_region2} areas"
                else:
                    return "is distributed in various parts of the scene"
        else:
            distinct_regions = sorted(list(set(obj.get("region", "") for obj in group_of_objects if obj.get("region"))))
            valid_regions = [r for r in distinct_regions if r and r != "unknown" and r.strip()]
            if not valid_regions:
                return "are visible in the scene"
            elif len(valid_regions) == 1:
                clean_region = valid_regions[0].replace('_', ' ')
                return f"are primarily in the {clean_region} area"
            elif len(valid_regions) == 2:
                clean_region1 = valid_regions[0].replace('_', ' ')
                clean_region2 = valid_regions[1].replace('_', ' ')
                return f"are mainly across the {clean_region1} and {clean_region2} areas"
            else:
                return "are distributed in various parts of the scene"

    def _get_spatial_description_phrase(self, region: str) -> str:
        """
        獲取空間描述短語的備用方法

        Args:
            region: 區域字符串

        Returns:
            str: 空間描述短語
        """
        if not region or region == "unknown":
            return ""

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

        return region_map.get(clean_region, "")

    def _normalize_object_class_name(self, class_name: str) -> str:
        """
        標準化物件類別名稱

        Args:
            class_name: 原始類別名稱

        Returns:
            str: 標準化後的類別名稱
        """
        if self.text_optimizer:
            return self.text_optimizer.normalize_object_class_name(class_name)
        else:
            # 備用標準化邏輯
            if not class_name or not isinstance(class_name, str):
                return "object"

            # 簡單的標準化處理
            normalized = class_name.replace('_', ' ').strip().lower()
            return normalized

    def _format_object_count_description(self, class_name: str, count: int,
                                       scene_type: Optional[str] = None,
                                       detected_objects: Optional[List[Dict]] = None,
                                       avg_confidence: float = 0.0) -> str:
        """
        格式化物件數量描述

        Args:
            class_name: 標準化後的類別名稱
            count: 物件數量
            scene_type: 場景類型
            detected_objects: 該類型的所有檢測物件
            avg_confidence: 平均檢測置信度

        Returns:
            str: 完整的格式化數量描述
        """
        if self.text_optimizer:
            return self.text_optimizer.format_object_count_description(
                class_name, count, scene_type, detected_objects, avg_confidence
            )
        else:
            # 備用格式化邏輯
            if count <= 0:
                return ""
            elif count == 1:
                article = "an" if class_name[0].lower() in 'aeiou' else "a"
                return f"{article} {class_name}"
            else:
                # 簡單的複數處理
                plural_form = class_name + "s" if not class_name.endswith("s") else class_name

                number_words = {
                    2: "two", 3: "three", 4: "four", 5: "five", 6: "six",
                    7: "seven", 8: "eight", 9: "nine", 10: "ten",
                    11: "eleven", 12: "twelve"
                }

                if count in number_words:
                    return f"{number_words[count]} {plural_form}"
                elif count <= 20:
                    return f"several {plural_form}"
                else:
                    return f"numerous {plural_form}"
