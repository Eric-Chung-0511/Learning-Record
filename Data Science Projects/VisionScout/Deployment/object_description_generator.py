import re
import logging
import traceback
from typing import Dict, List, Tuple, Optional, Any
import numpy as np

from prominence_calculator import ProminenceCalculator
from spatial_location_handler import SpatialLocationHandler
from text_optimizer import TextOptimizer
from object_group_processor import ObjectGroupProcessor

class ObjectDescriptionError(Exception):
    """物件描述生成過程中的自定義異常"""
    pass


class ObjectDescriptionGenerator:
    """
    物件描述生成器 - 負責將檢測到的物件轉換為自然語言描述
    匯總於EnhancedSceneDescriber

    該類別處理物件相關的所有描述生成邏輯，包括重要物件的辨識、
    空間位置描述、物件列表格式化以及描述文本的優化。

    作為 Facade 模式的實現，協調四個專門的子組件來完成複雜的描述生成任務。
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
            region_analyzer: 可選的RegionAnalyzer實例
        """
        self.logger = logging.getLogger(self.__class__.__name__)

        self.min_prominence_score = min_prominence_score
        self.max_categories_to_return = max_categories_to_return
        self.max_total_objects = max_total_objects
        self.confidence_threshold_for_description = confidence_threshold_for_description
        self.region_analyzer = region_analyzer

        # 初始化子組件
        self.prominence_calculator = ProminenceCalculator(
            min_prominence_score=self.min_prominence_score
        )

        self.spatial_handler = SpatialLocationHandler(
            region_analyzer=self.region_analyzer
        )

        self.text_optimizer = TextOptimizer()

        self.object_group_processor = ObjectGroupProcessor(
            confidence_threshold_for_description=self.confidence_threshold_for_description,
            spatial_handler=self.spatial_handler,
            text_optimizer=self.text_optimizer
        )

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
        return self.prominence_calculator.filter_prominent_objects(
            detected_objects=detected_objects,
            min_prominence_score=min_prominence_score,
            max_categories_to_return=max_categories_to_return
        )

    def set_region_analyzer(self, region_analyzer: Any) -> None:
        """
        設置RegionAnalyzer，用於標準化空間描述生成

        Args:
            region_analyzer: RegionAnalyzer實例
        """
        try:
            self.region_analyzer = region_analyzer
            self.spatial_handler.set_region_analyzer(region_analyzer)
            self.logger.info("RegionAnalyzer instance set for ObjectDescriptionGenerator")
        except Exception as e:
            self.logger.warning(f"Error setting RegionAnalyzer: {str(e)}")

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
        return self.text_optimizer.format_object_list_for_description(
            objects=objects,
            use_indefinite_article_for_one=use_indefinite_article_for_one,
            count_threshold_for_generalization=count_threshold_for_generalization,
            max_types_to_list=max_types_to_list
        )

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
        return self.spatial_handler.generate_spatial_description(
            obj=obj,
            image_width=image_width,
            image_height=image_height,
            region_analyzer=region_analyzer
        )

    def optimize_object_description(self, description: str) -> str:
        """
        優化物件描述文本，消除多餘重複並改善表達流暢度

        Args:
            description: 原始的場景描述文本，可能包含重複或冗餘的表達

        Returns:
            str: 經過優化清理的描述文本，如果處理失敗則返回原始文本
        """
        return self.text_optimizer.optimize_object_description(description)

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
            scene_type = places365_info.get("scene", "") if places365_info else ""

            self.logger.debug(f"Generating dynamic description for {len(detected_objects)} objects, "
                            f"viewpoint: {viewpoint}, lighting: {lighting_info is not None}")

            # 1. 整體氛圍（照明和視角）- 移除室內外標籤
            ambiance_parts = []
            if lighting_info:
                time_of_day = lighting_info.get("time_of_day", "unknown lighting")
                is_indoor = lighting_info.get("is_indoor")

                # 直接描述照明條件，不加入室內外標籤
                readable_lighting = f"{time_of_day.replace('_', ' ')} lighting conditions"

                # 根據室內外環境調整描述但不直接標明
                if is_indoor is True:
                    ambiance_statement = f"The scene features {readable_lighting} characteristic of an interior space."
                elif is_indoor is False:
                    ambiance_statement = f"The scene displays {readable_lighting} typical of an outdoor environment."
                else:
                    ambiance_statement = f"The scene presents {readable_lighting}."

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
                    # 使用 ObjectGroupProcessor 處理物件分組和排序
                    objects_by_class = self.object_group_processor.group_objects_by_class(
                        confident_objects, object_statistics
                    )

                    if not objects_by_class:
                        description_segments.append("No common objects were confidently identified for detailed description.")
                    else:
                        # 移除重複物件
                        deduplicated_objects_by_class = self.object_group_processor.remove_duplicate_objects(
                            objects_by_class
                        )

                        # 排序物件組
                        sorted_object_groups = self.object_group_processor.sort_object_groups(
                            deduplicated_objects_by_class
                        )

                        # 生成物件描述子句
                        object_clauses = self.object_group_processor.generate_object_clauses(
                            sorted_object_groups, object_statistics, scene_type,
                            image_width, image_height, self.region_analyzer
                        )

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
            raw_description = self.text_optimizer.remove_repetitive_descriptors(raw_description)

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
                        normalized_name = self.text_optimizer.normalize_object_class_name(raw_name)
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

                    # 同步更新子組件的配置
                    if key == "min_prominence_score" and hasattr(self, 'prominence_calculator'):
                        self.prominence_calculator.min_prominence_score = value
                    elif key == "confidence_threshold_for_description" and hasattr(self, 'object_group_processor'):
                        self.object_group_processor.confidence_threshold_for_description = value

                else:
                    self.logger.warning(f"Unknown configuration parameter: {key}")

        except Exception as e:
            self.logger.error(f"Error updating configuration: {str(e)}")
            raise ObjectDescriptionError(f"Failed to update configuration: {str(e)}") from e
