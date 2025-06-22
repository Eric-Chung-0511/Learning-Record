
import logging
import traceback
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

class FunctionalZoneDetector:
    """
    負責基於物件關聯性的功能區域識別
    處理物件組合分析和描述性區域命名
    """

    def __init__(self):
        """初始化功能區域檢測器"""
        try:
            logger.info("FunctionalZoneDetector initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize FunctionalZoneDetector: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def identify_primary_functional_area(self, detected_objects: List[Dict]) -> Dict:
        """
        識別主要功能區域，基於最強的物件關聯性組合
        採用通用邏輯處理各種室內場景

        Args:
            detected_objects: 檢測到的物件列表

        Returns:
            主要功能區域字典或None
        """
        try:
            # 用餐區域檢測（桌椅組合）
            dining_area = self.detect_functional_combination(
                detected_objects,
                primary_objects=[60],  # dining table
                supporting_objects=[56, 40, 41, 42, 43],  # chair, wine glass, cup, fork, knife
                min_supporting=2,
                description_template="Dining area with table and seating arrangement"
            )
            if dining_area:
                return dining_area

            # 休息區域檢測（沙發電視組合或床）
            seating_area = self.detect_functional_combination(
                detected_objects,
                primary_objects=[57, 59],  # sofa, bed
                supporting_objects=[62, 58, 56],  # tv, potted plant, chair
                min_supporting=1,
                description_template="Seating and relaxation area"
            )
            if seating_area:
                return seating_area

            # 工作區域檢測（電子設備與家具組合）
            work_area = self.detect_functional_combination(
                detected_objects,
                primary_objects=[63, 66],  # laptop, keyboard
                supporting_objects=[60, 56, 64],  # dining table, chair, mouse
                min_supporting=2,
                description_template="Workspace area with electronics and furniture"
            )
            if work_area:
                return work_area

            return None

        except Exception as e:
            logger.error(f"Error identifying primary functional area: {str(e)}")
            logger.error(traceback.format_exc())
            return None

    def identify_secondary_functional_area(self, detected_objects: List[Dict], existing_zones: Dict) -> Dict:
        """
        識別次要功能區域，避免與主要區域重疊

        Args:
            detected_objects: 檢測到的物件列表
            existing_zones: 已存在的功能區域

        Returns:
            次要功能區域字典或None
        """
        try:
            # 獲取已使用的區域
            used_regions = set(zone.get("region") for zone in existing_zones.values())

            # 裝飾區域檢測（植物集中區域）
            decorative_area = self.detect_functional_combination(
                detected_objects,
                primary_objects=[58],  # potted plant
                supporting_objects=[75],  # vase
                min_supporting=0,
                min_primary=3,  # 至少需要3個植物
                description_template="Decorative area with plants and ornamental items",
                exclude_regions=used_regions
            )
            if decorative_area:
                return decorative_area

            # 儲存區域檢測（廚房電器組合）
            storage_area = self.detect_functional_combination(
                detected_objects,
                primary_objects=[72, 68, 69],  # refrigerator, microwave, oven
                supporting_objects=[71],  # sink
                min_supporting=0,
                min_primary=2,
                description_template="Kitchen appliance and storage area",
                exclude_regions=used_regions
            )
            if storage_area:
                return storage_area

            return None

        except Exception as e:
            logger.error(f"Error identifying secondary functional area: {str(e)}")
            logger.error(traceback.format_exc())
            return None

    def detect_functional_combination(self, detected_objects: List[Dict], primary_objects: List[int],
                                    supporting_objects: List[int], min_supporting: int,
                                    description_template: str, min_primary: int = 1,
                                    exclude_regions: set = None) -> Dict:
        """
        通用的功能組合檢測方法
        基於主要物件和支持物件的組合判斷功能區域

        Args:
            detected_objects: 檢測到的物件列表
            primary_objects: 主要物件的class_id列表
            supporting_objects: 支持物件的class_id列表
            min_supporting: 最少需要的支持物件數量
            description_template: 描述模板
            min_primary: 最少需要的主要物件數量
            exclude_regions: 需要排除的區域集合

        Returns:
            功能區域資訊字典，如果不符合條件則返回None
        """
        try:
            if exclude_regions is None:
                exclude_regions = set()

            # 收集主要物件
            primary_objs = [obj for obj in detected_objects
                        if obj.get("class_id") in primary_objects and obj.get("confidence", 0) >= 0.4]

            # 收集支持物件
            supporting_objs = [obj for obj in detected_objects
                            if obj.get("class_id") in supporting_objects and obj.get("confidence", 0) >= 0.4]

            # 檢查是否滿足最少數量要求
            if len(primary_objs) < min_primary or len(supporting_objs) < min_supporting:
                return None

            # 按區域組織物件
            region_combinations = {}
            all_relevant_objs = primary_objs + supporting_objs

            for obj in all_relevant_objs:
                region = obj.get("region")

                # 排除指定區域
                if region in exclude_regions:
                    continue

                if region not in region_combinations:
                    region_combinations[region] = {"primary": [], "supporting": [], "all": []}

                region_combinations[region]["all"].append(obj)

                if obj.get("class_id") in primary_objects:
                    region_combinations[region]["primary"].append(obj)
                else:
                    region_combinations[region]["supporting"].append(obj)

            # 找到最佳區域組合
            best_region = None
            best_score = 0

            for region, objs in region_combinations.items():
                # 計算該區域的評分
                primary_count = len(objs["primary"])
                supporting_count = len(objs["supporting"])

                # 必須滿足最低要求
                if primary_count < min_primary or supporting_count < min_supporting:
                    continue

                # 計算組合評分（主要物件權重較高）
                score = primary_count * 2 + supporting_count

                if score > best_score:
                    best_score = score
                    best_region = region

            if best_region is None:
                return None

            best_combination = region_combinations[best_region]
            all_objects = [obj["class_name"] for obj in best_combination["all"]]

            return {
                "region": best_region,
                "objects": all_objects,
                "description": description_template
            }

        except Exception as e:
            logger.error(f"Error detecting functional combination: {str(e)}")
            logger.error(traceback.format_exc())
            return None

    def generate_descriptive_zone_key_from_data(self, zone_data: Dict, priority_level: str) -> str:
        """
        基於區域與物品名產生一個比較有描述性的區域

        Args:
            zone_data: 區域數據字典
            priority_level: 優先級別（primary/secondary）

        Returns:
            str: 描述性區域鍵名
        """
        try:
            objects = zone_data.get("objects", [])
            region = zone_data.get("region", "")
            description = zone_data.get("description", "")

            # 確保只有在明確檢測到廚房設備時才產生 kitchen area
            kitchen_objects = ["refrigerator", "microwave", "oven", "sink", "dishwasher", "stove"]
            explicit_kitchen_detected = any(
                any(kitchen_item in obj.lower() for kitchen_item in kitchen_objects)
                for obj in objects
            )

            # 基於物件內容確定功能類型（保持原有順序，但加強廚房確認, 因為與dining room混淆）
            if any("dining" in obj.lower() or "table" in obj.lower() for obj in objects):
                base_name = "dining area"
            elif any("chair" in obj.lower() or "sofa" in obj.lower() for obj in objects):
                base_name = "seating area"
            elif any("bed" in obj.lower() for obj in objects):
                base_name = "sleeping area"
            elif any("laptop" in obj.lower() or "keyboard" in obj.lower() for obj in objects):
                base_name = "workspace area"
            elif any("plant" in obj.lower() or "vase" in obj.lower() for obj in objects):
                base_name = "decorative area"
            elif explicit_kitchen_detected:
                # 只有在明確檢測到廚房設備時才使用 kitchen area
                base_name = "kitchen area"
            else:
                # 基於描述內容推斷，但避免不當的 kitchen area 判斷
                if "dining" in description.lower() and any("table" in obj.lower() for obj in objects):
                    # 只有當描述中提到 dining 且確實有桌子時才使用 dining area
                    base_name = "dining area"
                elif "seating" in description.lower() or "relaxation" in description.lower():
                    base_name = "seating area"
                elif "work" in description.lower() and any("laptop" in obj.lower() or "keyboard" in obj.lower() for obj in objects):
                    # 只有當描述中提到 work 且確實有工作設備時才使用 workspace area
                    base_name = "workspace area"
                elif "decorative" in description.lower():
                    base_name = "decorative area"
                else:
                    # 根據主要物件類型決定預設區域類型，避免使用 kitchen area
                    if objects:
                        # 根據最常見的物件類型決定區域名稱
                        object_counts = {}
                        for obj in objects:
                            obj_lower = obj.lower()
                            if "chair" in obj_lower:
                                object_counts["seating"] = object_counts.get("seating", 0) + 1
                            elif "table" in obj_lower:
                                object_counts["dining"] = object_counts.get("dining", 0) + 1
                            elif "person" in obj_lower:
                                object_counts["activity"] = object_counts.get("activity", 0) + 1
                            else:
                                object_counts["general"] = object_counts.get("general", 0) + 1

                        # 選擇最常見的類型
                        if object_counts:
                            most_common = max(object_counts, key=object_counts.get)
                            if most_common == "seating":
                                base_name = "seating area"
                            elif most_common == "dining":
                                base_name = "dining area"
                            elif most_common == "activity":
                                base_name = "activity area"
                            else:
                                base_name = "functional area"
                        else:
                            base_name = "functional area"
                    else:
                        base_name = "functional area"

            # 為次要區域添加位置標識以區分
            if priority_level == "secondary" and region:
                spatial_context = self.get_spatial_context_description(region)
                if spatial_context:
                    return f"{spatial_context} {base_name}"

            return base_name

        except Exception as e:
            logger.warning(f"Error generating descriptive zone key: {str(e)}")
            return "activity area"

    def get_spatial_context_description(self, region: str) -> str:
        """
        獲取空間上下文描述

        Args:
            region: 區域位置標識

        Returns:
            str: 空間上下文描述
        """
        try:
            spatial_mapping = {
                "top_left": "upper left",
                "top_center": "upper",
                "top_right": "upper right",
                "middle_left": "left side",
                "middle_center": "central",
                "middle_right": "right side",
                "bottom_left": "lower left",
                "bottom_center": "lower",
                "bottom_right": "lower right"
            }

            return spatial_mapping.get(region, "")

        except Exception as e:
            logger.warning(f"Error getting spatial context for region '{region}': {str(e)}")
            return ""
