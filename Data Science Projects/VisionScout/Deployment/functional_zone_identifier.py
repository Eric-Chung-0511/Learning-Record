
import logging
import traceback
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

class FunctionalZoneIdentifier:
    """
    作為功能區域辨識的主要窗口
    整合區域評估和場景特定的區域辨識邏輯，提供統一的功能區域辨識接口
    """

    def __init__(self, zone_evaluator=None, scene_zone_identifier=None, scene_viewpoint_analyzer=None):
        """
        初始化功能區域識別器

        Args:
            zone_evaluator: 區域評估器實例
            scene_zone_identifier: 場景區域辨識器實例
            scene_viewpoint_analyzer: 場景視角分析器
        """
        try:
            self.zone_evaluator = zone_evaluator
            self.scene_zone_identifier = scene_zone_identifier

            self.scene_viewpoint_analyzer = scene_viewpoint_analyzer
            self.viewpoint_detector = scene_viewpoint_analyzer

            logger.info("FunctionalZoneIdentifier initialized successfully with SceneViewpointAnalyzer")

        except Exception as e:
            logger.error(f"Failed to initialize FunctionalZoneIdentifier: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def identify_functional_zones(self, detected_objects: List[Dict], scene_type: str) -> Dict:
        """
        識別場景內的功能區域，具有針對不同視角和文化背景的改進檢測能力。
        如果偵測到 is_landmark=True 的物件，則優先直接呼叫 identify_landmark_zones 並回傳結果。
        """

        try:
            # 1. 如果沒有啟用地標功能，就先把所有有 is_landmark=True 的物件過濾掉
            if not getattr(self, 'enable_landmark', True):
                detected_objects = [obj for obj in detected_objects if not obj.get("is_landmark", False)]

            # 2. 只要檢測到任何 is_landmark=True 的物件，立即優先使用 identify_landmark_zones
            landmark_objects = [obj for obj in detected_objects if obj.get("is_landmark", False)]
            if landmark_objects and self.scene_zone_identifier:
                lm_zones = self.scene_zone_identifier.identify_landmark_zones(landmark_objects)
                return self._standardize_zone_keys_and_descriptions(lm_zones)

            # 3. city_street
            if scene_type in ["tourist_landmark", "natural_landmark", "historical_monument"]:
                scene_type = "city_street"

            # 4.  判斷與物件數量檢查
            if self.zone_evaluator:
                should_identify = self.zone_evaluator.evaluate_zone_identification_feasibility(
                    detected_objects, scene_type
                )
                if not should_identify:
                    logger.info(f"Zone identification not feasible for scene type '{scene_type}'")
                    return {}
            else:
                if len(detected_objects) < 2:
                    logger.info("Insufficient objects for zone identification")
                    return {}

            # 5. 建立 category_regions 
            category_regions = self._build_category_regions_mapping(detected_objects)
            zones = {}

            # 6. 檢測場景視角
            viewpoint_info = {"viewpoint": "eye_level"}
            if self.scene_viewpoint_analyzer:
                viewpoint_info = self.scene_viewpoint_analyzer.detect_scene_viewpoint(detected_objects)

            # 7. 根據不同 scene_type 使用各種自己的區域辨識
            if scene_type in ["living_room", "bedroom", "dining_area", "kitchen", "office_workspace", "meeting_room"]:
                if self.scene_zone_identifier:
                    raw_zones = self.scene_zone_identifier.identify_indoor_zones(
                        category_regions, detected_objects, scene_type
                    )
                    zones.update(self._standardize_zone_keys_and_descriptions(raw_zones))

            elif scene_type in ["city_street", "parking_lot", "park_area"]:
                if self.scene_zone_identifier:
                    raw_zones = self.scene_zone_identifier.identify_outdoor_general_zones(
                        category_regions, detected_objects, scene_type
                    )
                    zones.update(self._standardize_zone_keys_and_descriptions(raw_zones))

            elif "aerial" in scene_type or viewpoint_info.get("viewpoint") == "aerial":
                if self.scene_zone_identifier:
                    raw_zones = self.scene_zone_identifier.identify_aerial_view_zones(
                        category_regions, detected_objects, scene_type
                    )
                    zones.update(self._standardize_zone_keys_and_descriptions(raw_zones))

            elif "asian" in scene_type:
                if self.scene_zone_identifier:
                    asian_zones = self.scene_zone_identifier.identify_asian_cultural_zones(
                        category_regions, detected_objects, scene_type
                    )
                    zones.update(self._standardize_zone_keys_and_descriptions(asian_zones))

            elif scene_type == "urban_intersection":
                if self.scene_zone_identifier:
                    raw_zones = self.scene_zone_identifier.identify_intersection_zones(
                        category_regions, detected_objects, viewpoint_info.get("viewpoint")
                    )
                    zones.update(self._standardize_zone_keys_and_descriptions(raw_zones))
                    used_tl_count_per_region = {}
                    for zone_info in raw_zones.values():
                        obj_list = zone_info.get("objects", [])
                        if "traffic light" in obj_list:
                            rg = zone_info.get("region", "")
                            count_in_zone = obj_list.count("traffic light")
                            used_tl_count_per_region[rg] = used_tl_count_per_region.get(rg, 0) + count_in_zone

                    signal_regions = {}
                    for t in [obj for obj in detected_objects if obj.get("class_id") == 9]:
                        region = t.get("region", "")
                        signal_regions.setdefault(region, []).append(t)

                    for idx, (region, signals) in enumerate(signal_regions.items()):
                        total_in_region = len(signals)
                        used_in_region = used_tl_count_per_region.get(region, 0)
                        remaining_in_region = total_in_region - used_in_region

                        if remaining_in_region > 0:
                            direction = self._get_directional_description(region)
                            if direction and direction != "central":
                                zone_key = f"{direction} traffic control area"
                            else:
                                zone_key = "primary traffic control area" if idx == 0 else "auxiliary traffic control area"

                            if zone_key in zones:
                                suffix = 1
                                new_key = f"{zone_key} ({suffix})"
                                while new_key in zones:
                                    suffix += 1
                                    new_key = f"{zone_key} ({suffix})"
                                zone_key = new_key

                            zones[zone_key] = {
                                "region": region,
                                "objects": ["traffic light"] * remaining_in_region,
                                "description": f"Traffic control area with {remaining_in_region} traffic lights in {region}"
                            }

                    for region, signals in signal_regions.items():
                        used = used_tl_count_per_region.get(region, 0)
                        total = len(signals)
                        remaining = total - used
                        # print(f"[DEBUG] Region '{region}': Total TL = {total}, Used in crossing = {used}, Remaining = {remaining}")

            elif scene_type == "financial_district":
                if self.scene_zone_identifier:
                    fd_zones = self.scene_zone_identifier.identify_financial_district_zones(
                        category_regions, detected_objects
                    )
                    zones.update(self._standardize_zone_keys_and_descriptions(fd_zones))

            elif scene_type == "upscale_dining":
                if self.scene_zone_identifier:
                    ud_zones = self.scene_zone_identifier.identify_upscale_dining_zones(
                        category_regions, detected_objects
                    )
                    zones.update(self._standardize_zone_keys_and_descriptions(ud_zones))

            else:
                # 如果不是上述任何一種場景，就用「預設功能區」
                default_zones = self._identify_default_zones(category_regions, detected_objects)
                zones.update(self._standardize_zone_keys_and_descriptions(default_zones))

            # 8. 如果此時 zones 仍為空，就會變成 default → basic → fallback
            if not zones:
                default_zones = self._identify_default_zones(category_regions, detected_objects)
                if default_zones:
                    zones.update(self._standardize_zone_keys_and_descriptions(default_zones))
                else:
                    basic_zones = self._create_basic_zones_from_objects(detected_objects, scene_type)
                    zones.update(self._standardize_zone_keys_and_descriptions(basic_zones))

            # 通用 fallback：把所有還沒被列出的 (class_name, region) 通通補進去
            fallback_zones = self._generate_category_fallback_zones(detected_objects, zones)
            zones.update(fallback_zones)

            # Debug: 列印出各功能區的 traffic light 統計
            total_tl_in_zones = 0
            for zone_key, zone_info in zones.items():
                if isinstance(zone_info, dict):
                    sub_objs = zone_info.get("objects", [])
                else:
                    sub_objs = []
                t_in_zone = [obj for obj in sub_objs if obj == "traffic light"]
                # print(f"[DEBUG] identify_functional_zones - Zone '{zone_key}' has {len(t_in_zone)} traffic light(s).")
                total_tl_in_zones += len(t_in_zone)
            # print(f"[DEBUG] identify_functional_zones - Total traffic lights in zones: {total_tl_in_zones}")

            logger.info(f"Identified {len(zones)} functional zones for scene type '{scene_type}'")
            return zones

        except Exception as e:
            logger.error(f"Error identifying functional zones: {str(e)}")
            logger.error(traceback.format_exc())
            return {}

    def _standardize_zone_keys_and_descriptions(self, raw_zones: Dict) -> Dict:
        """
        標準化區域鍵名和描述，將內部標識符轉換為描述性名稱

        Args:
            raw_zones: 原始區域識別結果

        Returns:
            Dict: 標準化後的區域字典
        """
        try:
            standardized_zones = {}

            for zone_key, zone_data in raw_zones.items():
                # 生成描述性的區域鍵名
                descriptive_key = self._generate_descriptive_zone_key(zone_key, zone_data)

                # 確保區域描述也經過標準化
                if isinstance(zone_data, dict) and "description" in zone_data:
                    zone_data["description"] = self._enhance_zone_description(zone_data["description"], zone_data)

                standardized_zones[descriptive_key] = zone_data

            return standardized_zones

        except Exception as e:
            logger.error(f"Error standardizing zone keys and descriptions: {str(e)}")
            return raw_zones

    def _generate_descriptive_zone_key(self, original_key: str, zone_data: Dict) -> str:
        """
        基於區域內容生成描述性的鍵名
        核心修改：只要該區域內有任一個 'traffic light'，就優先回傳 'traffic control zone'，
        """
        try:
            objects = zone_data.get("objects", [])
            region = zone_data.get("region", "")

            # 優先檢查是否含有 traffic light 
            if any(obj == "traffic light" or "traffic light" in obj for obj in objects):
                return "traffic control zone"

            # 如果沒有 traffic light，才繼續分析「主要物件」順序
            primary_objects = self._analyze_primary_objects(objects)

            # 依序檢查人、車、家具、紅綠燈等
            if "person" in primary_objects:
                if len([o for o in objects if o == "person"]) > 1:
                    return "pedestrian activity area"
                else:
                    return "individual activity zone"
            elif any(vehicle in primary_objects for vehicle in ["car", "truck", "bus", "motorcycle"]):
                return "vehicle movement area"
            elif any(furniture in primary_objects for furniture in ["chair", "table", "sofa", "bed"]):
                return "furniture arrangement area"

            # 若上述都不符合，改用「基於位置」做 fallback
            position_descriptions = {
                "top_left": "upper left area",
                "top_center": "upper central area",
                "top_right": "upper right area",
                "middle_left": "left side area",
                "middle_center": "main crossing area",
                "middle_right": "right side area",
                "bottom_left": "lower left area",
                "bottom_center": "lower central area",
                "bottom_right": "lower right area"
            }
            if region in position_descriptions:
                return position_descriptions[region]

            # 再次檢查主要物件，給出另一種 fallback 命名
            if primary_objects:
                if "traffic light" in primary_objects:
                    return "traffic control zone"
                elif any(vehicle in primary_objects for vehicle in ["car", "truck", "bus"]):
                    return "vehicle movement area"
                elif "person" in primary_objects:
                    return "pedestrian activity area"

            # 最後最後的備用名稱
            return "activity area"

        except Exception as e:
            logger.warning(f"Error generating descriptive key for '{original_key}': {str(e)}")
            return "activity area"

    def _analyze_primary_objects(self, objects: List[str]) -> List[str]:
        """
        分析區域中的主要物件類型

        Args:
            objects: 物件名稱列表

        Returns:
            List[str]: 主要物件類型列表
        """
        try:
            # 計算物件出現頻率
            object_counts = {}
            for obj in objects:
                normalized_obj = obj.replace('_', ' ').lower().strip()
                object_counts[normalized_obj] = object_counts.get(normalized_obj, 0) + 1

            # 按出現頻率排序，返回前三個主要物件
            sorted_objects = sorted(object_counts.items(), key=lambda x: x[1], reverse=True)
            return [obj[0] for obj in sorted_objects[:3]]

        except Exception as e:
            logger.warning(f"Error analyzing primary objects: {str(e)}")
            return []

    def _enhance_zone_description(self, original_description: str, zone_data: Dict) -> str:
        """
        增強區域描述的自然性和完整性
        """
        try:
            if not original_description or not original_description.strip():
                return self._generate_fallback_description(zone_data)

            import re
            enhanced = original_description.strip()

            # 改善技術性表達為自然語言
            enhanced = re.sub(r'\bin central direction\b', 'in the center', enhanced)
            enhanced = re.sub(r'\bin west area\b', 'on the left side', enhanced)
            enhanced = re.sub(r'\bin east direction\b', 'on the right side', enhanced)
            enhanced = re.sub(r'\bnear traffic signals\b', 'near the traffic lights', enhanced)
            enhanced = re.sub(r'\bwith (\d+) (\w+)\b', r'where \1 \2 can be seen', enhanced)

            # 移除重複和冗餘表達
            enhanced = re.sub(r'\barea with.*?in.*?area\b', lambda m: m.group(0).split(' in ')[0], enhanced)
            enhanced = enhanced.replace('traffic area', 'area').replace('crossing area', 'crossing')

            # 標準化描述結構
            if enhanced.startswith('Pedestrian'):
                enhanced = re.sub(r'^Pedestrian crossing area', 'The main pedestrian crossing', enhanced)
            elif enhanced.startswith('Vehicle'):
                enhanced = re.sub(r'^Vehicle traffic area', 'The vehicle movement area', enhanced)
            elif enhanced.startswith('Traffic control'):
                enhanced = re.sub(r'^Traffic control area', 'Traffic management elements', enhanced)

            # 移除內部標識符格式
            enhanced = re.sub(r'\b\w+_\w+(?:_\w+)*\b', lambda m: m.group(0).replace('_', ' '), enhanced)

            # 確保描述的完整性
            if not enhanced.endswith('.'):
                enhanced += '.'

            # 改善描述的自然性
            enhanced = enhanced.replace('with with', 'with')
            enhanced = re.sub(r'\s{2,}', ' ', enhanced)

            return enhanced

        except Exception as e:
            logger.warning(f"Error enhancing zone description: {str(e)}")
            return original_description if original_description else "A functional area within the scene."

    def _generate_fallback_description(self, zone_data: Dict) -> str:
        """
        為缺少描述的區域生成備用描述

        Args:
            zone_data: 區域數據

        Returns:
            str: 備用描述
        """
        try:
            objects = zone_data.get("objects", [])
            region = zone_data.get("region", "")

            if objects:
                object_count = len(objects)
                unique_objects = list(set(objects))

                if object_count == 1:
                    return f"Area containing {unique_objects[0].replace('_', ' ')}."
                elif len(unique_objects) <= 3:
                    obj_list = ", ".join([obj.replace('_', ' ') for obj in unique_objects])
                    return f"Area featuring {obj_list}."
                else:
                    return f"Multi-functional area with {object_count} elements including various objects."

            return "Functional area within the scene."

        except Exception as e:
            logger.warning(f"Error generating fallback description: {str(e)}")
            return "Activity area."

    def _build_category_regions_mapping(self, detected_objects: List[Dict]) -> Dict:
        """
        建立物件按類別和區域的分組映射

        Args:
            detected_objects: 檢測到的物件列表

        Returns:
            按類別和區域分組的物件字典
        """
        try:
            category_regions = {}

            for obj in detected_objects:
                category = self._categorize_object(obj)
                if not category:
                    continue

                if category not in category_regions:
                    category_regions[category] = {}

                region = obj.get("region", "center")
                if region not in category_regions[category]:
                    category_regions[category][region] = []

                category_regions[category][region].append(obj)

            logger.debug(f"Built category regions mapping with {len(category_regions)} categories")
            return category_regions

        except Exception as e:
            logger.error(f"Error building category regions mapping: {str(e)}")
            logger.error(traceback.format_exc())
            return {}

    def _categorize_object(self, obj: Dict) -> str:
        """
        將檢測到的物件分類到功能類別中，用於區域識別

        Args:
            obj: 物件字典

        Returns:
            物件功能類別字串
        """
        try:
            class_id = obj.get("class_id", -1)
            class_name = obj.get("class_name", "").lower()

            # 使用現有的類別映射（如果可用）
            if hasattr(self, 'OBJECT_CATEGORIES') and self.OBJECT_CATEGORIES:
                for category, ids in self.OBJECT_CATEGORIES.items():
                    if class_id in ids:
                        return category

            # 基於COCO類別名稱的後備分類
            furniture_items = ["chair", "couch", "bed", "dining table", "toilet"]
            plant_items = ["potted plant"]
            electronic_items = ["tv", "laptop", "mouse", "remote", "keyboard", "cell phone"]
            vehicle_items = ["bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat"]
            person_items = ["person"]
            kitchen_items = ["bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
                            "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog",
                            "pizza", "donut", "cake", "refrigerator", "oven", "toaster", "sink", "microwave"]
            sports_items = ["frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
                        "baseball glove", "skateboard", "surfboard", "tennis racket"]
            personal_items = ["handbag", "tie", "suitcase", "umbrella", "backpack"]

            if any(item in class_name for item in furniture_items):
                return "furniture"
            elif any(item in class_name for item in plant_items):
                return "plant"
            elif any(item in class_name for item in electronic_items):
                return "electronics"
            elif any(item in class_name for item in vehicle_items):
                return "vehicle"
            elif any(item in class_name for item in person_items):
                return "person"
            elif any(item in class_name for item in kitchen_items):
                return "kitchen_items"
            elif any(item in class_name for item in sports_items):
                return "sports"
            elif any(item in class_name for item in personal_items):
                return "personal_items"
            else:
                return "misc"

        except Exception as e:
            logger.error(f"Error categorizing object: {str(e)}")
            logger.error(traceback.format_exc())
            return "misc"

    def _identify_default_zones(self, category_regions: Dict, detected_objects: List[Dict]) -> Dict:
        """
        當沒有匹配到特定場景類型時的一般功能區域識別

        Args:
            category_regions: 按類別和區域分組的物件字典
            detected_objects: 檢測到的物件列表

        Returns:
            預設功能區域字典
        """
        try:
            zones = {}

            # 按類別分組物件並找到主要集中區域
            for category, regions in category_regions.items():
                if not regions:
                    continue

                # 找到此類別中物件最多的區域
                main_region = max(regions.items(),
                            key=lambda x: len(x[1]),
                            default=(None, []))

                if main_region[0] is None or len(main_region[1]) < 2:
                    continue

                # 創建基於物件類別的區域
                zone_objects = [obj["class_name"] for obj in main_region[1]]

                # 如果物件太少，跳過
                if len(zone_objects) < 2:
                    continue

                # 根據類別創建區域名稱和描述
                if category == "furniture":
                    zones["furniture arrangement area"] = {
                        "region": main_region[0],
                        "objects": zone_objects,
                        "description": f"Furniture arrangement area featuring {self._format_object_list_naturally(zone_objects[:3])}"
                    }
                elif category == "electronics":
                    zones["electronics area"] = {
                        "region": main_region[0],
                        "objects": zone_objects,
                        "description": f"Electronics area containing {self._format_object_list_naturally(zone_objects[:3])}"
                    }
                elif category == "kitchen_items":
                    zones["dining_zone"] = {
                        "region": main_region[0],
                        "objects": zone_objects,
                        "description": f"Dining or food area with {', '.join(zone_objects[:3])}"
                    }
                elif category == "vehicle":
                    zones["vehicle_zone"] = {
                        "region": main_region[0],
                        "objects": zone_objects,
                        "description": f"Area with vehicles including {', '.join(zone_objects[:3])}"
                    }
                elif category == "personal_items":
                    zones["personal_items_zone"] = {
                        "region": main_region[0],
                        "objects": zone_objects,
                        "description": f"Area with personal items including {', '.join(zone_objects[:3])}"
                    }

            # 檢查人群聚集
            people_objs = [obj for obj in detected_objects if obj["class_id"] == 0]
            if len(people_objs) >= 2:
                people_regions = {}
                for obj in people_objs:
                    region = obj["region"]
                    if region not in people_regions:
                        people_regions[region] = []
                    people_regions[region].append(obj)

                if people_regions:
                    main_people_region = max(people_regions.items(),
                                        key=lambda x: len(x[1]),
                                        default=(None, []))

                    if main_people_region[0] is not None:
                        zones["people_zone"] = {
                            "region": main_people_region[0],
                            "objects": ["person"] * len(main_people_region[1]),
                            "description": f"Area with {len(main_people_region[1])} people"
                        }

            logger.debug(f"Identified {len(zones)} default zones")
            return zones

        except Exception as e:
            logger.error(f"Error identifying default zones: {str(e)}")
            logger.error(traceback.format_exc())
            return {}

    def _format_object_list_naturally(self, object_list: List[str]) -> str:
        """
        將物件列表格式化為自然語言表達

        Args:
            object_list: 物件名稱列表

        Returns:
            str: 自然語言格式的物件列表
        """
        try:
            if not object_list:
                return "various items"

            # 標準化物件名稱
            normalized_objects = []
            for obj in object_list:
                normalized = obj.replace('_', ' ').strip()
                if normalized:
                    normalized_objects.append(normalized)

            if not normalized_objects:
                return "various items"

            # 格式化列表
            if len(normalized_objects) == 1:
                return normalized_objects[0]
            elif len(normalized_objects) == 2:
                return f"{normalized_objects[0]} and {normalized_objects[1]}"
            else:
                return ", ".join(normalized_objects[:-1]) + f", and {normalized_objects[-1]}"

        except Exception as e:
            logger.warning(f"Error formatting object list naturally: {str(e)}")
            return "various items"

    def _create_basic_zones_from_objects(self, detected_objects: List[Dict], scene_type: str) -> Dict:
        """
        從個別高置信度物件創建基本功能區域
        這是標準區域識別失敗時的後備方案

        Args:
            detected_objects: 檢測到的物件列表
            scene_type: 場景類型

        Returns:
            基本區域字典
        """
        try:
            zones = {}

            # 專注於高置信度物件
            high_conf_objects = [obj for obj in detected_objects if obj.get("confidence", 0) >= 0.6]

            if not high_conf_objects:
                high_conf_objects = detected_objects  # 後備到所有物件

            # 基於個別重要物件創建區域
            processed_objects = set()  # 避免重複處理相同類型的物件

            for obj in high_conf_objects[:3]:  # 限制為前3個物件
                class_name = obj["class_name"]
                region = obj.get("region", "center")

                # 避免為同一類型物件創建多個區域
                if class_name in processed_objects:
                    continue
                processed_objects.add(class_name)

                # 基於物件類型創建描述性區域
                zone_description = self._get_basic_zone_description(class_name, scene_type)
                descriptive_key = self._generate_object_based_zone_key(class_name, region)

                if zone_description and descriptive_key:
                    zones[descriptive_key] = {
                        "region": region,
                        "objects": [class_name],
                        "description": zone_description
                    }

            logger.debug(f"Created {len(zones)} basic zones from high confidence objects")
            return zones

        except Exception as e:
            logger.error(f"Error creating basic zones from objects: {str(e)}")
            logger.error(traceback.format_exc())
            return {}

    def _generate_object_based_zone_key(self, class_name: str, region: str) -> str:
        """
        基於物件類型和位置生成描述性的區域鍵名

        Args:
            class_name: 物件類別名稱
            region: 區域位置

        Returns:
            str: 描述性區域鍵名
        """
        try:
            # 標準化物件名稱
            normalized_class = class_name.replace('_', ' ').lower().strip()

            # 物件類型對應的區域描述
            object_zone_mapping = {
                'person': 'activity area',
                'car': 'vehicle area',
                'truck': 'vehicle area',
                'bus': 'vehicle area',
                'motorcycle': 'vehicle area',
                'bicycle': 'cycling area',
                'traffic light': 'traffic control area',
                'chair': 'seating area',
                'sofa': 'seating area',
                'bed': 'rest area',
                'dining table': 'dining area',
                'tv': 'entertainment area',
                'laptop': 'workspace area',
                'potted plant': 'decorative area'
            }

            base_description = object_zone_mapping.get(normalized_class, f"{normalized_class} area")

            # 添加位置信息以提供更具體的描述
            position_modifiers = {
                'top_left': 'upper left',
                'top_center': 'upper central',
                'top_right': 'upper right',
                'middle_left': 'left side',
                'middle_center': 'central',
                'middle_right': 'right side',
                'bottom_left': 'lower left',
                'bottom_center': 'lower central',
                'bottom_right': 'lower right'
            }

            if region in position_modifiers:
                return f"{position_modifiers[region]} {base_description}"

            return base_description

        except Exception as e:
            logger.warning(f"Error generating object-based zone key for '{class_name}': {str(e)}")
            return "activity area"

    def _get_basic_zone_description(self, class_name: str, scene_type: str) -> str:
        """
        基於物件和場景類型生成基本區域描述

        Args:
            class_name: 物件類別名稱
            scene_type: 場景類型

        Returns:
            區域描述字串
        """
        try:
            # 物件特定描述
            descriptions = {
                "bed": "Sleeping and rest area",
                "sofa": "Seating and relaxation area",
                "chair": "Seating area",
                "dining table": "Dining and meal area",
                "tv": "Entertainment and media area",
                "laptop": "Work and computing area",
                "potted plant": "Decorative and green space area",
                "refrigerator": "Food storage and kitchen area",
                "car": "Vehicle and transportation area",
                "person": "Activity and social area"
            }

            return descriptions.get(class_name, f"Functional area with {class_name}")

        except Exception as e:
            logger.error(f"Error getting basic zone description for '{class_name}': {str(e)}")
            return f"Functional area with {class_name}"


    def _generate_category_fallback_zones(self, all_detected_objects: List[Dict], current_zones: Dict) -> Dict:
        """
        通用 fallback：針對 all_detected_objects 裡，每一個 (class_name, region) 組合是否已經
        在 current_zones 裡出現過。如果還沒，就為它們產生一個 fallback zone。
        """
        general_fallback = {
                0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus',
                6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant',
                11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat',
                16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear',
                22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag',
                27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard',
                32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove',
                36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle',
                40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl',
                46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli',
                51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair',
                57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet',
                62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard',
                67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink',
                72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors',
                77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'

        }

        # 1. 統計 current_zones 裡，已使用掉的 (class_name, region) 次數 
        used_count = {}
        for zone_info in current_zones.values():
            rg = zone_info.get("region", "")
            for obj_name in zone_info.get("objects", []):
                key = (obj_name, rg)
                used_count[key] = used_count.get(key, 0) + 1

        # 2. 統計 all_detected_objects 裡的 (class_name, region) 總次數 
        total_count = {}
        for obj in all_detected_objects:
            cname = obj.get("class_name", "")
            rg = obj.get("region", "")
            key = (cname, rg)
            total_count[key] = total_count.get(key, 0) + 1

        # 3. 把 default_classes 轉換成「class_name → fallback 區域 type」的對照表 
        category_to_fallback = {
            # 行人與交通工具
            "person":        "pedestrian area",
            "bicycle":       "vehicle movement area",
            "car":           "vehicle movement area",
            "motorcycle":    "vehicle movement area",
            "airplane":      "vehicle movement area",
            "bus":           "vehicle movement area",
            "train":         "vehicle movement area",
            "truck":         "vehicle movement area",
            "boat":          "vehicle movement area",
            "traffic light": "traffic control area",
            "fire hydrant":  "traffic control area",
            "stop sign":     "traffic control area",
            "parking meter": "traffic control area",
            "bench":         "public furniture area",

            # 動物類、鳥類
            "bird":          "animal area",
            "cat":           "animal area",
            "dog":           "animal area",
            "horse":         "animal area",
            "sheep":         "animal area",
            "cow":           "animal area",
            "elephant":      "animal area",
            "bear":          "animal area",
            "zebra":         "animal area",
            "giraffe":       "animal area",

            # 托運與行李
            "backpack":      "personal items area",
            "umbrella":      "personal items area",
            "handbag":       "personal items area",
            "tie":           "personal items area",
            "suitcase":      "personal items area",

            # 運動器材
            "frisbee":       "sports area",
            "skis":          "sports area",
            "snowboard":     "sports area",
            "sports ball":   "sports area",
            "kite":          "sports area",
            "baseball bat":  "sports area",
            "baseball glove":"sports area",
            "skateboard":    "sports area",
            "surfboard":     "sports area",
            "tennis racket": "sports area",

            # 廚房與食品（Kitchen）
            "bottle":        "kitchen area",
            "wine glass":    "kitchen area",
            "cup":           "kitchen area",
            "fork":          "kitchen area",
            "knife":         "kitchen area",
            "spoon":         "kitchen area",
            "bowl":          "kitchen area",
            "banana":        "kitchen area",
            "apple":         "kitchen area",
            "sandwich":      "kitchen area",
            "orange":        "kitchen area",
            "broccoli":      "kitchen area",
            "carrot":        "kitchen area",
            "hot dog":       "kitchen area",
            "pizza":         "kitchen area",
            "donut":         "kitchen area",
            "cake":          "kitchen area",
            "dining table":  "furniture arrangement area",
            "refrigerator":  "kitchen area",
            "oven":          "kitchen area",
            "microwave":     "kitchen area",
            "toaster":       "kitchen area",
            "sink":          "kitchen area",
            "book":          "miscellaneous area",
            "clock":         "miscellaneous area",
            "vase":          "decorative area",
            "scissors":      "miscellaneous area",
            "teddy bear":    "miscellaneous area",
            "hair drier":    "miscellaneous area",
            "toothbrush":    "miscellaneous area",

            # 電子產品
            "tv":            "electronics area",
            "laptop":        "electronics area",
            "mouse":         "electronics area",
            "remote":        "electronics area",
            "keyboard":      "electronics area",
            "cell phone":    "electronics area",

            # 家具類
            "chair":         "furniture arrangement area",
            "couch":         "furniture arrangement area",
            "bed":           "furniture arrangement area",
            "toilet":        "furniture arrangement area",

            # 植物（室內植物或戶外綠化）
            "potted plant":  "decorative area",
        }

        # 4. 計算缺少的 (class_name, region) 並建立 fallback zone 
        for (cname, rg), total in total_count.items():
            used = used_count.get((cname, rg), 0)
            missing = total - used
            if missing <= 0:
                continue  

            # (A) 決定這個 cname 在 fallback 裡屬於哪個大 class（zone_type）
            zone_type = category_to_fallback.get(cname, "miscellaneous area")

            # (B) 根據 region 與 zone_type 組合成 fallback_key
            fallback_key = f"{rg} {zone_type}"

            # (C) 如果名稱重複，就在後面加 (1),(2),… 避免掉衝突
            if fallback_key in current_zones or fallback_key in general_fallback:
                suffix = 1
                new_key = f"{fallback_key} ({suffix})"
                while new_key in current_zones or new_key in general_fallback:
                    suffix += 1
                    new_key = f"{fallback_key} ({suffix})"
                fallback_key = new_key

            # (D) 建立這支 fallback zone，objects 裡放 missing 個 cname
            general_fallback[fallback_key] = {
                "region": rg,
                "objects": [cname] * missing,
                "description": f"{missing} {cname}(s) placed in fallback {zone_type} for region {rg}"
            }

        return general_fallback
