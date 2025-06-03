
import logging
import traceback
import numpy as np
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

class SceneZoneIdentifier:
    """
    負責不同場景類型的區域識別邏輯
    專注於根據場景類型執行相應的功能區域識別策略
    """

    def __init__(self):
        """初始化場景區域辨識器"""
        try:
            logger.info("SceneZoneIdentifier initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize SceneZoneIdentifier: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def identify_indoor_zones(self, category_regions: Dict, detected_objects: List[Dict], scene_type: str) -> Dict:
        """
        平衡化的室內功能區域識別並標準化命名
        採用通用的物件關聯性分析，避免只針對特定場景

        Args:
            category_regions: 按類別和區域分組的物件字典
            detected_objects: 檢測到的物件列表
            scene_type: 場景類型

        Returns:
            識別出的室內功能區域字典，使用描述性鍵名
        """
        try:
            zones = {}

            # 主要功能區域（基於物件關聯性而非場景類型）
            primary_zone = self._identify_primary_functional_area(detected_objects)
            if primary_zone:
                # 基於區域內容生成描述性鍵名
                descriptive_key = self._generate_descriptive_zone_key_from_data(primary_zone, "primary")
                zones[descriptive_key] = primary_zone

            # 只有明確證據且物件數量足夠時創建次要功能區域
            if len(zones) >= 1 and len(detected_objects) >= 6:
                secondary_zone = self._identify_secondary_functional_area(detected_objects, zones)
                if secondary_zone:
                    # 基於區域內容生成描述性鍵名
                    descriptive_key = self._generate_descriptive_zone_key_from_data(secondary_zone, "secondary")
                    zones[descriptive_key] = secondary_zone

            logger.info(f"Identified {len(zones)} indoor zones for scene type '{scene_type}'")
            return zones

        except Exception as e:
            logger.error(f"Error identifying indoor zones: {str(e)}")
            logger.error(traceback.format_exc())
            return {}

    def _generate_descriptive_zone_key_from_data(self, zone_data: Dict, priority_level: str) -> str:
        """
        基於區域數據生成描述性鍵名

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

            # 基於物件內容確定功能類型
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
            elif any("refrigerator" in obj.lower() or "microwave" in obj.lower() for obj in objects):
                base_name = "kitchen area"
            else:
                # 基於描述內容推斷
                if "dining" in description.lower():
                    base_name = "dining area"
                elif "seating" in description.lower() or "relaxation" in description.lower():
                    base_name = "seating area"
                elif "work" in description.lower():
                    base_name = "workspace area"
                elif "decorative" in description.lower():
                    base_name = "decorative area"
                else:
                    base_name = "functional area"

            # 為次要區域添加位置標識以區分
            if priority_level == "secondary" and region:
                spatial_context = self._get_spatial_context_description(region)
                if spatial_context:
                    return f"{spatial_context} {base_name}"

            return base_name

        except Exception as e:
            logger.warning(f"Error generating descriptive zone key: {str(e)}")
            return "activity area"

    def _get_spatial_context_description(self, region: str) -> str:
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

    def identify_outdoor_general_zones(self, category_regions: Dict, detected_objects: List[Dict], scene_type: str) -> Dict:
        """
        識別一般戶外場景的功能區域

        Args:
            category_regions: 按類別和區域分組的物件字典
            detected_objects: 檢測到的物件列表
            scene_type: 特定戶外場景類型

        Returns:
            戶外功能區域字典
        """
        try:
            zones = {}

            # 識別行人區域
            people_objs = [obj for obj in detected_objects if obj["class_id"] == 0]
            if people_objs:
                people_regions = {}
                for obj in people_objs:
                    region = obj["region"]
                    if region not in people_regions:
                        people_regions[region] = []
                    people_regions[region].append(obj)

                if people_regions:
                    # 找到主要的行人活動區域
                    main_people_regions = sorted(people_regions.items(),
                                            key=lambda x: len(x[1]),
                                            reverse=True)[:2]  # 取前2個區域

                    for idx, (region, objs) in enumerate(main_people_regions):
                        if len(objs) > 0:
                            # 生成基於位置的描述性鍵名
                            spatial_desc = self._get_directional_description(region)
                            if spatial_desc and spatial_desc != "central":
                                zone_key = f"{spatial_desc} pedestrian area"
                            else:
                                zone_key = "main pedestrian area" if idx == 0 else "secondary pedestrian area"

                            zones[zone_key] = {
                                "region": region,
                                "objects": ["person"] * len(objs),
                                "description": f"Pedestrian area with {len(objs)} {'people' if len(objs) > 1 else 'person'}"
                            }

            # 識別車輛區域，適用於街道和停車場
            vehicle_objs = [obj for obj in detected_objects if obj["class_id"] in [1, 2, 3, 5, 6, 7]]
            if vehicle_objs:
                vehicle_regions = {}
                for obj in vehicle_objs:
                    region = obj["region"]
                    if region not in vehicle_regions:
                        vehicle_regions[region] = []
                    vehicle_regions[region].append(obj)

                if vehicle_regions:
                    main_vehicle_region = max(vehicle_regions.items(),
                                        key=lambda x: len(x[1]),
                                        default=(None, []))

                    if main_vehicle_region[0] is not None:
                        vehicle_types = [obj["class_name"] for obj in main_vehicle_region[1]]
                        zones["vehicle_zone"] = {
                            "region": main_vehicle_region[0],
                            "objects": vehicle_types,
                            "description": f"Traffic area with {', '.join(list(set(vehicle_types))[:3])}"
                        }

            # 針對公園區域的特殊處理
            if scene_type == "park_area":
                zones.update(self._identify_park_recreational_zones(detected_objects))

            # 針對停車場的特殊處理
            if scene_type == "parking_lot":
                zones.update(self._identify_parking_zones(detected_objects))

            logger.info(f"Identified {len(zones)} outdoor zones for scene type '{scene_type}'")
            return zones

        except Exception as e:
            logger.error(f"Error identifying outdoor general zones: {str(e)}")
            logger.error(traceback.format_exc())
            return {}

    def identify_intersection_zones(self, category_regions: Dict, detected_objects: List[Dict], viewpoint: str) -> Dict:
        """
        辨識城市十字路口的功能區域，無論是否有行人，只要偵測到紅綠燈就一定顯示 Traffic Control Area；
        若有行人，則額外建立 Crossing Zone 並把行人 + 同 region 的紅綠燈歸在一起。

        Args:
            category_regions: 按類別和 region 分組的物件字典
            detected_objects: YOLO 檢測到的所有物件列表
            viewpoint: 偵測到的視角字串

        Returns:
            zones: 最終的十字路口功能區域字典
        """
        try:
            zones = {}

            # 1. 按 class_id 分出行人、車輛、紅綠燈
            pedestrian_objs    = [obj for obj in detected_objects if obj["class_id"] == 0]
            vehicle_objs       = [obj for obj in detected_objects if obj["class_id"] in [1, 2, 3, 5, 7]]
            traffic_light_objs = [obj for obj in detected_objects if obj["class_id"] == 9]

            # 2. Step A: 無條件建立 Traffic Control Area
            #    把每個 region 下的紅綠燈都先分群，生成對應 zone，確保「只要偵測到紅綠燈就一定顯示」
            signal_regions_all = {}
            for t in traffic_light_objs:
                region = t["region"]
                signal_regions_all.setdefault(region, []).append(t)

            for idx, (region, signals) in enumerate(signal_regions_all.items()):
                # 先決定 zone_key (依 direction 或 primary/auxiliary)
                direction = self._get_directional_description(region)
                if direction and direction != "central":
                    zone_key = f"{direction} traffic control area"
                else:
                    zone_key = "primary traffic control area" if idx == 0 else "auxiliary traffic control area"

                # 確保命名不衝突
                if zone_key in zones:
                    suffix = 1
                    new_key = f"{zone_key} ({suffix})"
                    while new_key in zones:
                        suffix += 1
                        new_key = f"{zone_key} ({suffix})"
                    zone_key = new_key

                zones[zone_key] = {
                    "region": region,
                    "objects": ["traffic light"] * len(signals),
                    "description": f"Traffic control area with {len(signals)} traffic lights in {region}"
                }

            # (用於後面計算 Crossing 使用掉的 traffic light)
            used_tl_count_per_region = dict.fromkeys(signal_regions_all.keys(), 0)

            # 3. Step B: 如果有行人，就建立 Crossing Zone，並移除已被打包的紅綠燈
            if pedestrian_objs:
                # 先呼叫 _analyze_crossing_patterns，讓它回傳「行人 + 同 region 的紅綠燈」區
                crossing_zones = self._analyze_crossing_patterns(pedestrian_objs, traffic_light_objs)

                # 把 Crossing Zone 加到最終 zones，並同時記錄已使用掉的紅綠燈數量
                for zone_key, zone_info in crossing_zones.items():
                    region = zone_info.get("region", "")
                    obj_list = zone_info.get("objects", [])

                    # 如果該 zone_info["objects"] 裡含有紅綠燈，就累加到 used_tl_count_per_region
                    count_in_zone = obj_list.count("traffic light")
                    if count_in_zone > 0:
                        used_tl_count_per_region[region] = used_tl_count_per_region.get(region, 0) + count_in_zone

                    # 加入最終結果
                    # 如果 key 重複，也可以在此加上 index，或直接覆蓋
                    if zone_key in zones:
                        suffix = 1
                        new_key = f"{zone_key} ({suffix})"
                        while new_key in zones:
                            suffix += 1
                            new_key = f"{zone_key} ({suffix})"
                        zone_key = new_key

                    zones[zone_key] = {
                        "region": region,
                        "objects": obj_list,
                        "description": zone_info.get("description", "")
                    }

            # 4. Step C: 計算並顯示 debug 資訊 (Total / Used / Remaining)
            for region, signals in signal_regions_all.items():
                total = len(signals)
                used = used_tl_count_per_region.get(region, 0)
                remaining = total - used
                # print(f"[DEBUG] Region '{region}': Total TL = {total}, Used in crossing = {used}, Remaining = {remaining}")

            # 5. Step D: 分析車輛交通區域（Vehicle Zones）
            if vehicle_objs:
                traffic_zones = self._analyze_traffic_zones(vehicle_objs)
                # _analyze_traffic_zones 內部已用英文 debug，直接更新
                for zone_key, zone_info in traffic_zones.items():
                    if zone_key in zones:
                        suffix = 1
                        new_key = f"{zone_key} ({suffix})"
                        while new_key in zones:
                            suffix += 1
                            new_key = f"{zone_key} ({suffix})"
                        zone_key = new_key
                    zones[zone_key] = zone_info

            logger.info(f"Identified {len(zones)} intersection zones")
            return zones

        except Exception as e:
            logger.error(f"Error in identify_intersection_zones: {str(e)}")
            logger.error(traceback.format_exc())
            return {}

    def identify_aerial_view_zones(self, category_regions: Dict, detected_objects: List[Dict], scene_type: str) -> Dict:
        """
        辨識空中視角場景的功能區域
        專注於模式和流動而非特定區域

        Args:
            category_regions: 按類別和區域分組的物件字典
            detected_objects: 檢測到的物件列表
            scene_type: 特定場景類型

        Returns:
            空中視角功能區域字典
        """
        try:
            zones = {}

            # 識別行人模式
            people_objs = [obj for obj in detected_objects if obj["class_id"] == 0]
            if people_objs:
                # 將位置轉換為數組進行模式分析
                positions = np.array([obj["normalized_center"] for obj in people_objs])

                if len(positions) >= 3:
                    # 計算分布指標
                    x_coords = positions[:, 0]
                    y_coords = positions[:, 1]

                    x_mean = np.mean(x_coords)
                    y_mean = np.mean(y_coords)
                    x_std = np.std(x_coords)
                    y_std = np.std(y_coords)

                    # 判斷人群是否組織成線性模式
                    if x_std < 0.1 or y_std < 0.1:
                        # 沿一個軸的線性分布
                        pattern_direction = "vertical" if x_std < y_std else "horizontal"

                        zones["pedestrian_pattern"] = {
                            "region": "central",
                            "objects": ["person"] * len(people_objs),
                            "description": f"Aerial view shows a {pattern_direction} pedestrian movement pattern"
                        }
                    else:
                        # 更分散的模式
                        zones["pedestrian_distribution"] = {
                            "region": "wide",
                            "objects": ["person"] * len(people_objs),
                            "description": f"Aerial view shows pedestrians distributed across the area"
                        }

            # 識別車輛模式進行交通分析
            vehicle_objs = [obj for obj in detected_objects if obj["class_id"] in [1, 2, 3, 5, 6, 7]]
            if vehicle_objs:
                zones.update(self._analyze_aerial_traffic_patterns(vehicle_objs))

            # 針對十字路口特定空中視角的處理
            if "intersection" in scene_type:
                zones.update(self._identify_aerial_intersection_features(detected_objects))

            # 針對廣場空中視角的處理
            if "plaza" in scene_type:
                zones.update(self._identify_aerial_plaza_features(people_objs))

            logger.info(f"Identified {len(zones)} aerial view zones")
            return zones

        except Exception as e:
            logger.error(f"Error identifying aerial view zones: {str(e)}")
            logger.error(traceback.format_exc())
            return {}

    def identify_asian_cultural_zones(self, category_regions: Dict, detected_objects: List[Dict], scene_type: str) -> Dict:
        """
        辨識有亞洲文化背景的場景功能區域

        Args:
            category_regions: 按類別和區域分組的物件字典
            detected_objects: 檢測到的物件列表
            scene_type: 特定場景類型

        Returns:
            亞洲文化功能區域字典
        """
        try:
            zones = {}

            # 識別店面區域
            # 由於店面不能直接檢測，從情境推斷
            # 例如，尋找有標誌、行人和小物件的區域
            storefront_regions = {}
            for obj in detected_objects:
                if obj["class_id"] == 0:  # Person
                    region = obj["region"]
                    if region not in storefront_regions:
                        storefront_regions[region] = []
                    storefront_regions[region].append(obj)

            # 將人最多的區域作為店面區域
            if storefront_regions:
                main_storefront_regions = sorted(storefront_regions.items(),
                                            key=lambda x: len(x[1]),
                                            reverse=True)[:2]  # 前2個區域

                for idx, (region, objs) in enumerate(main_storefront_regions):
                    # 生成基於位置的描述性鍵名
                    spatial_desc = self._get_directional_description(region)
                    if spatial_desc and spatial_desc != "central":
                        zone_key = f"{spatial_desc} commercial area"
                    else:
                        zone_key = "main commercial area" if idx == 0 else "secondary commercial area"

                    zones[zone_key] = {
                        "region": region,
                        "objects": [obj["class_name"] for obj in objs],
                        "description": f"Asian commercial storefront with pedestrian activity"
                    }

            # 辨識行人通道 
            zones.update(self._identify_asian_pedestrian_pathway(detected_objects))

            # 辨識攤販區域（小攤/商店 - 從情境推斷）
            zones.update(self._identify_vendor_zones(detected_objects))

            # 針對夜市的特殊處理
            if scene_type == "asian_night_market":
                zones["food_stall_zone"] = {
                    "region": "middle_center",
                    "objects": ["inferred food stalls"],
                    "description": "Food stall area typical of Asian night markets"
                }

            logger.info(f"Identified {len(zones)} Asian cultural zones")
            return zones

        except Exception as e:
            logger.error(f"Error identifying Asian cultural zones: {str(e)}")
            logger.error(traceback.format_exc())
            return {}

    def identify_upscale_dining_zones(self, category_regions: Dict, detected_objects: List[Dict]) -> Dict:
        """
        辨識高級餐飲設置的功能區域

        Args:
            category_regions: 按類別和區域分組的物件字典
            detected_objects: 檢測到的物件列表

        Returns:
            高級餐飲功能區域字典
        """
        try:
            zones = {}

            # 辨識餐桌區域
            dining_items = []
            dining_regions = {}

            for obj in detected_objects:
                if obj["class_id"] in [40, 41, 42, 43, 44, 45, 60]:  # Wine glass, cup, fork, knife, spoon, bowl, table
                    region = obj["region"]
                    if region not in dining_regions:
                        dining_regions[region] = []
                    dining_regions[region].append(obj)
                    dining_items.append(obj["class_name"])

            if dining_items:
                main_dining_region = max(dining_regions.items(),
                                    key=lambda x: len(x[1]),
                                    default=(None, []))

                if main_dining_region[0] is not None:
                    zones["formal_dining_zone"] = {
                        "region": main_dining_region[0],
                        "objects": list(set(dining_items)),
                        "description": f"Formal dining area with {', '.join(list(set(dining_items))[:3])}"
                    }

            # 識別裝飾區域，增強檢測
            zones.update(self._identify_upscale_decorative_zones(detected_objects))

            # 識別座位安排區域
            zones.update(self._identify_dining_seating_zones(detected_objects))

            # 識別服務區域（如果與餐飲區域不同）
            zones.update(self._identify_serving_zones(detected_objects, zones))

            logger.info(f"Identified {len(zones)} upscale dining zones")
            return zones

        except Exception as e:
            logger.error(f"Error identifying upscale dining zones: {str(e)}")
            logger.error(traceback.format_exc())
            return {}

    def identify_financial_district_zones(self, category_regions: Dict, detected_objects: List[Dict]) -> Dict:
        """
        金融區場景的功能區域

        Args:
            category_regions: 按類別和區域分組的物件字典
            detected_objects: 檢測到的物件列表

        Returns:
            金融區功能區域字典
        """
        try:
            zones = {}

            # 識別交通區域
            traffic_items = []
            traffic_regions = {}

            for obj in detected_objects:
                if obj["class_id"] in [1, 2, 3, 5, 6, 7, 9]:  # 各種車輛和交通燈
                    region = obj["region"]
                    if region not in traffic_regions:
                        traffic_regions[region] = []
                    traffic_regions[region].append(obj)
                    traffic_items.append(obj["class_name"])

            if traffic_items:
                main_traffic_region = max(traffic_regions.items(),
                                    key=lambda x: len(x[1]),
                                    default=(None, []))

                if main_traffic_region[0] is not None:
                    zones["traffic_zone"] = {
                        "region": main_traffic_region[0],
                        "objects": list(set(traffic_items)),
                        "description": f"Urban traffic area with {', '.join(list(set(traffic_items))[:3])}"
                    }

            # 側邊建築區域（從場景情境推斷）
            zones.update(self._identify_building_zones(detected_objects))

            # 行人區域
            zones.update(self._identify_financial_pedestrian_zones(detected_objects))

            logger.info(f"Identified {len(zones)} financial district zones")
            return zones

        except Exception as e:
            logger.error(f"Error identifying financial district zones: {str(e)}")
            logger.error(traceback.format_exc())
            return {}

    def identify_landmark_zones(self, landmark_objects: List[Dict]) -> Dict:
        """
        辨識與地標相關的功能區域

        Args:
            landmark_objects: 被辨識為地標的物體列表

        Returns:
            地標相關的功能區域字典
        """
        try:
            landmark_zones = {}

            # 如果沒有任何地標，就直接回空字典
            if not landmark_objects:
                logger.warning("No landmark objects provided to identify_landmark_zones")
                return landmark_zones

            # 只取第一個地標來示範：至少產生一個地標
            landmark = landmark_objects[0]
            # 確保傳入的 landmark 是 dict
            if not isinstance(landmark, dict):
                logger.warning("First landmark object is not a dict")
                return landmark_zones

            # 從 landmark dict 拿出必要欄位
            landmark_id = landmark.get("landmark_id", "unknown_landmark")
            landmark_name = landmark.get("class_name", "Landmark")
            landmark_type = landmark.get("landmark_type", "architectural")
            landmark_region = landmark.get("region", "middle_center")

            # 如果 location 沒提供，就給預設 "this area"
            location = landmark.get("location")
            if not location:
                location = "this area"

            # 為地標創建主要觀景區
            zone_id = f"{landmark_name.lower().replace(' ', '_')}_viewing_area"
            zone_name = f"{landmark_name} Viewing Area"

            # 根據地標類型調整描述，並確保帶入地點
            if landmark_type == "natural":
                zone_description = (
                    f"Scenic viewpoint for observing {landmark_name}, "
                    f"a notable natural landmark in {location}."
                )
                primary_function = "Nature observation and photography"
            elif landmark_type == "monument":
                zone_description = (
                    f"Viewing area around {landmark_name}, "
                    f"a significant monument in {location}."
                )
                primary_function = "Historical appreciation and cultural tourism"
            else:  # architectural
                zone_description = (
                    f"Area centered around {landmark_name}, "
                    f"where visitors can observe and appreciate this iconic structure in {location}."
                )
                primary_function = "Architectural tourism and photography"

            # 確定與地標相關的物體（如果被偵測到）
            related_objects = []
            for o in landmark_objects:
                cn = o.get("class_name", "").lower()
                if cn in ["person", "camera", "cell phone", "backpack"]:
                    related_objects.append(cn)

            # 建立地標功能區
            landmark_zones[zone_id] = {
                "name": zone_name,
                "description": zone_description,
                "objects": ["landmark"] + related_objects,
                "region": landmark_region,
                "primary_function": primary_function
            }

            # 創建相關輔助功能區，如攝影區、紀念品販賣區
            auxiliary_zones = self._create_landmark_auxiliary_zones(landmark, 0)
            if auxiliary_zones:
                landmark_zones.update(auxiliary_zones)

            logger.info(f"Identified {len(landmark_zones)} landmark zones")
            return landmark_zones

        except Exception as e:
            logger.error(f"Error in identify_landmark_zones: {str(e)}")
            logger.error(traceback.format_exc())
            return {}


    def _identify_primary_functional_area(self, detected_objects: List[Dict]) -> Dict:
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
            dining_area = self._detect_functional_combination(
                detected_objects,
                primary_objects=[60],  # dining table
                supporting_objects=[56, 40, 41, 42, 43],  # chair, wine glass, cup, fork, knife
                min_supporting=2,
                description_template="Dining area with table and seating arrangement"
            )
            if dining_area:
                return dining_area

            # 休息區域檢測（沙發電視組合或床）
            seating_area = self._detect_functional_combination(
                detected_objects,
                primary_objects=[57, 59],  # sofa, bed
                supporting_objects=[62, 58, 56],  # tv, potted plant, chair
                min_supporting=1,
                description_template="Seating and relaxation area"
            )
            if seating_area:
                return seating_area

            # 工作區域檢測（電子設備與家具組合）
            work_area = self._detect_functional_combination(
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

    def _identify_secondary_functional_area(self, detected_objects: List[Dict], existing_zones: Dict) -> Dict:
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
            decorative_area = self._detect_functional_combination(
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
            storage_area = self._detect_functional_combination(
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

    def _detect_functional_combination(self, detected_objects: List[Dict], primary_objects: List[int],
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

    def _analyze_crossing_patterns(self, pedestrians: List[Dict], traffic_lights: List[Dict]) -> Dict:
        """
        Analyze pedestrian crossing patterns to identify crossing zones.
        若同一 region 中同時有行人與紅綠燈，則將兩者都放入該區域的 objects。

        Args:
            pedestrians: 行人物件列表（每個 obj 應包含 'class_id', 'region', 'confidence' 等）
            traffic_lights: 紅綠燈物件列表（每個 obj 應包含 'class_id', 'region', 'confidence' 等）

        Returns:
            crossing_zones: 字典，key 為 zone 名稱，value 包含 'region', 'objects', 'description'
        """
        try:
            crossing_zones = {}

            # 如果沒有任何行人，就不辨識任何 crossing zone
            if not pedestrians:
                return crossing_zones

            # (1) 按照 region 分組行人
            pedestrian_regions = {}
            for p in pedestrians:
                region = p["region"]
                pedestrian_regions.setdefault(region, []).append(p)

            # (2) 針對每個 region，看是否同時有紅綠燈
            # 建立一個 mapping： region -> { "pedestrians": [...], "traffic_lights": [...] }
            combined_regions = {}
            for region, peds in pedestrian_regions.items():
                # 取得該 region 下所有紅綠燈
                tls_in_region = [t for t in traffic_lights if t["region"] == region]
                combined_regions[region] = {
                    "pedestrians": peds,
                    "traffic_lights": tls_in_region
                }

            # (3) 按照行人數量排序，找出前兩個需要建立 crossing zone 的 region
            sorted_regions = sorted(
                combined_regions.items(),
                key=lambda x: len(x[1]["pedestrians"]),
                reverse=True
            )

            # (4) 將前兩個 region 建立 Crossing Zone，objects 同時包含行人與紅綠燈
            for idx, (region, group) in enumerate(sorted_regions[:2]):
                peds = group["pedestrians"]
                tls  = group["traffic_lights"]
                has_nearby_signals = len(tls) > 0

                # 生成 zone_name（基於 region 方向 + idx 決定主/次 crossing）
                direction = self._get_directional_description(region)
                if direction and direction != "central":
                    zone_name = f"{direction} crossing area"
                else:
                    zone_name = "main crossing area" if idx == 0 else "secondary crossing area"

                # 組合 description
                description = f"Pedestrian crossing area with {len(peds)} "
                description += "person" if len(peds) == 1 else "people"
                if direction:
                    description += f" in {direction} direction"
                if has_nearby_signals:
                    description += " near traffic signals"

                # ======= 將行人 + 同區紅綠燈一併放入 objects =======
                obj_list = ["pedestrian"] * len(peds)
                if has_nearby_signals:
                    obj_list += ["traffic light"] * len(tls)

                crossing_zones[zone_name] = {
                    "region": region,
                    "objects": obj_list,
                    "description": description
                }

            return crossing_zones

        except Exception as e:
            logger.error(f"Error in _analyze_crossing_patterns: {str(e)}")
            logger.error(traceback.format_exc())
            return {}


    def _analyze_traffic_zones(self, vehicles: List[Dict]) -> Dict:
        """
        分析車輛分布以識別具有方向感知的交通區域

        Args:
            vehicles: 車輛物件列表

        Returns:
            識別出的交通區域字典
        """
        try:
            traffic_zones = {}

            if not vehicles:
                return traffic_zones

            # 按區域分組車輛
            vehicle_regions = {}
            for v in vehicles:
                region = v["region"]
                if region not in vehicle_regions:
                    vehicle_regions[region] = []
                vehicle_regions[region].append(v)

            # 為有車輛的區域創建交通區域
            main_traffic_region = max(vehicle_regions.items(), key=lambda x: len(x[1]), default=(None, []))

            if main_traffic_region[0] is not None:
                region = main_traffic_region[0]
                vehicles_in_region = main_traffic_region[1]

                # 獲取車輛類型列表用於描述
                vehicle_types = [v["class_name"] for v in vehicles_in_region]
                unique_types = list(set(vehicle_types))

                # 獲取方向描述
                direction = self._get_directional_description(region)

                # 創建描述性區域
                traffic_zones["vehicle_zone"] = {
                    "region": region,
                    "objects": vehicle_types,
                    "description": f"Vehicle traffic area with {', '.join(unique_types[:3])}" +
                                (f" in {direction} area" if direction else "")
                }

                # 如果車輛分布在多個區域，創建次要區域
                if len(vehicle_regions) > 1:
                    # 獲取第二大車輛聚集區域
                    sorted_regions = sorted(vehicle_regions.items(), key=lambda x: len(x[1]), reverse=True)
                    if len(sorted_regions) > 1:
                        second_region, second_vehicles = sorted_regions[1]
                        direction = self._get_directional_description(second_region)
                        vehicle_types = [v["class_name"] for v in second_vehicles]
                        unique_types = list(set(vehicle_types))

                        traffic_zones["secondary_vehicle_zone"] = {
                            "region": second_region,
                            "objects": vehicle_types,
                            "description": f"Secondary traffic area with {', '.join(unique_types[:2])}" +
                                        (f" in {direction} direction" if direction else "")
                        }

            return traffic_zones

        except Exception as e:
            logger.error(f"Error analyzing traffic zones: {str(e)}")
            logger.error(traceback.format_exc())
            return {}

    def _get_directional_description(self, region: str) -> str:
        """
        將區域名稱轉換為方位描述（東西南北）

        Args:
            region: 區域名稱

        Returns:
            方位描述字串
        """
        try:
            region_lower = region.lower()

            if "top" in region_lower and "left" in region_lower:
                return "northwest"
            elif "top" in region_lower and "right" in region_lower:
                return "northeast"
            elif "bottom" in region_lower and "left" in region_lower:
                return "southwest"
            elif "bottom" in region_lower and "right" in region_lower:
                return "southeast"
            elif "top" in region_lower:
                return "north"
            elif "bottom" in region_lower:
                return "south"
            elif "left" in region_lower:
                return "west"
            elif "right" in region_lower:
                return "east"
            else:
                return "central"

        except Exception as e:
            logger.error(f"Error getting directional description for region '{region}': {str(e)}")
            return "central"

    def _identify_park_recreational_zones(self, detected_objects: List[Dict]) -> Dict:
        """
        識別公園的休閒活動區域

        Args:
            detected_objects: 檢測到的物件列表

        Returns:
            休閒區域字典
        """
        try:
            zones = {}

            # 尋找休閒物件（運動球、風箏等）
            rec_items = []
            rec_regions = {}

            for obj in detected_objects:
                if obj["class_id"] in [32, 33, 34, 35, 38]:  # sports ball, kite, baseball bat, glove, tennis racket
                    region = obj["region"]
                    if region not in rec_regions:
                        rec_regions[region] = []
                    rec_regions[region].append(obj)
                    rec_items.append(obj["class_name"])

            if rec_items:
                main_rec_region = max(rec_regions.items(),
                                key=lambda x: len(x[1]),
                                default=(None, []))

                if main_rec_region[0] is not None:
                    zones["recreational_zone"] = {
                        "region": main_rec_region[0],
                        "objects": list(set(rec_items)),
                        "description": f"Recreational area with {', '.join(list(set(rec_items)))}"
                    }

            return zones

        except Exception as e:
            logger.error(f"Error identifying park recreational zones: {str(e)}")
            logger.error(traceback.format_exc())
            return {}

    def _identify_parking_zones(self, detected_objects: List[Dict]) -> Dict:
        """
        停車場的停車區域

        Args:
            detected_objects: 檢測到的物件列表

        Returns:
            停車區域字典
        """
        try:
            zones = {}

            # 尋找停放的汽車
            car_objs = [obj for obj in detected_objects if obj["class_id"] == 2]  # cars

            if len(car_objs) >= 3:
                # 檢查汽車是否按模式排列（簡化）
                car_positions = [obj["normalized_center"] for obj in car_objs]

                # 通過分析垂直位置檢查行模式
                y_coords = [pos[1] for pos in car_positions]
                y_clusters = {}

                # 簡化聚類 - 按相似y坐標分組汽車
                for i, y in enumerate(y_coords):
                    assigned = False
                    for cluster_y in y_clusters.keys():
                        if abs(y - cluster_y) < 0.1:  # 圖像高度的10%內
                            y_clusters[cluster_y].append(i)
                            assigned = True
                            break

                    if not assigned:
                        y_clusters[y] = [i]

                # 如果有行模式
                if max(len(indices) for indices in y_clusters.values()) >= 2:
                    zones["parking_row"] = {
                        "region": "central",
                        "objects": ["car"] * len(car_objs),
                        "description": f"Organized parking area with vehicles arranged in rows"
                    }
                else:
                    zones["parking_area"] = {
                        "region": "wide",
                        "objects": ["car"] * len(car_objs),
                        "description": f"Parking area with {len(car_objs)} vehicles"
                    }

            return zones

        except Exception as e:
            logger.error(f"Error identifying parking zones: {str(e)}")
            logger.error(traceback.format_exc())
            return {}

    def _analyze_aerial_traffic_patterns(self, vehicle_objs: List[Dict]) -> Dict:
        """
        分析空中視角的車輛交通模式

        Args:
            vehicle_objs: 車輛物件列表

        Returns:
            交通模式區域字典
        """
        try:
            zones = {}

            if not vehicle_objs:
                return zones

            # 將位置轉換為數組進行模式分析
            positions = np.array([obj["normalized_center"] for obj in vehicle_objs])

            if len(positions) >= 2:
                # 計算分布指標
                x_coords = positions[:, 0]
                y_coords = positions[:, 1]

                x_mean = np.mean(x_coords)
                y_mean = np.mean(y_coords)
                x_std = np.std(x_coords)
                y_std = np.std(y_coords)

                # 判斷車輛是否組織成車道
                if x_std < y_std * 0.5:
                    # 車輛垂直對齊 - 表示南北交通
                    zones["vertical_traffic_flow"] = {
                        "region": "central_vertical",
                        "objects": [obj["class_name"] for obj in vehicle_objs[:5]],
                        "description": "North-south traffic flow visible from aerial view"
                    }
                elif y_std < x_std * 0.5:
                    # 車輛水平對齊 - 表示東西交通
                    zones["horizontal_traffic_flow"] = {
                        "region": "central_horizontal",
                        "objects": [obj["class_name"] for obj in vehicle_objs[:5]],
                        "description": "East-west traffic flow visible from aerial view"
                    }
                else:
                    # 車輛多方向 - 表示十字路口
                    zones["intersection_traffic"] = {
                        "region": "central",
                        "objects": [obj["class_name"] for obj in vehicle_objs[:5]],
                        "description": "Multi-directional traffic at intersection visible from aerial view"
                    }

            return zones

        except Exception as e:
            logger.error(f"Error analyzing aerial traffic patterns: {str(e)}")
            logger.error(traceback.format_exc())
            return {}

    def _identify_aerial_intersection_features(self, detected_objects: List[Dict]) -> Dict:
        """
        空中視角十字路口特徵

        Args:
            detected_objects: 檢測到的物件列表

        Returns:
            十字路口特徵區域字典
        """
        try:
            zones = {}

            # 檢查交通信號
            traffic_light_objs = [obj for obj in detected_objects if obj["class_id"] == 9]
            if traffic_light_objs:
                zones["traffic_control_pattern"] = {
                    "region": "intersection",
                    "objects": ["traffic light"] * len(traffic_light_objs),
                    "description": f"Intersection traffic control with {len(traffic_light_objs)} signals visible from above"
                }

            # 人行道從空中視角的情境推斷
            zones["crossing_pattern"] = {
                "region": "central",
                "objects": ["inferred crosswalk"],
                "description": "Crossing pattern visible from aerial perspective"
            }

            return zones

        except Exception as e:
            logger.error(f"Error identifying aerial intersection features: {str(e)}")
            logger.error(traceback.format_exc())
            return {}

    def _identify_aerial_plaza_features(self, people_objs: List[Dict]) -> Dict:
        """
        識別空中視角廣場特徵

        Args:
            people_objs: 行人物件列表

        Returns:
            廣場特徵區域字典
        """
        try:
            zones = {}

            if people_objs:
                # 檢查人群是否聚集在中央區域
                central_people = [obj for obj in people_objs
                                if "middle" in obj["region"]]

                if central_people:
                    zones["central_gathering"] = {
                        "region": "middle_center",
                        "objects": ["person"] * len(central_people),
                        "description": f"Central plaza gathering area with {len(central_people)} people viewed from above"
                    }

            return zones

        except Exception as e:
            logger.error(f"Error identifying aerial plaza features: {str(e)}")
            logger.error(traceback.format_exc())
            return {}

    def _identify_asian_pedestrian_pathway(self, detected_objects: List[Dict]) -> Dict:
        """
        亞洲文化場景中的行人通道

        Args:
            detected_objects: 檢測到的物件列表

        Returns:
            行人通道區域字典
        """
        try:
            zones = {}

            pathway_items = []
            pathway_regions = {}

            # 提取人群用於通道分析
            people_objs = [obj for obj in detected_objects if obj["class_id"] == 0]

            # 分析人群是否形成線形（商業街的特徵）
            people_positions = [obj["normalized_center"] for obj in people_objs]

            structured_path = False
            path_direction = "meandering"

            if len(people_positions) >= 3:
                # 檢查人群是否沿相似y坐標排列（水平路徑）
                y_coords = [pos[1] for pos in people_positions]
                y_mean = sum(y_coords) / len(y_coords)
                y_variance = sum((y - y_mean)**2 for y in y_coords) / len(y_coords)

                horizontal_path = y_variance < 0.05  # 低變異表示水平對齊

                # 檢查人群是否沿相似x坐標排列（垂直路徑）
                x_coords = [pos[0] for pos in people_positions]
                x_mean = sum(x_coords) / len(x_coords)
                x_variance = sum((x - x_mean)**2 for x in x_coords) / len(x_coords)

                vertical_path = x_variance < 0.05  # 低變異表示垂直對齊

                structured_path = horizontal_path or vertical_path
                path_direction = "horizontal" if horizontal_path else "vertical" if vertical_path else "meandering"

            # 收集通道物件（人、自行車、摩托車在中間區域）
            for obj in detected_objects:
                if obj["class_id"] in [0, 1, 3]:  # Person, bicycle, motorcycle
                    y_pos = obj["normalized_center"][1]
                    # 按垂直位置分組（圖像中間可能是通道）
                    if 0.25 <= y_pos <= 0.75:
                        region = obj["region"]
                        if region not in pathway_regions:
                            pathway_regions[region] = []
                        pathway_regions[region].append(obj)
                        pathway_items.append(obj["class_name"])

            if pathway_items:
                path_desc = "Pedestrian walkway with people moving through the commercial area"
                if structured_path:
                    path_desc = f"{path_direction.capitalize()} pedestrian walkway with organized foot traffic"

                zones["pedestrian_pathway"] = {
                    "region": "middle_center",  # 假設：通道通常在中間
                    "objects": list(set(pathway_items)),
                    "description": path_desc
                }

            return zones

        except Exception as e:
            logger.error(f"Error identifying Asian pedestrian pathway: {str(e)}")
            logger.error(traceback.format_exc())
            return {}

    def _identify_vendor_zones(self, detected_objects: List[Dict]) -> Dict:
        """
        識別攤販區域

        Args:
            detected_objects: 檢測到的物件列表

        Returns:
            攤販區域字典
        """
        try:
            zones = {}

            # 識別攤販區域（小攤/商店 - 從情境推斷）
            has_small_objects = any(obj["class_id"] in [24, 26, 39, 41] for obj in detected_objects)  # bags, bottles, cups
            has_people = any(obj["class_id"] == 0 for obj in detected_objects)

            if has_small_objects and has_people:
                # 可能的攤販區域是人群和小物件聚集的地方
                small_obj_regions = {}

                for obj in detected_objects:
                    if obj["class_id"] in [24, 26, 39, 41, 67]:  # bags, bottles, cups, phones
                        region = obj["region"]
                        if region not in small_obj_regions:
                            small_obj_regions[region] = []
                        small_obj_regions[region].append(obj)

                if small_obj_regions:
                    main_vendor_region = max(small_obj_regions.items(),
                                        key=lambda x: len(x[1]),
                                        default=(None, []))

                    if main_vendor_region[0] is not None:
                        vendor_items = [obj["class_name"] for obj in main_vendor_region[1]]
                        zones["vendor_zone"] = {
                            "region": main_vendor_region[0],
                            "objects": list(set(vendor_items)),
                            "description": "Vendor or market stall area with small merchandise"
                        }

            return zones

        except Exception as e:
            logger.error(f"Error identifying vendor zones: {str(e)}")
            logger.error(traceback.format_exc())
            return {}

    def _identify_upscale_decorative_zones(self, detected_objects: List[Dict]) -> Dict:
        """
        識別高級餐飲的裝飾區域

        Args:
            detected_objects: 檢測到的物件列表

        Returns:
            裝飾區域字典
        """
        try:
            zones = {}

            decor_items = []
            decor_regions = {}

            # 尋找裝飾元素（花瓶、酒杯、未使用的餐具）
            for obj in detected_objects:
                if obj["class_id"] in [75, 40]:  # Vase, wine glass
                    region = obj["region"]
                    if region not in decor_regions:
                        decor_regions[region] = []
                    decor_regions[region].append(obj)
                    decor_items.append(obj["class_name"])

            if decor_items:
                main_decor_region = max(decor_regions.items(),
                                    key=lambda x: len(x[1]),
                                    default=(None, []))

                if main_decor_region[0] is not None:
                    zones["decorative_zone"] = {
                        "region": main_decor_region[0],
                        "objects": list(set(decor_items)),
                        "description": f"Decorative area with {', '.join(list(set(decor_items)))}"
                    }

            return zones

        except Exception as e:
            logger.error(f"Error identifying upscale decorative zones: {str(e)}")
            logger.error(traceback.format_exc())
            return {}

    def _identify_dining_seating_zones(self, detected_objects: List[Dict]) -> Dict:
        """
        識別餐廳座位安排區域

        Args:
            detected_objects: 檢測到的物件列表

        Returns:
            座位區域字典
        """
        try:
            zones = {}

            # 識別座位安排區域
            chairs = [obj for obj in detected_objects if obj["class_id"] == 56]  # chairs
            if len(chairs) >= 2:
                chair_regions = {}
                for obj in chairs:
                    region = obj["region"]
                    if region not in chair_regions:
                        chair_regions[region] = []
                    chair_regions[region].append(obj)

                if chair_regions:
                    main_seating_region = max(chair_regions.items(),
                                        key=lambda x: len(x[1]),
                                        default=(None, []))

                    if main_seating_region[0] is not None:
                        zones["dining_seating_zone"] = {
                            "region": main_seating_region[0],
                            "objects": ["chair"] * len(main_seating_region[1]),
                            "description": f"Formal dining seating arrangement with {len(main_seating_region[1])} chairs"
                        }

            return zones

        except Exception as e:
            logger.error(f"Error identifying dining seating zones: {str(e)}")
            logger.error(traceback.format_exc())
            return {}

    def _identify_serving_zones(self, detected_objects: List[Dict], existing_zones: Dict) -> Dict:
        """
        識別服務區域

        Args:
            detected_objects: 檢測到的物件列表
            existing_zones: 已存在的功能區域

        Returns:
            服務區域字典
        """
        try:
            zones = {}

            serving_items = []
            serving_regions = {}

            # 服務區域可能有瓶子、碗、容器
            for obj in detected_objects:
                if obj["class_id"] in [39, 45]:  # Bottle, bowl
                    # 檢查是否在與主餐桌不同的區域
                    if "formal_dining_zone" in existing_zones and obj["region"] != existing_zones["formal_dining_zone"]["region"]:
                        region = obj["region"]
                        if region not in serving_regions:
                            serving_regions[region] = []
                        serving_regions[region].append(obj)
                        serving_items.append(obj["class_name"])

            if serving_items:
                main_serving_region = max(serving_regions.items(),
                                    key=lambda x: len(x[1]),
                                    default=(None, []))

                if main_serving_region[0] is not None:
                    zones["serving_zone"] = {
                        "region": main_serving_region[0],
                        "objects": list(set(serving_items)),
                        "description": f"Serving or sideboard area with {', '.join(list(set(serving_items)))}"
                    }

            return zones

        except Exception as e:
            logger.error(f"Error identifying serving zones: {str(e)}")
            logger.error(traceback.format_exc())
            return {}

    def _identify_building_zones(self, detected_objects: List[Dict]) -> Dict:
        """
        識別建築區域（從場景情境推斷）

        Args:
            detected_objects: 檢測到的物件列表

        Returns:
            建築區域字典
        """
        try:
            zones = {}

            # 側邊建築區域（從場景情境推斷）
            # 檢查是否有實際可能包含建築物的區域
            left_side_regions = ["top_left", "middle_left", "bottom_left"]
            right_side_regions = ["top_right", "middle_right", "bottom_right"]

            # 檢查左側
            left_building_evidence = True
            for region in left_side_regions:
                # 如果此區域有很多車輛或人群，不太可能是建築物
                vehicle_in_region = any(obj["region"] == region and obj["class_id"] in [1, 2, 3, 5, 7]
                                    for obj in detected_objects)
                people_in_region = any(obj["region"] == region and obj["class_id"] == 0
                                    for obj in detected_objects)

                if vehicle_in_region or people_in_region:
                    left_building_evidence = False
                    break

            # 檢查右側
            right_building_evidence = True
            for region in right_side_regions:
                # 如果此區域有很多車輛或人群，不太可能是建築物
                vehicle_in_region = any(obj["region"] == region and obj["class_id"] in [1, 2, 3, 5, 7]
                                    for obj in detected_objects)
                people_in_region = any(obj["region"] == region and obj["class_id"] == 0
                                    for obj in detected_objects)

                if vehicle_in_region or people_in_region:
                    right_building_evidence = False
                    break

            # 如果證據支持，添加建築區域
            if left_building_evidence:
                zones["building_zone_left"] = {
                    "region": "middle_left",
                    "objects": ["building"],  # 推斷
                    "description": "Tall buildings line the left side of the street"
                }

            if right_building_evidence:
                zones["building_zone_right"] = {
                    "region": "middle_right",
                    "objects": ["building"],  # 推斷
                    "description": "Tall buildings line the right side of the street"
                }

            return zones

        except Exception as e:
            logger.error(f"Error identifying building zones: {str(e)}")
            logger.error(traceback.format_exc())
            return {}

    def _identify_financial_pedestrian_zones(self, detected_objects: List[Dict]) -> Dict:
        """
        識別金融區的行人區域

        Args:
            detected_objects: 檢測到的物件列表

        Returns:
            行人區域字典
        """
        try:
            zones = {}

            # 識別行人區域（如果有人群）
            people_objs = [obj for obj in detected_objects if obj["class_id"] == 0]
            if people_objs:
                people_regions = {}
                for obj in people_objs:
                    region = obj["region"]
                    if region not in people_regions:
                        people_regions[region] = []
                    people_regions[region].append(obj)

                if people_regions:
                    main_pedestrian_region = max(people_regions.items(),
                                            key=lambda x: len(x[1]),
                                            default=(None, []))

                    if main_pedestrian_region[0] is not None:
                        zones["pedestrian_zone"] = {
                            "region": main_pedestrian_region[0],
                            "objects": ["person"] * len(main_pedestrian_region[1]),
                            "description": f"Pedestrian area with {len(main_pedestrian_region[1])} people navigating the financial district"
                        }

            return zones

        except Exception as e:
            logger.error(f"Error identifying financial pedestrian zones: {str(e)}")
            logger.error(traceback.format_exc())
            return {}

    def _create_landmark_auxiliary_zones(self, landmark: Dict, index: int) -> Dict:
        """
        創建地標相關的輔助區域（攝影區、紀念品區等）

        Args:
            landmark: 地標物件字典
            index: 地標索引

        Returns:
            輔助區域字典
        """
        try:
            auxiliary_zones = {}
            landmark_region = landmark.get("region", "middle_center")
            landmark_name = landmark.get("class_name", "Landmark")

            # 創建攝影區
            # 根據地標位置調整攝影區位置（地標前方通常是攝影區）
            region_mapping = {
                "top_left": "bottom_right",
                "top_center": "bottom_center",
                "top_right": "bottom_left",
                "middle_left": "middle_right",
                "middle_center": "bottom_center",
                "middle_right": "middle_left",
                "bottom_left": "top_right",
                "bottom_center": "top_center",
                "bottom_right": "top_left"
            }

            photo_region = region_mapping.get(landmark_region, landmark_region)

            photo_key = f"{landmark_name.lower().replace(' ', '_')}_photography_spot"
            auxiliary_zones[photo_key] = {
                "name": f"{landmark_name} Photography Spot",
                "description": f"Popular position for photographing {landmark_name} with optimal viewing angle.",
                "objects": ["camera", "person", "cell phone"],
                "region": photo_region,
                "primary_function": "Tourist photography"
            }

            # 如果是著名地標，可能有紀念品販售區
            if landmark.get("confidence", 0) > 0.7:  # 高置信度地標更可能有紀念品區
                # 根據地標位置找到適合的紀念品區位置（通常在地標附近但不直接在地標上）
                adjacent_regions = {
                    "top_left": ["top_center", "middle_left"],
                    "top_center": ["top_left", "top_right"],
                    "top_right": ["top_center", "middle_right"],
                    "middle_left": ["top_left", "bottom_left"],
                    "middle_center": ["middle_left", "middle_right"],
                    "middle_right": ["top_right", "bottom_right"],
                    "bottom_left": ["middle_left", "bottom_center"],
                    "bottom_center": ["bottom_left", "bottom_right"],
                    "bottom_right": ["bottom_center", "middle_right"]
                }

                if landmark_region in adjacent_regions:
                    souvenir_region = adjacent_regions[landmark_region][0]  # 選擇第一個相鄰區域

                    souvenir_key = f"{landmark_name.lower().replace(' ', '_')}_souvenir_area"
                    auxiliary_zones[souvenir_key] = {
                        "name": f"{landmark_name} Souvenir Area",
                        "description": f"Area where visitors can purchase souvenirs and memorabilia related to {landmark_name}.",
                        "objects": ["person", "handbag", "backpack"],
                        "region": souvenir_region,
                        "primary_function": "Tourism commerce"
                    }

            return auxiliary_zones

        except Exception as e:
            logger.error(f"Error creating landmark auxiliary zones: {str(e)}")
            logger.error(traceback.format_exc())
            return {}
