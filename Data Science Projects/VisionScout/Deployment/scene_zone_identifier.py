
import logging
import traceback
import numpy as np
from typing import Dict, List, Any, Optional
from functional_zone_detector import FunctionalZoneDetector
from pattern_analyzer import PatternAnalyzer
from specialized_scene_processor import SpecializedSceneProcessor

logger = logging.getLogger(__name__)

class SceneZoneIdentifier:
    """
    負責不同場景類型的區域識別邏輯
    專注於根據場景類型執行相應的功能區域識別策略
    整合所有專門的區域辨識組件，主要須整合至SpatialAnalyzer
    """

    def __init__(self):
        """初始化場景區域辨識器"""
        try:
            # 初始化各個專門組件
            self.functional_detector = FunctionalZoneDetector()
            self.pattern_analyzer = PatternAnalyzer()
            self.scene_processor = SpecializedSceneProcessor()

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
            primary_zone = self.functional_detector.identify_primary_functional_area(detected_objects)
            if primary_zone:
                # 基於區域內容生成描述性鍵名
                descriptive_key = self.functional_detector.generate_descriptive_zone_key_from_data(primary_zone, "primary")
                zones[descriptive_key] = primary_zone

            # 只有明確證據且物件數量足夠時創建次要功能區域
            if len(zones) >= 1 and len(detected_objects) >= 6:
                secondary_zone = self.functional_detector.identify_secondary_functional_area(detected_objects, zones)
                if secondary_zone:
                    # 基於區域內容生成描述性鍵名
                    descriptive_key = self.functional_detector.generate_descriptive_zone_key_from_data(secondary_zone, "secondary")
                    zones[descriptive_key] = secondary_zone

            logger.info(f"Identified {len(zones)} indoor zones for scene type '{scene_type}'")
            return zones

        except Exception as e:
            logger.error(f"Error identifying indoor zones: {str(e)}")
            logger.error(traceback.format_exc())
            return {}

    def identify_outdoor_general_zones(self, category_regions: Dict, detected_objects: List[Dict], scene_type: str) -> Dict:
        """
        辨識一般戶外場景的功能區域

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
                zones.update(self.pattern_analyzer.identify_park_recreational_zones(detected_objects))

            # 針對停車場的特殊處理
            if scene_type == "parking_lot":
                zones.update(self.pattern_analyzer.identify_parking_zones(detected_objects))

            logger.info(f"Identified {len(zones)} outdoor zones for scene type '{scene_type}'")
            return zones

        except Exception as e:
            logger.error(f"Error identifying outdoor general zones: {str(e)}")
            logger.error(traceback.format_exc())
            return {}

    def identify_intersection_zones(self, category_regions: Dict, detected_objects: List[Dict], viewpoint: str) -> Dict:
        """
        辨識城市十字路口的功能區域，無論是否有行人，只要偵測到紅綠燈就一定顯示 Traffic Control Area；
        如果有行人，則額外建立 Crossing Zone 並把行人 + 同 region 的紅綠燈歸在一起。

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
            #    把每個 region 下的紅綠燈都先分群，生成對應 zone，確保"只要偵測到紅綠燈就一定顯示"
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
                # 先呼叫 analyze_crossing_patterns，讓它回傳「行人 + 同 region 的紅綠燈」區
                crossing_zones = self.pattern_analyzer.analyze_crossing_patterns(pedestrian_objs, traffic_light_objs)

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
                traffic_zones = self.pattern_analyzer.analyze_traffic_zones(vehicle_objs)
                # analyze_traffic_zones 內部已用英文 debug，直接更新
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
                zones.update(self.pattern_analyzer.analyze_aerial_traffic_patterns(vehicle_objs))

            # 針對十字路口特定空中視角的處理
            if "intersection" in scene_type:
                zones.update(self.scene_processor.identify_aerial_intersection_features(detected_objects))

            # 針對廣場空中視角的處理
            if "plaza" in scene_type:
                zones.update(self.scene_processor.identify_aerial_plaza_features(people_objs))

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
            zones.update(self.scene_processor.identify_asian_pedestrian_pathway(detected_objects))

            # 辨識攤販區域（小攤/商店 - 從情境推斷）
            zones.update(self.scene_processor.identify_vendor_zones(detected_objects))

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
            zones.update(self.scene_processor.identify_upscale_decorative_zones(detected_objects))

            # 識別座位安排區域
            zones.update(self.scene_processor.identify_dining_seating_zones(detected_objects))

            # 識別服務區域（如果與餐飲區域不同）
            zones.update(self.scene_processor.identify_serving_zones(detected_objects, zones))

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
            zones.update(self.scene_processor.identify_building_zones(detected_objects))

            # 行人區域
            zones.update(self.scene_processor.identify_financial_pedestrian_zones(detected_objects))

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
            auxiliary_zones = self.scene_processor.create_landmark_auxiliary_zones(landmark, 0)
            if auxiliary_zones:
                landmark_zones.update(auxiliary_zones)

            logger.info(f"Identified {len(landmark_zones)} landmark zones")
            return landmark_zones

        except Exception as e:
            logger.error(f"Error in identify_landmark_zones: {str(e)}")
            logger.error(traceback.format_exc())
            return {}

    def _get_directional_description(self, region: str) -> str:
        """
        將區域名稱轉換為方位描述（東西南北）
        這是核心工具方法，供所有組件使用

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
