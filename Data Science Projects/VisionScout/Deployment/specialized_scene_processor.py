
import logging
import traceback
import numpy as np
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

class SpecializedSceneProcessor:
    """
    負責處理特殊場景類型和地標識別
    包含亞洲文化場景、高級餐飲、金融區、空中視角等專門處理邏輯
    """

    def __init__(self):
        """初始化特殊場景處理器"""
        try:
            logger.info("SpecializedSceneProcessor initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize SpecializedSceneProcessor: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def identify_aerial_intersection_features(self, detected_objects: List[Dict]) -> Dict:
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

    def identify_aerial_plaza_features(self, people_objs: List[Dict]) -> Dict:
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

    def identify_asian_pedestrian_pathway(self, detected_objects: List[Dict]) -> Dict:
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
                    "region": "middle_center",  # 通道通常會在中間area
                    "objects": list(set(pathway_items)),
                    "description": path_desc
                }

            return zones

        except Exception as e:
            logger.error(f"Error identifying Asian pedestrian pathway: {str(e)}")
            logger.error(traceback.format_exc())
            return {}

    def identify_vendor_zones(self, detected_objects: List[Dict]) -> Dict:
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

    def identify_upscale_decorative_zones(self, detected_objects: List[Dict]) -> Dict:
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

    def identify_dining_seating_zones(self, detected_objects: List[Dict]) -> Dict:
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

    def identify_serving_zones(self, detected_objects: List[Dict], existing_zones: Dict) -> Dict:
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

    def identify_building_zones(self, detected_objects: List[Dict]) -> Dict:
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

    def identify_financial_pedestrian_zones(self, detected_objects: List[Dict]) -> Dict:
        """
        識別金融區的行人區域

        Args:
            detected_objects: 檢測到的物件列表

        Returns:
            行人區域字典
        """
        try:
            zones = {}

            # 辨識行人區域（如果有人群）
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

    def create_landmark_auxiliary_zones(self, landmark: Dict, index: int) -> Dict:
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
