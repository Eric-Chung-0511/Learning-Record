
import os
import numpy as np
from typing import Dict, List, Tuple, Any, Optional

from scene_type import SCENE_TYPES
from enhance_scene_describer import EnhancedSceneDescriber

class SpatialAnalyzer:
    """
    Analyzes spatial relationships between objects in an image.
    Handles region assignment, object positioning, and functional zone identification.
    """

    def __init__(self, class_names: Dict[int, str] = None, object_categories=None):
        """Initialize the spatial analyzer with image regions"""
        # Define regions of the image (3x3 grid)
        self.regions = {
            "top_left": (0, 0, 1/3, 1/3),
            "top_center": (1/3, 0, 2/3, 1/3),
            "top_right": (2/3, 0, 1, 1/3),
            "middle_left": (0, 1/3, 1/3, 2/3),
            "middle_center": (1/3, 1/3, 2/3, 2/3),
            "middle_right": (2/3, 1/3, 1, 2/3),
            "bottom_left": (0, 2/3, 1/3, 1),
            "bottom_center": (1/3, 2/3, 2/3, 1),
            "bottom_right": (2/3, 2/3, 1, 1)
        }

        self.class_names = class_names
        self.OBJECT_CATEGORIES = object_categories or {}
        self.enhance_descriptor = EnhancedSceneDescriber(scene_types=SCENE_TYPES)

        # Distances thresholds for proximity analysis (normalized)
        self.proximity_threshold = 0.2


    def _determine_region(self, x: float, y: float) -> str:
        """
        Determine which region a point falls into.

        Args:
            x: Normalized x-coordinate (0-1)
            y: Normalized y-coordinate (0-1)

        Returns:
            Region name
        """
        for region_name, (x1, y1, x2, y2) in self.regions.items():
            if x1 <= x < x2 and y1 <= y < y2:
                return region_name

        return "unknown"

    def _analyze_regions(self, detected_objects: List[Dict]) -> Dict:
        """
        Analyze object distribution across image regions.

        Args:
            detected_objects: List of detected objects with position information

        Returns:
            Dictionary with region analysis
        """
        # Count objects in each region
        region_counts = {region: 0 for region in self.regions.keys()}
        region_objects = {region: [] for region in self.regions.keys()}

        for obj in detected_objects:
            region = obj["region"]
            if region in region_counts:
                region_counts[region] += 1
                region_objects[region].append({
                    "class_id": obj["class_id"],
                    "class_name": obj["class_name"]
                })

        # Determine main focus regions (top 1-2 regions by object count)
        sorted_regions = sorted(region_counts.items(), key=lambda x: x[1], reverse=True)
        main_regions = [region for region, count in sorted_regions if count > 0][:2]

        return {
            "counts": region_counts,
            "main_focus": main_regions,
            "objects_by_region": region_objects
        }

    def _extract_detected_objects(self, detection_result: Any, confidence_threshold: float = 0.25) -> List[Dict]:
        """
        Extract detected objects from detection result with position information.

        Args:
            detection_result: Detection result from YOLOv8
            confidence_threshold: Minimum confidence threshold

        Returns:
            List of dictionaries with detected object information
        """
        boxes = detection_result.boxes.xyxy.cpu().numpy()
        classes = detection_result.boxes.cls.cpu().numpy().astype(int)
        confidences = detection_result.boxes.conf.cpu().numpy()

        # Image dimensions
        img_height, img_width = detection_result.orig_shape[:2]

        detected_objects = []
        for box, class_id, confidence in zip(boxes, classes, confidences):
            # Skip objects with confidence below threshold
            if confidence < confidence_threshold:
                continue

            x1, y1, x2, y2 = box
            width = x2 - x1
            height = y2 - y1

            # Center point
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            # Normalized positions (0-1)
            norm_x = center_x / img_width
            norm_y = center_y / img_height
            norm_width = width / img_width
            norm_height = height / img_height

            # Area calculation
            area = width * height
            norm_area = area / (img_width * img_height)

            # Region determination
            object_region = self._determine_region(norm_x, norm_y)

            detected_objects.append({
                "class_id": int(class_id),
                "class_name": self.class_names[int(class_id)],
                "confidence": float(confidence),
                "box": [float(x1), float(y1), float(x2), float(y2)],
                "center": [float(center_x), float(center_y)],
                "normalized_center": [float(norm_x), float(norm_y)],
                "size": [float(width), float(height)],
                "normalized_size": [float(norm_width), float(norm_height)],
                "area": float(area),
                "normalized_area": float(norm_area),
                "region": object_region
            })

        return detected_objects


    def _detect_scene_viewpoint(self, detected_objects: List[Dict]) -> Dict:
        """
        檢測場景視角並識別特殊場景模式。

        Args:
            detected_objects: 檢測到的物體列表

        Returns:
            Dict: 包含視角和場景模式信息的字典
        """
        if not detected_objects:
            return {"viewpoint": "eye_level", "patterns": []}

        # 從物體位置中提取信息
        patterns = []

        # 檢測行人位置模式
        pedestrian_objs = [obj for obj in detected_objects if obj["class_id"] == 0]

        # 檢查是否有足夠的行人來識別模式
        if len(pedestrian_objs) >= 4:
            pedestrian_positions = [obj["normalized_center"] for obj in pedestrian_objs]

            # 檢測十字交叉模式
            if self._detect_cross_pattern(pedestrian_positions):
                patterns.append("crosswalk_intersection")

            # 檢測多方向行人流
            directions = self._analyze_movement_directions(pedestrian_positions)
            if len(directions) >= 2:
                patterns.append("multi_directional_movement")

        # 檢查物體的大小一致性 - 在空中俯視圖中，物體大小通常更一致
        if len(detected_objects) >= 5:
            sizes = [obj.get("normalized_area", 0) for obj in detected_objects]
            size_variance = np.var(sizes) / (np.mean(sizes) ** 2)  # 標準化變異數，不會受到平均值影響

            if size_variance < 0.3:  # 低變異表示大小一致
                patterns.append("consistent_object_size")

        # 基本視角檢測
        viewpoint = self.enhance_descriptor._detect_viewpoint(detected_objects)

        # 根據檢測到的模式增強視角判斷
        if "crosswalk_intersection" in patterns and viewpoint != "aerial":
            # 如果檢測到斑馬線交叉但視角判斷不是空中視角，優先採用模式判斷
            viewpoint = "aerial"

        return {
            "viewpoint": viewpoint,
            "patterns": patterns
        }

    def _detect_cross_pattern(self, positions):
        """
        檢測位置中的十字交叉模式

        Args:
            positions: 位置列表 [[x1, y1], [x2, y2], ...]

        Returns:
            bool: 是否檢測到十字交叉模式
        """
        if len(positions) < 8:  # 需要足夠多的點
            return False

        # 提取 x 和 y 坐標
        x_coords = [pos[0] for pos in positions]
        y_coords = [pos[1] for pos in positions]

        # 檢測 x 和 y 方向的聚類
        x_clusters = []
        y_clusters = []

        # 簡化的聚類分析
        x_mean = np.mean(x_coords)
        y_mean = np.mean(y_coords)

        # 計算在中心線附近的點
        near_x_center = sum(1 for x in x_coords if abs(x - x_mean) < 0.1)
        near_y_center = sum(1 for y in y_coords if abs(y - y_mean) < 0.1)

        # 如果有足夠的點在中心線附近，可能是十字交叉
        return near_x_center >= 3 and near_y_center >= 3

    def _analyze_movement_directions(self, positions):
        """
        分析位置中的移動方向

        Args:
            positions: 位置列表 [[x1, y1], [x2, y2], ...]

        Returns:
            list: 檢測到的主要方向
        """
        if len(positions) < 6:
            return []

        # extract x 和 y 坐標
        x_coords = [pos[0] for pos in positions]
        y_coords = [pos[1] for pos in positions]

        directions = []

        # horizontal move (left --> right)
        x_std = np.std(x_coords)
        x_range = max(x_coords) - min(x_coords)

        # vertical move(up --> down)
        y_std = np.std(y_coords)
        y_range = max(y_coords) - min(y_coords)

        # 足夠大的範圍表示該方向有運動
        if x_range > 0.4:
            directions.append("horizontal")
        if y_range > 0.4:
            directions.append("vertical")

        return directions

    def _identify_functional_zones(self, detected_objects: List[Dict], scene_type: str) -> Dict:
        """
        Identify functional zones within the scene with improved detection for different viewpoints
        and cultural contexts.

        Args:
            detected_objects: List of detected objects
            scene_type: Identified scene type

        Returns:
            Dictionary of functional zones with their descriptions
        """
        # Group objects by category and region
        category_regions = {}

        for obj in detected_objects:
            # Find object category
            category = "other"
            for cat_name, cat_ids in self.OBJECT_CATEGORIES.items():
                if obj["class_id"] in cat_ids:
                    category = cat_name
                    break

            # Add to category-region mapping
            if category not in category_regions:
                category_regions[category] = {}

            region = obj["region"]
            if region not in category_regions[category]:
                category_regions[category][region] = []

            category_regions[category][region].append(obj)

        # Identify zones based on object groupings
        zones = {}

        # Detect viewpoint to adjust zone identification strategy
        viewpoint = self._detect_scene_viewpoint(detected_objects)

        # Choose appropriate zone identification strategy based on scene type and viewpoint
        if scene_type in ["living_room", "bedroom", "dining_area", "kitchen", "office_workspace", "meeting_room"]:
            # Indoor scenes
            zones.update(self._identify_indoor_zones(category_regions, detected_objects, scene_type))
        elif scene_type in ["city_street", "parking_lot", "park_area"]:
            # Outdoor general scenes
            zones.update(self._identify_outdoor_general_zones(category_regions, detected_objects, scene_type))
        elif "aerial" in scene_type or viewpoint == "aerial":
            # Aerial viewpoint scenes
            zones.update(self._identify_aerial_view_zones(category_regions, detected_objects, scene_type))
        elif "asian" in scene_type:
            # Asian cultural context scenes
            zones.update(self._identify_asian_cultural_zones(category_regions, detected_objects, scene_type))
        elif scene_type == "urban_intersection":
            # Specific urban intersection logic
            zones.update(self._identify_intersection_zones(category_regions, detected_objects, viewpoint))
        elif scene_type == "financial_district":
            # Financial district specific logic
            zones.update(self._identify_financial_district_zones(category_regions, detected_objects))
        elif scene_type == "upscale_dining":
            # Upscale dining specific logic
            zones.update(self._identify_upscale_dining_zones(category_regions, detected_objects))
        else:
            # Default zone identification for other scene types
            zones.update(self._identify_default_zones(category_regions, detected_objects))

        # If no zones were identified, try the default approach
        if not zones:
            zones.update(self._identify_default_zones(category_regions, detected_objects))

        return zones

    def _identify_indoor_zones(self, category_regions: Dict, detected_objects: List[Dict], scene_type: str) -> Dict:
        """
        Identify functional zones for indoor scenes.

        Args:
            category_regions: Objects grouped by category and region
            detected_objects: List of detected objects
            scene_type: Specific indoor scene type

        Returns:
            Dict: Indoor functional zones
        """
        zones = {}

        # Seating/social zone
        if "furniture" in category_regions:
            furniture_regions = category_regions["furniture"]
            main_furniture_region = max(furniture_regions.items(),
                                    key=lambda x: len(x[1]),
                                    default=(None, []))

            if main_furniture_region[0] is not None and len(main_furniture_region[1]) >= 2:
                zone_objects = [obj["class_name"] for obj in main_furniture_region[1]]
                zones["social_zone"] = {
                    "region": main_furniture_region[0],
                    "objects": zone_objects,
                    "description": f"Social or seating area with {', '.join(zone_objects)}"
                }

        # Entertainment zone
        if "electronics" in category_regions:
            electronics_items = []
            for region_objects in category_regions["electronics"].values():
                electronics_items.extend([obj["class_name"] for obj in region_objects])

            if electronics_items:
                zones["entertainment_zone"] = {
                    "region": self._find_main_region(category_regions.get("electronics", {})),
                    "objects": electronics_items,
                    "description": f"Entertainment or media area with {', '.join(electronics_items)}"
                }

        # Dining/food zone
        food_zone_categories = ["kitchen_items", "food"]
        food_items = []
        food_regions = {}

        for category in food_zone_categories:
            if category in category_regions:
                for region, objects in category_regions[category].items():
                    if region not in food_regions:
                        food_regions[region] = []
                    food_regions[region].extend(objects)
                    food_items.extend([obj["class_name"] for obj in objects])

        if food_items:
            main_food_region = max(food_regions.items(),
                                key=lambda x: len(x[1]),
                                default=(None, []))

            if main_food_region[0] is not None:
                zones["dining_zone"] = {
                    "region": main_food_region[0],
                    "objects": list(set(food_items)),
                    "description": f"Dining or food preparation area with {', '.join(list(set(food_items))[:3])}"
                }

        # Work/study zone - enhanced to detect even when scene_type is not explicitly office
        work_items = []
        work_regions = {}

        for obj in detected_objects:
            if obj["class_id"] in [56, 60, 63, 64, 66, 73]:  # chair, table, laptop, mouse, keyboard, book
                region = obj["region"]
                if region not in work_regions:
                    work_regions[region] = []
                work_regions[region].append(obj)
                work_items.append(obj["class_name"])

        # Check for laptop and table/chair combinations that suggest a workspace
        has_laptop = any(obj["class_id"] == 63 for obj in detected_objects)
        has_keyboard = any(obj["class_id"] == 66 for obj in detected_objects)
        has_table = any(obj["class_id"] == 60 for obj in detected_objects)
        has_chair = any(obj["class_id"] == 56 for obj in detected_objects)

        # If we have electronics with furniture in the same region, likely a workspace
        workspace_detected = (has_laptop or has_keyboard) and (has_table or has_chair)

        if (workspace_detected or scene_type in ["office_workspace", "meeting_room"]) and work_items:
            main_work_region = max(work_regions.items(),
                                key=lambda x: len(x[1]),
                                default=(None, []))

            if main_work_region[0] is not None:
                zones["workspace_zone"] = {
                    "region": main_work_region[0],
                    "objects": list(set(work_items)),
                    "description": f"Work or study area with {', '.join(list(set(work_items))[:3])}"
                }

        # Bedroom-specific zones
        if scene_type == "bedroom":
            bed_objects = [obj for obj in detected_objects if obj["class_id"] == 59]  # Bed
            if bed_objects:
                bed_region = bed_objects[0]["region"]
                zones["sleeping_zone"] = {
                    "region": bed_region,
                    "objects": ["bed"],
                    "description": "Sleeping area with bed"
                }

        # Kitchen-specific zones
        if scene_type == "kitchen":
            # Look for appliances (refrigerator, oven, microwave, sink)
            appliance_ids = [68, 69, 71, 72]  # microwave, oven, sink, refrigerator
            appliance_objects = [obj for obj in detected_objects if obj["class_id"] in appliance_ids]

            if appliance_objects:
                appliance_regions = {}
                for obj in appliance_objects:
                    region = obj["region"]
                    if region not in appliance_regions:
                        appliance_regions[region] = []
                    appliance_regions[region].append(obj)

                if appliance_regions:
                    main_appliance_region = max(appliance_regions.items(),
                                            key=lambda x: len(x[1]),
                                            default=(None, []))

                    if main_appliance_region[0] is not None:
                        appliance_names = [obj["class_name"] for obj in main_appliance_region[1]]
                        zones["kitchen_appliance_zone"] = {
                            "region": main_appliance_region[0],
                            "objects": appliance_names,
                            "description": f"Kitchen appliance area with {', '.join(appliance_names)}"
                        }

        return zones

    def _identify_intersection_zones(self, category_regions: Dict, detected_objects: List[Dict], viewpoint: str) -> Dict:
        """
        Identify functional zones for urban intersections with enhanced spatial awareness.

        Args:
            category_regions: Objects grouped by category and region
            detected_objects: List of detected objects
            viewpoint: Detected viewpoint

        Returns:
            Dict: Refined intersection functional zones
        """
        zones = {}

        # Get pedestrians, vehicles and traffic signals
        pedestrian_objs = [obj for obj in detected_objects if obj["class_id"] == 0]
        vehicle_objs = [obj for obj in detected_objects if obj["class_id"] in [1, 2, 3, 5, 7]]  # bicycle, car, motorcycle, bus, truck
        traffic_light_objs = [obj for obj in detected_objects if obj["class_id"] == 9]

        # Create distribution maps for better spatial understanding
        regions_distribution = self._create_distribution_map(detected_objects)

        # Analyze pedestrian crossing patterns
        crossing_zones = self._analyze_crossing_patterns(pedestrian_objs, traffic_light_objs, regions_distribution)
        zones.update(crossing_zones)

        # Analyze vehicle traffic zones with directional awareness
        traffic_zones = self._analyze_traffic_zones(vehicle_objs, regions_distribution)
        zones.update(traffic_zones)

        # Identify traffic control zones based on signal placement
        if traffic_light_objs:
            # Group traffic lights by region for better organization
            signal_regions = {}
            for obj in traffic_light_objs:
                region = obj["region"]
                if region not in signal_regions:
                    signal_regions[region] = []
                signal_regions[region].append(obj)

            # Create traffic control zones for each region with signals
            for idx, (region, signals) in enumerate(signal_regions.items()):
                # Check if this region has a directional name
                direction = self._get_directional_description(region)

                zones[f"traffic_control_zone_{idx+1}"] = {
                    "region": region,
                    "objects": ["traffic light"] * len(signals),
                    "description": f"Traffic control area with {len(signals)} traffic signals" +
                                (f" in {direction} area" if direction else "")
                }

        return zones

    def _analyze_crossing_patterns(self, pedestrians: List[Dict], traffic_lights: List[Dict],
                                region_distribution: Dict) -> Dict:
        """
        Analyze pedestrian crossing patterns to identify crosswalk zones.

        Args:
            pedestrians: List of pedestrian objects
            traffic_lights: List of traffic light objects
            region_distribution: Distribution of objects by region

        Returns:
            Dict: Identified crossing zones
        """
        crossing_zones = {}

        if not pedestrians:
            return crossing_zones

        # Group pedestrians by region
        pedestrian_regions = {}
        for p in pedestrians:
            region = p["region"]
            if region not in pedestrian_regions:
                pedestrian_regions[region] = []
            pedestrian_regions[region].append(p)

        # Sort regions by pedestrian count to find main crossing areas
        sorted_regions = sorted(pedestrian_regions.items(), key=lambda x: len(x[1]), reverse=True)

        # Create crossing zones for regions with pedestrians
        for idx, (region, peds) in enumerate(sorted_regions[:2]):  # Focus on top 2 regions
            # Check if there are traffic lights nearby to indicate a crosswalk
            has_nearby_signals = any(t["region"] == region for t in traffic_lights)

            # Create crossing zone with descriptive naming
            zone_name = f"crossing_zone_{idx+1}"
            direction = self._get_directional_description(region)

            description = f"Pedestrian crossing area with {len(peds)} "
            description += "person" if len(peds) == 1 else "people"
            if direction:
                description += f" in {direction} direction"
            if has_nearby_signals:
                description += " near traffic signals"

            crossing_zones[zone_name] = {
                "region": region,
                "objects": ["pedestrian"] * len(peds),
                "description": description
            }

        return crossing_zones

    def _analyze_traffic_zones(self, vehicles: List[Dict], region_distribution: Dict) -> Dict:
        """
        Analyze vehicle distribution to identify traffic zones with directional awareness.

        Args:
            vehicles: List of vehicle objects
            region_distribution: Distribution of objects by region

        Returns:
            Dict: Identified traffic zones
        """
        traffic_zones = {}

        if not vehicles:
            return traffic_zones

        # Group vehicles by region
        vehicle_regions = {}
        for v in vehicles:
            region = v["region"]
            if region not in vehicle_regions:
                vehicle_regions[region] = []
            vehicle_regions[region].append(v)

        # Create traffic zones for regions with vehicles
        main_traffic_region = max(vehicle_regions.items(), key=lambda x: len(x[1]), default=(None, []))

        if main_traffic_region[0] is not None:
            region = main_traffic_region[0]
            vehicles_in_region = main_traffic_region[1]

            # Get a list of vehicle types for description
            vehicle_types = [v["class_name"] for v in vehicles_in_region]
            unique_types = list(set(vehicle_types))

            # Get directional description
            direction = self._get_directional_description(region)

            # Create descriptive zone
            traffic_zones["vehicle_zone"] = {
                "region": region,
                "objects": vehicle_types,
                "description": f"Vehicle traffic area with {', '.join(unique_types[:3])}" +
                            (f" in {direction} area" if direction else "")
            }

            # If vehicles are distributed across multiple regions, create secondary zones
            if len(vehicle_regions) > 1:
                # Get second most populated region
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

    def _get_directional_description(self, region: str) -> str:
        """
        Convert region name to a directional description.

        Args:
            region: Region name from the grid

        Returns:
            str: Directional description
        """
        if "top" in region and "left" in region:
            return "northwest"
        elif "top" in region and "right" in region:
            return "northeast"
        elif "bottom" in region and "left" in region:
            return "southwest"
        elif "bottom" in region and "right" in region:
            return "southeast"
        elif "top" in region:
            return "north"
        elif "bottom" in region:
            return "south"
        elif "left" in region:
            return "west"
        elif "right" in region:
            return "east"
        else:
            return "central"

    def _create_distribution_map(self, detected_objects: List[Dict]) -> Dict:
        """
        Create a distribution map of objects across regions for spatial analysis.

        Args:
            detected_objects: List of detected objects

        Returns:
            Dict: Distribution map of objects by region and class
        """
        distribution = {}

        # Initialize all regions
        for region in self.regions.keys():
            distribution[region] = {
                "total": 0,
                "objects": {},
                "density": 0
            }

        # Populate the distribution
        for obj in detected_objects:
            region = obj["region"]
            class_id = obj["class_id"]
            class_name = obj["class_name"]

            distribution[region]["total"] += 1

            if class_id not in distribution[region]["objects"]:
                distribution[region]["objects"][class_id] = {
                    "name": class_name,
                    "count": 0,
                    "positions": []
                }

            distribution[region]["objects"][class_id]["count"] += 1

            # Store position for spatial relationship analysis
            if "normalized_center" in obj:
                distribution[region]["objects"][class_id]["positions"].append(obj["normalized_center"])

        # Calculate object density for each region
        for region, data in distribution.items():
            # Assuming all regions are equal size in the grid
            data["density"] = data["total"] / 1

        return distribution

    def _identify_asian_cultural_zones(self, category_regions: Dict, detected_objects: List[Dict], scene_type: str) -> Dict:
        """
        Identify functional zones for scenes with Asian cultural context.

        Args:
            category_regions: Objects grouped by category and region
            detected_objects: List of detected objects
            scene_type: Specific scene type

        Returns:
            Dict: Asian cultural functional zones
        """
        zones = {}

        # Identify storefront zone
        storefront_items = []
        storefront_regions = {}

        # Since storefronts aren't directly detectable, infer from context
        # For example, look for regions with signs, people, and smaller objects
        sign_regions = set()
        for obj in detected_objects:
            if obj["class_id"] == 0:  # Person
                region = obj["region"]
                if region not in storefront_regions:
                    storefront_regions[region] = []
                storefront_regions[region].append(obj)

                # Add regions with people as potential storefront areas
                sign_regions.add(region)

        # Use the areas with most people as storefront zones
        if storefront_regions:
            main_storefront_regions = sorted(storefront_regions.items(),
                                        key=lambda x: len(x[1]),
                                        reverse=True)[:2]  # Top 2 regions

            for idx, (region, objs) in enumerate(main_storefront_regions):
                zones[f"commercial_zone_{idx+1}"] = {
                    "region": region,
                    "objects": [obj["class_name"] for obj in objs],
                    "description": f"Asian commercial storefront with pedestrian activity"
                }

        # Identify pedestrian pathway - enhanced to better detect linear pathways
        pathway_items = []
        pathway_regions = {}

        # Extract people for pathway analysis
        people_objs = [obj for obj in detected_objects if obj["class_id"] == 0]

        # Analyze if people form a line (typical of shopping streets)
        people_positions = [obj["normalized_center"] for obj in people_objs]

        structured_path = False
        if len(people_positions) >= 3:
            # Check if people are arranged along a similar y-coordinate (horizontal path)
            y_coords = [pos[1] for pos in people_positions]
            y_mean = sum(y_coords) / len(y_coords)
            y_variance = sum((y - y_mean)**2 for y in y_coords) / len(y_coords)

            horizontal_path = y_variance < 0.05  # Low variance indicates horizontal alignment

            # Check if people are arranged along a similar x-coordinate (vertical path)
            x_coords = [pos[0] for pos in people_positions]
            x_mean = sum(x_coords) / len(x_coords)
            x_variance = sum((x - x_mean)**2 for x in x_coords) / len(x_coords)

            vertical_path = x_variance < 0.05  # Low variance indicates vertical alignment

            structured_path = horizontal_path or vertical_path
            path_direction = "horizontal" if horizontal_path else "vertical" if vertical_path else "meandering"

        # Collect pathway objects (people, bicycles, motorcycles in middle area)
        for obj in detected_objects:
            if obj["class_id"] in [0, 1, 3]:  # Person, bicycle, motorcycle
                y_pos = obj["normalized_center"][1]
                # Group by vertical position (middle of image likely pathway)
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
                "region": "middle_center",  # Assumption: pathway often in middle
                "objects": list(set(pathway_items)),
                "description": path_desc
            }

        # Identify vendor zone (small stalls/shops - inferred from context)
        has_small_objects = any(obj["class_id"] in [24, 26, 39, 41] for obj in detected_objects)  # bags, bottles, cups
        has_people = any(obj["class_id"] == 0 for obj in detected_objects)

        if has_small_objects and has_people:
            # Likely vendor areas are where people and small objects cluster
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

        # For night markets, identify illuminated zones
        if scene_type == "asian_night_market":
            # Night markets typically have bright spots for food stalls
            # This would be enhanced with lighting analysis integration
            zones["food_stall_zone"] = {
                "region": "middle_center",
                "objects": ["inferred food stalls"],
                "description": "Food stall area typical of Asian night markets"
            }

        return zones

    def _identify_upscale_dining_zones(self, category_regions: Dict, detected_objects: List[Dict]) -> Dict:
        """
        Identify functional zones for upscale dining settings.

        Args:
            category_regions: Objects grouped by category and region
            detected_objects: List of detected objects

        Returns:
            Dict: Upscale dining functional zones
        """
        zones = {}

        # Identify dining table zone
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

        # Identify decorative zone with enhanced detection
        decor_items = []
        decor_regions = {}

        # Look for decorative elements (vases, wine glasses, unused dishes)
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

        # Identify seating arrangement zone
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

        # Identify serving area (if different from dining area)
        serving_items = []
        serving_regions = {}

        # Serving areas might have bottles, bowls, containers
        for obj in detected_objects:
            if obj["class_id"] in [39, 45]:  # Bottle, bowl
                # Check if it's in a different region from the main dining table
                if "formal_dining_zone" in zones and obj["region"] != zones["formal_dining_zone"]["region"]:
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

    def _identify_financial_district_zones(self, category_regions: Dict, detected_objects: List[Dict]) -> Dict:
        """
        Identify functional zones for financial district scenes.

        Args:
            category_regions: Objects grouped by category and region
            detected_objects: List of detected objects

        Returns:
            Dict: Financial district functional zones
        """
        zones = {}

        # Identify traffic zone
        traffic_items = []
        traffic_regions = {}

        for obj in detected_objects:
            if obj["class_id"] in [1, 2, 3, 5, 6, 7, 9]:  # Various vehicles and traffic lights
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

        # Building zones on the sides (inferred from scene context)
        # Enhanced to check if there are actual regions that might contain buildings
        # Check for regions without vehicles or pedestrians - likely building areas
        left_side_regions = ["top_left", "middle_left", "bottom_left"]
        right_side_regions = ["top_right", "middle_right", "bottom_right"]

        # Check left side
        left_building_evidence = True
        for region in left_side_regions:
            # If many vehicles or people in this region, less likely to be buildings
            vehicle_in_region = any(obj["region"] == region and obj["class_id"] in [1, 2, 3, 5, 7]
                                for obj in detected_objects)
            people_in_region = any(obj["region"] == region and obj["class_id"] == 0
                                for obj in detected_objects)

            if vehicle_in_region or people_in_region:
                left_building_evidence = False
                break

        # Check right side
        right_building_evidence = True
        for region in right_side_regions:
            # If many vehicles or people in this region, less likely to be buildings
            vehicle_in_region = any(obj["region"] == region and obj["class_id"] in [1, 2, 3, 5, 7]
                                for obj in detected_objects)
            people_in_region = any(obj["region"] == region and obj["class_id"] == 0
                                for obj in detected_objects)

            if vehicle_in_region or people_in_region:
                right_building_evidence = False
                break

        # Add building zones if evidence supports them
        if left_building_evidence:
            zones["building_zone_left"] = {
                "region": "middle_left",
                "objects": ["building"],  # Inferred
                "description": "Tall buildings line the left side of the street"
            }

        if right_building_evidence:
            zones["building_zone_right"] = {
                "region": "middle_right",
                "objects": ["building"],  # Inferred
                "description": "Tall buildings line the right side of the street"
            }

        # Identify pedestrian zone if people are present
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

    def _identify_aerial_view_zones(self, category_regions: Dict, detected_objects: List[Dict], scene_type: str) -> Dict:
        """
        Identify functional zones for scenes viewed from an aerial perspective.

        Args:
            category_regions: Objects grouped by category and region
            detected_objects: List of detected objects
            scene_type: Specific scene type

        Returns:
            Dict: Aerial view functional zones
        """
        zones = {}

        # For aerial views, we focus on patterns and flows rather than specific zones

        # Identify pedestrian patterns
        people_objs = [obj for obj in detected_objects if obj["class_id"] == 0]
        if people_objs:
            # Convert positions to arrays for pattern analysis
            positions = np.array([obj["normalized_center"] for obj in people_objs])

            if len(positions) >= 3:
                # Calculate distribution metrics
                x_coords = positions[:, 0]
                y_coords = positions[:, 1]

                x_mean = np.mean(x_coords)
                y_mean = np.mean(y_coords)
                x_std = np.std(x_coords)
                y_std = np.std(y_coords)

                # Determine if people are organized in a linear pattern
                if x_std < 0.1 or y_std < 0.1:
                    # Linear distribution along one axis
                    pattern_direction = "vertical" if x_std < y_std else "horizontal"

                    zones["pedestrian_pattern"] = {
                        "region": "central",
                        "objects": ["person"] * len(people_objs),
                        "description": f"Aerial view shows a {pattern_direction} pedestrian movement pattern"
                    }
                else:
                    # More dispersed pattern
                    zones["pedestrian_distribution"] = {
                        "region": "wide",
                        "objects": ["person"] * len(people_objs),
                        "description": f"Aerial view shows pedestrians distributed across the area"
                    }

        # Identify vehicle patterns for traffic analysis
        vehicle_objs = [obj for obj in detected_objects if obj["class_id"] in [1, 2, 3, 5, 6, 7]]
        if vehicle_objs:
            # Convert positions to arrays for pattern analysis
            positions = np.array([obj["normalized_center"] for obj in vehicle_objs])

            if len(positions) >= 2:
                # Calculate distribution metrics
                x_coords = positions[:, 0]
                y_coords = positions[:, 1]

                x_mean = np.mean(x_coords)
                y_mean = np.mean(y_coords)
                x_std = np.std(x_coords)
                y_std = np.std(y_coords)

                # Determine if vehicles are organized in lanes
                if x_std < y_std * 0.5:
                    # Vehicles aligned vertically - indicates north-south traffic
                    zones["vertical_traffic_flow"] = {
                        "region": "central_vertical",
                        "objects": [obj["class_name"] for obj in vehicle_objs[:5]],
                        "description": "North-south traffic flow visible from aerial view"
                    }
                elif y_std < x_std * 0.5:
                    # Vehicles aligned horizontally - indicates east-west traffic
                    zones["horizontal_traffic_flow"] = {
                        "region": "central_horizontal",
                        "objects": [obj["class_name"] for obj in vehicle_objs[:5]],
                        "description": "East-west traffic flow visible from aerial view"
                    }
                else:
                    # Vehicles in multiple directions - indicates intersection
                    zones["intersection_traffic"] = {
                        "region": "central",
                        "objects": [obj["class_name"] for obj in vehicle_objs[:5]],
                        "description": "Multi-directional traffic at intersection visible from aerial view"
                    }

        # For intersection specific aerial views, identify crossing patterns
        if "intersection" in scene_type:
            # Check for traffic signals
            traffic_light_objs = [obj for obj in detected_objects if obj["class_id"] == 9]
            if traffic_light_objs:
                zones["traffic_control_pattern"] = {
                    "region": "intersection",
                    "objects": ["traffic light"] * len(traffic_light_objs),
                    "description": f"Intersection traffic control with {len(traffic_light_objs)} signals visible from above"
                }

            # Crosswalks are inferred from context in aerial views
            zones["crossing_pattern"] = {
                "region": "central",
                "objects": ["inferred crosswalk"],
                "description": "Crossing pattern visible from aerial perspective"
            }

        # For plaza aerial views, identify gathering patterns
        if "plaza" in scene_type:
            # Plazas typically have central open area with people
            if people_objs:
                # Check if people are clustered in central region
                central_people = [obj for obj in people_objs
                                if "middle" in obj["region"]]

                if central_people:
                    zones["central_gathering"] = {
                        "region": "middle_center",
                        "objects": ["person"] * len(central_people),
                        "description": f"Central plaza gathering area with {len(central_people)} people viewed from above"
                    }

        return zones

    def _identify_outdoor_general_zones(self, category_regions: Dict, detected_objects: List[Dict], scene_type: str) -> Dict:
        """
        Identify functional zones for general outdoor scenes.

        Args:
            category_regions: Objects grouped by category and region
            detected_objects: List of detected objects
            scene_type: Specific outdoor scene type

        Returns:
            Dict: Outdoor functional zones
        """
        zones = {}

        # Identify pedestrian zones
        people_objs = [obj for obj in detected_objects if obj["class_id"] == 0]
        if people_objs:
            people_regions = {}
            for obj in people_objs:
                region = obj["region"]
                if region not in people_regions:
                    people_regions[region] = []
                people_regions[region].append(obj)

            if people_regions:
                # Find main pedestrian areas
                main_people_regions = sorted(people_regions.items(),
                                        key=lambda x: len(x[1]),
                                        reverse=True)[:2]  # Top 2 regions

                for idx, (region, objs) in enumerate(main_people_regions):
                    if len(objs) > 0:
                        zones[f"pedestrian_zone_{idx+1}"] = {
                            "region": region,
                            "objects": ["person"] * len(objs),
                            "description": f"Pedestrian area with {len(objs)} {'people' if len(objs) > 1 else 'person'}"
                        }

        # Identify vehicle zones for streets and parking lots
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

        # For park areas, identify recreational zones
        if scene_type == "park_area":
            # Look for recreational objects (sports balls, kites, etc.)
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

        # For parking lots, identify parking zones
        if scene_type == "parking_lot":
            # Look for parked cars with consistent spacing
            car_objs = [obj for obj in detected_objects if obj["class_id"] == 2]  # cars

            if len(car_objs) >= 3:
                # Check if cars are arranged in patterns (simplified)
                car_positions = [obj["normalized_center"] for obj in car_objs]

                # Check for row patterns by analyzing vertical positions
                y_coords = [pos[1] for pos in car_positions]
                y_clusters = {}

                # Simplified clustering - group cars by similar y-coordinates
                for i, y in enumerate(y_coords):
                    assigned = False
                    for cluster_y in y_clusters.keys():
                        if abs(y - cluster_y) < 0.1:  # Within 10% of image height
                            y_clusters[cluster_y].append(i)
                            assigned = True
                            break

                    if not assigned:
                        y_clusters[y] = [i]

                # If we have row patterns
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

    def _identify_default_zones(self, category_regions: Dict, detected_objects: List[Dict]) -> Dict:
        """
        Identify general functional zones when no specific scene type is matched.

        Args:
            category_regions: Objects grouped by category and region
            detected_objects: List of detected objects

        Returns:
            Dict: Default functional zones
        """
        zones = {}

        # Group objects by category and find main concentrations
        for category, regions in category_regions.items():
            if not regions:
                continue

            # Find region with most objects in this category
            main_region = max(regions.items(),
                        key=lambda x: len(x[1]),
                        default=(None, []))

            if main_region[0] is None or len(main_region[1]) < 2:
                continue

            # Create zone based on object category
            zone_objects = [obj["class_name"] for obj in main_region[1]]

            # Skip if too few objects
            if len(zone_objects) < 2:
                continue

            # Create appropriate zone name and description based on category
            if category == "furniture":
                zones["furniture_zone"] = {
                    "region": main_region[0],
                    "objects": zone_objects,
                    "description": f"Area with furniture including {', '.join(zone_objects[:3])}"
                }
            elif category == "electronics":
                zones["electronics_zone"] = {
                    "region": main_region[0],
                    "objects": zone_objects,
                    "description": f"Area with electronic devices including {', '.join(zone_objects[:3])}"
                }
            elif category == "kitchen_items":
                zones["dining_zone"] = {
                    "region": main_region[0],
                    "objects": zone_objects,
                    "description": f"Dining or food area with {', '.join(zone_objects[:3])}"
                }
            elif category == "vehicles":
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

        # Check for people groups
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

        return zones

    def _find_main_region(self, region_objects_dict: Dict) -> str:
        """Find the main region with the most objects"""
        if not region_objects_dict:
            return "unknown"

        return max(region_objects_dict.items(),
                key=lambda x: len(x[1]),
                default=("unknown", []))[0]

    def _find_main_region(self, region_objects_dict: Dict) -> str:
        """Find the main region with the most objects"""
        if not region_objects_dict:
            return "unknown"

        return max(region_objects_dict.items(),
                 key=lambda x: len(x[1]),
                 default=("unknown", []))[0]
