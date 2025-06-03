import os
import json
from typing import Dict, List, Tuple, Any, Optional

from scene_type import SCENE_TYPES
from scene_detail_templates import SCENE_DETAIL_TEMPLATES
from object_template_fillers import OBJECT_TEMPLATE_FILLERS
from activity_templates import ACTIVITY_TEMPLATES
from safety_templates import SAFETY_TEMPLATES
from confidence_templates import CONFIDENCE_TEMPLATES

class SceneDescriptor:
    """
    Generates natural language descriptions of scenes.
    Handles scene descriptions, activity inference, and safety concerns identification.
    """

    def __init__(self, scene_types=None, object_categories=None):
        """
        Initialize the scene descriptor

        Args:
            scene_types: Dictionary of scene type definitions
        """
        self.scene_types = scene_types or {}
        self.SCENE_TYPES = scene_types or {}

        if object_categories:
            self.OBJECT_CATEGORIES = object_categories
        else:
            # 從 JSON 加載或使用默認值
            self.OBJECT_CATEGORIES = self._load_json_data("object_categories") or {
                "furniture": [56, 57, 58, 59, 60, 61],
                "electronics": [62, 63, 64, 65, 66, 67, 68, 69, 70],
                "kitchen_items": [39, 40, 41, 42, 43, 44, 45],
                "food": [46, 47, 48, 49, 50, 51, 52, 53, 54, 55],
                "vehicles": [1, 2, 3, 4, 5, 6, 7, 8],
                "personal_items": [24, 25, 26, 27, 28, 73, 78, 79]
            }

        # 加載所有模板數據
        self._load_templates()

    def _load_templates(self):
        """Load all template data from script or fallback to imported defaults"""
        self.confidence_templates = CONFIDENCE_TEMPLATES
        self.scene_detail_templates = SCENE_DETAIL_TEMPLATES
        self.object_template_fillers = OBJECT_TEMPLATE_FILLERS
        self.safety_templates = SAFETY_TEMPLATES
        self.activity_templates = ACTIVITY_TEMPLATES


    def _initialize_fallback_templates(self):
        """Initialize fallback templates when no external data is available"""
        # 只在無法從文件或導入加載時使用
        self.confidence_templates = {
            "high": "{description} {details}",
            "medium": "This appears to be {description} {details}",
            "low": "This might be {description}, but the confidence is low. {details}"
        }

        # 只提供最基本的模板作為後備
        self.scene_detail_templates = {
            "default": ["A space with various objects."]
        }

        self.object_template_fillers = {
            "default": ["various items"]
        }

        self.safety_templates = {
            "general": "Pay attention to {safety_element}."
        }

        self.activity_templates = {
            "default": ["General activity"]
        }

    def _get_alternative_scenes(self, scene_scores: Dict[str, float],
                            threshold: float, top_k: int = 2) -> List[Dict]:
        """
        Get alternative scene interpretations with their scores.

        Args:
            scene_scores: Dictionary of scene type scores
            threshold: Minimum confidence threshold
            top_k: Number of alternatives to return

        Returns:
            List of dictionaries with alternative scenes
        """
        # Sort scenes by score in descending order
        sorted_scenes = sorted(scene_scores.items(), key=lambda x: x[1], reverse=True)

        # Skip the first one (best match) and take the next top_k
        alternatives = []
        for scene_type, score in sorted_scenes[1:1+top_k]:
            if score >= threshold:
                alternatives.append({
                    "type": scene_type,
                    "name": self.SCENE_TYPES.get(scene_type, {}).get("name", "Unknown"),
                    "confidence": score
                })

        return alternatives


    def _infer_possible_activities(self, scene_type: str, detected_objects: List[Dict], enable_landmark: bool = True, scene_scores: Optional[Dict] = None) -> List[str]:
        """
        Infer possible activities based on scene type and detected objects.

        Args:
            scene_type: Identified scene type
            detected_objects: List of detected objects
            enable_landmark: Whether landmark detection is enabled
            scene_scores: Optional dictionary of scene type scores

        Returns:
            List of possible activities
        """
        activities = []

        # Dynamically replace landmark scene types when landmark detection is disabled
        if not enable_landmark and scene_type in ["tourist_landmark", "natural_landmark", "historical_monument"]:
            alternative_scene_type = self._get_alternative_scene_type(scene_type, detected_objects, scene_scores)
            print(f"Replacing landmark scene type '{scene_type}' with '{alternative_scene_type}' for activity inference")
            scene_type = alternative_scene_type

        # Process aerial view scenes
        if scene_type.startswith("aerial_view_"):
            if scene_type == "aerial_view_intersection":
                # Use predefined intersection activities
                activities.extend(self.activity_templates.get("aerial_view_intersection", []))

                # Add pedestrian and vehicle specific activities
                pedestrians = [obj for obj in detected_objects if obj["class_id"] == 0]
                vehicles = [obj for obj in detected_objects if obj["class_id"] in [2, 5, 7]]  # Car, bus, truck

                if pedestrians and vehicles:
                    activities.append("Waiting for an opportunity to cross the street")
                    activities.append("Obeying traffic signals")

            elif scene_type == "aerial_view_commercial_area":
                activities.extend(self.activity_templates.get("aerial_view_commercial_area", []))

            elif scene_type == "aerial_view_plaza":
                activities.extend(self.activity_templates.get("aerial_view_plaza", []))

            else:
                # Handle other undefined aerial view scenes
                aerial_activities = [
                    "Street crossing",
                    "Waiting for signals",
                    "Following traffic rules",
                    "Pedestrian movement"
                ]
                activities.extend(aerial_activities)

        # Add scene-specific activities from templates
        if scene_type in self.activity_templates:
            activities.extend(self.activity_templates[scene_type])
        elif "default" in self.activity_templates:
            activities.extend(self.activity_templates["default"])

        # Filter out landmark-related activities when landmark detection is disabled
        if not enable_landmark:
            filtered_activities = []
            landmark_keywords = ["sightseeing", "landmark", "tourist", "monument", "historical",
                                "guided tour", "photography", "cultural tourism", "heritage"]

            for activity in activities:
                if not any(keyword in activity.lower() for keyword in landmark_keywords):
                    filtered_activities.append(activity)

            activities = filtered_activities

        # If we filtered out all activities, add some generic ones based on scene type
        if not activities:
            generic_activities = {
                "city_street": ["Walking", "Commuting", "Shopping"],
                "intersection": ["Crossing the street", "Waiting for traffic signals"],
                "commercial_district": ["Shopping", "Walking", "Dining"],
                "pedestrian_area": ["Walking", "Socializing", "Shopping"],
                "park_area": ["Relaxing", "Walking", "Exercise"],
                "outdoor_natural_area": ["Walking", "Nature observation", "Relaxation"],
                "urban_architecture": ["Walking", "Urban exploration", "Photography"]
            }

            activities.extend(generic_activities.get(scene_type, ["Walking", "Observing surroundings"]))

        # Add activities based on detected objects
        detected_class_ids = [obj["class_id"] for obj in detected_objects]

        # Add activities based on specific object combinations
        if 62 in detected_class_ids and 57 in detected_class_ids:  # TV and sofa
            activities.append("Watching shows or movies")

        if 63 in detected_class_ids:  # laptop
            activities.append("Using a computer/laptop")

        if 67 in detected_class_ids:  # cell phone
            activities.append("Using a mobile phone")

        if 73 in detected_class_ids:  # book
            activities.append("Reading")

        if any(food_id in detected_class_ids for food_id in [46, 47, 48, 49, 50, 51, 52, 53, 54, 55]):
            activities.append("Eating or preparing food")

        # Person-specific activities
        if 0 in detected_class_ids:  # Person
            if any(vehicle in detected_class_ids for vehicle in [1, 2, 3, 5, 7]):  # Vehicles
                activities.append("Commuting or traveling")

            if 16 in detected_class_ids:  # Dog
                activities.append("Walking a dog")

            if 24 in detected_class_ids or 26 in detected_class_ids:  # Backpack or handbag
                activities.append("Carrying personal items")

            # Add more person count-dependent activities
            person_count = detected_class_ids.count(0)
            if person_count > 3:
                activities.append("Group gathering")
            elif person_count > 1:
                activities.append("Social interaction")

        # Add additional activities based on significant objects
        if 43 in detected_class_ids:  # cup
            activities.append("Drinking beverages")

        if 32 in detected_class_ids:  # sports ball
            activities.append("Playing sports")

        if 25 in detected_class_ids:  # umbrella
            activities.append("Sheltering from weather")

        # Add location-specific activities based on environment objects
        if any(furniture in detected_class_ids for furniture in [56, 57, 58, 59, 60]):  # furniture items
            activities.append("Using indoor facilities")

        if any(outdoor_item in detected_class_ids for outdoor_item in [13, 14, 15]):  # bench, outdoor items
            activities.append("Enjoying outdoor spaces")

        # Remove duplicates and ensure reasonable number of activities
        unique_activities = list(set(activities))

        # Limit to reasonable number (maximum 8 activities)
        if len(unique_activities) > 8:
            # Prioritize more specific activities over general ones
            general_activities = ["Walking", "Observing surroundings", "Commuting", "Using indoor facilities"]
            specific_activities = [a for a in unique_activities if a not in general_activities]

            # Take all specific activities first, then fill with general ones if needed
            if len(specific_activities) <= 8:
                result = specific_activities + general_activities[:8-len(specific_activities)]
            else:
                result = specific_activities[:8]
        else:
            result = unique_activities

        return result

    def _identify_safety_concerns(self, detected_objects: List[Dict], scene_type: str) -> List[str]:
        """
        Identify potential safety concerns based on objects and scene type.

        Args:
            detected_objects: List of detected objects
            scene_type: Identified scene type

        Returns:
            List of potential safety concerns
        """
        concerns = []
        detected_class_ids = [obj["class_id"] for obj in detected_objects]

        # General safety concerns
        if 42 in detected_class_ids or 43 in detected_class_ids:  # Fork or knife
            concerns.append("Sharp utensils present")

        if 76 in detected_class_ids:  # Scissors
            concerns.append("Cutting tools present")

        # Traffic-related concerns
        if scene_type in ["city_street", "parking_lot"]:
            if 0 in detected_class_ids:  # Person
                if any(vehicle in detected_class_ids for vehicle in [2, 3, 5, 7, 8]):  # Vehicles
                    concerns.append("Pedestrians near vehicles")

            if 9 in detected_class_ids:  # Traffic light
                concerns.append("Monitor traffic signals")

        # Identify crowded scenes
        person_count = detected_class_ids.count(0)
        if person_count > 5:
            concerns.append(f"Crowded area with multiple people ({person_count})")

        # Scene-specific concerns
        if scene_type == "kitchen":
            if 68 in detected_class_ids or 69 in detected_class_ids:  # Microwave or oven
                concerns.append("Hot cooking equipment")

        # Potentially unstable objects
        for obj in detected_objects:
            if obj["class_id"] in [39, 40, 41, 45]:  # Bottle, wine glass, cup, bowl
                if obj["region"] in ["top_left", "top_center", "top_right"] and obj["normalized_area"] > 0.05:
                    concerns.append(f"Elevated {obj['class_name']} might be unstable")

        # Upscale dining safety concerns
        if scene_type == "upscale_dining":
            # Check for fragile items
            if 40 in detected_class_ids:  # Wine glass
                concerns.append("Fragile glassware present")

            # Check for lit candles (can't directly detect but can infer from context)
            # Look for small bright spots that might be candles
            if any(obj["class_id"] == 41 for obj in detected_objects):  # Cup (which might include candle holders)
                # We can't reliably detect candles, but if the scene appears to be formal dining,
                # we can suggest this as a possibility
                concerns.append("Possible lit candles or decorative items requiring care")

            # Check for overcrowded table
            table_objs = [obj for obj in detected_objects if obj["class_id"] == 60]  # Dining table
            if table_objs:
                table_region = table_objs[0]["region"]
                items_on_table = 0

                for obj in detected_objects:
                    if obj["class_id"] in [39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55]:
                        if obj["region"] == table_region:
                            items_on_table += 1

                if items_on_table > 8:
                    concerns.append("Dining table has multiple items which should be handled with care")

        # Asian commercial street safety concerns
        elif scene_type == "asian_commercial_street":
            # Check for crowded walkways
            if 0 in detected_class_ids:  # Person
                person_count = detected_class_ids.count(0)
                if person_count > 3:
                    # Calculate person density (simplified)
                    person_positions = []
                    for obj in detected_objects:
                        if obj["class_id"] == 0:
                            person_positions.append(obj["normalized_center"])

                    if len(person_positions) >= 2:
                        # Calculate average distance between people
                        total_distance = 0
                        count = 0
                        for i in range(len(person_positions)):
                            for j in range(i+1, len(person_positions)):
                                p1 = person_positions[i]
                                p2 = person_positions[j]
                                distance = ((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)**0.5
                                total_distance += distance
                                count += 1

                        if count > 0:
                            avg_distance = total_distance / count
                            if avg_distance < 0.1:  # Close proximity
                                concerns.append("Crowded walkway with limited personal space")

            # Check for motorcycles/bicycles near pedestrians
            if (1 in detected_class_ids or 3 in detected_class_ids) and 0 in detected_class_ids:  # Bicycle/motorcycle and person
                concerns.append("Two-wheeled vehicles in pedestrian areas")

            # Check for potential trip hazards
            if scene_type == "asian_commercial_street" and "bottom" in " ".join([obj["region"] for obj in detected_objects if obj["class_id"] == 0]):
                # If people are in bottom regions, they might be walking on uneven surfaces
                concerns.append("Potential uneven walking surfaces in commercial area")

        # Financial district safety concerns
        elif scene_type == "financial_district":
            # Check for heavy traffic conditions
            vehicle_count = sum(1 for obj_id in detected_class_ids if obj_id in [2, 5, 7])  # Car, bus, truck
            if vehicle_count > 5:
                concerns.append("Heavy vehicle traffic in urban area")

            # Check for pedestrians crossing busy streets
            if 0 in detected_class_ids:  # Person
                person_count = detected_class_ids.count(0)
                vehicle_nearby = any(vehicle in detected_class_ids for vehicle in [2, 3, 5, 7])

                if person_count > 0 and vehicle_nearby:
                    concerns.append("Pedestrians navigating busy urban traffic")

            # Check for traffic signals
            if 9 in detected_class_ids:  # Traffic light
                concerns.append("Observe traffic signals when navigating this area")
            else:
                # If no traffic lights detected but it's a busy area, it's worth noting
                if vehicle_count > 3:
                    concerns.append("Busy traffic area potentially without visible traffic signals in view")

            # Time of day considerations
            vehicle_objs = [obj for obj in detected_objects if obj["class_id"] in [2, 5, 7]]
            if vehicle_objs and any("lighting_conditions" in obj for obj in detected_objects):
                # If vehicles are present and it might be evening/night
                concerns.append("Reduced visibility conditions during evening commute")

        # Urban intersection safety concerns
        elif scene_type == "urban_intersection":
            # Check for pedestrians in crosswalks
            pedestrian_objs = [obj for obj in detected_objects if obj["class_id"] == 0]
            vehicle_objs = [obj for obj in detected_objects if obj["class_id"] in [2, 3, 5, 7]]

            if pedestrian_objs:
                # Calculate distribution of pedestrians to see if they're crossing
                pedestrian_positions = [obj["normalized_center"] for obj in pedestrian_objs]

                # Simplified check for pedestrians in crossing pattern
                if len(pedestrian_positions) >= 3:
                    # Check if pedestrians are distributed across different regions
                    pedestrian_regions = set(obj["region"] for obj in pedestrian_objs)
                    if len(pedestrian_regions) >= 2:
                        concerns.append("Multiple pedestrians crossing the intersection")

            # Check for traffic signal observation
            if 9 in detected_class_ids:  # Traffic light
                concerns.append("Observe traffic signals when crossing")

            # Check for busy intersection
            if len(vehicle_objs) > 3:
                concerns.append("Busy intersection with multiple vehicles")

            # Check for pedestrians potentially jay-walking
            if pedestrian_objs and not 9 in detected_class_ids:  # People but no traffic lights
                concerns.append("Pedestrians should use designated crosswalks")

            # Visibility concerns based on lighting
            # This would be better with actual lighting data
            pedestrian_count = len(pedestrian_objs)
            if pedestrian_count > 5:
                concerns.append("High pedestrian density at crossing points")

        # Transit hub safety concerns
        elif scene_type == "transit_hub":
            # These would be for transit areas like train stations or bus terminals
            if 0 in detected_class_ids:  # Person
                person_count = detected_class_ids.count(0)
                if person_count > 8:
                    concerns.append("Crowded transit area requiring careful navigation")

            # Check for luggage/bags that could be trip hazards
            if 24 in detected_class_ids or 28 in detected_class_ids:  # Backpack or suitcase
                concerns.append("Luggage and personal items may create obstacles")

            # Public transportation vehicles
            if any(vehicle in detected_class_ids for vehicle in [5, 6, 7]):  # Bus, train, truck
                concerns.append("Stay clear of arriving and departing transit vehicles")

        # Shopping district safety concerns
        elif scene_type == "shopping_district":
            # Check for crowded shopping areas
            if 0 in detected_class_ids:  # Person
                person_count = detected_class_ids.count(0)
                if person_count > 5:
                    concerns.append("Crowded shopping area with multiple people")

            # Check for shopping bags and personal items
            if 24 in detected_class_ids or 26 in detected_class_ids:  # Backpack or handbag
                concerns.append("Mind personal belongings in busy retail environment")

            # Check for store entrances/exits which might have automatic doors
            # We can't directly detect this, but can infer from context
            if scene_type == "shopping_district" and 0 in detected_class_ids:
                concerns.append("Be aware of store entrances and exits with potential automatic doors")

        return concerns
