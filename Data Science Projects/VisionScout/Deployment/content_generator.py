import logging
import random
import re
from typing import Dict, List, Optional, Union, Any

class ContentGenerator:
    """
    內容生成器 - 負責基礎內容生成和佔位符替換邏輯

    此類別專門處理模板中的動態內容生成，包括物件摘要、
    場景特定內容生成，以及提供默認的替換字典。
    """

    def __init__(self):
        """初始化內容生成器"""
        self.logger = logging.getLogger(self.__class__.__name__)

        # 預載入默認替換內容 
        self.default_replacements = self._generate_default_replacements()

        self.logger.debug("ContentGenerator initialized successfully")

    def _generate_default_replacements(self) -> Dict[str, str]:
        """
        生成默認的模板替換內容

        Returns:
            Dict[str, str]: 默認替換內容字典
        """
        return {
            # 場景介紹相關
            "scene_introduction": "this scene",
            "location_prefix": "this location",
            "setting_description": "this setting",
            "area_description": "this area",
            "environment_description": "this environment",
            "spatial_introduction": "this space",

            # 室內相關
            "furniture": "various furniture pieces",
            "seating": "comfortable seating",
            "electronics": "entertainment devices",
            "bed_type": "a bed",
            "bed_location": "room",
            "bed_description": "sleeping arrangements",
            "extras": "personal items",
            "table_setup": "a dining table and chairs",
            "table_description": "a dining surface",
            "dining_items": "dining furniture and tableware",
            "appliances": "kitchen appliances",
            "kitchen_items": "cooking utensils and dishware",
            "cooking_equipment": "cooking equipment",
            "office_equipment": "work-related furniture and devices",
            "desk_setup": "a desk and chair",
            "computer_equipment": "electronic devices",

            # 室外/城市相關
            "traffic_description": "vehicles and pedestrians",
            "people_and_vehicles": "people and various vehicles",
            "street_elements": "urban infrastructure",
            "park_features": "benches and greenery",
            "outdoor_elements": "natural features",
            "park_description": "outdoor amenities",
            "store_elements": "merchandise displays",
            "shopping_activity": "customers browse and shop",
            "store_items": "products for sale",

            # 高級餐廳相關
            "design_elements": "elegant decor",
            "lighting": "stylish lighting fixtures",

            # 亞洲商業街相關
            "storefront_features": "compact shops",
            "pedestrian_flow": "people walking",
            "asian_elements": "distinctive cultural elements",
            "cultural_elements": "traditional design features",
            "signage": "colorful signs",
            "street_activities": "busy urban activity",

            # 金融區相關
            "buildings": "tall buildings",
            "traffic_elements": "vehicles",
            "skyscrapers": "high-rise buildings",
            "road_features": "wide streets",
            "architectural_elements": "modern architecture",
            "city_landmarks": "prominent structures",

            # 十字路口相關
            "crossing_pattern": "clearly marked pedestrian crossings",
            "pedestrian_behavior": "careful pedestrian movement",
            "pedestrian_density": "multiple groups of pedestrians",
            "traffic_pattern": "well-regulated traffic flow",
            "pedestrian_flow": "steady pedestrian movement",
            "traffic_description": "active urban traffic",
            "people_and_vehicles": "pedestrians and vehicles",
            "street_elements": "urban infrastructure elements",

            # 交通相關
            "transit_vehicles": "public transportation vehicles",
            "passenger_activity": "commuter movement",
            "transportation_modes": "various transit options",
            "passenger_needs": "waiting areas",
            "transit_infrastructure": "transit facilities",
            "passenger_movement": "commuter flow",

            # 購物區相關
            "retail_elements": "shops and displays",
            "store_types": "various retail establishments",
            "walkway_features": "pedestrian pathways",
            "commercial_signage": "store signs",
            "consumer_behavior": "shopping activities",

            # 空中視角相關
            "commercial_layout": "organized retail areas",
            "pedestrian_pattern": "people movement patterns",
            "gathering_features": "public gathering spaces",
            "movement_pattern": "crowd flow patterns",
            "urban_elements": "city infrastructure",
            "public_activity": "social interaction",

            # 文化特定元素
            "stall_elements": "vendor booths",
            "lighting_features": "decorative lights",
            "food_elements": "food offerings",
            "vendor_stalls": "market stalls",
            "nighttime_activity": "evening commerce",
            "cultural_lighting": "traditional lighting",
            "night_market_sounds": "lively market sounds",
            "evening_crowd_behavior": "nighttime social activity",
            "architectural_elements": "cultural buildings",
            "religious_structures": "sacred buildings",
            "decorative_features": "ornamental designs",
            "cultural_practices": "traditional activities",
            "temple_architecture": "religious structures",
            "sensory_elements": "atmospheric elements",
            "visitor_activities": "cultural experiences",
            "ritual_activities": "ceremonial practices",
            "cultural_symbols": "meaningful symbols",
            "architectural_style": "historical buildings",
            "historic_elements": "traditional architecture",
            "urban_design": "city planning elements",
            "social_behaviors": "public interactions",
            "european_features": "European architectural details",
            "tourist_activities": "visitor activities",
            "local_customs": "regional practices",

            # 時間特定元素
            "lighting_effects": "artificial lighting",
            "shadow_patterns": "light and shadow",
            "urban_features": "city elements",
            "illuminated_elements": "lit structures",
            "evening_activities": "nighttime activities",
            "light_sources": "lighting points",
            "lit_areas": "illuminated spaces",
            "shadowed_zones": "darker areas",
            "illuminated_signage": "bright signs",
            "colorful_lighting": "multicolored lights",
            "neon_elements": "neon signs",
            "night_crowd_behavior": "evening social patterns",
            "light_displays": "lighting installations",
            "building_features": "architectural elements",
            "nightlife_activities": "evening entertainment",
            "lighting_modifier": "bright",

            # 混合環境元素
            "transitional_elements": "connecting features",
            "indoor_features": "interior elements",
            "outdoor_setting": "exterior spaces",
            "interior_amenities": "inside comforts",
            "exterior_features": "outside elements",
            "inside_elements": "interior design",
            "outside_spaces": "outdoor areas",
            "dual_environment_benefits": "combined settings",
            "passenger_activities": "waiting behaviors",
            "transportation_types": "transit vehicles",
            "sheltered_elements": "covered areas",
            "exposed_areas": "open sections",
            "waiting_behaviors": "passenger activities",
            "indoor_facilities": "inside services",
            "platform_features": "transit platform elements",
            "transit_routines": "transportation procedures",

            # 專門場所元素
            "seating_arrangement": "spectator seating",
            "playing_surface": "athletic field",
            "sporting_activities": "sports events",
            "spectator_facilities": "viewer accommodations",
            "competition_space": "sports arena",
            "sports_events": "athletic competitions",
            "viewing_areas": "audience sections",
            "field_elements": "field markings and equipment",
            "game_activities": "competitive play",
            "construction_equipment": "building machinery",
            "building_materials": "construction supplies",
            "construction_activities": "building work",
            "work_elements": "construction tools",
            "structural_components": "building structures",
            "site_equipment": "construction gear",
            "raw_materials": "building supplies",
            "construction_process": "building phases",
            "medical_elements": "healthcare equipment",
            "clinical_activities": "medical procedures",
            "facility_design": "healthcare layout",
            "healthcare_features": "medical facilities",
            "patient_interactions": "care activities",
            "equipment_types": "medical devices",
            "care_procedures": "health services",
            "treatment_spaces": "clinical areas",
            "educational_furniture": "learning furniture",
            "learning_activities": "educational practices",
            "instructional_design": "teaching layout",
            "classroom_elements": "school equipment",
            "teaching_methods": "educational approaches",
            "student_engagement": "learning participation",
            "learning_spaces": "educational areas",
            "educational_tools": "teaching resources",
            "knowledge_transfer": "learning exchanges"
        }

    def generate_objects_summary(self, detected_objects: List[Dict]) -> str:
        """
        基於檢測物件生成自然語言摘要，按重要性排序

        Args:
            detected_objects: 檢測到的物件列表

        Returns:
            str: 物件摘要描述
        """
        try:
            # detected_objects 裡有幾個 traffic light)
            tl_count = len([obj for obj in detected_objects if obj.get("class_name","") == "traffic light"])
            # print(f"[DEBUG] _generate_objects_summary 傳入的 detected_objects 中 traffic light: {tl_count} 個")
            for obj in detected_objects:
                if obj.get("class_name","") == "traffic light":
                    print(f"    - conf={obj.get('confidence',0):.4f}, bbox={obj.get('bbox')}, region={obj.get('region')}")

            if not detected_objects:
                return "various elements"

            # 計算物件統計 
            object_counts = {}
            total_confidence = 0

            for obj in detected_objects:
                class_name = obj.get("class_name", "unknown")
                confidence = obj.get("confidence", 0.5)

                if class_name not in object_counts:
                    object_counts[class_name] = {"count": 0, "total_confidence": 0}

                object_counts[class_name]["count"] += 1
                object_counts[class_name]["total_confidence"] += confidence
                total_confidence += confidence

            # 計算平均置信度並排序
            sorted_objects = []
            for class_name, stats in object_counts.items():
                avg_confidence = stats["total_confidence"] / stats["count"]
                count = stats["count"]

                # 重要性評分：結合數量和置信度
                importance_score = (count * 0.6) + (avg_confidence * 0.4)
                sorted_objects.append((class_name, count, importance_score))

            # 按重要性排序，取前5個最重要的物件
            sorted_objects.sort(key=lambda x: x[2], reverse=True)
            top_objects = sorted_objects[:5]

            # 生成自然語言描述
            descriptions = []
            for class_name, count, _ in top_objects:
                clean_name = class_name.replace('_', ' ')
                if count == 1:
                    article = "an" if clean_name[0].lower() in 'aeiou' else "a"
                    descriptions.append(f"{article} {clean_name}")
                else:
                    descriptions.append(f"{count} {clean_name}s")

            # 組合描述
            if len(descriptions) == 1:
                return descriptions[0]
            elif len(descriptions) == 2:
                return f"{descriptions[0]} and {descriptions[1]}"
            else:
                return ", ".join(descriptions[:-1]) + f", and {descriptions[-1]}"

        except Exception as e:
            self.logger.warning(f"Error generating objects summary: {str(e)}")
            return "various elements"

    def get_placeholder_replacement(self, placeholder: str, fillers: Dict,
                                   all_replacements: Dict, detected_objects: List[Dict],
                                   scene_type: str) -> str:
        """
        獲取特定佔位符的替換內容，確保永遠不返回空值

        Args:
            placeholder: 佔位符名稱
            fillers: 模板填充器字典
            all_replacements: 所有替換內容字典
            detected_objects: 檢測到的物體列表
            scene_type: 場景類型

        Returns:
            str: 替換內容
        """
        try:
            # 優先處理動態內容生成的佔位符
            dynamic_placeholders = [
                'primary_objects', 'detected_objects_summary', 'main_objects',
                'functional_area', 'functional_zones_description', 'scene_elements'
            ]

            if placeholder in dynamic_placeholders:
                dynamic_content = self.generate_objects_summary(detected_objects)
                if dynamic_content and dynamic_content.strip():
                    return dynamic_content.strip()

            # 檢查預定義替換內容
            if placeholder in all_replacements:
                replacement = all_replacements[placeholder]
                if replacement and replacement.strip():
                    return replacement.strip()

            # 檢查物體模板填充器
            if placeholder in fillers:
                options = fillers[placeholder]
                if options and isinstance(options, list):
                    valid_options = [opt.strip() for opt in options if opt and str(opt).strip()]
                    if valid_options:
                        num_items = min(len(valid_options), random.randint(1, 3))
                        selected_items = random.sample(valid_options, num_items)

                        if len(selected_items) == 1:
                            return selected_items[0]
                        elif len(selected_items) == 2:
                            return f"{selected_items[0]} and {selected_items[1]}"
                        else:
                            return ", ".join(selected_items[:-1]) + f", and {selected_items[-1]}"

            # 基於檢測對象生成動態內容
            scene_specific_replacement = self.generate_scene_specific_content(
                placeholder, detected_objects, scene_type
            )
            if scene_specific_replacement and scene_specific_replacement.strip():
                return scene_specific_replacement.strip()

            # 通用備用字典 
            fallback_replacements = {
                # 交通和城市相關
                "crossing_pattern": "pedestrian crosswalks",
                "pedestrian_behavior": "people moving carefully",
                "traffic_pattern": "vehicle movement",
                "urban_elements": "city infrastructure",
                "street_elements": "urban features",
                "intersection_features": "traffic management systems",
                "pedestrian_density": "groups of people",
                "pedestrian_flow": "pedestrian movement",
                "traffic_description": "vehicle traffic",
                "people_and_vehicles": "pedestrians and cars",

                # 場景設置相關
                "scene_setting": "this urban environment",
                "location_context": "the area",
                "spatial_context": "the scene",
                "environmental_context": "this location",

                # 常見的家具和設備
                "furniture": "various furniture pieces",
                "seating": "seating arrangements",
                "electronics": "electronic devices",
                "appliances": "household appliances",

                # 活動和行為
                "activities": "various activities",
                "interactions": "people interacting",
                "movement": "movement patterns",

                # 照明和氛圍
                "lighting_conditions": "ambient lighting",
                "atmosphere": "the overall atmosphere",
                "ambiance": "environmental ambiance",

                # 空間描述
                "spatial_arrangement": "spatial organization",
                "layout": "the layout",
                "composition": "visual composition",

                # 物體和元素
                "objects": "various objects",
                "elements": "scene elements",
                "features": "notable features",
                "details": "observable details"
            }

            if placeholder in fallback_replacements:
                return fallback_replacements[placeholder]

            # 基於場景類型的智能默認值
            scene_based_defaults = self.get_scene_based_default(placeholder, scene_type)
            if scene_based_defaults:
                return scene_based_defaults

            # 最終備用：將下劃線轉換為有意義的短語
            cleaned_placeholder = placeholder.replace('_', ' ')

            # 對常見模式提供更好的默認值
            if placeholder.endswith('_pattern'):
                return f"{cleaned_placeholder.replace(' pattern', '')} arrangement"
            elif placeholder.endswith('_behavior'):
                return f"{cleaned_placeholder.replace(' behavior', '')} activity"
            elif placeholder.endswith('_description'):
                return f"{cleaned_placeholder.replace(' description', '')} elements"
            elif placeholder.endswith('_elements'):
                return cleaned_placeholder
            elif placeholder.endswith('_features'):
                return cleaned_placeholder
            else:
                return cleaned_placeholder if cleaned_placeholder != placeholder else "various elements"

        except Exception as e:
            self.logger.warning(f"Error getting replacement for placeholder '{placeholder}': {str(e)}")
            # 確保即使在異常情況下也返回有意義的內容
            return placeholder.replace('_', ' ') if placeholder else "scene elements"

    def get_scene_based_default(self, placeholder: str, scene_type: str) -> Optional[str]:
        """
        基於場景類型提供智能默認值

        Args:
            placeholder: 佔位符名稱
            scene_type: 場景類型

        Returns:
            Optional[str]: 場景特定的默認值或None
        """
        try:
            # 針對不同場景類型的特定默認值
            scene_defaults = {
                "urban_intersection": {
                    "crossing_pattern": "marked crosswalks",
                    "pedestrian_behavior": "pedestrians crossing carefully",
                    "traffic_pattern": "controlled traffic flow"
                },
                "city_street": {
                    "traffic_description": "urban vehicle traffic",
                    "street_elements": "city infrastructure",
                    "people_and_vehicles": "pedestrians and vehicles"
                },
                "living_room": {
                    "furniture": "comfortable living room furniture",
                    "seating": "sofas and chairs",
                    "electronics": "entertainment equipment"
                },
                "kitchen": {
                    "appliances": "kitchen appliances",
                    "cooking_equipment": "cooking tools and equipment"
                },
                "office_workspace": {
                    "office_equipment": "work furniture and devices",
                    "desk_setup": "desk and office chair"
                }
            }

            if scene_type in scene_defaults and placeholder in scene_defaults[scene_type]:
                return scene_defaults[scene_type][placeholder]

            return None

        except Exception as e:
            self.logger.warning(f"Error getting scene-based default for '{placeholder}' in '{scene_type}': {str(e)}")
            return None

    def generate_scene_specific_content(self, placeholder: str, detected_objects: List[Dict],
                                       scene_type: str) -> Optional[str]:
        """
        基於場景特定邏輯生成佔位符內容

        Args:
            placeholder: 佔位符名稱
            detected_objects: 檢測到的物體列表
            scene_type: 場景類型

        Returns:
            Optional[str]: 生成的內容或None
        """
        try:
            if placeholder == "furniture":
                # 提取家具物品
                furniture_ids = [56, 57, 58, 59, 60, 61]  # 家具類別ID
                furniture_objects = [obj for obj in detected_objects if obj.get("class_id") in furniture_ids]

                if furniture_objects:
                    furniture_names = [obj.get("class_name", "furniture") for obj in furniture_objects[:3]]
                    unique_names = list(set(furniture_names))
                    return ", ".join(unique_names) if len(unique_names) > 1 else unique_names[0]
                return "various furniture items"

            elif placeholder == "electronics":
                # 提取電子設備
                electronics_ids = [62, 63, 64, 65, 66, 67, 68, 69, 70]  # 電子設備類別ID
                electronics_objects = [obj for obj in detected_objects if obj.get("class_id") in electronics_ids]

                if electronics_objects:
                    electronics_names = [obj.get("class_name", "electronic device") for obj in electronics_objects[:3]]
                    unique_names = list(set(electronics_names))
                    return ", ".join(unique_names) if len(unique_names) > 1 else unique_names[0]
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
                    unique_names = list(set(seating_names))
                    return ", ".join(unique_names) if len(unique_names) > 1 else unique_names[0]
                return "seating arrangements"

            # 如果沒有匹配的特定邏輯，返回None
            return None

        except Exception as e:
            self.logger.warning(f"Error generating scene-specific content for '{placeholder}': {str(e)}")
            return None

    def get_emergency_replacement(self, placeholder: str) -> str:
        """
        獲取緊急替換值，確保不會產生語法錯誤

        Args:
            placeholder: 佔位符名稱

        Returns:
            str: 安全的替換值
        """
        emergency_replacements = {
            "crossing_pattern": "pedestrian walkways",
            "pedestrian_behavior": "people moving through the area",
            "traffic_pattern": "vehicle movement",
            "scene_setting": "this location",
            "urban_elements": "city features",
            "street_elements": "urban components"
        }

        if placeholder in emergency_replacements:
            return emergency_replacements[placeholder]

        # 基於佔位符名稱生成合理的替換
        cleaned = placeholder.replace('_', ' ')
        if len(cleaned.split()) > 1:
            return cleaned
        else:
            return f"various {cleaned}"
