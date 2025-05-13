import os
import re
import json
import random
import numpy as np
from typing import Dict, List, Tuple, Any, Optional

from scene_type import SCENE_TYPES
from scene_detail_templates import SCENE_DETAIL_TEMPLATES
from object_template_fillers import OBJECT_TEMPLATE_FILLERS
from lighting_conditions import LIGHTING_CONDITIONS
from viewpoint_templates import VIEWPOINT_TEMPLATES
from cultural_templates import CULTURAL_TEMPLATES
from confifence_templates import CONFIDENCE_TEMPLATES

class EnhancedSceneDescriber:
    """
    Enhanced scene description generator with improved template handling,
    viewpoint awareness, and cultural context recognition.
    Provides detailed natural language descriptions of scenes based on
    detection results and scene classification.
    """

    def __init__(self, templates_db: Optional[Dict] = None, scene_types: Optional[Dict] = None):
        """
        Initialize the enhanced scene describer.

        Args:
            templates_db: Optional custom templates database
            scene_types: Dictionary of scene type definitions
        """
        # Load or use provided scene types
        self.scene_types = scene_types or self._load_default_scene_types()

        # Load templates database
        self.templates = templates_db or self._load_templates()

        # Initialize viewpoint detection parameters
        self._initialize_viewpoint_parameters()

    def _load_default_scene_types(self) -> Dict:
        """
        Load default scene types.

        Returns:
            Dict: Scene type definitions
        """

        return SCENE_TYPES

    def _load_templates(self) -> Dict:
        """
        Load description templates from imported Python modules.

        Returns:
            Dict: Template collections for different description components
        """
        templates = {}

        # 直接從導入的 Python 模組中獲取模板
        templates["scene_detail_templates"] = SCENE_DETAIL_TEMPLATES
        templates["object_template_fillers"] = OBJECT_TEMPLATE_FILLERS
        templates["viewpoint_templates"] = VIEWPOINT_TEMPLATES
        templates["cultural_templates"] = CULTURAL_TEMPLATES

        # 從 LIGHTING_CONDITIONS 獲取照明模板
        templates["lighting_templates"] = {
            key: data["general"] for key, data in LIGHTING_CONDITIONS.get("time_descriptions", {}).items()
        }

        # 設置默認的置信度模板
        templates["confidence_templates"] = {
            "high": "{description} {details}",
            "medium": "This appears to be {description} {details}",
            "low": "This might be {description}, but the confidence is low. {details}"
        }

        # 初始化其他必要的模板（現在這個函數簡化了很多）
        self._initialize_default_templates(templates)

        return templates

    def _initialize_default_templates(self, templates: Dict):
        """
        檢查模板字典並填充任何缺失的默認模板。

        在將模板移至專門的模組後，此方法主要作為安全機制，
        確保即使導入失敗或某些模板未在外部定義，系統仍能正常運行。

        Args:
            templates: 要檢查和更新的模板字典
        """
        # 檢查關鍵模板類型是否存在，如果不存在則添加默認值

        # 置信度模板 - 用於控制描述的語氣
        if "confidence_templates" not in templates:
            templates["confidence_templates"] = {
                "high": "{description} {details}",
                "medium": "This appears to be {description} {details}",
                "low": "This might be {description}, but the confidence is low. {details}"
            }

        # 場景細節模板 - 如果未從外部導入
        if "scene_detail_templates" not in templates:
            templates["scene_detail_templates"] = {
                "default": ["A space with various objects."]
            }

        # 物體填充模板 - 用於生成物體描述
        if "object_template_fillers" not in templates:
            templates["object_template_fillers"] = {
                "default": ["various items"]
            }

        # 視角模板 - 雖然我們現在從專門模組導入，但作為備份
        if "viewpoint_templates" not in templates:
            # 使用簡化版的默認視角模板
            templates["viewpoint_templates"] = {
                "eye_level": {
                    "prefix": "From eye level, ",
                    "observation": "the scene is viewed straight on."
                },
                "aerial": {
                    "prefix": "From above, ",
                    "observation": "the scene is viewed from a bird's-eye perspective."
                }
            }

        # 文化模板
        if "cultural_templates" not in templates:
            templates["cultural_templates"] = {
                "asian": {
                    "elements": ["cultural elements"],
                    "description": "The scene has Asian characteristics."
                },
                "european": {
                    "elements": ["architectural features"],
                    "description": "The scene has European characteristics."
                }
            }

        # 照明模板 - 用於描述光照條件
        if "lighting_templates" not in templates:
            templates["lighting_templates"] = {
                "day_clear": "The scene is captured during daylight.",
                "night": "The scene is captured at night.",
                "unknown": "The lighting conditions are not easily determined."
            }

    def _initialize_viewpoint_parameters(self):
        """
        Initialize parameters used for viewpoint detection.
        """
        self.viewpoint_params = {
            # Parameters for detecting aerial views
            "aerial_threshold": 0.7,  # High object density viewed from top
            "aerial_size_variance_threshold": 0.15,  # Low size variance in aerial views

            # Parameters for detecting low angle views
            "low_angle_threshold": 0.3,  # Bottom-heavy object distribution
            "vertical_size_ratio_threshold": 1.8,  # Vertical objects appear taller

            # Parameters for detecting elevated views
            "elevated_threshold": 0.6,  # Objects mostly in middle/bottom
            "elevated_top_threshold": 0.3  # Few objects at top of frame
        }


    def generate_description(self,
                        scene_type: str,
                        detected_objects: List[Dict],
                        confidence: float,
                        lighting_info: Optional[Dict] = None,
                        functional_zones: Optional[Dict] = None) -> str:
        """
        Generate enhanced scene description based on detection results, scene type,
        and additional contextual information.

        This is the main entry point that replaces the original _generate_scene_description.

        Args:
            scene_type: Identified scene type
            detected_objects: List of detected objects
            confidence: Scene classification confidence
            lighting_info: Optional lighting condition information
            functional_zones: Optional identified functional zones

        Returns:
            str: Natural language description of the scene
        """
        # Handle unknown scene type or very low confidence
        if scene_type == "unknown" or confidence < 0.4:
            return self._format_final_description(self._generate_generic_description(detected_objects, lighting_info))

        # Detect viewpoint
        viewpoint = self._detect_viewpoint(detected_objects)

        # Process aerial viewpoint scene types
        if viewpoint == "aerial":
            if "intersection" in scene_type or self._is_intersection(detected_objects):
                scene_type = "aerial_view_intersection"
            elif any(keyword in scene_type for keyword in ["commercial", "shopping", "retail"]):
                scene_type = "aerial_view_commercial_area"
            elif any(keyword in scene_type for keyword in ["plaza", "square"]):
                scene_type = "aerial_view_plaza"
            else:
                scene_type = "aerial_view_intersection"

        # Detect cultural context - only for non-aerial viewpoints
        cultural_context = None
        if viewpoint != "aerial":
            cultural_context = self._detect_cultural_context(scene_type, detected_objects)

        # Select appropriate template based on confidence
        if confidence > 0.75:
            confidence_level = "high"
        elif confidence > 0.5:
            confidence_level = "medium"
        else:
            confidence_level = "low"

        # Get base description for the scene type
        if viewpoint == "aerial":
            if 'base_description' not in locals():
                base_description = "An aerial view showing the layout and movement patterns from above"
        elif scene_type in self.scene_types:
            base_description = self.scene_types[scene_type].get("description", "A scene")
        else:
            base_description = "A scene"

        # Generate detailed scene information
        scene_details = self._generate_scene_details(
            scene_type,
            detected_objects,
            lighting_info,
            viewpoint
        )

        # Start with the base description
        description = base_description

        # If there's a secondary description from the scene type template, append it properly
        if scene_type in self.scene_types and "secondary_description" in self.scene_types[scene_type]:
            secondary_desc = self.scene_types[scene_type]["secondary_description"]
            if secondary_desc:
                description = self._smart_append(description, secondary_desc)

        # Improve description based on people count
        people_objs = [obj for obj in detected_objects if obj["class_id"] == 0]  # Person class
        if people_objs:
            people_count = len(people_objs)
            if people_count > 5:
                people_phrase = f"numerous people ({people_count})"
            else:
                people_phrase = f"{people_count} {'people' if people_count > 1 else 'person'}"

            # Add people information to the scene details if not already mentioned
            if "people" not in description.lower() and "pedestrian" not in description.lower():
                description = self._smart_append(description, f"The scene includes {people_phrase}")

        # Apply cultural context if detected (only for non-aerial viewpoints)
        if cultural_context and viewpoint != "aerial":
            cultural_elements = self._generate_cultural_elements(cultural_context)
            if cultural_elements:
                description = self._smart_append(description, cultural_elements)

        # Now append the detailed scene information if available
        if scene_details:
            # Use smart_append to ensure proper formatting between base description and details
            description = self._smart_append(description, scene_details)

        # Include lighting information if available
        lighting_description = ""
        if lighting_info and "time_of_day" in lighting_info:
            lighting_type = lighting_info["time_of_day"]
            if lighting_type in self.templates.get("lighting_templates", {}):
                lighting_description = self.templates["lighting_templates"][lighting_type]

        # Add lighting description if available
        if lighting_description and lighting_description not in description:
            description = self._smart_append(description, lighting_description)

        # Process viewpoint information
        if viewpoint != "eye_level" and viewpoint in self.templates.get("viewpoint_templates", {}):
            viewpoint_template = self.templates["viewpoint_templates"][viewpoint]

            # Special handling for viewpoint prefix
            prefix = viewpoint_template.get('prefix', '')
            if prefix and not description.startswith(prefix):
                # Prefix is a phrase like "From above, " that should precede the description
                if description and description[0].isupper():
                    # Maintain the flow by lowercasing the first letter after the prefix
                    description = prefix + description[0].lower() + description[1:]
                else:
                    description = prefix + description

            # Get appropriate scene elements description based on viewpoint
            if viewpoint == "aerial":
                scene_elements = "the crossing patterns and pedestrian movement"
            else:
                scene_elements = "objects and layout"

            viewpoint_desc = viewpoint_template.get("observation", "").format(
                scene_elements=scene_elements
            )

            # Add viewpoint observation if not already included
            if viewpoint_desc and viewpoint_desc not in description:
                description = self._smart_append(description, viewpoint_desc)

        # Add information about functional zones if available
        if functional_zones and len(functional_zones) > 0:
            zones_desc = self._describe_functional_zones(functional_zones)
            if zones_desc:
                description = self._smart_append(description, zones_desc)

        # Calculate actual people count
        people_count = len([obj for obj in detected_objects if obj["class_id"] == 0])

        # Check for inconsistencies in people count descriptions
        if people_count > 5:
            # Identify fragments that might contain smaller people counts
            small_people_patterns = [
                r"Area with \d+ people\.",
                r"Area with \d+ person\.",
                r"with \d+ people",
                r"with \d+ person"
            ]

            # Check and remove each pattern
            filtered_description = description
            for pattern in small_people_patterns:
                matches = re.findall(pattern, filtered_description)
                for match in matches:
                    # Extract the number from the match
                    number_match = re.search(r'\d+', match)
                    if number_match:
                        try:
                            people_mentioned = int(number_match.group())
                            # If the mentioned count is less than total, remove the entire sentence
                            if people_mentioned < people_count:
                                # Split description into sentences
                                sentences = re.split(r'(?<=[.!?])\s+', filtered_description)
                                # Remove sentences containing the match
                                filtered_sentences = []
                                for sentence in sentences:
                                    if match not in sentence:
                                        filtered_sentences.append(sentence)
                                # Recombine the description
                                filtered_description = " ".join(filtered_sentences)
                        except ValueError:
                            # Failed number conversion, continue processing
                            continue

            # Use the filtered description
            description = filtered_description

        # Final formatting to ensure correct punctuation and capitalization
        description = self._format_final_description(description)

        description_lines = description.split('\n')
        clean_description = []
        skip_block = False  # 添加這個變數的定義

        for line in description_lines:
            # 檢查是否需要跳過這行
            if line.strip().startswith(':param') or line.strip().startswith('"""'):
                continue
            if line.strip().startswith("Exercise") or "class SceneDescriptionSystem" in line:
                skip_block = True
                continue
            if ('def generate_scene_description' in line or
                'def enhance_scene_descriptions' in line or
                'def __init__' in line):
                skip_block = True
                continue
            if line.strip().startswith('#TEST'):
                skip_block = True
                continue

            # 空行結束跳過模式
            if skip_block and line.strip() == "":
                skip_block = False

            # 如果不需要跳過，添加這行到結果
            if not skip_block:
                clean_description.append(line)

        # 如果過濾後的描述為空，返回原始描述
        if not clean_description:
            return description
        else:
            return '\n'.join(clean_description)

    def _smart_append(self, current_text: str, new_fragment: str) -> str:
        """
        Intelligently append a new text fragment to the current text,
        handling punctuation and capitalization correctly.

        Args:
            current_text: The existing text to append to
            new_fragment: The new text fragment to append

        Returns:
            str: The combined text with proper formatting
        """
        # Handle empty cases
        if not new_fragment:
            return current_text

        if not current_text:
            # Ensure first character is uppercase for the first fragment
            return new_fragment[0].upper() + new_fragment[1:] if new_fragment else ""

        # Clean up existing text
        current_text = current_text.rstrip()

        # Check for ending punctuation
        ends_with_sentence = current_text.endswith(('.', '!', '?'))
        ends_with_comma = current_text.endswith(',')

        # Specifically handle the "A xxx A yyy" pattern that's causing issues
        if (current_text.startswith("A ") or current_text.startswith("An ")) and \
        (new_fragment.startswith("A ") or new_fragment.startswith("An ")):
            return current_text + ". " + new_fragment

        # Decide how to join the texts
        if ends_with_sentence:
            # After a sentence, start with uppercase and add proper spacing
            joined_text = current_text + " " + (new_fragment[0].upper() + new_fragment[1:])
        elif ends_with_comma:
            # After a comma, maintain flow with lowercase unless it's a proper noun or special case
            if new_fragment.startswith(('I ', 'I\'', 'A ', 'An ', 'The ')) or new_fragment[0].isupper():
                joined_text = current_text + " " + new_fragment
            else:
                joined_text = current_text + " " + new_fragment[0].lower() + new_fragment[1:]
        elif "scene is" in new_fragment.lower() or "scene includes" in new_fragment.lower():
            # When adding a new sentence about the scene, use a period
            joined_text = current_text + ". " + new_fragment
        else:
            # For other cases, decide based on the content
            if self._is_related_phrases(current_text, new_fragment):
                if new_fragment.startswith(('I ', 'I\'', 'A ', 'An ', 'The ')) or new_fragment[0].isupper():
                    joined_text = current_text + ", " + new_fragment
                else:
                    joined_text = current_text + ", " + new_fragment[0].lower() + new_fragment[1:]
            else:
                # Use period for unrelated phrases
                joined_text = current_text + ". " + (new_fragment[0].upper() + new_fragment[1:])

        return joined_text

    def _is_related_phrases(self, text1: str, text2: str) -> bool:
        """
        Determine if two phrases are related and should be connected with a comma
        rather than separated with a period.

        Args:
            text1: The first text fragment
            text2: The second text fragment to be appended

        Returns:
            bool: Whether the phrases appear to be related
        """
        # Check if either phrase starts with "A" or "An" - these are likely separate descriptions
        if (text1.startswith("A ") or text1.startswith("An ")) and \
        (text2.startswith("A ") or text2.startswith("An ")):
            return False  # These are separate descriptions, not related phrases

        # Check if the second phrase starts with a connecting word
        connecting_words = ["which", "where", "who", "whom", "whose", "with", "without",
                        "this", "these", "that", "those", "and", "or", "but"]

        first_word = text2.split()[0].lower() if text2 else ""
        if first_word in connecting_words:
            return True

        # Check if the first phrase ends with something that suggests continuity
        ending_patterns = ["such as", "including", "like", "especially", "particularly",
                        "for example", "for instance", "namely", "specifically"]

        for pattern in ending_patterns:
            if text1.lower().endswith(pattern):
                return True

        # Check if both phrases are about the scene
        if "scene" in text1.lower() and "scene" in text2.lower():
            return False  # Separate statements about the scene should be separate sentences

        return False

    def _format_final_description(self, text: str) -> str:
        """
        Format the final description text to ensure correct punctuation,
        capitalization, and spacing.

        Args:
            text: The text to format

        Returns:
            str: The properly formatted text
        """
        import re

        if not text:
            return ""

        # 1. 特別處理連續以"A"開頭的片段 (這是一個常見問題)
        text = re.sub(r'(A\s[^.!?]+?)\s+(A\s)', r'\1. \2', text, flags=re.IGNORECASE)
        text = re.sub(r'(An\s[^.!?]+?)\s+(An?\s)', r'\1. \2', text, flags=re.IGNORECASE)

        # 2. 確保第一個字母大寫
        text = text[0].upper() + text[1:] if text else ""

        # 3. 修正詞之間的空格問題
        text = re.sub(r'\s{2,}', ' ', text)  # 多個空格改為一個
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)  # 小寫後大寫間加空格

        # 4. 修正詞連接問題
        text = re.sub(r'([a-zA-Z])and', r'\1 and', text)  # "xxx"和"and"間加空格
        text = re.sub(r'([a-zA-Z])with', r'\1 with', text)  # "xxx"和"with"間加空格
        text = re.sub(r'plants(and|with|or)', r'plants \1', text)  # 修正"plantsand"這類問題

        # 5. 修正標點符號後的大小寫問題
        text = re.sub(r'\.(\s+)([a-z])', lambda m: f'.{m.group(1)}{m.group(2).upper()}', text)  # 句號後大寫

        # 6. 修正逗號後接大寫單詞的問題
        def fix_capitalization_after_comma(match):
            word = match.group(2)
            # 例外情況：保留專有名詞、人稱代詞等的大寫
            if word in ["I", "I'm", "I've", "I'd", "I'll"]:
                return match.group(0)  # 保持原樣

            # 保留月份、星期、地名等專有名詞的大寫
            proper_nouns = ["January", "February", "March", "April", "May", "June", "July",
                            "August", "September", "October", "November", "December",
                            "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            if word in proper_nouns:
                return match.group(0)  # 保持原樣

            # 其他情況：將首字母改為小寫
            return match.group(1) + word[0].lower() + word[1:]

        # 匹配逗號後接空格再接大寫單詞的模式
        text = re.sub(r'(,\s+)([A-Z][a-zA-Z]*)', fix_capitalization_after_comma, text)


        common_phrases = [
            (r'Social or seating area', r'social or seating area'),
            (r'Sleeping area', r'sleeping area'),
            (r'Dining area', r'dining area'),
            (r'Living space', r'living space')
        ]

        for phrase, replacement in common_phrases:
            # 只修改句中的術語，保留句首的大寫
            text = re.sub(r'(?<=[.!?]\s)' + phrase, replacement, text)
            # 修改句中的術語，但保留句首的大寫
            text = re.sub(r'(?<=,\s)' + phrase, replacement, text)

        # 7. 確保標點符號後有空格
        text = re.sub(r'\s+([.,;:!?])', r'\1', text)  # 標點符號前不要空格
        text = re.sub(r'([.,;:!?])([a-zA-Z0-9])', r'\1 \2', text)  # 標點符號後要有空格

        # 8. 修正重複標點符號
        text = re.sub(r'\.{2,}', '.', text)  # 多個句號變一個
        text = re.sub(r',{2,}', ',', text)  # 多個逗號變一個

        # 9. 確保文本以標點結束
        if text and not text[-1] in '.!?':
            text += '.'

        return text

    def _is_intersection(self, detected_objects: List[Dict]) -> bool:
        """
        通過分析物體分佈來判斷場景是否為十字路口
        """
        # 檢查行人分佈模式
        pedestrians = [obj for obj in detected_objects if obj["class_id"] == 0]

        if len(pedestrians) >= 8:  # 需要足夠的行人來形成十字路口
            # 抓取行人位置
            positions = [obj.get("normalized_center", (0, 0)) for obj in pedestrians]

            # 分析 x 和 y 坐標分佈
            x_coords = [pos[0] for pos in positions]
            y_coords = [pos[1] for pos in positions]

            # 計算 x 和 y 坐標的變異數
            x_variance = np.var(x_coords) if len(x_coords) > 1 else 0
            y_variance = np.var(y_coords) if len(y_coords) > 1 else 0

            # 計算範圍
            x_range = max(x_coords) - min(x_coords)
            y_range = max(y_coords) - min(y_coords)

            # 如果 x 和 y 方向都有較大範圍且範圍相似，那就有可能是十字路口
            if x_range > 0.5 and y_range > 0.5 and 0.7 < (x_range / y_range) < 1.3:
                return True

        return False

    def _generate_generic_description(self, detected_objects: List[Dict], lighting_info: Optional[Dict] = None) -> str:
        """
        Generate a generic description when scene type is unknown or confidence is very low.

        Args:
            detected_objects: List of detected objects
            lighting_info: Optional lighting condition information

        Returns:
            str: Generic description based on detected objects
        """
        # Count object occurrences
        obj_counts = {}
        for obj in detected_objects:
            class_name = obj["class_name"]
            if class_name not in obj_counts:
                obj_counts[class_name] = 0
            obj_counts[class_name] += 1

        # Get top objects by count
        top_objects = sorted(obj_counts.items(), key=lambda x: x[1], reverse=True)[:5]

        if not top_objects:
            base_desc = "No clearly identifiable objects are visible in this scene."
        else:
            # Format object list
            objects_text = []
            for name, count in top_objects:
                if count > 1:
                    objects_text.append(f"{count} {name}s")
                else:
                    objects_text.append(name)

            if len(objects_text) == 1:
                objects_list = objects_text[0]
            elif len(objects_text) == 2:
                objects_list = f"{objects_text[0]} and {objects_text[1]}"
            else:
                objects_list = ", ".join(objects_text[:-1]) + f", and {objects_text[-1]}"

            base_desc = f"This scene contains {objects_list}."

        # Add lighting information if available
        if lighting_info and "time_of_day" in lighting_info:
            lighting_type = lighting_info["time_of_day"]
            if lighting_type in self.templates.get("lighting_templates", {}):
                lighting_desc = self.templates["lighting_templates"][lighting_type]
                base_desc += f" {lighting_desc}"

        return base_desc

    def _generate_scene_details(self,
                              scene_type: str,
                              detected_objects: List[Dict],
                              lighting_info: Optional[Dict] = None,
                              viewpoint: str = "eye_level") -> str:
        """
        Generate detailed description based on scene type and detected objects.

        Args:
            scene_type: Identified scene type
            detected_objects: List of detected objects
            lighting_info: Optional lighting condition information
            viewpoint: Detected viewpoint (aerial, eye_level, etc.)

        Returns:
            str: Detailed scene description
        """
        # Get scene-specific templates
        scene_details = ""
        scene_templates = self.templates.get("scene_detail_templates", {})

        # Handle specific scene types
        if scene_type in scene_templates:
            # Select a template appropriate for the viewpoint if available
            viewpoint_key = f"{scene_type}_{viewpoint}"

            if viewpoint_key in scene_templates:
                # We have a viewpoint-specific template
                templates_list = scene_templates[viewpoint_key]
            else:
                # Fall back to general templates for this scene type
                templates_list = scene_templates[scene_type]

            # Select a random template from the list
            if templates_list:
                detail_template = random.choice(templates_list)

                # Fill the template with object information
                scene_details = self._fill_detail_template(
                    detail_template,
                    detected_objects,
                    scene_type
                )
        else:
            # Use default templates if specific ones aren't available
            if "default" in scene_templates:
                detail_template = random.choice(scene_templates["default"])
                scene_details = self._fill_detail_template(
                    detail_template,
                    detected_objects,
                    "default"
                )
            else:
                # Fall back to basic description if no templates are available
                scene_details = self._generate_basic_details(scene_type, detected_objects)

        return scene_details

    def _fill_detail_template(self, template: str, detected_objects: List[Dict], scene_type: str) -> str:
        """
        Fill a template with specific details based on detected objects.

        Args:
            template: Template string with placeholders
            detected_objects: List of detected objects
            scene_type: Identified scene type

        Returns:
            str: Filled template
        """
        # Find placeholders in the template using simple {placeholder} syntax
        import re
        placeholders = re.findall(r'\{([^}]+)\}', template)

        filled_template = template

        # Get object template fillers
        fillers = self.templates.get("object_template_fillers", {})

        # 為所有可能的變數設置默認值
        default_replacements = {
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
            "crossing_pattern": "marked pedestrian crossings",
            "pedestrian_behavior": "careful walking",
            "pedestrian_density": "groups of pedestrians",
            "traffic_pattern": "regulated traffic flow",

            # 交通樞紐相關
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

        # For each placeholder, try to fill with appropriate content
        for placeholder in placeholders:
            if placeholder in fillers:
                # Get random filler for this placeholder
                options = fillers[placeholder]
                if options:
                    # Select 1-3 items from the options list
                    num_items = min(len(options), random.randint(1, 3))
                    selected_items = random.sample(options, num_items)

                    # Create a formatted list
                    if len(selected_items) == 1:
                        replacement = selected_items[0]
                    elif len(selected_items) == 2:
                        replacement = f"{selected_items[0]} and {selected_items[1]}"
                    else:
                        replacement = ", ".join(selected_items[:-1]) + f", and {selected_items[-1]}"

                    # Replace the placeholder
                    filled_template = filled_template.replace(f"{{{placeholder}}}", replacement)
            else:
                # Try to fill with scene-specific logic
                replacement = self._generate_placeholder_content(placeholder, detected_objects, scene_type)
                if replacement:
                    filled_template = filled_template.replace(f"{{{placeholder}}}", replacement)
                elif placeholder in default_replacements:
                    # Use default replacement if available
                    filled_template = filled_template.replace(f"{{{placeholder}}}", default_replacements[placeholder])
                else:
                    # Last resort default
                    filled_template = filled_template.replace(f"{{{placeholder}}}", "various items")

        return filled_template

    def _generate_placeholder_content(self, placeholder: str, detected_objects: List[Dict], scene_type: str) -> str:
        """
        Generate content for a template placeholder based on scene-specific logic.

        Args:
            placeholder: Template placeholder
            detected_objects: List of detected objects
            scene_type: Identified scene type

        Returns:
            str: Content for the placeholder
        """
        # Handle different types of placeholders with custom logic
        if placeholder == "furniture":
            # Extract furniture items
            furniture_ids = [56, 57, 58, 59, 60, 61]  # Example furniture IDs
            furniture_objects = [obj for obj in detected_objects if obj["class_id"] in furniture_ids]

            if furniture_objects:
                furniture_names = [obj["class_name"] for obj in furniture_objects[:3]]
                return ", ".join(set(furniture_names))
            return "various furniture items"

        elif placeholder == "electronics":
            # Extract electronic items
            electronics_ids = [62, 63, 64, 65, 66, 67, 68, 69, 70]  # Example electronics IDs
            electronics_objects = [obj for obj in detected_objects if obj["class_id"] in electronics_ids]

            if electronics_objects:
                electronics_names = [obj["class_name"] for obj in electronics_objects[:3]]
                return ", ".join(set(electronics_names))
            return "electronic devices"

        elif placeholder == "people_count":
            # Count people
            people_count = len([obj for obj in detected_objects if obj["class_id"] == 0])

            if people_count == 0:
                return "no people"
            elif people_count == 1:
                return "one person"
            elif people_count < 5:
                return f"{people_count} people"
            else:
                return "several people"

        elif placeholder == "seating":
            # Extract seating items
            seating_ids = [56, 57]  # chair, sofa
            seating_objects = [obj for obj in detected_objects if obj["class_id"] in seating_ids]

            if seating_objects:
                seating_names = [obj["class_name"] for obj in seating_objects[:2]]
                return ", ".join(set(seating_names))
            return "seating arrangements"

        # Default case - empty string
        return ""

    def _generate_basic_details(self, scene_type: str, detected_objects: List[Dict]) -> str:
        """
        Generate basic details when templates aren't available.

        Args:
            scene_type: Identified scene type
            detected_objects: List of detected objects

        Returns:
            str: Basic scene details
        """
        # Handle specific scene types with custom logic
        if scene_type == "living_room":
            tv_objs = [obj for obj in detected_objects if obj["class_id"] == 62]  # TV
            sofa_objs = [obj for obj in detected_objects if obj["class_id"] == 57]  # Sofa

            if tv_objs and sofa_objs:
                tv_region = tv_objs[0]["region"]
                sofa_region = sofa_objs[0]["region"]

                arrangement = f"The TV is in the {tv_region.replace('_', ' ')} of the image, "
                arrangement += f"while the sofa is in the {sofa_region.replace('_', ' ')}. "

                return f"{arrangement}This appears to be a space designed for relaxation and entertainment."

        elif scene_type == "bedroom":
            bed_objs = [obj for obj in detected_objects if obj["class_id"] == 59]  # Bed

            if bed_objs:
                bed_region = bed_objs[0]["region"]
                extra_items = []

                for obj in detected_objects:
                    if obj["class_id"] == 74:  # Clock
                        extra_items.append("clock")
                    elif obj["class_id"] == 73:  # Book
                        extra_items.append("book")

                extras = ""
                if extra_items:
                    extras = f" There is also a {' and a '.join(extra_items)} visible."

                return f"The bed is located in the {bed_region.replace('_', ' ')} of the image.{extras}"

        elif scene_type in ["dining_area", "kitchen"]:
            # Count food and dining-related items
            food_items = []
            for obj in detected_objects:
                if obj["class_id"] in [39, 41, 42, 43, 44, 45]:  # Kitchen items
                    food_items.append(obj["class_name"])

            food_str = ""
            if food_items:
                unique_items = list(set(food_items))
                if len(unique_items) <= 3:
                    food_str = f" with {', '.join(unique_items)}"
                else:
                    food_str = f" with {', '.join(unique_items[:3])} and other items"

            return f"{food_str}."

        elif scene_type == "city_street":
            # Count people and vehicles
            people_count = len([obj for obj in detected_objects if obj["class_id"] == 0])
            vehicle_count = len([obj for obj in detected_objects
                               if obj["class_id"] in [1, 2, 3, 5, 7]])  # Bicycle, car, motorbike, bus, truck

            traffic_desc = ""
            if people_count > 0 and vehicle_count > 0:
                traffic_desc = f" with {people_count} {'people' if people_count > 1 else 'person'} and "
                traffic_desc += f"{vehicle_count} {'vehicles' if vehicle_count > 1 else 'vehicle'}"
            elif people_count > 0:
                traffic_desc = f" with {people_count} {'people' if people_count > 1 else 'person'}"
            elif vehicle_count > 0:
                traffic_desc = f" with {vehicle_count} {'vehicles' if vehicle_count > 1 else 'vehicle'}"

            return f"{traffic_desc}."

        # Handle more specialized scenes
        elif scene_type == "asian_commercial_street":
            # Look for key urban elements
            people_count = len([obj for obj in detected_objects if obj["class_id"] == 0])
            vehicle_count = len([obj for obj in detected_objects if obj["class_id"] in [1, 2, 3]])

            # Analyze pedestrian distribution
            people_positions = []
            for obj in detected_objects:
                if obj["class_id"] == 0:  # Person
                    people_positions.append(obj["normalized_center"])

            # Check if people are distributed along a line (indicating a walking path)
            structured_path = False
            if len(people_positions) >= 3:
                # Simplified check - see if y-coordinates are similar for multiple people
                y_coords = [pos[1] for pos in people_positions]
                y_mean = sum(y_coords) / len(y_coords)
                y_variance = sum((y - y_mean)**2 for y in y_coords) / len(y_coords)
                if y_variance < 0.05:  # Low variance indicates linear arrangement
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

            # Add cultural elements
            street_desc += ". The signage and architectural elements suggest an Asian urban setting."

            return street_desc

        # Default general description
        return "The scene contains various elements characteristic of this environment."

    def _detect_viewpoint(self, detected_objects: List[Dict]) -> str:
        """
        改進視角檢測，特別加強對空中俯視視角的識別。

        Args:
            detected_objects: 檢測到的物體列表

        Returns:
            str: 檢測到的視角類型
        """
        if not detected_objects:
            return "eye_level"  # default

        # 提取物體位置和大小
        top_region_count = 0
        bottom_region_count = 0
        total_objects = len(detected_objects)

        # 追蹤大小分布以檢測空中視角
        sizes = []

        # 垂直大小比例用於低角度檢測
        height_width_ratios = []

        # 用於檢測規則圖案的變數
        people_positions = []
        crosswalk_pattern_detected = False

        for obj in detected_objects:
            # 計算頂部/底部區域中的物體
            region = obj["region"]
            if "top" in region:
                top_region_count += 1
            elif "bottom" in region:
                bottom_region_count += 1

            # 計算標準化大小（面積）
            if "normalized_area" in obj:
                sizes.append(obj["normalized_area"])

            # 計算高度/寬度比例
            if "normalized_size" in obj:
                width, height = obj["normalized_size"]
                if width > 0:
                    height_width_ratios.append(height / width)

            # 收集人的位置用於圖案檢測
            if obj["class_id"] == 0:  # 人
                if "normalized_center" in obj:
                    people_positions.append(obj["normalized_center"])

        # 專門為斑馬線十字路口添加檢測邏輯
        # 檢查是否有明顯的垂直和水平行人分布
        people_objs = [obj for obj in detected_objects if obj["class_id"] == 0]  # 人

        if len(people_objs) >= 8:  # 需要足夠多的人才能形成十字路口模式
            # 檢查是否有斑馬線模式 - 新增功能
            if len(people_positions) >= 4:
                # 對位置進行聚類分析，尋找線性分布
                x_coords = [pos[0] for pos in people_positions]
                y_coords = [pos[1] for pos in people_positions]

                # 計算 x 和 y 坐標的變異數和範圍
                x_variance = np.var(x_coords) if len(x_coords) > 1 else 0
                y_variance = np.var(y_coords) if len(y_coords) > 1 else 0

                x_range = max(x_coords) - min(x_coords)
                y_range = max(y_coords) - min(y_coords)

                # 嘗試檢測十字形分布
                # 如果 x 和 y 方向都有較大範圍，且範圍相似，可能是十字路口
                if x_range > 0.5 and y_range > 0.5 and 0.7 < (x_range / y_range) < 1.3:

                    # 計算到中心點的距離
                    center_x = np.mean(x_coords)
                    center_y = np.mean(y_coords)

                    # 將點映射到十字架的軸上（水平和垂直）
                    x_axis_distance = [abs(x - center_x) for x in x_coords]
                    y_axis_distance = [abs(y - center_y) for y in y_coords]

                    # 點應該接近軸線（水平或垂直）
                    # 對於每個點，檢查它是否接近水平或垂直軸線
                    close_to_axis_count = 0
                    for i in range(len(x_coords)):
                        if x_axis_distance[i] < 0.1 or y_axis_distance[i] < 0.1:
                            close_to_axis_count += 1

                    # 如果足夠多的點接近軸線，認為是十字路口
                    if close_to_axis_count >= len(x_coords) * 0.6:
                        crosswalk_pattern_detected = True

                # 如果沒有檢測到十字形，嘗試檢測線性聚類分布
                if not crosswalk_pattern_detected:
                    # 檢查 x 和 y 方向的聚類
                    x_clusters = self._detect_linear_clusters(x_coords)
                    y_clusters = self._detect_linear_clusters(y_coords)

                    # 如果在 x 和 y 方向上都有多個聚類，可能是交叉的斑馬線
                    if len(x_clusters) >= 2 and len(y_clusters) >= 2:
                        crosswalk_pattern_detected = True

        # 檢測斑馬線模式 - 優先判斷
        if crosswalk_pattern_detected:
            return "aerial"

        # 檢測行人分布情況
        if len(people_objs) >= 10:
            people_region_counts = {}
            for obj in people_objs:
                region = obj["region"]
                if region not in people_region_counts:
                    people_region_counts[region] = 0
                people_region_counts[region] += 1

            # 計算不同區域中的行人數量
            region_count = len([r for r, c in people_region_counts.items() if c >= 2])

            # 如果行人分布在多個區域中，可能是空中視角
            if region_count >= 4:
                # 檢查行人分布的模式
                # 特別是檢查不同區域中行人數量的差異
                region_counts = list(people_region_counts.values())
                region_counts_variance = np.var(region_counts) if len(region_counts) > 1 else 0
                region_counts_mean = np.mean(region_counts) if region_counts else 0

                # 如果行人分布較為均勻（變異係數小），可能是空中視角
                if region_counts_mean > 0:
                    variation_coefficient = region_counts_variance / region_counts_mean
                    if variation_coefficient < 0.5:
                        return "aerial"

        # 計算指標
        top_ratio = top_region_count / total_objects if total_objects > 0 else 0
        bottom_ratio = bottom_region_count / total_objects if total_objects > 0 else 0

        # 大小變異數（標準化）
        size_variance = 0
        if sizes:
            mean_size = sum(sizes) / len(sizes)
            size_variance = sum((s - mean_size) ** 2 for s in sizes) / len(sizes)
            size_variance = size_variance / (mean_size ** 2)  # 標準化

        # 平均高度/寬度比例
        avg_height_width_ratio = sum(height_width_ratios) / len(height_width_ratios) if height_width_ratios else 1.0

        # 空中視角：低大小差異，物體均勻分布，底部很少或沒有物體
        if (size_variance < self.viewpoint_params["aerial_size_variance_threshold"] and
            bottom_ratio < 0.3 and top_ratio > self.viewpoint_params["aerial_threshold"]):
            return "aerial"

        # 低角度視角：物體傾向於比寬高，頂部較多物體
        elif (avg_height_width_ratio > self.viewpoint_params["vertical_size_ratio_threshold"] and
            top_ratio > self.viewpoint_params["low_angle_threshold"]):
            return "low_angle"

        # 高視角：底部較多物體，頂部較少
        elif (bottom_ratio > self.viewpoint_params["elevated_threshold"] and
            top_ratio < self.viewpoint_params["elevated_top_threshold"]):
            return "elevated"

        # 默認：平視角
        return "eye_level"

    def _detect_linear_clusters(self, coords, threshold=0.05):
        """
        檢測坐標中的線性聚類

        Args:
            coords: 一維坐標列表
            threshold: 聚類閾值

        Returns:
            list: 聚類列表
        """
        if not coords:
            return []

        # 排序坐標
        sorted_coords = sorted(coords)

        clusters = []
        current_cluster = [sorted_coords[0]]

        for i in range(1, len(sorted_coords)):
            # 如果當前坐標與前一個接近，添加到當前聚類
            if sorted_coords[i] - sorted_coords[i-1] < threshold:
                current_cluster.append(sorted_coords[i])
            else:
                # 否則開始新的聚類
                if len(current_cluster) >= 2:  # 至少需要2個點形成聚類
                    clusters.append(current_cluster)
                current_cluster = [sorted_coords[i]]

        # 添加最後一個cluster
        if len(current_cluster) >= 2:
            clusters.append(current_cluster)

        return clusters

    def _detect_cultural_context(self, scene_type: str, detected_objects: List[Dict]) -> Optional[str]:
        """
        Detect the likely cultural context of the scene.

        Args:
            scene_type: Identified scene type
            detected_objects: List of detected objects

        Returns:
            Optional[str]: Detected cultural context (asian, european, etc.) or None
        """
        # Scene types with explicit cultural contexts
        cultural_scene_mapping = {
            "asian_commercial_street": "asian",
            "asian_night_market": "asian",
            "asian_temple_area": "asian",
            "european_plaza": "european"
        }

        # Check if scene type directly indicates cultural context
        if scene_type in cultural_scene_mapping:
            return cultural_scene_mapping[scene_type]

        # No specific cultural context detected
        return None

    def _generate_cultural_elements(self, cultural_context: str) -> str:
        """
        Generate description of cultural elements for the detected context.

        Args:
            cultural_context: Detected cultural context

        Returns:
            str: Description of cultural elements
        """
        # Get template for this cultural context
        cultural_templates = self.templates.get("cultural_templates", {})

        if cultural_context in cultural_templates:
            template = cultural_templates[cultural_context]
            elements = template.get("elements", [])

            if elements:
                # Select 1-2 random elements
                num_elements = min(len(elements), random.randint(1, 2))
                selected_elements = random.sample(elements, num_elements)

                # Format elements list
                elements_text = " and ".join(selected_elements) if num_elements == 2 else selected_elements[0]

                # Fill template
                return template.get("description", "").format(elements=elements_text)

        return ""

    def _optimize_object_description(self, description: str) -> str:
        """
        優化物品描述，避免重複列舉相同物品
        """
        import re

        # 處理床鋪重複描述
        if "bed in the room" in description:
            description = description.replace("a bed in the room", "a bed")

        # 處理重複的物品列表
        # 尋找格式如 "item, item, item" 的模式
        object_lists = re.findall(r'with ([^\.]+?)(?:\.|\band\b)', description)

        for obj_list in object_lists:
            # 計算每個物品出現次數
            items = re.findall(r'([a-zA-Z\s]+)(?:,|\band\b|$)', obj_list)
            item_counts = {}

            for item in items:
                item = item.strip()
                if item and item not in ["and", "with"]:
                    if item not in item_counts:
                        item_counts[item] = 0
                    item_counts[item] += 1

            # 生成優化後的物品列表
            if item_counts:
                new_items = []
                for item, count in item_counts.items():
                    if count > 1:
                        new_items.append(f"{count} {item}s")
                    else:
                        new_items.append(item)

                # 格式化新列表
                if len(new_items) == 1:
                    new_list = new_items[0]
                elif len(new_items) == 2:
                    new_list = f"{new_items[0]} and {new_items[1]}"
                else:
                    new_list = ", ".join(new_items[:-1]) + f", and {new_items[-1]}"

                # 替換原始列表
                description = description.replace(obj_list, new_list)

        return description

    def _describe_functional_zones(self, functional_zones: Dict) -> str:
        """
        生成場景功能區域的描述，優化處理行人區域、人數統計和物品重複問題。

        Args:
            functional_zones: 識別出的功能區域字典

        Returns:
            str: 功能區域描述
        """
        if not functional_zones:
            return ""

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
        max_mentioned_people = 0  # 跟踪已經提到的最大人數

        # 如果總人數顯著且還沒在主描述中提到，添加總人數描述
        if total_people_count > 5:
            summary = f"The scene contains a significant number of pedestrians ({total_people_count} people). "
            max_mentioned_people = total_people_count  # 更新已提到的最大人數

        # 處理每個區域的描述，確保人數信息的一致性
        processed_zones = []

        for zone_name, zone_info in top_zones:
            zone_desc = zone_info.get("description", "a functional zone")
            zone_people_count = people_by_zone.get(zone_name, 0)

            # 檢查描述中是否包含人數信息
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

        return self._optimize_object_description(final_desc)
