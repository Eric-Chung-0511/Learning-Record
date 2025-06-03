import logging
import traceback
import random
from typing import Dict, List, Optional, Any

from cultural_templates import CULTURAL_TEMPLATES

class CulturalContextError(Exception):
    """文化語境分析過程中的自定義異常"""
    pass


class CulturalContextAnalyzer:
    """
    文化語境分析器 - 檢測場景中的文化特徵並生成相關的描述

    該類別負責識別場景中的文化語境線索，包括建築風格、標誌特徵
    和物件配置，然後生成適當的文化描述元素。
    """

    def __init__(self, cultural_templates: Optional[Dict] = None):
        """
        初始化文化語境分析器

        Args:
            cultural_templates: 可選的自定義文化模板，如果提供則會與默認模板合併
        """
        self.logger = logging.getLogger(self.__class__.__name__)

        try:
            # 載入文化模板
            self.cultural_templates = self._load_cultural_templates()

            # 如果提供了自定義模板，進行合併
            if cultural_templates:
                self._merge_custom_templates(cultural_templates)

            # 初始化場景類型到文化語境的映射
            self.scene_cultural_mapping = self._initialize_scene_cultural_mapping()

            self.logger.info("CulturalContextAnalyzer initialized with %d cultural templates",
                           len(self.cultural_templates))

        except Exception as e:
            error_msg = f"Failed to initialize CulturalContextAnalyzer: {str(e)}"
            self.logger.error(f"{error_msg}\n{traceback.format_exc()}")
            raise CulturalContextError(error_msg) from e

    def _load_cultural_templates(self) -> Dict:
        """
        載入文化模板

        Returns:
            Dict: 文化模板字典

        Raises:
            CulturalContextError: 當模板載入失敗時
        """
        try:
            self.logger.debug("Loading cultural templates")

            # 從配置模組載入文化模板
            templates = CULTURAL_TEMPLATES.copy()

            # 確保模板結構正確
            self._validate_cultural_templates(templates)

            # 如果沒有載入到模板，使用默認模板
            if not templates:
                self.logger.warning("No cultural templates loaded, using defaults")
                templates = self._get_default_cultural_templates()

            self.logger.debug("Successfully loaded %d cultural template categories", len(templates))
            return templates

        except ImportError as e:
            self.logger.warning(f"Failed to import cultural templates: {str(e)}, using defaults")
            return self._get_default_cultural_templates()
        except Exception as e:
            error_msg = f"Error loading cultural templates: {str(e)}"
            self.logger.error(f"{error_msg}\n{traceback.format_exc()}")
            raise CulturalContextError(error_msg) from e

    def _get_default_cultural_templates(self) -> Dict:
        """
        獲取默認文化模板

        Returns:
            Dict: 默認文化模板字典
        """
        return {
            "asian": {
                "elements": [
                    "traditional architectural elements",
                    "cultural signage",
                    "Asian design features",
                    "oriental decorative patterns",
                    "traditional building materials",
                    "characteristic roofline styles",
                    "cultural landscaping elements"
                ],
                "description": "The scene displays distinctive Asian cultural characteristics with {elements}."
            },
            "european": {
                "elements": [
                    "classical architecture",
                    "European design elements",
                    "historic features",
                    "traditional stonework",
                    "characteristic window styles",
                    "ornamental facades",
                    "heritage building elements"
                ],
                "description": "The scene exhibits European architectural and cultural elements including {elements}."
            },
            "american": {
                "elements": [
                    "modern architectural styles",
                    "contemporary design features",
                    "commercial signage",
                    "urban planning elements",
                    "standardized building designs"
                ],
                "description": "The scene shows American urban characteristics featuring {elements}."
            },
            "mediterranean": {
                "elements": [
                    "coastal architectural styles",
                    "warm climate adaptations",
                    "traditional building colors",
                    "characteristic outdoor spaces"
                ],
                "description": "The scene reflects Mediterranean cultural influences with {elements}."
            }
        }

    def _validate_cultural_templates(self, templates: Dict):
        """
        驗證文化模板結構

        Args:
            templates: 要驗證的模板字典

        Raises:
            CulturalContextError: 當模板結構無效時
        """
        try:
            for culture, template_data in templates.items():
                if not isinstance(template_data, dict):
                    self.logger.warning(f"Invalid cultural template structure for '{culture}': not a dictionary")
                    continue

                required_keys = ["elements", "description"]
                for key in required_keys:
                    if key not in template_data:
                        self.logger.warning(f"Missing required key '{key}' in cultural template '{culture}'")

                # 驗證元素列表
                if "elements" in template_data:
                    if not isinstance(template_data["elements"], list):
                        self.logger.warning(f"Cultural template '{culture}' elements should be a list")
                    elif not template_data["elements"]:
                        self.logger.warning(f"Cultural template '{culture}' has empty elements list")

                # 驗證描述模板
                if "description" in template_data:
                    if not isinstance(template_data["description"], str):
                        self.logger.warning(f"Cultural template '{culture}' description should be a string")
                    elif "{elements}" not in template_data["description"]:
                        self.logger.warning(f"Cultural template '{culture}' description missing {{elements}} placeholder")

            self.logger.debug("Cultural templates validation completed")

        except Exception as e:
            self.logger.warning(f"Error validating cultural templates: {str(e)}")

    def _merge_custom_templates(self, custom_templates: Dict):
        """
        合併自定義文化模板

        Args:
            custom_templates: 自定義模板字典
        """
        try:
            for culture, template_data in custom_templates.items():
                if culture in self.cultural_templates:
                    # 合併現有文化的模板
                    if isinstance(self.cultural_templates[culture], dict) and isinstance(template_data, dict):
                        # 合併元素列表
                        if "elements" in template_data and "elements" in self.cultural_templates[culture]:
                            existing_elements = self.cultural_templates[culture]["elements"]
                            new_elements = template_data["elements"]
                            if isinstance(existing_elements, list) and isinstance(new_elements, list):
                                self.cultural_templates[culture]["elements"] = existing_elements + new_elements

                        # 更新其他鍵值
                        for key, value in template_data.items():
                            if key != "elements":
                                self.cultural_templates[culture][key] = value
                    else:
                        self.cultural_templates[culture] = template_data
                else:
                    # 添加新的文化模板
                    self.cultural_templates[culture] = template_data

                self.logger.debug(f"Merged custom template for culture: {culture}")

            self.logger.info("Successfully merged custom cultural templates")

        except Exception as e:
            self.logger.warning(f"Error merging custom cultural templates: {str(e)}")

    def _initialize_scene_cultural_mapping(self) -> Dict[str, str]:
        """
        初始化場景類型到文化語境的display

        Returns:
            Dict[str, str]: 場景類型到文化語境的映射字典
        """
        return {
            "asian_commercial_street": "asian",
            "asian_night_market": "asian",
            "asian_temple_area": "asian",
            "chinese_restaurant": "asian",
            "japanese_restaurant": "asian",
            "korean_restaurant": "asian",
            "european_plaza": "european",
            "european_cafe": "european",
            "mediterranean_restaurant": "mediterranean",
            "american_diner": "american",
            "american_fast_food": "american"
        }

    def detect_cultural_context(self, scene_type: str, detected_objects: List[Dict]) -> Optional[str]:
        """
        檢測場景的文化語境

        Args:
            scene_type: 識別的場景類型
            detected_objects: 檢測到的物件列表

        Returns:
            Optional[str]: 檢測到的文化語境（asian, european等）或None
        """
        try:
            self.logger.debug(f"Detecting cultural context for scene_type: {scene_type}")

            # 檢查場景類型是否直接指示文化語境
            if scene_type in self.scene_cultural_mapping:
                cultural_context = self.scene_cultural_mapping[scene_type]
                self.logger.debug(f"Direct cultural mapping found: {scene_type} -> {cultural_context}")
                return cultural_context

            # 基於場景類型名稱的模式匹配
            cultural_context = self._detect_from_scene_name_patterns(scene_type)
            if cultural_context:
                self.logger.debug(f"Cultural context detected from name patterns: {cultural_context}")
                return cultural_context

            # 基於檢測物件的文化特徵分析
            cultural_context = self._detect_from_object_analysis(detected_objects)
            if cultural_context:
                self.logger.debug(f"Cultural context detected from object analysis: {cultural_context}")
                return cultural_context

            # 沒有檢測到特定文化語境
            self.logger.debug("No specific cultural context detected")
            return None

        except Exception as e:
            self.logger.warning(f"Error detecting cultural context: {str(e)}")
            return None

    def _detect_from_scene_name_patterns(self, scene_type: str) -> Optional[str]:
        """
        基於場景類型名稱模式檢測文化語境

        Args:
            scene_type: 場景類型名稱

        Returns:
            Optional[str]: 檢測到的文化語境或None
        """
        try:
            scene_lower = scene_type.lower()

            # Asia
            asian_keywords = [
                "asian", "chinese", "japanese", "korean", "thai", "vietnamese",
                "temple", "pagoda", "zen", "oriental", "bamboo", "tatami"
            ]

            # Europe
            european_keywords = [
                "european", "french", "italian", "spanish", "german", "british",
                "plaza", "piazza", "cathedral", "gothic", "baroque", "renaissance",
                "cafe", "bistro", "pub"
            ]

            # 地中海文化
            mediterranean_keywords = [
                "mediterranean", "greek", "turkish", "coastal", "terrace",
                "villa", "courtyard"
            ]

            # 美國
            american_keywords = [
                "american", "diner", "fast_food", "mall", "suburban",
                "downtown", "strip_mall"
            ]

            # 檢查各文化的key word
            if any(keyword in scene_lower for keyword in asian_keywords):
                return "asian"
            elif any(keyword in scene_lower for keyword in european_keywords):
                return "european"
            elif any(keyword in scene_lower for keyword in mediterranean_keywords):
                return "mediterranean"
            elif any(keyword in scene_lower for keyword in american_keywords):
                return "american"

            return None

        except Exception as e:
            self.logger.warning(f"Error detecting cultural context from scene name patterns: {str(e)}")
            return None

    def _detect_from_object_analysis(self, detected_objects: List[Dict]) -> Optional[str]:
        """
        基於檢測物件分析文化特徵

        Args:
            detected_objects: 檢測到的物件列表

        Returns:
            Optional[str]: 檢測到的文化語境或None
        """
        try:
            if not detected_objects:
                return None

            # 統計文化相關物件
            cultural_indicators = {
                "asian": 0,
                "european": 0,
                "american": 0,
                "mediterranean": 0
            }

            for obj in detected_objects:
                class_name = obj.get("class_name", "").lower()

                # Asia 特色
                if any(indicator in class_name for indicator in [
                    "lantern", "chopsticks", "rice", "noodles", "tea",
                    "bamboo", "pagoda", "shrine", "torii"
                ]):
                    cultural_indicators["asian"] += 1

                # 歐洲的特色
                elif any(indicator in class_name for indicator in [
                    "wine", "cheese", "bread", "fountain", "column",
                    "statue", "cathedral", "clock_tower"
                ]):
                    cultural_indicators["european"] += 1

                # 地中海的特色
                elif any(indicator in class_name for indicator in [
                    "olive", "terracotta", "pergola", "villa",
                    "coastal", "maritime"
                ]):
                    cultural_indicators["mediterranean"] += 1

                # 美國的特色
                elif any(indicator in class_name for indicator in [
                    "burger", "pizza", "hotdog", "soda",
                    "drive_through", "parking_lot"
                ]):
                    cultural_indicators["american"] += 1

            # 找出得分最高的文化語境
            if max(cultural_indicators.values()) > 0:
                dominant_culture = max(cultural_indicators.items(), key=lambda x: x[1])[0]
                max_score = cultural_indicators[dominant_culture]

                # 需要至少2個指標物件才算有效檢測
                if max_score >= 2:
                    return dominant_culture

            return None

        except Exception as e:
            self.logger.warning(f"Error detecting cultural context from object analysis: {str(e)}")
            return None

    def generate_cultural_elements(self, cultural_context: str) -> str:
        """
        為檢測到的文化語境生成描述元素

        Args:
            cultural_context: 檢測到的文化語境

        Returns:
            str: 文化元素描述

        Raises:
            CulturalContextError: 當文化元素生成失敗時
        """
        try:
            if not cultural_context:
                return ""

            self.logger.debug(f"Generating cultural elements for context: {cultural_context}")

            # 獲取該文化語境的模板
            if cultural_context not in self.cultural_templates:
                self.logger.warning(f"No template found for cultural context: {cultural_context}")
                return ""

            template = self.cultural_templates[cultural_context]
            elements = template.get("elements", [])

            if not elements:
                self.logger.warning(f"No elements found for cultural context: {cultural_context}")
                return ""

            # 選擇1-2個隨機元素
            num_elements = min(len(elements), random.randint(1, 2))
            selected_elements = random.sample(elements, num_elements)

            # 格式化元素列表
            if len(selected_elements) == 1:
                elements_text = selected_elements[0]
            else:
                elements_text = " and ".join(selected_elements)

            # 填充模板
            description_template = template.get("description", "")
            if not description_template:
                return f"The scene displays {cultural_context} cultural characteristics."

            # 替換佔位符
            cultural_description = description_template.format(elements=elements_text)

            self.logger.debug(f"Generated cultural description: {cultural_description}")
            return cultural_description

        except Exception as e:
            error_msg = f"Error generating cultural elements for context '{cultural_context}': {str(e)}"
            self.logger.error(f"{error_msg}\n{traceback.format_exc()}")
            raise CulturalContextError(error_msg) from e

    def get_cultural_template(self, cultural_context: str) -> Dict[str, Any]:
        """
        獲取指定文化語境的模板

        Args:
            cultural_context: 文化語境名稱

        Returns:
            Dict[str, Any]: 文化模板字典
        """
        try:
            if cultural_context in self.cultural_templates:
                return self.cultural_templates[cultural_context].copy()

            # 返回備用模板
            self.logger.warning(f"Cultural template not found for '{cultural_context}', using fallback")
            return {
                "elements": ["various cultural elements"],
                "description": f"The scene displays {cultural_context} cultural characteristics."
            }

        except Exception as e:
            self.logger.warning(f"Error getting cultural template for '{cultural_context}': {str(e)}")
            return {
                "elements": ["various elements"],
                "description": "The scene displays cultural characteristics."
            }

    def add_cultural_template(self, cultural_context: str, template: Dict[str, Any]):
        """
        添加或更新文化模板

        Args:
            cultural_context: 文化語境名稱
            template: 文化模板字典

        Raises:
            CulturalContextError: 當模板格式無效時
        """
        try:
            # 驗證模板格式
            if not isinstance(template, dict):
                raise CulturalContextError("Template must be a dictionary")

            required_keys = ["elements", "description"]
            for key in required_keys:
                if key not in template:
                    raise CulturalContextError(f"Template missing required key: {key}")

            if not isinstance(template["elements"], list):
                raise CulturalContextError("Template 'elements' must be a list")

            if not isinstance(template["description"], str):
                raise CulturalContextError("Template 'description' must be a string")

            # 添加模板
            self.cultural_templates[cultural_context] = template.copy()

            self.logger.info(f"Added cultural template for context: {cultural_context}")

        except CulturalContextError:
            raise
        except Exception as e:
            error_msg = f"Error adding cultural template for '{cultural_context}': {str(e)}"
            self.logger.error(f"{error_msg}\n{traceback.format_exc()}")
            raise CulturalContextError(error_msg) from e

    def get_supported_cultures(self) -> List[str]:
        """
        獲取所有支援的文化語境列表

        Returns:
            List[str]: 支援的文化語境名稱列表
        """
        return list(self.cultural_templates.keys())

    def has_cultural_context(self, cultural_context: str) -> bool:
        """
        檢查是否支援指定的文化語境

        Args:
            cultural_context: 文化語境名稱

        Returns:
            bool: 是否支援該文化語境
        """
        return cultural_context in self.cultural_templates

    def analyze_cultural_diversity(self, detected_objects: List[Dict]) -> Dict[str, int]:
        """
        分析場景中的文化多樣性

        Args:
            detected_objects: 檢測到的物件列表

        Returns:
            Dict[str, int]: 各文化語境的指標物件計數
        """
        try:
            cultural_scores = {culture: 0 for culture in self.cultural_templates.keys()}

            if not detected_objects:
                return cultural_scores

            for obj in detected_objects:
                class_name = obj.get("class_name", "").lower()

                # 為每個文化語境計算指標分數
                for culture in cultural_scores:
                    if self._is_cultural_indicator(class_name, culture):
                        cultural_scores[culture] += 1

            self.logger.debug(f"Cultural diversity analysis: {cultural_scores}")
            return cultural_scores

        except Exception as e:
            self.logger.warning(f"Error analyzing cultural diversity: {str(e)}")
            return {culture: 0 for culture in self.cultural_templates.keys()}

    def _is_cultural_indicator(self, object_name: str, culture: str) -> bool:
        """
        檢查物件名稱是否為特定文化的指標

        Args:
            object_name: 物件名稱
            culture: 文化語境

        Returns:
            bool: 是否為該文化的指標物件
        """
        try:
            cultural_keywords = {
                "asian": [
                    "lantern", "chopsticks", "rice", "noodles", "tea",
                    "bamboo", "pagoda", "shrine", "torii", "kimono",
                    "sushi", "ramen", "dim_sum"
                ],
                "european": [
                    "wine", "cheese", "bread", "fountain", "column",
                    "statue", "cathedral", "clock_tower", "baguette",
                    "croissant", "espresso", "gelato"
                ],
                "mediterranean": [
                    "olive", "terracotta", "pergola", "villa",
                    "coastal", "maritime", "cypress", "vineyard"
                ],
                "american": [
                    "burger", "pizza", "hotdog", "soda",
                    "drive_through", "parking_lot", "diner",
                    "strip_mall", "suburb"
                ]
            }

            if culture not in cultural_keywords:
                return False

            keywords = cultural_keywords[culture]
            return any(keyword in object_name for keyword in keywords)

        except Exception as e:
            self.logger.warning(f"Error checking cultural indicator for {object_name}, {culture}: {str(e)}")
            return False

    def get_template_summary(self) -> Dict[str, Dict[str, Any]]:
        """
        獲取所有文化模板的摘要信息

        Returns:
            Dict[str, Dict[str, Any]]: 文化模板摘要
        """
        try:
            summary = {}

            for culture, template in self.cultural_templates.items():
                summary[culture] = {
                    "element_count": len(template.get("elements", [])),
                    "has_description": bool(template.get("description", "")),
                    "sample_elements": template.get("elements", [])[:3]  # 前3個元素作為樣本
                }

            return summary

        except Exception as e:
            self.logger.warning(f"Error generating template summary: {str(e)}")
            return {}
