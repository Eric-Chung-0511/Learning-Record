import logging
import traceback
from typing import Dict, List, Optional, Any

from scene_detail_templates import SCENE_DETAIL_TEMPLATES
from object_template_fillers import OBJECT_TEMPLATE_FILLERS
from viewpoint_templates import VIEWPOINT_TEMPLATES
from cultural_templates import CULTURAL_TEMPLATES
from lighting_conditions import LIGHTING_CONDITIONS
from confidence_templates import CONFIDENCE_TEMPLATES

class TemplateRepository:
    """
    模板資料的管理器 - 負責模板的載入、儲存、檢索和驗證

    此類別專門處理模板資源的管理，包括從各種來源載入模板、
    驗證模板完整性，以及提供統一的模板檢索介面。
    """

    def __init__(self, custom_templates_db: Optional[Dict] = None):
        """
        初始化模板庫管理器

        Args:
            custom_templates_db: 可選的自定義模板數據庫，如果提供則會與默認模板合併
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.templates = {}
        self.template_registry = {}

        try:
            # 載入模板數據庫
            self.templates = self._load_templates()

            # 初始化模板註冊表
            self.template_registry = self._initialize_template_registry()

            # 如果提供了自定義模板，則進行合併
            if custom_templates_db:
                self._merge_custom_templates(custom_templates_db)

            # 驗證模板完整性
            self._validate_templates()

            self.logger.info("TemplateRepository initialized successfully with %d template categories",
                           len(self.templates))

        except Exception as e:
            error_msg = f"Failed to initialize TemplateRepository: {str(e)}"
            self.logger.error(f"{error_msg}\n{traceback.format_exc()}")
            # 初始化基本的空模板
            self.templates = self._initialize_fallback_templates()

    def _load_templates(self) -> Dict:
        """
        載入所有描述模板

        Returns:
            Dict: 包含所有模板類別的字典
        """
        try:
            templates = {}

            # 載入場景詳細描述模板
            self.logger.debug("Loading scene detail templates")
            try:
                templates["scene_detail_templates"] = SCENE_DETAIL_TEMPLATES
            except NameError:
                self.logger.warning("SCENE_DETAIL_TEMPLATES not defined, using empty dict")
                templates["scene_detail_templates"] = {}

            # 載入物體模板填充器
            self.logger.debug("Loading object template fillers")
            try:
                templates["object_template_fillers"] = OBJECT_TEMPLATE_FILLERS
            except NameError:
                self.logger.warning("OBJECT_TEMPLATE_FILLERS not defined, using empty dict")
                templates["object_template_fillers"] = {}

            # 載入視角模板
            self.logger.debug("Loading viewpoint templates")
            try:
                templates["viewpoint_templates"] = VIEWPOINT_TEMPLATES
            except NameError:
                self.logger.warning("VIEWPOINT_TEMPLATES not defined, using empty dict")
                templates["viewpoint_templates"] = {}

            # 載入文化模板
            self.logger.debug("Loading cultural templates")
            try:
                templates["cultural_templates"] = CULTURAL_TEMPLATES
            except NameError:
                self.logger.warning("CULTURAL_TEMPLATES not defined, using empty dict")
                templates["cultural_templates"] = {}

            # 從照明條件模組載入照明模板
            self.logger.debug("Loading lighting templates")
            try:
                templates["lighting_templates"] = self._extract_lighting_templates()
            except Exception as e:
                self.logger.warning(f"Failed to extract lighting templates: {str(e)}")
                templates["lighting_templates"] = {}

            # 載入信心度模板
            self.logger.debug("Loading confidence templates")
            try:
                templates["confidence_templates"] = CONFIDENCE_TEMPLATES
            except NameError:
                self.logger.warning("CONFIDENCE_TEMPLATES not defined, using empty dict")
                templates["confidence_templates"] = {}

            # 初始化默認模板（當成備份）
            self._initialize_default_templates(templates)

            self.logger.info("Successfully loaded %d template categories", len(templates))
            return templates

        except Exception as e:
            error_msg = f"Unexpected error during template loading: {str(e)}"
            self.logger.error(f"{error_msg}\n{traceback.format_exc()}")
            # 返回基本模板
            return self._initialize_fallback_templates()

    def _initialize_template_registry(self) -> Dict[str, Dict[str, Any]]:
        """
        初始化模板註冊表，包含各種場景類型的結構化模板

        Returns:
            Dict[str, Dict[str, Any]]: 模板註冊表字典
        """
        try:
            template_registry = {
                "indoor_detailed": {
                    "scene_type": "indoor",
                    "complexity": "high",
                    "structure": [
                        {
                            "type": "opening",
                            "content": "This indoor scene presents a comprehensive view of a well-organized living space."
                        },
                        {
                            "type": "zone_analysis",
                            "priority": "functional_areas",
                            "detail_level": "detailed"
                        },
                        {
                            "type": "object_summary",
                            "grouping": "by_category",
                            "include_counts": True
                        },
                        {
                            "type": "conclusion",
                            "style": "analytical"
                        }
                    ]
                },

                "indoor_moderate": {
                    "scene_type": "indoor",
                    "complexity": "medium",
                    "structure": [
                        {
                            "type": "opening",
                            "content": "The indoor environment displays organized functional areas."
                        },
                        {
                            "type": "zone_analysis",
                            "priority": "main_areas",
                            "detail_level": "moderate"
                        },
                        {
                            "type": "object_summary",
                            "grouping": "by_function",
                            "include_counts": False
                        },
                        {
                            "type": "conclusion",
                            "style": "descriptive"
                        }
                    ]
                },

                "indoor_simple": {
                    "scene_type": "indoor",
                    "complexity": "low",
                    "structure": [
                        {
                            "type": "opening",
                            "content": "An indoor space with visible furniture and household items."
                        },
                        {
                            "type": "zone_analysis",
                            "priority": "basic_areas",
                            "detail_level": "simple"
                        },
                        {
                            "type": "object_summary",
                            "grouping": "general",
                            "include_counts": False
                        }
                    ]
                },

                "outdoor_detailed": {
                    "scene_type": "outdoor",
                    "complexity": "high",
                    "structure": [
                        {
                            "type": "opening",
                            "content": "This outdoor scene captures a dynamic urban environment with multiple activity zones."
                        },
                        {
                            "type": "zone_analysis",
                            "priority": "activity_areas",
                            "detail_level": "detailed"
                        },
                        {
                            "type": "object_summary",
                            "grouping": "by_location",
                            "include_counts": True
                        },
                        {
                            "type": "conclusion",
                            "style": "environmental"
                        }
                    ]
                },

                "outdoor_moderate": {
                    "scene_type": "outdoor",
                    "complexity": "medium",
                    "structure": [
                        {
                            "type": "opening",
                            "content": "The outdoor scene shows organized public spaces and pedestrian areas."
                        },
                        {
                            "type": "zone_analysis",
                            "priority": "public_areas",
                            "detail_level": "moderate"
                        },
                        {
                            "type": "object_summary",
                            "grouping": "by_type",
                            "include_counts": False
                        },
                        {
                            "type": "conclusion",
                            "style": "observational"
                        }
                    ]
                },

                "outdoor_simple": {
                    "scene_type": "outdoor",
                    "complexity": "low",
                    "structure": [
                        {
                            "type": "opening",
                            "content": "An outdoor area with pedestrians and urban elements."
                        },
                        {
                            "type": "zone_analysis",
                            "priority": "basic_areas",
                            "detail_level": "simple"
                        },
                        {
                            "type": "object_summary",
                            "grouping": "general",
                            "include_counts": False
                        }
                    ]
                },

                "commercial_detailed": {
                    "scene_type": "commercial",
                    "complexity": "high",
                    "structure": [
                        {
                            "type": "opening",
                            "content": "This commercial environment demonstrates organized retail and customer service areas."
                        },
                        {
                            "type": "zone_analysis",
                            "priority": "service_areas",
                            "detail_level": "detailed"
                        },
                        {
                            "type": "object_summary",
                            "grouping": "by_function",
                            "include_counts": True
                        },
                        {
                            "type": "conclusion",
                            "style": "business"
                        }
                    ]
                },

                "transportation_detailed": {
                    "scene_type": "transportation",
                    "complexity": "high",
                    "structure": [
                        {
                            "type": "opening",
                            "content": "This transportation hub features organized passenger facilities and transit infrastructure."
                        },
                        {
                            "type": "zone_analysis",
                            "priority": "transit_areas",
                            "detail_level": "detailed"
                        },
                        {
                            "type": "object_summary",
                            "grouping": "by_transit_function",
                            "include_counts": True
                        },
                        {
                            "type": "conclusion",
                            "style": "infrastructure"
                        }
                    ]
                },

                "default": {
                    "scene_type": "general",
                    "complexity": "medium",
                    "structure": [
                        {
                            "type": "opening",
                            "content": "The scene displays various elements organized across functional areas."
                        },
                        {
                            "type": "zone_analysis",
                            "priority": "general_areas",
                            "detail_level": "moderate"
                        },
                        {
                            "type": "object_summary",
                            "grouping": "general",
                            "include_counts": False
                        },
                        {
                            "type": "conclusion",
                            "style": "general"
                        }
                    ]
                }
            }

            self.logger.debug(f"Initialized template registry with {len(template_registry)} templates")
            return template_registry

        except Exception as e:
            error_msg = f"Error initializing template registry: {str(e)}"
            self.logger.error(f"{error_msg}\n{traceback.format_exc()}")
            # 返回最基本的註冊表
            return {
                "default": {
                    "scene_type": "general",
                    "complexity": "low",
                    "structure": [
                        {
                            "type": "opening",
                            "content": "Scene analysis completed with identified objects and areas."
                        }
                    ]
                }
            }

    def _extract_lighting_templates(self) -> Dict:
        """
        從照明條件模組提取照明描述模板

        Returns:
            Dict: 照明模板字典
        """
        try:
            lighting_templates = {}

            # 從 LIGHTING_CONDITIONS 提取時間描述
            time_descriptions = LIGHTING_CONDITIONS.get("time_descriptions", {})

            for time_key, time_data in time_descriptions.items():
                if isinstance(time_data, dict) and "general" in time_data:
                    lighting_templates[time_key] = time_data["general"]
                else:
                    # 如果數據結構不符合預期，使用備用描述
                    lighting_templates[time_key] = f"The scene is captured during {time_key.replace('_', ' ')}."

            # 確保至少有基本的照明模板
            if not lighting_templates:
                self.logger.warning("No lighting templates found, using defaults")
                lighting_templates = self._get_default_lighting_templates()

            self.logger.debug("Extracted %d lighting templates", len(lighting_templates))
            return lighting_templates

        except Exception as e:
            self.logger.warning(f"Error extracting lighting templates: {str(e)}, using defaults")
            return self._get_default_lighting_templates()

    def _get_default_lighting_templates(self) -> Dict:
        """獲取默認照明模板"""
        return {
            "day_clear": "The scene is captured during clear daylight conditions.",
            "day_overcast": "The scene is captured during overcast daylight.",
            "night": "The scene is captured at night with artificial lighting.",
            "dawn": "The scene is captured during dawn with soft natural lighting.",
            "dusk": "The scene is captured during dusk with diminishing natural light.",
            "unknown": "The lighting conditions are not clearly identifiable."
        }

    def _initialize_default_templates(self, templates: Dict):
        """
        初始化默認模板作為備份機制

        Args:
            templates: 要檢查和補充的模板字典
        """
        try:
            # 置信度模板備份
            if "confidence_templates" not in templates or not templates["confidence_templates"]:
                templates["confidence_templates"] = {
                    "high": "{description} {details}",
                    "medium": "This appears to be {description} {details}",
                    "low": "This might be {description}, but the confidence is low. {details}"
                }

            # 場景詳細模板備份
            if "scene_detail_templates" not in templates or not templates["scene_detail_templates"]:
                templates["scene_detail_templates"] = {
                    "default": ["A scene with various elements and objects."]
                }

            # 物體填充模板備份
            if "object_template_fillers" not in templates or not templates["object_template_fillers"]:
                templates["object_template_fillers"] = {
                    "default": ["various items", "different objects", "multiple elements"]
                }

            # 視角模板備份
            if "viewpoint_templates" not in templates or not templates["viewpoint_templates"]:
                templates["viewpoint_templates"] = {
                    "eye_level": {
                        "prefix": "From eye level, ",
                        "observation": "the scene is viewed straight ahead.",
                        "short_desc": "at eye level"
                    },
                    "aerial": {
                        "prefix": "From above, ",
                        "observation": "the scene is viewed from a bird's-eye perspective.",
                        "short_desc": "from above"
                    },
                    "low_angle": {
                        "prefix": "From a low angle, ",
                        "observation": "the scene is viewed from below looking upward.",
                        "short_desc": "from below"
                    },
                    "elevated": {
                        "prefix": "From an elevated position, ",
                        "observation": "the scene is viewed from a higher vantage point.",
                        "short_desc": "from an elevated position"
                    }
                }

            # 文化模板備份
            if "cultural_templates" not in templates or not templates["cultural_templates"]:
                templates["cultural_templates"] = {
                    "asian": {
                        "elements": ["traditional architectural elements", "cultural signage", "Asian design features"],
                        "description": "The scene displays distinctive Asian cultural characteristics with {elements}."
                    },
                    "european": {
                        "elements": ["classical architecture", "European design elements", "historic features"],
                        "description": "The scene exhibits European architectural and cultural elements including {elements}."
                    }
                }

            self.logger.debug("Default templates initialized as backup")

        except Exception as e:
            self.logger.error(f"Error initializing default templates: {str(e)}")

    def _merge_custom_templates(self, custom_templates: Dict):
        """
        合併自定義模板到現有模板庫

        Args:
            custom_templates: 自定義模板字典
        """
        try:
            for template_category, custom_content in custom_templates.items():
                if template_category in self.templates:
                    if isinstance(self.templates[template_category], dict) and isinstance(custom_content, dict):
                        self.templates[template_category].update(custom_content)
                        self.logger.debug(f"Merged custom templates for category: {template_category}")
                    else:
                        self.templates[template_category] = custom_content
                        self.logger.debug(f"Replaced templates for category: {template_category}")
                else:
                    self.templates[template_category] = custom_content
                    self.logger.debug(f"Added new template category: {template_category}")

            self.logger.info("Successfully merged custom templates")

        except Exception as e:
            self.logger.warning(f"Error merging custom templates: {str(e)}")

    def _validate_templates(self):
        """
        驗證模板完整性和有效性
        """
        try:
            required_categories = [
                "scene_detail_templates",
                "object_template_fillers",
                "viewpoint_templates",
                "cultural_templates",
                "lighting_templates",
                "confidence_templates"
            ]

            missing_categories = []
            for category in required_categories:
                if category not in self.templates:
                    missing_categories.append(category)
                elif not self.templates[category]:
                    self.logger.warning(f"Template category '{category}' is empty")

            if missing_categories:
                error_msg = f"Missing required template categories: {missing_categories}"
                self.logger.warning(error_msg)
                # 為缺失的類別創建空模板
                for category in missing_categories:
                    self.templates[category] = {}

            # 驗證視角模板結構
            self._validate_viewpoint_templates()

            # 驗證文化模板結構
            self._validate_cultural_templates()

            self.logger.debug("Template validation completed successfully")

        except Exception as e:
            error_msg = f"Template validation failed: {str(e)}"
            self.logger.error(f"{error_msg}\n{traceback.format_exc()}")

    def _validate_viewpoint_templates(self):
        """驗證視角模板結構"""
        viewpoint_templates = self.templates.get("viewpoint_templates", {})

        for viewpoint, template_data in viewpoint_templates.items():
            if not isinstance(template_data, dict):
                self.logger.warning(f"Invalid viewpoint template structure for '{viewpoint}'")
                continue

            required_keys = ["prefix", "observation"]
            for key in required_keys:
                if key not in template_data:
                    self.logger.warning(f"Missing '{key}' in viewpoint template '{viewpoint}'")

    def _validate_cultural_templates(self):
        """驗證文化模板結構"""
        cultural_templates = self.templates.get("cultural_templates", {})

        for culture, template_data in cultural_templates.items():
            if not isinstance(template_data, dict):
                self.logger.warning(f"Invalid cultural template structure for '{culture}'")
                continue

            if "elements" not in template_data or "description" not in template_data:
                self.logger.warning(f"Missing required keys in cultural template '{culture}'")

    def _initialize_fallback_templates(self) -> Dict:
        """
        初始化備用模板系統，當主要載入失敗時使用

        Returns:
            Dict: 最基本的模板字典
        """
        return {
            "scene_detail_templates": {"default": ["A scene with various elements."]},
            "object_template_fillers": {"default": ["various items"]},
            "viewpoint_templates": {
                "eye_level": {
                    "prefix": "From eye level, ",
                    "observation": "the scene is viewed straight ahead.",
                    "short_desc": "at eye level"
                }
            },
            "cultural_templates": {"default": {"elements": ["elements"], "description": "The scene displays cultural elements."}},
            "lighting_templates": {"unknown": "The lighting conditions are not clearly identifiable."},
            "confidence_templates": {"medium": "{description} {details}"}
        }

    def get_template(self, category: str, key: Optional[str] = None) -> Any:
        """
        獲取指定類別的模板

        Args:
            category: 模板類別名稱
            key: 可選的具體模板鍵值

        Returns:
            Any: 請求的模板內容，如果不存在則返回空字典或空字符串
        """
        try:
            if category not in self.templates:
                self.logger.warning(f"Template category '{category}' not found")
                return {} if key is None else ""

            if key is None:
                return self.templates[category]

            category_templates = self.templates[category]
            if not isinstance(category_templates, dict):
                self.logger.warning(f"Template category '{category}' is not a dictionary")
                return ""

            if key not in category_templates:
                self.logger.warning(f"Template key '{key}' not found in category '{category}'")
                return ""

            return category_templates[key]

        except Exception as e:
            error_msg = f"Error retrieving template {category}.{key}: {str(e)}"
            self.logger.error(error_msg)
            return {} if key is None else ""

    def get_template_categories(self) -> List[str]:
        """
        獲取所有可用的模板類別名稱

        Returns:
            List[str]: 模板類別名稱列表
        """
        return list(self.templates.keys())

    def template_exists(self, category: str, key: Optional[str] = None) -> bool:
        """
        檢查模板是否存在

        Args:
            category: 模板類別
            key: 可選的模板鍵值

        Returns:
            bool: 模板是否存在
        """
        try:
            if category not in self.templates:
                return False

            if key is None:
                return True

            category_templates = self.templates[category]
            if isinstance(category_templates, dict):
                return key in category_templates

            return False

        except Exception as e:
            self.logger.warning(f"Error checking template existence for {category}.{key}: {str(e)}")
            return False

    def get_confidence_template(self, confidence_level: str) -> str:
        """
        獲取指定信心度級別的模板

        Args:
            confidence_level: 信心度級別 ('high', 'medium', 'low')

        Returns:
            str: 信心度模板字符串
        """
        try:
            confidence_templates = self.templates.get("confidence_templates", {})

            if confidence_level in confidence_templates:
                return confidence_templates[confidence_level]

            # 備用模板
            fallback_templates = {
                "high": "{description} {details}",
                "medium": "This appears to be {description} {details}",
                "low": "This might be {description}, but the confidence is low. {details}"
            }

            return fallback_templates.get(confidence_level, "{description} {details}")

        except Exception as e:
            self.logger.warning(f"Error getting confidence template for '{confidence_level}': {str(e)}")
            return "{description} {details}"

    def get_lighting_template(self, lighting_type: str) -> str:
        """
        獲取指定照明類型的模板

        Args:
            lighting_type: 照明類型

        Returns:
            str: 照明描述模板
        """
        try:
            lighting_templates = self.templates.get("lighting_templates", {})

            if lighting_type in lighting_templates:
                return lighting_templates[lighting_type]

            # 備用模板
            return f"The scene is captured with {lighting_type.replace('_', ' ')} lighting conditions."

        except Exception as e:
            self.logger.warning(f"Error getting lighting template for '{lighting_type}': {str(e)}")
            return "The lighting conditions are not clearly identifiable."

    def get_viewpoint_template(self, viewpoint: str) -> Dict[str, str]:
        """
        獲取指定視角的模板

        Args:
            viewpoint: 視角類型

        Returns:
            Dict[str, str]: 包含prefix、observation等鍵的視角模板字典
        """
        try:
            viewpoint_templates = self.templates.get("viewpoint_templates", {})

            if viewpoint in viewpoint_templates:
                return viewpoint_templates[viewpoint]

            # 備用模板
            fallback_templates = {
                "eye_level": {
                    "prefix": "From eye level, ",
                    "observation": "the scene is viewed straight ahead.",
                    "short_desc": "at eye level"
                },
                "aerial": {
                    "prefix": "From above, ",
                    "observation": "the scene is viewed from a bird's-eye perspective.",
                    "short_desc": "from above"
                },
                "low_angle": {
                    "prefix": "From a low angle, ",
                    "observation": "the scene is viewed from below looking upward.",
                    "short_desc": "from below"
                },
                "elevated": {
                    "prefix": "From an elevated position, ",
                    "observation": "the scene is viewed from a higher vantage point.",
                    "short_desc": "from an elevated position"
                }
            }

            return fallback_templates.get(viewpoint, fallback_templates["eye_level"])

        except Exception as e:
            self.logger.warning(f"Error getting viewpoint template for '{viewpoint}': {str(e)}")
            return {
                "prefix": "",
                "observation": "the scene is viewed normally.",
                "short_desc": "normally"
            }

    def get_cultural_template(self, cultural_context: str) -> Dict[str, Any]:
        """
        獲取指定文化語境的模板

        Args:
            cultural_context: 文化語境

        Returns:
            Dict[str, Any]: 文化模板字典
        """
        try:
            cultural_templates = self.templates.get("cultural_templates", {})

            if cultural_context in cultural_templates:
                return cultural_templates[cultural_context]

            # 備用模板
            return {
                "elements": ["cultural elements"],
                "description": f"The scene displays {cultural_context} cultural characteristics."
            }

        except Exception as e:
            self.logger.warning(f"Error getting cultural template for '{cultural_context}': {str(e)}")
            return {
                "elements": ["various elements"],
                "description": "The scene displays cultural characteristics."
            }

    def get_scene_detail_templates(self, scene_type: str, viewpoint: Optional[str] = None) -> List[str]:
        """
        獲取場景詳細描述模板

        Args:
            scene_type: 場景類型
            viewpoint: 可選的視角類型

        Returns:
            List[str]: 場景描述模板列表
        """
        try:
            scene_templates = self.templates.get("scene_detail_templates", {})

            # 首先嘗試獲取特定視角的模板
            if viewpoint:
                viewpoint_key = f"{scene_type}_{viewpoint}"
                if viewpoint_key in scene_templates:
                    return scene_templates[viewpoint_key]

            # 然後嘗試獲取場景類型的通用模板
            if scene_type in scene_templates:
                return scene_templates[scene_type]

            # 最後使用默認模板
            if "default" in scene_templates:
                return scene_templates["default"]

            # 備用模板
            return ["A scene with various elements and objects."]

        except Exception as e:
            self.logger.warning(f"Error getting scene detail templates for '{scene_type}': {str(e)}")
            return ["A scene with various elements and objects."]
