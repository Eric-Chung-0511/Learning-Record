import logging
import traceback
import re
import random
from typing import Dict, List, Optional, Union, Any
import json

from scene_detail_templates import SCENE_DETAIL_TEMPLATES
from object_template_fillers import OBJECT_TEMPLATE_FILLERS
from viewpoint_templates import VIEWPOINT_TEMPLATES
from cultural_templates import CULTURAL_TEMPLATES
from lighting_conditions import LIGHTING_CONDITIONS
from confidence_templates import CONFIDENCE_TEMPLATES

class TemplateLoadingError(Exception):
    """模板載入或處理相關錯誤的自訂例外"""
    pass

class TemplateFillError(Exception):
    pass

class TemplateManager:
    """
    模板管理器 - 負責描述模板的載入、管理和填充

    此class 管理所有用於場景描述生成的模板資源，提供模板填充功能，
    並根據場景類型、物體檢測結果和上下文的資訊給出適當的描述內容。
    """

    def __init__(self, custom_templates_db: Optional[Dict] = None):
        """
        初始化模板管理器

        Args:
            custom_templates_db: 可選的自定義模板數據庫，如果提供則會與默認模板合併
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.template_registry = {}

        try:
            # 載入模板數據庫
            self.templates = self._load_templates()

            self.template_registry = self._initialize_template_registry()

            # 如果提供了自定義模板，則進行合併
            if custom_templates_db:
                self._merge_custom_templates(custom_templates_db)

            # 驗證模板完整性
            self._validate_templates()

            self.logger.info("TemplateManager initialized successfully with %d template categories",
                        len(self.templates))

        except Exception as e:
            error_msg = f"Failed to initialize TemplateManager: {str(e)}"
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
        初始化模板，包含各種場景類型的結構化模板

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

    def get_template_by_scene_type(self, scene_type: str, detected_objects: List[Dict],
                              functional_zones: Dict) -> str:
        """
        根據場景類型選擇合適的模板並進行標準化處理

        Args:
            scene_type: 場景類型
            detected_objects: 檢測到的物件列表
            functional_zones: 功能區域字典

        Returns:
            str: 標準化後的模板字符串
        """
        try:
            # 獲取場景的物件統計信息
            object_stats = self._analyze_scene_composition(detected_objects)
            zone_count = len(functional_zones) if functional_zones else 0

            # 根據場景複雜度和類型選擇模板
            if scene_type in self.templates:
                scene_templates = self.templates[scene_type]

                # 根據複雜度選擇合適的模板變體
                if zone_count >= 3 and object_stats.get("total_objects", 0) >= 10:
                    template_key = "complex"
                elif zone_count >= 2 or object_stats.get("total_objects", 0) >= 5:
                    template_key = "moderate"
                else:
                    template_key = "simple"

                if template_key in scene_templates:
                    raw_template = scene_templates[template_key]
                else:
                    raw_template = scene_templates.get("default", scene_templates[list(scene_templates.keys())[0]])
            else:
                # 如果沒有特定場景的模板，使用通用模板
                raw_template = self._get_generic_template(object_stats, zone_count)

            # 標準化模板中的佔位符和格式
            standardized_template = self._standardize_template_format(raw_template)
            return standardized_template

        except Exception as e:
            logger.error(f"Error selecting template for scene type '{scene_type}': {str(e)}")
            return self._get_fallback_template()

    def _analyze_scene_composition(self, detected_objects: List[Dict]) -> Dict:
        """
        分析場景組成以確定模板複雜度

        Args:
            detected_objects: 檢測到的物件列表

        Returns:
            Dict: 場景組成統計信息
        """
        try:
            total_objects = len(detected_objects)

            # 統計不同類型的物件
            object_categories = {}
            for obj in detected_objects:
                class_name = obj.get("class_name", "unknown")
                object_categories[class_name] = object_categories.get(class_name, 0) + 1

            # 計算場景多樣性
            unique_categories = len(object_categories)

            return {
                "total_objects": total_objects,
                "unique_categories": unique_categories,
                "category_distribution": object_categories,
                "complexity_score": min(total_objects * 0.3 + unique_categories * 0.7, 10)
            }

        except Exception as e:
            logger.warning(f"Error analyzing scene composition: {str(e)}")
            return {"total_objects": 0, "unique_categories": 0, "complexity_score": 0}

    def _get_generic_template(self, object_stats: Dict, zone_count: int) -> str:
        """
        獲取通用模板

        Args:
            object_stats: 物件統計信息
            zone_count: 功能區域數量

        Returns:
            str: 通用模板字符串
        """
        try:
            complexity_score = object_stats.get("complexity_score", 0)

            if complexity_score >= 7 or zone_count >= 3:
                return "This scene presents a comprehensive view featuring {functional_area} with {primary_objects}. The spatial organization demonstrates {spatial_arrangement} across multiple {activity_areas}, creating a dynamic environment with diverse elements and clear functional zones."
            elif complexity_score >= 4 or zone_count >= 2:
                return "The scene displays {functional_area} containing {primary_objects}. The arrangement shows {spatial_organization} with distinct areas serving different purposes within the overall space."
            else:
                return "A {scene_description} featuring {primary_objects} arranged in {basic_layout} within the visible area."

        except Exception as e:
            logger.warning(f"Error getting generic template: {str(e)}")
            return self._get_fallback_template()

    def _get_fallback_template(self) -> str:
        """
        獲取備用模板

        Returns:
            str: 備用模板字符串
        """
        return "A scene featuring various elements and organized areas of activity within the visible space."

    def _standardize_template_format(self, template: str) -> str:
        """
        標準化模板格式，確保佔位符和表達方式符合自然語言要求

        Args:
            template: 原始模板字符串

        Returns:
            str: 標準化後的模板字符串
        """
        try:
            if not template:
                return self._get_fallback_template()

            import re
            standardized = template

            # 標準化佔位符格式，移除技術性標記
            placeholder_mapping = {
                r'\{zone_\d+\}': '{functional_area}',
                r'\{object_group_\d+\}': '{primary_objects}',
                r'\{region_\d+\}': '{spatial_area}',
                r'\{category_\d+\}': '{object_category}',
                r'\{area_\d+\}': '{activity_area}',
                r'\{section_\d+\}': '{scene_section}'
            }

            for pattern, replacement in placeholder_mapping.items():
                standardized = re.sub(pattern, replacement, standardized)

            # 標準化常見的技術性術語
            term_replacements = {
                'functional_zones': 'areas of activity',
                'object_detection': 'visible elements',
                'category_regions': 'organized sections',
                'spatial_distribution': 'arrangement throughout the space',
                'viewpoint_analysis': 'perspective view'
            }

            for tech_term, natural_term in term_replacements.items():
                standardized = standardized.replace(tech_term, natural_term)

            # 確保模板語法的自然性
            standardized = self._improve_template_readability(standardized)

            return standardized

        except Exception as e:
            logger.warning(f"Error standardizing template format: {str(e)}")
            return template if template else self._get_fallback_template()

    def _improve_template_readability(self, template: str) -> str:
        """
        改善模板的可讀性和自然性

        Args:
            template: 模板字符串

        Returns:
            str: 改善後的模板字符串
        """
        try:
            import re

            # 移除多餘的空格和換行
            improved = re.sub(r'\s+', ' ', template).strip()

            # 改善句子連接
            improved = improved.replace(' . ', '. ')
            improved = improved.replace(' , ', ', ')
            improved = improved.replace(' ; ', '; ')

            # 確保適當的句號結尾
            if improved and not improved.endswith(('.', '!', '?')):
                improved += '.'

            # 改善常見的表達問題
            readability_fixes = [
                (r'\bthe the\b', 'the'),
                (r'\ba a\b', 'a'),
                (r'\ban an\b', 'an'),
                (r'\bwith with\b', 'with'),
                (r'\bin in\b', 'in'),
                (r'\bof of\b', 'of'),
                (r'\band and\b', 'and')
            ]

            for pattern, replacement in readability_fixes:
                improved = re.sub(pattern, replacement, improved, flags=re.IGNORECASE)

            return improved

        except Exception as e:
            logger.warning(f"Error improving template readability: {str(e)}")
            return template

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

    def fill_template(self, template: str, detected_objects: List[Dict], scene_type: str,
             places365_info: Optional[Dict] = None,
             object_statistics: Optional[Dict] = None) -> str:
        """
        填充模板中的佔位符，增強容錯處理

        Args:
            template: 包含佔位符的模板字符串
            detected_objects: 檢測到的物體列表
            scene_type: 場景類型
            places365_info: Places365場景分類信息
            object_statistics: 物體統計信息

        Returns:
            str: 填充後的模板字符串，確保語法正確
        """
        try:
            self.logger.debug(f"Filling template for scene_type: {scene_type}")

            if not template or not template.strip():
                return "A scene with various elements."

            # 預處理模板，移除可能的問題模式
            template = self._preprocess_template(template)

            # 查找模板中的佔位符
            placeholders = re.findall(r'\{([^}]+)\}', template)
            filled_template = template

            # 獲取模板填充器
            fillers = self.templates.get("object_template_fillers", {})

            # 基於物體統計信息生成替換內容
            statistics_based_replacements = self._generate_statistics_replacements(object_statistics)

            # 生成默認替換內容
            default_replacements = self._generate_default_replacements()

            # 添加Places365上下文信息
            places365_replacements = self._generate_places365_replacements(places365_info)

            # 添加功能區域信息到場景數據中以便後續使用
            scene_functional_zones = None
            if hasattr(self, '_current_functional_zones'):
                scene_functional_zones = self._current_functional_zones

            # 合併所有替換內容（優先順序是統計信息 > Places365 > 默認）
            all_replacements = {**default_replacements, **places365_replacements, **statistics_based_replacements}

            # 填充每個佔位符
            for placeholder in placeholders:
                try:
                    replacement = self._get_placeholder_replacement(
                        placeholder, fillers, all_replacements, detected_objects, scene_type
                    )

                    # 確保替換內容不為空且有意義
                    if not replacement or not replacement.strip():
                        replacement = self._get_emergency_replacement(placeholder)

                    filled_template = filled_template.replace(f"{{{placeholder}}}", replacement)

                except Exception as placeholder_error:
                    self.logger.warning(f"Failed to replace placeholder '{placeholder}': {str(placeholder_error)}")
                    # 使用緊急替換值
                    emergency_replacement = self._get_emergency_replacement(placeholder)
                    filled_template = filled_template.replace(f"{{{placeholder}}}", emergency_replacement)

            # 修復可能的語法問題
            filled_template = self._postprocess_filled_template(filled_template)

            self.logger.debug("Template filling completed successfully")
            return filled_template

        except Exception as e:
            error_msg = f"Error filling template: {str(e)}"
            self.logger.error(f"{error_msg}\n{traceback.format_exc()}")
            # 返回安全的備用內容
            return self._generate_fallback_description(scene_type, detected_objects)

    def _preprocess_template(self, template: str) -> str:
        """
        預處理模板，修復常見問題

        Args:
            template: 原始模板字符串

        Returns:
            str: 預處理後的模板
        """
        try:
            # 移除可能導致問題的模式
            template = re.sub(r'\{[^}]*\}\s*,\s*\{[^}]*\}', '{combined_elements}', template)

            # 確保模板不以逗號開始
            template = re.sub(r'^[,\s]*', '', template)

            return template.strip()

        except Exception as e:
            self.logger.warning(f"Error preprocessing template: {str(e)}")
            return template

    def _get_emergency_replacement(self, placeholder: str) -> str:
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

    def _postprocess_filled_template(self, filled_template: str) -> str:
        """
        後處理填充完成的模板，修復語法問題

        Args:
            filled_template: 填充後的模板字符串

        Returns:
            str: 修復後的模板字符串
        """
        try:
            # 修復 "In , " 模式
            filled_template = re.sub(r'\bIn\s*,\s*', 'In this scene, ', filled_template)
            filled_template = re.sub(r'\bAt\s*,\s*', 'At this location, ', filled_template)
            filled_template = re.sub(r'\bWithin\s*,\s*', 'Within this area, ', filled_template)

            # 修復連續逗號
            filled_template = re.sub(r',\s*,', ',', filled_template)

            # 修復開頭的逗號
            filled_template = re.sub(r'^[,\s]*', '', filled_template)

            # 確保首字母大寫
            if filled_template and not filled_template[0].isupper():
                filled_template = filled_template[0].upper() + filled_template[1:]

            # 確保以句號結尾
            if filled_template and not filled_template.endswith(('.', '!', '?')):
                filled_template += '.'

            return filled_template.strip()

        except Exception as e:
            self.logger.warning(f"Error postprocessing filled template: {str(e)}")
            return filled_template

    def _generate_fallback_description(self, scene_type: str, detected_objects: List[Dict]) -> str:
        """
        生成備用描述，當模板填充完全失敗時使用

        Args:
            scene_type: 場景類型
            detected_objects: 檢測到的物體列表

        Returns:
            str: 備用描述
        """
        try:
            object_count = len(detected_objects)

            if object_count == 0:
                return f"A {scene_type.replace('_', ' ')} scene."
            elif object_count == 1:
                return f"A {scene_type.replace('_', ' ')} scene with one visible element."
            else:
                return f"A {scene_type.replace('_', ' ')} scene with {object_count} visible elements."

        except Exception as e:
            self.logger.warning(f"Error generating fallback description: {str(e)}")
            return "A scene with various elements."

    def _generate_statistics_replacements(self, object_statistics: Optional[Dict]) -> Dict[str, str]:
        """
        基於物體統計信息生成模板替換內容

        Args:
            object_statistics: 物體統計信息

        Returns:
            Dict[str, str]: 統計信息基礎的替換內容
        """
        replacements = {}

        if not object_statistics:
            return replacements

        try:
            # 處理植物元素
            if "potted plant" in object_statistics:
                count = object_statistics["potted plant"]["count"]
                if count == 1:
                    replacements["plant_elements"] = "a potted plant"
                elif count <= 3:
                    replacements["plant_elements"] = f"{count} potted plants"
                else:
                    replacements["plant_elements"] = f"multiple potted plants ({count} total)"

            # 處理座位(椅子)相關
            if "chair" in object_statistics:
                count = object_statistics["chair"]["count"]

                # 使用統一的數字轉換邏輯
                number_words = {
                    1: "one", 2: "two", 3: "three", 4: "four",
                    5: "five", 6: "six", 7: "seven", 8: "eight",
                    9: "nine", 10: "ten", 11: "eleven", 12: "twelve"
                }

                if count == 1:
                    replacements["seating"] = "a chair"
                    replacements["furniture"] = "a chair"
                elif count in number_words:
                    word_count = number_words[count]
                    replacements["seating"] = f"{word_count} chairs"
                    replacements["furniture"] = f"{word_count} chairs"
                elif count <= 20:
                    replacements["seating"] = f"several chairs"
                    replacements["furniture"] = f"several chairs"
                else:
                    replacements["seating"] = f"numerous chairs ({count} total)"
                    replacements["furniture"] = f"numerous chairs"


            # 處理混合家具情況（當存在多種家具類型時）
            furniture_items = []
            furniture_counts = []

            # 收集所有家具類型的統計
            for furniture_type in ["chair", "dining table", "couch", "bed"]:
                if furniture_type in object_statistics:
                    count = object_statistics[furniture_type]["count"]
                    if count > 0:
                        furniture_items.append(furniture_type)
                        furniture_counts.append(count)

            # 如果只有椅子,那就用上面的方式
            # 如果有多種家具類型，生成組合描述
            if len(furniture_items) > 1 and "furniture" not in replacements:
                main_furniture = furniture_items[0]  # 數量最多的家具類型
                main_count = furniture_counts[0]

                if main_furniture == "chair":
                    number_words = ["", "one", "two", "three", "four", "five", "six"]
                    if main_count <= 6:
                        replacements["furniture"] = f"{number_words[main_count]} chairs and other furniture"
                    else:
                        replacements["furniture"] = "multiple chairs and other furniture"

            # 處理人員
            if "person" in object_statistics:
                count = object_statistics["person"]["count"]
                if count == 1:
                    replacements["people_and_vehicles"] = "a person"
                    replacements["pedestrian_flow"] = "an individual walking"
                elif count <= 5:
                    replacements["people_and_vehicles"] = f"{count} people"
                    replacements["pedestrian_flow"] = f"{count} people walking"
                else:
                    replacements["people_and_vehicles"] = f"many people ({count} individuals)"
                    replacements["pedestrian_flow"] = f"a crowd of {count} people"

            # 處理桌子設置
            if "dining table" in object_statistics:
                count = object_statistics["dining table"]["count"]
                if count == 1:
                    replacements["table_setup"] = "a dining table"
                    replacements["table_description"] = "a dining surface"
                else:
                    replacements["table_setup"] = f"{count} dining tables"
                    replacements["table_description"] = f"{count} dining surfaces"

            self.logger.debug(f"Generated {len(replacements)} statistics-based replacements")

        except Exception as e:
            self.logger.warning(f"Error generating statistics replacements: {str(e)}")

        return replacements

    def _generate_places365_replacements(self, places365_info: Optional[Dict]) -> Dict[str, str]:
        """
        基於Places365信息生成模板替換內容

        Args:
            places365_info: Places365場景分類信息

        Returns:
            Dict[str, str]: Places365基礎的替換內容
        """
        replacements = {}

        if not places365_info or places365_info.get('confidence', 0) <= 0.35:
            replacements["places365_context"] = ""
            replacements["places365_atmosphere"] = ""
            return replacements

        try:
            scene_label = places365_info.get('scene_label', '').replace('_', ' ')
            attributes = places365_info.get('attributes', [])

            # 生成場景上下文
            if scene_label:
                replacements["places365_context"] = f"characteristic of a {scene_label}"
            else:
                replacements["places365_context"] = ""

            # 生成氛圍描述
            if 'natural_lighting' in attributes:
                replacements["places365_atmosphere"] = "with natural illumination"
            elif 'artificial_lighting' in attributes:
                replacements["places365_atmosphere"] = "under artificial lighting"
            else:
                replacements["places365_atmosphere"] = ""

            self.logger.debug("Generated Places365-based replacements")

        except Exception as e:
            self.logger.warning(f"Error generating Places365 replacements: {str(e)}")
            replacements["places365_context"] = ""
            replacements["places365_atmosphere"] = ""

        return replacements

    def _generate_default_replacements(self) -> Dict[str, str]:
        """
        生成默認的模板替換內容

        Returns:
            Dict[str, str]: 默認替換內容
        """
        return {

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

            # 亞洲商業街相
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

    def _generate_objects_summary(self, detected_objects: List[Dict]) -> str:
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

            # calculate object statistic
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

    def _get_placeholder_replacement(self, placeholder: str, fillers: Dict,
                           all_replacements: Dict, detected_objects: List[Dict],
                           scene_type: str) -> str:
        """
        獲取特定佔位符的替換內容，確保永遠不返回空值
        """
        try:
            # 優先處理動態內容生成的佔位符
            dynamic_placeholders = [
                'primary_objects', 'detected_objects_summary', 'main_objects',
                'functional_area', 'functional_zones_description', 'scene_elements'
            ]

            if placeholder in dynamic_placeholders:
                dynamic_content = self._generate_objects_summary(detected_objects)
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
            scene_specific_replacement = self._generate_scene_specific_content(
                placeholder, detected_objects, scene_type
            )
            if scene_specific_replacement and scene_specific_replacement.strip():
                return scene_specific_replacement.strip()

            # 通用備用字典 - 擴展版本
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
            scene_based_defaults = self._get_scene_based_default(placeholder, scene_type)
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

    def _get_scene_based_default(self, placeholder: str, scene_type: str) -> Optional[str]:
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

    def _generate_scene_specific_content(self, placeholder: str, detected_objects: List[Dict],
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

    def apply_template(self, template: Union[str, Dict[str, Any]], scene_data: Dict[str, Any]) -> str:
        """
        應用選定的模板來生成場景描述

        Args:
            template: 模板字符串或模板內容字典
            scene_data: 場景分析的資料字典

        Returns:
            str: 最終生成的場景描述
        """
        try:
            # 如果傳入的是字符串模板，直接使用填充邏輯
            if isinstance(template, str):
                self.logger.debug("Processing string template directly")

                # 提取場景數據
                detected_objects = scene_data.get("detected_objects", [])
                scene_type = scene_data.get("scene_type", "general")
                places365_info = scene_data.get("places365_info")
                object_statistics = scene_data.get("object_statistics")
                functional_zones = scene_data.get("functional_zones", {})

                # 暫存功能區域資訊供填充邏輯使用
                self._current_functional_zones = functional_zones

                # 使用現有的填充邏輯
                filled_description = self.fill_template(
                    template,
                    detected_objects,
                    scene_type,
                    places365_info,
                    object_statistics
                )

                # 清理暫存資訊
                if hasattr(self, '_current_functional_zones'):
                    delattr(self, '_current_functional_zones')

                return filled_description

            # 如果傳入的是字典結構模板
            elif isinstance(template, dict):
                self.logger.debug("Processing structured template")
                return self._process_structured_template(template, scene_data)

            # 如果是模板名稱字符串且需要從registry獲取
            elif hasattr(self, 'template_registry') and template in self.template_registry:
                template_dict = self.template_registry[template]
                return self._process_structured_template(template_dict, scene_data)

            else:
                self.logger.warning(f"Invalid template format or template not found: {type(template)}")
                return self._generate_fallback_scene_description(scene_data)

        except Exception as e:
            self.logger.error(f"Error applying template: {str(e)}")
            return self._generate_fallback_scene_description(scene_data)

    def _process_structured_template(self, template: Dict[str, Any], scene_data: Dict[str, Any]) -> str:
        """
        處理結構化模板字典

        Args:
            template: 結構化模板字典
            scene_data: 場景分析資料

        Returns:
            str: 生成的場景描述
        """
        try:
            # 提取 scene_data 中各區塊資料
            zone_data = scene_data.get("functional_zones", scene_data.get("zones", {}))
            object_data = scene_data.get("detected_objects", [])
            scene_context = scene_data.get("scene_context", "")

            # 獲取模板結構
            structure = template.get("structure", [])
            if not structure:
                self.logger.warning("Template has no structure defined")
                return self._generate_fallback_scene_description(scene_data)

            description_parts = []

            # 按照模板結構生成描述
            for section in structure:
                section_type = section.get("type", "")
                content = section.get("content", "")

                if section_type == "opening":
                    description_parts.append(content)

                elif section_type == "zone_analysis":
                    zone_descriptions = self._generate_zone_descriptions(zone_data, section)
                    if zone_descriptions:
                        description_parts.extend(zone_descriptions)

                elif section_type == "object_summary":
                    object_summary = self._generate_object_summary(object_data, section)
                    if object_summary:
                        description_parts.append(object_summary)

                elif section_type == "conclusion":
                    conclusion = self._generate_conclusion(template, zone_data, object_data)
                    if conclusion:
                        description_parts.append(conclusion)

            # 合併並標準化輸出
            final_description = self._standardize_final_description(" ".join(description_parts))
            self.logger.info("Successfully applied structured template")
            return final_description

        except Exception as e:
            self.logger.error(f"Error processing structured template: {str(e)}")
            return self._generate_fallback_scene_description(scene_data)

    def _generate_fallback_scene_description(self, scene_data: Dict[str, Any]) -> str:
        """
        生成備用場景描述

        Args:
            scene_data: 場景分析資料

        Returns:
            str: 備用場景描述
        """
        try:
            detected_objects = scene_data.get("detected_objects", [])
            zones = scene_data.get("functional_zones", scene_data.get("zones", {}))
            scene_type = scene_data.get("scene_type", "general")

            object_count = len(detected_objects)
            zone_count = len(zones)

            if zone_count > 0 and object_count > 0:
                return f"Scene analysis completed with {zone_count} functional areas containing {object_count} identified objects."
            elif object_count > 0:
                return f"Scene analysis identified {object_count} objects in this {scene_type.replace('_', ' ')} environment."
            else:
                return f"Scene analysis completed for this {scene_type.replace('_', ' ')} environment."

        except Exception as e:
            self.logger.warning(f"Error generating fallback description: {str(e)}")
            return "Scene analysis completed with detected objects and functional areas."


    def _generate_zone_descriptions(self, zone_data: Dict[str, Any], section: Dict[str, Any]) -> List[str]:
        """
        生成功能區域描述
        """
        try:
            descriptions = []

            if not zone_data:
                return descriptions

            # 直接處理區域資料（zone_data 本身就是區域字典）
            sorted_zones = sorted(zone_data.items(),
                                key=lambda x: len(x[1].get("objects", [])),
                                reverse=True)

            for zone_name, zone_info in sorted_zones:
                description = zone_info.get("description", "")
                objects = zone_info.get("objects", [])

                if objects:
                    # 使用現有描述或生成基於物件的描述
                    if description and not any(tech in description.lower() for tech in ['zone', 'area', 'region']):
                        zone_desc = description
                    else:
                        # 生成更自然的區域描述
                        clean_zone_name = zone_name.replace('_', ' ').replace(' area', '').replace(' zone', '')
                        object_list = ', '.join(objects[:3])

                        if 'crossing' in zone_name or 'pedestrian' in zone_name:
                            zone_desc = f"In the central crossing area, there are {object_list}."
                        elif 'vehicle' in zone_name or 'traffic' in zone_name:
                            zone_desc = f"The vehicle movement area includes {object_list}."
                        elif 'control' in zone_name:
                            zone_desc = f"Traffic control elements include {object_list}."
                        else:
                            zone_desc = f"The {clean_zone_name} contains {object_list}."

                        if len(objects) > 3:
                            zone_desc += f" Along with {len(objects) - 3} additional elements."

                    descriptions.append(zone_desc)

            return descriptions

        except Exception as e:
            logger.error(f"Error generating zone descriptions: {str(e)}")
            return []

    def _generate_object_summary(self, object_data: List[Dict], section: Dict[str, Any]) -> str:
        """
        生成物件摘要描述
        """
        try:
            if not object_data:
                return ""

            # 統計物件類型並計算重要性
            object_stats = {}
            for obj in object_data:
                class_name = obj.get("class_name", "unknown")
                confidence = obj.get("confidence", 0.5)

                if class_name not in object_stats:
                    object_stats[class_name] = {"count": 0, "total_confidence": 0}

                object_stats[class_name]["count"] += 1
                object_stats[class_name]["total_confidence"] += confidence

            # 按重要性排序（結合數量和置信度）
            sorted_objects = []
            for class_name, stats in object_stats.items():
                count = stats["count"]
                avg_confidence = stats["total_confidence"] / count
                importance = count * 0.6 + avg_confidence * 0.4
                sorted_objects.append((class_name, count, importance))

            sorted_objects.sort(key=lambda x: x[2], reverse=True)

            # 生成自然語言描述
            descriptions = []
            for class_name, count, _ in sorted_objects[:5]:
                clean_name = class_name.replace('_', ' ')
                if count == 1:
                    article = "an" if clean_name[0].lower() in 'aeiou' else "a"
                    descriptions.append(f"{article} {clean_name}")
                else:
                    descriptions.append(f"{count} {clean_name}s")

            if len(descriptions) == 1:
                return f"The scene features {descriptions[0]}."
            elif len(descriptions) == 2:
                return f"The scene features {descriptions[0]} and {descriptions[1]}."
            else:
                main_items = ", ".join(descriptions[:-1])
                return f"The scene features {main_items}, and {descriptions[-1]}."

        except Exception as e:
            self.logger.error(f"Error generating object summary: {str(e)}")
            return ""

    def _generate_conclusion(self, template: Dict[str, Any], zone_data: Dict[str, Any],
                            object_data: List[Dict]) -> str:
        """
        生成結論描述
        """
        try:
            scene_type = template.get("scene_type", "general")
            zones_count = len(zone_data)
            objects_count = len(object_data)

            if scene_type == "indoor":
                conclusion = f"This indoor environment demonstrates clear functional organization with {zones_count} distinct areas and {objects_count} identified objects."
            elif scene_type == "outdoor":
                conclusion = f"This outdoor scene shows dynamic activity patterns across {zones_count} functional zones with {objects_count} detected elements."
            else:
                conclusion = f"The scene analysis reveals {zones_count} functional areas containing {objects_count} identifiable objects."

            return conclusion

        except Exception as e:
            logger.error(f"Error generating conclusion: {str(e)}")
            return ""

    def _standardize_final_description(self, description: str) -> str:
        """
        對最終描述進行標準化處理

        Args:
            description: 原始描述文本

        Returns:
            str: 標準化後的描述文本
        """
        try:
            # 移除多餘空格
            description = " ".join(description.split())

            # 確保句子間有適當間距
            description = description.replace(". ", ". ")

            # 移除任何殘留的技術性標識符
            technical_patterns = [
                r'zone_\d+', r'area_\d+', r'region_\d+',
                r'_zone', r'_area', r'_region'
            ]

            for pattern in technical_patterns:
                description = re.sub(pattern, '', description, flags=re.IGNORECASE)

            return description.strip()

        except Exception as e:
            logger.error(f"Error standardizing final description: {str(e)}")
            return description
