import logging
import traceback
import re
from typing import Dict, List, Optional, Union, Any

from template_repository import TemplateRepository
from content_generator import ContentGenerator
from statistics_processor import StatisticsProcessor
from template_processor import TemplateProcessor

class TemplateManager:
    """
    模板管理器 - 負責描述模板的載入、管理和填充的統一介面,
    匯總於EnhancedSceneDescriber

    此類別作為模板系統的統一門面，協調四個專業化組件的運作：
    - TemplateRepository: 模板載入和存儲
    - ContentGenerator: 基礎內容生成
    - StatisticsProcessor: 複雜統計分析
    - TemplateProcessor: 模板處理和渲染

    重要：此類別必須維持與 EnhancedSceneDescriber 的完全相容性
    """

    def __init__(self, custom_templates_db: Optional[Dict] = None):
        """
        初始化模板管理器

        Args:
            custom_templates_db: 可選的自定義模板數據庫，如果提供則會與默認模板合併
        """
        self.logger = logging.getLogger(self.__class__.__name__)

        try:
            # 初始化四個專業化組件
            self.template_repository = TemplateRepository(custom_templates_db)
            self.content_generator = ContentGenerator()
            self.statistics_processor = StatisticsProcessor()
            self.template_processor = TemplateProcessor()

            # 提供向後相容性的屬性存取 
            self.templates = self.template_repository.templates
            self.template_registry = self.template_repository.template_registry

            self.logger.info("TemplateManager facade initialized successfully with %d template categories",
                           len(self.templates))

        except Exception as e:
            error_msg = f"Failed to initialize TemplateManager: {str(e)}"
            self.logger.error(f"{error_msg}\n{traceback.format_exc()}")
            # 初始化基本的空模板以保持系統穩定性
            self._initialize_fallback_state()

    def _initialize_fallback_state(self):
        """
        初始化備用狀態，確保系統在組件初始化失敗時仍能運作
        """
        try:
            # 創建最基本的組件實例
            self.template_repository = TemplateRepository()
            self.content_generator = ContentGenerator()
            self.statistics_processor = StatisticsProcessor()
            self.template_processor = TemplateProcessor()

            # 確保向後相容性屬性可用
            self.templates = getattr(self.template_repository, 'templates', {})
            self.template_registry = getattr(self.template_repository, 'template_registry', {})

        except Exception as e:
            self.logger.critical(f"Failed to initialize fallback state: {str(e)}")
            # 最後手段：創建空的屬性以避免 AttributeError
            self.templates = {}
            self.template_registry = {}

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
            template = self.template_processor.preprocess_template(template)

            # 查找模板中的佔位符
            placeholders = re.findall(r'\{([^}]+)\}', template)
            filled_template = template

            # 獲取模板填充器
            fillers = self.templates.get("object_template_fillers", {})

            # 基於物體統計信息生成替換內容
            statistics_based_replacements = self.statistics_processor.generate_statistics_replacements(object_statistics)

            # 生成默認替換內容
            default_replacements = self.content_generator.default_replacements

            # 添加Places365上下文信息
            places365_replacements = self.statistics_processor.generate_places365_replacements(places365_info)

            # 添加功能區域信息到場景數據中以便後續使用 - 重要：保持原有邏輯
            if hasattr(self, '_current_functional_zones'):
                scene_functional_zones = self._current_functional_zones

            # 合併所有替換內容（優先順序是統計的資訊 > Places365 > 默認）
            all_replacements = {**default_replacements, **places365_replacements, **statistics_based_replacements}

            # 填充每個佔位符
            for placeholder in placeholders:
                try:
                    replacement = self.content_generator.get_placeholder_replacement(
                        placeholder, fillers, all_replacements, detected_objects, scene_type
                    )

                    # 確保替換內容不為空且有意義
                    if not replacement or not replacement.strip():
                        replacement = self.content_generator.get_emergency_replacement(placeholder)

                    filled_template = filled_template.replace(f"{{{placeholder}}}", replacement)

                except Exception as placeholder_error:
                    self.logger.warning(f"Failed to replace placeholder '{placeholder}': {str(placeholder_error)}")
                    # 使用緊急替換值
                    emergency_replacement = self.content_generator.get_emergency_replacement(placeholder)
                    filled_template = filled_template.replace(f"{{{placeholder}}}", emergency_replacement)

            # 修復可能的語法問題
            filled_template = self.template_processor.postprocess_filled_template(filled_template)

            self.logger.debug("Template filling completed successfully")
            return filled_template

        except Exception as e:
            error_msg = f"Error filling template: {str(e)}"
            self.logger.error(f"{error_msg}\n{traceback.format_exc()}")
            # 返回安全的備用內容
            return self.template_processor.generate_fallback_description(scene_type, detected_objects)

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
                return self.template_processor.process_structured_template(
                    template, scene_data, self.statistics_processor
                )

            # 如果是模板名稱字符串且需要從registry獲取
            elif hasattr(self, 'template_registry') and template in self.template_registry:
                template_dict = self.template_registry[template]
                return self.template_processor.process_structured_template(
                    template_dict, scene_data, self.statistics_processor
                )

            else:
                self.logger.warning(f"Invalid template format or template not found: {type(template)}")
                return self.template_processor._generate_fallback_scene_description(scene_data)

        except Exception as e:
            self.logger.error(f"Error applying template: {str(e)}")
            return self.template_processor._generate_fallback_scene_description(scene_data)

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
        return self.template_processor.get_template_by_scene_type(
            scene_type, detected_objects, functional_zones, self.template_repository
        )

    def get_template(self, category: str, key: Optional[str] = None) -> Any:
        """
        獲取指定類別的模板

        Args:
            category: 模板類別名稱
            key: 可選的具體模板鍵值

        Returns:
            Any: 請求的模板內容，如果不存在則返回空字典或空字符串
        """
        return self.template_repository.get_template(category, key)

    def get_template_categories(self) -> List[str]:
        """
        獲取所有可用的模板類別名稱

        Returns:
            List[str]: 模板類別名稱列表
        """
        return self.template_repository.get_template_categories()

    def template_exists(self, category: str, key: Optional[str] = None) -> bool:
        """
        檢查模板是否存在

        Args:
            category: 模板類別
            key: 可選的模板鍵值

        Returns:
            bool: 模板是否存在
        """
        return self.template_repository.template_exists(category, key)

    def get_confidence_template(self, confidence_level: str) -> str:
        """
        獲取指定信心度級別的模板

        Args:
            confidence_level: 信心度級別 ('high', 'medium', 'low')

        Returns:
            str: 信心度模板字符串
        """
        return self.template_repository.get_confidence_template(confidence_level)

    def get_lighting_template(self, lighting_type: str) -> str:
        """
        獲取指定照明類型的模板

        Args:
            lighting_type: 照明類型

        Returns:
            str: 照明描述模板
        """
        return self.template_repository.get_lighting_template(lighting_type)

    def get_viewpoint_template(self, viewpoint: str) -> Dict[str, str]:
        """
        獲取指定視角的模板

        Args:
            viewpoint: 視角類型

        Returns:
            Dict[str, str]: 包含prefix、observation等鍵的視角模板字典
        """
        return self.template_repository.get_viewpoint_template(viewpoint)

    def get_cultural_template(self, cultural_context: str) -> Dict[str, Any]:
        """
        獲取指定文化語境的模板

        Args:
            cultural_context: 文化語境

        Returns:
            Dict[str, Any]: 文化模板字典
        """
        return self.template_repository.get_cultural_template(cultural_context)

    def get_scene_detail_templates(self, scene_type: str, viewpoint: Optional[str] = None) -> List[str]:
        """
        獲取場景詳細描述模板

        Args:
            scene_type: 場景類型
            viewpoint: 可選的視角類型

        Returns:
            List[str]: 場景描述模板列表
        """
        return self.template_repository.get_scene_detail_templates(scene_type, viewpoint)
