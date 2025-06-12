import logging
import traceback
import re
from typing import Dict, List, Optional, Union, Any

class TemplateProcessor:
    """
    模板處理器 - 負責模板填充、後處理和結構化模板渲染

    此類別專門處理模板的最終填充過程、文本格式化、
    語法修復以及結構化模板的渲染邏輯。
    """

    def __init__(self):
        """初始化模板處理器"""
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.debug("TemplateProcessor initialized successfully")

    def preprocess_template(self, template: str) -> str:
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

    def postprocess_filled_template(self, filled_template: str) -> str:
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

    def get_template_by_scene_type(self, scene_type: str, detected_objects: List[Dict],
                                  functional_zones: Dict, template_repository) -> str:
        """
        根據場景類型選擇合適的模板並進行標準化處理

        Args:
            scene_type: 場景類型
            detected_objects: 檢測到的物件列表
            functional_zones: 功能區域字典
            template_repository: 模板庫實例

        Returns:
            str: 標準化後的模板字符串
        """
        try:
            # 獲取場景的物件統計信息
            object_stats = self._analyze_scene_composition(detected_objects)
            zone_count = len(functional_zones) if functional_zones else 0

            # 根據場景複雜度和類型選擇模板
            templates = template_repository.templates
            if scene_type in templates:
                scene_templates = templates[scene_type]

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
            self.logger.error(f"Error selecting template for scene type '{scene_type}': {str(e)}")
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
            self.logger.warning(f"Error analyzing scene composition: {str(e)}")
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
            self.logger.warning(f"Error getting generic template: {str(e)}")
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
            self.logger.warning(f"Error standardizing template format: {str(e)}")
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
            self.logger.warning(f"Error improving template readability: {str(e)}")
            return template

    def process_structured_template(self, template: Dict[str, Any], scene_data: Dict[str, Any],
                                  statistics_processor) -> str:
        """
        處理結構化模板字典

        Args:
            template: 結構化模板字典
            scene_data: 場景分析資料
            statistics_processor: 統計處理器實例

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
                    zone_descriptions = statistics_processor.generate_zone_descriptions(zone_data, section)
                    if zone_descriptions:
                        description_parts.extend(zone_descriptions)

                elif section_type == "object_summary":
                    object_summary = statistics_processor.generate_object_summary(object_data, section)
                    if object_summary:
                        description_parts.append(object_summary)

                elif section_type == "conclusion":
                    conclusion = statistics_processor.generate_conclusion(template, zone_data, object_data)
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
            self.logger.error(f"Error standardizing final description: {str(e)}")
            return description

    def generate_fallback_description(self, scene_type: str, detected_objects: List[Dict]) -> str:
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
