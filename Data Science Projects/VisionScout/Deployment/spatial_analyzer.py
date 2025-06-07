
import os
import numpy as np
import logging
import traceback
from typing import Dict, List, Tuple, Any, Optional

from object_categories import OBJECT_CATEGORIES
from region_analyzer import RegionAnalyzer
from object_extractor import ObjectExtractor
from scene_viewpoint_analyzer import SceneViewpointAnalyzer
from zone_evaluator import ZoneEvaluator
from scene_zone_identifier import SceneZoneIdentifier
from functional_zone_identifier import FunctionalZoneIdentifier

logger = logging.getLogger(__name__)

class SpatialAnalyzer:
    """
    分析圖像中物件間空間關係的主要類別
    處理區域分配、物件定位和功能區域識別
    使用Facade模式整合多個子組件，保持外部接口的穩定性
    """

    def __init__(self, class_names: Dict[int, str] = None, object_categories=None):
        """
        初始化空間分析器，包含圖像區域定義

        Args:
            class_names: 類別ID到類別名稱的映射字典
            object_categories: 物件類別分組字典
        """
        try:
            # 初始化所有子組件
            self.class_names = class_names
            self.OBJECT_CATEGORIES = object_categories or {}

            self.region_analyzer = RegionAnalyzer()
            self.object_extractor = ObjectExtractor(class_names, object_categories)

            self.scene_viewpoint_analyzer = SceneViewpointAnalyzer()

            self.zone_evaluator = ZoneEvaluator()
            self.scene_zone_identifier = SceneZoneIdentifier()
            self.functional_zone_identifier = FunctionalZoneIdentifier(
                zone_evaluator=self.zone_evaluator,
                scene_zone_identifier=self.scene_zone_identifier,
                scene_viewpoint_analyzer=self.scene_viewpoint_analyzer,
                object_categories=self.OBJECT_CATEGORIES
            )

            self.enhance_descriptor = None

            # 接近分析的距離閾值（標準化）
            self.proximity_threshold = 0.2

            logger.info("SpatialAnalyzer initialized successfully with all sub-components")

        except Exception as e:
            logger.error(f"Failed to initialize SpatialAnalyzer: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def update_class_names(self, class_names: Dict[int, str]):
        """
        更新類別名稱映射並傳遞給 ObjectExtractor

        Args:
            class_names: 新的類別名稱映射字典
        """
        try:
            self.class_names = class_names
            if hasattr(self, 'object_extractor') and self.object_extractor:
                self.object_extractor.update_class_names(class_names)
                logger.info(f"Updated class names in SpatialAnalyzer and ObjectExtractor")
        except Exception as e:
            logger.error(f"Failed to update class names in SpatialAnalyzer: {str(e)}")

    def _determine_region(self, x: float, y: float) -> str:
        """
        判斷點位於哪個區域

        Args:
            x: 標準化x座標 (0-1)
            y: 標準化y座標 (0-1)

        Returns:
            區域名稱
        """
        try:
            return self.region_analyzer.determine_region(x, y)
        except Exception as e:
            logger.error(f"Error in _determine_region: {str(e)}")
            logger.error(traceback.format_exc())
            return "unknown"

    def _analyze_regions(self, detected_objects: List[Dict]) -> Dict:
        """
        分析物件在各區域的分布情況

        Args:
            detected_objects: 包含位置資訊的檢測物件列表

        Returns:
            包含區域分析結果的字典
        """
        try:
            return self.region_analyzer.analyze_regions(detected_objects)
        except Exception as e:
            logger.error(f"Error in _analyze_regions: {str(e)}")
            logger.error(traceback.format_exc())
            return {
                "counts": {},
                "main_focus": [],
                "objects_by_region": {}
            }

    def _extract_detected_objects(self, detection_result: Any, confidence_threshold: float = 0.25) -> List[Dict]:
        """
        從檢測結果中提取物件資訊，包含位置資訊

        Args:
            detection_result: YOLOv8檢測結果
            confidence_threshold: 最小信心度閾值

        Returns:
            包含檢測物件資訊的字典列表
        """
        try:
            return self.object_extractor.extract_detected_objects(
                detection_result,
                confidence_threshold,
                region_analyzer=self.region_analyzer
            )
        except Exception as e:
            logger.error(f"Error in _extract_detected_objects: {str(e)}")
            logger.error(traceback.format_exc())
            return []

    def _detect_scene_viewpoint(self, detected_objects: List[Dict]) -> Dict:
        """
        檢測場景視角並識別特殊場景模式

        Args:
            detected_objects: 檢測到的物件列表

        Returns:
            包含視角和場景模式資訊的字典
        """
        try:
            # 委託給新的場景視角分析器
            return self.scene_viewpoint_analyzer.detect_scene_viewpoint(detected_objects)
        except Exception as e:
            logger.error(f"Error in _detect_scene_viewpoint: {str(e)}")
            logger.error(traceback.format_exc())
            return {"viewpoint": "eye_level", "patterns": []}

    def _identify_functional_zones(self, detected_objects: List[Dict], scene_type: str) -> Dict:
        """
        識別場景內的功能區域，具有針對不同視角和文化背景的改進檢測能力

        Args:
            detected_objects: 檢測到的物件列表
            scene_type: 識別出的場景類型

        Returns:
            包含功能區域及其描述的字典
        """
        try:
            return self.functional_zone_identifier.identify_functional_zones(detected_objects, scene_type)
        except Exception as e:
            logger.error(f"Error in _identify_functional_zones: {str(e)}")
            logger.error(traceback.format_exc())
            return {}

    def _get_object_categories(self, detected_objects: List[Dict]) -> set:
        """
        從檢測到的物件中獲取唯一的物件類別

        Args:
            detected_objects: 檢測到的物件列表

        Returns:
            唯一物件類別的集合
        """
        try:
            return self.object_extractor.get_object_categories(detected_objects)
        except Exception as e:
            logger.error(f"Error in _get_object_categories: {str(e)}")
            logger.error(traceback.format_exc())
            return set()

    def _identify_core_objects_for_scene(self, detected_objects: List[Dict], scene_type: str) -> List[Dict]:
        """
        識別定義特定場景類型的核心物件

        Args:
            detected_objects: 檢測到的物件列表
            scene_type: 場景類型

        Returns:
            場景的核心物件列表
        """
        try:
            return self.object_extractor.identify_core_objects_for_scene(detected_objects, scene_type)
        except Exception as e:
            logger.error(f"Error in _identify_core_objects_for_scene: {str(e)}")
            logger.error(traceback.format_exc())
            return []

    def _evaluate_zone_identification_feasibility(self, detected_objects: List[Dict], scene_type: str) -> bool:
        """
        基於物件關聯性和分布特徵的彈性可行性評估

        Args:
            detected_objects: 檢測到的物件列表
            scene_type: 場景類型

        Returns:
            是否適合進行區域識別
        """
        try:
            return self.zone_evaluator.evaluate_zone_identification_feasibility(detected_objects, scene_type)
        except Exception as e:
            logger.error(f"Error in _evaluate_zone_identification_feasibility: {str(e)}")
            logger.error(traceback.format_exc())
            return False

    def _calculate_functional_relationships(self, detected_objects: List[Dict]) -> float:
        """
        計算物件間的功能關聯性評分

        Args:
            detected_objects: 檢測到的物件列表

        Returns:
            功能關聯性評分 (0.0-1.0)
        """
        try:
            return self.zone_evaluator.calculate_functional_relationships(detected_objects)
        except Exception as e:
            logger.error(f"Error in _calculate_functional_relationships: {str(e)}")
            logger.error(traceback.format_exc())
            return 0.0

    def _calculate_spatial_diversity(self, detected_objects: List[Dict]) -> float:
        """
        計算物件空間分布的多樣性

        Args:
            detected_objects: 檢測到的物件列表

        Returns:
            空間多樣性評分 (0.0-1.0)
        """
        try:
            return self.zone_evaluator.calculate_spatial_diversity(detected_objects)
        except Exception as e:
            logger.error(f"Error in _calculate_spatial_diversity: {str(e)}")
            logger.error(traceback.format_exc())
            return 0.0

    def _get_complexity_threshold(self, scene_type: str) -> float:
        """
        根據場景類型返回適當的複雜度閾值

        Args:
            scene_type: 場景類型

        Returns:
            複雜度閾值 (0.0-1.0)
        """
        try:
            return self.zone_evaluator.get_complexity_threshold(scene_type)
        except Exception as e:
            logger.error(f"Error in _get_complexity_threshold: {str(e)}")
            logger.error(traceback.format_exc())
            return 0.55

    def _create_distribution_map(self, detected_objects: List[Dict]) -> Dict:
        """
        創建物件在各區域分布的詳細地圖，用於空間分析

        Args:
            detected_objects: 檢測到的物件列表

        Returns:
            包含各區域分布詳情的字典
        """
        try:
            return self.region_analyzer.create_distribution_map(detected_objects)
        except Exception as e:
            logger.error(f"Error in _create_distribution_map: {str(e)}")
            logger.error(traceback.format_exc())
            return {}

    def _find_main_region(self, region_objects_dict: Dict) -> str:
        """
        找到物件最多的主要區域

        Args:
            region_objects_dict: 區域物件字典

        Returns:
            主要區域名稱
        """
        try:
            if not region_objects_dict:
                return "unknown"

            return max(region_objects_dict.items(),
                    key=lambda x: len(x[1]),
                    default=("unknown", []))[0]
        except Exception as e:
            logger.error(f"Error in _find_main_region: {str(e)}")
            logger.error(traceback.format_exc())
            return "unknown"

    def _detect_cross_pattern(self, positions):
        """檢測位置中的十字交叉模式 - 委託給SceneViewpointAnalyzer"""
        try:
            return self.scene_viewpoint_analyzer._detect_cross_pattern(positions)
        except Exception as e:
            logger.error(f"Error in _detect_cross_pattern: {str(e)}")
            return False

    def _analyze_movement_directions(self, positions):
        """分析位置中的移動方向 - 委託給SceneViewpointAnalyzer"""
        try:
            return self.scene_viewpoint_analyzer._analyze_movement_directions(positions)
        except Exception as e:
            logger.error(f"Error in _analyze_movement_directions: {str(e)}")
            return []

    def _get_directional_description(self, region: str) -> str:
        """將區域名稱轉換為方位描述 - 委託給RegionAnalyzer"""
        try:
            return self.region_analyzer.get_directional_description(region)
        except Exception as e:
            logger.error(f"Error in _get_directional_description: {str(e)}")
            return "central"

    @property
    def regions(self):
        """提供對區域定義的向後兼容訪問"""
        return self.region_analyzer.regions
