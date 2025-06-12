import logging
import traceback
import numpy as np
from typing import Dict, List, Optional, Any, Tuple

class SpatialLocationHandler:
    """
    空間位置處理器 - 專門處理空間描述生成和排列模式分析
    負責生成物件的空間位置描述、分析排列模式以及與 RegionAnalyzer 的整合
    """

    def __init__(self, region_analyzer: Optional[Any] = None):
        """
        初始化空間位置處理器

        Args:
            region_analyzer: RegionAnalyzer實例
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.region_analyzer = region_analyzer

    def set_region_analyzer(self, region_analyzer: Any) -> None:
        """
        設置RegionAnalyzer，用於標準化空間描述生成

        Args:
            region_analyzer: RegionAnalyzer實例
        """
        try:
            self.region_analyzer = region_analyzer
            self.logger.info("RegionAnalyzer instance set for SpatialLocationHandler")
        except Exception as e:
            self.logger.warning(f"Error setting RegionAnalyzer: {str(e)}")

    def generate_spatial_description(self, obj: Dict, image_width: Optional[int] = None,
                                   image_height: Optional[int] = None,
                                   region_analyzer: Optional[Any] = None) -> str:
        """
        為物件生成空間位置描述

        Args:
            obj: 物件字典
            image_width: 可選的圖像寬度
            image_height: 可選的圖像高度
            region_analyzer: 可選的RegionAnalyzer實例，用於生成標準化描述

        Returns:
            str: 空間描述字符串，空值region時返回空字串
        """
        try:
            region = obj.get("region") or ""
            object_type = obj.get("class_name", "")

            # 處理空值或無效region，直接返回空字串避免不完整描述
            if not region.strip() or region == "unknown":
                # 根據物件類型提供合適的預設位置描述
                if object_type and any(vehicle in object_type.lower() for vehicle in ["car", "truck", "bus"]):
                    return "positioned in the scene"
                elif object_type and "person" in object_type.lower():
                    return "present in the area"
                else:
                    return "located in the scene"

            # 如果提供了RegionAnalyzer實例，使用其標準化方法
            if region_analyzer and hasattr(region_analyzer, 'get_spatial_description_phrase'):
                if hasattr(region_analyzer, 'get_contextual_spatial_description'):
                    spatial_desc = region_analyzer.get_contextual_spatial_description(region, object_type)
                else:
                    spatial_desc = region_analyzer.get_spatial_description_phrase(region)

                if spatial_desc:
                    return spatial_desc

            # 備用邏輯：使用改進的內建映射
            clean_region = region.replace('_', ' ').strip().lower()

            region_map = {
                "top left": "in the upper left area",
                "top center": "in the upper area",
                "top right": "in the upper right area",
                "middle left": "on the left side",
                "middle center": "in the center",
                "center": "in the center",
                "middle right": "on the right side",
                "bottom left": "in the lower left area",
                "bottom center": "in the lower area",
                "bottom right": "in the lower right area"
            }

            # 直接映射匹配
            if clean_region in region_map:
                return region_map[clean_region]

            # 比較模糊籠統的方位匹配
            if "top" in clean_region and "left" in clean_region:
                return "in the upper left area"
            elif "top" in clean_region and "right" in clean_region:
                return "in the upper right area"
            elif "bottom" in clean_region and "left" in clean_region:
                return "in the lower left area"
            elif "bottom" in clean_region and "right" in clean_region:
                return "in the lower right area"
            elif "top" in clean_region:
                return "in the upper area"
            elif "bottom" in clean_region:
                return "in the lower area"
            elif "left" in clean_region:
                return "on the left side"
            elif "right" in clean_region:
                return "on the right side"
            elif "center" in clean_region or "middle" in clean_region:
                return "in the center"

            # 如果region無法辨識，使用normalized_center作為備用
            norm_center = obj.get("normalized_center")
            if norm_center and image_width and image_height:
                x_norm, y_norm = norm_center
                h_pos = "left" if x_norm < 0.4 else "right" if x_norm > 0.6 else "center"
                v_pos = "upper" if y_norm < 0.4 else "lower" if y_norm > 0.6 else "center"

                if h_pos == "center" and v_pos == "center":
                    return "in the center"
                return f"in the {v_pos} {h_pos} area"

            # 如果所有方法都失敗，返回空字串
            return ""

        except Exception as e:
            self.logger.warning(f"Error generating spatial description: {str(e)}")
            return ""

    def get_standardized_spatial_description(self, obj: Dict) -> str:
        """
        使用RegionAnalyzer生成標準化空間描述的內部方法

        Args:
            obj: 物件字典

        Returns:
            str: 標準化空間描述，失敗時返回空字串
        """
        try:
            if hasattr(self, 'region_analyzer') and self.region_analyzer:
                region = obj.get("region", "")
                object_type = obj.get("class_name", "")

                if hasattr(self.region_analyzer, 'get_contextual_spatial_description'):
                    return self.region_analyzer.get_contextual_spatial_description(region, object_type)
                elif hasattr(self.region_analyzer, 'get_spatial_description_phrase'):
                    return self.region_analyzer.get_spatial_description_phrase(region)

            return ""

        except Exception as e:
            self.logger.warning(f"Error getting standardized spatial description: {str(e)}")
            object_type = obj.get("class_name", "")
            if object_type:
                return "visible in the scene"
            return "present in the view"

    def analyze_spatial_arrangement(self, class_name: str, scene_type: Optional[str],
                                  detected_objects: Optional[List[Dict]],
                                  count: int) -> Optional[str]:
        """
        分析物件的空間排列模式並生成相應描述

        Args:
            class_name: 物件類別名稱
            scene_type: 場景類型
            detected_objects: 該類型的所有檢測物件
            count: 物件數量

        Returns:
            Optional[str]: 空間排列描述，如果無法分析則返回None
        """
        if not detected_objects or len(detected_objects) < 2:
            return None

        try:
            # 提取物件的標準化位置
            positions = []
            for obj in detected_objects:
                center = obj.get("normalized_center", [0.5, 0.5])
                if isinstance(center, (list, tuple)) and len(center) >= 2:
                    positions.append(center)

            if len(positions) < 2:
                return None

            # 分析排列模式
            arrangement_pattern = self._analyze_arrangement_pattern(positions)

            # 根據物件類型和場景生成描述
            return self._generate_arrangement_description(class_name, scene_type,
                                                        arrangement_pattern, count)

        except Exception as e:
            self.logger.warning(f"Error analyzing spatial arrangement: {str(e)}")
            return None

    def _analyze_arrangement_pattern(self, positions: List[List[float]]) -> str:
        """
        分析位置點的排列模式

        Args:
            positions: 標準化的位置座標列表

        Returns:
            str: 排列模式類型（linear, clustered, scattered, circular等）
        """
        if len(positions) < 2:
            return "single"

        # 轉換為numpy陣列便於計算
        pos_array = np.array(positions)

        # 計算位置的分布特徵
        x_coords = pos_array[:, 0]
        y_coords = pos_array[:, 1]

        # 分析x和y方向的變異程度
        x_variance = np.var(x_coords)
        y_variance = np.var(y_coords)

        # 計算物件間的平均距離
        distances = []
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                dist = np.sqrt((positions[i][0] - positions[j][0])**2 +
                            (positions[i][1] - positions[j][1])**2)
                distances.append(dist)

        avg_distance = np.mean(distances) if distances else 0
        distance_variance = np.var(distances) if distances else 0

        # 判斷排列模式
        if len(positions) >= 4 and self._is_circular_pattern(positions):
            return "circular"
        elif x_variance < 0.05 or y_variance < 0.05:  # 一個方向變異很小
            return "linear"
        elif avg_distance < 0.3 and distance_variance < 0.02:  # 物件聚集且距離相近
            return "clustered"
        elif avg_distance > 0.6:  # 物件分散
            return "scattered"
        elif distance_variance < 0.03:  # 距離一致，可能是規則排列
            return "regular"
        else:
            return "distributed"

    def _is_circular_pattern(self, positions: List[List[float]]) -> bool:
        """
        檢查位置是否形成圓形或環形排列

        Args:
            positions: 位置座標列表

        Returns:
            bool: 是否為圓形排列
        """
        if len(positions) < 4:
            return False

        try:
            pos_array = np.array(positions)

            # 計算中心點
            center_x = np.mean(pos_array[:, 0])
            center_y = np.mean(pos_array[:, 1])

            # 計算每個點到中心的距離
            distances_to_center = []
            for pos in positions:
                dist = np.sqrt((pos[0] - center_x)**2 + (pos[1] - center_y)**2)
                distances_to_center.append(dist)

            # 如果所有距離都相近，可能是圓形排列
            distance_variance = np.var(distances_to_center)
            return distance_variance < 0.05 and np.mean(distances_to_center) > 0.2

        except:
            return False

    def _generate_arrangement_description(self, class_name: str, scene_type: Optional[str],
                                        arrangement_pattern: str, count: int) -> Optional[str]:
        """
        根據物件類型、場景和排列模式生成空間描述

        Args:
            class_name: 物件類別名稱
            scene_type: 場景類型
            arrangement_pattern: 排列模式
            count: 物件數量

        Returns:
            Optional[str]: 生成的空間排列描述
        """
        # 基於物件類型的描述模板
        arrangement_templates = {
            "chair": {
                "linear": "arranged in a row",
                "clustered": "grouped together for conversation",
                "circular": "arranged around the table",
                "scattered": "positioned throughout the space",
                "regular": "evenly spaced",
                "distributed": "thoughtfully positioned"
            },
            "dining table": {
                "linear": "aligned to create a unified dining space",
                "clustered": "grouped to form intimate dining areas",
                "scattered": "distributed to optimize space flow",
                "regular": "systematically positioned",
                "distributed": "strategically placed"
            },
            "car": {
                "linear": "parked in sequence",
                "clustered": "grouped in the parking area",
                "scattered": "distributed throughout the lot",
                "regular": "neatly parked",
                "distributed": "positioned across the area"
            },
            "person": {
                "linear": "moving in a line",
                "clustered": "gathered together",
                "circular": "forming a circle",
                "scattered": "spread across the area",
                "distributed": "positioned throughout the scene"
            }
        }

        # 獲取對應的描述模板
        if class_name in arrangement_templates:
            template_dict = arrangement_templates[class_name]
            base_description = template_dict.get(arrangement_pattern, "positioned in the scene")
        else:
            # 通用的排列描述
            generic_templates = {
                "linear": "arranged in a line",
                "clustered": "grouped together",
                "circular": "arranged in a circular pattern",
                "scattered": "distributed across the space",
                "regular": "evenly positioned",
                "distributed": "thoughtfully placed"
            }
            base_description = generic_templates.get(arrangement_pattern, "positioned in the scene")

        return base_description
