import logging
import traceback
from typing import Dict, List, Tuple, Optional
import numpy as np

class ViewpointDetectionError(Exception):
    """Custom exception for errors during viewpoint detection."""
    pass


class ViewpointDetector:
    """
    視角檢測器 - 分析物體分布模式以識別圖像視角類型

    此class負責通過分析檢測到的物體在圖像中的空間分布、大小變化和位置模式，
    來確定圖像的拍攝視角。特別針對行人密集的十字路口場景進行了優化。
    """

    def __init__(self,
                 aerial_threshold: float = 0.7,
                 aerial_size_variance_threshold: float = 0.15,
                 low_angle_threshold: float = 0.3,
                 vertical_size_ratio_threshold: float = 1.8,
                 elevated_threshold: float = 0.6,
                 elevated_top_threshold: float = 0.3,
                 crosswalk_position_tolerance: float = 0.1,
                 crosswalk_axis_tolerance: float = 0.15,
                 min_people_for_crosswalk: int = 8,
                 min_people_for_aerial: int = 10):
        """
        初始化視角檢測器

        Args:
            aerial_threshold: 空中視角檢測的物體密度閾值
            aerial_size_variance_threshold: 空中視角的大小變異閾值
            low_angle_threshold: 低角度視角的底部分布閾值
            vertical_size_ratio_threshold: 垂直大小比例閾值
            elevated_threshold: 高位視角的物體分布閾值
            elevated_top_threshold: 高位視角的頂部物體閾值
            crosswalk_position_tolerance: 十字路口位置容差
            crosswalk_axis_tolerance: 十字路口軸線容差
            min_people_for_crosswalk: 檢測十字路口所需的最少人數
            min_people_for_aerial: 檢測空中視角所需的最少人數
        """
        self.logger = logging.getLogger(self.__class__.__name__)

        # 視角檢測參數配置
        self.viewpoint_params = {
            "aerial_threshold": aerial_threshold,
            "aerial_size_variance_threshold": aerial_size_variance_threshold,
            "low_angle_threshold": low_angle_threshold,
            "vertical_size_ratio_threshold": vertical_size_ratio_threshold,
            "elevated_threshold": elevated_threshold,
            "elevated_top_threshold": elevated_top_threshold,
            "crosswalk_position_tolerance": crosswalk_position_tolerance,
            "crosswalk_axis_tolerance": crosswalk_axis_tolerance,
            "min_people_for_crosswalk": min_people_for_crosswalk,
            "min_people_for_aerial": min_people_for_aerial
        }

        self.logger.info("ViewpointDetector initialized with parameters: %s", self.viewpoint_params)

    def detect_viewpoint(self, detected_objects: List[Dict]) -> str:
        """
        檢測圖像視角類型

        Args:
            detected_objects: 檢測到的物體列表，每個物體應包含位置、大小等信息

        Returns:
            str: 檢測到的視角類型 ('aerial', 'low_angle', 'elevated', 'eye_level')
        """
        try:
            if not detected_objects:
                self.logger.warning("No detected objects provided for viewpoint detection")
                return "eye_level"

            self.logger.info(f"Starting viewpoint detection with {len(detected_objects)} objects")

            # 優先檢測十字路口模式（通常為空中視角）
            if self._detect_crosswalk_pattern(detected_objects):
                self.logger.info("Crosswalk pattern detected - returning aerial viewpoint")
                return "aerial"

            # 檢測基於行人分布的空中視角
            if self._detect_aerial_from_pedestrian_distribution(detected_objects):
                self.logger.info("Aerial viewpoint detected from pedestrian distribution")
                return "aerial"

            # 標準視角檢測流程
            return self._detect_standard_viewpoint(detected_objects)

        except Exception as e:
            error_msg = f"Error during viewpoint detection: {str(e)}"
            self.logger.error(f"{error_msg}\n{traceback.format_exc()}")
            return "eye_level"  # 返回默認值

    def _detect_crosswalk_pattern(self, detected_objects: List[Dict]) -> bool:
        """
        檢測十字路口/斑馬線模式

        Args:
            detected_objects: 檢測到的物體列表

        Returns:
            bool: 是否檢測到十字路口模式
        """
        try:
            people_objs = [obj for obj in detected_objects if obj.get("class_id") == 0]

            if len(people_objs) < self.viewpoint_params["min_people_for_crosswalk"]:
                return False

            # 提取行人位置
            people_positions = []
            for obj in people_objs:
                if "normalized_center" in obj:
                    people_positions.append(obj["normalized_center"])

            if len(people_positions) < 4:
                return False

            # 檢測十字形分布
            if self._detect_cross_pattern(people_positions):
                self.logger.debug("Cross pattern detected in pedestrian positions")
                return True

            # 檢測線性聚類分布
            if self._detect_linear_crosswalk_clusters(people_positions):
                self.logger.debug("Linear crosswalk clusters detected")
                return True

            return False

        except Exception as e:
            self.logger.warning(f"Error in crosswalk pattern detection: {str(e)}")
            return False

    def _detect_cross_pattern(self, positions: List[Tuple[float, float]]) -> bool:
        """
        檢測十字形分布模式

        Args:
            positions: 物體位置列表 [(x, y), ...]

        Returns:
            bool: 是否檢測到十字形模式
        """
        try:
            x_coords = [pos[0] for pos in positions]
            y_coords = [pos[1] for pos in positions]

            x_range = max(x_coords) - min(x_coords)
            y_range = max(y_coords) - min(y_coords)

            # 檢查 x 和 y 方向都有較大範圍且範圍相似
            if x_range <= 0.5 or y_range <= 0.5:
                return False

            if not (0.7 < (x_range / y_range) < 1.3):
                return False

            # 計算到中心點的距離並檢查軸線分布
            center_x = np.mean(x_coords)
            center_y = np.mean(y_coords)

            close_to_axis_count = 0
            axis_tolerance = self.viewpoint_params["crosswalk_axis_tolerance"]

            for x, y in positions:
                x_distance_to_center = abs(x - center_x)
                y_distance_to_center = abs(y - center_y)

                # 檢查是否接近水平或垂直軸線
                if x_distance_to_center < axis_tolerance or y_distance_to_center < axis_tolerance:
                    close_to_axis_count += 1

            # 如果足夠多的點接近軸線，認為是十字路口
            axis_ratio = close_to_axis_count / len(positions)
            return axis_ratio >= 0.6

        except Exception as e:
            self.logger.warning(f"Error detecting cross pattern: {str(e)}")
            return False

    def _detect_linear_crosswalk_clusters(self, positions: List[Tuple[float, float]]) -> bool:
        """
        檢測線性聚類分布（交叉的斑馬線）

        Args:
            positions: 物體位置列表

        Returns:
            bool: 是否檢測到線性交叉模式
        """
        try:
            x_coords = [pos[0] for pos in positions]
            y_coords = [pos[1] for pos in positions]

            # 檢測 x 和 y 方向的聚類
            x_clusters = self._detect_linear_clusters(x_coords)
            y_clusters = self._detect_linear_clusters(y_coords)

            # 如果在 x 和 y 方向上都有多個聚類，可能是交叉的斑馬線
            return len(x_clusters) >= 2 and len(y_clusters) >= 2

        except Exception as e:
            self.logger.warning(f"Error detecting linear crosswalk clusters: {str(e)}")
            return False

    def _detect_linear_clusters(self, coords: List[float], threshold: float = 0.05) -> List[List[float]]:
        """
        檢測坐標中的線性聚類

        Args:
            coords: 一維坐標列表
            threshold: 聚類閾值

        Returns:
            List[List[float]]: 聚類列表
        """
        if not coords:
            return []

        try:
            sorted_coords = sorted(coords)
            clusters = []
            current_cluster = [sorted_coords[0]]

            for i in range(1, len(sorted_coords)):
                if sorted_coords[i] - sorted_coords[i-1] < threshold:
                    current_cluster.append(sorted_coords[i])
                else:
                    if len(current_cluster) >= 2:
                        clusters.append(current_cluster)
                    current_cluster = [sorted_coords[i]]

            # 添加最後一個聚類
            if len(current_cluster) >= 2:
                clusters.append(current_cluster)

            return clusters

        except Exception as e:
            self.logger.warning(f"Error in linear cluster detection: {str(e)}")
            return []

    def _detect_aerial_from_pedestrian_distribution(self, detected_objects: List[Dict]) -> bool:
        """
        基於行人分布檢測空中視角

        Args:
            detected_objects: 檢測到的物體列表

        Returns:
            bool: 是否為空中視角
        """
        try:
            people_objs = [obj for obj in detected_objects if obj.get("class_id") == 0]

            if len(people_objs) < self.viewpoint_params["min_people_for_aerial"]:
                return False

            # 統計不同區域的行人數量
            people_region_counts = {}
            for obj in people_objs:
                region = obj.get("region", "unknown")
                people_region_counts[region] = people_region_counts.get(region, 0) + 1

            # 檢查行人是否分布在多個區域
            regions_with_multiple_people = sum(1 for count in people_region_counts.values() if count >= 2)

            if regions_with_multiple_people < 4:
                return False

            # 檢查行人分布的均勻性
            region_counts = list(people_region_counts.values())
            if not region_counts:
                return False

            region_counts_variance = np.var(region_counts)
            region_counts_mean = np.mean(region_counts)

            if region_counts_mean > 0:
                variation_coefficient = region_counts_variance / region_counts_mean
                return variation_coefficient < 0.5

            return False

        except Exception as e:
            self.logger.warning(f"Error in aerial detection from pedestrian distribution: {str(e)}")
            return False

    def _detect_standard_viewpoint(self, detected_objects: List[Dict]) -> str:
        """
        標準視角檢測流程

        Args:
            detected_objects: 檢測到的物體列表

        Returns:
            str: 檢測到的視角類型
        """
        try:
            # 計算基本統計指標
            metrics = self._calculate_viewpoint_metrics(detected_objects)

            # 基於計算的指標判斷視角類型
            if self._is_aerial_viewpoint(metrics):
                return "aerial"
            elif self._is_low_angle_viewpoint(metrics):
                return "low_angle"
            elif self._is_elevated_viewpoint(metrics):
                return "elevated"
            else:
                return "eye_level"

        except Exception as e:
            self.logger.warning(f"Error in standard viewpoint detection: {str(e)}")
            return "eye_level"

    def _calculate_viewpoint_metrics(self, detected_objects: List[Dict]) -> Dict:
        """
        計算視角檢測所需的各項指標

        Args:
            detected_objects: 檢測到的物體列表

        Returns:
            Dict: 包含各項指標的字典
        """
        total_objects = len(detected_objects)
        top_region_count = 0
        bottom_region_count = 0
        sizes = []
        height_width_ratios = []

        try:
            for obj in detected_objects:
                # 統計頂部和底部區域的物體數量
                region = obj.get("region", "")
                if "top" in region:
                    top_region_count += 1
                elif "bottom" in region:
                    bottom_region_count += 1

                # 收集大小信息
                if "normalized_area" in obj:
                    sizes.append(obj["normalized_area"])

                # 計算高寬比
                if "normalized_size" in obj:
                    width, height = obj["normalized_size"]
                    if width > 0:
                        height_width_ratios.append(height / width)

            # 計算比例
            top_ratio = top_region_count / total_objects if total_objects > 0 else 0
            bottom_ratio = bottom_region_count / total_objects if total_objects > 0 else 0

            # 計算大小變異係數
            size_variance_coefficient = 0
            if sizes and len(sizes) > 1:
                mean_size = np.mean(sizes)
                if mean_size > 0:
                    size_variance = np.var(sizes)
                    size_variance_coefficient = size_variance / (mean_size ** 2)

            # 計算平均高寬比
            avg_height_width_ratio = np.mean(height_width_ratios) if height_width_ratios else 1.0

            metrics = {
                "top_ratio": top_ratio,
                "bottom_ratio": bottom_ratio,
                "size_variance_coefficient": size_variance_coefficient,
                "avg_height_width_ratio": avg_height_width_ratio,
                "total_objects": total_objects
            }

            self.logger.debug(f"Calculated viewpoint metrics: {metrics}")
            return metrics

        except Exception as e:
            self.logger.error(f"Error calculating viewpoint metrics: {str(e)}")
            return {
                "top_ratio": 0,
                "bottom_ratio": 0,
                "size_variance_coefficient": 0,
                "avg_height_width_ratio": 1.0,
                "total_objects": total_objects
            }

    def _is_aerial_viewpoint(self, metrics: Dict) -> bool:
        """判斷是否為空中視角"""
        return (metrics["size_variance_coefficient"] < self.viewpoint_params["aerial_size_variance_threshold"] and
                metrics["bottom_ratio"] < 0.3 and
                metrics["top_ratio"] > self.viewpoint_params["aerial_threshold"])

    def _is_low_angle_viewpoint(self, metrics: Dict) -> bool:
        """判斷是否為低角度視角"""
        return (metrics["avg_height_width_ratio"] > self.viewpoint_params["vertical_size_ratio_threshold"] and
                metrics["top_ratio"] > self.viewpoint_params["low_angle_threshold"])

    def _is_elevated_viewpoint(self, metrics: Dict) -> bool:
        """判斷是否為高位視角"""
        return (metrics["bottom_ratio"] > self.viewpoint_params["elevated_threshold"] and
                metrics["top_ratio"] < self.viewpoint_params["elevated_top_threshold"])

    def get_viewpoint_confidence(self, detected_objects: List[Dict]) -> Tuple[str, float]:
        """
        獲取視角檢測結果及其信心度

        Args:
            detected_objects: 檢測到的物體列表

        Returns:
            Tuple[str, float]: (視角類型, 信心度)
        """
        try:
            viewpoint = self.detect_viewpoint(detected_objects)

            # 基於檢測條件計算信心度
            if viewpoint == "aerial" and self._detect_crosswalk_pattern(detected_objects):
                confidence = 0.95  # 十字路口模式有很高信心度
            elif viewpoint == "aerial":
                confidence = 0.8
            elif viewpoint == "eye_level":
                confidence = 0.7  # 默認視角信心度較低
            else:
                confidence = 0.85

            self.logger.info(f"Viewpoint detection result: {viewpoint} (confidence: {confidence:.2f})")
            return viewpoint, confidence

        except Exception as e:
            self.logger.warning("Using fallback viewpoint due to detection error")
            return "eye_level", 0.3
