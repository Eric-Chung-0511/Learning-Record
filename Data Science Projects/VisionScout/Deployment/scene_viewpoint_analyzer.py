
import logging
import traceback
import numpy as np
from typing import Dict, List, Any, Optional, Tuple

logger = logging.getLogger(__name__)

class SceneViewpointAnalyzer:
    """
    負責場景視角檢測和模式識別
    專注於檢測場景視角（俯視、平視等）並識別特殊場景模式（如十字路口、人流方向等）
    提供詳細的場景空間分析和視角相關的場景理解功能
    """

    def __init__(self, enhanced_scene_describer=None):
        """
        初始化場景視角分析器

        Args:
            enhanced_scene_describer: 增強場景描述器實例，用於基本視角檢測
        """
        try:
            self.enhanced_scene_describer = enhanced_scene_describer
            logger.info("SceneViewpointAnalyzer initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize SceneViewpointAnalyzer: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def detect_viewpoint(self, detected_objects: List[Dict]) -> str:
        """
        檢測圖像視角類型

        Args:
            detected_objects: 檢測到的物件列表

        Returns:
            str: 檢測到的視角類型
        """
        try:
            # 使用內部的場景視角檢測方法
            viewpoint_info = self.detect_scene_viewpoint(detected_objects)
            return viewpoint_info.get("viewpoint", "eye_level")
        except Exception as e:
            logger.warning(f"Error detecting viewpoint: {str(e)}")
            return "eye_level"

    def get_viewpoint_confidence(self, detected_objects: List[Dict]) -> Tuple[str, float]:
        """
        獲取視角檢測結果及其信心度

        Args:
            detected_objects: 檢測到的物件列表

        Returns:
            Tuple[str, float]: (視角類型, 信心度)
        """
        try:
            viewpoint_info = self.detect_scene_viewpoint(detected_objects)
            viewpoint = viewpoint_info.get("viewpoint", "eye_level")

            # 根據檢測到的模式計算信心度
            patterns = viewpoint_info.get("patterns", [])
            confidence = 0.5  # 基礎信心度

            if "crosswalk_intersection" in patterns:
                confidence += 0.3
            if "consistent_object_size" in patterns:
                confidence += 0.2
            if "multi_directional_movement" in patterns:
                confidence += 0.1

            confidence = min(confidence, 1.0)
            return viewpoint, confidence

        except Exception as e:
            logger.error(f"Error getting viewpoint confidence: {str(e)}")
            return "eye_level", 0.5

    def detect_scene_viewpoint(self, detected_objects: List[Dict]) -> Dict:
        """
        檢測場景視角並識別特殊場景模式

        Args:
            detected_objects: 檢測到的物件列表

        Returns:
            包含視角和場景模式資訊的字典
        """
        try:
            if not detected_objects:
                logger.warning("No detected objects provided for viewpoint detection")
                return {"viewpoint": "eye_level", "patterns": []}

            # 從物件位置中提取資訊
            patterns = []

            # 檢測行人位置模式 - 篩選出所有行人物件
            pedestrian_objs = [obj for obj in detected_objects if obj.get("class_id") == 0]

            # 檢查是否有足夠的行人來識別模式 - 至少需要4個行人才能進行模式分析
            if len(pedestrian_objs) >= 4:
                # 提取行人的標準化中心座標用於模式分析
                pedestrian_positions = [obj["normalized_center"] for obj in pedestrian_objs]

                # 檢測十字交叉模式 - 這通常出現在斑馬線交叉口的俯視圖
                if self._detect_cross_pattern(pedestrian_positions):
                    patterns.append("crosswalk_intersection")

                # 檢測多方向行人流 - 分析行人是否在多個方向移動
                directions = self._analyze_movement_directions(pedestrian_positions)
                if len(directions) >= 2:
                    patterns.append("multi_directional_movement")

            # 檢查物件的大小一致性 - 在空中俯視圖中，物件大小通常更一致
            # 因為距離相對均勻，不像地面視角會有遠近差異
            if len(detected_objects) >= 5:
                sizes = [obj.get("normalized_area", 0) for obj in detected_objects]
                # 計算標準化變異數，避免受平均值影響
                size_variance = np.var(sizes) / (np.mean(sizes) ** 2) if np.mean(sizes) > 0 else 0

                # 低變異表示大小一致，可能是俯視角度
                if size_variance < 0.3:
                    patterns.append("consistent_object_size")

            # 基本視角檢測 - 使用增強場景描述器進行基礎視角判斷
            viewpoint = "eye_level"  # 預設值
            if self.enhanced_scene_describer and hasattr(self.enhanced_scene_describer, '_detect_viewpoint'):
                viewpoint = self.enhanced_scene_describer._detect_viewpoint(detected_objects)

            # 根據檢測到的模式增強視角判斷
            # 如果檢測到斑馬線交叉但視角判斷不是空中視角，優先採用模式判斷
            if "crosswalk_intersection" in patterns and viewpoint != "aerial":
                viewpoint = "aerial"

            result = {
                "viewpoint": viewpoint,
                "patterns": patterns
            }

            logger.info(f"Viewpoint detection completed: {viewpoint}, patterns: {patterns}")
            return result

        except Exception as e:
            logger.error(f"Error in scene viewpoint detection: {str(e)}")
            logger.error(traceback.format_exc())
            return {"viewpoint": "eye_level", "patterns": []}

    def _detect_cross_pattern(self, positions: List[List[float]]) -> bool:
        """
        檢測位置中的十字交叉模式
        這種模式通常出現在十字路口的俯視圖中，行人分布呈現十字形

        Args:
            positions: 位置列表 [[x1, y1], [x2, y2], ...]

        Returns:
            是否檢測到十字交叉模式
        """
        try:
            if len(positions) < 8:  # 需要足夠多的點才能形成有意義的十字模式
                return False

            # 提取 x 和 y 座標進行分析
            x_coords = [pos[0] for pos in positions]
            y_coords = [pos[1] for pos in positions]

            # 計算座標的平均值，用於確定中心線位置
            x_mean = np.mean(x_coords)
            y_mean = np.mean(y_coords)

            # 計算在中心線附近的點數量
            # 如果有足夠多的點在垂直和水平中心線附近，可能是十字交叉
            near_x_center = sum(1 for x in x_coords if abs(x - x_mean) < 0.1)  # 容忍10%的偏差
            near_y_center = sum(1 for y in y_coords if abs(y - y_mean) < 0.1)  # 容忍10%的偏差

            # 十字交叉模式的判斷條件：垂直和水平方向都有足夠的點聚集
            is_cross_pattern = near_x_center >= 3 and near_y_center >= 3

            if is_cross_pattern:
                logger.info(f"Cross pattern detected with {near_x_center} points near vertical center and {near_y_center} points near horizontal center")

            return is_cross_pattern

        except Exception as e:
            logger.error(f"Error detecting cross pattern: {str(e)}")
            logger.error(traceback.format_exc())
            return False

    def _analyze_movement_directions(self, positions: List[List[float]]) -> List[str]:
        """
        分析位置中的移動方向
        通過分析座標分布範圍來推斷主要的移動方向

        Args:
            positions: 位置列表 [[x1, y1], [x2, y2], ...]

        Returns:
            檢測到的主要方向列表
        """
        try:
            if len(positions) < 6:  # 需要足夠的點才能分析方向性
                return []

            # 提取 x 和 y 座標
            x_coords = [pos[0] for pos in positions]
            y_coords = [pos[1] for pos in positions]

            directions = []

            # 水平移動分析（左右移動）
            # 計算x座標的標準差和範圍來判斷水平方向的分散程度
            x_std = np.std(x_coords)
            x_range = max(x_coords) - min(x_coords)

            # 垂直移動分析（上下移動）
            # 計算y座標的標準差和範圍來判斷垂直方向的分散程度
            y_std = np.std(y_coords)
            y_range = max(y_coords) - min(y_coords)

            # 足夠大的範圍表示該方向有明顯的運動或分散
            # 40%的圖像範圍被認為是有意義的移動範圍
            if x_range > 0.4:
                directions.append("horizontal")
                logger.debug(f"Horizontal movement detected with range: {x_range:.3f}")

            if y_range > 0.4:
                directions.append("vertical")
                logger.debug(f"Vertical movement detected with range: {y_range:.3f}")

            logger.info(f"Movement directions analyzed: {directions}")
            return directions

        except Exception as e:
            logger.error(f"Error analyzing movement directions: {str(e)}")
            logger.error(traceback.format_exc())
            return []

    def detect_aerial_view_indicators(self, detected_objects: List[Dict]) -> Dict:
        """
        檢測俯視角度的指標
        分析物件分布特徵來判斷是否為俯視角度

        Args:
            detected_objects: 檢測到的物件列表

        Returns:
            包含俯視角度指標的字典
        """
        try:
            indicators = {
                "consistent_sizing": False,
                "grid_like_distribution": False,
                "high_object_density": False,
                "aerial_score": 0.0
            }

            if not detected_objects:
                return indicators

            # 檢查物件大小的一致性
            sizes = [obj.get("normalized_area", 0) for obj in detected_objects]
            if len(sizes) >= 3:
                size_variance = np.var(sizes) / (np.mean(sizes) ** 2) if np.mean(sizes) > 0 else 1
                # 俯視角度通常物件大小較為一致
                indicators["consistent_sizing"] = size_variance < 0.3

            # 檢查是否有網格狀分布（如停車場的俯視圖）
            positions = [obj.get("normalized_center", [0.5, 0.5]) for obj in detected_objects]
            if len(positions) >= 6:
                # 簡化的網格檢測：檢查是否有規律的行列分布
                x_coords = [pos[0] for pos in positions]
                y_coords = [pos[1] for pos in positions]

                # 計算座標的分布是否接近規律網格
                x_unique = len(set([round(x, 1) for x in x_coords]))  # 四捨五入到0.1精度
                y_unique = len(set([round(y, 1) for y in y_coords]))

                # 如果x和y方向都有多個不同的規律位置，可能是網格分布
                indicators["grid_like_distribution"] = x_unique >= 3 and y_unique >= 3

            # 檢查物件密度
            total_objects = len(detected_objects)
            # 俯視角度通常能看到更多物件
            indicators["high_object_density"] = total_objects >= 8

            # 計算俯視角度評分
            score = 0
            if indicators["consistent_sizing"]:
                score += 0.4
            if indicators["grid_like_distribution"]:
                score += 0.4
            if indicators["high_object_density"]:
                score += 0.2

            indicators["aerial_score"] = score

            logger.info(f"Aerial view indicators: score={score:.2f}, consistent_sizing={indicators['consistent_sizing']}, grid_distribution={indicators['grid_like_distribution']}, high_density={indicators['high_object_density']}")
            return indicators

        except Exception as e:
            logger.error(f"Error detecting aerial view indicators: {str(e)}")
            logger.error(traceback.format_exc())
            return {
                "consistent_sizing": False,
                "grid_like_distribution": False,
                "high_object_density": False,
                "aerial_score": 0.0
            }
