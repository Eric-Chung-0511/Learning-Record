
import logging
import traceback
from typing import Dict, Any, Optional

class ConfidenceManager:
    """
    專門管理信心度相關邏輯，包括動態閾值調整、信心度乘數管理和地標類型特定的閾值處理
    """

    def __init__(self):
        """
        初始化置信度管理器
        """
        self.logger = logging.getLogger(__name__)

        # 初始化批處理參數
        self.batch_size = 16  # 默認批處理大小

        # 置信度閾值乘數配置
        self.confidence_threshold_multipliers = {
            "close_up": 0.9,     # 近景標準閾值
            "partial": 0.6,      # 部分可見降低閾值要求
            "distant": 0.5,      # 遠景更低閾值要求
            "full_image": 0.7    # 整張圖像需要更高閾值
        }

        # 地標類型閾值配置
        self.landmark_type_thresholds = {
            "tower": 0.5,         # 塔型建築需要更高閾值
            "skyscraper": 0.4,    # 摩天大樓使用較低閾值
            "building": 0.55,     # 一般的建築物閾值略微降低
            "monument": 0.5,      # 紀念碑閾值
            "natural": 0.6        # 自然景觀可以使用較低閾值
        }

    def set_batch_size(self, batch_size: int):
        """
        設置批處理大小

        Args:
            batch_size: 新的批處理大小
        """
        self.batch_size = max(1, batch_size)
        self.logger.info(f"Batch size set to {self.batch_size}")

    def adjust_confidence_threshold(self, detection_type: str, multiplier: float):
        """
        調整特定檢測類型的信心度的threshold

        Args:
            detection_type: 檢測類型 ('close_up', 'partial', 'distant', 'full_image')
            multiplier: 置信度閾值乘數
        """
        if detection_type in self.confidence_threshold_multipliers:
            self.confidence_threshold_multipliers[detection_type] = max(0.1, min(1.5, multiplier))
            self.logger.info(f"Adjusted confidence threshold multiplier for {detection_type} to {multiplier}")
        else:
            self.logger.warning(f"Unknown detection type: {detection_type}")

    def get_detection_type_multiplier(self, detection_type: str) -> float:
        """
        獲取檢測類型的置信度乘數

        Args:
            detection_type: 檢測類型

        Returns:
            float: 置信度乘數
        """
        return self.confidence_threshold_multipliers.get(detection_type, 1.0)

    def get_landmark_type_threshold(self, landmark_type: str) -> float:
        """
        獲取地標類型的閾值

        Args:
            landmark_type: 地標類型

        Returns:
            float: 地標類型閾值
        """
        return self.landmark_type_thresholds.get(landmark_type, 0.5)

    def calculate_adjusted_threshold(self, base_threshold: float, detection_type: str) -> float:
        """
        根據檢測類型計算調整後的閾值

        Args:
            base_threshold: 基礎閾值
            detection_type: 檢測type

        Returns:
            float: 調整後的閾值
        """
        try:
            base_multiplier = self.get_detection_type_multiplier(detection_type)
            adjusted_threshold = base_threshold * base_multiplier
            return adjusted_threshold
        except Exception as e:
            self.logger.error(f"Error calculating adjusted threshold: {e}")
            self.logger.error(traceback.format_exc())
            return base_threshold

    def calculate_final_threshold(self, base_threshold: float, detection_type: str,
                                landmark_type: str) -> float:
        """
        計算最終閾值，結合檢測類型和地標類型

        Args:
            base_threshold: 基礎閾值
            detection_type: 檢測type
            landmark_type: 地標type

        Returns:
            float: 最終閾值
        """
        try:
            # 根據檢測類型調整
            adjusted_threshold = self.calculate_adjusted_threshold(base_threshold, detection_type)

            # 根據地標類型進一步調整
            if landmark_type == "distinctive":
                # 特殊建築的閾值降低25%
                type_multiplier = 0.75
            else:
                # 使用已有的類型閾值
                type_multiplier = self.get_landmark_type_threshold(landmark_type) / 0.5

            final_threshold = adjusted_threshold * type_multiplier
            return final_threshold

        except Exception as e:
            self.logger.error(f"Error calculating final threshold: {e}")
            self.logger.error(traceback.format_exc())
            return base_threshold

    def evaluate_confidence(self, confidence: float, threshold: float) -> bool:
        """
        評估置信度是否達到閾值

        Args:
            confidence: 信心度score
            threshold: 閾值

        Returns:
            bool: 是否達到閾值
        """
        return confidence >= threshold

    def apply_architectural_boost(self, confidence: float, architectural_analysis: Dict[str, Any],
                                landmark_id: str) -> float:
        """
        根據建築特徵分析調整信心度

        Args:
            confidence: 原始置信度
            architectural_analysis: 建築特徵分析結果
            landmark_id: 地標ID

        Returns:
            float: 調整後的信心度
        """
        try:
            confidence_boost = 0
            landmark_id_lower = landmark_id.lower()

            top_features = architectural_analysis.get("architectural_features", [])
            primary_category = architectural_analysis.get("primary_category", "")

            # 使用主要建築類別來調整置信度，使用通用條件而非特定地標名稱
            if primary_category == "tower" and any(term in landmark_id_lower for term in ["tower", "spire", "needle"]):
                confidence_boost += 0.05
            elif primary_category == "skyscraper" and any(term in landmark_id_lower for term in ["building", "skyscraper", "tall"]):
                confidence_boost += 0.05
            elif primary_category == "historical" and any(term in landmark_id_lower for term in ["monument", "castle", "palace", "temple"]):
                confidence_boost += 0.05
            elif primary_category == "distinctive" and any(term in landmark_id_lower for term in ["unusual", "unique", "special", "famous"]):
                confidence_boost += 0.05

            # 根據特定特徵進一步微調，使用通用特徵描述而非特定地標
            for feature, score in top_features:
                if feature == "time_display" and "clock" in landmark_id_lower:
                    confidence_boost += 0.03
                elif feature == "segmented_exterior" and "segmented" in landmark_id_lower:
                    confidence_boost += 0.03
                elif feature == "slanted_design" and "leaning" in landmark_id_lower:
                    confidence_boost += 0.03

            # 應用信心度調整
            if confidence_boost > 0:
                adjusted_confidence = confidence + confidence_boost
                self.logger.info(f"Boosted confidence by {confidence_boost:.2f} based on architectural features ({primary_category})")
                return adjusted_confidence

            return confidence

        except Exception as e:
            self.logger.error(f"Error applying architectural boost: {e}")
            self.logger.error(traceback.format_exc())
            return confidence

    def determine_detection_type_from_region(self, region_width: int, region_height: int,
                                           image_width: int, image_height: int) -> str:
        """
        根據區域大小自動判斷檢測類型

        Args:
            region_width: 區域寬度
            region_height: 區域高度
            image_width: 圖像寬度
            image_height: 圖像高度

        Returns:
            str: 檢測類型
        """
        try:
            region_area_ratio = (region_width * region_height) / (image_width * image_height)

            if region_area_ratio > 0.5:
                return "close_up"
            elif region_area_ratio > 0.2:
                return "partial"
            else:
                return "distant"

        except Exception as e:
            self.logger.error(f"Error determining detection type from region: {e}")
            self.logger.error(traceback.format_exc())
            return "partial"

    def adjust_detection_type_by_viewpoint(self, detection_type: str, dominant_viewpoint: str) -> str:
        """
        根據視角調整檢測類型

        Args:
            detection_type: 原始檢測類型
            dominant_viewpoint: 主要視角

        Returns:
            str: 調整後的檢測類型
        """
        try:
            if dominant_viewpoint == "close_up" and detection_type != "close_up":
                return "close_up"
            elif dominant_viewpoint == "distant" and detection_type != "distant":
                return "distant"
            elif dominant_viewpoint == "angled_view":
                return "partial"  # 角度視圖可能是部分可見
            else:
                return detection_type

        except Exception as e:
            self.logger.error(f"Error adjusting detection type by viewpoint: {e}")
            self.logger.error(traceback.format_exc())
            return detection_type

    def get_batch_size(self) -> int:
        """
        獲取當前批處理大小

        Returns:
            int: 批處理大小
        """
        return self.batch_size

    def get_all_threshold_multipliers(self) -> Dict[str, float]:
        """
        獲取所有置信度閾值乘數

        Returns:
            Dict[str, float]: 閾值乘數字典
        """
        return self.confidence_threshold_multipliers.copy()

    def get_all_landmark_type_thresholds(self) -> Dict[str, float]:
        """
        獲取所有地標類型閾值

        Returns:
            Dict[str, float]: 地標類型閾值字典
        """
        return self.landmark_type_thresholds.copy()
