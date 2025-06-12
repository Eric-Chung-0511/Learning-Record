import logging
import numpy as np
from typing import Dict, List, Optional, Any

class ProminenceCalculator:
    """
    重要性計算器 - 專門處理物件重要性評估和篩選邏輯
    負責計算物件的重要性分數、類別重要性係數以及重要物件的篩選
    """

    def __init__(self, min_prominence_score: float = 0.1):
        """
        初始化重要性計算器

        Args:
            min_prominence_score: 物件顯著性的最低分數閾值
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.min_prominence_score = min_prominence_score

    def calculate_prominence_score(self, obj: Dict) -> float:
        """
        計算物件的重要性評分
        基本上權重設定為信心度 > 尺寸 > 空間 > 類別重要性

        Args:
            obj: 物件字典，包含檢測信息

        Returns:
            float: 重要性評分 (0.0-1.0)
        """
        try:
            # 基礎置信度評分 (權重: 40%)
            confidence = obj.get("confidence", 0.5)
            confidence_score = confidence * 0.4

            # 大小評分 (權重: 30%)
            normalized_area = obj.get("normalized_area", 0.1)
            # 使用對數縮放避免過大物件主導評分
            size_score = min(np.log(normalized_area * 10 + 1) / np.log(11), 1.0) * 0.3

            # 位置評分 (權重: 20%)
            # 中心區域的物件通常更重要
            center_x, center_y = obj.get("normalized_center", [0.5, 0.5])
            distance_from_center = np.sqrt((center_x - 0.5)**2 + (center_y - 0.5)**2)
            position_score = (1 - min(distance_from_center * 2, 1.0)) * 0.2

            # 類別重要性評分 (權重: 10%)
            class_importance = self.get_class_importance(obj.get("class_name", "unknown"))
            class_score = class_importance * 0.1

            total_score = confidence_score + size_score + position_score + class_score

            # 確保評分在有效範圍內
            return max(0.0, min(1.0, total_score))

        except Exception as e:
            self.logger.warning(f"Error calculating prominence score for object: {str(e)}")
            return 0.5  # 返回中等評分作為備用

    def get_class_importance(self, class_name: str) -> float:
        """
        根據物件類別返回重要性係數

        Args:
            class_name: 物件類別名稱

        Returns:
            float: 類別重要性係數 (0.0-1.0)
        """
        # 高重要性物件（人、車輛、建築）
        high_importance = ["person", "car", "truck", "bus", "motorcycle", "bicycle", "building"]

        # 中等重要性物件（家具、電器）
        medium_importance = ["chair", "couch", "tv", "laptop", "refrigerator", "dining table", "bed"]

        # 低重要性物件（小物品、配件）
        low_importance = ["handbag", "backpack", "umbrella", "cell phone", "remote", "mouse"]

        class_name_lower = class_name.lower()

        if any(item in class_name_lower for item in high_importance):
            return 1.0
        elif any(item in class_name_lower for item in medium_importance):
            return 0.7
        elif any(item in class_name_lower for item in low_importance):
            return 0.4
        else:
            return 0.6  # 預設中等重要性

    def filter_prominent_objects(self, detected_objects: List[Dict],
                                min_prominence_score: float = 0.5,
                                max_categories_to_return: Optional[int] = None) -> List[Dict]:
        """
        獲取最重要的物件，基於置信度、大小和位置計算重要性評分

        Args:
            detected_objects: 檢測到的物件列表
            min_prominence_score: 最小重要性分數閾值，範圍 0.0-1.0
            max_categories_to_return: 可選的最大返回類別數量限制

        Returns:
            List[Dict]: 按重要性排序的物件列表
        """
        try:
            if not detected_objects:
                return []

            prominent_objects = []

            for obj in detected_objects:
                # 計算重要性評分
                prominence_score = self.calculate_prominence_score(obj)

                # 只保留超過閾值的物件
                if prominence_score >= min_prominence_score:
                    obj_copy = obj.copy()
                    obj_copy['prominence_score'] = prominence_score
                    prominent_objects.append(obj_copy)

            # 按重要性評分排序（從高到低）
            prominent_objects.sort(key=lambda x: x.get('prominence_score', 0), reverse=True)

            # 如果指定了最大類別數量限制，進行過濾
            if max_categories_to_return is not None and max_categories_to_return > 0:
                categories_seen = set()
                filtered_objects = []

                for obj in prominent_objects:
                    class_name = obj.get("class_name", "unknown")

                    # 如果是新類別且未達到限制
                    if class_name not in categories_seen:
                        if len(categories_seen) < max_categories_to_return:
                            categories_seen.add(class_name)
                            filtered_objects.append(obj)
                    else:
                        # 已見過的類別，直接添加
                        filtered_objects.append(obj)

                return filtered_objects

            return prominent_objects

        except Exception as e:
            self.logger.error(f"Error calculating prominent objects: {str(e)}")
            return []
