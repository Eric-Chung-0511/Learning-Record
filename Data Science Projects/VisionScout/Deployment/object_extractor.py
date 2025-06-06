
import logging
import traceback
from typing import Dict, List, Any, Optional

# 設置日誌記錄器
logger = logging.getLogger(__name__)

class ObjectExtractor:
    """
    專門處理物件檢測結果的提取和預處理
    負責從YOLO檢測結果提取物件資訊、物件分類和核心物件的辨識
    """

    def __init__(self, class_names: Dict[int, str] = None, object_categories: Dict[str, List[int]] = None):
        """
        初始化物件提取器

        Args:
            class_names: 類別ID到類別名稱的映射字典
            object_categories: 物件類別分組字典
        """
        try:
            self.class_names = class_names or {}
            self.object_categories = object_categories or {}

            # 1. 讀取並設定基本信心度門檻（如果外部沒傳，就預設 0.25）
            self.base_conf_threshold = 0.25

            # 2. 動態信心度調整映射表 (key: 小寫 class_name, value: 調整係數)
            #    最終的門檻 = base_conf_threshold * factor
            #    如果某個 class_name 沒在這裡，就直接用 base_conf_threshold（相當於 factor=1.0）
            self.dynamic_conf_map = {
                "traffic light": 0.6,  # 0.25 * 0.6 = 0.15
                "car": 0.8,            # 0.25 * 0.8 = 0.20
                "person": 0.7,         # 0.25 * 0.7 = 0.175
                
            }

            logger.info(f"ObjectExtractor initialized with {len(self.class_names)} class names and {len(self.object_categories)} object categories")

        except Exception as e:
            logger.error(f"Failed to initialize ObjectExtractor: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def _get_dynamic_threshold(self, class_name: str) -> float:
        """
        根據 class_name 從 dynamic_conf_map 拿到 factor，計算最終的信心度門檻：
            threshold = base_conf_threshold * factor

        如果 class_name 不在映射表裡，就回傳 base_conf_threshold。
        """
        # 使用小寫做匹配，確保在 dynamic_conf_map 裡的 key 也都用小寫
        key = class_name.lower()
        factor = self.dynamic_conf_map.get(key, 1.0)
        return self.base_conf_threshold * factor

    def extract_detected_objects(
            self,
            detection_result: Any,
            confidence_threshold: float = 0.25,
            region_analyzer=None
        ) -> List[Dict]:
            """
            從檢測結果中提取物件資訊，包含位置資訊

            Args:
                detection_result: YOLO檢測結果
                confidence_threshold: 改由動態門檻決定
                region_analyzer: 區域分析器實例，用於判斷物件所屬區域

            Returns:
                包含檢測物件資訊的字典列表
            """
            try:
                # 調試信息：記錄當前類別映射狀態
                logger.info(f"ObjectExtractor.extract_detected_objects called")
                logger.info(f"Current class_names keys: {list(self.class_names.keys()) if self.class_names else 'None'}")

                if detection_result is None:
                    logger.warning("Detection result is None")
                    return []

                if not hasattr(detection_result, 'boxes'):
                    logger.error("Detection result does not have boxes attribute")
                    return []

                boxes = detection_result.boxes.xyxy.cpu().numpy()
                classes = detection_result.boxes.cls.cpu().numpy().astype(int)
                confidences = detection_result.boxes.conf.cpu().numpy()

                # 獲取圖像尺寸
                img_height, img_width = detection_result.orig_shape[:2]

                detected_objects = []

                for box, class_id, confidence in zip(boxes, classes, confidences):
                    try:
                        # 1. 先拿到這筆偵測物件的 class_name
                        class_name = self.class_names.get(int(class_id), f"unknown_class_{class_id}")
                        # 2. 計算這個 class 應該採用的動態 threshold
                        dyn_thr = self._get_dynamic_threshold(class_name)  # e.g. 0.25 * factor
                        # 3. 如果 confidence < dyn_thr，就跳過這一筆
                        if confidence < dyn_thr:
                            continue

                        # 後面維持原本的座標、中心、大小、區域等資訊計算
                        x1, y1, x2, y2 = box
                        width = x2 - x1
                        height = y2 - y1

                        # 中心點計算
                        center_x = (x1 + x2) / 2
                        center_y = (y1 + y2) / 2

                        # 標準化位置 (0-1)
                        norm_x = center_x / img_width
                        norm_y = center_y / img_height
                        norm_width = width / img_width
                        norm_height = height / img_height

                        # 面積計算
                        area = width * height
                        norm_area = area / (img_width * img_height)

                        # 區域判斷
                        object_region = "unknown"
                        if region_analyzer:
                            object_region = region_analyzer.determine_region(norm_x, norm_y)

                        # 調試信息：記錄映射過程
                        if class_name.startswith("unknown_class_"):
                            logger.warning(
                                f"Class ID {class_id} not found in class_names. "
                                f"Available keys: {list(self.class_names.keys())}"
                            )
                        else:
                            logger.debug(f"Successfully mapped class ID {class_id} to '{class_name}'")

                        detected_objects.append({
                            "class_id": int(class_id),
                            "class_name": class_name,
                            "confidence": float(confidence),
                            "box": [float(x1), float(y1), float(x2), float(y2)],
                            "center": [float(center_x), float(center_y)],
                            "normalized_center": [float(norm_x), float(norm_y)],
                            "size": [float(width), float(height)],
                            "normalized_size": [float(norm_width), float(norm_height)],
                            "area": float(area),
                            "normalized_area": float(norm_area),
                            "region": object_region
                        })

                    except Exception as e:
                        logger.error(f"Error processing object with class_id {class_id}: {str(e)}")
                        continue

                logger.info(f"Extracted {len(detected_objects)} objects from detection result")
                return detected_objects

            except Exception as e:
                logger.error(f"Error extracting detected objects: {str(e)}")
                logger.error(traceback.format_exc())
                return []

    def update_class_names(self, class_names: Dict[int, str]):
        """
        動態更新類別名稱映射

        Args:
            class_names: 新的類別名稱映射字典
        """
        try:
            self.class_names = class_names or {}
            logger.info(f"Class names updated: {len(self.class_names)} classes")
            logger.debug(f"Updated class names: {self.class_names}")
        except Exception as e:
            logger.error(f"Failed to update class names: {str(e)}")

    def categorize_object(self, obj: Dict) -> str:
        """
        將檢測到的物件分類到功能類別中，用於區域識別

        Args:
            obj: 物件字典

        Returns:
            物件功能類別字串
        """
        try:
            class_id = obj.get("class_id", -1)
            class_name = obj.get("class_name", "").lower()

            # 使用現有的類別映射（如果可用）
            if self.object_categories:
                for category, ids in self.object_categories.items():
                    if class_id in ids:
                        return category

            # 基於COCO類別名稱的後備分類
            furniture_items = ["chair", "couch", "bed", "dining table", "toilet"]
            plant_items = ["potted plant"]
            electronic_items = ["tv", "laptop", "mouse", "remote", "keyboard", "cell phone"]
            vehicle_items = ["bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat"]
            person_items = ["person"]
            kitchen_items = ["bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
                            "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog",
                            "pizza", "donut", "cake", "refrigerator", "oven", "toaster", "sink", "microwave"]
            sports_items = ["frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
                        "baseball glove", "skateboard", "surfboard", "tennis racket"]
            personal_items = ["handbag", "tie", "suitcase", "umbrella", "backpack"]

            if any(item in class_name for item in furniture_items):
                return "furniture"
            elif any(item in class_name for item in plant_items):
                return "plant"
            elif any(item in class_name for item in electronic_items):
                return "electronics"
            elif any(item in class_name for item in vehicle_items):
                return "vehicle"
            elif any(item in class_name for item in person_items):
                return "person"
            elif any(item in class_name for item in kitchen_items):
                return "kitchen_items"
            elif any(item in class_name for item in sports_items):
                return "sports"
            elif any(item in class_name for item in personal_items):
                return "personal_items"
            else:
                return "misc"

        except Exception as e:
            logger.error(f"Error categorizing object: {str(e)}")
            logger.error(traceback.format_exc())
            return "misc"

    def get_object_categories(self, detected_objects: List[Dict]) -> set:
        """
        從檢測到的物件中取得唯一的物件類別

        Args:
            detected_objects: 檢測到的物件列表

        Returns:
            唯一物件類別的集合
        """
        try:
            object_categories = set()
            for obj in detected_objects:
                category = self.categorize_object(obj)
                if category:
                    object_categories.add(category)

            logger.info(f"Found {len(object_categories)} unique object categories")
            return object_categories

        except Exception as e:
            logger.error(f"Error getting object categories: {str(e)}")
            logger.error(traceback.format_exc())
            return set()

    def identify_core_objects_for_scene(self, detected_objects: List[Dict], scene_type: str) -> List[Dict]:
        """
        識別定義特定場景類型的核心物件

        Args:
            detected_objects: 檢測到的物件列表
            scene_type: 場景類型

        Returns:
            場景的核心物件列表
        """
        try:
            core_objects = []

            # 場景核心物件映射
            scene_core_mapping = {
                "bedroom": [59],  # bed
                "kitchen": [68, 69, 71, 72],  # microwave, oven, sink, refrigerator
                "living_room": [57, 58, 62],  # sofa, chair, tv
                "dining_area": [60, 42, 43],  # dining table, fork, knife
                "office_workspace": [63, 64, 66, 73]  # laptop, mouse, keyboard, book
            }

            if scene_type in scene_core_mapping:
                core_class_ids = scene_core_mapping[scene_type]
                for obj in detected_objects:
                    if obj.get("class_id") in core_class_ids and obj.get("confidence", 0) >= 0.4:
                        core_objects.append(obj)

            logger.info(f"Identified {len(core_objects)} core objects for scene type '{scene_type}'")
            return core_objects

        except Exception as e:
            logger.error(f"Error identifying core objects for scene '{scene_type}': {str(e)}")
            logger.error(traceback.format_exc())
            return []

    def group_objects_by_category_and_region(self, detected_objects: List[Dict]) -> Dict:
        """
        將物件按類別和區域分組

        Args:
            detected_objects: 檢測到的物件列表

        Returns:
            按類別和區域分組的物件字典
        """
        try:
            category_regions = {}

            for obj in detected_objects:
                category = self.categorize_object(obj)
                if not category:
                    continue

                if category not in category_regions:
                    category_regions[category] = {}

                region = obj.get("region", "center")
                if region not in category_regions[category]:
                    category_regions[category][region] = []

                category_regions[category][region].append(obj)

            logger.info(f"Grouped objects into {len(category_regions)} categories across regions")
            return category_regions

        except Exception as e:
            logger.error(f"Error grouping objects by category and region: {str(e)}")
            logger.error(traceback.format_exc())
            return {}

    def filter_objects_by_confidence(self, detected_objects: List[Dict], min_confidence: float) -> List[Dict]:
        """
        根據信心度過濾物件

        Args:
            detected_objects: 檢測到的物件列表
            min_confidence: 最小信心度閾值

        Returns:
            過濾後的物件列表
        """
        try:
            filtered_objects = [
                obj for obj in detected_objects
                if obj.get("confidence", 0) >= min_confidence
            ]

            logger.info(f"Filtered {len(detected_objects)} objects to {len(filtered_objects)} objects with confidence >= {min_confidence}")
            return filtered_objects

        except Exception as e:
            logger.error(f"Error filtering objects by confidence: {str(e)}")
            logger.error(traceback.format_exc())
            return detected_objects  # 發生錯誤時返回原始列表
