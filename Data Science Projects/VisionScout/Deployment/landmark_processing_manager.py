import re
import logging
import traceback
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from PIL import Image

from clip_zero_shot_classifier import CLIPZeroShotClassifier
from landmark_activities import LANDMARK_ACTIVITIES
from landmark_data import ALL_LANDMARKS


class LandmarkProcessingManager:
    """
    負責處理所有地標相關的檢測和處理邏輯，包括未知物體的地標識別、
    地標物體的創建和驗證，以及地標引用的清理。
    """

    def __init__(self, enable_landmark: bool = True, use_clip: bool = True):
        """
        初始化地標處理管理器。

        Args:
            enable_landmark: 是否啟用地標檢測功能
            use_clip: 是否啟用 CLIP 分析功能
        """
        self.logger = logging.getLogger(__name__)
        self.enable_landmark = enable_landmark
        self.use_clip = use_clip

        # 載入地標相關數據
        self.landmark_activities = {}
        self.all_landmarks = {}
        self._load_landmark_data()

        # 地標分類器將按需初始化
        self.landmark_classifier = None

    def _load_landmark_data(self):
        """載入地標相關的數據結構。"""
        try:
            self.landmark_activities = LANDMARK_ACTIVITIES
            self.logger.info("Loaded LANDMARK_ACTIVITIES successfully")
        except ImportError as e:
            self.logger.warning(f"Failed to load LANDMARK_ACTIVITIES: {e}")
            self.landmark_activities = {}

        try:
            self.all_landmarks = ALL_LANDMARKS
            self.logger.info("Loaded ALL_LANDMARKS successfully")
        except ImportError as e:
            self.logger.warning(f"Failed to load ALL_LANDMARKS: {e}")
            self.all_landmarks = {}

    def set_landmark_classifier(self, landmark_classifier):
        """
        設置地標分類器實例。

        Args:
            landmark_classifier: CLIPZeroShotClassifier 實例
        """
        self.landmark_classifier = landmark_classifier

    def process_unknown_objects(self, detection_result, detected_objects, clip_analyzer=None):
        """
        對 YOLO 未能識別或信心度低的物體進行地標檢測。

        Args:
            detection_result: YOLO 檢測結果
            detected_objects: 已識別的物體列表
            clip_analyzer: CLIP 分析器實例（用於按需初始化地標分類器）

        Returns:
            tuple: (更新後的物體列表, 地標物體列表)
        """
        if (not self.enable_landmark or not self.use_clip or
            not hasattr(self, 'use_landmark_detection') or not self.use_landmark_detection):
            # 未啟用地標識別時，確保返回的物體列表中不包含任何地標物體
            cleaned_objects = [obj for obj in detected_objects if not obj.get("is_landmark", False)]
            return cleaned_objects, []

        try:
            # 獲取原始圖像
            original_image = None
            if detection_result is not None and hasattr(detection_result, 'orig_img'):
                original_image = detection_result.orig_img

            # 檢查原始圖像是否存在
            if original_image is None:
                self.logger.warning("Original image not available for landmark detection")
                return detected_objects, []

            # 確保原始圖像為 PIL 格式或可轉換為 PIL 格式
            if not isinstance(original_image, Image.Image):
                if isinstance(original_image, np.ndarray):
                    try:
                        if original_image.ndim == 3 and original_image.shape[2] == 4:  # RGBA
                            original_image = original_image[:, :, :3]  # 轉換為 RGB
                        if original_image.ndim == 2:  # 灰度圖
                            original_image = Image.fromarray(original_image).convert("RGB")
                        else:  # 假設為 RGB 或 BGR
                            original_image = Image.fromarray(original_image)

                        if hasattr(original_image, 'mode') and original_image.mode == 'BGR':  # 從 OpenCV 明確將 BGR 轉換為 RGB
                            original_image = original_image.convert('RGB')
                    except Exception as e:
                        self.logger.warning(f"Error converting image for landmark detection: {e}")
                        return detected_objects, []
                else:
                    self.logger.warning(f"Cannot process image of type {type(original_image)}")
                    return detected_objects, []

            # 獲取圖像維度
            if isinstance(original_image, np.ndarray):
                h, w = original_image.shape[:2]
            elif isinstance(original_image, Image.Image):
                w, h = original_image.size
            else:
                self.logger.warning(f"Unable to determine image dimensions for type {type(original_image)}")
                return detected_objects, []

            # 收集可能含有地標的區域
            candidate_boxes = []
            low_conf_boxes = []

            # 即使沒有 YOLO 檢測到的物體，也嘗試進行更詳細的地標分析
            if len(detected_objects) == 0:
                # 創建一個包含整個圖像的框
                full_image_box = [0, 0, w, h]
                low_conf_boxes.append(full_image_box)
                candidate_boxes.append((full_image_box, "full_image"))

                # 加入網格分析以增加檢測成功率
                grid_size = 2  # 2x2 網格
                for i in range(grid_size):
                    for j in range(grid_size):
                        # 創建網格框
                        grid_box = [
                            j * w / grid_size,
                            i * h / grid_size,
                            (j + 1) * w / grid_size,
                            (i + 1) * h / grid_size
                        ]
                        low_conf_boxes.append(grid_box)
                        candidate_boxes.append((grid_box, "grid"))

                # 創建更大的中心框（覆蓋中心 70% 區域）
                center_box = [
                    w * 0.15, h * 0.15,
                    w * 0.85, h * 0.85
                ]
                low_conf_boxes.append(center_box)
                candidate_boxes.append((center_box, "center"))

                self.logger.info("No YOLO detections, attempting detailed landmark analysis with multiple regions")
            else:
                try:
                    # 獲取原始 YOLO 檢測結果中的低置信度物體
                    if (hasattr(detection_result, 'boxes') and
                        hasattr(detection_result.boxes, 'xyxy') and
                        hasattr(detection_result.boxes, 'conf') and
                        hasattr(detection_result.boxes, 'cls')):
                        all_boxes = (detection_result.boxes.xyxy.cpu().numpy()
                                   if hasattr(detection_result.boxes.xyxy, 'cpu')
                                   else detection_result.boxes.xyxy)
                        all_confs = (detection_result.boxes.conf.cpu().numpy()
                                   if hasattr(detection_result.boxes.conf, 'cpu')
                                   else detection_result.boxes.conf)
                        all_cls = (detection_result.boxes.cls.cpu().numpy()
                                 if hasattr(detection_result.boxes.cls, 'cpu')
                                 else detection_result.boxes.cls)

                        # 收集低置信度區域和可能含有地標的區域（如建築物）
                        for i, (box, conf, cls) in enumerate(zip(all_boxes, all_confs, all_cls)):
                            is_low_conf = conf < 0.4 and conf > 0.1

                            # 根據物體類別 ID 識別建築物 - 使用通用分類
                            common_building_classes = [11, 12, 13, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65]  # 常見建築類別 ID
                            is_building = int(cls) in common_building_classes

                            # 計算相對面積 - 大物體
                            is_large_object = (box[2] - box[0]) * (box[3] - box[1]) > (0.1 * w * h)

                            if is_low_conf or is_building:
                                # 確保 box 是一個有效的數組或列表
                                if isinstance(box, (list, tuple, np.ndarray)) and len(box) >= 4:
                                    low_conf_boxes.append(box)
                                    if is_large_object:
                                        candidate_boxes.append((box, "building" if is_building else "low_conf"))
                except Exception as e:
                    self.logger.error(f"Error processing YOLO detections: {e}")
                    traceback.print_exc()

            # 按需初始化地標分類器
            if not self.landmark_classifier:
                if clip_analyzer and hasattr(clip_analyzer, 'get_clip_instance'):
                    try:
                        self.logger.info("Initializing landmark classifier for process_unknown_objects")
                        model, preprocess, device = clip_analyzer.get_clip_instance()
                        self.landmark_classifier = CLIPZeroShotClassifier(device=device)
                    except Exception as e:
                        self.logger.error(f"Error initializing landmark classifier: {e}")
                        return detected_objects, []
                else:
                    self.logger.warning("landmark_classifier not available and cannot be initialized")
                    return detected_objects, []

            # 使用智能地標搜索
            landmark_results = None
            try:
                # 確保有有效的框
                if not low_conf_boxes:
                    # 如果沒有低置信度框，添加全圖
                    low_conf_boxes.append([0, 0, w, h])

                landmark_results = self.landmark_classifier.intelligent_landmark_search(
                    original_image,
                    yolo_boxes=low_conf_boxes,
                    base_threshold=0.25
                )
            except Exception as e:
                self.logger.error(f"Error in intelligent_landmark_search: {e}")
                traceback.print_exc()
                return detected_objects, []

            # 處理識別結果
            landmark_objects = []

            # 如果有效的地標結果
            if landmark_results and landmark_results.get("is_landmark_scene", False):
                for landmark_info in landmark_results.get("detected_landmarks", []):
                    try:
                        # 使用 landmark_classifier 的閾值判斷
                        base_threshold = 0.25  # 基礎閾值

                        # 獲取地標類型並設定閾值
                        landmark_type = "architectural"  # 預設類型
                        type_threshold = 0.5  # 預設閾值

                        # 優先使用 landmark_classifier
                        if (hasattr(self.landmark_classifier, '_determine_landmark_type') and
                            landmark_info.get("landmark_id")):
                            landmark_type = self.landmark_classifier._determine_landmark_type(landmark_info.get("landmark_id"))
                            type_threshold = getattr(self.landmark_classifier, 'landmark_type_thresholds', {}).get(landmark_type, 0.5)
                        # 否則使用本地方法
                        elif hasattr(self, '_determine_landmark_type'):
                            landmark_type = self._determine_landmark_type(landmark_info.get("landmark_id", ""))
                            # 依據地標類型調整閾值
                            if landmark_type == "skyscraper":
                                type_threshold = 0.4
                            elif landmark_type == "natural":
                                type_threshold = 0.6
                        # 或者直接從地標 ID 推斷
                        else:
                            landmark_id = landmark_info.get("landmark_id", "").lower()
                            if any(term in landmark_id for term in ["mountain", "canyon", "waterfall", "lake", "river", "natural"]):
                                landmark_type = "natural"
                                type_threshold = 0.6
                            elif any(term in landmark_id for term in ["skyscraper", "building", "tower", "tall"]):
                                landmark_type = "skyscraper"
                                type_threshold = 0.4
                            elif any(term in landmark_id for term in ["monument", "memorial", "statue", "historical"]):
                                landmark_type = "monument"
                                type_threshold = 0.5

                        effective_threshold = base_threshold * (type_threshold / 0.5)

                        # 如果置信度足夠高
                        if landmark_info.get("confidence", 0) > effective_threshold:
                            # 獲取邊界框
                            if "box" in landmark_info:
                                box = landmark_info["box"]
                            else:
                                # 如果沒有邊界框，使用整個圖像的 90% 區域
                                margin_x, margin_y = w * 0.05, h * 0.05
                                box = [margin_x, margin_y, w - margin_x, h - margin_y]

                            # 計算中心點和其他必要信息
                            center_x = (box[0] + box[2]) / 2
                            center_y = (box[1] + box[3]) / 2
                            norm_center_x = center_x / w if w > 0 else 0.5
                            norm_center_y = center_y / h if h > 0 else 0.5

                            # 獲取區域位置（需要 spatial_analyzer 的支持）
                            region = "center"  # 預設

                            # 創建地標物體
                            landmark_obj = {
                                "class_id": (landmark_info.get("landmark_id", "")[:15]
                                           if isinstance(landmark_info.get("landmark_id", ""), str)
                                           else "-100"),  # 截斷過長的 ID
                                "class_name": landmark_info.get("landmark_name", "Unknown Landmark"),
                                "confidence": landmark_info.get("confidence", 0.0),
                                "box": box,
                                "center": (center_x, center_y),
                                "normalized_center": (norm_center_x, norm_center_y),
                                "size": (box[2] - box[0], box[3] - box[1]),
                                "normalized_size": (
                                    (box[2] - box[0]) / w if w > 0 else 0,
                                    (box[3] - box[1]) / h if h > 0 else 0
                                ),
                                "area": (box[2] - box[0]) * (box[3] - box[1]),
                                "normalized_area": (
                                    (box[2] - box[0]) * (box[3] - box[1]) / (w * h) if w * h > 0 else 0
                                ),
                                "region": region,
                                "is_landmark": True,
                                "landmark_id": landmark_info.get("landmark_id", ""),
                                "location": landmark_info.get("location", "Unknown Location")
                            }

                            # 添加額外信息
                            for key in ["year_built", "architectural_style", "significance"]:
                                if key in landmark_info:
                                    landmark_obj[key] = landmark_info[key]

                            # 添加地標類型
                            landmark_obj["landmark_type"] = landmark_type

                            # 添加到檢測物體列表
                            detected_objects.append(landmark_obj)
                            landmark_objects.append(landmark_obj)
                            self.logger.info(f"Detected landmark: {landmark_info.get('landmark_name', 'Unknown')} with confidence {landmark_info.get('confidence', 0.0):.2f}")
                    except Exception as e:
                        self.logger.error(f"Error processing landmark: {e}")
                        continue

                return detected_objects, landmark_objects

            return detected_objects, []

        except Exception as e:
            self.logger.error(f"Error in landmark detection: {e}")
            traceback.print_exc()
            return detected_objects, []

    def remove_landmark_references(self, text):
        """
        從文本中移除所有地標引用。

        Args:
            text: 輸入文本

        Returns:
            str: 清除地標引用後的文本
        """
        if not text:
            return text

        try:
            # 動態收集所有地標名稱和位置
            landmark_names = []
            locations = []

            for landmark_id, info in self.all_landmarks.items():
                # 收集地標名稱及其別名
                landmark_names.append(info["name"])
                landmark_names.extend(info.get("aliases", []))

                # 收集地理位置
                if "location" in info:
                    location = info["location"]
                    locations.append(location)

                    # 處理分離的城市和國家名稱
                    parts = location.split(",")
                    if len(parts) >= 1:
                        locations.append(parts[0].strip())
                    if len(parts) >= 2:
                        locations.append(parts[1].strip())

            # 使用正則表達式動態替換所有地標名稱
            for name in landmark_names:
                if name and len(name) > 2:  # 避免過短的名稱
                    text = re.sub(r'\b' + re.escape(name) + r'\b', "tall structure", text, flags=re.IGNORECASE)

            # 動態替換所有位置引用
            for location in locations:
                if location and len(location) > 2:
                    # 替換常見位置表述模式
                    text = re.sub(r'in ' + re.escape(location), "in the urban area", text, flags=re.IGNORECASE)
                    text = re.sub(r'of ' + re.escape(location), "of the urban area", text, flags=re.IGNORECASE)
                    text = re.sub(r'\b' + re.escape(location) + r'\b', "the urban area", text, flags=re.IGNORECASE)

        except Exception as e:
            self.logger.warning(f"Error in dynamic landmark reference removal, using generic patterns: {e}")
            # 通用地標描述模式
            landmark_patterns = [
                # 地標地點模式
                (r'an iconic structure in ([A-Z][a-zA-Z\s,]+)', r'an urban structure'),
                (r'a famous (monument|tower|landmark) in ([A-Z][a-zA-Z\s,]+)', r'an urban structure'),
                (r'(the [A-Z][a-zA-Z\s]+ Tower)', r'the tower'),
                (r'(the [A-Z][a-zA-Z\s]+ Building)', r'the building'),
                (r'(the CN Tower)', r'the tower'),
                (r'([A-Z][a-zA-Z\s]+) Tower', r'tall structure'),

                # 地標位置關係模式
                (r'(centered|built|located|positioned) around the ([A-Z][a-zA-Z\s]+? (Tower|Monument|Landmark))', r'located in this area'),

                # 地標活動模式
                (r'(sightseeing|guided tours|cultural tourism) (at|around|near) (this landmark|the [A-Z][a-zA-Z\s]+)', r'\1 in this area'),

                # 一般性地標形容模式
                (r'this (famous|iconic|historic|well-known) (landmark|monument|tower|structure)', r'this urban structure'),
                (r'landmark scene', r'urban scene'),
                (r'tourist destination', r'urban area'),
                (r'tourist attraction', r'urban area')
            ]

            for pattern, replacement in landmark_patterns:
                text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        return text

    def get_alternative_scene_type(self, landmark_scene_type, detected_objects, scene_scores):
        """
        為地標場景類型選擇適合的替代類型。

        Args:
            landmark_scene_type: 原始地標場景類型
            detected_objects: 檢測到的物體列表
            scene_scores: 所有場景類型的分數

        Returns:
            str: 適合的替代場景類型
        """
        # 1. 嘗試從現有場景分數中找出第二高的非地標場景
        landmark_types = {"tourist_landmark", "natural_landmark", "historical_monument"}
        alternative_scores = {k: v for k, v in scene_scores.items() if k not in landmark_types and v > 0.2}

        if alternative_scores:
            # 返回分數最高的非地標場景類型
            return max(alternative_scores.items(), key=lambda x: x[1])[0]

        # 2. 基於物體組合推斷場景類型
        object_counts = {}
        for obj in detected_objects:
            class_name = obj.get("class_name", "")
            if class_name not in object_counts:
                object_counts[class_name] = 0
            object_counts[class_name] += 1

        # 根據物體組合決定場景類型
        if "car" in object_counts or "truck" in object_counts or "bus" in object_counts:
            # 有車輛，可能是街道或交叉路口
            if "traffic light" in object_counts or "stop sign" in object_counts:
                return "intersection"
            else:
                return "city_street"

        if "building" in object_counts and object_counts.get("person", 0) > 0:
            # 有建築物和人，可能是商業區
            return "commercial_district"

        if object_counts.get("person", 0) > 3:
            # 多個行人，可能是行人區
            return "pedestrian_area"

        if "bench" in object_counts or "potted plant" in object_counts:
            # 有長椅或盆栽，可能是公園區域
            return "park_area"

        # 3. 根據原始地標場景類型選擇合適的替代場景
        if landmark_scene_type == "natural_landmark":
            return "outdoor_natural_area"
        elif landmark_scene_type == "historical_monument":
            return "urban_architecture"

        # 默認回退到城市街道
        return "city_street"

    def extract_landmark_specific_activities(self, landmark_objects):
        """
        從識別的地標中提取特定活動。

        Args:
            landmark_objects: 地標物體列表

        Returns:
            List[str]: 地標特定活動列表
        """
        landmark_specific_activities = []

        # 優先收集來自識別地標的特定活動
        for lm_obj in landmark_objects:
            lm_id = lm_obj.get("landmark_id")
            if lm_id and lm_id in self.landmark_activities:
                landmark_specific_activities.extend(self.landmark_activities[lm_id])

        if landmark_specific_activities:
            landmark_names = [lm.get('landmark_name', 'unknown') for lm in landmark_objects if lm.get('is_landmark', False)]
            self.logger.info(f"Added {len(landmark_specific_activities)} landmark-specific activities for {', '.join(landmark_names)}")

        return landmark_specific_activities

    def update_enable_landmark_status(self, enable_landmark: bool):
        """
        更新地標檢測的啟用狀態。

        Args:
            enable_landmark: 是否啟用地標檢測
        """
        self.enable_landmark = enable_landmark

    def update_use_landmark_detection_status(self, use_landmark_detection: bool):
        """
        更新地標檢測使用狀態。

        Args:
            use_landmark_detection: 是否使用地標檢測
        """
        self.use_landmark_detection = use_landmark_detection
