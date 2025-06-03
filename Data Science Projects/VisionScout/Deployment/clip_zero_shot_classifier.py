
import torch
import clip
from PIL import Image
import numpy as np
import logging
import traceback
from typing import List, Dict, Tuple, Optional, Union, Any

from clip_model_manager import CLIPModelManager
from landmark_data_manager import LandmarkDataManager
from image_analyzer import ImageAnalyzer
from confidence_manager import ConfidenceManager
from result_cache_manager import ResultCacheManager

class CLIPZeroShotClassifier:
    """
    使用CLIP模型進行zero shot，專注於辨識世界知名地標。
    作為YOLO的補充，處理YOLO無法辨識到的地標。

    這是一個總窗口class，協調各個組件的工作以提供統一的對外接口。
    """

    def __init__(self, model_name: str = "ViT-B/16", device: str = None):
        """
        初始化CLIP零樣本分類器

        Args:
            model_name: CLIP模型名稱，默認為"ViT-B/16"
            device: 運行設備，None則自動選擇
        """
        self.logger = logging.getLogger(__name__)

        # 初始化各個組件
        self.clip_model_manager = CLIPModelManager(model_name, device)
        self.landmark_data_manager = LandmarkDataManager()
        self.image_analyzer = ImageAnalyzer()
        self.confidence_manager = ConfidenceManager()
        self.cache_manager = ResultCacheManager()

        # 預計算地標文本特徵
        self.landmark_text_features = None
        self._precompute_landmark_features()

        self.logger.info(f"Initializing CLIP Zero-Shot Landmark Classifier ({model_name}) on {self.clip_model_manager.get_device()}")

    def _precompute_landmark_features(self):
        """
        預計算地標文本特徵，提高批處理效率
        """
        try:
            if self.landmark_data_manager.is_landmark_enabled():
                landmark_prompts = self.landmark_data_manager.get_landmark_prompts()
                if landmark_prompts:
                    self.landmark_text_features = self.clip_model_manager.encode_text_batch(landmark_prompts)
                    self.logger.info(f"Precomputed text features for {len(landmark_prompts)} landmark prompts")
                else:
                    self.logger.warning("No landmark prompts available for precomputation")
            else:
                self.logger.warning("Landmark data not enabled, skipping feature precomputation")
        except Exception as e:
            self.logger.error(f"Error precomputing landmark features: {e}")
            self.logger.error(traceback.format_exc())

    def set_batch_size(self, batch_size: int):
        """
        設置批處理大小

        Args:
            batch_size: 新的批處理大小
        """
        self.confidence_manager.set_batch_size(batch_size)

    def adjust_confidence_threshold(self, detection_type: str, multiplier: float):
        """
        調整特定檢測類型的置信度閾值乘數

        Args
            detection_type: 檢測類型 ('close_up', 'partial', 'distant', 'full_image')
            multiplier: 置信度閾值乘數
        """
        self.confidence_manager.adjust_confidence_threshold(detection_type, multiplier)

    def classify_image_region(self,
                            image: Union[Image.Image, np.ndarray],
                            box: List[float],
                            threshold: float = 0.25,
                            detection_type: str = "close_up") -> Dict[str, Any]:
        """
        對圖像的特定區域進行地標分類，具有增強的多尺度和部分識別能力

        Args:
            image: 原始圖像 (PIL Image 或 numpy數組)
            box: 邊界框 [x1, y1, x2, y2]
            threshold: 基礎分類置信度閾值
            detection_type: 檢測類型，影響置信度調整

        Returns:
            Dict: 地標分類結果
        """
        try:
            if not self.landmark_data_manager.is_landmark_enabled():
                return {"is_landmark": False, "confidence": 0.0}

            # 確保圖像是PIL格式
            if not isinstance(image, Image.Image):
                if isinstance(image, np.ndarray):
                    image = Image.fromarray(image)
                else:
                    raise ValueError("Unsupported image format. Expected PIL Image or numpy array.")

            # 生成圖像區域的hash用於快取
            image_hash = self.image_analyzer.get_image_hash(image)
            region_key = self.cache_manager.get_region_cache_key(image_hash, tuple(box), detection_type)

            # 檢查快取
            cached_result = self.cache_manager.get_cached_result(region_key)
            if cached_result is not None:
                return cached_result

            # 裁剪區域
            x1, y1, x2, y2 = map(int, box)
            cropped_image = image.crop((x1, y1, x2, y2))
            enhanced_image = self.image_analyzer.enhance_features(cropped_image)

            # 分析視角信息
            viewpoint_info = self.image_analyzer.analyze_viewpoint(enhanced_image, self.clip_model_manager)
            dominant_viewpoint = viewpoint_info["dominant_viewpoint"]

            # 計算區域信息
            region_width = x2 - x1
            region_height = y2 - y1
            image_width, image_height = image.size

            # 根據區域大小判斷可能的檢測類型
            if detection_type == "auto":
                detection_type = self.confidence_manager.determine_detection_type_from_region(
                    region_width, region_height, image_width, image_height
                )

            # 根據視角調整檢測類型
            detection_type = self.confidence_manager.adjust_detection_type_by_viewpoint(detection_type, dominant_viewpoint)

            # 調整置信度閾值
            adjusted_threshold = self.confidence_manager.calculate_adjusted_threshold(threshold, detection_type)

            # 準備多尺度和縱橫比分析
            scales = [1.0]
            if detection_type in ["partial", "distant"]:
                scales = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]

            if dominant_viewpoint in ["angled_view", "low_angle"]:
                scales = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4]

            aspect_ratios = [1.0, 0.8, 1.2]
            if dominant_viewpoint in ["angled_view", "unique_feature"]:
                aspect_ratios = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5]

            best_result = {
                "landmark_id": None,
                "landmark_name": None,
                "confidence": 0.0,
                "is_landmark": False
            }

            # 多尺度和縱橫比分析
            for scale in scales:
                for aspect_ratio in aspect_ratios:
                    try:
                        # 縮放裁剪區域
                        current_width, current_height = cropped_image.size

                        if aspect_ratio != 1.0:
                            new_width = int(current_width * scale * (1/aspect_ratio)**0.5)
                            new_height = int(current_height * scale * aspect_ratio**0.5)
                        else:
                            new_width = int(current_width * scale)
                            new_height = int(current_height * scale)

                        new_width = max(1, new_width)
                        new_height = max(1, new_height)

                        scaled_image = cropped_image.resize((new_width, new_height), Image.LANCZOS)

                        # 預處理並獲取特徵
                        image_input = self.clip_model_manager.preprocess_image(scaled_image)
                        image_features = self.clip_model_manager.encode_image(image_input)

                        # 計算相似度
                        similarity = self.clip_model_manager.calculate_similarity(image_features, self.landmark_text_features)

                        # 找到最佳匹配
                        best_idx = similarity[0].argmax().item()
                        best_score = similarity[0][best_idx]

                        # 如果當前尺度結果更好，則更新
                        if best_score > best_result["confidence"]:
                            landmark_id, landmark_info = self.landmark_data_manager.get_landmark_by_index(best_idx)

                            if landmark_id:
                                # 先從 LandmarkDataManager 拿 location
                                loc = landmark_info.get("location", "")
                                # 如果 loc 為空，就從全域 ALL_LANDMARKS 補上
                                if not loc and landmark_id in ALL_LANDMARKS:
                                    loc = ALL_LANDMARKS[landmark_id].get("location", "")
                                best_result = {
                                    "landmark_id": landmark_id,
                                    "landmark_name": landmark_info.get("name", "Unknown"),
                                    "location": loc or "Unknown Location",
                                    "confidence": float(best_score),
                                    "is_landmark": best_score >= adjusted_threshold,
                                    "scale_used": scale,
                                    "aspect_ratio_used": aspect_ratio,
                                    "viewpoint": dominant_viewpoint
                                }

                                # 添加額外可用信息
                                for key in ["year_built", "architectural_style", "significance"]:
                                    if key in landmark_info:
                                        best_result[key] = landmark_info[key]

                    except Exception as e:
                        self.logger.error(f"Error in scale analysis: {e}")
                        continue

            # 應用地標類型閾值調整
            if best_result["landmark_id"]:
                landmark_type = self.landmark_data_manager.determine_landmark_type(best_result["landmark_id"])
                final_threshold = self.confidence_manager.calculate_final_threshold(adjusted_threshold, detection_type, landmark_type)

                best_result["is_landmark"] = self.confidence_manager.evaluate_confidence(best_result["confidence"], final_threshold)
                best_result["landmark_type"] = landmark_type
                best_result["threshold_applied"] = final_threshold

            # 快取結果
            self.cache_manager.set_cached_result(region_key, best_result)

            return best_result

        except Exception as e:
            self.logger.error(f"Error in classify_image_region: {e}")
            self.logger.error(traceback.format_exc())
            return {"is_landmark": False, "confidence": 0.0}


    def classify_batch_regions(self,
                              image: Union[Image.Image, np.ndarray],
                              boxes: List[List[float]],
                              threshold: float = 0.28) -> List[Dict[str, Any]]:
        """
        批量處理多個圖像區域，提高效率

        Args:
            image: 原始圖像
            boxes: 邊界框列表
            threshold: 置信度閾值

        Returns:
            List[Dict]: 分類結果列表
        """
        try:
            if not self.landmark_data_manager.is_landmark_enabled() or self.landmark_text_features is None:
                return [{"is_landmark": False, "confidence": 0.0} for _ in boxes]

            # 確保圖像是PIL格式
            if not isinstance(image, Image.Image):
                if isinstance(image, np.ndarray):
                    image = Image.fromarray(image)
                else:
                    raise ValueError("Unsupported image format. Expected PIL Image or numpy array.")

            if not boxes:
                return []

            # 批量處理所有區域
            batch_features = self.clip_model_manager.batch_process_regions(image, boxes)

            # 計算相似度
            similarity = self.clip_model_manager.calculate_similarity(batch_features, self.landmark_text_features)

            # 處理每個區域的結果
            results = []
            for i, sim in enumerate(similarity):
                best_idx = sim.argmax().item()
                best_score = sim[best_idx]

                if best_score >= threshold:
                    landmark_id, landmark_info = self.landmark_data_manager.get_landmark_by_index(best_idx)

                    if landmark_id:
                        # 如果landmark_info["location"] 為空，則從 ALL_LANDMARKS 補
                        loc = landmark_info.get("location", "")
                        if not loc and landmark_id in ALL_LANDMARKS:
                            loc = ALL_LANDMARKS[landmark_id].get("location", "")
                        results.append({
                            "landmark_id": landmark_id,
                            "landmark_name": landmark_info.get("name", "Unknown"),
                            "location": loc or "Unknown Location",
                            "confidence": float(best_score),
                            "is_landmark": True,
                            "box": boxes[i]
                        })
                    else:
                        results.append({
                            "landmark_id": None,
                            "landmark_name": None,
                            "confidence": float(best_score),
                            "is_landmark": False,
                            "box": boxes[i]
                        })
                else:
                    results.append({
                        "landmark_id": None,
                        "landmark_name": None,
                        "confidence": float(best_score),
                        "is_landmark": False,
                        "box": boxes[i]
                    })

            return results

        except Exception as e:
            self.logger.error(f"Error in classify_batch_regions: {e}")
            self.logger.error(traceback.format_exc())
            return [{"is_landmark": False, "confidence": 0.0} for _ in boxes]

    def search_entire_image(self,
                           image: Union[Image.Image, np.ndarray],
                           threshold: float = 0.35,
                           detailed_analysis: bool = False) -> Dict[str, Any]:
        """
        檢查整張圖像是否包含地標，具有增強的分析能力

        Args:
            image: 原始圖像
            threshold: 置信度閾值
            detailed_analysis: 是否進行詳細分析，包括多區域檢測

        Returns:
            Dict: 地標分類結果
        """
        try:
            if not self.landmark_data_manager.is_landmark_enabled() or self.landmark_text_features is None:
                return {"is_landmark": False, "confidence": 0.0}

            # 確保圖像是PIL格式
            if not isinstance(image, Image.Image):
                if isinstance(image, np.ndarray):
                    image = Image.fromarray(image)
                else:
                    raise ValueError("Unsupported image format. Expected PIL Image or numpy array.")

            # 檢查cache
            image_hash = self.image_analyzer.get_image_hash(image)
            image_key = self.cache_manager.get_image_cache_key(image_hash, "entire_image", detailed_analysis)

            cached_result = self.cache_manager.get_cached_result(image_key)
            if cached_result is not None:
                return cached_result

            # 調整閾值
            adjusted_threshold = self.confidence_manager.calculate_adjusted_threshold(threshold, "full_image")

            # 預處理並獲取特徵
            image_input = self.clip_model_manager.preprocess_image(image)
            image_features = self.clip_model_manager.encode_image(image_input)

            # calculate相似度
            similarity = self.clip_model_manager.calculate_similarity(image_features, self.landmark_text_features)

            # 找到最佳匹配
            best_idx = similarity[0].argmax().item()
            best_score = similarity[0][best_idx]

            # 獲取top3地標
            top_indices = similarity[0].argsort()[-3:][::-1]
            top_landmarks = []

            for idx in top_indices:
                score = similarity[0][idx]
                landmark_id, landmark_info = self.landmark_data_manager.get_landmark_by_index(idx)

                if landmark_id:
                    # 補 location
                    loc_top = landmark_info.get("location", "")
                    if not loc_top and landmark_id in ALL_LANDMARKS:
                        loc_top = ALL_LANDMARKS[landmark_id].get("location", "")
                    landmark_result = {
                        "landmark_id": landmark_id,
                        "landmark_name": landmark_info.get("name", "Unknown"),
                        "location": loc_top or "Unknown Location",
                        "confidence": float(score)
                    }

                    # 加額外可用信息
                    for key in ["year_built", "architectural_style", "significance"]:
                        if key in landmark_info:
                            landmark_result[key] = landmark_info[key]

                    top_landmarks.append(landmark_result)

            # main result
            result = {}
            if best_score >= adjusted_threshold:
                landmark_id, landmark_info = self.landmark_data_manager.get_landmark_by_index(best_idx)

                if landmark_id:
                    # 應用地標類型特定閾值
                    landmark_type = self.landmark_data_manager.determine_landmark_type(landmark_id)
                    final_threshold = self.confidence_manager.calculate_final_threshold(adjusted_threshold, "full_image", landmark_type)

                    if self.confidence_manager.evaluate_confidence(best_score, final_threshold):
                        # 補 location
                        loc_main = landmark_info.get("location", "")
                        if not loc_main and landmark_id in ALL_LANDMARKS:
                            loc_main = ALL_LANDMARKS[landmark_id].get("location", "")
                        result = {
                            "landmark_id": landmark_id,
                            "landmark_name": landmark_info.get("name", "Unknown"),
                            "location": loc_main or "Unknown Location",
                            "confidence": float(best_score),
                            "is_landmark": True,
                            "landmark_type": landmark_type,
                            "top_landmarks": top_landmarks
                        }

                        # 添加額外可用信息
                        for key in ["year_built", "architectural_style", "significance"]:
                            if key in landmark_info:
                                result[key] = landmark_info[key]
                    else:
                        result = {
                            "landmark_id": None,
                            "landmark_name": None,
                            "confidence": float(best_score),
                            "is_landmark": False,
                            "top_landmarks": top_landmarks
                        }
            else:
                result = {
                    "landmark_id": None,
                    "landmark_name": None,
                    "confidence": float(best_score),
                    "is_landmark": False,
                    "top_landmarks": top_landmarks
                }

            # 詳細分析
            if detailed_analysis and result.get("is_landmark", False):
                width, height = image.size
                regions = [
                    [width * 0.25, height * 0.25, width * 0.75, height * 0.75],
                    [0, 0, width * 0.5, height],
                    [width * 0.5, 0, width, height],
                    [0, 0, width, height * 0.5],
                    [0, height * 0.5, width, height]
                ]

                region_results = []
                for i, box in enumerate(regions):
                    region_result = self.classify_image_region(
                        image,
                        box,
                        threshold=threshold * 0.9,
                        detection_type="partial"
                    )
                    if region_result["is_landmark"]:
                        region_result["region_name"] = ["center", "left", "right", "top", "bottom"][i]
                        region_results.append(region_result)

                if region_results:
                    result["region_analyses"] = region_results

            # 快取結果
            self.cache_manager.set_cached_result(image_key, result)

            return result

        except Exception as e:
            self.logger.error(f"Error in search_entire_image: {e}")
            self.logger.error(traceback.format_exc())
            return {"is_landmark": False, "confidence": 0.0}


    def intelligent_landmark_search(self,
                                  image: Union[Image.Image, np.ndarray],
                                  yolo_boxes: Optional[List[List[float]]] = None,
                                  base_threshold: float = 0.25) -> Dict[str, Any]:
        """
        對圖像進行地標搜索，綜合整張圖像分析和區域分析

        Args:
            image: 原始圖像
            yolo_boxes: YOLO檢測到的邊界框 (可選)
            base_threshold: 基礎置信度閾值

        Returns:
            Dict: 包含所有檢測結果的綜合分析
        """
        try:
            if not self.landmark_data_manager.is_landmark_enabled():
                return {
                    "full_image_analysis": {},
                    "is_landmark_scene": False,
                    "detected_landmarks": []
                }

            # 確保圖像是PIL格式
            if not isinstance(image, Image.Image):
                if isinstance(image, np.ndarray):
                    image = Image.fromarray(image)
                else:
                    raise ValueError("Unsupported image format. Expected PIL Image or numpy array.")

            # 調整閾值
            actual_threshold = base_threshold * 0.85 if yolo_boxes is None or len(yolo_boxes) == 0 else base_threshold

            # 首先對整張圖像進行分析
            full_image_result = self.search_entire_image(
                image,
                threshold=actual_threshold,
                detailed_analysis=True
            )

            # 如果沒有YOLO框且全圖分析未發現地標，進行金字塔分析
            if (yolo_boxes is None or len(yolo_boxes) == 0) and (not full_image_result or not full_image_result.get("is_landmark", False)):
                self.logger.info("No YOLO boxes provided, attempting multi-scale pyramid analysis")
                pyramid_results = self.image_analyzer.perform_pyramid_analysis(
                    image,
                    self.clip_model_manager,
                    self.landmark_data_manager,
                    levels=4,
                    base_threshold=actual_threshold,
                    aspect_ratios=[1.0, 0.75, 1.5, 0.5, 2.0]
                )

                if pyramid_results and pyramid_results.get("is_landmark", False) and pyramid_results.get("best_result", {}).get("confidence", 0) > actual_threshold:
                    if not full_image_result or not full_image_result.get("is_landmark", False):
                        full_image_result = {
                            "is_landmark": True,
                            "landmark_id": pyramid_results["best_result"]["landmark_id"],
                            "landmark_name": pyramid_results["best_result"]["landmark_name"],
                            "confidence": pyramid_results["best_result"]["confidence"],
                            "location": pyramid_results["best_result"].get("location", "Unknown Location")
                        }
                        self.logger.info(f"Pyramid analysis detected landmark: {pyramid_results['best_result']['landmark_name']} with confidence {pyramid_results['best_result']['confidence']:.3f}")

            # 初始化結果dict
            result = {
                "full_image_analysis": full_image_result if full_image_result else {},
                "is_landmark_scene": False,
                "detected_landmarks": []
            }

            # 處理上下文感知比較
            if full_image_result and "top_landmarks" in full_image_result and len(full_image_result["top_landmarks"]) >= 2:
                top_landmarks = full_image_result["top_landmarks"]

                if len(top_landmarks) >= 2 and abs(top_landmarks[0]["confidence"] - top_landmarks[1]["confidence"]) < 0.1:
                    architectural_analysis = self.image_analyzer.analyze_architectural_features(image, self.clip_model_manager)

                    for i, landmark in enumerate(top_landmarks[:2]):
                        if i >= len(top_landmarks):
                            continue

                        adjusted_confidence = self.confidence_manager.apply_architectural_boost(
                            landmark["confidence"],
                            architectural_analysis,
                            landmark.get("landmark_id", "")
                        )

                        if adjusted_confidence != landmark["confidence"]:
                            top_landmarks[i]["confidence"] = adjusted_confidence

                    # 重新排序
                    top_landmarks.sort(key=lambda x: x["confidence"], reverse=True)
                    full_image_result["top_landmarks"] = top_landmarks
                    if top_landmarks:
                        full_image_result["landmark_id"] = top_landmarks[0]["landmark_id"]
                        full_image_result["landmark_name"] = top_landmarks[0]["landmark_name"]
                        full_image_result["confidence"] = top_landmarks[0]["confidence"]
                        full_image_result["location"] = top_landmarks[0].get("location", "Unknown Location")

            # 處理全圖結果
            if full_image_result and full_image_result.get("is_landmark", False):
                result["is_landmark_scene"] = True
                landmark_id = full_image_result.get("landmark_id", "unknown")

                landmark_specific_info = self.landmark_data_manager.extract_landmark_specific_info(landmark_id)

                landmark_info = {
                    "landmark_id": landmark_id,
                    "landmark_name": full_image_result.get("landmark_name", "Unknown Landmark"),
                    "confidence": full_image_result.get("confidence", 0.0),
                    "location": full_image_result.get("location", "Unknown Location"),
                    "region_type": "full_image",
                    "box": [0, 0, getattr(image, 'width', 0), getattr(image, 'height', 0)]
                }

                landmark_info.update(landmark_specific_info)

                if landmark_specific_info.get("landmark_name"):
                    landmark_info["landmark_name"] = landmark_specific_info["landmark_name"]

                result["detected_landmarks"].append(landmark_info)

                if landmark_specific_info.get("has_specific_activities", False):
                    result["primary_landmark_activities"] = landmark_specific_info.get("landmark_specific_activities", [])
                    self.logger.info(f"Set primary landmark activities: {len(result['primary_landmark_activities'])} activities for {landmark_info['landmark_name']}")

            # 處理YOLO邊界框
            if yolo_boxes and len(yolo_boxes) > 0:
                for box in yolo_boxes:
                    try:
                        box_result = self.classify_image_region(
                            image,
                            box,
                            threshold=base_threshold,
                            detection_type="auto"
                        )

                        if box_result and box_result.get("is_landmark", False):
                            is_duplicate = False
                            for existing in result["detected_landmarks"]:
                                if existing.get("landmark_id") == box_result.get("landmark_id"):
                                    if box_result.get("confidence", 0) > existing.get("confidence", 0):
                                        existing.update({
                                            "confidence": box_result.get("confidence", 0),
                                            "region_type": "yolo_box",
                                            "box": box
                                        })
                                    is_duplicate = True
                                    break

                            if not is_duplicate:
                                result["detected_landmarks"].append({
                                    "landmark_id": box_result.get("landmark_id", "unknown"),
                                    "landmark_name": box_result.get("landmark_name", "Unknown Landmark"),
                                    "confidence": box_result.get("confidence", 0.0),
                                    "location": box_result.get("location", "Unknown Location"),
                                    "region_type": "yolo_box",
                                    "box": box
                                })
                    except Exception as e:
                        self.logger.error(f"Error in analyzing YOLO box: {e}")
                        continue

            # 網格搜索（如果需要）
            should_do_grid_search = (
                len(result["detected_landmarks"]) == 0 or
                max([landmark.get("confidence", 0) for landmark in result["detected_landmarks"]], default=0) < 0.5
            )

            if should_do_grid_search:
                try:
                    width, height = getattr(image, 'size', (getattr(image, 'width', 0), getattr(image, 'height', 0)))
                    if not isinstance(width, (int, float)) or width <= 0:
                        width = getattr(image, 'width', 0)
                    if not isinstance(height, (int, float)) or height <= 0:
                        height = getattr(image, 'height', 0)

                    if width > 0 and height > 0:
                        grid_boxes = []
                        for i in range(5):
                            for j in range(5):
                                grid_boxes.append([
                                    width * (j/5), height * (i/5),
                                    width * ((j+1)/5), height * ((i+1)/5)
                                ])

                        for box in grid_boxes:
                            try:
                                grid_result = self.classify_image_region(
                                    image,
                                    box,
                                    threshold=base_threshold * 0.9,
                                    detection_type="partial"
                                )

                                if grid_result and grid_result.get("is_landmark", False):
                                    is_duplicate = False
                                    for existing in result["detected_landmarks"]:
                                        if existing.get("landmark_id") == grid_result.get("landmark_id"):
                                            is_duplicate = True
                                            break

                                    if not is_duplicate:
                                        result["detected_landmarks"].append({
                                            "landmark_id": grid_result.get("landmark_id", "unknown"),
                                            "landmark_name": grid_result.get("landmark_name", "Unknown Landmark"),
                                            "confidence": grid_result.get("confidence", 0.0),
                                            "location": grid_result.get("location", "Unknown Location"),
                                            "region_type": "grid",
                                            "box": box
                                        })
                            except Exception as e:
                                self.logger.error(f"Error in analyzing grid region: {e}")
                                continue
                except Exception as e:
                    self.logger.error(f"Error in grid search: {e}")
                    self.logger.error(traceback.format_exc())

            # 按置信度排序檢測結果
            result["detected_landmarks"].sort(key=lambda x: x.get("confidence", 0), reverse=True)

            # 更新整體場景類型判斷
            if len(result["detected_landmarks"]) > 0:
                result["is_landmark_scene"] = True
                result["primary_landmark"] = result["detected_landmarks"][0]

                if full_image_result and "clip_analysis" in full_image_result:
                    result["clip_analysis_on_full_image"] = full_image_result["clip_analysis"]

            return result

        except Exception as e:
            self.logger.error(f"Error in intelligent_landmark_search: {e}")
            self.logger.error(traceback.format_exc())
            return {
                "full_image_analysis": {},
                "is_landmark_scene": False,
                "detected_landmarks": []
            }

    def enhanced_landmark_detection(self,
                                  image: Union[Image.Image, np.ndarray],
                                  threshold: float = 0.3) -> Dict[str, Any]:
        """
        使用多種分析技術進行增強地標檢測

        Args:
            image: 輸入圖像
            threshold: 基礎置信度閾值

        Returns:
            Dict: 綜合地標檢測結果
        """
        try:
            if not self.landmark_data_manager.is_landmark_enabled():
                return {"is_landmark_scene": False, "detected_landmarks": []}

            # 確保圖像是PIL格式
            if not isinstance(image, Image.Image):
                if isinstance(image, np.ndarray):
                    image = Image.fromarray(image)
                else:
                    raise ValueError("Unsupported image format. Expected PIL Image or numpy array.")

            # 1: 分析視角以調整檢測參數
            viewpoint_info = self.image_analyzer.analyze_viewpoint(image, self.clip_model_manager)
            viewpoint = viewpoint_info["dominant_viewpoint"]

            # 根據視角調整閾值
            if viewpoint == "distant":
                adjusted_threshold = threshold * 0.7
            elif viewpoint == "close_up":
                adjusted_threshold = threshold * 1.1
            else:
                adjusted_threshold = threshold

            # 2: 執行多尺度金字塔分析
            pyramid_results = self.image_analyzer.perform_pyramid_analysis(
                image,
                self.clip_model_manager,
                self.landmark_data_manager,
                levels=3,
                base_threshold=adjusted_threshold
            )

            # 3: 執行基於網格的區域分析
            grid_results = []
            width, height = image.size

            # 根據視角創建自適應網格
            if viewpoint == "distant":
                grid_size = 3
            elif viewpoint == "close_up":
                grid_size = 5
            else:
                grid_size = 4

            # 生成網格區域
            for i in range(grid_size):
                for j in range(grid_size):
                    box = [
                        width * (j/grid_size),
                        height * (i/grid_size),
                        width * ((j+1)/grid_size),
                        height * ((i+1)/grid_size)
                    ]

                    region_result = self.classify_image_region(
                        image,
                        box,
                        threshold=adjusted_threshold,
                        detection_type="auto"
                    )

                    if region_result["is_landmark"]:
                        region_result["grid_position"] = (i, j)
                        grid_results.append(region_result)

            # 4: 交叉驗證並合併結果
            all_detections = []

            # 添加金字塔結果
            if pyramid_results["is_landmark"] and pyramid_results["best_result"]:
                all_detections.append({
                    "source": "pyramid",
                    "landmark_id": pyramid_results["best_result"]["landmark_id"],
                    "landmark_name": pyramid_results["best_result"]["landmark_name"],
                    "confidence": pyramid_results["best_result"]["confidence"],
                    "scale_factor": pyramid_results["best_result"].get("scale_factor", 1.0)
                })

            # 添加網格結果
            for result in grid_results:
                all_detections.append({
                    "source": "grid",
                    "landmark_id": result["landmark_id"],
                    "landmark_name": result["landmark_name"],
                    "confidence": result["confidence"],
                    "grid_position": result.get("grid_position", (0, 0))
                })

            # 搜索整張圖像
            full_image_result = self.search_entire_image(image, threshold=adjusted_threshold)
            if full_image_result and full_image_result.get("is_landmark", False):
                all_detections.append({
                    "source": "full_image",
                    "landmark_id": full_image_result["landmark_id"],
                    "landmark_name": full_image_result["landmark_name"],
                    "confidence": full_image_result["confidence"]
                })

            # 按地標ID分組並計算總體置信度
            landmark_groups = {}
            for detection in all_detections:
                landmark_id = detection["landmark_id"]
                if landmark_id not in landmark_groups:
                    landmark_groups[landmark_id] = {
                        "landmark_id": landmark_id,
                        "landmark_name": detection["landmark_name"],
                        "detections": [],
                        "sources": set()
                    }

                landmark_groups[landmark_id]["detections"].append(detection)
                landmark_groups[landmark_id]["sources"].add(detection["source"])

            # 計算每個地標的總體置信度
            for landmark_id, group in landmark_groups.items():
                detections = group["detections"]

                # 基礎置信度是任何來源的最大置信度
                max_confidence = max(d["confidence"] for d in detections)

                # 多來源檢測獎勵
                source_count = len(group["sources"])
                source_bonus = min(0.15, (source_count - 1) * 0.05)

                # 一致性獎勵
                detection_count = len(detections)
                consistency_bonus = min(0.1, (detection_count - 1) * 0.02)

                # 計算最終置信度
                aggregate_confidence = min(1.0, max_confidence + source_bonus + consistency_bonus)

                group["confidence"] = aggregate_confidence
                group["detection_count"] = detection_count
                group["source_count"] = source_count

            # 照信心度排序地標
            sorted_landmarks = sorted(
                landmark_groups.values(),
                key=lambda x: x["confidence"],
                reverse=True
            )

            return {
                "is_landmark_scene": len(sorted_landmarks) > 0,
                "detected_landmarks": sorted_landmarks,
                "viewpoint_info": viewpoint_info,
                "primary_landmark": sorted_landmarks[0] if sorted_landmarks else None
            }

        except Exception as e:
            self.logger.error(f"Error in enhanced_landmark_detection: {e}")
            self.logger.error(traceback.format_exc())
            return {"is_landmark_scene": False, "detected_landmarks": []}
