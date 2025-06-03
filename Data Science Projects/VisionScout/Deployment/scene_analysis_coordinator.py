import logging
import traceback
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from PIL import Image

class SceneAnalysisCoordinator:
    """
    負責整個場景分析流程的協調和控制邏輯，包含主要的分析流程、
    處理無檢測結果的回退邏輯，以及多源分析結果的整合。
    """

    def __init__(self, component_initializer, scene_scoring_engine, landmark_processing_manager,
                 scene_confidence_threshold: float = 0.6):
        """
        初始化場景分析協調器。

        Args:
            component_initializer: 組件初始化器實例
            scene_scoring_engine: 場景評分引擎實例
            landmark_processing_manager: 地標處理管理器實例
            scene_confidence_threshold: 場景置信度閾值
        """
        self.logger = logging.getLogger(__name__)
        self.component_initializer = component_initializer
        self.scene_scoring_engine = scene_scoring_engine
        self.landmark_processing_manager = landmark_processing_manager
        self.scene_confidence_threshold = scene_confidence_threshold

        # 獲取必要的組件和數據
        self.spatial_analyzer = component_initializer.get_component('spatial_analyzer')
        self.descriptor = component_initializer.get_component('descriptor')
        self.scene_describer = component_initializer.get_component('scene_describer')
        self.clip_analyzer = component_initializer.get_component('clip_analyzer')
        self.llm_enhancer = component_initializer.get_component('llm_enhancer')

        self.scene_types = component_initializer.get_data_structure('SCENE_TYPES')

        # 從組件初始化器獲取功能開關狀態
        self.use_clip = component_initializer.use_clip
        self.use_llm = component_initializer.use_llm
        self.enable_landmark = component_initializer.enable_landmark

    def analyze(self, detection_result: Any, lighting_info: Optional[Dict] = None,
                class_confidence_threshold: float = 0.25, scene_confidence_threshold: float = 0.6,
                enable_landmark: bool = True, places365_info: Optional[Dict] = None) -> Dict:
        """
        分析檢測結果以確定場景類型並提供理解。

        Args:
            detection_result: 來自 YOLOv8 或類似系統的檢測結果
            lighting_info: 可選的照明條件分析結果
            class_confidence_threshold: 考慮物體的最小置信度
            scene_confidence_threshold: 確定場景的最小置信度
            enable_landmark: 是否為此次運行啟用地標檢測和識別
            places365_info: 可選的 Places365 場景分類結果

        Returns:
            包含場景分析結果的字典
        """
        current_run_enable_landmark = enable_landmark
        self.logger.info(f"DIAGNOSTIC (SceneAnalyzer.analyze): Called with current_run_enable_landmark={current_run_enable_landmark}")
        self.logger.debug(f"SceneAnalyzer received lighting_info type: {type(lighting_info)}")
        self.logger.debug(f"SceneAnalyzer lighting_info source: {lighting_info.get('source', 'unknown') if isinstance(lighting_info, dict) else 'not_dict'}")

        # 記錄 Places365 資訊
        if places365_info:
            self.logger.info(f"DIAGNOSTIC: Places365 info received - scene: {places365_info.get('scene_label', 'unknown')}, "
                           f"mapped: {places365_info.get('mapped_scene_type', 'unknown')}, "
                           f"confidence: {places365_info.get('confidence', 0.0):.3f}")

        # 同步 enable_landmark 狀態到子組件（為此次分析運行）
        self._sync_landmark_status_to_components(current_run_enable_landmark)

        # 提取和處理原始圖像
        original_image_pil, image_dims_val = self._extract_image_info(detection_result)

        # 處理無 YOLO 檢測結果的情況
        no_yolo_detections = self._check_no_yolo_detections(detection_result)

        if no_yolo_detections:
            return self._handle_no_yolo_detections(
                original_image_pil, image_dims_val, current_run_enable_landmark,
                lighting_info, places365_info
            )

        # 主處理流程（有 YOLO 檢測結果）
        return self._handle_main_analysis_flow(
            detection_result, original_image_pil, image_dims_val,
            class_confidence_threshold, scene_confidence_threshold,
            current_run_enable_landmark, lighting_info, places365_info
        )

    def _sync_landmark_status_to_components(self, current_run_enable_landmark: bool):
        """同步地標狀態到所有相關組件。"""
        # 更新場景評分引擎
        self.scene_scoring_engine.update_enable_landmark_status(current_run_enable_landmark)

        # 更新地標處理管理器
        self.landmark_processing_manager.update_enable_landmark_status(current_run_enable_landmark)

        # 更新其他組件的地標狀態
        for component_name in ['scene_describer', 'clip_analyzer', 'landmark_classifier']:
            component = self.component_initializer.get_component(component_name)
            if component and hasattr(component, 'enable_landmark'):
                component.enable_landmark = current_run_enable_landmark

        # 更新實例狀態
        self.enable_landmark = current_run_enable_landmark

    def _extract_image_info(self, detection_result) -> Tuple[Optional[Image.Image], Optional[Tuple[int, int]]]:
        """從檢測結果中提取圖像信息。"""
        original_image_pil = None
        image_dims_val = None  # 將是 (width, height)

        if (detection_result is not None and hasattr(detection_result, 'orig_img') and
            detection_result.orig_img is not None):
            if isinstance(detection_result.orig_img, np.ndarray):
                try:
                    img_array = detection_result.orig_img
                    if img_array.ndim == 3 and img_array.shape[2] == 4:  # RGBA
                        img_array = img_array[:, :, :3]  # 轉換為 RGB
                    if img_array.ndim == 2:  # 灰度
                        original_image_pil = Image.fromarray(img_array).convert("RGB")
                    else:  # 假設 RGB 或 BGR（如果源是 cv2 BGR，PIL 在 fromarray 時會處理 BGR->RGB，但明確處理更好）
                        original_image_pil = Image.fromarray(img_array)

                    if hasattr(original_image_pil, 'mode') and original_image_pil.mode == 'BGR':  # 明確將 OpenCV 的 BGR 轉換為 PIL 的 RGB
                        original_image_pil = original_image_pil.convert('RGB')

                    image_dims_val = (original_image_pil.width, original_image_pil.height)
                except Exception as e:
                    self.logger.warning(f"Error converting NumPy orig_img to PIL: {e}")
            elif hasattr(detection_result.orig_img, 'size') and callable(getattr(detection_result.orig_img, 'convert', None)):
                original_image_pil = detection_result.orig_img.copy().convert("RGB")  # 確保 RGB
                image_dims_val = original_image_pil.size
            else:
                self.logger.warning(f"detection_result.orig_img (type: {type(detection_result.orig_img)}) is not a recognized NumPy array or PIL Image.")
        else:
            self.logger.warning("detection_result.orig_img not available. Image-based analysis will be limited.")

        return original_image_pil, image_dims_val

    def _check_no_yolo_detections(self, detection_result) -> bool:
        """檢查是否沒有 YOLO 檢測結果。"""
        return (detection_result is None or
                not hasattr(detection_result, 'boxes') or
                not hasattr(detection_result.boxes, 'xyxy') or
                len(detection_result.boxes.xyxy) == 0)

    def _handle_no_yolo_detections(self, original_image_pil, image_dims_val,
                                 current_run_enable_landmark, lighting_info, places365_info) -> Dict:
        """處理無 YOLO 檢測結果的情況。"""
        tried_landmark_detection = False
        landmark_detection_result = None

        # 嘗試地標檢測
        if original_image_pil and self.use_clip and current_run_enable_landmark:
            landmark_detection_result = self._attempt_landmark_detection_no_yolo(
                original_image_pil, image_dims_val, lighting_info
            )
            tried_landmark_detection = True

            if landmark_detection_result:
                return landmark_detection_result

        # 如果地標檢測失敗或未嘗試，使用 CLIP 進行一般場景分析
        if not landmark_detection_result and self.use_clip and original_image_pil:
            clip_fallback_result = self._attempt_clip_fallback_analysis(
                original_image_pil, image_dims_val, current_run_enable_landmark, lighting_info
            )
            if clip_fallback_result:
                return clip_fallback_result

        # 最終回退邏輯
        return self._get_final_fallback_result(places365_info, lighting_info)

    def _attempt_landmark_detection_no_yolo(self, original_image_pil, image_dims_val, lighting_info) -> Optional[Dict]:
        """在無 YOLO 檢測的情況下嘗試地標檢測。"""
        try:
            # 初始化地標分類器（如果需要）
            landmark_classifier = self.component_initializer.get_component('landmark_classifier')
            if not landmark_classifier and self.clip_analyzer:
                if hasattr(self.clip_analyzer, 'get_clip_instance'):
                    try:
                        model, preprocess, device = self.clip_analyzer.get_clip_instance()
                        landmark_classifier = CLIPZeroShotClassifier(device=device)
                        self.landmark_processing_manager.set_landmark_classifier(landmark_classifier)
                        self.logger.info("Initialized landmark classifier with shared CLIP model")
                    except Exception as e:
                        self.logger.warning(f"Could not initialize landmark classifier: {e}")
                        return None

            if landmark_classifier:
                self.logger.info("Attempting landmark detection with no YOLO boxes")
                landmark_results_no_yolo = landmark_classifier.intelligent_landmark_search(
                    original_image_pil, yolo_boxes=None, base_threshold=0.2  # 略微降低閾值，提高靈敏度
                )

                # 確保在無地標場景時返回有效結果
                if landmark_results_no_yolo is None:
                    landmark_results_no_yolo = {"is_landmark_scene": False, "detected_landmarks": []}

                if (landmark_results_no_yolo and landmark_results_no_yolo.get("is_landmark_scene", False)):
                    return self._process_landmark_detection_result(
                        landmark_results_no_yolo, image_dims_val, lighting_info
                    )
        except Exception as e:
            self.logger.error(f"Error in landmark-only detection path (analyze method): {e}")
            traceback.print_exc()

        return None

    def _process_landmark_detection_result(self, landmark_results, image_dims_val, lighting_info) -> Dict:
        """處理地標檢測結果並生成最終輸出。"""
        primary_landmark = landmark_results.get("primary_landmark")

        # 放寬閾值條件，以便捕獲更多潛在地標
        if not primary_landmark or primary_landmark.get("confidence", 0) <= 0.25:
            return None

        detected_objects_from_landmarks_list = []
        w_img, h_img = image_dims_val if image_dims_val else (1, 1)

        for lm_info_item in landmark_results.get("detected_landmarks", []):
            if lm_info_item.get("confidence", 0) > 0.25:  # 降低閾值與上面保持一致
                # 安全獲取 box 值，避免索引錯誤
                box = lm_info_item.get("box", [0, 0, w_img, h_img])
                if len(box) < 4:
                    box = [0, 0, w_img, h_img]

                # 計算中心點和標準化坐標
                center_x, center_y = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
                norm_cx = center_x / w_img if w_img > 0 else 0.5
                norm_cy = center_y / h_img if h_img > 0 else 0.5

                # 決定地標類型
                landmark_type = "architectural"  # 預設類型
                landmark_id = lm_info_item.get("landmark_id", "")

                landmark_classifier = self.component_initializer.get_component('landmark_classifier')
                if (landmark_classifier and hasattr(landmark_classifier, '_determine_landmark_type') and landmark_id):
                    try:
                        landmark_type = landmark_classifier._determine_landmark_type(landmark_id)
                    except Exception as e:
                        self.logger.error(f"Error determining landmark type: {e}")
                else:
                    # 使用簡單的基於 ID 的啟發式方法推斷類型
                    landmark_id_lower = landmark_id.lower() if isinstance(landmark_id, str) else ""
                    if "natural" in landmark_id_lower or any(term in landmark_id_lower for term in ["mountain", "waterfall", "canyon", "lake"]):
                        landmark_type = "natural"
                    elif "monument" in landmark_id_lower or "memorial" in landmark_id_lower or "historical" in landmark_id_lower:
                        landmark_type = "monument"

                # 決定區域位置
                region = "center"  # 預設值
                if self.spatial_analyzer and hasattr(self.spatial_analyzer, '_determine_region'):
                    try:
                        region = self.spatial_analyzer._determine_region(norm_cx, norm_cy)
                    except Exception as e:
                        self.logger.error(f"Error determining region: {e}")

                # 取得並補 location
                loc_lm = lm_info_item.get("location", "")
                if not loc_lm and landmark_id in ALL_LANDMARKS:
                    loc_lm = ALL_LANDMARKS[landmark_id].get("location", "")

                # 創建地標物體
                landmark_obj = {
                    "class_id": lm_info_item.get("landmark_id", f"LM_{lm_info_item.get('landmark_name','unk')}")[:15],
                    "class_name": lm_info_item.get("landmark_name", "Unknown Landmark"),
                    "confidence": lm_info_item.get("confidence", 0.0),
                    "box": box,
                    "center": (center_x, center_y),
                    "normalized_center": (norm_cx, norm_cy),
                    "size": (box[2] - box[0], box[3] - box[1]),
                    "normalized_size": (
                        (box[2] - box[0])/(w_img if w_img>0 else 1),
                        (box[3] - box[1])/(h_img if h_img>0 else 1)
                    ),
                    "area": (box[2] - box[0]) * (box[3] - box[1]),
                    "normalized_area": (
                        (box[2] - box[0]) * (box[3] - box[1])
                    ) / ((w_img*h_img) if w_img*h_img >0 else 1),
                    "is_landmark": True,
                    "landmark_id": landmark_id,
                    "location": loc_lm or "Unknown Location",
                    "region": region,
                    "year_built": lm_info_item.get("year_built", ""),
                    "architectural_style": lm_info_item.get("architectural_style", ""),
                    "significance": lm_info_item.get("significance", ""),
                    "landmark_type": landmark_type
                }
                detected_objects_from_landmarks_list.append(landmark_obj)

        if not detected_objects_from_landmarks_list:
            return None

        # 設定場景類型
        best_scene_val = "tourist_landmark"  # 預設
        if primary_landmark:
            try:
                lm_type = primary_landmark.get("landmark_type", "architectural")
                if lm_type and "natural" in lm_type.lower():
                    best_scene_val = "natural_landmark"
                elif lm_type and ("historical" in lm_type.lower() or "monument" in lm_type.lower()):
                    best_scene_val = "historical_monument"
            except Exception as e:
                self.logger.error(f"Error determining scene type from landmark type: {e}")

        # 確保場景類型有效
        if best_scene_val not in self.scene_types:
            best_scene_val = "tourist_landmark"  # 預設場景類型

        # 設定置信度
        scene_confidence = primary_landmark.get("confidence", 0.0) if primary_landmark else 0.0

        # 生成其他必要的分析結果
        region_analysis = self._generate_region_analysis(detected_objects_from_landmarks_list)

        functional_zones = self._generate_functional_zones(
            detected_objects_from_landmarks_list,
            best_scene_val
        )

        scene_description = self._generate_scene_description(
            best_scene_val, detected_objects_from_landmarks_list, scene_confidence,
            lighting_info, functional_zones, image_dims_val
        )

        enhanced_description = self._enhance_description_with_llm(
            scene_description, best_scene_val, detected_objects_from_landmarks_list,
            scene_confidence, lighting_info, functional_zones, landmark_results, image_dims_val
        )
        possible_activities = self._extract_possible_activities(detected_objects_from_landmarks_list, landmark_results)

        # 準備最終結果
        return {
            "scene_type": best_scene_val,
            "scene_name": self.scene_types.get(best_scene_val, {}).get("name", "Landmark"),
            "confidence": round(float(scene_confidence), 4),
            "description": scene_description,
            "enhanced_description": enhanced_description,
            "objects_present": detected_objects_from_landmarks_list,
            "object_count": len(detected_objects_from_landmarks_list),
            "regions": region_analysis,
            "possible_activities": possible_activities,
            "functional_zones": functional_zones,
            "detected_landmarks": [lm for lm in detected_objects_from_landmarks_list if lm.get("is_landmark", False)],
            "primary_landmark": primary_landmark,
            "lighting_conditions": lighting_info or {"time_of_day": "unknown", "confidence": 0.0}
        }


    def _attempt_clip_fallback_analysis(self, original_image_pil, image_dims_val,
                                      current_run_enable_landmark, lighting_info) -> Optional[Dict]:
        """嘗試使用 CLIP 進行一般場景分析。"""
        try:
            clip_analysis_val = None
            if self.clip_analyzer and hasattr(self.clip_analyzer, 'analyze_image'):
                try:
                    clip_analysis_val = self.clip_analyzer.analyze_image(
                        original_image_pil,
                        enable_landmark=current_run_enable_landmark
                    )
                except Exception as e:
                    self.logger.error(f"Error in CLIP analysis: {e}")

            scene_type_llm = "llm_inferred_no_yolo"
            confidence_llm = 0.0

            if clip_analysis_val and isinstance(clip_analysis_val, dict):
                top_scene = clip_analysis_val.get("top_scene")
                if top_scene and isinstance(top_scene, tuple) and len(top_scene) >= 2:
                    confidence_llm = top_scene[1]
                    if isinstance(top_scene[0], str):
                        scene_type_llm = top_scene[0]

            desc_llm = "Primary object detection did not yield results. This description is based on overall image context."

            w_llm, h_llm = image_dims_val if image_dims_val else (1, 1)
            enhanced_desc_llm = self._enhance_no_detection_description(
                desc_llm, scene_type_llm, confidence_llm, lighting_info,
                clip_analysis_val, current_run_enable_landmark, w_llm, h_llm
            )

            # 安全類型轉換
            try:
                confidence_float = float(confidence_llm)
            except (ValueError, TypeError):
                confidence_float = 0.0

            # 確保增強描述不為空
            if not enhanced_desc_llm or not isinstance(enhanced_desc_llm, str):
                enhanced_desc_llm = desc_llm

            # 返回結果
            return {
                "scene_type": scene_type_llm,
                "confidence": round(confidence_float, 4),
                "description": desc_llm,
                "enhanced_description": enhanced_desc_llm,
                "objects_present": [],
                "object_count": 0,
                "regions": {},
                "possible_activities": [],
                "safety_concerns": [],
                "lighting_conditions": lighting_info or {"time_of_day": "unknown", "confidence": 0.0}
            }
        except Exception as e:
            self.logger.error(f"Error in CLIP no-detection fallback (analyze method): {e}")
            traceback.print_exc()
            return None

    def _get_final_fallback_result(self, places365_info, lighting_info) -> Dict:
        """獲取最終的回退結果。"""
        # 檢查 Places365 是否提供有用的場景信息（即使沒有 YOLO 檢測）
        fallback_scene_type = "unknown"
        fallback_confidence = 0.0
        fallback_description = "No objects were detected in the image, and contextual analysis could not be performed or failed."

        if places365_info and places365_info.get('confidence', 0) > 0.3:
            fallback_scene_type = places365_info.get('mapped_scene_type', 'unknown')
            fallback_confidence = places365_info.get('confidence', 0.0)
            fallback_description = f"Scene appears to be {places365_info.get('scene_label', 'an unidentified location')} based on overall visual context."

        return {
            "scene_type": fallback_scene_type,
            "confidence": fallback_confidence,
            "description": fallback_description,
            "enhanced_description": "The image analysis system could not detect any recognizable objects or landmarks in this image.",
            "objects_present": [],
            "object_count": 0,
            "regions": {},
            "possible_activities": [],
            "safety_concerns": [],
            "lighting_conditions": lighting_info or {"time_of_day": "unknown", "confidence": 0.0}
        }

    def _handle_main_analysis_flow(self, detection_result, original_image_pil, image_dims_val,
                                 class_confidence_threshold, scene_confidence_threshold,
                                 current_run_enable_landmark, lighting_info, places365_info) -> Dict:
        """處理主要的分析流程（有 YOLO 檢測結果）。"""
        # 更新類別名稱映射
        if hasattr(detection_result, 'names'):
            if hasattr(self.spatial_analyzer, 'class_names'):
                self.spatial_analyzer.class_names = detection_result.names

        # 提取檢測到的物體
        detected_objects_main = self.spatial_analyzer._extract_detected_objects(
            detection_result,
            confidence_threshold=class_confidence_threshold
        )

        if not detected_objects_main:
            return {
                "scene_type": "unknown", "confidence": 0.0,
                "description": "No objects detected with sufficient confidence by the primary vision system.",
                "objects_present": [], "object_count": 0, "regions": {}, "possible_activities": [],
                "safety_concerns": [], "lighting_conditions": lighting_info or {"time_of_day": "unknown", "confidence": 0.0}
            }

        # 空間分析
        region_analysis_val = self.spatial_analyzer._analyze_regions(detected_objects_main)

        # 地標處理和整合
        landmark_objects_identified = []
        landmark_specific_activities = []
        final_landmark_info = {}

        if self.use_clip and current_run_enable_landmark:
            detected_objects_main, landmark_objects_identified = self.landmark_processing_manager.process_unknown_objects(
                detection_result, detected_objects_main, self.clip_analyzer
            )

            if landmark_objects_identified:
                landmark_specific_activities = self.landmark_processing_manager.extract_landmark_specific_activities(
                    landmark_objects_identified
                )
                final_landmark_info = {
                    "detected_landmarks": landmark_objects_identified,
                    "primary_landmark": max(landmark_objects_identified, key=lambda x: x.get("confidence", 0.0), default=None),
                    "detailed_landmarks": landmark_objects_identified
                }

        # 如果當前運行禁用地標檢測，清理地標物體
        if not current_run_enable_landmark:
            detected_objects_main = [obj for obj in detected_objects_main if not obj.get("is_landmark", False)]
            final_landmark_info = {}

        # 計算場景分數並進行融合
        yolo_scene_scores = self.scene_scoring_engine.compute_scene_scores(
            detected_objects_main, spatial_analysis_results=region_analysis_val
        )

        clip_scene_scores = {}
        clip_analysis_results = None
        if self.use_clip and original_image_pil is not None:
            clip_analysis_results, clip_scene_scores = self._perform_clip_analysis(
                original_image_pil, current_run_enable_landmark, lighting_info
            )

        # 融合場景分數
        yolo_only_objects = [obj for obj in detected_objects_main if not obj.get("is_landmark")]
        num_yolo_detections = len(yolo_only_objects)
        avg_yolo_confidence = (sum(obj.get('confidence', 0.0) for obj in yolo_only_objects) / num_yolo_detections
                              if num_yolo_detections > 0 else 0.0)

        scene_scores_fused = self.scene_scoring_engine.fuse_scene_scores(
            yolo_scene_scores, clip_scene_scores,
            num_yolo_detections=num_yolo_detections,
            avg_yolo_confidence=avg_yolo_confidence,
            lighting_info=lighting_info,
            places365_info=places365_info
        )

        # 確定最終場景類型
        final_best_scene, final_scene_confidence = self.scene_scoring_engine.determine_scene_type(scene_scores_fused)

        # 處理禁用地標檢測時的替代場景類型
        if (not current_run_enable_landmark and
            final_best_scene in ["tourist_landmark", "natural_landmark", "historical_monument"]):
            alt_scene_type = self.landmark_processing_manager.get_alternative_scene_type(
                final_best_scene, detected_objects_main, scene_scores_fused
            )
            final_best_scene = alt_scene_type
            final_scene_confidence = scene_scores_fused.get(alt_scene_type, 0.6)

        # 生成最終的描述性內容
        final_result = self._generate_final_result(
            final_best_scene, final_scene_confidence, detected_objects_main,
            landmark_specific_activities, landmark_objects_identified, final_landmark_info,
            region_analysis_val, lighting_info, scene_scores_fused, current_run_enable_landmark,
            clip_analysis_results, image_dims_val, scene_confidence_threshold
        )

        return final_result

    def _perform_clip_analysis(self, original_image_pil, current_run_enable_landmark, lighting_info) -> Tuple[Optional[Dict], Dict]:
        """執行 CLIP 分析。"""
        clip_analysis_results = None
        clip_scene_scores = {}

        try:
            clip_analysis_results = self.clip_analyzer.analyze_image(
                original_image_pil,
                enable_landmark=current_run_enable_landmark,
                exclude_categories=["landmark", "tourist", "monument", "tower", "attraction", "scenic", "historical", "famous"] if not current_run_enable_landmark else None
            )

            if isinstance(clip_analysis_results, dict):
                clip_scene_scores = clip_analysis_results.get("scene_scores", {})

                # 如果禁用地標檢測，再次過濾
                if not current_run_enable_landmark:
                    clip_scene_scores = {k: v for k, v in clip_scene_scores.items()
                                       if not any(kw in k.lower() for kw in ["landmark", "monument", "tourist"])}
                    if "cultural_analysis" in clip_analysis_results:
                        del clip_analysis_results["cultural_analysis"]
                    if ("top_scene" in clip_analysis_results and
                        any(term in clip_analysis_results.get("top_scene", ["unknown", 0.0])[0].lower()
                            for term in ["landmark", "monument", "tourist"])):
                        non_lm_cs = sorted([item for item in clip_scene_scores.items() if item[1] > 0],
                                         key=lambda x: x[1], reverse=True)
                        clip_analysis_results["top_scene"] = non_lm_cs[0] if non_lm_cs else ("unknown", 0.0)

                # 處理照明信息回退
                if (not lighting_info and "lighting_condition" in clip_analysis_results):
                    lt, lc = clip_analysis_results.get("lighting_condition", ("unknown", 0.0))
                    lighting_info = {"time_of_day": lt, "confidence": lc, "source": "CLIP_fallback"}
        except Exception as e:
            self.logger.error(f"Error in main CLIP analysis for YOLO path (analyze method): {e}")

        return clip_analysis_results, clip_scene_scores

    def _generate_final_result(self, final_best_scene, final_scene_confidence, detected_objects_main,
                             landmark_specific_activities, landmark_objects_identified, final_landmark_info,
                             region_analysis_val, lighting_info, scene_scores_fused, current_run_enable_landmark,
                             clip_analysis_results, image_dims_val, scene_confidence_threshold) -> Dict:
        """生成最終的分析結果。"""
        # 生成最終的描述性內容（活動、安全、區域）
        final_activities = []

        # 通用活動推斷
        generic_activities = []
        if self.descriptor and hasattr(self.descriptor, '_infer_possible_activities'):
            generic_activities = self.descriptor._infer_possible_activities(
                final_best_scene, detected_objects_main,
                enable_landmark=current_run_enable_landmark, scene_scores=scene_scores_fused
            )

        # 優先處理策略：使用特定地標活動，不足時才從通用活動補充
        if landmark_specific_activities:
            # 如果有特定活動，優先保留，去除與特定活動重複的通用活動
            unique_generic_activities = [act for act in generic_activities if act not in landmark_specific_activities]

            # 如果特定活動少於3個，從通用活動中補充
            if len(landmark_specific_activities) < 3:
                # 補充通用活動但總數不超過7個
                supplement_count = min(3 - len(landmark_specific_activities), len(unique_generic_activities))
                if supplement_count > 0:
                    final_activities.extend(unique_generic_activities[:supplement_count])
        else:
            # 若無特定活動，則使用所有通用活動
            final_activities.extend(generic_activities)

        # 去重並排序，但確保特定地標活動保持在前面
        final_activities_set = set(final_activities)
        final_activities = []

        # 先加入特定地標活動（按原順序）
        for activity in landmark_specific_activities:
            if activity in final_activities_set:
                final_activities.append(activity)
                final_activities_set.remove(activity)

        # 再加入通用活動（按字母排序）
        final_activities.extend(sorted(list(final_activities_set)))

        # 安全問題識別
        final_safety_concerns = []
        if self.descriptor and hasattr(self.descriptor, '_identify_safety_concerns'):
            final_safety_concerns = self.descriptor._identify_safety_concerns(detected_objects_main, final_best_scene)

        # 功能區域識別
        final_functional_zones = {}
        if self.spatial_analyzer and hasattr(self.spatial_analyzer, '_identify_functional_zones'):
            general_zones = self.spatial_analyzer._identify_functional_zones(detected_objects_main, final_best_scene)
            final_functional_zones.update(general_zones)

        # 地標相關的功能區域
        if landmark_objects_identified and self.spatial_analyzer and hasattr(self.spatial_analyzer, '_identify_landmark_zones'):
            landmark_zones = self.spatial_analyzer._identify_landmark_zones(landmark_objects_identified)
            final_functional_zones.update(landmark_zones)

        # 如果當前運行禁用地標檢測，過濾相關內容
        if not current_run_enable_landmark:
            final_functional_zones = {
                        str(k): v
                        for k, v in final_functional_zones.items()
                        if (not str(k).isdigit())
                        and (not any(kw in str(k).lower() for kw in ["landmark", "monument", "viewing", "tourist"]))
                    }


            current_activities_temp = [act for act in final_activities
                                     if not any(kw in act.lower() for kw in ["sightsee", "photograph", "tour", "histor", "landmark", "monument", "cultur"])]
            final_activities = current_activities_temp
            if not final_activities and self.descriptor and hasattr(self.descriptor, '_infer_possible_activities'):
                final_activities = self.descriptor._infer_possible_activities("generic_street_view", detected_objects_main, enable_landmark=False)

        # 創建淨化的光線資訊，避免不合理的時間描述
        lighting_info_clean = None
        if lighting_info:
            lighting_info_clean = {
                "is_indoor": lighting_info.get("is_indoor"),
                "confidence": lighting_info.get("confidence", 0.0),
                "time_of_day": lighting_info.get("time_of_day", "unknown")
            }

        # 生成場景描述
        base_scene_description = self._generate_scene_description(
            final_best_scene, detected_objects_main, final_scene_confidence,
            lighting_info_clean, final_functional_zones, image_dims_val
        )

        # 清理地標引用（如果禁用地標檢測）
        if not current_run_enable_landmark:
            base_scene_description = self.landmark_processing_manager.remove_landmark_references(base_scene_description)

        # LLM 增強
        enhanced_final_description = self._enhance_final_description(
            base_scene_description, final_best_scene, final_scene_confidence, detected_objects_main,
            final_functional_zones, final_activities, final_safety_concerns, lighting_info,
            clip_analysis_results, current_run_enable_landmark, image_dims_val, final_landmark_info
        )

        # 清理增強描述的地標引用
        if not current_run_enable_landmark:
            enhanced_final_description = self.landmark_processing_manager.remove_landmark_references(enhanced_final_description)

        # 構建最終輸出字典
        output_result = {
            "scene_type": final_best_scene if final_scene_confidence >= scene_confidence_threshold else "unknown",
            "scene_name": (self.scene_types.get(final_best_scene, {}).get("name", "Unknown Scene")
                          if final_scene_confidence >= scene_confidence_threshold else "Unknown Scene"),
            "confidence": round(float(final_scene_confidence), 4),
            "description": base_scene_description,
            "enhanced_description": enhanced_final_description,
            "objects_present": [{"class_id": obj.get("class_id", -1),
                               "class_name": obj.get("class_name", "unknown"),
                               "confidence": round(float(obj.get("confidence", 0.0)), 4)}
                              for obj in detected_objects_main],
            "object_count": len(detected_objects_main),
            "regions": region_analysis_val,
            "possible_activities": final_activities,
            "safety_concerns": final_safety_concerns,
            "functional_zones": final_functional_zones,
            "lighting_conditions": lighting_info if lighting_info else {"time_of_day": "unknown", "confidence": 0.0, "source": "default"}
        }

        # 添加替代場景
        if self.descriptor and hasattr(self.descriptor, '_get_alternative_scenes'):
            output_result["alternative_scenes"] = self.descriptor._get_alternative_scenes(
                scene_scores_fused, scene_confidence_threshold, top_k=2
            )

        # 添加地標相關信息
        if current_run_enable_landmark and final_landmark_info and final_landmark_info.get("detected_landmarks"):
            output_result.update(final_landmark_info)
            if final_best_scene in ["tourist_landmark", "natural_landmark", "historical_monument"]:
                output_result["scene_source"] = "landmark_detection"
        elif not current_run_enable_landmark:
            for key_rm in ["detected_landmarks", "primary_landmark", "detailed_landmarks", "scene_source"]:
                if key_rm in output_result:
                    del output_result[key_rm]

        # 添加 CLIP 分析結果
        if clip_analysis_results and isinstance(clip_analysis_results, dict) and "error" not in clip_analysis_results:
            top_scene_clip = clip_analysis_results.get("top_scene", ("unknown", 0.0))
            output_result["clip_analysis"] = {
                "top_scene": (top_scene_clip[0], round(float(top_scene_clip[1]), 4)),
                "cultural_analysis": clip_analysis_results.get("cultural_analysis", {}) if current_run_enable_landmark else {}
            }

        return output_result

    # 輔助方法
    def _generate_region_analysis(self, detected_objects):
        """生成區域分析結果。"""
        if self.spatial_analyzer and hasattr(self.spatial_analyzer, '_analyze_regions'):
            try:
                return self.spatial_analyzer._analyze_regions(detected_objects)
            except Exception as e:
                self.logger.error(f"Error analyzing regions: {e}")
        return {}

    def _generate_functional_zones(self, detected_objects, scene_type):
        """
        生成功能區域。
        由於原本直接呼叫 _identify_landmark_zones，導致非地標場景必定回 {}。
        這裡改為呼叫 _identify_functional_zones，並帶入 scene_type。
        """
        try:
            # 如果 spatial_analyzer 可以識別 functional zones，就調用它
            if self.spatial_analyzer and hasattr(self.spatial_analyzer, '_identify_functional_zones'):
                return self.spatial_analyzer._identify_functional_zones(detected_objects, scene_type)
        except Exception as e:
            self.logger.error(f"Error identifying functional zones: {e}")
            self.logger.error(traceback.format_exc())
        return {}


    def _generate_scene_description(self, scene_type, detected_objects, confidence,
                                  lighting_info, functional_zones, image_dims):
        """生成場景描述。"""
        if self.scene_describer and hasattr(self.scene_describer, 'generate_description'):
            try:
                for obj in detected_objects:
                    if obj.get("is_landmark"):
                        loc_obj = obj.get("location", "")
                        lm_id_obj = obj.get("landmark_id")
                        if (not loc_obj) and lm_id_obj and lm_id_obj in ALL_LANDMARKS:
                            obj["location"] = ALL_LANDMARKS[lm_id_obj].get("location", "")

                return self.scene_describer.generate_description(
                    scene_type=scene_type,
                    detected_objects=detected_objects,
                    confidence=confidence,
                    lighting_info=lighting_info,
                    functional_zones=list(functional_zones.keys()) if functional_zones else [],
                    enable_landmark=self.enable_landmark,
                    scene_scores={scene_type: confidence},
                    spatial_analysis={},
                    image_dimensions=image_dims
                )
            except Exception as e:
                self.logger.error(f"Error generating scene description: {e}")
        return f"A {scene_type} scene."

    def _enhance_description_with_llm(self, scene_description, scene_type, detected_objects,
                                    confidence, lighting_info, functional_zones, landmark_results, image_dims):
        """使用 LLM 增強描述。"""
        if not self.use_llm or not self.llm_enhancer:
            return scene_description

        try:
            prominent_objects_detail = ""
            if self.scene_describer and hasattr(self.scene_describer, 'format_object_list_for_description'):
                try:
                    prominent_objects_detail = self.scene_describer.format_object_list_for_description(
                        detected_objects[:min(1, len(detected_objects))]
                    )
                except Exception as e:
                    self.logger.error(f"Error formatting object list: {e}")

            w_img, h_img = image_dims if image_dims else (1, 1)
            scene_data_llm = {
                "original_description": scene_description,
                "scene_type": scene_type,
                "scene_name": self.scene_types.get(scene_type, {}).get("name", "Landmark"),
                "detected_objects": detected_objects,
                "object_list": "landmark",
                "confidence": confidence,
                "lighting_info": lighting_info,
                "functional_zones": functional_zones,
                "clip_analysis": landmark_results.get("clip_analysis_on_full_image", {}),
                "enable_landmark": True,
                "image_width": w_img,
                "image_height": h_img,
                "prominent_objects_detail": prominent_objects_detail
            }

            return self.llm_enhancer.enhance_description(scene_data_llm)
        except Exception as e:
            self.logger.error(f"Error enhancing description with LLM: {e}")
            traceback.print_exc()
            return scene_description

    def _enhance_no_detection_description(self, desc, scene_type, confidence, lighting_info,
                                        clip_analysis, enable_landmark, width, height):
        """增強無檢測結果的描述。"""
        if not self.use_llm or not self.llm_enhancer:
            return desc

        try:
            clip_analysis_safe = {}
            if isinstance(clip_analysis, dict):
                clip_analysis_safe = clip_analysis

            scene_data_llm = {
                "original_description": desc,
                "scene_type": scene_type,
                "scene_name": "Contextually Inferred (No Detections)",
                "detected_objects": [],
                "object_list": "general ambiance",
                "confidence": confidence,
                "lighting_info": lighting_info or {"time_of_day": "unknown", "confidence": 0.0},
                "clip_analysis": clip_analysis_safe,
                "enable_landmark": enable_landmark,
                "image_width": width,
                "image_height": height,
                "prominent_objects_detail": "the overall visual context"
            }

            if hasattr(self.llm_enhancer, 'enhance_description'):
                try:
                    enhanced = self.llm_enhancer.enhance_description(scene_data_llm)
                    if enhanced and len(enhanced.strip()) >= 20:
                        return enhanced
                except Exception as e:
                    self.logger.error(f"Error in enhance_description: {e}")

            if hasattr(self.llm_enhancer, 'handle_no_detection'):
                try:
                    return self.llm_enhancer.handle_no_detection(clip_analysis_safe)
                except Exception as e:
                    self.logger.error(f"Error in handle_no_detection: {e}")
        except Exception as e:
            self.logger.error(f"Error preparing data for LLM enhancement: {e}")
            traceback.print_exc()

        return desc

    def _extract_possible_activities(self, detected_objects, landmark_results):
        """提取可能的活動。"""
        possible_activities = ["Sightseeing"]

        # 檢查是否有主要地標活動從 CLIP 分析結果中獲取
        primary_landmark_activities = landmark_results.get("primary_landmark_activities", [])

        if primary_landmark_activities:
            self.logger.info(f"Using {len(primary_landmark_activities)} landmark-specific activities")
            possible_activities = primary_landmark_activities
        else:
            # 從檢測到的地標中提取特定活動
            landmark_specific_activities = self.landmark_processing_manager.extract_landmark_specific_activities(detected_objects)

            if landmark_specific_activities:
                possible_activities = list(set(landmark_specific_activities))  # 去重
                self.logger.info(f"Extracted {len(possible_activities)} activities from landmark data")
            else:
                # 回退到通用活動推斷
                if self.descriptor and hasattr(self.descriptor, '_infer_possible_activities'):
                    try:
                        possible_activities = self.descriptor._infer_possible_activities(
                            "tourist_landmark",
                            detected_objects,
                            enable_landmark=True,
                            scene_scores={"tourist_landmark": 0.8}
                        )
                    except Exception as e:
                        self.logger.error(f"Error inferring possible activities: {e}")

        return possible_activities

    def _enhance_final_description(self, base_description, scene_type, scene_confidence, detected_objects,
                                 functional_zones, activities, safety_concerns, lighting_info,
                                 clip_analysis_results, enable_landmark, image_dims, landmark_info):
        """增強最終描述。"""
        if not self.use_llm or not self.llm_enhancer:
            return base_description

        try:
            obj_list_for_llm = ", ".join(sorted(list(set(
                obj["class_name"] for obj in detected_objects
                if obj.get("confidence", 0) > 0.4 and not obj.get("is_landmark")
            ))))

            if not obj_list_for_llm and enable_landmark and landmark_info.get("primary_landmark"):
                obj_list_for_llm = landmark_info["primary_landmark"].get("class_name", "a prominent feature")
            elif not obj_list_for_llm:
                obj_list_for_llm = "various visual elements"

            # 生成物體統計信息
            object_statistics = {}
            for obj in detected_objects:
                class_name = obj.get("class_name", "unknown")
                if class_name not in object_statistics:
                    object_statistics[class_name] = {
                        "count": 0,
                        "avg_confidence": 0.0,
                        "max_confidence": 0.0,
                        "instances": []
                    }

                stats = object_statistics[class_name]
                stats["count"] += 1
                stats["instances"].append(obj)
                stats["max_confidence"] = max(stats["max_confidence"], obj.get("confidence", 0.0))

            # 計算平均信心度
            for class_name, stats in object_statistics.items():
                if stats["count"] > 0:
                    total_conf = sum(inst.get("confidence", 0.0) for inst in stats["instances"])
                    stats["avg_confidence"] = total_conf / stats["count"]

            llm_scene_data = {
                "original_description": base_description,
                "scene_type": scene_type,
                "scene_name": self.scene_types.get(scene_type, {}).get("name", "Unknown Scene"),
                "detected_objects": detected_objects,
                "object_list": obj_list_for_llm,
                "object_statistics": object_statistics,
                "confidence": scene_confidence,
                "lighting_info": lighting_info,
                "functional_zones": functional_zones,
                "activities": activities,
                "safety_concerns": safety_concerns,
                "clip_analysis": clip_analysis_results if isinstance(clip_analysis_results, dict) else None,
                "enable_landmark": enable_landmark,
                "image_width": image_dims[0] if image_dims else None,
                "image_height": image_dims[1] if image_dims else None,
                "prominent_objects_detail": ""
            }

            # 添加顯著物體詳細信息
            if self.scene_describer and hasattr(self.scene_describer, 'get_prominent_objects') and hasattr(self.scene_describer, 'format_object_list_for_description'):
                try:
                    prominent_objects = self.scene_describer.get_prominent_objects(
                        detected_objects, min_prominence_score=0.1, max_categories_to_return=3, max_total_objects=7
                    )
                    llm_scene_data["prominent_objects_detail"] = self.scene_describer.format_object_list_for_description(prominent_objects)
                except Exception as e:
                    self.logger.error(f"Error getting prominent objects: {e}")

            if enable_landmark and landmark_info.get("primary_landmark"):
                llm_scene_data["primary_landmark_info"] = landmark_info["primary_landmark"]

            return self.llm_enhancer.enhance_description(llm_scene_data)
        except Exception as e:
            self.logger.error(f"Error in LLM Enhancement in main flow (analyze method): {e}")
            return base_description
