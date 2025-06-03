import os
import logging
import numpy as np
import torch
import cv2
from PIL import Image
import tempfile
import uuid
from typing import Dict, List, Any, Optional, Tuple

from detection_model import DetectionModel
from color_mapper import ColorMapper
from visualization_helper import VisualizationHelper
from evaluation_metrics import EvaluationMetrics
from lighting_analyzer import LightingAnalyzer
from scene_analyzer import SceneAnalyzer
from places365_model import Places365Model

class ImageProcessor:
    """
    Class for handling image processing and object detection operations
    Separates processing logic from UI components
    """

    def __init__(self, use_llm=True, llm_model_path=None, enable_places365=True, places365_model_name='resnet50_places365'):
        """Initialize the image processor with required components"""
        print(f"Initializing ImageProcessor with use_llm={use_llm}, enable_places365={enable_places365}")
        self.logger = logging.getLogger(__name__)
        try:
            # Initialize basic components first
            self.use_llm = use_llm
            self.llm_model_path = llm_model_path
            self.enable_places365 = enable_places365
            self.model_instances = {}

            self.coco_class_names = {
                0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane',
                5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light',
                10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench',
                14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow',
                20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack',
                25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee',
                30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat',
                35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket',
                39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife',
                44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich',
                49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza',
                54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant',
                59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop',
                64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone',
                68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator',
                73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear',
                78: 'hair drier', 79: 'toothbrush'
            }

            # Initialize ColorMapper
            self.color_mapper = ColorMapper()
            print("ColorMapper initialized successfully")

            # Initialize LightingAnalyzer
            self.lighting_analyzer = LightingAnalyzer()
            print("LightingAnalyzer initialized successfully")

            # Initialize Places365 model if enabled
            self.places365_model = None
            if self.enable_places365:
                try:
                    self.places365_model = Places365Model(
                        model_name=places365_model_name,
                        device=None
                    )
                    print(f"Places365 model initialized successfully with {places365_model_name}")
                except Exception as e:
                    print(f"Warning: Failed to initialize Places365 model: {e}")
                    print("Continuing without Places365 analysis")
                    self.enable_places365 = False
                    self.places365_model = None

            # Initialize SceneAnalyzer with error handling
            self.scene_analyzer = None
            self.class_names = self.coco_class_names

            try:
                # Initialize SceneAnalyzer without class_names (will be set later)
                self.scene_analyzer = SceneAnalyzer(
                    class_names=self.coco_class_names,
                    use_llm=self.use_llm,
                    use_clip=True,
                    enable_landmark=True,
                    llm_model_path=self.llm_model_path
                )
                print("SceneAnalyzer initialized successfully")

                # Verify critical components
                if self.scene_analyzer is not None:
                    print(f"SceneAnalyzer status - spatial_analyzer: {hasattr(self.scene_analyzer, 'spatial_analyzer')}, "
                        f"descriptor: {hasattr(self.scene_analyzer, 'descriptor')}, "
                        f"scene_describer: {hasattr(self.scene_analyzer, 'scene_describer')}")
                else:
                    print("WARNING: scene_analyzer is None after initialization")

            except Exception as e:
                print(f"Error initializing SceneAnalyzer: {e}")
                import traceback
                traceback.print_exc()
                self.scene_analyzer = None

            print("ImageProcessor initialization completed successfully")

        except Exception as e:
            print(f"Critical error during ImageProcessor initialization: {e}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Failed to initialize ImageProcessor: {str(e)}")

    def get_model_instance(self, model_name: str, confidence: float = 0.25, iou: float = 0.25) -> DetectionModel:
        """
        Get or create a model instance based on model name

        Args:
            model_name: Name of the model to use
            confidence: Confidence threshold for detection
            iou: IoU threshold for non-maximum suppression

        Returns:
            DetectionModel instance
        """
        if model_name not in self.model_instances:
            print(f"Creating new model instance for {model_name}")
            self.model_instances[model_name] = DetectionModel(
                model_name=model_name,
                confidence=confidence,
                iou=iou
            )
        else:
            print(f"Using existing model instance for {model_name}")
            self.model_instances[model_name].confidence = confidence

        return self.model_instances[model_name]

    def analyze_scene(self, detection_result: Any, lighting_info: Optional[Dict] = None, enable_landmark=True, places365_info=None) -> Dict:
        """
        Perform scene analysis on detection results

        Args:
            detection_result: Object detection result from YOLOv8
            lighting_info: Lighting condition analysis results (optional)
            enable_landmark: Whether to enable landmark detection
            places365_info: Places365 analysis results (optional)

        Returns:
            Dictionary containing scene analysis results
        """
        print(f"DEBUG: analyze_scene received enable_landmark={enable_landmark}")
        try:
            # Check if detection_result has valid names
            class_names = getattr(detection_result, 'names', None) if detection_result else None

            # Initialize or reinitialize scene analyzer if needed
            if self.scene_analyzer is None:
                print("Scene analyzer not initialized, creating new instance")
                self.scene_analyzer = SceneAnalyzer(
                    class_names=class_names,
                    use_llm=self.use_llm,
                    use_clip=True,
                    enable_landmark=enable_landmark,
                    llm_model_path=self.llm_model_path
                )

                if self.scene_analyzer is None:
                    raise ValueError("Failed to create SceneAnalyzer instance")
            else:
                # Update existing scene analyzer settings
                self.scene_analyzer.enable_landmark = enable_landmark

                # Update class names if available and different
                if class_names and self.scene_analyzer.class_names != class_names:
                    self.scene_analyzer.class_names = class_names
                    if hasattr(self.scene_analyzer, 'spatial_analyzer') and self.scene_analyzer.spatial_analyzer:
                        self.scene_analyzer.spatial_analyzer.class_names = class_names

                # Update landmark detection settings in child components
                if hasattr(self.scene_analyzer, 'spatial_analyzer') and self.scene_analyzer.spatial_analyzer:
                    self.scene_analyzer.spatial_analyzer.enable_landmark = enable_landmark

            # Perform scene analysis with lighting info and Places365 context
            scene_analysis = self.scene_analyzer.analyze(
                detection_result=detection_result,
                lighting_info=lighting_info,
                class_confidence_threshold=0.35,
                scene_confidence_threshold=0.6,
                enable_landmark=enable_landmark,
                places365_info=places365_info
            )

            return scene_analysis

        except Exception as e:
            print(f"Error in scene analysis: {str(e)}")
            import traceback
            traceback.print_exc()

            # Return a valid default result
            return {
                "scene_type": "unknown",
                "confidence": 0.0,
                "description": f"Error during scene analysis: {str(e)}",
                "enhanced_description": "Scene analysis could not be completed due to an error.",
                "objects_present": [],
                "object_count": 0,
                "regions": {},
                "possible_activities": [],
                "safety_concerns": [],
                "lighting_conditions": lighting_info or {"time_of_day": "unknown", "confidence": 0.0}
            }

    def analyze_lighting_conditions(self, image, places365_info: Optional[Dict] = None):
        """
        分析光照條件並考慮 Places365 場景資訊。

        Args:
            image: 輸入圖像
            places365_info: Places365 場景分析結果，用於覆蓋邏輯

        Returns:
            Dict: 光照分析結果
        """
        return self.lighting_analyzer.analyze(image, places365_info=places365_info)

    def analyze_places365_scene(self, image):
        """
        Analyze scene using Places365 model.

        Args:
            image: Input image (PIL Image)

        Returns:
            Dict: Places365 analysis results or None if disabled/failed
        """
        if not self.enable_places365 or self.places365_model is None:
            return None

        try:
            if not isinstance(image, Image.Image):
                if isinstance(image, np.ndarray):
                    image = Image.fromarray(image)
                else:
                    print(f"Warning: Cannot process image of type {type(image)} for Places365")
                    return None

            places365_result = self.places365_model.predict(image)

            if places365_result and places365_result.get('confidence', 0) > 0.1:
                print(f"Places365 detected: {places365_result['scene_label']} "
                    f"(mapped: {places365_result['mapped_scene_type']}) "
                    f"confidence: {places365_result['confidence']:.3f}")
                return places365_result
            else:
                print("Places365 analysis failed or low confidence")
                return None

        except Exception as e:
            print(f"Error in Places365 analysis: {str(e)}")
            return None

    def process_image(self, image: Any, model_name: str, confidence_threshold: float, filter_classes: Optional[List[int]] = None,  enable_landmark: bool = True) -> Tuple[Any, str, Dict]:
        """
        Process an image for object detection and scene analysis.
        Args:
            image: Input image (numpy array or PIL Image).
            model_name: Name of the model to use.
            confidence_threshold: Confidence threshold for detection.
            filter_classes: Optional list of classes to filter results.
            enable_landmark: Whether to enable landmark detection for this run.
        Returns:
            Tuple of (result_image_pil, result_text, stats_data_with_scene_analysis).
        """
        model_instance = self.get_model_instance(model_name, confidence_threshold)
        if model_instance is None:
            return None, f"Failed to load model: {model_name}. Please check model configuration.", {}

        result = None
        stats_data = {}
        temp_path = None
        pil_image_for_processing = None # Use this to store the consistently processed PIL image

        try:
            if isinstance(image, np.ndarray):
                if image.ndim == 3 and image.shape[2] == 3: # RGB or BGR
                    # Assuming BGR from OpenCV, convert to RGB for PIL standard
                    image_rgb_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    pil_image_for_processing = Image.fromarray(image_rgb_np)
                elif image.ndim == 3 and image.shape[2] == 4: # RGBA or BGRA
                    image_rgba_np = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA) # Ensure RGBA
                    pil_image_for_processing = Image.fromarray(image_rgba_np).convert("RGB") # Convert to RGB
                elif image.ndim == 2: # Grayscale
                    pil_image_for_processing = Image.fromarray(image).convert("RGB")
                else:
                    pil_image_for_processing = Image.fromarray(image) # Hope for the best
            elif isinstance(image, Image.Image):
                pil_image_for_processing = image.copy() # Use a copy
            elif image is None:
                return None, "No image provided. Please upload an image.", {}
            else:
                return None, f"Unsupported image type: {type(image)}. Please provide a NumPy array or PIL Image.", {}

            if pil_image_for_processing.mode != "RGB": # Ensure final image is RGB
                pil_image_for_processing = pil_image_for_processing.convert("RGB")

            # Add Places365 scene analysis parallel to lighting analysis
            places365_info = self.analyze_places365_scene(pil_image_for_processing)

            lighting_info = self.analyze_lighting_conditions(pil_image_for_processing, places365_info=places365_info)

            temp_dir = tempfile.gettempdir()
            temp_filename = f"temp_{uuid.uuid4().hex}.jpg"
            temp_path = os.path.join(temp_dir, temp_filename)
            pil_image_for_processing.save(temp_path, format="JPEG")

            result = model_instance.detect(temp_path)

            if result is None or not hasattr(result, 'boxes'):
                scene_analysis_no_yolo = self.analyze_scene(result, lighting_info, enable_landmark=enable_landmark, places365_info=places365_info)
                desc_no_yolo = scene_analysis_no_yolo.get("enhanced_description", scene_analysis_no_yolo.get("description", "Detection failed, scene context analysis attempted."))
                stats_data["scene_analysis"] = scene_analysis_no_yolo
                if places365_info:
                    stats_data["places365_analysis"] = places365_info
                return pil_image_for_processing, desc_no_yolo, stats_data

            # 統計資訊
            stats_data = EvaluationMetrics.calculate_basic_stats(result)
            spatial_metrics = EvaluationMetrics.calculate_distance_metrics(result)
            stats_data["spatial_metrics"] = spatial_metrics
            stats_data["lighting_conditions"] = lighting_info
            if places365_info:
                stats_data["places365_analysis"] = places365_info

            if filter_classes and len(filter_classes) > 0:
                classes = result.boxes.cls.cpu().numpy().astype(int)
                confs = result.boxes.conf.cpu().numpy()
                mask = np.isin(classes, filter_classes)
                filtered_stats_data = {
                    "total_objects": int(np.sum(mask)), "class_statistics": {},
                    "average_confidence": float(np.mean(confs[mask])) if np.any(mask) else 0.0,
                    "spatial_metrics": stats_data.get("spatial_metrics",{}),
                    "lighting_conditions": lighting_info
                }
                if places365_info:
                    filtered_stats_data["places365_analysis"] = places365_info
                names = result.names
                class_conf_sums = {}
                for cls_id_int, conf_val in zip(classes[mask], confs[mask]):
                    cls_name = names[cls_id_int]
                    if cls_name not in filtered_stats_data["class_statistics"]:
                        filtered_stats_data["class_statistics"][cls_name] = {"count": 0}
                        class_conf_sums[cls_name] = 0.0
                    filtered_stats_data["class_statistics"][cls_name]["count"] += 1 # 累計統計資訊
                    class_conf_sums[cls_name] += conf_val
                for cls_name_stat, data_stat in filtered_stats_data["class_statistics"].items():
                    data_stat["average_confidence"] = round(class_conf_sums[cls_name_stat] / data_stat["count"] if data_stat["count"] > 0 else 0.0, 4)
                stats_data = filtered_stats_data

            viz_data = EvaluationMetrics.generate_visualization_data(result, self.color_mapper.get_all_colors())

            result_image_pil = VisualizationHelper.visualize_detection(
                temp_path, result, color_mapper=self.color_mapper,
                figsize=(12, 12), return_pil=True, filter_classes=filter_classes
            )

            result_text_summary = EvaluationMetrics.format_detection_summary(viz_data)

            #  Pass the enable_landmark parameter from function signature
            # Initialize or update scene analyzer if needed
            if self.scene_analyzer is None:
                print("Creating SceneAnalyzer in process_image")
                self.scene_analyzer = SceneAnalyzer(
                    class_names=result.names if result else None,
                    use_llm=self.use_llm,
                    use_clip=True,
                    enable_landmark=enable_landmark,
                    llm_model_path=self.llm_model_path
                )

                if self.scene_analyzer is None:
                    print("ERROR: Failed to create SceneAnalyzer in process_image")
            else:
                # Update existing scene analyzer with current settings
                if result and hasattr(result, 'names'):
                    # 使用檢測結果的類別名稱或回退到預定義映射
                    current_class_names = result.names if result.names else self.coco_class_names

                    self.scene_analyzer.class_names = current_class_names
                    if hasattr(self.scene_analyzer, 'spatial_analyzer') and self.scene_analyzer.spatial_analyzer:
                        self.scene_analyzer.spatial_analyzer.update_class_names(current_class_names)

                    logger.info(f"Updated class names in scene analyzer: {list(current_class_names.keys())}")

                self.scene_analyzer.enable_landmark = enable_landmark
                if hasattr(self.scene_analyzer, 'spatial_analyzer') and self.scene_analyzer.spatial_analyzer:
                    self.scene_analyzer.spatial_analyzer.enable_landmark = enable_landmark

            # Perform scene analysis using the existing analyze_scene method
            scene_analysis_result = self.analyze_scene(
                detection_result=result,
                lighting_info=lighting_info,
                enable_landmark=enable_landmark,
                places365_info=places365_info
            )

            stats_data["scene_analysis"] = scene_analysis_result

            final_result_text = result_text_summary

            # Use enable_landmark parameter for landmark block
            if enable_landmark and "detected_landmarks" in scene_analysis_result:
                landmarks_detected = scene_analysis_result.get("detected_landmarks", [])
                if not landmarks_detected and scene_analysis_result.get("primary_landmark"):
                    primary_lm = scene_analysis_result.get("primary_landmark")
                    if isinstance(primary_lm, dict): landmarks_detected = [primary_lm]

                if landmarks_detected:
                    final_result_text += "\n\n--- Detected Landmarks ---\n"
                    # Ensure drawing on the correct PIL image
                    img_to_draw_on = result_image_pil.copy() # Draw on a copy
                    img_for_drawing_cv2 = cv2.cvtColor(np.array(img_to_draw_on), cv2.COLOR_RGB2BGR)

                    for landmark_item in landmarks_detected:
                        if not isinstance(landmark_item, dict): continue

                        # Use .get() for all potentially missing keys 比較保險
                        landmark_name_disp = landmark_item.get("class_name", landmark_item.get("name", "N/A"))
                        landmark_loc_disp = landmark_item.get("location", "N/A")
                        landmark_conf_disp = landmark_item.get("confidence", 0.0)

                        final_result_text += f"• {landmark_name_disp} ({landmark_loc_disp}, confidence: {landmark_conf_disp:.2f})\n"

                        if "box" in landmark_item:
                            box = landmark_item["box"]
                            pt1 = (int(box[0]), int(box[1])); pt2 = (int(box[2]), int(box[3]))
                            color_lm = (255, 0, 255); thickness_lm = 3 # Magenta BGR
                            cv2.rectangle(img_for_drawing_cv2, pt1, pt2, color_lm, thickness_lm)

                            label_lm = f"{landmark_name_disp} ({landmark_conf_disp:.2f})"
                            font_scale_lm = 0.6; font_thickness_lm = 1
                            (w_text, h_text), baseline = cv2.getTextSize(label_lm, cv2.FONT_HERSHEY_SIMPLEX, font_scale_lm, font_thickness_lm)

                            # Label position logic (simplified from your extensive one for brevity)
                            label_y_pos = pt1[1] - baseline - 3
                            if label_y_pos < h_text : # If label goes above image, put it below box
                                label_y_pos = pt2[1] + h_text + baseline + 3

                            label_bg_pt1 = (pt1[0], label_y_pos - h_text - baseline)
                            label_bg_pt2 = (pt1[0] + w_text, label_y_pos + baseline)

                            cv2.rectangle(img_for_drawing_cv2, label_bg_pt1, label_bg_pt2, color_lm, -1)
                            cv2.putText(img_for_drawing_cv2, label_lm, (pt1[0], label_y_pos),
                                        cv2.FONT_HERSHEY_SIMPLEX, font_scale_lm, (255,255,255), font_thickness_lm, cv2.LINE_AA)

                    result_image_pil = Image.fromarray(cv2.cvtColor(img_for_drawing_cv2, cv2.COLOR_BGR2RGB))

            return result_image_pil, final_result_text, stats_data

        except Exception as e:
            error_message = f"Error in ImageProcessor.process_image: {str(e)}"
            import traceback
            traceback.print_exc()
            return pil_image_for_processing if pil_image_for_processing else None, error_message, {}
        finally:
            if temp_path and os.path.exists(temp_path):
                try: os.remove(temp_path)
                except Exception as e: print(f"Warning: Cannot delete temp file {temp_path}: {str(e)}")

    def format_result_text(self, stats: Dict) -> str:
        """
        Format detection statistics into readable text with improved spacing

        Args:
            stats: Dictionary containing detection statistics

        Returns:
            Formatted text summary
        """
        if not stats or "total_objects" not in stats:
            return "No objects detected."

        # 減少不必要的空行
        lines = [
            f"Detected {stats['total_objects']} objects.",
            f"Average confidence: {stats.get('average_confidence', 0):.2f}",
            "Objects by class:"
        ]

        if "class_statistics" in stats and stats["class_statistics"]:
            # 按計數排序類別
            sorted_classes = sorted(
                stats["class_statistics"].items(),
                key=lambda x: x[1]["count"],
                reverse=True
            )

            for cls_name, cls_stats in sorted_classes:
                count = cls_stats["count"]
                conf = cls_stats.get("average_confidence", 0)

                item_text = "item" if count == 1 else "items"
                lines.append(f"• {cls_name}: {count} {item_text} (avg conf: {conf:.2f})")
        else:
            lines.append("No class information available.")

        # 添加空間資訊
        if "spatial_metrics" in stats and "spatial_distribution" in stats["spatial_metrics"]:
            lines.append("Object Distribution:")

            dist = stats["spatial_metrics"]["spatial_distribution"]
            x_mean = dist.get("x_mean", 0)
            y_mean = dist.get("y_mean", 0)

            # 描述物體的大致位置
            if x_mean < 0.33:
                h_pos = "on the left side"
            elif x_mean < 0.67:
                h_pos = "in the center"
            else:
                h_pos = "on the right side"

            if y_mean < 0.33:
                v_pos = "in the upper part"
            elif y_mean < 0.67:
                v_pos = "in the middle"
            else:
                v_pos = "in the lower part"

            lines.append(f"• Most objects appear {h_pos} {v_pos} of the image")

        return "\n".join(lines)

    def format_json_for_display(self, stats: Dict) -> Dict:
        """
        Format statistics JSON for better display

        Args:
            stats: Raw statistics dictionary

        Returns:
            Formatted statistics structure for display
        """
        # Create a cleaner copy of the stats for display
        display_stats = {}

        # Add summary section
        display_stats["summary"] = {
            "total_objects": stats.get("total_objects", 0),
            "average_confidence": round(stats.get("average_confidence", 0), 3)
        }

        # Add class statistics in a more organized way
        if "class_statistics" in stats and stats["class_statistics"]:
            # Sort classes by count (descending)
            sorted_classes = sorted(
                stats["class_statistics"].items(),
                key=lambda x: x[1].get("count", 0),
                reverse=True
            )

            class_stats = {}
            for cls_name, cls_data in sorted_classes:
                class_stats[cls_name] = {
                    "count": cls_data.get("count", 0),
                    "average_confidence": round(cls_data.get("average_confidence", 0), 3)
                }

            display_stats["detected_objects"] = class_stats

        # Simplify spatial metrics
        if "spatial_metrics" in stats:
            spatial = stats["spatial_metrics"]

            # Simplify spatial distribution
            if "spatial_distribution" in spatial:
                dist = spatial["spatial_distribution"]
                display_stats["spatial"] = {
                    "distribution": {
                        "x_mean": round(dist.get("x_mean", 0), 3),
                        "y_mean": round(dist.get("y_mean", 0), 3),
                        "x_std": round(dist.get("x_std", 0), 3),
                        "y_std": round(dist.get("y_std", 0), 3)
                    }
                }

            # Add simplified size information
            if "size_distribution" in spatial:
                size = spatial["size_distribution"]
                display_stats["spatial"]["size"] = {
                    "mean_area": round(size.get("mean_area", 0), 3),
                    "min_area": round(size.get("min_area", 0), 3),
                    "max_area": round(size.get("max_area", 0), 3)
                }

        return display_stats

    def prepare_visualization_data(self, stats: Dict, available_classes: Dict[int, str]) -> Dict:
        """
        Prepare data for visualization based on detection statistics

        Args:
            stats: Detection statistics
            available_classes: Dictionary of available class IDs and names

        Returns:
            Visualization data dictionary
        """
        if not stats or "class_statistics" not in stats or not stats["class_statistics"]:
            return {"error": "No detection data available"}

        # Prepare visualization data
        viz_data = {
            "total_objects": stats.get("total_objects", 0),
            "average_confidence": stats.get("average_confidence", 0),
            "class_data": []
        }

        # Class data
        for cls_name, cls_stats in stats.get("class_statistics", {}).items():
            # Search class ID
            class_id = -1
            for id, name in available_classes.items():
                if name == cls_name:
                    class_id = id
                    break

            cls_data = {
                "name": cls_name,
                "class_id": class_id,
                "count": cls_stats.get("count", 0),
                "average_confidence": cls_stats.get("average_confidence", 0),
                "color": self.color_mapper.get_color(class_id if class_id >= 0 else cls_name)
            }

            viz_data["class_data"].append(cls_data)

        # Descending order
        viz_data["class_data"].sort(key=lambda x: x["count"], reverse=True)

        return viz_data
