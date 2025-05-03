import os
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

class ImageProcessor:
    """
    Class for handling image processing and object detection operations
    Separates processing logic from UI components
    """

    def __init__(self):
        """Initialize the image processor with required components"""
        self.color_mapper = ColorMapper()
        self.model_instances = {}
        self.lighting_analyzer = LightingAnalyzer()

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

    def analyze_scene(self, detection_result: Any, lighting_info: Optional[Dict] = None) -> Dict:
        """
        Perform scene analysis on detection results

        Args:
            detection_result: Object detection result from YOLOv8
            lighting_info: Lighting condition analysis results (optional)

        Returns:
            Dictionary containing scene analysis results
        """
        try:
            # Initialize scene analyzer if not already done
            if not hasattr(self, 'scene_analyzer'):
                self.scene_analyzer = SceneAnalyzer(class_names=detection_result.names)

            # 確保類名正確更新
            if self.scene_analyzer.class_names is None:
                self.scene_analyzer.class_names = detection_result.names
                self.scene_analyzer.spatial_analyzer.class_names = detection_result.names

            # Perform scene analysis with lighting info
            scene_analysis = self.scene_analyzer.analyze(
                detection_result=detection_result,
                lighting_info=lighting_info,
                class_confidence_threshold=0.35,
                scene_confidence_threshold=0.6
            )

            return scene_analysis
        except Exception as e:
            print(f"Error in scene analysis: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                "scene_type": "unknown",
                "confidence": 0.0,
                "description": f"Error during scene analysis: {str(e)}",
                "objects_present": [],
                "object_count": 0,
                "regions": {},
                "possible_activities": [],
                "safety_concerns": [],
                "lighting_conditions": lighting_info or {"time_of_day": "unknown", "confidence": 0.0}
            }

    def analyze_lighting_conditions(self, image):
        """
        分析光照條件。

        Args:
            image: 輸入圖像

        Returns:
            Dict: 光照分析結果
        """
        return self.lighting_analyzer.analyze(image)

    def process_image(self, image, model_name: str, confidence_threshold: float, filter_classes: Optional[List[int]] = None) -> Tuple[Any, str, Dict]:
        """
        Process an image for object detection

        Args:
            image: Input image (numpy array or PIL Image)
            model_name: Name of the model to use
            confidence_threshold: Confidence threshold for detection
            filter_classes: Optional list of classes to filter results

        Returns:
            Tuple of (result_image, result_text, stats_data)
        """
        # Get model instance
        model_instance = self.get_model_instance(model_name, confidence_threshold)

        # Initialize key variables
        result = None
        stats = {}
        temp_path = None

        try:
            # Processing input image
            if isinstance(image, np.ndarray):
                # Convert BGR to RGB if needed
                if image.shape[2] == 3:
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                else:
                    image_rgb = image
                pil_image = Image.fromarray(image_rgb)
            elif image is None:
                return None, "No image provided. Please upload an image.", {}
            else:
                pil_image = image

            # Analyze lighting conditions
            lighting_info = self.analyze_lighting_conditions(pil_image)

            # Store temp files
            temp_dir = tempfile.gettempdir()  # Use system temp directory
            temp_filename = f"temp_{uuid.uuid4().hex}.jpg"
            temp_path = os.path.join(temp_dir, temp_filename)
            pil_image.save(temp_path)

            # Object detection
            result = model_instance.detect(temp_path)

            if result is None:
                return None, "Detection failed. Please try again with a different image.", {}

            # Calculate stats
            stats = EvaluationMetrics.calculate_basic_stats(result)

            # Add space calculation
            spatial_metrics = EvaluationMetrics.calculate_distance_metrics(result)
            stats["spatial_metrics"] = spatial_metrics

            # Add lighting information
            stats["lighting_conditions"] = lighting_info

            # Apply filter if specified
            if filter_classes and len(filter_classes) > 0:
                # Get classes, boxes, confidence
                classes = result.boxes.cls.cpu().numpy().astype(int)
                confs = result.boxes.conf.cpu().numpy()
                boxes = result.boxes.xyxy.cpu().numpy()

                mask = np.zeros_like(classes, dtype=bool)
                for cls_id in filter_classes:
                    mask = np.logical_or(mask, classes == cls_id)

                filtered_stats = {
                    "total_objects": int(np.sum(mask)),
                    "class_statistics": {},
                    "average_confidence": float(np.mean(confs[mask])) if np.any(mask) else 0,
                    "spatial_metrics": stats["spatial_metrics"],
                    "lighting_conditions": lighting_info
                }

                # Update stats
                names = result.names
                for cls, conf in zip(classes[mask], confs[mask]):
                    cls_name = names[int(cls)]
                    if cls_name not in filtered_stats["class_statistics"]:
                        filtered_stats["class_statistics"][cls_name] = {
                            "count": 0,
                            "average_confidence": 0
                        }

                    filtered_stats["class_statistics"][cls_name]["count"] += 1
                    filtered_stats["class_statistics"][cls_name]["average_confidence"] = conf

                stats = filtered_stats

            viz_data = EvaluationMetrics.generate_visualization_data(
                result,
                self.color_mapper.get_all_colors()
            )

            result_image = VisualizationHelper.visualize_detection(
                temp_path, result, color_mapper=self.color_mapper, figsize=(12, 12), return_pil=True, filter_classes=filter_classes
            )

            result_text = EvaluationMetrics.format_detection_summary(viz_data)

            if result is not None:
                # Perform scene analysis with lighting info
                scene_analysis = self.analyze_scene(result, lighting_info)

                # Add scene analysis to stats
                stats["scene_analysis"] = scene_analysis

            return result_image, result_text, stats

        except Exception as e:
            error_message = f"Error Occurs: {str(e)}"
            import traceback
            traceback.print_exc()
            print(error_message)
            return None, error_message, {}

        finally:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception as e:
                    print(f"Cannot delete temp files {temp_path}: {str(e)}")


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

        # 添加空間信息
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
