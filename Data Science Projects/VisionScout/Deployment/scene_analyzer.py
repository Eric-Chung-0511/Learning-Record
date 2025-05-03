import os
import numpy as np
from typing import Dict, List, Tuple, Any, Optional

from spatial_analyzer import SpatialAnalyzer
from scene_description import SceneDescriptor
from enhance_scene_describer import EnhancedSceneDescriber
from clip_analyzer import CLIPAnalyzer
from scene_type import SCENE_TYPES
from object_categories import OBJECT_CATEGORIES

class SceneAnalyzer:
    """
    Core class for scene analysis and understanding based on object detection results.
    Analyzes detected objects, their relationships, and infers the scene type.
    """
    def __init__(self, class_names: Dict[int, str] = None):
        """
        Initialize the scene analyzer with optional class name mappings.
        Args:
            class_names: Dictionary mapping class IDs to class names (optional)
        """
        self.class_names = class_names

        # 加載場景類型和物體類別
        self.SCENE_TYPES = SCENE_TYPES
        self.OBJECT_CATEGORIES = OBJECT_CATEGORIES

        # 初始化其他組件，將數據傳遞給 SceneDescriptor
        self.spatial_analyzer = SpatialAnalyzer(class_names=class_names, object_categories=self.OBJECT_CATEGORIES)
        self.descriptor = SceneDescriptor(scene_types=self.SCENE_TYPES, object_categories=self.OBJECT_CATEGORIES)
        self.scene_describer = EnhancedSceneDescriber(scene_types=self.SCENE_TYPES)

        # 初始化 CLIP 分析器（新增）
        try:
            self.clip_analyzer = CLIPAnalyzer()
            self.use_clip = True
        except Exception as e:
            print(f"Warning: Could not initialize CLIP analyzer: {e}")
            print("Scene analysis will proceed without CLIP. Install CLIP with 'pip install clip' for enhanced scene understanding.")
            self.use_clip = False

    def generate_scene_description(self,
                             scene_type,
                             detected_objects,
                             confidence,
                             lighting_info=None,
                             functional_zones=None):
        """
        生成場景描述。
        Args:
            scene_type: 識別的場景類型
            detected_objects: 檢測到的物體列表
            confidence: 場景分類置信度
            lighting_info: 照明條件信息（可選）
            functional_zones: 功能區域信息（可選）
        Returns:
            str: 生成的場景描述
        """
        return self.scene_describer.generate_description(
            scene_type,
            detected_objects,
            confidence,
            lighting_info,
            functional_zones
        )

    def _generate_scene_description(self, scene_type, detected_objects, confidence, lighting_info=None):
        """
        Use new implement
        """
        # get the functional zones info
        functional_zones = self.spatial_analyzer._identify_functional_zones(detected_objects, scene_type)

        return self.generate_scene_description(
            scene_type,
            detected_objects,
            confidence,
            lighting_info,
            functional_zones
        )

    def _define_image_regions(self):
        """Define regions of the image for spatial analysis (3x3 grid)"""
        self.regions = {
            "top_left": (0, 0, 1/3, 1/3),
            "top_center": (1/3, 0, 2/3, 1/3),
            "top_right": (2/3, 0, 1, 1/3),
            "middle_left": (0, 1/3, 1/3, 2/3),
            "middle_center": (1/3, 1/3, 2/3, 2/3),
            "middle_right": (2/3, 1/3, 1, 2/3),
            "bottom_left": (0, 2/3, 1/3, 1),
            "bottom_center": (1/3, 2/3, 2/3, 1),
            "bottom_right": (2/3, 2/3, 1, 1)
        }


    def analyze(self, detection_result: Any, lighting_info: Optional[Dict] = None, class_confidence_threshold: float = 0.35, scene_confidence_threshold: float = 0.6) -> Dict:
        """
        Analyze detection results to determine scene type and provide understanding.
        Args:
            detection_result: Detection result from YOLOv8
            lighting_info: Optional lighting condition analysis results
            class_confidence_threshold: Minimum confidence to consider an object
            scene_confidence_threshold: Minimum confidence to determine a scene
        Returns:
            Dictionary with scene analysis results
        """
        # If no result or no detections, return empty analysis
        if detection_result is None or len(detection_result.boxes) == 0:
            return {
                "scene_type": "unknown",
                "confidence": 0,
                "description": "No objects detected in the image.",
                "objects_present": [],
                "object_count": 0,
                "regions": {},
                "possible_activities": [],
                "safety_concerns": [],
                "lighting_conditions": lighting_info or {"time_of_day": "unknown", "confidence": 0}
            }

        # Get class names from detection result if not already set
        if self.class_names is None:
            self.class_names = detection_result.names
            # Also update class names in spatial analyzer
            self.spatial_analyzer.class_names = self.class_names

        # Extract detected objects with confidence above threshold
        detected_objects = self.spatial_analyzer._extract_detected_objects(
            detection_result,
            confidence_threshold=class_confidence_threshold
        )

        # No objects above confidence threshold
        if not detected_objects:
            return {
                "scene_type": "unknown",
                "confidence": 0.0,
                "description": "No objects with sufficient confidence detected.",
                "objects_present": [],
                "object_count": 0,
                "regions": {},
                "possible_activities": [],
                "safety_concerns": [],
                "lighting_conditions": lighting_info or {"time_of_day": "unknown", "confidence": 0}
            }

        # Analyze object distribution in regions
        region_analysis = self.spatial_analyzer._analyze_regions(detected_objects)

        # Compute scene type scores based on object detection
        yolo_scene_scores = self._compute_scene_scores(detected_objects)

        # 使用 CLIP 分析圖像
        clip_scene_scores = {}
        clip_analysis = None
        if self.use_clip:
            try:
                # 獲取原始圖像
                original_image = detection_result.orig_img

                # Use CLIP analyze image
                clip_analysis = self.clip_analyzer.analyze_image(original_image)

                # get CLIP's score
                clip_scene_scores = clip_analysis.get("scene_scores", {})

                if "asian_commercial_street" in clip_scene_scores and clip_scene_scores["asian_commercial_street"] > 0.2:
                    # 使用對比提示進一步區分室內/室外
                    comparative_results = self.clip_analyzer.calculate_similarity(
                        original_image,
                        self.clip_analyzer.comparative_prompts["indoor_vs_outdoor"]
                    )

                    # 分析對比結果
                    indoor_score = sum(s for p, s in comparative_results.items() if "indoor" in p or "enclosed" in p)
                    outdoor_score = sum(s for p, s in comparative_results.items() if "outdoor" in p or "open-air" in p)

                    # 如果 CLIP 認為這是室外場景，且光照分析認為是室內
                    if outdoor_score > indoor_score and lighting_info and lighting_info.get("is_indoor", False):
                        # 修正光照分析結果
                        print(f"CLIP indicates outdoor commercial street (score: {outdoor_score:.2f} vs {indoor_score:.2f}), adjusting lighting analysis")
                        lighting_info["is_indoor"] = False
                        lighting_info["indoor_probability"] = 0.3
                        # 把CLIP 分析結果加到光照診斷
                        if "diagnostics" not in lighting_info:
                            lighting_info["diagnostics"] = {}
                        lighting_info["diagnostics"]["clip_override"] = {
                            "reason": "CLIP detected outdoor commercial street",
                            "outdoor_score": float(outdoor_score),
                            "indoor_score": float(indoor_score)
                        }

                # 如果 CLIP 檢測到了光照條件但沒有提供 lighting_info
                if not lighting_info and "lighting_condition" in clip_analysis:
                    lighting_type, lighting_conf = clip_analysis["lighting_condition"]
                    lighting_info = {
                        "time_of_day": lighting_type,
                        "confidence": lighting_conf
                    }
            except Exception as e:
                print(f"Error in CLIP analysis: {e}")

        # 融合 YOLO 和 CLIP 的場景分數
        scene_scores = self._fuse_scene_scores(yolo_scene_scores, clip_scene_scores)

        # Determine best matching scene type
        best_scene, scene_confidence = self._determine_scene_type(scene_scores)

        # Generate possible activities based on scene
        activities = self.descriptor._infer_possible_activities(best_scene, detected_objects)

        # Identify potential safety concerns
        safety_concerns = self.descriptor._identify_safety_concerns(detected_objects, best_scene)

        # Calculate functional zones
        functional_zones = self.spatial_analyzer._identify_functional_zones(detected_objects, best_scene)

        # Generate scene description
        scene_description = self.generate_scene_description(
            best_scene,
            detected_objects,
            scene_confidence,
            lighting_info=lighting_info,
            functional_zones=functional_zones
        )

        # Return comprehensive analysis
        result = {
            "scene_type": best_scene if scene_confidence >= scene_confidence_threshold else "unknown",
            "scene_name": self.SCENE_TYPES.get(best_scene, {}).get("name", "Unknown")
                        if scene_confidence >= scene_confidence_threshold else "Unknown Scene",
            "confidence": scene_confidence,
            "description": scene_description,
            "objects_present": [
                {"class_id": obj["class_id"],
                "class_name": obj["class_name"],
                "confidence": obj["confidence"]}
                for obj in detected_objects
            ],
            "object_count": len(detected_objects),
            "regions": region_analysis,
            "possible_activities": activities,
            "safety_concerns": safety_concerns,
            "functional_zones": functional_zones,
            "alternative_scenes": self.descriptor._get_alternative_scenes(scene_scores, scene_confidence_threshold, top_k=2),
            "lighting_conditions": lighting_info or {"time_of_day": "unknown", "confidence": 0}
        }

        # 添加 CLIP 特定的結果（新增）
        if clip_analysis and "error" not in clip_analysis:
            result["clip_analysis"] = {
                "top_scene": clip_analysis.get("top_scene", ("unknown", 0)),
                "cultural_analysis": clip_analysis.get("cultural_analysis", {})
            }

        return result

    def _compute_scene_scores(self, detected_objects: List[Dict]) -> Dict[str, float]:
        """
        Compute confidence scores for each scene type based on detected objects.
        Args:
            detected_objects: List of detected objects
        Returns:
            Dictionary mapping scene types to confidence scores
        """
        scene_scores = {}
        detected_class_ids = [obj["class_id"] for obj in detected_objects]
        detected_classes_set = set(detected_class_ids)

        # Count occurrence of each class
        class_counts = {}
        for obj in detected_objects:
            class_id = obj["class_id"]
            if class_id not in class_counts:
                class_counts[class_id] = 0
            class_counts[class_id] += 1

        # Evaluate each scene type
        for scene_type, scene_def in self.SCENE_TYPES.items():
            # Count required objects present
            required_objects = set(scene_def["required_objects"])
            required_present = required_objects.intersection(detected_classes_set)

            # Count optional objects present
            optional_objects = set(scene_def["optional_objects"])
            optional_present = optional_objects.intersection(detected_classes_set)

            # Skip if minimum required objects aren't present
            if len(required_present) < scene_def["minimum_required"]:
                scene_scores[scene_type] = 0
                continue

            # Base score from required objects
            required_ratio = len(required_present) / max(1, len(required_objects))
            required_score = required_ratio * 0.7  # 70% of score from required objects

            # Additional score from optional objects
            optional_ratio = len(optional_present) / max(1, len(optional_objects))
            optional_score = optional_ratio * 0.3  # 30% of score from optional objects

            # Bonus for having multiple instances of key objects
            multiple_bonus = 0.0
            for class_id in required_present:
                if class_counts.get(class_id, 0) > 1:
                    multiple_bonus += 0.05  # 5% bonus per additional key object type

            # Cap the bonus at 15%
            multiple_bonus = min(0.15, multiple_bonus)

            # Calculate final score
            final_score = required_score + optional_score + multiple_bonus

            if "priority" in scene_def:
                final_score *= scene_def["priority"]

            # Normalize to 0-1 range
            scene_scores[scene_type] = min(1.0, final_score)

        return scene_scores

    def _determine_scene_type(self, scene_scores: Dict[str, float]) -> Tuple[str, float]:
        """
        Determine the most likely scene type based on scores.
        Args:
            scene_scores: Dictionary mapping scene types to confidence scores
        Returns:
            Tuple of (best_scene_type, confidence)
        """
        if not scene_scores:
            return "unknown", 0

        # Find scene with highest score
        best_scene = max(scene_scores, key=scene_scores.get)
        best_score = scene_scores[best_scene]

        return best_scene, best_score


    def _fuse_scene_scores(self, yolo_scene_scores: Dict[str, float], clip_scene_scores: Dict[str, float]) -> Dict[str, float]:
        """
        融合基於 YOLO 物體檢測和 CLIP 分析的場景分數。
        Args:
            yolo_scene_scores: 基於 YOLO 物體檢測的場景分數
            clip_scene_scores: 基於 CLIP 分析的場景分數
        Returns:
            Dict: 融合後的場景分數
        """
        # 如果沒有 CLIP 分數，直接返回 YOLO 分數
        if not clip_scene_scores:
            return yolo_scene_scores

        # 如果沒有 YOLO 分數，直接返回 CLIP 分數
        if not yolo_scene_scores:
            return clip_scene_scores

        # 融合分數
        fused_scores = {}

        # 獲取所有場景類型
        all_scene_types = set(list(yolo_scene_scores.keys()) + list(clip_scene_scores.keys()))

        for scene_type in all_scene_types:
            # 獲取兩個模型的分數
            yolo_score = yolo_scene_scores.get(scene_type, 0)
            clip_score = clip_scene_scores.get(scene_type, 0)

            # 設置基本權重
            yolo_weight = 0.7  # YOLO 可提供比較好的物體資訊
            clip_weight = 0.3  # CLIP 強項是理解整體的場景關係

            # 對特定類型場景調整權重
            # 文化特定場景或具有特殊布局的場景，CLIP可能比較能理解
            if any(keyword in scene_type for keyword in ["asian", "cultural", "aerial"]):
                yolo_weight = 0.3
                clip_weight = 0.7

            # 對室內家居場景，物體檢測通常更準確
            elif any(keyword in scene_type for keyword in ["room", "kitchen", "office", "bedroom"]):
                yolo_weight = 0.8
                clip_weight = 0.2
            elif scene_type == "beach_water_recreation":
                yolo_weight = 0.8  # 衝浪板等特定物品的檢測
                clip_weight = 0.2
            elif scene_type == "sports_venue":
                yolo_weight = 0.7
                clip_weight = 0.3
            elif scene_type == "professional_kitchen":
                yolo_weight = 0.8  # 廚房用具的檢測非常重要
                clip_weight = 0.2

            # 計算加權分數
            fused_scores[scene_type] = (yolo_score * yolo_weight) + (clip_score * clip_weight)

        return fused_scores
