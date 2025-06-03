
import os
import numpy as np
import logging
import traceback
from typing import Dict, List, Tuple, Any, Optional
from PIL import Image

from component_initializer import ComponentInitializer
from scene_scoring_engine import SceneScoringEngine
from landmark_processing_manager import LandmarkProcessingManager
from scene_analysis_coordinator import SceneAnalysisCoordinator

class SceneAnalyzer:
    """
    Core class for scene analysis and understanding based on object detection results.
    Analyzes detected objects, their relationships, and infers the scene type.
    此class為場景理解的總窗口

    This is the main Facade class that coordinates all scene analysis components
    while maintaining the original public interface for backward compatibility.
    """

    EVERYDAY_SCENE_TYPE_KEYS = [
        "general_indoor_space", "generic_street_view",
        "desk_area_workspace", "outdoor_gathering_spot",
        "kitchen_counter_or_utility_area"
    ]

    def __init__(self, class_names: Dict[int, str] = None, use_llm: bool = True,
                 use_clip: bool = True, enable_landmark: bool = True,
                 llm_model_path: str = None):
        """
        Initialize the scene analyzer with optional class name mappings.

        Args:
            class_names: Dictionary mapping class IDs to class names (optional)
            use_llm: Whether to enable LLM enhancement functionality
            use_clip: Whether to enable CLIP analysis functionality
            enable_landmark: Whether to enable landmark detection functionality
            llm_model_path: Path to LLM model (optional)
        """
        self.logger = logging.getLogger(__name__)

        try:
            # Initialize all components through the component initializer
            self.component_initializer = ComponentInitializer(
                class_names=class_names,
                use_llm=use_llm,
                use_clip=use_clip,
                enable_landmark=enable_landmark,
                llm_model_path=llm_model_path
            )

            # Get data structures for easy access
            self.SCENE_TYPES = self.component_initializer.get_data_structure('SCENE_TYPES')
            self.OBJECT_CATEGORIES = self.component_initializer.get_data_structure('OBJECT_CATEGORIES')
            self.LANDMARK_ACTIVITIES = self.component_initializer.get_data_structure('LANDMARK_ACTIVITIES')

            # Initialize specialized engines
            self.scene_scoring_engine = SceneScoringEngine(
                scene_types=self.SCENE_TYPES,
                enable_landmark=enable_landmark
            )

            self.landmark_processing_manager = LandmarkProcessingManager(
                enable_landmark=enable_landmark,
                use_clip=use_clip
            )

            # Initialize the main coordinator
            self.scene_analysis_coordinator = SceneAnalysisCoordinator(
                component_initializer=self.component_initializer,
                scene_scoring_engine=self.scene_scoring_engine,
                landmark_processing_manager=self.landmark_processing_manager
            )

            # Store configuration for backward compatibility
            self.class_names = class_names
            self.use_clip = use_clip
            self.use_llm = use_llm
            self.enable_landmark = enable_landmark
            self.use_landmark_detection = enable_landmark

            # Get component references for backward compatibility
            self.spatial_analyzer = self.component_initializer.get_component('spatial_analyzer')
            self.descriptor = self.component_initializer.get_component('descriptor')
            self.scene_describer = self.component_initializer.get_component('scene_describer')
            self.clip_analyzer = self.component_initializer.get_component('clip_analyzer')
            self.llm_enhancer = self.component_initializer.get_component('llm_enhancer')
            self.landmark_classifier = self.component_initializer.get_component('landmark_classifier')

            # Set landmark classifier in the processing manager
            if self.landmark_classifier:
                self.landmark_processing_manager.set_landmark_classifier(self.landmark_classifier)

            self.logger.info("SceneAnalyzer initialized successfully with all components")

        except Exception as e:
            self.logger.error(f"Critical error during SceneAnalyzer initialization: {e}")
            traceback.print_exc()
            raise

    def analyze(self, detection_result: Any, lighting_info: Optional[Dict] = None,
                class_confidence_threshold: float = 0.25, scene_confidence_threshold: float = 0.6,
                enable_landmark: bool = True, places365_info: Optional[Dict] = None) -> Dict:
        """
        Analyze detection results to determine scene type and provide understanding.

        Args:
            detection_result: Detection result from YOLOv8 or similar
            lighting_info: Optional lighting condition analysis results
            class_confidence_threshold: Minimum confidence to consider an object
            scene_confidence_threshold: Minimum confidence to determine a scene
            enable_landmark: Whether to enable landmark detection and recognition for this run
            places365_info: Optional Places365 scene classification results

        Returns:
            Dictionary with scene analysis results
        """
        try:
            return self.scene_analysis_coordinator.analyze(
                detection_result=detection_result,
                lighting_info=lighting_info,
                class_confidence_threshold=class_confidence_threshold,
                scene_confidence_threshold=scene_confidence_threshold,
                enable_landmark=enable_landmark,
                places365_info=places365_info
            )
        except Exception as e:
            self.logger.error(f"Error in scene analysis: {e}")
            traceback.print_exc()
            # Return a safe fallback result
            return {
                "scene_type": "unknown",
                "confidence": 0.0,
                "description": "Scene analysis failed due to an internal error.",
                "enhanced_description": "An error occurred during scene analysis. Please check the system logs for details.",
                "objects_present": [],
                "object_count": 0,
                "regions": {},
                "possible_activities": [],
                "safety_concerns": [],
                "lighting_conditions": lighting_info or {"time_of_day": "unknown", "confidence": 0.0}
            }

    def generate_scene_description(self, scene_type: str, detected_objects: List[Dict],
                                 confidence: float, lighting_info: Optional[Dict] = None,
                                 functional_zones: Optional[Dict] = None,
                                 enable_landmark: bool = True,
                                 scene_scores: Optional[Dict] = None,
                                 spatial_analysis: Optional[Dict] = None,
                                 image_dimensions: Optional[Tuple[int, int]] = None) -> str:
        """
        Generate scene description and pass all necessary context to the underlying describer.

        Args:
            scene_type: Identified scene type
            detected_objects: List of detected objects
            confidence: Scene classification confidence
            lighting_info: Lighting condition information (optional)
            functional_zones: Functional zone information (optional)
            enable_landmark: Whether to enable landmark description (optional)
            scene_scores: Scene scores (optional)
            spatial_analysis: Spatial analysis results (optional)
            image_dimensions: Image dimensions (width, height) (optional)

        Returns:
            str: Generated scene description
        """
        try:
            # Convert functional_zones from Dict to List[str] and filter technical terms
            functional_zones_list = []
            if functional_zones and isinstance(functional_zones, dict):
                # Filter out technical terms, keep only meaningful descriptions
                filtered_zones = {k: v for k, v in functional_zones.items()
                                if not k.endswith('_zone') or k in ['dining_zone', 'seating_zone', 'work_zone']}
                functional_zones_list = [v.get('description', k) for k, v in filtered_zones.items()
                                       if isinstance(v, dict) and v.get('description')]
            elif functional_zones and isinstance(functional_zones, list):
                # Filter technical terms from list
                functional_zones_list = [zone for zone in functional_zones
                                       if not zone.endswith('_zone') or 'area' in zone]

            # Generate detailed object statistics
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

            # Calculate average confidence
            for class_name, stats in object_statistics.items():
                if stats["count"] > 0:
                    total_conf = sum(inst.get("confidence", 0.0) for inst in stats["instances"])
                    stats["avg_confidence"] = total_conf / stats["count"]

            if self.scene_describer:
                return self.scene_describer.generate_description(
                    scene_type=scene_type,
                    detected_objects=detected_objects,
                    confidence=confidence,
                    lighting_info=lighting_info,
                    functional_zones=functional_zones_list,
                    enable_landmark=enable_landmark,
                    scene_scores=scene_scores,
                    spatial_analysis=spatial_analysis,
                    image_dimensions=image_dimensions,
                    object_statistics=object_statistics
                )
            else:
                return f"A {scene_type} scene with {len(detected_objects)} detected objects."

        except Exception as e:
            self.logger.error(f"Error generating scene description: {e}")
            return f"A {scene_type} scene."

    def process_unknown_objects(self, detection_result, detected_objects):
        """
        Process objects that YOLO failed to identify or have low confidence for landmark detection.

        Args:
            detection_result: YOLO detection results
            detected_objects: List of identified objects

        Returns:
            tuple: (updated object list, landmark object list)
        """
        try:
            return self.landmark_processing_manager.process_unknown_objects(
                detection_result, detected_objects, self.clip_analyzer
            )
        except Exception as e:
            self.logger.error(f"Error processing unknown objects: {e}")
            traceback.print_exc()
            return detected_objects, []

    def _compute_scene_scores(self, detected_objects: List[Dict],
                            spatial_analysis_results: Optional[Dict] = None) -> Dict[str, float]:
        """
        Compute confidence scores for each scene type based on detected objects.

        Args:
            detected_objects: List of detected objects with their details
            spatial_analysis_results: Optional output from SpatialAnalyzer

        Returns:
            Dictionary mapping scene types to confidence scores
        """
        return self.scene_scoring_engine.compute_scene_scores(
            detected_objects, spatial_analysis_results
        )

    def _determine_scene_type(self, scene_scores: Dict[str, float]) -> Tuple[str, float]:
        """
        Determine the most likely scene type based on scores.

        Args:
            scene_scores: Dictionary mapping scene types to confidence scores

        Returns:
            Tuple of (best_scene_type, confidence)
        """
        return self.scene_scoring_engine.determine_scene_type(scene_scores)

    def _fuse_scene_scores(self, yolo_scene_scores: Dict[str, float],
                          clip_scene_scores: Dict[str, float],
                          num_yolo_detections: int = 0,
                          avg_yolo_confidence: float = 0.0,
                          lighting_info: Optional[Dict] = None,
                          places365_info: Optional[Dict] = None) -> Dict[str, float]:
        """
        Fuse scene scores from YOLO-based object detection, CLIP-based analysis, and Places365.

        Args:
            yolo_scene_scores: Scene scores based on YOLO object detection
            clip_scene_scores: Scene scores based on CLIP analysis
            num_yolo_detections: Total number of non-landmark objects detected by YOLO
            avg_yolo_confidence: Average confidence of non-landmark objects detected by YOLO
            lighting_info: Optional lighting condition analysis results
            places365_info: Optional Places365 scene classification results

        Returns:
            Dict: Fused scene scores incorporating all analysis sources
        """
        return self.scene_scoring_engine.fuse_scene_scores(
            yolo_scene_scores, clip_scene_scores, num_yolo_detections,
            avg_yolo_confidence, lighting_info, places365_info
        )

    def _get_alternative_scene_type(self, landmark_scene_type, detected_objects, scene_scores):
        """
        Select appropriate alternative type for landmark scene types.

        Args:
            landmark_scene_type: Original landmark scene type
            detected_objects: List of detected objects
            scene_scores: All scene type scores

        Returns:
            str: Appropriate alternative scene type
        """
        return self.landmark_processing_manager.get_alternative_scene_type(
            landmark_scene_type, detected_objects, scene_scores
        )

    def _remove_landmark_references(self, text):
        """
        Remove all landmark references from text.

        Args:
            text: Input text

        Returns:
            str: Text with landmark references removed
        """
        return self.landmark_processing_manager.remove_landmark_references(text)

    def _define_image_regions(self):
        """Define regions of the image for spatial analysis (3x3 grid)."""
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

    def get_component_status(self) -> Dict[str, bool]:
        """
        Get the initialization status of all components.

        Returns:
            Dictionary mapping component names to their initialization status
        """
        return self.component_initializer.get_initialization_summary()

    def is_component_available(self, component_name: str) -> bool:
        """
        Check if a specific component is available and properly initialized.

        Args:
            component_name: Name of the component to check

        Returns:
            bool: Whether the component is available
        """
        return self.component_initializer.is_component_available(component_name)

    def update_landmark_enable_status(self, enable_landmark: bool):
        """
        Update the landmark detection enable status across all components.

        Args:
            enable_landmark: Whether to enable landmark detection
        """
        self.enable_landmark = enable_landmark
        self.use_landmark_detection = enable_landmark

        # Update all related components
        self.component_initializer.update_landmark_enable_status(enable_landmark)
        self.scene_scoring_engine.update_enable_landmark_status(enable_landmark)
        self.landmark_processing_manager.update_enable_landmark_status(enable_landmark)

        # Update the coordinator's enable_landmark status
        if hasattr(self.scene_analysis_coordinator, 'enable_landmark'):
            self.scene_analysis_coordinator.enable_landmark = enable_landmark
