import numpy as np
import cv2
import logging
import traceback
from typing import Dict, Any, Optional

from configuration_manager import ConfigurationManager
from feature_extractor import FeatureExtractor
from indoor_outdoor_classifier import IndoorOutdoorClassifier
from lighting_condition_analyzer import LightingConditionAnalyzer


class LightingAnalyzer:
    """
    Comprehensive lighting analysis system facade that coordinates feature extraction,
    indoor/outdoor classification, and lighting condition determination.
    此class是一個總窗口，主要匯總各式光線分析相關的class

    This facade class maintains the original interface while internally delegating
    work to specialized components for improved maintainability and modularity.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the lighting analyzer with configuration.

        Args:
            config: Optional configuration dictionary. If None, uses default configuration.
        """
        self.logger = self._setup_logger()

        try:
            # Initialize configuration manager
            self.config_manager = ConfigurationManager()

            # Override default configuration if provided
            if config is not None:
                self._update_configuration(config)

            # Initialize specialized components
            self.feature_extractor = FeatureExtractor(self.config_manager)
            self.indoor_outdoor_classifier = IndoorOutdoorClassifier(self.config_manager)
            self.lighting_condition_analyzer = LightingConditionAnalyzer(self.config_manager)

            # Legacy configuration access for backward compatibility
            self.config = self.config_manager.get_legacy_config_dict()

            self.logger.info("LightingAnalyzer initialized successfully")

        except Exception as e:
            self.logger.error(f"Error initializing LightingAnalyzer: {str(e)}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            raise

    def _setup_logger(self) -> logging.Logger:
        """Set up logger for lighting analysis operations."""
        logger = logging.getLogger(f"{__name__}.LightingAnalyzer")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

    def _update_configuration(self, config: Dict[str, Any]) -> None:
        """
        Update configuration manager with provided configuration dictionary.

        Args:
            config: Configuration dictionary to update existing configuration.
        """
        try:
            # Update configuration through the manager's internal method
            self.config_manager._update_from_dict(config)
            self.logger.debug("Configuration updated successfully")

        except Exception as e:
            self.logger.warning(f"Error updating configuration: {str(e)}")
            self.logger.warning("Continuing with default configuration")

    def analyze(self, image, places365_info: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Analyze lighting conditions of an image.

        This is the main entry point that maintains compatibility with the original interface
        while leveraging the new modular architecture internally.

        Args:
            image: Input image (numpy array or PIL Image).
            places365_info: Optional Places365 classification information containing
                          scene type, confidence, attributes, and indoor/outdoor classification.

        Returns:
            Dictionary containing comprehensive lighting analysis results including:
            - time_of_day: Specific lighting condition classification
            - confidence: Confidence score for the classification
            - is_indoor: Boolean indicating indoor/outdoor classification
            - indoor_probability: Probability score for indoor classification
            - brightness: Brightness analysis metrics
            - color_info: Color characteristic analysis
            - texture_info: Texture and gradient analysis
            - structure_info: Structural feature analysis
            - diagnostics: Detailed diagnostic information (if enabled)
        """
        try:
            self.logger.debug("Starting comprehensive lighting analysis")

            # Step 1: Validate and preprocess input image
            processed_image = self._preprocess_image(image)
            if processed_image is None:
                return self._get_error_result("Invalid image input")

            # Step 2: Extract comprehensive features
            self.logger.debug("Extracting image features")
            features = self.feature_extractor.extract_features(processed_image)

            if not features or "avg_brightness" not in features:
                return self._get_error_result("Feature extraction failed")

            # Step 3: Classify indoor/outdoor with Places365 integration
            self.logger.debug("Performing indoor/outdoor classification")
            indoor_outdoor_result = self.indoor_outdoor_classifier.classify(
                features, places365_info
            )

            is_indoor = indoor_outdoor_result["is_indoor"]
            indoor_probability = indoor_outdoor_result["indoor_probability"]

            # Step 4: Determine specific lighting conditions
            self.logger.debug(f"Analyzing lighting conditions for {'indoor' if is_indoor else 'outdoor'} scene")
            lighting_result = self.lighting_condition_analyzer.analyze_lighting_conditions(
                features, is_indoor, places365_info
            )

            # Step 5: Consolidate comprehensive results
            result = self._consolidate_analysis_results(
                lighting_result, indoor_outdoor_result, features
            )

            self.logger.info(f"Analysis complete: {result['time_of_day']} "
                           f"({'indoor' if result['is_indoor'] else 'outdoor'}) "
                           f"confidence: {result['confidence']:.3f}")

            return result

        except Exception as e:
            self.logger.error(f"Error in lighting analysis: {str(e)}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return self._get_error_result(str(e))

    def _preprocess_image(self, image) -> Optional[np.ndarray]:
        """
        Preprocess input image to ensure consistent format for analysis.

        Args:
            image: Input image in various possible formats.

        Returns:
            Preprocessed image as RGB numpy array, or None if preprocessing failed.
        """
        try:
            # Convert to numpy array if needed
            if not isinstance(image, np.ndarray):
                image_np = np.array(image)
            else:
                image_np = image.copy()

            # Validate basic image properties
            if len(image_np.shape) < 2:
                self.logger.error("Image must have at least 2 dimensions")
                return None

            height, width = image_np.shape[:2]
            if height == 0 or width == 0:
                self.logger.error(f"Invalid image dimensions: {height}x{width}")
                return None

            # Handle different color formats and convert to RGB
            if len(image_np.shape) == 2:
                # 灰階 to RGB
                image_rgb = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
            elif image_np.shape[2] == 3:
                # Handle BGR vs RGB
                if not isinstance(image, np.ndarray):
                    # PIL images are typically RGB
                    image_rgb = image_np
                else:
                    # OpenCV arrays are typically BGR, convert to RGB
                    image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
            elif image_np.shape[2] == 4:
                # RGBA to RGB
                if not isinstance(image, np.ndarray):
                    # PIL RGBA to RGB
                    image_rgb = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)
                else:
                    # OpenCV BGRA to RGB
                    image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGRA2RGB)
            else:
                self.logger.error(f"Unsupported image format with shape: {image_np.shape}")
                return None

            # Ensure uint8 data type
            if image_rgb.dtype != np.uint8:
                if image_rgb.dtype in [np.float32, np.float64]:
                    # Assume normalized float values
                    if image_rgb.max() <= 1.0:
                        image_rgb = (image_rgb * 255).astype(np.uint8)
                    else:
                        image_rgb = image_rgb.astype(np.uint8)
                else:
                    image_rgb = image_rgb.astype(np.uint8)

            self.logger.debug(f"Preprocessed image: {image_rgb.shape}, dtype: {image_rgb.dtype}")
            return image_rgb

        except Exception as e:
            self.logger.error(f"Error preprocessing image: {str(e)}")
            return None

    def _consolidate_analysis_results(self, lighting_result: Dict[str, Any],
                                     indoor_outdoor_result: Dict[str, Any],
                                     features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Consolidate results from all analysis components into final output format.

        Args:
            lighting_result: Results from lighting condition analysis.
            indoor_outdoor_result: Results from indoor/outdoor classification.
            features: Extracted image features.

        Returns:
            Consolidated analysis results in the expected output format.
        """
        # Extract core results
        time_of_day = lighting_result["time_of_day"]
        confidence = lighting_result["confidence"]
        is_indoor = indoor_outdoor_result["is_indoor"]
        indoor_probability = indoor_outdoor_result["indoor_probability"]

        # Organize brightness information
        brightness_info = {
            "average": float(features.get("avg_brightness", 0.0)),
            "std_dev": float(features.get("brightness_std", 0.0)),
            "dark_ratio": float(features.get("dark_pixel_ratio", 0.0)),
            "bright_ratio": float(features.get("bright_pixel_ratio", 0.0))
        }

        # Organize color information
        color_info = {
            "blue_ratio": float(features.get("blue_ratio", 0.0)),
            "sky_like_blue_ratio": float(features.get("sky_like_blue_ratio", 0.0)),
            "yellow_orange_ratio": float(features.get("yellow_orange_ratio", 0.0)),
            "gray_ratio": float(features.get("gray_ratio", 0.0)),
            "avg_saturation": float(features.get("avg_saturation", 0.0)),
            "sky_region_brightness_ratio": float(features.get("sky_region_brightness_ratio", 1.0)),
            "sky_region_saturation": float(features.get("sky_region_saturation", 0.0)),
            "sky_region_blue_dominance": float(features.get("sky_region_blue_dominance", 0.0)),
            "color_atmosphere": features.get("color_atmosphere", "neutral"),
            "warm_ratio": float(features.get("warm_ratio", 0.0)),
            "cool_ratio": float(features.get("cool_ratio", 0.0))
        }

        # Organize texture information
        texture_info = {
            "gradient_ratio_vertical_horizontal": float(features.get("gradient_ratio_vertical_horizontal", 0.0)),
            "top_region_texture_complexity": float(features.get("top_region_texture_complexity", 0.0)),
            "shadow_clarity_score": float(features.get("shadow_clarity_score", 0.5))
        }

        # Organize structure information
        structure_info = {
            "ceiling_likelihood": float(features.get("ceiling_likelihood", 0.0)),
            "boundary_clarity": float(features.get("boundary_clarity", 0.0)),
            "openness_top_edge": float(features.get("openness_top_edge", 0.5))
        }

        # Compile final result
        result = {
            "time_of_day": time_of_day,
            "confidence": float(confidence),
            "is_indoor": is_indoor,
            "indoor_probability": float(indoor_probability),
            "brightness": brightness_info,
            "color_info": color_info,
            "texture_info": texture_info,
            "structure_info": structure_info
        }

        # Add diagnostic information if enabled
        if self.config_manager.algorithm_parameters.include_diagnostics:
            diagnostics = {}

            # Combine diagnostics from all components
            if "diagnostics" in lighting_result:
                diagnostics["lighting_diagnostics"] = lighting_result["diagnostics"]

            if "diagnostics" in indoor_outdoor_result:
                diagnostics["indoor_outdoor_diagnostics"] = indoor_outdoor_result["diagnostics"]

            if "feature_contributions" in indoor_outdoor_result:
                diagnostics["feature_contributions"] = indoor_outdoor_result["feature_contributions"]

            result["diagnostics"] = diagnostics

        return result

    def _get_error_result(self, error_message: str) -> Dict[str, Any]:
        """
        Generate standardized error result format.

        Args:
            error_message: Description of the error that occurred.

        Returns:
            Dictionary containing error result with safe default values.
        """
        return {
            "time_of_day": "unknown",
            "confidence": 0.0,
            "is_indoor": False,
            "indoor_probability": 0.5,
            "brightness": {
                "average": 100.0,
                "std_dev": 50.0,
                "dark_ratio": 0.0,
                "bright_ratio": 0.0
            },
            "color_info": {
                "blue_ratio": 0.0,
                "sky_like_blue_ratio": 0.0,
                "yellow_orange_ratio": 0.0,
                "gray_ratio": 0.0,
                "avg_saturation": 100.0,
                "sky_region_brightness_ratio": 1.0,
                "sky_region_saturation": 0.0,
                "sky_region_blue_dominance": 0.0,
                "color_atmosphere": "neutral",
                "warm_ratio": 0.0,
                "cool_ratio": 0.0
            },
            "texture_info": {
                "gradient_ratio_vertical_horizontal": 1.0,
                "top_region_texture_complexity": 0.5,
                "shadow_clarity_score": 0.5
            },
            "structure_info": {
                "ceiling_likelihood": 0.0,
                "boundary_clarity": 0.0,
                "openness_top_edge": 0.5
            },
            "error": error_message
        }

    def get_configuration(self) -> Dict[str, Any]:
        """
        Get current configuration as dictionary for backward compatibility.

        Returns:
            Dictionary containing all current configuration parameters.
        """
        return self.config_manager.get_legacy_config_dict()

    def update_configuration(self, config_updates: Dict[str, Any]) -> None:
        """
        Update configuration parameters.

        Args:
            config_updates: Dictionary containing configuration parameters to update.
        """
        try:
            self.config_manager._update_from_dict(config_updates)
            # Update legacy config reference
            self.config = self.config_manager.get_legacy_config_dict()
            self.logger.info("Configuration updated successfully")

        except Exception as e:
            self.logger.error(f"Error updating configuration: {str(e)}")
            raise

    def validate_configuration(self) -> bool:
        """
        Validate current configuration for logical consistency.

        Returns:
            True if configuration is valid, False otherwise.
        """
        try:
            validation_errors = self.config_manager.validate_configuration()

            if validation_errors:
                self.logger.error("Configuration validation failed:")
                for error in validation_errors:
                    self.logger.error(f"  - {error}")
                return False

            self.logger.info("Configuration validation passed")
            return True

        except Exception as e:
            self.logger.error(f"Error during configuration validation: {str(e)}")
            return False

    def save_configuration(self, filepath: str) -> None:
        """
        Save current configuration to file.

        Args:
            filepath: Path where to save the configuration file.
        """
        try:
            self.config_manager.save_to_file(filepath)
            self.logger.info(f"Configuration saved to {filepath}")

        except Exception as e:
            self.logger.error(f"Error saving configuration: {str(e)}")
            raise

    def load_configuration(self, filepath: str) -> None:
        """
        Load configuration from file.

        Args:
            filepath: Path to the configuration file to load.
        """
        try:
            self.config_manager.load_from_file(filepath)
            # Update legacy config reference
            self.config = self.config_manager.get_legacy_config_dict()
            self.logger.info(f"Configuration loaded from {filepath}")

        except Exception as e:
            self.logger.error(f"Error loading configuration: {str(e)}")
            raise
