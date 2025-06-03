import numpy as np
import logging
import traceback
from typing import Dict, Any, Optional, List, Tuple
from configuration_manager import ConfigurationManager


class LightingConditionAnalyzer:
    """
    Determines specific lighting conditions and time of day based on scene analysis.
    此class 會判斷一些光線的特定場景

    This class analyzes lighting characteristics including natural and artificial illumination,
    color temperature patterns, and temporal indicators to classify scenes into specific
    lighting categories such as day clear, night with lights, indoor artificial, etc.
    """

    def __init__(self, config_manager: ConfigurationManager):
        """
        Initialize the lighting condition analyzer.

        Args:
            config_manager: Configuration manager instance for accessing thresholds and parameters.
        """
        self.config_manager = config_manager
        self.logger = self._setup_logger()

        # Internal threshold constants for Places365 analysis
        self.P365_ATTRIBUTE_CONF_THRESHOLD = 0.60
        self.P365_SCENE_MODERATE_CONF_THRESHOLD = 0.45
        self.P365_SCENE_HIGH_CONF_THRESHOLD = 0.70

        # Scene type keyword definitions
        self.P365_OUTDOOR_SCENE_KEYWORDS = [
            "street", "road", "highway", "park", "beach", "mountain", "forest", "field",
            "outdoor", "sky", "coast", "courtyard", "square", "plaza", "bridge",
            "parking", "playground", "stadium", "construction", "river", "ocean", "desert",
            "garden", "trail", "natural_landmark", "airport_outdoor", "train_station_outdoor",
            "bus_station_outdoor", "intersection", "crosswalk", "sidewalk", "pathway"
        ]

        self.P365_INDOOR_RESTAURANT_KEYWORDS = [
            "restaurant", "bar", "cafe", "dining_room", "pub", "bistro", "eatery"
        ]

    def _setup_logger(self) -> logging.Logger:
        """Set up logger for lighting condition analysis operations."""
        logger = logging.getLogger(f"{__name__}.LightingConditionAnalyzer")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

    def analyze_lighting_conditions(self, features: Dict[str, Any], is_indoor: bool,
                                   places365_info: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Determine specific lighting conditions based on features and scene context.

        Args:
            features: Dictionary containing extracted image features.
            is_indoor: Boolean indicating whether the scene is indoor (from previous classification).
            places365_info: Optional Places365 classification information.

        Returns:
            Dictionary containing lighting analysis results including time_of_day, confidence,
            and diagnostic information.
        """
        try:
            self.logger.debug(f"Starting lighting analysis for {'indoor' if is_indoor else 'outdoor'} scene")

            # Initialize analysis results
            time_of_day = "unknown"
            confidence = 0.5
            diagnostics = {}

            # Extract Places365 context
            p365_context = self._extract_places365_context(places365_info, diagnostics)

            # Priority 1: Use Places365 attributes if highly confident
            attribute_result = self._analyze_places365_attributes(
                p365_context, is_indoor, features, diagnostics
            )

            if attribute_result["determined"] and attribute_result["confidence"] >= 0.75:
                self.logger.debug(f"High-confidence Places365 attribute determination: {attribute_result['time_of_day']}")
                return {
                    "time_of_day": attribute_result["time_of_day"],
                    "confidence": attribute_result["confidence"],
                    "diagnostics": diagnostics
                }

            # Priority 2: Visual feature analysis with Places365 scene context
            visual_result = self._analyze_visual_features(
                features, is_indoor, p365_context, diagnostics
            )

            time_of_day = visual_result["time_of_day"]
            confidence = visual_result["confidence"]

            # Combine with attribute result if it exists but wasn't decisive
            if attribute_result["determined"]:
                combined_result = self._combine_attribute_and_visual_results(
                    attribute_result, visual_result, diagnostics
                )
                time_of_day = combined_result["time_of_day"]
                confidence = combined_result["confidence"]

            # Priority 3: Special lighting refinement (neon, sodium vapor)
            refined_result = self._apply_special_lighting_refinement(
                time_of_day, confidence, features, is_indoor, p365_context, diagnostics
            )

            time_of_day = refined_result["time_of_day"]
            confidence = refined_result["confidence"]

            # Final confidence clamping
            confidence = min(0.95, max(0.50, confidence))

            # Record final results
            diagnostics["final_lighting_time_of_day"] = time_of_day
            diagnostics["final_lighting_confidence"] = round(confidence, 3)

            self.logger.debug(f"Lighting analysis complete: {time_of_day} (confidence: {confidence:.3f})")

            return {
                "time_of_day": time_of_day,
                "confidence": confidence,
                "diagnostics": diagnostics
            }

        except Exception as e:
            self.logger.error(f"Error in lighting condition analysis: {str(e)}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return self._get_default_lighting_result()

    def _extract_places365_context(self, places365_info: Optional[Dict],
                                  diagnostics: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and validate Places365 context information for lighting analysis."""
        context = {
            "mapped_scene": "unknown",
            "attributes": [],
            "confidence": 0.0
        }

        if places365_info:
            context["mapped_scene"] = places365_info.get('mapped_scene_type', 'unknown').lower()
            context["attributes"] = [attr.lower() for attr in places365_info.get('attributes', [])]
            context["confidence"] = places365_info.get('confidence', 0.0)

            diagnostics["p365_context_for_lighting"] = (
                f"P365 Scene: {context['mapped_scene']}, Attrs: {context['attributes']}, "
                f"Conf: {context['confidence']:.2f}"
            )

        return context

    def _analyze_places365_attributes(self, p365_context: Dict[str, Any], is_indoor: bool,
                                     features: Dict[str, Any], diagnostics: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze Places365 attributes for lighting condition determination."""
        if (not p365_context["attributes"] or
            p365_context["confidence"] <= self.P365_ATTRIBUTE_CONF_THRESHOLD):
            return {"determined": False, "time_of_day": "unknown", "confidence": 0.5}

        confidence = p365_context["confidence"]
        attributes = p365_context["attributes"]
        mapped_scene = p365_context["mapped_scene"]

        # Outdoor attribute analysis
        if not is_indoor:
            outdoor_result = self._analyze_outdoor_attributes(
                attributes, mapped_scene, confidence, diagnostics
            )
            if outdoor_result["determined"]:
                return outdoor_result

        # Indoor attribute analysis
        if is_indoor:
            indoor_result = self._analyze_indoor_attributes(
                attributes, mapped_scene, features, confidence, diagnostics
            )
            if indoor_result["determined"]:
                return indoor_result

        return {"determined": False, "time_of_day": "unknown", "confidence": 0.5}

    def _analyze_outdoor_attributes(self, attributes: List[str], mapped_scene: str,
                                   confidence: float, diagnostics: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze Places365 attributes for outdoor lighting conditions."""
        base_confidence_boost = (confidence - self.P365_ATTRIBUTE_CONF_THRESHOLD) * 0.25

        if "sunny" in attributes or "clear sky" in attributes:
            final_confidence = 0.85 + base_confidence_boost
            diagnostics["reason"] = "P365 attribute: sunny/clear sky (Outdoor)."
            return {
                "determined": True,
                "time_of_day": "day_clear",
                "confidence": final_confidence
            }

        elif "nighttime" in attributes or "night" in attributes:
            if ("artificial lighting" in attributes or "man-made lighting" in attributes or
                any(kw in mapped_scene for kw in ["street", "city", "road", "urban", "downtown"])):
                final_confidence = 0.82 + base_confidence_boost * 0.8
                diagnostics["reason"] = "P365 attribute: nighttime with artificial/street lights (Outdoor)."
                return {
                    "determined": True,
                    "time_of_day": "night_with_lights",
                    "confidence": final_confidence
                }
            else:
                final_confidence = 0.78 + base_confidence_boost * 0.8
                diagnostics["reason"] = "P365 attribute: nighttime, dark (Outdoor)."
                return {
                    "determined": True,
                    "time_of_day": "night_dark",
                    "confidence": final_confidence
                }

        elif "cloudy" in attributes or "overcast" in attributes:
            final_confidence = 0.80 + base_confidence_boost
            diagnostics["reason"] = "P365 attribute: cloudy/overcast (Outdoor)."
            return {
                "determined": True,
                "time_of_day": "day_cloudy_overcast",
                "confidence": final_confidence
            }

        return {"determined": False, "time_of_day": "unknown", "confidence": 0.5}

    def _analyze_indoor_attributes(self, attributes: List[str], mapped_scene: str,
                                  features: Dict[str, Any], confidence: float,
                                  diagnostics: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze Places365 attributes for indoor lighting conditions."""
        base_confidence_boost = (confidence - self.P365_ATTRIBUTE_CONF_THRESHOLD) * 0.20
        avg_brightness = features.get("avg_brightness", 128.0)

        if "artificial lighting" in attributes or "man-made lighting" in attributes:
            base_indoor_conf = 0.70 + base_confidence_boost
            thresholds = self.config_manager.lighting_thresholds

            if avg_brightness > thresholds.indoor_bright_thresh:
                time_of_day = "indoor_bright_artificial"
                final_confidence = base_indoor_conf + 0.10
            elif avg_brightness > thresholds.indoor_moderate_thresh:
                time_of_day = "indoor_moderate_artificial"
                final_confidence = base_indoor_conf
            else:
                time_of_day = "indoor_dim_artificial"
                final_confidence = base_indoor_conf - 0.05

            diagnostics["reason"] = (
                f"P365 attribute: artificial lighting (Indoor), "
                f"brightness based category: {time_of_day}."
            )
            return {
                "determined": True,
                "time_of_day": time_of_day,
                "confidence": final_confidence
            }

        elif "natural lighting" in attributes:
            is_applicable_scene = (
                self._check_home_environment_pattern(features) or
                any(kw in mapped_scene for kw in ["living_room", "bedroom", "sunroom"])
            )
            if is_applicable_scene:
                final_confidence = 0.80 + base_confidence_boost
                diagnostics["reason"] = "P365 attribute: natural lighting in residential/applicable indoor scene."
                return {
                    "determined": True,
                    "time_of_day": "indoor_residential_natural",
                    "confidence": final_confidence
                }

        return {"determined": False, "time_of_day": "unknown", "confidence": 0.5}

    def _analyze_visual_features(self, features: Dict[str, Any], is_indoor: bool,
                                p365_context: Dict[str, Any], diagnostics: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze visual features for lighting condition determination."""
        if is_indoor:
            return self._analyze_indoor_visual_features(features, p365_context, diagnostics)
        else:
            return self._analyze_outdoor_visual_features(features, p365_context, diagnostics)

    def _analyze_indoor_visual_features(self, features: Dict[str, Any], p365_context: Dict[str, Any],
                                       diagnostics: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze visual features for indoor lighting conditions."""
        avg_brightness = features.get("avg_brightness", 128.0)
        thresholds = self.config_manager.lighting_thresholds

        # Extract relevant features
        sky_blue_in_sky_region = features.get("sky_region_blue_dominance", 0.0)
        sky_region_is_brighter = features.get("sky_region_brightness_ratio", 1.0) > 1.05
        is_likely_home_environment = self._check_home_environment_pattern(features)

        # Lighting and structural features
        circular_lights = features.get("circular_light_count", 0)
        bright_spots_overall = features.get("bright_spot_count", 0)
        brightness_uniformity = features.get("brightness_uniformity", 0.0)
        warm_ratio = features.get("warm_ratio", 0.0)

        # Natural light hints calculation
        natural_light_hints = 0.0
        if sky_blue_in_sky_region > 0.05 and sky_region_is_brighter:
            natural_light_hints += 1.0
        if brightness_uniformity > 0.65 and features.get("brightness_std", 100.0) < 70:
            natural_light_hints += 1.0
        if warm_ratio > 0.15 and avg_brightness > 110:
            natural_light_hints += 0.5

        # Designer lighting detection
        is_designer_lit = (
            (circular_lights > 0 or bright_spots_overall > 2) and
            brightness_uniformity > 0.6 and warm_ratio > 0.2 and avg_brightness > 90
        )

        # Brightness-based classification
        if avg_brightness > thresholds.indoor_bright_thresh:
            return self._classify_bright_indoor(
                features, natural_light_hints, is_designer_lit, is_likely_home_environment,
                p365_context, diagnostics
            )
        elif avg_brightness > thresholds.indoor_moderate_thresh:
            return self._classify_moderate_indoor(
                features, is_designer_lit, is_likely_home_environment, p365_context, diagnostics
            )
        else:
            return self._classify_dim_indoor(features, diagnostics)

    def _classify_bright_indoor(self, features: Dict[str, Any], natural_light_hints: float,
                               is_designer_lit: bool, is_likely_home_environment: bool,
                               p365_context: Dict[str, Any], diagnostics: Dict[str, Any]) -> Dict[str, Any]:
        """Classify bright indoor lighting conditions."""
        mapped_scene = p365_context["mapped_scene"]
        sky_blue_in_sky_region = features.get("sky_region_blue_dominance", 0.0)
        sky_region_is_brighter = features.get("sky_region_brightness_ratio", 1.0) > 1.05

        # Natural residential lighting
        if (natural_light_hints >= 1.5 and
            (is_likely_home_environment or any(kw in mapped_scene for kw in ["home", "residential", "living", "bedroom"]))):
            return {
                "time_of_day": "indoor_residential_natural",
                "confidence": 0.82
            }

        # Designer residential lighting
        elif (is_designer_lit and
              (is_likely_home_environment or any(kw in mapped_scene for kw in ["home", "designer", "modern_interior"]))):
            return {
                "time_of_day": "indoor_designer_residential",
                "confidence": 0.85
            }

        # Mixed natural/artificial lighting
        elif sky_blue_in_sky_region > 0.03 and sky_region_is_brighter:
            return {
                "time_of_day": "indoor_bright_natural_mix",
                "confidence": 0.78
            }

        # Pure artificial lighting
        else:
            return {
                "time_of_day": "indoor_bright_artificial",
                "confidence": 0.75
            }

    def _classify_moderate_indoor(self, features: Dict[str, Any], is_designer_lit: bool,
                                 is_likely_home_environment: bool, p365_context: Dict[str, Any],
                                 diagnostics: Dict[str, Any]) -> Dict[str, Any]:
        """Classify moderate brightness indoor lighting conditions."""
        mapped_scene = p365_context["mapped_scene"]
        confidence = p365_context["confidence"]
        warm_ratio = features.get("warm_ratio", 0.0)
        yellow_orange_ratio = features.get("yellow_orange_ratio", 0.0)

        # Designer residential lighting
        if (is_designer_lit and
            (is_likely_home_environment or any(kw in mapped_scene for kw in ["home", "designer"]))):
            return {
                "time_of_day": "indoor_designer_residential",
                "confidence": 0.78
            }

        # Restaurant/bar lighting
        elif warm_ratio > 0.35 and yellow_orange_ratio > 0.1:
            return self._classify_restaurant_bar_lighting(
                p365_context, features, diagnostics
            )

        # Standard moderate artificial
        else:
            return {
                "time_of_day": "indoor_moderate_artificial",
                "confidence": 0.70
            }

    def _classify_restaurant_bar_lighting(self, p365_context: Dict[str, Any],
                                         features: Dict[str, Any],
                                         diagnostics: Dict[str, Any]) -> Dict[str, Any]:
        """Classify restaurant/bar specific lighting conditions."""
        mapped_scene = p365_context["mapped_scene"]
        confidence = p365_context["confidence"]

        # Strong P365 restaurant/bar confirmation
        if (any(kw in mapped_scene for kw in self.P365_INDOOR_RESTAURANT_KEYWORDS) and
            confidence > self.P365_SCENE_MODERATE_CONF_THRESHOLD):
            diagnostics["visual_analysis_reason"] = (
                "Visual: Moderate warm tones. P365 context confirms restaurant/bar."
            )
            return {
                "time_of_day": "indoor_restaurant_bar",
                "confidence": 0.80 + confidence * 0.15
            }

        # P365 outdoor conflict detection
        elif (any(kw in mapped_scene for kw in self.P365_OUTDOOR_SCENE_KEYWORDS) and
              confidence > self.P365_SCENE_MODERATE_CONF_THRESHOLD):
            diagnostics["visual_analysis_reason"] = (
                "Visual: Moderate warm. CONFLICT: LA says indoor but P365 scene is outdoor. "
                "Defaulting to general indoor artificial."
            )
            diagnostics["conflict_is_indoor_vs_p365_scene_for_restaurant_bar"] = True
            return {
                "time_of_day": "indoor_moderate_artificial",
                "confidence": 0.55
            }

        # Neutral P365 context
        else:
            diagnostics["visual_analysis_reason"] = (
                "Visual: Moderate warm tones, typical of restaurant/bar. P365 context neutral or weak."
            )
            return {
                "time_of_day": "indoor_restaurant_bar",
                "confidence": 0.70
            }

    def _classify_dim_indoor(self, features: Dict[str, Any],
                            diagnostics: Dict[str, Any]) -> Dict[str, Any]:
        """Classify dim indoor lighting conditions."""
        warm_ratio = features.get("warm_ratio", 0.0)
        yellow_orange_ratio = features.get("yellow_orange_ratio", 0.0)

        if warm_ratio > 0.45 and yellow_orange_ratio > 0.15:
            return {
                "time_of_day": "indoor_dim_warm",
                "confidence": 0.75
            }
        else:
            return {
                "time_of_day": "indoor_dim_general",
                "confidence": 0.70
            }

    def _analyze_outdoor_visual_features(self, features: Dict[str, Any], p365_context: Dict[str, Any],
                                        diagnostics: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze visual features for outdoor lighting conditions."""
        avg_brightness = features.get("avg_brightness", 128.0)
        thresholds = self.config_manager.lighting_thresholds

        # P365 enhanced street scene analysis
        street_result = self._analyze_p365_enhanced_street_scenes(
            features, p365_context, diagnostics
        )
        if street_result["determined"]:
            return street_result

        # Brightness-based outdoor classification
        if avg_brightness < thresholds.outdoor_night_thresh_brightness:
            return self._classify_night_outdoor(features, diagnostics)
        elif (avg_brightness < thresholds.outdoor_dusk_dawn_thresh_brightness and
              self._check_warm_sunset_conditions(features)):
            return self._classify_sunset_sunrise(features, p365_context, diagnostics)
        elif avg_brightness > thresholds.outdoor_day_bright_thresh:
            return self._classify_bright_day_outdoor(features, diagnostics)
        elif avg_brightness > thresholds.outdoor_day_cloudy_thresh:
            return self._classify_cloudy_day_outdoor(features, diagnostics)
        else:
            return self._classify_general_outdoor(features, diagnostics)

    def _analyze_p365_enhanced_street_scenes(self, features: Dict[str, Any], p365_context: Dict[str, Any],
                                            diagnostics: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze outdoor scenes with Places365 street context enhancement."""
        mapped_scene = p365_context["mapped_scene"]
        confidence = p365_context["confidence"]
        thresholds = self.config_manager.lighting_thresholds

        # Check for street scene with warm lighting
        is_street_scene = (
            any(kw in mapped_scene for kw in ["street", "city", "road", "urban", "downtown", "intersection"]) and
            confidence > self.P365_SCENE_MODERATE_CONF_THRESHOLD and
            features.get("color_atmosphere") == "warm"
        )

        if not is_street_scene:
            return {"determined": False, "time_of_day": "unknown", "confidence": 0.5}

        avg_brightness = features.get("avg_brightness", 128.0)
        bright_spots_overall = features.get("bright_spot_count", 0)

        # Night with street lights
        if (avg_brightness < thresholds.outdoor_night_thresh_brightness and
            bright_spots_overall > thresholds.outdoor_night_lights_thresh):
            diagnostics["visual_analysis_reason"] = (
                f"P365 outdoor scene '{mapped_scene}' + visual low-warm light with spots -> night_with_lights."
            )
            return {
                "determined": True,
                "time_of_day": "night_with_lights",
                "confidence": 0.88 + confidence * 0.1
            }

        # Sunset/sunrise conditions
        elif avg_brightness >= thresholds.outdoor_night_thresh_brightness:
            diagnostics["visual_analysis_reason"] = (
                f"P365 outdoor scene '{mapped_scene}' + visual moderate-warm light -> sunset/sunrise."
            )
            return {
                "determined": True,
                "time_of_day": "sunset_sunrise",
                "confidence": 0.88 + confidence * 0.1
            }

        # Very dark conditions
        else:
            diagnostics["visual_analysis_reason"] = (
                f"P365 outdoor scene '{mapped_scene}' + visual very low light -> night_dark."
            )
            return {
                "determined": True,
                "time_of_day": "night_dark",
                "confidence": 0.75 + confidence * 0.1
            }

    def _classify_night_outdoor(self, features: Dict[str, Any],
                               diagnostics: Dict[str, Any]) -> Dict[str, Any]:
        """Classify nighttime outdoor conditions."""
        bright_spots_overall = features.get("bright_spot_count", 0)
        dark_pixel_ratio = features.get("dark_pixel_ratio", 0.0)
        thresholds = self.config_manager.lighting_thresholds

        if bright_spots_overall > thresholds.outdoor_night_lights_thresh:
            confidence = 0.82 + min(0.13, dark_pixel_ratio / 2.5)
            diagnostics["visual_analysis_reason"] = "Visual: Low brightness with light sources (street/car lights)."
            return {
                "time_of_day": "night_with_lights",
                "confidence": confidence
            }
        else:
            confidence = 0.78 + min(0.17, dark_pixel_ratio / 1.8)
            diagnostics["visual_analysis_reason"] = "Visual: Very low brightness outdoor, deep night."
            return {
                "time_of_day": "night_dark",
                "confidence": confidence
            }

    def _classify_sunset_sunrise(self, features: Dict[str, Any], p365_context: Dict[str, Any],
                                diagnostics: Dict[str, Any]) -> Dict[str, Any]:
        """Classify sunset/sunrise outdoor conditions."""
        yellow_orange_ratio = features.get("yellow_orange_ratio", 0.0)
        confidence = 0.75 + min(0.20, yellow_orange_ratio / 1.5)

        diagnostics["visual_analysis_reason"] = "Visual: Moderate brightness, warm tones -> sunset/sunrise."

        # P365 natural scene boost
        mapped_scene = p365_context["mapped_scene"]
        p365_confidence = p365_context["confidence"]

        if (any(kw in mapped_scene for kw in ["beach", "mountain", "lake", "ocean", "desert", "field", "natural_landmark", "sky"]) and
            p365_confidence > self.P365_SCENE_MODERATE_CONF_THRESHOLD):
            confidence = min(0.95, confidence + 0.15)
            diagnostics["visual_analysis_reason"] += f" P365 natural scene '{mapped_scene}' supports."

        return {
            "time_of_day": "sunset_sunrise",
            "confidence": confidence
        }

    def _classify_bright_day_outdoor(self, features: Dict[str, Any],
                                    diagnostics: Dict[str, Any]) -> Dict[str, Any]:
        """Classify bright daytime outdoor conditions."""
        sky_like_blue_in_sky_region = features.get("sky_region_blue_dominance", 0.0)
        sky_region_brightness_ratio = features.get("sky_region_brightness_ratio", 1.0)
        texture_complexity = features.get("top_region_texture_complexity", 0.5)
        thresholds = self.config_manager.lighting_thresholds

        # Clear sky conditions
        if (sky_like_blue_in_sky_region > thresholds.outdoor_day_blue_thresh or
            (sky_region_brightness_ratio > 1.05 and texture_complexity < 0.4)):

            confidence = 0.80 + min(0.15, sky_like_blue_in_sky_region * 2 +
                                  (sky_like_blue_in_sky_region * 1.5 if sky_region_brightness_ratio > 1.05 else 0))
            diagnostics["visual_analysis_reason"] = "Visual: High brightness with blue/sky tones or bright smooth top."

            return {
                "time_of_day": "day_clear",
                "confidence": confidence
            }

        # Stadium/floodlit detection
        brightness_uniformity = features.get("brightness_uniformity", 0.0)
        bright_spots_overall = features.get("bright_spot_count", 0)

        if (brightness_uniformity > 0.70 and
            bright_spots_overall > thresholds.stadium_min_spots_thresh):
            diagnostics["visual_analysis_reason"] = (
                "Visual: Very bright, uniform lighting with multiple sources, suggests floodlights (Outdoor)."
            )
            return {
                "time_of_day": "stadium_or_floodlit_area",
                "confidence": 0.78
            }

        # General bright day
        diagnostics["visual_analysis_reason"] = "Visual: High brightness outdoor, specific sky features unclear."
        return {
            "time_of_day": "day_bright_general",
            "confidence": 0.68
        }

    def _classify_cloudy_day_outdoor(self, features: Dict[str, Any],
                                    diagnostics: Dict[str, Any]) -> Dict[str, Any]:
        """Classify cloudy daytime outdoor conditions."""
        sky_region_brightness_ratio = features.get("sky_region_brightness_ratio", 1.0)
        texture_complexity = features.get("top_region_texture_complexity", 0.5)
        avg_saturation = features.get("avg_saturation", 100.0)
        gray_ratio = features.get("gray_ratio", 0.0)
        brightness_uniformity = features.get("brightness_uniformity", 0.0)
        thresholds = self.config_manager.lighting_thresholds

        # Overcast conditions
        if (sky_region_brightness_ratio > 1.05 and texture_complexity < 0.45 and avg_saturation < 70):
            confidence = 0.75 + min(0.20, gray_ratio / 1.5 + (brightness_uniformity - 0.5) / 1.5)
            diagnostics["visual_analysis_reason"] = (
                "Visual: Good brightness, uniform bright top, lower saturation -> overcast."
            )
            return {
                "time_of_day": "day_cloudy_overcast",
                "confidence": confidence
            }

        # Gray cloudy conditions
        elif gray_ratio > thresholds.outdoor_day_gray_thresh:
            confidence = 0.72 + min(0.23, gray_ratio / 1.8)
            diagnostics["visual_analysis_reason"] = "Visual: Good brightness with higher gray tones."
            return {
                "time_of_day": "day_cloudy_gray",
                "confidence": confidence
            }

        # General bright outdoor
        else:
            diagnostics["visual_analysis_reason"] = "Visual: Bright outdoor, specific type less clear."
            return {
                "time_of_day": "day_bright_general",
                "confidence": 0.68
            }

    def _classify_general_outdoor(self, features: Dict[str, Any],
                                 diagnostics: Dict[str, Any]) -> Dict[str, Any]:
        """Classify general outdoor conditions when specific patterns are unclear."""
        color_atmosphere = features.get("color_atmosphere", "neutral")
        yellow_orange_ratio = features.get("yellow_orange_ratio", 0.0)
        sky_like_blue_in_sky_region = features.get("sky_region_blue_dominance", 0.0)

        # Potential sunset/sunrise with low confidence
        if color_atmosphere == "warm" and yellow_orange_ratio > 0.08:
            diagnostics["visual_analysis_reason"] = (
                "Visual: Outdoor, specific conditions less clear; broader visual cues suggest warm lighting."
            )
            return {
                "time_of_day": "sunset_sunrise_low_confidence",
                "confidence": 0.62
            }

        # Potential hazy day conditions
        elif sky_like_blue_in_sky_region > 0.02:
            diagnostics["visual_analysis_reason"] = (
                "Visual: Outdoor, specific conditions less clear; some blue tones suggest daylight."
            )
            return {
                "time_of_day": "day_hazy_or_partly_cloudy",
                "confidence": 0.62
            }

        # Unknown outdoor daylight
        else:
            diagnostics["visual_analysis_reason"] = (
                "Visual: Outdoor, specific conditions less clear; broader visual cues."
            )
            return {
                "time_of_day": "outdoor_unknown_daylight",
                "confidence": 0.58
            }

    def _apply_commercial_indoor_refinement(self, features: Dict[str, Any], p365_context: Dict[str, Any],
                                           time_of_day: str, confidence: float) -> Dict[str, Any]:
        """Apply commercial indoor lighting refinement if conditions are met."""
        # Skip if already classified as residential, restaurant, or bar
        if any(category in time_of_day for category in ["residential", "restaurant", "bar"]):
            return {"time_of_day": time_of_day, "confidence": confidence}

        # Skip if P365 suggests home environment
        mapped_scene = p365_context["mapped_scene"]
        if any(kw in mapped_scene for kw in ["home", "residential"]):
            return {"time_of_day": time_of_day, "confidence": confidence}

        # Check commercial lighting indicators
        avg_brightness = features.get("avg_brightness", 100.0)
        bright_spots_overall = features.get("bright_spot_count", 0)
        light_dist_uniformity = features.get("light_distribution_uniformity", 0.5)
        ceiling_likelihood = features.get("ceiling_likelihood", 0.0)
        thresholds = self.config_manager.lighting_thresholds

        if (avg_brightness > thresholds.commercial_min_brightness_thresh and
            bright_spots_overall > thresholds.commercial_min_spots_thresh and
            (light_dist_uniformity > 0.5 or ceiling_likelihood > 0.4)):

            refined_confidence = 0.70 + min(0.2, bright_spots_overall * 0.02)
            return {
                "time_of_day": "indoor_commercial",
                "confidence": refined_confidence
            }

        return {"time_of_day": time_of_day, "confidence": confidence}

    def _apply_special_lighting_refinement(self, time_of_day: str, confidence: float,
                                          features: Dict[str, Any], is_indoor: bool,
                                          p365_context: Dict[str, Any],
                                          diagnostics: Dict[str, Any]) -> Dict[str, Any]:
        """Apply special lighting refinement for neon and sodium vapor lighting."""
        # Apply commercial refinement for indoor scenes first
        if is_indoor:
            commercial_result = self._apply_commercial_indoor_refinement(
                features, p365_context, time_of_day, confidence
            )
            time_of_day = commercial_result["time_of_day"]
            confidence = commercial_result["confidence"]

        # Check for neon/sodium vapor lighting conditions
        is_current_night_or_dim_warm = "night" in time_of_day or time_of_day == "indoor_dim_warm"

        if not is_current_night_or_dim_warm:
            return {"time_of_day": time_of_day, "confidence": confidence}

        # Extract features for neon detection
        yellow_orange_ratio = features.get("yellow_orange_ratio", 0.0)
        bright_spots_overall = features.get("bright_spot_count", 0)
        color_atmosphere = features.get("color_atmosphere", "neutral")
        avg_saturation = features.get("avg_saturation", 0.0)

        # Get neon detection thresholds
        thresholds = self.config_manager.lighting_thresholds

        # Check neon lighting conditions
        if (yellow_orange_ratio > thresholds.neon_yellow_orange_thresh and
            bright_spots_overall > thresholds.neon_bright_spots_thresh and
            color_atmosphere == "warm" and
            avg_saturation > thresholds.neon_avg_saturation_thresh):

            old_time_of_day = time_of_day
            old_confidence = confidence

            # Check P365 context for neon scenes
            mapped_scene = p365_context["mapped_scene"]
            attributes = p365_context["attributes"]
            is_p365_neon_context = (
                any(kw in mapped_scene for kw in ["neon", "nightclub", "bar_neon"]) or
                "neon" in attributes
            )

            if is_indoor:
                if (is_p365_neon_context or
                    any(kw in mapped_scene for kw in self.P365_INDOOR_RESTAURANT_KEYWORDS)):
                    time_of_day = "indoor_neon_lit"
                    confidence = max(confidence, 0.80)
                else:
                    time_of_day = "indoor_dim_warm_neon_accent"
                    confidence = max(confidence, 0.77)
            else:
                if (is_p365_neon_context or
                    any(kw in mapped_scene for kw in ["street_night", "city_night", "downtown_night"])):
                    time_of_day = "neon_or_sodium_vapor_night"
                    confidence = max(confidence, 0.82)
                else:
                    time_of_day = "night_with_neon_lights"
                    confidence = max(confidence, 0.79)

            # Record the refinement
            diagnostics["special_lighting_detected"] = (
                f"Refined from {old_time_of_day} (Conf:{old_confidence:.2f}) "
                f"to {time_of_day} (Conf:{confidence:.2f}) due to neon/sodium vapor light characteristics. "
                f"P365 Context: {mapped_scene if is_p365_neon_context else 'N/A'}."
            )

        return {"time_of_day": time_of_day, "confidence": confidence}

    def _combine_attribute_and_visual_results(self, attribute_result: Dict[str, Any],
                                             visual_result: Dict[str, Any],
                                             diagnostics: Dict[str, Any]) -> Dict[str, Any]:
        """Combine Places365 attribute and visual analysis results."""
        # If visual analysis provided a different and potentially more nuanced result
        if (attribute_result["time_of_day"] != visual_result["time_of_day"] and
            visual_result["confidence"] > 0.65):

            diagnostics["final_decision_source"] = "Visual features (potentially P365-context-refined)."
            diagnostics["p365_attr_overridden_by_visual"] = (
                f"P365 Attr ToD {attribute_result['time_of_day']} "
                f"(Conf {attribute_result['confidence']:.2f}) was less certain or overridden by "
                f"visual logic result {visual_result['time_of_day']} (Conf {visual_result['confidence']:.2f})."
            )
            return visual_result

        # Use attribute result if it was more confident
        elif attribute_result["confidence"] >= visual_result["confidence"]:
            diagnostics["final_decision_source"] = "High-confidence P365 attribute."
            return attribute_result

        # Use visual result
        else:
            diagnostics["final_decision_source"] = "Visual features (potentially P365-context-refined)."
            return visual_result

    def _check_home_environment_pattern(self, features: Dict[str, Any]) -> bool:
        """Check if features indicate a home/residential environment pattern."""
        thresholds = self.config_manager.indoor_outdoor_thresholds
        return features.get("home_environment_pattern", 0.0) > thresholds.home_pattern_thresh_moderate * 0.7

    def _check_warm_sunset_conditions(self, features: Dict[str, Any]) -> bool:
        """Check if features indicate warm sunset/sunrise lighting conditions."""
        thresholds = self.config_manager.lighting_thresholds
        yellow_orange_ratio = features.get("yellow_orange_ratio", 0.0)
        color_atmosphere = features.get("color_atmosphere", "neutral")
        sky_brightness_ratio = features.get("sky_region_brightness_ratio", 1.0)

        return (yellow_orange_ratio > thresholds.outdoor_dusk_dawn_color_thresh and
                color_atmosphere == "warm" and
                sky_brightness_ratio < 1.5)

    def _get_default_lighting_result(self) -> Dict[str, Any]:
        """Return default lighting analysis result in case of errors."""
        return {
            "time_of_day": "unknown",
            "confidence": 0.5,
            "diagnostics": {
                "error": "Lighting analysis failed, using default values"
            }
        }
