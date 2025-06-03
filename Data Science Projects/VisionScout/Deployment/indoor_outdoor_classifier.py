import numpy as np
import logging
import traceback
from typing import Dict, Any, Optional, List
from configuration_manager import ConfigurationManager


class IndoorOutdoorClassifier:
    """
    Classifies scenes as indoor or outdoor based on visual features and Places365 context.(判斷室內室外)
    此class會融入PLACES365，使判斷更準確

    This class implements sophisticated decision logic that combines multiple evidence sources
    including visual scene analysis, structural features, and external scene classification
    data to determine whether a scene is indoor or outdoor.
    """

    def __init__(self, config_manager: ConfigurationManager):
        """
        Initialize the indoor/outdoor classifier.

        Args:
            config_manager: Configuration manager instance for accessing thresholds and weights.
        """
        self.config_manager = config_manager
        self.logger = self._setup_logger()

        # Internal threshold constants for Places365 confidence levels
        self.P365_HIGH_CONF_THRESHOLD = 0.65
        self.P365_MODERATE_CONF_THRESHOLD = 0.4

        # 以下是絕對室內/室外的基本情況
        self.DEFINITELY_OUTDOOR_KEYWORDS_P365 = [
            "street", "road", "highway", "park", "beach", "mountain", "forest", "field",
            "outdoor", "sky", "coast", "courtyard", "square", "plaza", "bridge",
            "parking_lot", "playground", "stadium", "construction_site", "river", "ocean",
            "desert", "garden", "trail", "intersection", "crosswalk", "sidewalk", "pathway",
            "avenue", "boulevard", "downtown", "city_center", "market_outdoor"
        ]

        self.DEFINITELY_INDOOR_KEYWORDS_P365 = [
            "bedroom", "office", "kitchen", "library", "classroom", "conference_room", "living_room",
            "bathroom", "hospital", "hotel_room", "cabin", "interior", "museum", "gallery",
            "mall", "market_indoor", "basement", "corridor", "lobby", "restaurant_indoor",
            "bar_indoor", "shop_indoor", "gym_indoor"
        ]

    def _setup_logger(self) -> logging.Logger:
        """Set up logger for classification operations."""
        logger = logging.getLogger(f"{__name__}.IndoorOutdoorClassifier")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

    def classify(self, features: Dict[str, Any], places365_info: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Classify scene as indoor or outdoor based on features and Places365 context.

        Args:
            features: Dictionary containing extracted image features.
            places365_info: Optional Places365 classification information.

        Returns:
            Dictionary containing classification results including decision, probability,
            feature contributions, and diagnostic information.
        """
        try:
            self.logger.debug("Starting indoor/outdoor classification")

            # Initialize classification components
            visual_score = 0.0
            feature_contributions = {}
            diagnostics = {}

            # Extract Places365 information
            p365_context = self._extract_places365_context(places365_info, diagnostics)

            # Compute visual evidence score
            visual_analysis = self._analyze_visual_evidence(features, diagnostics)
            visual_score = visual_analysis["visual_score"]
            feature_contributions.update(visual_analysis["contributions"])

            # Incorporate Places365 influence
            p365_analysis = self._analyze_places365_influence(
                p365_context, visual_analysis.get("strong_sky_signal", False), diagnostics
            )
            p365_influence_score = p365_analysis["influence_score"]
            if abs(p365_influence_score) > 0.01:
                feature_contributions["places365_influence_score"] = round(p365_influence_score, 2)

            # Calculate final score and probability
            final_indoor_score = visual_score + p365_influence_score
            classification_result = self._compute_final_classification(
                final_indoor_score, visual_score, p365_influence_score, diagnostics
            )

            # Apply Places365 override if conditions are met
            override_result = self._apply_places365_override(
                classification_result, p365_context, diagnostics
            )

            # Ensure default values for missing contributions
            self._ensure_default_contributions(feature_contributions)

            # 最終結果
            result = {
                "is_indoor": override_result["is_indoor"],
                "indoor_probability": override_result["indoor_probability"],
                "indoor_score_raw": override_result["final_score"],
                "feature_contributions": feature_contributions,
                "diagnostics": diagnostics
            }

            self.logger.debug(f"Classification complete: indoor={result['is_indoor']}, "
                            f"probability={result['indoor_probability']:.3f}")

            return result

        except Exception as e:
            self.logger.error(f"Error in indoor/outdoor classification: {str(e)}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return self._get_default_classification_result()

    def _extract_places365_context(self, places365_info: Optional[Dict],
                                  diagnostics: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and validate Places365 context information."""
        context = {
            "mapped_scene": "unknown",
            "is_indoor_from_classification": None,
            "attributes": [],
            "confidence": 0.0,
            "is_indoor": None
        }

        if places365_info:
            context["mapped_scene"] = places365_info.get('mapped_scene_type', 'unknown').lower()
            context["attributes"] = [attr.lower() for attr in places365_info.get('attributes', [])]
            context["confidence"] = places365_info.get('confidence', 0.0)
            context["is_indoor_from_classification"] = places365_info.get('is_indoor_from_classification', None)
            context["is_indoor"] = places365_info.get('is_indoor', None)

            diagnostics["p365_context_received"] = (
                f"P365 Scene: {context['mapped_scene']}, P365 SceneConf: {context['confidence']:.2f}, "
                f"P365 DirectIndoor: {context['is_indoor_from_classification']}, "
                f"P365 Attrs: {context['attributes']}"
            )

        return context

    def _analyze_visual_evidence(self, features: Dict[str, Any],
                                diagnostics: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze visual evidence for indoor/outdoor classification."""
        visual_score = 0.0
        contributions = {}
        strong_sky_signal = False

        # Sky and openness analysis
        sky_analysis = self._analyze_sky_evidence(features, diagnostics)
        visual_score += sky_analysis["score"]
        if sky_analysis["score"] != 0:
            contributions["sky_openness_features_visual"] = round(sky_analysis["score"], 2)
        strong_sky_signal = sky_analysis["strong_signal"]

        # Enclosure and structural analysis
        enclosure_analysis = self._analyze_enclosure_evidence(features, strong_sky_signal, diagnostics)
        visual_score += enclosure_analysis["score"]
        if enclosure_analysis["score"] != 0:
            contributions["enclosure_features"] = round(enclosure_analysis["score"], 2)

        # Brightness uniformity analysis
        uniformity_analysis = self._analyze_brightness_uniformity(features, strong_sky_signal, diagnostics)
        visual_score += uniformity_analysis["score"]
        if uniformity_analysis["score"] != 0:
            contributions["brightness_uniformity_contribution"] = round(uniformity_analysis["score"], 2)

        # Light source analysis
        light_analysis = self._analyze_light_sources(features, strong_sky_signal, diagnostics)
        visual_score += light_analysis["score"]
        if light_analysis["score"] != 0:
            contributions["light_source_features"] = round(light_analysis["score"], 2)

        # Color atmosphere analysis
        atmosphere_analysis = self._analyze_color_atmosphere(features, strong_sky_signal, diagnostics)
        visual_score += atmosphere_analysis["score"]
        if atmosphere_analysis["score"] != 0:
            contributions["warm_atmosphere_indoor_visual_contrib"] = round(atmosphere_analysis["score"], 2)

        # Home environment pattern analysis
        home_analysis = self._analyze_home_environment_pattern(features, strong_sky_signal, diagnostics)
        visual_score += home_analysis["score"]
        if home_analysis["score"] != 0:
            contributions["home_environment_pattern_visual"] = round(home_analysis["score"], 2)

        # Aerial street pattern analysis
        aerial_analysis = self._analyze_aerial_street_pattern(features, strong_sky_signal, contributions, diagnostics)
        visual_score += aerial_analysis["score"]
        if aerial_analysis["score"] != 0:
            contributions["aerial_street_pattern_visual"] = round(aerial_analysis["score"], 2)

        diagnostics["visual_indoor_score_subtotal"] = round(visual_score, 3)

        return {
            "visual_score": visual_score,
            "contributions": contributions,
            "strong_sky_signal": strong_sky_signal
        }

    def _analyze_sky_evidence(self, features: Dict[str, Any],
                             diagnostics: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze sky-related evidence for outdoor classification."""
        sky_evidence_score = 0.0
        strong_sky_signal = False

        # Extract relevant features
        sky_blue_dominance = features.get("sky_region_blue_dominance", 0.0)
        sky_brightness_ratio = features.get("sky_region_brightness_ratio", 1.0)
        texture_complexity = features.get("top_region_texture_complexity", 0.5)
        openness_top_edge = features.get("openness_top_edge", 0.5)

        # Get thresholds
        thresholds = self.config_manager.indoor_outdoor_thresholds
        weights = self.config_manager.weighting_factors

        # Strong blue sky signal
        if sky_blue_dominance > thresholds.sky_blue_dominance_thresh:
            sky_evidence_score -= weights.sky_blue_dominance_w * sky_blue_dominance
            diagnostics["sky_detection_reason_visual"] = f"Visual: Strong sky-like blue ({sky_blue_dominance:.2f})"
            strong_sky_signal = True

        # Bright top region with low texture
        elif (sky_brightness_ratio > getattr(thresholds, 'sky_brightness_ratio_strong_thresh', 1.35) and
              texture_complexity < getattr(thresholds, 'sky_texture_complexity_clear_thresh', 0.25)):
            outdoor_push = weights.sky_brightness_ratio_w * (sky_brightness_ratio - 1.0)
            sky_evidence_score -= outdoor_push
            sky_evidence_score -= weights.sky_texture_w
            diagnostics["sky_detection_reason_visual"] = (
                f"Visual: Top brighter (ratio:{sky_brightness_ratio:.2f}) & low texture."
            )
            strong_sky_signal = True

        # High top edge openness
        elif openness_top_edge > getattr(thresholds, 'openness_top_strong_thresh', 0.80):
            sky_evidence_score -= weights.openness_top_w * openness_top_edge
            diagnostics["sky_detection_reason_visual"] = (
                f"Visual: Very high top edge openness ({openness_top_edge:.2f})."
            )
            strong_sky_signal = True

        # Weak sky signal (cloudy conditions)
        elif (not strong_sky_signal and
              texture_complexity < getattr(thresholds, 'sky_texture_complexity_cloudy_thresh', 0.20) and
              sky_brightness_ratio > getattr(thresholds, 'sky_brightness_ratio_cloudy_thresh', 0.95)):
            sky_evidence_score -= weights.sky_texture_w * (1.0 - texture_complexity) * 0.5
            diagnostics["sky_detection_reason_visual"] = (
                f"Visual: Weak sky signal (low texture, brightish top: {texture_complexity:.2f}), less weight."
            )

        if strong_sky_signal:
            diagnostics["strong_sky_signal_visual_detected"] = True

        return {
            "score": sky_evidence_score,
            "strong_signal": strong_sky_signal
        }

    def _analyze_enclosure_evidence(self, features: Dict[str, Any], strong_sky_signal: bool,
                                   diagnostics: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze enclosure evidence for indoor classification."""
        enclosure_score = 0.0

        # Extract features
        ceiling_likelihood = features.get("ceiling_likelihood", 0.0)
        boundary_clarity = features.get("boundary_clarity", 0.0)
        texture_complexity = features.get("top_region_texture_complexity", 0.5)
        openness_top_edge = features.get("openness_top_edge", 0.5)

        # Get configuration
        thresholds = self.config_manager.indoor_outdoor_thresholds
        weights = self.config_manager.weighting_factors
        override_factors = self.config_manager.override_factors

        # Ceiling likelihood analysis
        if ceiling_likelihood > thresholds.ceiling_likelihood_thresh:
            current_ceiling_score = weights.ceiling_likelihood_w * ceiling_likelihood
            if strong_sky_signal:
                current_ceiling_score *= override_factors.sky_override_factor_ceiling
            enclosure_score += current_ceiling_score
            diagnostics["indoor_reason_ceiling_visual"] = (
                f"Visual Ceiling: {ceiling_likelihood:.2f}, ScoreCont: {current_ceiling_score:.2f}"
            )

        # Boundary clarity analysis
        if boundary_clarity > thresholds.boundary_clarity_thresh:
            current_boundary_score = weights.boundary_clarity_w * boundary_clarity
            if strong_sky_signal:
                current_boundary_score *= override_factors.sky_override_factor_boundary
            enclosure_score += current_boundary_score
            diagnostics["indoor_reason_boundary_visual"] = (
                f"Visual Boundary: {boundary_clarity:.2f}, ScoreCont: {current_boundary_score:.2f}"
            )

        # Complex urban top detection
        if (not strong_sky_signal and texture_complexity > 0.7 and
            openness_top_edge < 0.3 and ceiling_likelihood < 0.35):
            diagnostics["complex_urban_top_visual"] = True
            if boundary_clarity > 0.5:
                enclosure_score *= 0.5
                diagnostics["reduced_enclosure_for_urban_top_visual"] = True

        return {"score": enclosure_score}

    def _analyze_brightness_uniformity(self, features: Dict[str, Any], strong_sky_signal: bool,
                                      diagnostics: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze brightness uniformity patterns."""
        uniformity_score = 0.0

        # Calculate brightness uniformity
        brightness_std = features.get("brightness_std", 50.0)
        avg_brightness = features.get("avg_brightness", 100.0)
        brightness_uniformity = 1.0 - min(1.0, brightness_std / max(avg_brightness, 1e-5))
        shadow_clarity = features.get("shadow_clarity_score", 0.5)

        # Get configuration
        thresholds = self.config_manager.indoor_outdoor_thresholds
        weights = self.config_manager.weighting_factors
        override_factors = self.config_manager.override_factors

        # High uniformity (indoor indicator)
        if brightness_uniformity > thresholds.brightness_uniformity_thresh_indoor:
            uniformity_score = weights.brightness_uniformity_w * brightness_uniformity
            if strong_sky_signal:
                uniformity_score *= override_factors.sky_override_factor_uniformity

        # Low uniformity (potential outdoor indicator)
        elif brightness_uniformity < thresholds.brightness_uniformity_thresh_outdoor:
            if shadow_clarity > 0.65:
                uniformity_score = -weights.brightness_non_uniformity_outdoor_w * (1.0 - brightness_uniformity)
            elif not strong_sky_signal:
                uniformity_score = weights.brightness_non_uniformity_indoor_penalty_w * (1.0 - brightness_uniformity)

        return {"score": uniformity_score}

    def _analyze_light_sources(self, features: Dict[str, Any], strong_sky_signal: bool,
                              diagnostics: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze artificial light source patterns."""
        light_score = 0.0

        # Extract light features
        indoor_light_score = features.get("indoor_light_score", 0.0)
        circular_light_count = features.get("circular_light_count", 0)
        bright_spot_count = features.get("bright_spot_count", 0)
        avg_brightness = features.get("avg_brightness", 100.0)
        gradient_ratio = features.get("gradient_ratio_vertical_horizontal", 1.0)
        edges_density = features.get("edges_density", 0.0)

        # Get configuration
        thresholds = self.config_manager.indoor_outdoor_thresholds
        weights = self.config_manager.weighting_factors
        override_factors = self.config_manager.override_factors

        # Circular lights detection
        if circular_light_count >= 1 and not strong_sky_signal:
            light_score += weights.circular_lights_w * circular_light_count

        # Indoor light score
        elif indoor_light_score > 0.55 and not strong_sky_signal:
            light_score += weights.indoor_light_score_w * indoor_light_score

        # Many bright spots in dim scenes
        elif (bright_spot_count > thresholds.many_bright_spots_thresh and
              avg_brightness < thresholds.dim_scene_for_spots_thresh and
              not strong_sky_signal):
            light_score += weights.many_bright_spots_indoor_w * min(bright_spot_count / 10.0, 1.5)

        # Street structure detection
        is_likely_street_structure = (0.7 < gradient_ratio < 1.5) and edges_density > 0.15

        if is_likely_street_structure and bright_spot_count > 3 and not strong_sky_signal:
            light_score *= 0.2
            diagnostics["street_lights_heuristic_visual"] = True
        elif strong_sky_signal:
            light_score *= override_factors.sky_override_factor_lights

        return {"score": light_score}

    def _analyze_color_atmosphere(self, features: Dict[str, Any], strong_sky_signal: bool,
                                 diagnostics: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze color atmosphere patterns."""
        atmosphere_score = 0.0

        # Extract features
        color_atmosphere = features.get("color_atmosphere", "neutral")
        avg_brightness = features.get("avg_brightness", 100.0)
        avg_saturation = features.get("avg_saturation", 100.0)
        gradient_ratio = features.get("gradient_ratio_vertical_horizontal", 1.0)
        edges_density = features.get("edges_density", 0.0)
        indoor_light_score = features.get("indoor_light_score", 0.0)

        # Get configuration
        thresholds = self.config_manager.indoor_outdoor_thresholds
        weights = self.config_manager.weighting_factors

        # Warm atmosphere analysis
        if (color_atmosphere == "warm" and
            avg_brightness < thresholds.warm_indoor_max_brightness_thresh):

            # Check exclusion conditions
            is_likely_street_structure = (0.7 < gradient_ratio < 1.5) and edges_density > 0.15
            is_complex_urban_top = diagnostics.get("complex_urban_top_visual", False)

            if (not strong_sky_signal and not is_complex_urban_top and
                not (is_likely_street_structure and avg_brightness > 80) and
                avg_saturation < 160):

                if indoor_light_score > 0.05:
                    atmosphere_score = weights.warm_atmosphere_indoor_w

        return {"score": atmosphere_score}

    def _analyze_home_environment_pattern(self, features: Dict[str, Any], strong_sky_signal: bool,
                                         diagnostics: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze home/residential environment patterns."""
        home_score = 0.0

        if strong_sky_signal:
            diagnostics["skipped_home_env_visual_due_to_sky"] = True
            return {"score": 0.0}

        # Calculate bedroom/home indicators
        bedroom_indicators = 0.0
        brightness_uniformity = features.get("brightness_uniformity", 0.0)
        boundary_clarity = features.get("boundary_clarity", 0.0)
        ceiling_likelihood = features.get("ceiling_likelihood", 0.0)
        bright_spot_count = features.get("bright_spot_count", 0)
        circular_light_count = features.get("circular_light_count", 0)
        warm_ratio = features.get("warm_ratio", 0.0)
        avg_saturation = features.get("avg_saturation", 100.0)

        # Accumulate indicators
        if brightness_uniformity > 0.65 and boundary_clarity > 0.40:
            bedroom_indicators += 1.1

        if ceiling_likelihood > 0.35 and (bright_spot_count > 0 or circular_light_count > 0):
            bedroom_indicators += 1.1

        if warm_ratio > 0.55 and brightness_uniformity > 0.65:
            bedroom_indicators += 1.0

        if brightness_uniformity > 0.70 and avg_saturation < 60:
            bedroom_indicators += 0.7

        # Get configuration
        thresholds = self.config_manager.indoor_outdoor_thresholds
        weights = self.config_manager.weighting_factors

        # Apply scoring based on indicator strength
        if bedroom_indicators >= thresholds.home_pattern_thresh_strong:
            home_score = weights.home_env_strong_w
        elif bedroom_indicators >= thresholds.home_pattern_thresh_moderate:
            home_score = weights.home_env_moderate_w

        if bedroom_indicators > 0:
            diagnostics["home_environment_pattern_visual_indicators"] = round(bedroom_indicators, 1)

        return {"score": home_score}

    def _analyze_aerial_street_pattern(self, features: Dict[str, Any], strong_sky_signal: bool,
                                      contributions: Dict[str, float],
                                      diagnostics: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze aerial view street patterns."""
        aerial_score = 0.0

        # Extract features
        sky_brightness_ratio = features.get("sky_region_brightness_ratio", 1.0)
        texture_complexity = features.get("top_region_texture_complexity", 0.5)
        avg_brightness = features.get("avg_brightness", 100.0)

        # Get configuration
        thresholds = self.config_manager.indoor_outdoor_thresholds
        weights = self.config_manager.weighting_factors

        # Aerial street pattern detection
        if (sky_brightness_ratio < thresholds.aerial_top_dark_ratio_thresh and
            texture_complexity > thresholds.aerial_top_complex_thresh and
            avg_brightness > thresholds.aerial_min_avg_brightness_thresh and
            not strong_sky_signal):

            aerial_score = -weights.aerial_street_w
            diagnostics["aerial_street_pattern_visual_detected"] = True

            # Reduce enclosure features if aerial pattern detected
            if ("enclosure_features" in contributions and
                contributions["enclosure_features"] > 0):

                reduction_factor = self.config_manager.override_factors.aerial_enclosure_reduction_factor
                positive_enclosure_score = max(0, contributions["enclosure_features"])
                reduction_amount = positive_enclosure_score * reduction_factor

                contributions["enclosure_features_reduced_by_aerial"] = round(-reduction_amount, 2)
                contributions["enclosure_features"] = round(
                    contributions["enclosure_features"] - reduction_amount, 2
                )

        return {"score": aerial_score}

    def _analyze_places365_influence(self, p365_context: Dict[str, Any],
                                    strong_sky_signal: bool,
                                    diagnostics: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze Places365 influence on classification."""
        p365_influence_score = 0.0

        if not p365_context or p365_context["confidence"] < self.P365_MODERATE_CONF_THRESHOLD:
            return {"influence_score": 0.0}

        # Places365 direct classification influence
        if p365_context["is_indoor_from_classification"] is not None:
            p365_influence_score += self._compute_direct_classification_influence(
                p365_context, strong_sky_signal, diagnostics
            )

        # Places365 scene context influence
        elif p365_context["confidence"] >= self.P365_MODERATE_CONF_THRESHOLD:
            p365_influence_score += self._compute_scene_context_influence(
                p365_context, strong_sky_signal, diagnostics
            )

        # Places365 attributes influence
        if p365_context["attributes"] and p365_context["confidence"] > 0.5:
            p365_influence_score += self._compute_attributes_influence(
                p365_context, strong_sky_signal, diagnostics
            )

        # High confidence street scene boost
        if (p365_context["confidence"] >= 0.85 and
            any(kw in p365_context["mapped_scene"] for kw in ["intersection", "crosswalk", "street", "road"])):

            additional_outdoor_push = -3.0 * p365_context["confidence"]
            p365_influence_score += additional_outdoor_push
            diagnostics["p365_street_scene_boost"] = (
                f"Additional outdoor push: {additional_outdoor_push:.2f} for street scene: "
                f"{p365_context['mapped_scene']}"
            )
            self.logger.debug(f"High confidence street scene detected - "
                            f"{p365_context['mapped_scene']} with confidence {p365_context['confidence']:.3f}")

        return {"influence_score": p365_influence_score}

    def _compute_direct_classification_influence(self, p365_context: Dict[str, Any],
                                               strong_sky_signal: bool,
                                               diagnostics: Dict[str, Any]) -> float:
        """Compute influence from Places365 direct indoor/outdoor classification."""
        P365_DIRECT_INDOOR_WEIGHT = 3.5
        P365_DIRECT_OUTDOOR_WEIGHT = 4.0

        confidence = p365_context["confidence"]
        is_indoor = p365_context["is_indoor_from_classification"]
        mapped_scene = p365_context["mapped_scene"]

        if is_indoor is True:
            current_contrib = P365_DIRECT_INDOOR_WEIGHT * confidence
            diagnostics["p365_influence_source"] = (
                f"P365_DirectIndoor(True,Conf:{confidence:.2f},Scene:{mapped_scene})"
            )
        else:
            current_contrib = -P365_DIRECT_OUTDOOR_WEIGHT * confidence
            diagnostics["p365_influence_source"] = (
                f"P365_DirectIndoor(False,Conf:{confidence:.2f},Scene:{mapped_scene})"
            )

        # Apply sky override for indoor predictions
        if strong_sky_signal and current_contrib > 0:
            sky_override_factor = self.config_manager.override_factors.sky_override_factor_p365_indoor_decision
            current_contrib *= sky_override_factor
            diagnostics["p365_indoor_push_reduced_by_visual_sky"] = f"Reduced to {current_contrib:.2f}"

        return current_contrib

    def _compute_scene_context_influence(self, p365_context: Dict[str, Any],
                                        strong_sky_signal: bool,
                                        diagnostics: Dict[str, Any]) -> float:
        """Compute influence from Places365 scene context."""
        P365_SCENE_CONTEXT_INDOOR_WEIGHT = 2.0
        P365_SCENE_CONTEXT_OUTDOOR_WEIGHT = 2.5

        confidence = p365_context["confidence"]
        mapped_scene = p365_context["mapped_scene"]

        is_def_indoor = any(kw in mapped_scene for kw in self.DEFINITELY_INDOOR_KEYWORDS_P365)
        is_def_outdoor = any(kw in mapped_scene for kw in self.DEFINITELY_OUTDOOR_KEYWORDS_P365)

        current_contrib = 0.0

        if is_def_indoor and not is_def_outdoor:
            current_contrib = P365_SCENE_CONTEXT_INDOOR_WEIGHT * confidence
            diagnostics["p365_influence_source"] = (
                f"P365_SceneContext(Indoor: {mapped_scene}, Conf:{confidence:.2f})"
            )
        elif is_def_outdoor and not is_def_indoor:
            current_contrib = -P365_SCENE_CONTEXT_OUTDOOR_WEIGHT * confidence
            diagnostics["p365_influence_source"] = (
                f"P365_SceneContext(Outdoor: {mapped_scene}, Conf:{confidence:.2f})"
            )

        # Apply sky override for indoor predictions
        if strong_sky_signal and current_contrib > 0:
            sky_override_factor = self.config_manager.override_factors.sky_override_factor_p365_indoor_decision
            current_contrib *= sky_override_factor
            diagnostics["p365_context_indoor_push_reduced_by_visual_sky"] = f"Reduced to {current_contrib:.2f}"

        return current_contrib

    def _compute_attributes_influence(self, p365_context: Dict[str, Any],
                                     strong_sky_signal: bool,
                                     diagnostics: Dict[str, Any]) -> float:
        """Compute influence from Places365 attributes."""
        P365_ATTRIBUTE_INDOOR_WEIGHT = 1.0
        P365_ATTRIBUTE_OUTDOOR_WEIGHT = 1.5

        confidence = p365_context["confidence"]
        attributes = p365_context["attributes"]

        attr_contrib = 0.0

        if "indoor" in attributes and "outdoor" not in attributes:
            attr_contrib += P365_ATTRIBUTE_INDOOR_WEIGHT * (confidence * 0.5)
            diagnostics["p365_attr_influence"] = f"+{attr_contrib:.2f} (indoor attr)"
        elif "outdoor" in attributes and "indoor" not in attributes:
            attr_contrib -= P365_ATTRIBUTE_OUTDOOR_WEIGHT * (confidence * 0.5)
            diagnostics["p365_attr_influence"] = f"{attr_contrib:.2f} (outdoor attr)"

        # Apply sky override for indoor attributes
        if strong_sky_signal and attr_contrib > 0:
            sky_override_factor = self.config_manager.override_factors.sky_override_factor_p365_indoor_decision
            attr_contrib *= sky_override_factor

        return attr_contrib

    def _compute_final_classification(self, final_indoor_score: float, visual_score: float,
                                     p365_influence_score: float, diagnostics: Dict[str, Any]) -> Dict[str, Any]:
        """Compute final classification probability and decision."""
        # Record score breakdown
        diagnostics["final_indoor_score_value"] = round(final_indoor_score, 3)
        diagnostics["final_score_breakdown"] = (
            f"VisualScore: {visual_score:.2f}, P365Influence: {p365_influence_score:.2f}"
        )

        # Apply sigmoid transformation
        sigmoid_scale = self.config_manager.algorithm_parameters.indoor_score_sigmoid_scale
        indoor_probability = 1 / (1 + np.exp(-final_indoor_score * sigmoid_scale))

        # Make decision
        decision_threshold = self.config_manager.algorithm_parameters.indoor_decision_threshold
        is_indoor = indoor_probability > decision_threshold

        return {
            "is_indoor": is_indoor,
            "indoor_probability": indoor_probability,
            "final_score": final_indoor_score
        }

    def _apply_places365_override(self, classification_result: Dict[str, Any],
                                 p365_context: Dict[str, Any],
                                 diagnostics: Dict[str, Any]) -> Dict[str, Any]:
        """Apply Places365 high-confidence override if conditions are met."""
        is_indoor = classification_result["is_indoor"]
        indoor_probability = classification_result["indoor_probability"]
        final_score = classification_result["final_score"]

        # Check for override conditions
        if not p365_context or p365_context["confidence"] < 0.5:
            diagnostics["final_indoor_probability_calculated"] = round(indoor_probability, 3)
            diagnostics["final_is_indoor_decision"] = bool(is_indoor)
            return classification_result

        p365_is_indoor_decision = p365_context.get("is_indoor", None)
        confidence = p365_context["confidence"]

        self.logger.debug(f"Override check: is_indoor={is_indoor}, p365_conf={confidence}, "
                         f"p365_raw_is_indoor={p365_is_indoor_decision}")

        # Apply override for high confidence Places365 decisions
        if p365_is_indoor_decision is not None:
            if p365_is_indoor_decision == False:
                self.logger.debug(f"Applying outdoor override. Original: {is_indoor}")
                original_decision = f"Indoor:{is_indoor}, Prob:{indoor_probability:.3f}, Score:{final_score:.2f}"

                is_indoor = False
                indoor_probability = 0.02
                final_score = -8.0

                diagnostics["p365_force_override_applied"] = (
                    f"P365 FORCED OUTDOOR (is_indoor: {p365_is_indoor_decision}, Conf: {confidence:.3f})"
                )
                diagnostics["p365_override_original_decision"] = original_decision
                self.logger.info(f"Places365 FORCED OUTDOOR override applied. New is_indoor: {is_indoor}")

            elif p365_is_indoor_decision == True:
                self.logger.debug(f"Applying indoor override. Original: {is_indoor}")
                original_decision = f"Indoor:{is_indoor}, Prob:{indoor_probability:.3f}, Score:{final_score:.2f}"

                is_indoor = True
                indoor_probability = 0.98
                final_score = 8.0

                diagnostics["p365_force_override_applied"] = (
                    f"P365 FORCED INDOOR (is_indoor: {p365_is_indoor_decision}, Conf: {confidence:.3f})"
                )
                diagnostics["p365_override_original_decision"] = original_decision
                self.logger.info(f"Places365 FORCED INDOOR override applied. New is_indoor: {is_indoor}")

        # Record final values
        diagnostics["final_indoor_probability_calculated"] = round(indoor_probability, 3)
        diagnostics["final_is_indoor_decision"] = bool(is_indoor)

        self.logger.debug(f"Final classification: is_indoor={is_indoor}, score={final_score}, prob={indoor_probability}")

        return {
            "is_indoor": is_indoor,
            "indoor_probability": indoor_probability,
            "final_score": final_score
        }

    def _ensure_default_contributions(self, feature_contributions: Dict[str, float]) -> None:
        """Ensure all expected feature contribution keys have default values."""
        default_keys = [
            "sky_openness_features", "enclosure_features",
            "brightness_uniformity_contribution", "light_source_features"
        ]

        for key in default_keys:
            if key not in feature_contributions:
                feature_contributions[key] = 0.0

    def _get_default_classification_result(self) -> Dict[str, Any]:
        """Return default classification result in case of errors."""
        return {
            "is_indoor": False,
            "indoor_probability": 0.5,
            "indoor_score_raw": 0.0,
            "feature_contributions": {
                "sky_openness_features": 0.0,
                "enclosure_features": 0.0,
                "brightness_uniformity_contribution": 0.0,
                "light_source_features": 0.0
            },
            "diagnostics": {
                "error": "Classification failed, using default values"
            }
        }
