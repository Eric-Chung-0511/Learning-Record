from typing import Dict, Any, List, Tuple, Optional, Union
import json
import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class FeatureThresholds:
    """Configuration class for feature extraction thresholds."""
    dark_pixel_threshold: float = 50.0
    bright_pixel_threshold: float = 220.0
    sky_blue_hue_min: float = 95.0
    sky_blue_hue_max: float = 135.0
    sky_blue_sat_min: float = 40.0
    sky_blue_val_min: float = 90.0
    gray_sat_max: float = 70.0
    gray_val_min: float = 60.0
    gray_val_max: float = 220.0
    light_source_abs_thresh: float = 220.0


@dataclass
class IndoorOutdoorThresholds:
    """Configuration class for indoor/outdoor classification thresholds."""
    sky_blue_dominance_thresh: float = 0.18
    sky_brightness_ratio_thresh: float = 1.25
    openness_top_thresh: float = 0.68
    sky_texture_complexity_thresh: float = 0.35
    ceiling_likelihood_thresh: float = 0.4
    boundary_clarity_thresh: float = 0.38
    brightness_uniformity_thresh_indoor: float = 0.6
    brightness_uniformity_thresh_outdoor: float = 0.40
    many_bright_spots_thresh: int = 6
    dim_scene_for_spots_thresh: float = 115.0
    home_pattern_thresh_strong: float = 2.0
    home_pattern_thresh_moderate: float = 1.0
    warm_indoor_max_brightness_thresh: float = 135.0
    aerial_top_dark_ratio_thresh: float = 0.9
    aerial_top_complex_thresh: float = 0.60
    aerial_min_avg_brightness_thresh: float = 65.0


@dataclass
class LightingThresholds:
    """Configuration class for lighting condition analysis thresholds."""
    outdoor_night_thresh_brightness: float = 80.0
    outdoor_night_lights_thresh: int = 2
    outdoor_dusk_dawn_thresh_brightness: float = 130.0
    outdoor_dusk_dawn_color_thresh: float = 0.10
    outdoor_day_bright_thresh: float = 140.0
    outdoor_day_blue_thresh: float = 0.05
    outdoor_day_cloudy_thresh: float = 120.0
    outdoor_day_gray_thresh: float = 0.18
    indoor_bright_thresh: float = 130.0
    indoor_moderate_thresh: float = 95.0
    commercial_min_brightness_thresh: float = 105.0
    commercial_min_spots_thresh: int = 3
    stadium_min_spots_thresh: int = 6
    neon_yellow_orange_thresh: float = 0.12
    neon_bright_spots_thresh: int = 4
    neon_avg_saturation_thresh: float = 60.0


@dataclass
class WeightingFactors:
    """Configuration class for feature weighting factors."""
    # Sky/Openness weights (negative values push towards outdoor)
    sky_blue_dominance_w: float = 3.5
    sky_brightness_ratio_w: float = 3.0
    openness_top_w: float = 2.8
    sky_texture_w: float = 2.0

    # Ceiling/Enclosure weights (positive values push towards indoor)
    ceiling_likelihood_w: float = 1.5
    boundary_clarity_w: float = 1.2

    # Brightness weights
    brightness_uniformity_w: float = 0.6
    brightness_non_uniformity_outdoor_w: float = 1.0
    brightness_non_uniformity_indoor_penalty_w: float = 0.1

    # Light source weights
    circular_lights_w: float = 1.2
    indoor_light_score_w: float = 0.8
    many_bright_spots_indoor_w: float = 0.3

    # Color atmosphere weights
    warm_atmosphere_indoor_w: float = 0.15

    # Environment pattern weights
    home_env_strong_w: float = 1.5
    home_env_moderate_w: float = 0.7

    # Structural pattern weights
    aerial_street_w: float = 2.5
    places365_outdoor_scene_w: float = 4.0
    places365_indoor_scene_w: float = 3.0
    places365_attribute_w: float = 1.5


@dataclass
class OverrideFactors:
    """Configuration class for override and reduction factors."""
    sky_override_factor_ceiling: float = 0.1
    sky_override_factor_boundary: float = 0.2
    sky_override_factor_uniformity: float = 0.15
    sky_override_factor_lights: float = 0.05
    sky_override_factor_p365_indoor_decision: float = 0.3
    aerial_enclosure_reduction_factor: float = 0.75
    ceiling_sky_override_factor: float = 0.1
    p365_outdoor_reduces_enclosure_factor: float = 0.3
    p365_indoor_boosts_ceiling_factor: float = 1.5


@dataclass
class ColorRanges:
    """Configuration class for color range definitions."""
    warm_hue_ranges: List[Tuple[float, float]] = field(
        default_factory=lambda: [(0, 50), (330, 360)]
    )
    cool_hue_ranges: List[Tuple[float, float]] = field(
        default_factory=lambda: [(90, 270)]
    )


@dataclass
class AlgorithmParameters:
    """Configuration class for algorithm-specific parameters."""
    indoor_score_sigmoid_scale: float = 0.3
    indoor_decision_threshold: float = 0.5
    places365_high_confidence_thresh: float = 0.75
    places365_moderate_confidence_thresh: float = 0.5
    places365_attribute_confidence_thresh: float = 0.6
    include_diagnostics: bool = True


class ConfigurationManager:
    """
    這主要是管理光線分析的參數，會有很多不同情況, 做parameters配置

    This class provides type-safe access to all configuration parameters,
    supports loading from external files, and includes validation mechanisms.
    """

    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize the configuration manager.

        Args:
            config_path: Optional path to external configuration file.
                        If None, uses default configuration.
        """
        self._feature_thresholds = FeatureThresholds()
        self._indoor_outdoor_thresholds = IndoorOutdoorThresholds()
        self._lighting_thresholds = LightingThresholds()
        self._weighting_factors = WeightingFactors()
        self._override_factors = OverrideFactors()
        self._color_ranges = ColorRanges()
        self._algorithm_parameters = AlgorithmParameters()

        if config_path is not None:
            self.load_from_file(config_path)

    @property
    def feature_thresholds(self) -> FeatureThresholds:
        """Get feature extraction thresholds."""
        return self._feature_thresholds

    @property
    def indoor_outdoor_thresholds(self) -> IndoorOutdoorThresholds:
        """Get indoor/outdoor classification thresholds."""
        return self._indoor_outdoor_thresholds

    @property
    def lighting_thresholds(self) -> LightingThresholds:
        """Get lighting condition analysis thresholds."""
        return self._lighting_thresholds

    @property
    def weighting_factors(self) -> WeightingFactors:
        """Get feature weighting factors."""
        return self._weighting_factors

    @property
    def override_factors(self) -> OverrideFactors:
        """Get override and reduction factors."""
        return self._override_factors

    @property
    def color_ranges(self) -> ColorRanges:
        """Get color range definitions."""
        return self._color_ranges

    @property
    def algorithm_parameters(self) -> AlgorithmParameters:
        """Get algorithm-specific parameters."""
        return self._algorithm_parameters

    def get_legacy_config_dict(self) -> Dict[str, Any]:
        """
        Generate legacy configuration dictionary for backward compatibility.

        Returns:
            Dictionary containing all configuration parameters in the original format.
        """
        config_dict = {}

        # Feature thresholds
        for field_name, field_value in self._feature_thresholds.__dict__.items():
            config_dict[field_name] = field_value

        # Indoor/outdoor thresholds
        for field_name, field_value in self._indoor_outdoor_thresholds.__dict__.items():
            config_dict[field_name] = field_value

        # Lighting thresholds
        for field_name, field_value in self._lighting_thresholds.__dict__.items():
            config_dict[field_name] = field_value

        # Override factors
        for field_name, field_value in self._override_factors.__dict__.items():
            config_dict[field_name] = field_value

        # Color ranges
        for field_name, field_value in self._color_ranges.__dict__.items():
            config_dict[field_name] = field_value

        # Algorithm parameters
        for field_name, field_value in self._algorithm_parameters.__dict__.items():
            config_dict[field_name] = field_value

        # Weighting factors - stored under 'indoor_outdoor_weights' key
        config_dict["indoor_outdoor_weights"] = self._weighting_factors.__dict__.copy()

        return config_dict

    def load_from_file(self, config_path: Union[str, Path]) -> None:
        """
        Load configuration from external JSON file.

        Args:
            config_path: Path to the configuration file.

        Raises:
            FileNotFoundError: If the configuration file doesn't exist.
            ValueError: If the configuration file contains invalid data.
        """
        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        try:
            with open(config_path, 'r', encoding='utf-8') as file:
                config_data = json.load(file)

            self._update_from_dict(config_data)

        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in configuration file: {e}")
        except Exception as e:
            raise ValueError(f"Error loading configuration: {e}")

    def save_to_file(self, config_path: Union[str, Path]) -> None:
        """
        Save current configuration to JSON file.

        Args:
            config_path: Path where to save the configuration file.
        """
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)

        config_dict = self.get_legacy_config_dict()

        with open(config_path, 'w', encoding='utf-8') as file:
            json.dump(config_dict, file, indent=2, ensure_ascii=False)

    def _update_from_dict(self, config_data: Dict[str, Any]) -> None:
        """
        Update configuration from dictionary data.

        Args:
            config_data: Dictionary containing configuration parameters.
        """
        # Update feature thresholds
        self._update_dataclass_from_dict(self._feature_thresholds, config_data)

        # Update indoor/outdoor thresholds
        self._update_dataclass_from_dict(self._indoor_outdoor_thresholds, config_data)

        # Update lighting thresholds
        self._update_dataclass_from_dict(self._lighting_thresholds, config_data)

        # Update override factors
        self._update_dataclass_from_dict(self._override_factors, config_data)

        # Update color ranges
        self._update_dataclass_from_dict(self._color_ranges, config_data)

        # Update algorithm parameters
        self._update_dataclass_from_dict(self._algorithm_parameters, config_data)

        # Update weighting factors from nested dictionary
        if "indoor_outdoor_weights" in config_data:
            self._update_dataclass_from_dict(
                self._weighting_factors,
                config_data["indoor_outdoor_weights"]
            )

    def _update_dataclass_from_dict(self, dataclass_instance: object, data_dict: Dict[str, Any]) -> None:
        """
        Update dataclass instance fields from dictionary.

        Args:
            dataclass_instance: The dataclass instance to update.
            data_dict: Dictionary containing the update values.
        """
        for field_name, field_value in data_dict.items():
            if hasattr(dataclass_instance, field_name):
                # Type validation could be added here
                setattr(dataclass_instance, field_name, field_value)

    def validate_configuration(self) -> List[str]:
        """
        Validate the current configuration for logical consistency.

        Returns:
            List of validation error messages. Empty list if configuration is valid.
        """
        errors = []

        # Validate threshold ranges
        ft = self._feature_thresholds
        if ft.dark_pixel_threshold >= ft.bright_pixel_threshold:
            errors.append("Dark pixel threshold must be less than bright pixel threshold")

        if ft.sky_blue_hue_min >= ft.sky_blue_hue_max:
            errors.append("Sky blue hue min must be less than sky blue hue max")

        if ft.gray_val_min >= ft.gray_val_max:
            errors.append("Gray value min must be less than gray value max")

        # Validate probability thresholds
        ap = self._algorithm_parameters
        if not (0.0 <= ap.indoor_decision_threshold <= 1.0):
            errors.append("Indoor decision threshold must be between 0 and 1")

        if not (0.0 <= ap.places365_high_confidence_thresh <= 1.0):
            errors.append("Places365 high confidence threshold must be between 0 and 1")

        # Validate color ranges
        for warm_range in self._color_ranges.warm_hue_ranges:
            if warm_range[0] >= warm_range[1]:
                errors.append(f"Invalid warm hue range: {warm_range}")

        for cool_range in self._color_ranges.cool_hue_ranges:
            if cool_range[0] >= cool_range[1]:
                errors.append(f"Invalid cool hue range: {cool_range}")

        return errors

    def get_threshold_value(self, threshold_name: str) -> Any:
        """
        Get a specific threshold value by name.

        Args:
            threshold_name: Name of the threshold parameter.

        Returns:
            The threshold value.

        Raises:
            AttributeError: If the threshold name doesn't exist.
        """
        # Search through all configuration sections
        for config_section in [
            self._feature_thresholds,
            self._indoor_outdoor_thresholds,
            self._lighting_thresholds,
            self._override_factors,
            self._algorithm_parameters
        ]:
            if hasattr(config_section, threshold_name):
                return getattr(config_section, threshold_name)

        # Check weighting factors
        if hasattr(self._weighting_factors, threshold_name):
            return getattr(self._weighting_factors, threshold_name)

        raise AttributeError(f"Threshold '{threshold_name}' not found")

    def update_threshold(self, threshold_name: str, value: Any) -> None:
        """
        Update a specific threshold value.

        Args:
            threshold_name: Name of the threshold parameter.
            value: New value for the threshold.

        Raises:
            AttributeError: If the threshold name doesn't exist.
        """
        # Search through all configuration sections
        for config_section in [
            self._feature_thresholds,
            self._indoor_outdoor_thresholds,
            self._lighting_thresholds,
            self._override_factors,
            self._algorithm_parameters,
            self._weighting_factors
        ]:
            if hasattr(config_section, threshold_name):
                setattr(config_section, threshold_name, value)
                return

        raise AttributeError(f"Threshold '{threshold_name}' not found")
