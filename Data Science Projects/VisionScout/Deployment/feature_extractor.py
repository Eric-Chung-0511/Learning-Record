import numpy as np
import cv2
import logging
import traceback
from typing import Dict, Any, Optional
from configuration_manager import ConfigurationManager


class FeatureExtractor:
    """
    Extracts comprehensive lighting and scene features from images.（主要從圖片提取光線資訊)

    This class handles all basic feature computation including brightness analysis,
    color characteristics, texture complexity, and structural features for
    lighting analysis and scene understanding.
    """

    def __init__(self, config_manager: ConfigurationManager):
        """
        Initialize the feature extractor.

        Args:
            config_manager: Configuration manager instance for accessing thresholds.
        """
        self.config_manager = config_manager
        self.logger = self._setup_logger()

    def _setup_logger(self) -> logging.Logger:
        """Set up logger for feature extraction operations."""
        logger = logging.getLogger(f"{__name__}.FeatureExtractor")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

    def extract_features(self, image_rgb: np.ndarray) -> Dict[str, Any]:
        """
        Extract all features from an RGB image.

        Args:
            image_rgb: Input image as numpy array in RGB format.

        Returns:
            Dictionary containing all extracted features.
        """
        try:
            # Validate input image
            if not self._validate_image(image_rgb):
                return self._get_default_features()

            # Get image dimensions and prepare processing parameters
            height, width = image_rgb.shape[:2]
            scale_factor = self._calculate_scale_factor(height, width)

            # Create processed image versions
            small_rgb = cv2.resize(
                image_rgb,
                (width // scale_factor, height // scale_factor),
                interpolation=cv2.INTER_AREA
            )
            hsv_img = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
            gray_img = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
            small_gray = cv2.cvtColor(small_rgb, cv2.COLOR_RGB2GRAY)

            # Extract features by category
            brightness_features = self.compute_brightness_features(hsv_img, height, width)
            color_features = self.compute_color_features(hsv_img, height, width)
            texture_features = self.compute_texture_features(small_gray, gray_img, height, width)
            structure_features = self.compute_structure_features(
                small_gray, gray_img, hsv_img, height, width, scale_factor
            )

            # Combine all features
            features = {**brightness_features, **color_features, **texture_features, **structure_features}

            # Add compatibility features for legacy code
            legacy_features = self._compute_legacy_compatibility_features(
                hsv_img, small_gray, features, scale_factor
            )
            features.update(legacy_features)

            self.logger.debug(f"Successfully extracted {len(features)} features from image")
            return features

        except Exception as e:
            self.logger.error(f"Error in feature extraction: {str(e)}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return self._get_default_features()

    def compute_brightness_features(self, hsv_img: np.ndarray, height: int, width: int) -> Dict[str, float]:
        """
        Compute brightness-related features from HSV image.

        Args:
            hsv_img: Image in HSV color space.
            height: Image height.
            width: Image width.

        Returns:
            Dictionary containing brightness features.
        """
        try:
            v_channel = hsv_img[:, :, 2]  # Value channel represents brightness

            # 基本的亮度統計
            avg_brightness = float(np.mean(v_channel))
            brightness_std = float(np.std(v_channel))

            # Pixel ratio calculations
            dark_threshold = self.config_manager.feature_thresholds.dark_pixel_threshold
            bright_threshold = self.config_manager.feature_thresholds.bright_pixel_threshold

            total_pixels = height * width
            dark_pixel_ratio = float(np.sum(v_channel < dark_threshold) / total_pixels)
            bright_pixel_ratio = float(np.sum(v_channel > bright_threshold) / total_pixels)

            # Brightness uniformity
            brightness_uniformity = 1.0 - min(1.0, brightness_std / max(avg_brightness, 1e-5))

            return {
                "avg_brightness": avg_brightness,
                "brightness_std": brightness_std,
                "dark_pixel_ratio": dark_pixel_ratio,
                "bright_pixel_ratio": bright_pixel_ratio,
                "brightness_uniformity": brightness_uniformity
            }

        except Exception as e:
            self.logger.error(f"Error computing brightness features: {str(e)}")
            return {
                "avg_brightness": 100.0,
                "brightness_std": 50.0,
                "dark_pixel_ratio": 0.0,
                "bright_pixel_ratio": 0.0,
                "brightness_uniformity": 0.5
            }

    def compute_color_features(self, hsv_img: np.ndarray, height: int, width: int) -> Dict[str, Any]:
        """
        Compute color-related features from HSV image.

        Args:
            hsv_img: Image in HSV color space.
            height: Image height.
            width: Image width.

        Returns:
            Dictionary containing color features.
        """
        try:
            h_channel, s_channel, v_channel = cv2.split(hsv_img)
            total_pixels = height * width

            # Color ratio calculations
            color_features = {}

            # Blue color detection (general and sky-specific)
            blue_mask = ((h_channel >= 90) & (h_channel <= 140))
            color_features["blue_ratio"] = float(np.sum(blue_mask) / total_pixels)

            # Sky-like blue detection
            ft = self.config_manager.feature_thresholds
            sky_blue_mask = (
                (h_channel >= ft.sky_blue_hue_min) & (h_channel <= ft.sky_blue_hue_max) &
                (s_channel > ft.sky_blue_sat_min) & (v_channel > ft.sky_blue_val_min)
            )
            color_features["sky_like_blue_ratio"] = float(np.sum(sky_blue_mask) / total_pixels)

            # Yellow-orange detection
            yellow_orange_mask = ((h_channel >= 15) & (h_channel <= 45))
            color_features["yellow_orange_ratio"] = float(np.sum(yellow_orange_mask) / total_pixels)

            # Gray detection
            gray_mask = (
                (s_channel < ft.gray_sat_max) &
                (v_channel > ft.gray_val_min) &
                (v_channel < ft.gray_val_max)
            )
            color_features["gray_ratio"] = float(np.sum(gray_mask) / total_pixels)

            # Saturation statistics
            color_features["avg_saturation"] = float(np.mean(s_channel))

            # Sky region analysis
            sky_region_features = self._analyze_sky_region(h_channel, s_channel, v_channel, height)
            color_features.update(sky_region_features)

            # Color atmosphere analysis
            atmosphere_features = self._analyze_color_atmosphere(h_channel, s_channel, total_pixels)
            color_features.update(atmosphere_features)

            return color_features

        except Exception as e:
            self.logger.error(f"Error computing color features: {str(e)}")
            return self._get_default_color_features()

    def compute_texture_features(self, small_gray: np.ndarray, gray_img: np.ndarray,
                                height: int, width: int) -> Dict[str, float]:
        """
        Compute texture and gradient features.

        Args:
            small_gray: Downscaled grayscale image for efficient processing.
            gray_img: Full-resolution grayscale image.
            height: Original image height.
            width: Original image width.

        Returns:
            Dictionary containing texture features.
        """
        try:
            # Compute gradients on small image for efficiency
            gx = cv2.Sobel(small_gray, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(small_gray, cv2.CV_32F, 0, 1, ksize=3)

            avg_abs_gx = float(np.mean(np.abs(gx)))
            avg_abs_gy = float(np.mean(np.abs(gy)))

            # Gradient ratio (vertical to horizontal)
            gradient_ratio_vertical_horizontal = avg_abs_gy / max(avg_abs_gx, 1e-5)

            # Top region texture complexity
            small_top_third_height = small_gray.shape[0] // 3
            small_sky_region_gray = small_gray[:small_top_third_height, :]

            if small_sky_region_gray.size > 0:
                laplacian_var_sky = cv2.Laplacian(small_sky_region_gray, cv2.CV_64F).var()
                top_region_texture_complexity = min(1.0, laplacian_var_sky / 1000.0)
            else:
                top_region_texture_complexity = 0.5

            # Shadow clarity estimation
            brightness_std = float(np.std(gray_img))
            avg_brightness = float(np.mean(gray_img))
            dark_pixel_ratio = float(np.sum(gray_img < 50) / (height * width))

            if brightness_std > 60 and dark_pixel_ratio < 0.15 and avg_brightness > 100:
                shadow_clarity_score = 0.7
            elif brightness_std < 30 and dark_pixel_ratio > 0.1:
                shadow_clarity_score = 0.3
            else:
                shadow_clarity_score = 0.5

            # Edge density
            edges_density = min(1.0, (avg_abs_gx + avg_abs_gy) / 100.0)

            return {
                "gradient_ratio_vertical_horizontal": gradient_ratio_vertical_horizontal,
                "top_region_texture_complexity": top_region_texture_complexity,
                "shadow_clarity_score": shadow_clarity_score,
                "vertical_strength": avg_abs_gy,
                "horizontal_strength": avg_abs_gx,
                "edges_density": edges_density
            }

        except Exception as e:
            self.logger.error(f"Error computing texture features: {str(e)}")
            return {
                "gradient_ratio_vertical_horizontal": 1.0,
                "top_region_texture_complexity": 0.5,
                "shadow_clarity_score": 0.5,
                "vertical_strength": 0.0,
                "horizontal_strength": 0.0,
                "edges_density": 0.0
            }

    def compute_structure_features(self, small_gray: np.ndarray, gray_img: np.ndarray,
                                  hsv_img: np.ndarray, height: int, width: int,
                                  scale_factor: int) -> Dict[str, float]:
        """
        Compute structural features including ceiling likelihood and boundary clarity.

        Args:
            small_gray: Downscaled grayscale image.
            gray_img: Full-resolution grayscale image.
            hsv_img: HSV image for brightness analysis.
            height: Original image height.
            width: Original image width.
            scale_factor: Downscaling factor used.

        Returns:
            Dictionary containing structural features.
        """
        try:
            # Compute gradients
            gx = cv2.Sobel(small_gray, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(small_gray, cv2.CV_32F, 0, 1, ksize=3)
            avg_abs_gx = float(np.mean(np.abs(gx)))
            avg_abs_gy = float(np.mean(np.abs(gy)))

            # Ceiling likelihood analysis
            ceiling_features = self._analyze_ceiling_likelihood(
                small_gray, hsv_img, gx, avg_abs_gx, height, scale_factor
            )

            # Boundary clarity analysis
            boundary_clarity = self._compute_boundary_clarity(small_gray, avg_abs_gx, avg_abs_gy)

            # Openness analysis
            openness_top_edge = self._compute_openness_top_edge(gy, avg_abs_gy)

            # Legacy compatibility features
            legacy_structure = self._compute_legacy_structure_features(gray_img, height)

            structure_features = {
                "ceiling_likelihood": ceiling_features["ceiling_likelihood"],
                "boundary_clarity": boundary_clarity,
                "openness_top_edge": openness_top_edge,
                **legacy_structure
            }

            return structure_features

        except Exception as e:
            self.logger.error(f"Error computing structure features: {str(e)}")
            return {
                "ceiling_likelihood": 0.0,
                "boundary_clarity": 0.0,
                "openness_top_edge": 0.5,
                "ceiling_uniformity": 0.5,
                "horizontal_line_ratio": 0.0
            }

    def _analyze_sky_region(self, h_channel: np.ndarray, s_channel: np.ndarray,
                           v_channel: np.ndarray, height: int) -> Dict[str, float]:
        """Analyze features specific to the sky region (top third of image)."""
        try:
            top_third_height = height // 3
            sky_region_v = v_channel[:top_third_height, :]
            sky_region_s = s_channel[:top_third_height, :]
            sky_region_h = h_channel[:top_third_height, :]

            if sky_region_v.size == 0:
                return self._get_default_sky_features()

            # Sky region brightness analysis
            sky_region_avg_brightness = float(np.mean(sky_region_v))
            overall_avg_brightness = float(np.mean(v_channel))
            sky_region_brightness_ratio = sky_region_avg_brightness / max(overall_avg_brightness, 1e-5)
            sky_region_saturation = float(np.mean(sky_region_s))

            # Sky blue dominance in sky region
            ft = self.config_manager.feature_thresholds
            sky_region_blue_pixels = np.sum(
                (sky_region_h >= ft.sky_blue_hue_min) & (sky_region_h <= ft.sky_blue_hue_max) &
                (sky_region_s > ft.sky_blue_sat_min) & (sky_region_v > ft.sky_blue_val_min)
            )
            sky_region_blue_dominance = float(sky_region_blue_pixels / max(1, sky_region_v.size))

            return {
                "sky_region_brightness_ratio": sky_region_brightness_ratio,
                "sky_region_saturation": sky_region_saturation,
                "sky_region_blue_dominance": sky_region_blue_dominance,
                "sky_brightness": sky_region_avg_brightness
            }

        except Exception as e:
            self.logger.error(f"Error analyzing sky region: {str(e)}")
            return self._get_default_sky_features()

    def _analyze_color_atmosphere(self, h_channel: np.ndarray, s_channel: np.ndarray,
                                 total_pixels: int) -> Dict[str, Any]:
        """Analyze warm/cool color atmosphere."""
        try:
            cr = self.config_manager.color_ranges

            # Warm colors detection
            warm_mask = np.zeros_like(h_channel, dtype=bool)
            for h_min, h_max in cr.warm_hue_ranges:
                warm_mask |= ((h_channel >= h_min) & (h_channel <= h_max))
            warm_ratio = float(np.sum(warm_mask & (s_channel > 30)) / total_pixels)

            # Cool colors detection
            cool_mask = np.zeros_like(h_channel, dtype=bool)
            for h_min, h_max in cr.cool_hue_ranges:
                cool_mask |= ((h_channel >= h_min) & (h_channel <= h_max))
            cool_ratio = float(np.sum(cool_mask & (s_channel > 30)) / total_pixels)

            # Determine overall atmosphere
            if warm_ratio > cool_ratio and warm_ratio > 0.3:
                color_atmosphere = "warm"
            elif cool_ratio > warm_ratio and cool_ratio > 0.3:
                color_atmosphere = "cool"
            else:
                color_atmosphere = "neutral"

            return {
                "warm_ratio": warm_ratio,
                "cool_ratio": cool_ratio,
                "color_atmosphere": color_atmosphere
            }

        except Exception as e:
            self.logger.error(f"Error analyzing color atmosphere: {str(e)}")
            return {
                "warm_ratio": 0.0,
                "cool_ratio": 0.0,
                "color_atmosphere": "neutral"
            }

    def _analyze_ceiling_likelihood(self, small_gray: np.ndarray, hsv_img: np.ndarray,
                                   gx: np.ndarray, avg_abs_gx: float, height: int,
                                   scale_factor: int) -> Dict[str, float]:
        """Analyze likelihood of ceiling presence."""
        try:
            ceiling_likelihood = 0.0
            config = self.config_manager.indoor_outdoor_thresholds

            # Get sky region brightness for analysis
            v_channel = hsv_img[:, :, 2]
            top_third_height = height // 3
            sky_region_v = v_channel[:top_third_height, :]
            sky_region_avg_brightness = float(np.mean(sky_region_v)) if sky_region_v.size > 0 else 0

            # Get top region texture complexity
            small_top_third_height = small_gray.shape[0] // 3
            small_sky_region_gray = small_gray[:small_top_third_height, :]

            if small_sky_region_gray.size > 0:
                laplacian_var = cv2.Laplacian(small_sky_region_gray, cv2.CV_64F).var()
                top_region_texture_complexity = min(1.0, laplacian_var / 1000.0)
            else:
                top_region_texture_complexity = 0.5

            # Condition 1: Simple texture and moderate brightness
            ceiling_texture_thresh = getattr(config, 'ceiling_texture_thresh', 0.4)
            ceiling_brightness_min = getattr(config, 'ceiling_brightness_min', 60)
            ceiling_brightness_max = getattr(config, 'ceiling_brightness_max', 230)

            if (top_region_texture_complexity < ceiling_texture_thresh and
                ceiling_brightness_min < sky_region_avg_brightness < ceiling_brightness_max):
                ceiling_likelihood += 0.45

            # Condition 2: Horizontal line strength
            top_horizontal_lines_strength = float(np.mean(np.abs(gx[:small_gray.shape[0]//3, :])))
            ceiling_horizontal_line_factor = getattr(config, 'ceiling_horizontal_line_factor', 1.15)

            if top_horizontal_lines_strength > avg_abs_gx * ceiling_horizontal_line_factor:
                ceiling_likelihood += 0.35

            # Condition 3: Central bright spot (lamp detection)
            center_y_sm, center_x_sm = small_gray.shape[0]//2, small_gray.shape[1]//2
            lamp_check_radius_y = small_gray.shape[0] // 8
            lamp_check_radius_x = small_gray.shape[1] // 8

            center_region = small_gray[
                max(0, center_y_sm - lamp_check_radius_y):min(small_gray.shape[0], center_y_sm + lamp_check_radius_y),
                max(0, center_x_sm - lamp_check_radius_x):min(small_gray.shape[1], center_x_sm + lamp_check_radius_x)
            ]

            if center_region.size > 0:
                avg_brightness = float(np.mean(small_gray))
                center_brightness = float(np.mean(center_region))
                ceiling_center_bright_factor = getattr(config, 'ceiling_center_bright_factor', 1.25)

                if center_brightness > avg_brightness * ceiling_center_bright_factor:
                    ceiling_likelihood += 0.30

            # Sky dominance analysis for penalty
            sky_region_blue_dominance = self._compute_sky_blue_dominance(hsv_img, height)
            sky_region_brightness_ratio = sky_region_avg_brightness / max(float(np.mean(v_channel)), 1e-5)

            # Penalties for strong sky signals
            ceiling_max_sky_blue_thresh = getattr(config, 'ceiling_max_sky_blue_thresh', 0.08)
            ceiling_max_sky_brightness_ratio = getattr(config, 'ceiling_max_sky_brightness_ratio', 1.15)

            if (sky_region_blue_dominance < ceiling_max_sky_blue_thresh and
                sky_region_brightness_ratio < ceiling_max_sky_brightness_ratio):
                ceiling_likelihood += 0.15

            # Strong sky override
            sky_blue_dominance_strong_thresh = getattr(config, 'sky_blue_dominance_strong_thresh', 0.25)
            sky_brightness_strong_thresh = getattr(config, 'sky_brightness_strong_thresh', 1.25)
            ceiling_sky_override_factor = getattr(config, 'ceiling_sky_override_factor', 0.1)

            if (sky_region_blue_dominance > sky_blue_dominance_strong_thresh and
                sky_region_brightness_ratio > sky_brightness_strong_thresh):
                ceiling_likelihood *= ceiling_sky_override_factor

            ceiling_likelihood = min(1.0, ceiling_likelihood)

            return {"ceiling_likelihood": ceiling_likelihood}

        except Exception as e:
            self.logger.error(f"Error analyzing ceiling likelihood: {str(e)}")
            return {"ceiling_likelihood": 0.0}

    def _compute_sky_blue_dominance(self, hsv_img: np.ndarray, height: int) -> float:
        """Compute blue dominance in sky region."""
        try:
            h_channel, s_channel, v_channel = cv2.split(hsv_img)
            top_third_height = height // 3
            sky_region_h = h_channel[:top_third_height, :]
            sky_region_s = s_channel[:top_third_height, :]
            sky_region_v = v_channel[:top_third_height, :]

            if sky_region_h.size == 0:
                return 0.0

            ft = self.config_manager.feature_thresholds
            sky_region_blue_pixels = np.sum(
                (sky_region_h >= ft.sky_blue_hue_min) & (sky_region_h <= ft.sky_blue_hue_max) &
                (sky_region_s > ft.sky_blue_sat_min) & (sky_region_v > ft.sky_blue_val_min)
            )

            return float(sky_region_blue_pixels / max(1, sky_region_h.size))

        except Exception as e:
            self.logger.error(f"Error computing sky blue dominance: {str(e)}")
            return 0.0

    def _compute_boundary_clarity(self, small_gray: np.ndarray, avg_abs_gx: float,
                                 avg_abs_gy: float) -> float:
        """Compute boundary clarity score."""
        try:
            edge_width_sm = max(1, small_gray.shape[1] // 10)
            edge_height_sm = max(1, small_gray.shape[0] // 10)

            # Edge gradients
            left_edge_grad_x = 0.0
            right_edge_grad_x = 0.0
            top_edge_grad_y = 0.0

            if small_gray.shape[1] > edge_width_sm:
                left_edge = small_gray[:, :edge_width_sm]
                right_edge = small_gray[:, -edge_width_sm:]
                left_edge_grad_x = float(np.mean(np.abs(cv2.Sobel(left_edge, cv2.CV_32F, 1, 0, ksize=3))))
                right_edge_grad_x = float(np.mean(np.abs(cv2.Sobel(right_edge, cv2.CV_32F, 1, 0, ksize=3))))

            if small_gray.shape[0] > edge_height_sm:
                top_edge = small_gray[:edge_height_sm, :]
                top_edge_grad_y = float(np.mean(np.abs(cv2.Sobel(top_edge, cv2.CV_32F, 0, 1, ksize=3))))

            # Normalize against average gradients
            boundary_clarity = (left_edge_grad_x + right_edge_grad_x + top_edge_grad_y) / (
                3 * max(avg_abs_gx, avg_abs_gy, 1e-5)
            )
            boundary_clarity = min(1.0, boundary_clarity / 1.5)

            return boundary_clarity

        except Exception as e:
            self.logger.error(f"Error computing boundary clarity: {str(e)}")
            return 0.0

    def _compute_openness_top_edge(self, gy: np.ndarray, avg_abs_gy: float) -> float:
        """Compute openness of top edge."""
        try:
            top_edge_strip_gy = float(np.mean(np.abs(gy[:max(1, gy.shape[0]//20), :])))
            openness_top_edge = 1.0 - min(1.0, top_edge_strip_gy / max(avg_abs_gy, 1e-5) / 0.5)
            return openness_top_edge
        except Exception as e:
            self.logger.error(f"Error computing top edge openness: {str(e)}")
            return 0.5

    def _compute_legacy_compatibility_features(self, hsv_img: np.ndarray, small_gray: np.ndarray,
                                             features: Dict[str, Any], scale_factor: int) -> Dict[str, Any]:
        """Compute additional features for backward compatibility."""
        try:
            v_channel = hsv_img[:, :, 2]

            # Light source detection
            light_features = self._detect_light_sources(v_channel, features["avg_brightness"],
                                                       features["brightness_std"], scale_factor)

            # Street line detection
            street_score = self._compute_street_line_score(small_gray)

            # Additional legacy features
            legacy_features = {
                **light_features,
                "street_line_score": street_score,
                "sky_blue_ratio": features.get("sky_like_blue_ratio", 0.0),  # Alias
                "gradient_ratio": features.get("gradient_ratio_vertical_horizontal", 1.0)  # Alias
            }

            return legacy_features

        except Exception as e:
            self.logger.error(f"Error computing legacy compatibility features: {str(e)}")
            return {}

    def _detect_light_sources(self, v_channel: np.ndarray, avg_brightness: float,
                             brightness_std: float, scale_factor: int) -> Dict[str, float]:
        """Detect artificial light sources in the image."""
        try:
            # Sample pixels for efficiency
            sampled_v = v_channel[::scale_factor*2, ::scale_factor*2]

            # Light threshold
            light_threshold = min(
                self.config_manager.feature_thresholds.light_source_abs_thresh,
                avg_brightness + 2 * brightness_std
            )

            is_bright_spots = sampled_v > light_threshold
            bright_spot_count = int(np.sum(is_bright_spots))

            # Initialize light features
            circular_light_count = 0
            indoor_light_score = 0.0
            light_distribution_uniformity = 0.5

            # Analyze light distribution if spots are found
            if 1 < bright_spot_count < 20:
                bright_y, bright_x = np.where(is_bright_spots)
                if len(bright_y) > 1:
                    mean_x, mean_y = np.mean(bright_x), np.mean(bright_y)
                    dist_from_center = np.sqrt((bright_x - mean_x)**2 + (bright_y - mean_y)**2)

                    if np.std(dist_from_center) < np.mean(dist_from_center):
                        circular_light_count = min(3, len(bright_y) // 2)
                        light_distribution_uniformity = 0.7

                    if np.mean(bright_y) < sampled_v.shape[0] / 2:
                        indoor_light_score = 0.6
                    else:
                        indoor_light_score = 0.3

            return {
                "bright_spot_count": bright_spot_count,
                "circular_light_count": circular_light_count,
                "indoor_light_score": indoor_light_score,
                "light_distribution_uniformity": light_distribution_uniformity
            }

        except Exception as e:
            self.logger.error(f"Error detecting light sources: {str(e)}")
            return {
                "bright_spot_count": 0,
                "circular_light_count": 0,
                "indoor_light_score": 0.0,
                "light_distribution_uniformity": 0.5
            }

    def _compute_street_line_score(self, small_gray: np.ndarray) -> float:
        """Compute street line detection score."""
        try:
            street_line_score = 0.0
            bottom_half_sm = small_gray[small_gray.shape[0]//2:, :]

            if bottom_half_sm.size > 0:
                bottom_vert_gradient = cv2.Sobel(bottom_half_sm, cv2.CV_32F, 0, 1, ksize=3)
                strong_vert_lines = np.abs(bottom_vert_gradient) > 50

                if np.sum(strong_vert_lines) > (bottom_half_sm.size * 0.05):
                    street_line_score = 0.7

            return street_line_score

        except Exception as e:
            self.logger.error(f"Error computing street line score: {str(e)}")
            return 0.0

    def _compute_legacy_structure_features(self, gray_img: np.ndarray, height: int) -> Dict[str, float]:
        """Compute legacy structure features for backward compatibility."""
        try:
            # Top region analysis for ceiling uniformity
            top_region = gray_img[:height//4, :]
            top_region_std = float(np.std(top_region)) if top_region.size > 0 else 0.0
            ceiling_uniformity = 1.0 - min(1.0, top_region_std / max(float(np.mean(top_region)) if top_region.size > 0 else 1e-5, 1e-5))

            # Horizontal line detection in top region
            if top_region.size > 0:
                top_gradients = np.abs(cv2.Sobel(top_region, cv2.CV_32F, 0, 1, ksize=3))
                horizontal_lines_strength = float(np.mean(top_gradients))
                horizontal_line_ratio = min(1.0, horizontal_lines_strength / 40.0)
            else:
                horizontal_line_ratio = 0.0

            # Boundary edge score computation
            boundary_edge_score = self._compute_legacy_boundary_score(gray_img)

            return {
                "ceiling_uniformity": ceiling_uniformity,
                "horizontal_line_ratio": horizontal_line_ratio,
                "top_region_std": top_region_std,
                "boundary_edge_score": boundary_edge_score
            }

        except Exception as e:
            self.logger.error(f"Error computing legacy structure features: {str(e)}")
            return {
                "ceiling_uniformity": 0.5,
                "horizontal_line_ratio": 0.0,
                "top_region_std": 0.0,
                "boundary_edge_score": 0.0
            }

    def _compute_legacy_boundary_score(self, gray_img: np.ndarray) -> float:
        """Compute legacy boundary edge score."""
        try:
            height, width = gray_img.shape

            # Create small version for boundary analysis
            small_height, small_width = height // 4, width // 4
            small_gray = cv2.resize(gray_img, (small_width, small_height), interpolation=cv2.INTER_AREA)

            # Edge regions
            left_edge_sm = small_gray[:, :small_width//6] if small_width > 6 else small_gray
            right_edge_sm = small_gray[:, 5*small_width//6:] if small_width > 6 else small_gray
            top_edge_sm = small_gray[:small_height//6, :] if small_height > 6 else small_gray

            # Compute gradients for each edge
            left_gradient = float(np.mean(np.abs(cv2.Sobel(left_edge_sm, cv2.CV_32F, 1, 0, ksize=3)))) if left_edge_sm.size > 0 else 0
            right_gradient = float(np.mean(np.abs(cv2.Sobel(right_edge_sm, cv2.CV_32F, 1, 0, ksize=3)))) if right_edge_sm.size > 0 else 0
            top_gradient = float(np.mean(np.abs(cv2.Sobel(top_edge_sm, cv2.CV_32F, 0, 1, ksize=3)))) if top_edge_sm.size > 0 else 0

            # Combine and normalize
            boundary_edge_score = (min(1.0, left_gradient/50) + min(1.0, right_gradient/50) + min(1.0, top_gradient/50)) / 3

            return boundary_edge_score

        except Exception as e:
            self.logger.error(f"Error computing legacy boundary score: {str(e)}")
            return 0.0

    def _validate_image(self, image_rgb: np.ndarray) -> bool:
        """Validate input image format and dimensions."""
        try:
            if not isinstance(image_rgb, np.ndarray):
                self.logger.error("Input is not a numpy array")
                return False

            if len(image_rgb.shape) != 3 or image_rgb.shape[2] != 3:
                self.logger.error(f"Invalid image shape: {image_rgb.shape}. Expected (H, W, 3)")
                return False

            height, width = image_rgb.shape[:2]
            if height == 0 or width == 0:
                self.logger.error(f"Invalid image dimensions: {height}x{width}")
                return False

            return True

        except Exception as e:
            self.logger.error(f"Error validating image: {str(e)}")
            return False

    def _calculate_scale_factor(self, height: int, width: int) -> int:
        """Calculate appropriate scale factor for image processing efficiency."""
        try:
            base_scale = 4
            scale_factor = base_scale + min(8, max(0, int((height * width) / (1000 * 1000)) if height * width > 0 else 0))
            return max(1, scale_factor)
        except Exception as e:
            self.logger.error(f"Error calculating scale factor: {str(e)}")
            return 4

    def _get_default_features(self) -> Dict[str, Any]:
        """Return default feature values in case of processing errors."""
        return {
            "avg_brightness": 100.0,
            "brightness_std": 50.0,
            "dark_pixel_ratio": 0.0,
            "bright_pixel_ratio": 0.0,
            "brightness_uniformity": 0.5,
            "blue_ratio": 0.0,
            "sky_like_blue_ratio": 0.0,
            "yellow_orange_ratio": 0.0,
            "gray_ratio": 0.0,
            "avg_saturation": 100.0,
            "sky_region_brightness_ratio": 1.0,
            "sky_region_saturation": 0.0,
            "sky_region_blue_dominance": 0.0,
            "sky_brightness": 100.0,
            "warm_ratio": 0.0,
            "cool_ratio": 0.0,
            "color_atmosphere": "neutral",
            "gradient_ratio_vertical_horizontal": 1.0,
            "top_region_texture_complexity": 0.5,
            "shadow_clarity_score": 0.5,
            "vertical_strength": 0.0,
            "horizontal_strength": 0.0,
            "edges_density": 0.0,
            "ceiling_likelihood": 0.0,
            "boundary_clarity": 0.0,
            "openness_top_edge": 0.5,
            "ceiling_uniformity": 0.5,
            "horizontal_line_ratio": 0.0,
            "top_region_std": 0.0,
            "boundary_edge_score": 0.0,
            "bright_spot_count": 0,
            "circular_light_count": 0,
            "indoor_light_score": 0.0,
            "light_distribution_uniformity": 0.5,
            "street_line_score": 0.0,
            "sky_blue_ratio": 0.0,
            "gradient_ratio": 1.0
        }

    def _get_default_color_features(self) -> Dict[str, Any]:
        """Return default color feature values."""
        return {
            "blue_ratio": 0.0,
            "sky_like_blue_ratio": 0.0,
            "yellow_orange_ratio": 0.0,
            "gray_ratio": 0.0,
            "avg_saturation": 100.0,
            "sky_region_brightness_ratio": 1.0,
            "sky_region_saturation": 0.0,
            "sky_region_blue_dominance": 0.0,
            "sky_brightness": 100.0,
            "warm_ratio": 0.0,
            "cool_ratio": 0.0,
            "color_atmosphere": "neutral"
        }

    def _get_default_sky_features(self) -> Dict[str, float]:
        """Return default sky region feature values."""
        return {
            "sky_region_brightness_ratio": 1.0,
            "sky_region_saturation": 0.0,
            "sky_region_blue_dominance": 0.0,
            "sky_brightness": 100.0
        }
