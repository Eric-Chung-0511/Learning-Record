import cv2
import numpy as np
import traceback
from PIL import Image
import logging
from typing import Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ImageBlender:
    """
    Advanced image blending with aggressive spill suppression and color replacement.

    Supports two primary modes:
    - Background generation: Foreground preservation with edge refinement
    - Inpainting: Seamless blending with adaptive color correction

    Attributes:
        enable_multi_scale: Whether multi-scale edge refinement is enabled
    """

    EDGE_EROSION_PIXELS = 1          # Pixels to erode from mask edge
    ALPHA_BINARIZE_THRESHOLD = 0.5   # Alpha threshold for binarization
    DARK_LUMINANCE_THRESHOLD = 60    # Luminance threshold for dark foreground
    FOREGROUND_PROTECTION_THRESHOLD = 140  # Mask value for strong protection
    BACKGROUND_COLOR_TOLERANCE = 30  # DeltaE tolerance for background detection

    # Inpainting-specific parameters
    INPAINT_FEATHER_SCALE = 1.2      # Scale factor for inpainting feathering
    INPAINT_COLOR_BLEND_RADIUS = 10  # Radius for color adaptation zone

    def __init__(self, enable_multi_scale: bool = True):
        """
        Initialize ImageBlender.

        Parameters
        ----------
        enable_multi_scale : bool
            Whether to enable multi-scale edge refinement (default True)
        """
        self.enable_multi_scale = enable_multi_scale
        self._debug_info = {}
        self._adaptive_strength_map = None

    def _erode_mask_edges(
        self,
        mask_array: np.ndarray,
        erosion_pixels: int = 2
    ) -> np.ndarray:
        """
        Erode mask edges to remove contaminated boundary pixels.

        This removes the outermost pixels of the foreground mask where
        color contamination from the original background is most likely.

        Args:
            mask_array: Input mask as numpy array (uint8, 0-255)
            erosion_pixels: Number of pixels to erode (default 2)

        Returns:
            Eroded mask array (uint8)
        """
        if erosion_pixels <= 0:
            return mask_array

        # Use elliptical kernel for natural-looking erosion
        kernel_size = max(2, erosion_pixels)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (kernel_size, kernel_size)
        )

        # Apply erosion
        eroded = cv2.erode(mask_array, kernel, iterations=1)

        # Slight blur to smooth the eroded edges
        eroded = cv2.GaussianBlur(eroded, (3, 3), 0)

        logger.debug(f"Mask erosion applied: {erosion_pixels}px, kernel size: {kernel_size}")
        return eroded

    def _binarize_edge_alpha(
        self,
        alpha: np.ndarray,
        mask_array: np.ndarray,
        orig_array: np.ndarray,
        threshold: float = 0.45
    ) -> np.ndarray:
        """
        Binarize semi-transparent edge pixels to eliminate color bleeding.

        Semi-transparent pixels at edges cause visible contamination because
        they blend the original (potentially dark) foreground with the new
        background. This method forces edge pixels to be either fully opaque
        or fully transparent.

        Args:
            alpha: Current alpha channel (float32, 0.0-1.0)
            mask_array: Original mask array (uint8, 0-255)
            orig_array: Original foreground image array (uint8, RGB)
            threshold: Alpha threshold for binarization decision (default 0.45)

        Returns:
            Modified alpha array with binarized edges (float32)
        """
        # Identify semi-transparent edge zone (not fully opaque, not fully transparent)
        edge_zone = (alpha > 0.05) & (alpha < 0.95)

        if not np.any(edge_zone):
            return alpha

        # Calculate local foreground luminance for adaptive thresholding
        gray = np.mean(orig_array, axis=2)

        # For dark foreground pixels, use slightly higher threshold
        # to preserve more of the dark subject
        is_dark = gray < self.DARK_LUMINANCE_THRESHOLD

        # Create adaptive threshold map
        adaptive_threshold = np.full_like(alpha, threshold)
        adaptive_threshold[is_dark] = threshold + 0.1  # Keep more dark pixels

        # Binarize: above threshold -> opaque, below -> transparent
        alpha_binarized = alpha.copy()

        # Pixels above threshold become fully opaque
        make_opaque = edge_zone & (alpha > adaptive_threshold)
        alpha_binarized[make_opaque] = 1.0

        # Pixels below threshold become fully transparent
        make_transparent = edge_zone & (alpha <= adaptive_threshold)
        alpha_binarized[make_transparent] = 0.0

        # Log statistics
        num_opaque = np.sum(make_opaque)
        num_transparent = np.sum(make_transparent)
        logger.info(f"Edge binarization: {num_opaque} pixels -> opaque, {num_transparent} pixels -> transparent")

        return alpha_binarized

    def _apply_edge_cleanup(
        self,
        result_array: np.ndarray,
        bg_array: np.ndarray,
        alpha: np.ndarray,
        cleanup_width: int = 2
    ) -> np.ndarray:
        """
        Final cleanup pass to remove any remaining edge artifacts.

        Detects remaining semi-transparent edges and replaces them with
        either pure foreground or pure background colors.

        Args:
            result_array: Current blended result (uint8, RGB)
            bg_array: Background image array (uint8, RGB)
            alpha: Final alpha channel (float32, 0.0-1.0)
            cleanup_width: Width of edge zone to clean (default 2)

        Returns:
            Cleaned result array (uint8)
        """
        # Find edge pixels that might still have artifacts
        # These are pixels with alpha close to but not exactly 0 or 1
        residual_edge = (alpha > 0.01) & (alpha < 0.99) & (alpha != 0.0) & (alpha != 1.0)

        if not np.any(residual_edge):
            return result_array

        result_cleaned = result_array.copy()

        # For residual edge pixels, snap to nearest pure state
        snap_to_bg = residual_edge & (alpha < 0.5)
        snap_to_fg = residual_edge & (alpha >= 0.5)

        # Replace with background
        result_cleaned[snap_to_bg] = bg_array[snap_to_bg]

        # For foreground, keep original but ensure no blending artifacts
        # (already handled by the blend, so no action needed for snap_to_fg)

        num_cleaned = np.sum(residual_edge)
        if num_cleaned > 0:
            logger.debug(f"Edge cleanup: {num_cleaned} residual pixels cleaned")

        return result_cleaned

    def _remove_background_color_contamination(
        self,
        image_array: np.ndarray,
        mask_array: np.ndarray,
        orig_bg_color_lab: np.ndarray,
        tolerance: float = 30.0
    ) -> np.ndarray:
        """
        Remove original background color contamination from foreground pixels.

        Scans the foreground area for pixels that match the original background
        color and replaces them with nearby clean foreground colors.

        Args:
            image_array: Foreground image array (uint8, RGB)
            mask_array: Mask array (uint8, 0-255)
            orig_bg_color_lab: Original background color in Lab space
            tolerance: DeltaE tolerance for detecting contaminated pixels

        Returns:
            Cleaned image array (uint8)
        """
        # Convert to Lab for color comparison
        image_lab = cv2.cvtColor(image_array, cv2.COLOR_RGB2LAB).astype(np.float32)

        # Only process foreground pixels (mask > 50)
        foreground_mask = mask_array > 50

        if not np.any(foreground_mask):
            return image_array

        # Calculate deltaE from original background color for all pixels
        delta_l = image_lab[:, :, 0] - orig_bg_color_lab[0]
        delta_a = image_lab[:, :, 1] - orig_bg_color_lab[1]
        delta_b = image_lab[:, :, 2] - orig_bg_color_lab[2]
        delta_e = np.sqrt(delta_l**2 + delta_a**2 + delta_b**2)

        # Find contaminated pixels: in foreground but color similar to original background
        contaminated = foreground_mask & (delta_e < tolerance)

        if not np.any(contaminated):
            logger.debug("No background color contamination detected in foreground")
            return image_array

        num_contaminated = np.sum(contaminated)
        logger.info(f"Found {num_contaminated} pixels with background color contamination")

        # Create output array
        result = image_array.copy()

        # For contaminated pixels, use inpainting to replace with surrounding colors
        inpaint_mask = contaminated.astype(np.uint8) * 255

        try:
            # Use inpainting to fill contaminated areas with surrounding foreground colors
            result = cv2.inpaint(result, inpaint_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
            logger.info(f"Inpainted {num_contaminated} contaminated pixels")
        except Exception as e:
            logger.warning(f"Inpainting failed: {e}, using median filter fallback")
            # Fallback: apply median filter to contaminated areas
            median_filtered = cv2.medianBlur(image_array, 5)
            result[contaminated] = median_filtered[contaminated]

        return result

    def _protect_foreground_core(
        self,
        result_array: np.ndarray,
        orig_array: np.ndarray,
        mask_array: np.ndarray,
        protection_threshold: int = 140
    ) -> np.ndarray:
        """
        Strongly protect core foreground pixels from any background influence.

        For pixels with high mask confidence, directly use the original foreground
        color without any blending, ensuring faces and bodies are not affected.

        Args:
            result_array: Current blended result (uint8, RGB)
            orig_array: Original foreground image (uint8, RGB)
            mask_array: Mask array (uint8, 0-255)
            protection_threshold: Mask value above which pixels are fully protected

        Returns:
            Protected result array (uint8)
        """
        # Identify strongly protected foreground pixels
        strong_foreground = mask_array >= protection_threshold

        if not np.any(strong_foreground):
            return result_array

        # For these pixels, use original foreground color directly
        result_protected = result_array.copy()
        result_protected[strong_foreground] = orig_array[strong_foreground]

        num_protected = np.sum(strong_foreground)
        logger.info(f"Protected {num_protected} core foreground pixels from background influence")

        return result_protected

    def multi_scale_edge_refinement(
        self,
        original_image: Image.Image,
        background_image: Image.Image,
        mask: Image.Image
    ) -> Image.Image:
        """
        Multi-scale edge refinement for better edge quality.
        Uses image pyramid to handle edges at different scales.

        Args:
            original_image: Foreground PIL Image
            background_image: Background PIL Image
            mask: Current mask PIL Image

        Returns:
            Refined mask PIL Image
        """
        logger.info("🔍 Starting multi-scale edge refinement...")

        try:
            # Convert to numpy arrays
            orig_array = np.array(original_image.convert('RGB'))
            mask_array = np.array(mask).astype(np.float32)
            height, width = mask_array.shape

            # Define scales for pyramid
            scales = [1.0, 0.5, 0.25]  # Original, half, quarter
            scale_masks = []
            scale_complexities = []

            # Convert to grayscale for edge detection
            gray = cv2.cvtColor(orig_array, cv2.COLOR_RGB2GRAY)

            for scale in scales:
                if scale == 1.0:
                    scaled_gray = gray
                    scaled_mask = mask_array
                else:
                    new_h = int(height * scale)
                    new_w = int(width * scale)
                    scaled_gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
                    scaled_mask = cv2.resize(mask_array, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

                # Compute local complexity using gradient standard deviation
                sobel_x = cv2.Sobel(scaled_gray, cv2.CV_64F, 1, 0, ksize=3)
                sobel_y = cv2.Sobel(scaled_gray, cv2.CV_64F, 0, 1, ksize=3)
                gradient_mag = np.sqrt(sobel_x**2 + sobel_y**2)

                # Calculate local complexity in 5x5 regions
                kernel_size = 5
                complexity = cv2.blur(gradient_mag, (kernel_size, kernel_size))

                # Resize back to original size
                if scale != 1.0:
                    scaled_mask = cv2.resize(scaled_mask, (width, height), interpolation=cv2.INTER_LANCZOS4)
                    complexity = cv2.resize(complexity, (width, height), interpolation=cv2.INTER_LANCZOS4)

                scale_masks.append(scaled_mask)
                scale_complexities.append(complexity)

            # Compute weights based on complexity
            # High complexity -> use high resolution mask
            # Low complexity -> use low resolution mask (smoother)
            weights = np.zeros((len(scales), height, width), dtype=np.float32)

            # Normalize complexities
            max_complexity = max(c.max() for c in scale_complexities) + 1e-6
            normalized_complexities = [c / max_complexity for c in scale_complexities]

            # Weight assignment: higher complexity at each scale means that scale is more reliable
            for i, complexity in enumerate(normalized_complexities):
                if i == 0:  # High resolution - prefer for high complexity regions
                    weights[i] = complexity
                elif i == 1:  # Medium resolution - moderate complexity
                    weights[i] = 0.5 * (1 - complexity) + 0.5 * complexity * 0.5
                else:  # Low resolution - prefer for low complexity regions
                    weights[i] = 1 - complexity

            # Normalize weights so they sum to 1 at each pixel
            weight_sum = weights.sum(axis=0, keepdims=True) + 1e-6
            weights = weights / weight_sum

            # Weighted blend of masks from different scales
            refined_mask = np.zeros((height, width), dtype=np.float32)
            for i, mask_i in enumerate(scale_masks):
                refined_mask += weights[i] * mask_i

            # Clip and convert to uint8
            refined_mask = np.clip(refined_mask, 0, 255).astype(np.uint8)

            logger.info("✅ Multi-scale edge refinement completed")
            return Image.fromarray(refined_mask, mode='L')

        except Exception as e:
            logger.error(f"❌ Multi-scale refinement failed: {e}, using original mask")
            return mask

    def simple_blend_images(
        self,
        original_image: Image.Image,
        background_image: Image.Image,
        combination_mask: Image.Image,
        use_multi_scale: Optional[bool] = None,
        feather_radius: int = 0
    ) -> Image.Image:
        """
        Aggressive spill suppression + color replacement: completely eliminate yellow edge residue, maintain sharp edges

        Args:
            original_image: Foreground PIL Image
            background_image: Background PIL Image
            combination_mask: Mask PIL Image (L mode)
            use_multi_scale: Override for multi-scale refinement (None = use class default)
            feather_radius: Gaussian blur radius for mask feathering (0 = disabled, default behavior)

        Returns:
            Blended PIL Image
        """
        logger.info("🎨 Starting advanced image blending process...")

        # Apply multi-scale edge refinement if enabled
        should_use_multi_scale = use_multi_scale if use_multi_scale is not None else self.enable_multi_scale
        if should_use_multi_scale:
            combination_mask = self.multi_scale_edge_refinement(
                original_image, background_image, combination_mask
            )

        # Convert to numpy arrays
        orig_array = np.array(original_image, dtype=np.uint8)
        bg_array = np.array(background_image, dtype=np.uint8)
        mask_array = np.array(combination_mask, dtype=np.uint8)

        # Apply feathering if requested
        if feather_radius > 0:
            kernel_size = feather_radius * 2 + 1
            mask_array = cv2.GaussianBlur(
                mask_array,
                (kernel_size, kernel_size),
                feather_radius / 2.0
            )
            logger.info(f"📐 Mask feathering applied: radius={feather_radius}, kernel={kernel_size}x{kernel_size}")

        logger.info(f"📊 Image dimensions - Original: {orig_array.shape}, Background: {bg_array.shape}, Mask: {mask_array.shape}")
        logger.info(f"📊 Mask statistics (before erosion) - Mean: {mask_array.mean():.1f}, Min: {mask_array.min()}, Max: {mask_array.max()}")

        # === NEW: Apply mask erosion to remove contaminated edge pixels ===
        mask_array = self._erode_mask_edges(mask_array, self.EDGE_EROSION_PIXELS)
        logger.info(f"📊 Mask statistics (after erosion) - Mean: {mask_array.mean():.1f}, Min: {mask_array.min()}, Max: {mask_array.max()}")

        # Enhanced parameters for better spill suppression
        RING_WIDTH_PX = 4           # Increased ring width for better coverage
        SPILL_STRENGTH = 0.85       # Stronger spill suppression
        L_MATCH_STRENGTH = 0.65     # Stronger luminance matching
        DELTAE_THRESHOLD = 18       # More aggressive contamination detection
        HARD_EDGE_PROTECT = True    # Black edge protection
        INPAINT_FALLBACK = True     # inpaint fallback repair
        MULTI_PASS_CORRECTION = True # Enable multi-pass correction

        # Estimate original background color and foreground representative color ===
        height, width = orig_array.shape[:2]

        # Take 15px from each side to estimate original background color
        edge_width = 15
        border_pixels = []

        # Collect border pixels (excluding foreground areas)
        border_mask = np.zeros((height, width), dtype=bool)
        border_mask[:edge_width, :] = True  # Top edge
        border_mask[-edge_width:, :] = True  # Bottom edge
        border_mask[:, :edge_width] = True  # Left edge
        border_mask[:, -edge_width:] = True  # Right edge

        # Exclude foreground areas
        fg_binary = mask_array > 50
        border_mask = border_mask & (~fg_binary)

        if np.any(border_mask):
            border_pixels = orig_array[border_mask].reshape(-1, 3)

            # Simplified background color estimation (no sklearn dependency)
            try:
                if len(border_pixels) > 100:
                    # Use histogram to find mode colors
                    # Quantize RGB to coarser grid to find main colors
                    quantized = (border_pixels // 32) * 32  # 8-level quantization

                    # Find most frequent color
                    unique_colors, counts = np.unique(quantized.reshape(-1, quantized.shape[-1]),
                                                    axis=0, return_counts=True)
                    most_common_idx = np.argmax(counts)
                    orig_bg_color_rgb = unique_colors[most_common_idx].astype(np.uint8)
                else:
                    orig_bg_color_rgb = np.median(border_pixels, axis=0).astype(np.uint8)
            except:
                # Fallback: use four corners average
                corners = np.array([orig_array[0,0], orig_array[0,-1],
                                  orig_array[-1,0], orig_array[-1,-1]])
                orig_bg_color_rgb = np.mean(corners, axis=0).astype(np.uint8)
        else:
            orig_bg_color_rgb = np.array([200, 180, 120], dtype=np.uint8)  # Default yellow

        # Convert to Lab space
        orig_bg_color_lab = cv2.cvtColor(orig_bg_color_rgb.reshape(1,1,3), cv2.COLOR_RGB2LAB)[0,0].astype(np.float32)
        logger.info(f"🎨 Detected original background color: RGB{tuple(orig_bg_color_rgb)}")

        # Remove original background color contamination from foreground
        orig_array = self._remove_background_color_contamination(
            orig_array,
            mask_array,
            orig_bg_color_lab,
            tolerance=self.BACKGROUND_COLOR_TOLERANCE
        )

        # Redefine trimap, optimized for cartoon characters
        try:
            kernel_3x3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

            # FG_CORE: Reduce erosion iterations from 2 to 1 to avoid losing thin limbs
            mask_eroded_once = cv2.erode(mask_array, kernel_3x3, iterations=1)
            fg_core = mask_eroded_once > 127  # Adjustable parameter: erosion iterations

            # RING: Use morphological gradient to redefine, ensuring only thin edge band
            mask_dilated = cv2.dilate(mask_array, kernel_3x3, iterations=1)
            mask_eroded = cv2.erode(mask_array, kernel_3x3, iterations=1)

            # Ensure consistent data types to avoid overflow
            morphological_gradient = cv2.subtract(mask_dilated, mask_eroded)
            ring_zone = morphological_gradient > 0  # Areas with morphological gradient > 0 are edge bands

            # BG: background area
            bg_zone = mask_array < 30

            logger.info(f"🔍 Trimap regions - FG_CORE: {fg_core.sum()}, RING: {ring_zone.sum()}, BG: {bg_zone.sum()}")

        except Exception as e:
            logger.error(f"❌ Trimap definition failed: {e}")
            logger.error(f"📍 Traceback: {traceback.format_exc()}")
            print(f"❌ TRIMAP ERROR: {e}")
            print(f"Traceback: {traceback.format_exc()}")
            # Fallback to simple definition
            fg_core = mask_array > 200
            ring_zone = (mask_array > 50) & (mask_array <= 200)
            bg_zone = mask_array <= 50

        # Foreground representative color: estimated from FG_CORE
        if np.any(fg_core):
            fg_pixels = orig_array[fg_core].reshape(-1, 3)
            fg_rep_color_rgb = np.median(fg_pixels, axis=0).astype(np.uint8)
        else:
            fg_rep_color_rgb = np.array([80, 60, 40], dtype=np.uint8)  # Default dark

        fg_rep_color_lab = cv2.cvtColor(fg_rep_color_rgb.reshape(1,1,3), cv2.COLOR_RGB2LAB)[0,0].astype(np.float32)

        # Edge band spill suppression and repair
        if np.any(ring_zone):
            # Convert to Lab space
            orig_lab = cv2.cvtColor(orig_array, cv2.COLOR_RGB2LAB).astype(np.float32)
            orig_array_working = orig_array.copy().astype(np.float32)

            # ΔE detect contaminated pixels
            ring_pixels_lab = orig_lab[ring_zone]

            # Calculate ΔE with original background color (simplified version)
            delta_l = ring_pixels_lab[:, 0] - orig_bg_color_lab[0]
            delta_a = ring_pixels_lab[:, 1] - orig_bg_color_lab[1]
            delta_b = ring_pixels_lab[:, 2] - orig_bg_color_lab[2]
            delta_e = np.sqrt(delta_l**2 + delta_a**2 + delta_b**2)

            # Contaminated pixel mask
            contaminated_mask = delta_e < DELTAE_THRESHOLD

            if np.any(contaminated_mask):
                # Calculate adaptive strength based on delta_e for each pixel
                # Pixels closer to background color get stronger correction
                contaminated_delta_e = delta_e[contaminated_mask]

                # Adaptive strength formula: inverse relationship with delta_e
                # Pixels very close to bg color (low delta_e) -> strong correction
                # Pixels further from bg color (high delta_e) -> lighter correction
                adaptive_strength = SPILL_STRENGTH * np.maximum(
                    0.0,
                    1.0 - (contaminated_delta_e / DELTAE_THRESHOLD)
                )

                # Clamp adaptive strength to reasonable range (30% - 100% of base strength)
                min_strength = SPILL_STRENGTH * 0.3
                adaptive_strength = np.clip(adaptive_strength, min_strength, SPILL_STRENGTH)

                # Store for debug visualization
                self._adaptive_strength_map = np.zeros_like(delta_e)
                self._adaptive_strength_map[contaminated_mask] = adaptive_strength

                logger.info(f"📊 Adaptive strength stats - Mean: {adaptive_strength.mean():.3f}, Min: {adaptive_strength.min():.3f}, Max: {adaptive_strength.max():.3f}")

                # Chroma vector deprojection
                bg_chroma = np.array([orig_bg_color_lab[1], orig_bg_color_lab[2]])
                bg_chroma_norm = bg_chroma / (np.linalg.norm(bg_chroma) + 1e-6)

                # Color correction for contaminated pixels
                contaminated_pixels = ring_pixels_lab[contaminated_mask]

                # Remove background chroma component with adaptive strength (per-pixel)
                pixel_chroma = contaminated_pixels[:, 1:3]  # a, b channels
                projection = np.dot(pixel_chroma, bg_chroma_norm)[:, np.newaxis] * bg_chroma_norm

                # Apply adaptive strength per pixel
                adaptive_strength_2d = adaptive_strength[:, np.newaxis]
                corrected_chroma = pixel_chroma - projection * adaptive_strength_2d

                # Converge toward foreground representative color with adaptive strength
                convergence_factor = adaptive_strength_2d * 0.6
                corrected_chroma = (corrected_chroma * (1 - convergence_factor) +
                                  fg_rep_color_lab[1:3] * convergence_factor)

                # Adaptive luminance matching
                adaptive_l_strength = adaptive_strength * (L_MATCH_STRENGTH / SPILL_STRENGTH)
                corrected_l = (contaminated_pixels[:, 0] * (1 - adaptive_l_strength) +
                             fg_rep_color_lab[0] * adaptive_l_strength)

                # Update Lab values
                ring_pixels_lab[contaminated_mask, 0] = corrected_l
                ring_pixels_lab[contaminated_mask, 1:3] = corrected_chroma

                # Write back to original image
                orig_lab[ring_zone] = ring_pixels_lab

            # Dark edge protection
            if HARD_EDGE_PROTECT:
                gray = np.mean(orig_array, axis=2)
                # Detect dark and high gradient areas
                sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
                sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
                gradient_mag = np.sqrt(sobel_x**2 + sobel_y**2)

                dark_edge_zone = ring_zone & (gray < 60) & (gradient_mag > 20)
                # Protect these areas from excessive modification, copy directly from original
                if np.any(dark_edge_zone):
                    orig_lab[dark_edge_zone] = cv2.cvtColor(orig_array, cv2.COLOR_RGB2LAB)[dark_edge_zone]

            # Multi-pass correction for stubborn spill
            if MULTI_PASS_CORRECTION:
                # Second pass for remaining contamination
                ring_pixels_lab_pass2 = orig_lab[ring_zone]
                delta_l_pass2 = ring_pixels_lab_pass2[:, 0] - orig_bg_color_lab[0]
                delta_a_pass2 = ring_pixels_lab_pass2[:, 1] - orig_bg_color_lab[1]
                delta_b_pass2 = ring_pixels_lab_pass2[:, 2] - orig_bg_color_lab[2]
                delta_e_pass2 = np.sqrt(delta_l_pass2**2 + delta_a_pass2**2 + delta_b_pass2**2)

                still_contaminated = delta_e_pass2 < (DELTAE_THRESHOLD * 0.8)

                if np.any(still_contaminated):
                    # Apply stronger correction to remaining contaminated pixels
                    remaining_pixels = ring_pixels_lab_pass2[still_contaminated]

                    # More aggressive chroma neutralization
                    remaining_chroma = remaining_pixels[:, 1:3]
                    neutralized_chroma = remaining_chroma * 0.3 + fg_rep_color_lab[1:3] * 0.7

                    # Stronger luminance matching
                    neutralized_l = remaining_pixels[:, 0] * 0.4 + fg_rep_color_lab[0] * 0.6

                    ring_pixels_lab_pass2[still_contaminated, 0] = neutralized_l
                    ring_pixels_lab_pass2[still_contaminated, 1:3] = neutralized_chroma
                    orig_lab[ring_zone] = ring_pixels_lab_pass2

            # Convert back to RGB
            orig_lab_clipped = np.clip(orig_lab, 0, 255).astype(np.uint8)
            orig_array_corrected = cv2.cvtColor(orig_lab_clipped, cv2.COLOR_LAB2RGB)

            # inpaint fallback repair
            if INPAINT_FALLBACK:
                # inpaint still contaminated outermost pixels
                final_contaminated = ring_zone.copy()

                # Check if there's still contamination after repair
                final_lab = cv2.cvtColor(orig_array_corrected, cv2.COLOR_RGB2LAB).astype(np.float32)
                final_ring_lab = final_lab[ring_zone]
                final_delta_l = final_ring_lab[:, 0] - orig_bg_color_lab[0]
                final_delta_a = final_ring_lab[:, 1] - orig_bg_color_lab[1]
                final_delta_b = final_ring_lab[:, 2] - orig_bg_color_lab[2]
                final_delta_e = np.sqrt(final_delta_l**2 + final_delta_a**2 + final_delta_b**2)

                still_contaminated = final_delta_e < (DELTAE_THRESHOLD * 0.5)
                if np.any(still_contaminated):
                    # Create inpaint mask
                    inpaint_mask = np.zeros((height, width), dtype=np.uint8)
                    ring_coords = np.where(ring_zone)
                    inpaint_coords = (ring_coords[0][still_contaminated], ring_coords[1][still_contaminated])
                    inpaint_mask[inpaint_coords] = 255

                    # Execute inpaint
                    try:
                        orig_array_corrected = cv2.inpaint(orig_array_corrected, inpaint_mask, 3, cv2.INPAINT_TELEA)
                    except:
                        # Fallback: directly cover with foreground representative color
                        orig_array_corrected[inpaint_coords] = fg_rep_color_rgb

            orig_array = orig_array_corrected

        # === Linear space blending (keep original logic) ===
        def srgb_to_linear(img):
            img_f = img.astype(np.float32) / 255.0
            return np.where(img_f <= 0.04045, img_f / 12.92, np.power((img_f + 0.055) / 1.055, 2.4))

        def linear_to_srgb(img):
            img_clipped = np.clip(img, 0, 1)
            return np.where(img_clipped <= 0.0031308,
                           12.92 * img_clipped,
                           1.055 * np.power(img_clipped, 1/2.4) - 0.055)

        orig_linear = srgb_to_linear(orig_array)
        bg_linear = srgb_to_linear(bg_array)

        # Cartoon-optimized Alpha calculation
        alpha = mask_array.astype(np.float32) / 255.0

        # Core foreground region - fully opaque
        alpha[fg_core] = 1.0

        # Background region - fully transparent
        alpha[bg_zone] = 0.0

        # [Key Fix] Force pixels with mask≥160 to α=1.0, avoiding white fill areas being limited to 0.9
        high_confidence_pixels = mask_array >= 160
        alpha[high_confidence_pixels] = 1.0
        logger.info(f"💯 High confidence pixels set to full opacity: {high_confidence_pixels.sum()}")

        # Ring area can be dehaloed, but doesn't affect already set high confidence pixels
        ring_without_high_conf = ring_zone & (~high_confidence_pixels)
        alpha[ring_without_high_conf] = np.clip(alpha[ring_without_high_conf], 0.2, 0.9)

        # Retain existing black outline/strong edge protection
        orig_gray = np.mean(orig_array, axis=2)

        # Detect strong edge areas
        sobel_x = cv2.Sobel(orig_gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(orig_gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_mag = np.sqrt(sobel_x**2 + sobel_y**2)

        # Black outline/strong edge protection: nearly fully opaque
        black_edge_threshold = 60  # black edge threshold
        gradient_threshold = 25    # gradient threshold
        strong_edges = (orig_gray < black_edge_threshold) & (gradient_mag > gradient_threshold) & (mask_array > 10)
        alpha[strong_edges] = np.maximum(alpha[strong_edges], 0.995)  # black edge alpha

        logger.info(f"🛡️ Protection applied - High conf: {high_confidence_pixels.sum()}, Strong edges: {strong_edges.sum()}")

        # Apply edge alpha binarization to eliminate semi-transparent artifacts
        alpha = self._binarize_edge_alpha(
            alpha,
            mask_array,
            orig_array,
            threshold=self.ALPHA_BINARIZE_THRESHOLD
        )

        # Final blending
        alpha_3d = alpha[:, :, np.newaxis]
        result_linear = orig_linear * alpha_3d + bg_linear * (1 - alpha_3d)
        result_srgb = linear_to_srgb(result_linear)
        result_array = (result_srgb * 255).astype(np.uint8)

        # Final edge cleanup pass
        result_array = self._apply_edge_cleanup(result_array, bg_array, alpha)

        # Protect core foreground from any background influence
        # This ensures faces and bodies retain original colors
        result_array = self._protect_foreground_core(
            result_array,
            np.array(original_image, dtype=np.uint8),  # Use original unprocessed image
            mask_array,
            protection_threshold=self.FOREGROUND_PROTECTION_THRESHOLD
        )

        # Store debug information (for debug output)
        self._debug_info = {
            'orig_bg_color_rgb': orig_bg_color_rgb,
            'fg_rep_color_rgb': fg_rep_color_rgb,
            'orig_bg_color_lab': orig_bg_color_lab,
            'fg_rep_color_lab': fg_rep_color_lab,
            'ring_zone': ring_zone,
            'fg_core': fg_core,
            'alpha_final': alpha
        }

        return Image.fromarray(result_array)

    def create_debug_images(
        self,
        original_image: Image.Image,
        generated_background: Image.Image,
        combination_mask: Image.Image,
        combined_image: Image.Image
    ) -> Dict[str, Image.Image]:
        """
        Generate debug images: (a) Final mask grayscale (b) Alpha heatmap (c) Ring visualization overlay
        """
        debug_images = {}

        # Final Mask grayscale
        debug_images["mask_gray"] = combination_mask.convert('L')

        # Alpha Heatmap
        mask_array = np.array(combination_mask.convert('L'))
        heatmap_colored = cv2.applyColorMap(mask_array, cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        debug_images["alpha_heatmap"] = Image.fromarray(heatmap_rgb)

        # Ring visualization overlay - show ring areas on original image
        if hasattr(self, '_debug_info') and 'ring_zone' in self._debug_info:
            ring_zone = self._debug_info['ring_zone']
            orig_array = np.array(original_image)
            ring_overlay = orig_array.copy()

            # Mark ring areas with red semi-transparent overlay
            ring_overlay[ring_zone] = ring_overlay[ring_zone] * 0.7 + np.array([255, 0, 0]) * 0.3
            debug_images["ring_visualization"] = Image.fromarray(ring_overlay.astype(np.uint8))
        else:
            # If no ring information, use original image
            debug_images["ring_visualization"] = original_image

        # Adaptive strength heatmap - visualize per-pixel correction strength
        if hasattr(self, '_adaptive_strength_map') and self._adaptive_strength_map is not None:
            # Normalize adaptive strength to 0-255 for visualization
            strength_map = self._adaptive_strength_map
            if strength_map.max() > 0:
                normalized_strength = (strength_map / strength_map.max() * 255).astype(np.uint8)
            else:
                normalized_strength = np.zeros_like(strength_map, dtype=np.uint8)

            # Apply colormap
            strength_heatmap = cv2.applyColorMap(normalized_strength, cv2.COLORMAP_VIRIDIS)
            strength_heatmap_rgb = cv2.cvtColor(strength_heatmap, cv2.COLOR_BGR2RGB)
            debug_images["adaptive_strength_heatmap"] = Image.fromarray(strength_heatmap_rgb)

        return debug_images

    # INPAINTING-SPECIFIC BLENDING METHODS
    def blend_inpainting(
        self,
        original: Image.Image,
        generated: Image.Image,
        mask: Image.Image,
        feather_radius: int = 8,
        apply_color_correction: bool = True
    ) -> Image.Image:
        """
        Blend inpainted region with original image.

        Specialized blending for inpainting that focuses on seamless integration
        rather than foreground protection. Performs blending in linear color space
        with optional adaptive color correction at boundaries.

        Parameters
        ----------
        original : PIL.Image
            Original image before inpainting
        generated : PIL.Image
            Generated/inpainted result from the model
        mask : PIL.Image
            Inpainting mask (white = inpainted area)
        feather_radius : int
            Feathering radius for smooth transitions
        apply_color_correction : bool
            Whether to apply adaptive color correction at boundaries

        Returns
        -------
        PIL.Image
            Blended result
        """
        logger.info(f"Inpainting blend: feather={feather_radius}, color_correction={apply_color_correction}")

        # Ensure same size
        if generated.size != original.size:
            generated = generated.resize(original.size, Image.LANCZOS)
        if mask.size != original.size:
            mask = mask.resize(original.size, Image.LANCZOS)

        # Convert to arrays
        orig_array = np.array(original.convert('RGB')).astype(np.float32)
        gen_array = np.array(generated.convert('RGB')).astype(np.float32)
        mask_array = np.array(mask.convert('L')).astype(np.float32) / 255.0

        # Apply feathering to mask
        if feather_radius > 0:
            scaled_radius = int(feather_radius * self.INPAINT_FEATHER_SCALE)
            kernel_size = scaled_radius * 2 + 1
            mask_array = cv2.GaussianBlur(
                mask_array,
                (kernel_size, kernel_size),
                scaled_radius / 2
            )

        # Apply adaptive color correction if enabled
        if apply_color_correction:
            gen_array = self._apply_inpaint_color_correction(
                orig_array, gen_array, mask_array
            )

        # sRGB to linear conversion for accurate blending
        def srgb_to_linear(img):
            img_norm = img / 255.0
            return np.where(
                img_norm <= 0.04045,
                img_norm / 12.92,
                np.power((img_norm + 0.055) / 1.055, 2.4)
            )

        def linear_to_srgb(img):
            img_clipped = np.clip(img, 0, 1)
            return np.where(
                img_clipped <= 0.0031308,
                12.92 * img_clipped,
                1.055 * np.power(img_clipped, 1/2.4) - 0.055
            )

        # Convert to linear space
        orig_linear = srgb_to_linear(orig_array)
        gen_linear = srgb_to_linear(gen_array)

        # Alpha blending in linear space
        alpha = mask_array[:, :, np.newaxis]
        result_linear = gen_linear * alpha + orig_linear * (1 - alpha)

        # Convert back to sRGB
        result_srgb = linear_to_srgb(result_linear)
        result_array = (result_srgb * 255).astype(np.uint8)

        logger.debug("Inpainting blend completed in linear color space")

        return Image.fromarray(result_array)

    def _apply_inpaint_color_correction(
        self,
        original: np.ndarray,
        generated: np.ndarray,
        mask: np.ndarray
    ) -> np.ndarray:
        """
        Apply adaptive color correction to match generated region with surroundings.

        Analyzes the boundary region and adjusts the generated content's
        luminance and color to better match the original context.

        Parameters
        ----------
        original : np.ndarray
            Original image (float32, 0-255)
        generated : np.ndarray
            Generated image (float32, 0-255)
        mask : np.ndarray
            Blend mask (float32, 0-1)

        Returns
        -------
        np.ndarray
            Color-corrected generated image
        """
        # Find boundary region
        mask_binary = (mask > 0.5).astype(np.uint8)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self.INPAINT_COLOR_BLEND_RADIUS * 2 + 1, self.INPAINT_COLOR_BLEND_RADIUS * 2 + 1)
        )
        dilated = cv2.dilate(mask_binary, kernel, iterations=1)
        boundary_zone = (dilated > 0) & (mask < 0.3)

        if not np.any(boundary_zone):
            return generated

        # Convert to Lab for perceptual color matching
        orig_lab = cv2.cvtColor(
            original.astype(np.uint8), cv2.COLOR_RGB2LAB
        ).astype(np.float32)
        gen_lab = cv2.cvtColor(
            generated.astype(np.uint8), cv2.COLOR_RGB2LAB
        ).astype(np.float32)

        # Calculate statistics in boundary zone (original)
        boundary_orig_l = orig_lab[boundary_zone, 0]
        boundary_orig_a = orig_lab[boundary_zone, 1]
        boundary_orig_b = orig_lab[boundary_zone, 2]

        orig_mean_l = np.median(boundary_orig_l)
        orig_mean_a = np.median(boundary_orig_a)
        orig_mean_b = np.median(boundary_orig_b)

        # Calculate statistics in generated inpaint region
        inpaint_zone = mask > 0.5
        if not np.any(inpaint_zone):
            return generated

        gen_inpaint_l = gen_lab[inpaint_zone, 0]
        gen_inpaint_a = gen_lab[inpaint_zone, 1]
        gen_inpaint_b = gen_lab[inpaint_zone, 2]

        gen_mean_l = np.median(gen_inpaint_l)
        gen_mean_a = np.median(gen_inpaint_a)
        gen_mean_b = np.median(gen_inpaint_b)

        # Calculate correction deltas
        delta_l = orig_mean_l - gen_mean_l
        delta_a = orig_mean_a - gen_mean_a
        delta_b = orig_mean_b - gen_mean_b

        # Limit correction to avoid over-adjustment
        max_correction = 15
        delta_l = np.clip(delta_l, -max_correction, max_correction)
        delta_a = np.clip(delta_a, -max_correction * 0.5, max_correction * 0.5)
        delta_b = np.clip(delta_b, -max_correction * 0.5, max_correction * 0.5)

        logger.debug(f"Color correction deltas: L={delta_l:.1f}, a={delta_a:.1f}, b={delta_b:.1f}")

        # Apply correction with spatial falloff from boundary
        # Create distance map from boundary
        distance = cv2.distanceTransform(
            mask_binary, cv2.DIST_L2, 5
        )
        max_dist = np.max(distance)
        if max_dist > 0:
            # Correction strength falls off from boundary toward center
            correction_strength = 1.0 - np.clip(distance / (max_dist * 0.5), 0, 1)
        else:
            correction_strength = np.ones_like(distance)

        # Apply correction to Lab channels
        corrected_lab = gen_lab.copy()
        corrected_lab[:, :, 0] += delta_l * correction_strength * 0.7
        corrected_lab[:, :, 1] += delta_a * correction_strength * 0.5
        corrected_lab[:, :, 2] += delta_b * correction_strength * 0.5

        # Clip to valid Lab ranges
        corrected_lab[:, :, 0] = np.clip(corrected_lab[:, :, 0], 0, 255)
        corrected_lab[:, :, 1] = np.clip(corrected_lab[:, :, 1], 0, 255)
        corrected_lab[:, :, 2] = np.clip(corrected_lab[:, :, 2], 0, 255)

        # Convert back to RGB
        corrected_rgb = cv2.cvtColor(
            corrected_lab.astype(np.uint8), cv2.COLOR_LAB2RGB
        ).astype(np.float32)

        return corrected_rgb

    def blend_inpainting_with_guided_filter(
        self,
        original: Image.Image,
        generated: Image.Image,
        mask: Image.Image,
        feather_radius: int = 8,
        guide_radius: int = 8,
        guide_eps: float = 0.01
    ) -> Image.Image:
        """
        Blend inpainted region using guided filter for edge-aware transitions.

        Combines standard alpha blending with guided filtering to preserve
        edges in the original image while seamlessly integrating new content.

        Parameters
        ----------
        original : PIL.Image
            Original image
        generated : PIL.Image
            Generated/inpainted result
        mask : PIL.Image
            Inpainting mask
        feather_radius : int
            Base feathering radius
        guide_radius : int
            Guided filter radius
        guide_eps : float
            Guided filter regularization

        Returns
        -------
        PIL.Image
            Blended result with edge-aware transitions
        """
        logger.info("Applying guided filter inpainting blend")

        # Ensure same size
        if generated.size != original.size:
            generated = generated.resize(original.size, Image.LANCZOS)
        if mask.size != original.size:
            mask = mask.resize(original.size, Image.LANCZOS)

        # Convert to arrays
        orig_array = np.array(original.convert('RGB')).astype(np.float32)
        gen_array = np.array(generated.convert('RGB')).astype(np.float32)
        mask_array = np.array(mask.convert('L')).astype(np.float32) / 255.0

        # Apply base feathering
        if feather_radius > 0:
            kernel_size = feather_radius * 2 + 1
            mask_feathered = cv2.GaussianBlur(
                mask_array,
                (kernel_size, kernel_size),
                feather_radius / 2
            )
        else:
            mask_feathered = mask_array

        # Use original image as guide for the filter
        guide = cv2.cvtColor(orig_array.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        guide = guide.astype(np.float32) / 255.0

        # Apply guided filter to the mask
        try:
            mask_guided = cv2.ximgproc.guidedFilter(
                guide=guide,
                src=mask_feathered,
                radius=guide_radius,
                eps=guide_eps
            )
            logger.debug("Guided filter applied successfully")
        except Exception as e:
            logger.warning(f"Guided filter failed: {e}, using standard feathering")
            mask_guided = mask_feathered

        # Alpha blending
        alpha = mask_guided[:, :, np.newaxis]
        result = gen_array * alpha + orig_array * (1 - alpha)
        result = np.clip(result, 0, 255).astype(np.uint8)

        return Image.fromarray(result)
