import cv2
import numpy as np
import traceback
from PIL import Image, ImageFilter, ImageDraw
import logging
from typing import Optional, Tuple
from scipy.ndimage import binary_erosion, binary_dilation
import io
import gc
import torch
from transformers import AutoModelForImageSegmentation
from torchvision import transforms
from rembg import remove, new_session

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Dark background detection thresholds
DARK_BG_LUMINANCE_THRESHOLD = 50  # Average luminance below this = dark background
DARK_BG_EDGE_SAMPLE_WIDTH = 20    # Pixels from edge to sample for background detection
DARK_BG_DILATION_PIXELS = 5       # Default dilation for dark backgrounds
DARK_BG_ENHANCED_DILATION = 8     # Enhanced dilation when user enables option


class MaskGenerator:
    """
    Intelligent mask generation using deep learning models with traditional fallback.
    Priority: BiRefNet > U²-Net (rembg) > Traditional gradient-based methods
    """

    def __init__(self, max_image_size: int = 1024, device: str = "auto"):
        self.max_image_size = max_image_size
        self.device = self._setup_device(device)

        # BiRefNet model (lazy loading)
        self._birefnet_model = None
        self._birefnet_transform = None

        # Log initialization
        logger.info(f"🎭 MaskGenerator initialized on {self.device}")

    def _setup_device(self, device: str) -> str:
        """Setup computation device"""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        return device

    def _load_birefnet_model(self) -> bool:
        """
        Lazy load BiRefNet model for memory efficiency.
        Returns True if model loaded successfully, False otherwise.
        """
        if self._birefnet_model is not None:
            return True

        try:
            logger.info("📥 Loading BiRefNet model (ZhengPeng7/BiRefNet)...")

            # Load model with fp16 for memory efficiency on GPU
            dtype = torch.float16 if self.device == "cuda" else torch.float32

            self._birefnet_model = AutoModelForImageSegmentation.from_pretrained(
                "ZhengPeng7/BiRefNet",
                trust_remote_code=True,
                torch_dtype=dtype
            )
            self._birefnet_model.to(self.device)
            self._birefnet_model.eval()

            # Define preprocessing transform
            self._birefnet_transform = transforms.Compose([
                transforms.Resize((1024, 1024)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

            logger.info("✅ BiRefNet model loaded successfully")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to load BiRefNet: {e}")
            self._birefnet_model = None
            self._birefnet_transform = None
            return False

    def _unload_birefnet_model(self):
        """Unload BiRefNet model to free memory"""
        if self._birefnet_model is not None:
            del self._birefnet_model
            self._birefnet_model = None
            self._birefnet_transform = None

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            logger.info("🧹 BiRefNet model unloaded")

    def detect_dark_background(self, image: Image.Image, mask: Optional[np.ndarray] = None) -> Tuple[bool, float]:
        """
        Detect if the image has a dark background.

        Analyzes the edge regions of the image (where background is likely) to determine
        if the background is predominantly dark, which can cause mask detection issues.

        Args:
            image: Input PIL Image
            mask: Optional existing mask to exclude foreground from analysis

        Returns:
            Tuple of (is_dark_background: bool, avg_luminance: float)
        """
        try:
            img_array = np.array(image.convert('RGB'))
            height, width = img_array.shape[:2]

            # Convert to grayscale for luminance analysis
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

            # Sample from edge regions (likely background)
            edge_width = min(DARK_BG_EDGE_SAMPLE_WIDTH, width // 10, height // 10)

            # Create edge sampling mask
            edge_sample_mask = np.zeros((height, width), dtype=bool)
            edge_sample_mask[:edge_width, :] = True  # Top
            edge_sample_mask[-edge_width:, :] = True  # Bottom
            edge_sample_mask[:, :edge_width] = True  # Left
            edge_sample_mask[:, -edge_width:] = True  # Right

            # Exclude foreground if mask is provided
            if mask is not None:
                foreground_mask = mask > 127
                edge_sample_mask = edge_sample_mask & (~foreground_mask)

            if not np.any(edge_sample_mask):
                # Fallback: use corners only
                corner_pixels = np.array([
                    gray[0, 0], gray[0, -1],
                    gray[-1, 0], gray[-1, -1]
                ])
                avg_luminance = np.mean(corner_pixels)
            else:
                avg_luminance = np.mean(gray[edge_sample_mask])

            is_dark = avg_luminance < DARK_BG_LUMINANCE_THRESHOLD

            logger.info(f"🔍 Background analysis - Avg luminance: {avg_luminance:.1f}, Dark: {is_dark}")

            return is_dark, avg_luminance

        except Exception as e:
            logger.error(f"❌ Dark background detection failed: {e}")
            return False, 128.0  # Default: not dark

    def enhance_mask_for_dark_background(
        self,
        mask: Image.Image,
        original_image: Image.Image,
        dilation_pixels: int = DARK_BG_DILATION_PIXELS,
        enhance_gray_areas: bool = True
    ) -> Image.Image:
        """
        Enhance mask for images with dark backgrounds.

        Applies dilation and gray area enhancement to capture foreground elements
        that may have been missed due to low contrast with dark backgrounds.

        Args:
            mask: Input mask PIL Image (L mode)
            original_image: Original image for reference
            dilation_pixels: Number of pixels to dilate the mask
            enhance_gray_areas: Whether to boost gray (uncertain) areas

        Returns:
            Enhanced mask PIL Image
        """
        try:
            mask_array = np.array(mask)
            orig_array = np.array(original_image.convert('RGB'))

            logger.info(f"🔧 Enhancing mask for dark background (dilation: {dilation_pixels}px)")

            # Step 1: Identify gray (uncertain) areas in the mask
            if enhance_gray_areas:
                gray_areas = (mask_array > 30) & (mask_array < 200)

                if np.any(gray_areas):
                    # For gray areas, check if they're near high-confidence foreground
                    high_conf = mask_array >= 200

                    # Dilate high confidence area to find nearby gray pixels
                    kernel_check = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
                    high_conf_dilated = cv2.dilate(high_conf.astype(np.uint8), kernel_check, iterations=2)

                    # Gray pixels near high confidence foreground -> boost them
                    boost_candidates = gray_areas & (high_conf_dilated > 0)

                    # Boost gray areas near foreground
                    mask_array[boost_candidates] = np.clip(
                        mask_array[boost_candidates] * 1.5 + 50,
                        0, 255
                    ).astype(np.uint8)

                    logger.info(f"📈 Boosted {np.sum(boost_candidates)} gray pixels near foreground")

            # Step 2: Apply dilation to expand foreground coverage
            if dilation_pixels > 0:
                kernel = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE,
                    (dilation_pixels * 2 + 1, dilation_pixels * 2 + 1)
                )

                # Threshold to get foreground region for dilation
                fg_binary = (mask_array > 50).astype(np.uint8) * 255
                fg_dilated = cv2.dilate(fg_binary, kernel, iterations=1)

                # Blend: keep original high values, expand into new areas
                # New areas from dilation get moderate confidence
                new_areas = (fg_dilated > 0) & (mask_array < 50)
                mask_array[new_areas] = 180  # Moderate confidence for expanded areas

                logger.info(f"📐 Dilated mask by {dilation_pixels}px, added {np.sum(new_areas)} pixels")

            # Step 3: Smooth the transitions
            mask_array = cv2.GaussianBlur(mask_array, (3, 3), 0.8)

            # Step 4: Re-strengthen core foreground
            core_fg = np.array(mask) >= 220
            mask_array[core_fg] = 255

            logger.info(f"✅ Dark background enhancement complete - Final mean: {mask_array.mean():.1f}")

            return Image.fromarray(mask_array, mode='L')

        except Exception as e:
            logger.error(f"❌ Mask enhancement failed: {e}")
            return mask

    def apply_guided_filter(
        self,
        mask: np.ndarray,
        guide_image: Image.Image,
        radius: int = 8,
        eps: float = 0.01
    ) -> np.ndarray:
        """
        Apply guided filter to mask for edge-preserving smoothing.
        Falls back to Gaussian blur if guided filter is not available.

        Args:
            mask: Input mask as numpy array (0-255)
            guide_image: Original image to use as guide
            radius: Filter radius (larger = more smoothing)
            eps: Regularization parameter (smaller = more edge-preserving)

        Returns:
            Filtered mask as numpy array (0-255)
        """
        try:
            # Convert guide image to grayscale
            guide_gray = np.array(guide_image.convert('L')).astype(np.float32) / 255.0
            mask_float = mask.astype(np.float32) / 255.0

            logger.info(f"🔧 Applying guided filter (radius={radius}, eps={eps})")

            # Apply guided filter
            filtered = cv2.ximgproc.guidedFilter(
                guide=guide_gray,
                src=mask_float,
                radius=radius,
                eps=eps
            )

            # Convert back to 0-255 range
            result = (np.clip(filtered, 0, 1) * 255).astype(np.uint8)
            logger.info("✅ Guided filter applied successfully")
            return result

        except Exception as e:
            logger.error(f"❌ Guided filter failed: {e}, using original mask")
            return mask

    def try_birefnet_mask(self, original_image: Image.Image) -> Optional[Image.Image]:
        """
        Generate foreground mask using BiRefNet model.
        BiRefNet provides high-quality segmentation with clean edges.

        Args:
            original_image: Input PIL Image

        Returns:
            PIL Image (L mode) mask or None if failed
        """
        try:
            # Lazy load model
            if not self._load_birefnet_model():
                return None

            logger.info("🤖 Starting BiRefNet foreground extraction...")
            original_size = original_image.size

            # Convert to RGB if needed
            if original_image.mode != 'RGB':
                image_rgb = original_image.convert('RGB')
            else:
                image_rgb = original_image

            # Preprocess image
            input_tensor = self._birefnet_transform(image_rgb).unsqueeze(0)

            # Move to device with appropriate dtype
            if self.device == "cuda":
                input_tensor = input_tensor.to(self.device, dtype=torch.float16)
            else:
                input_tensor = input_tensor.to(self.device)

            # Run inference
            with torch.no_grad():
                outputs = self._birefnet_model(input_tensor)

                # BiRefNet outputs a list, get the final prediction
                if isinstance(outputs, (list, tuple)):
                    pred = outputs[-1]
                else:
                    pred = outputs

                # Sigmoid to get probability map
                pred = torch.sigmoid(pred)

                # Convert to numpy
                pred_np = pred.squeeze().cpu().numpy()

            # Convert to 0-255 range
            mask_array = (pred_np * 255).astype(np.uint8)

            # Resize back to original size
            mask_pil = Image.fromarray(mask_array, mode='L')
            mask_pil = mask_pil.resize(original_size, Image.LANCZOS)
            mask_array = np.array(mask_pil)

            # Quality check
            mean_val = mask_array.mean()
            nonzero_ratio = np.count_nonzero(mask_array > 50) / mask_array.size

            logger.info(f"📊 BiRefNet mask stats - Mean: {mean_val:.1f}, Coverage: {nonzero_ratio:.1%}")

            if mean_val < 10:
                logger.warning("⚠️ BiRefNet mask too weak, falling back")
                return None

            if nonzero_ratio < 0.03:
                logger.warning("⚠️ BiRefNet foreground coverage too low, falling back")
                return None

            # Light post-processing for edge refinement
            # Use morphological operations to clean up
            kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask_array = cv2.morphologyEx(mask_array, cv2.MORPH_CLOSE, kernel_small)

            logger.info("✅ BiRefNet mask generation successful!")
            return Image.fromarray(mask_array, mode='L')

        except torch.cuda.OutOfMemoryError:
            logger.error("❌ BiRefNet: GPU memory exhausted")
            self._unload_birefnet_model()
            return None

        except Exception as e:
            logger.error(f"❌ BiRefNet mask generation failed: {e}")
            logger.error(f"📍 Traceback: {traceback.format_exc()}")
            return None

    def try_deep_learning_mask(self, original_image: Image.Image) -> Optional[Image.Image]:
        """
        Intelligent foreground extraction with model priority:
        1. BiRefNet (best quality, clean edges)
        2. U²-Net via rembg (good fallback)
        3. Return None to trigger traditional methods

        Args:
            original_image: Input PIL Image

        Returns:
            PIL Image (L mode) mask or None if all methods failed
        """
        # Priority 1: Try BiRefNet first
        logger.info("🤖 Attempting BiRefNet mask generation...")
        birefnet_mask = self.try_birefnet_mask(original_image)
        if birefnet_mask is not None:
            logger.info("✅ Using BiRefNet generated mask")
            return birefnet_mask

        # Priority 2: Fallback to rembg (U²-Net)
        logger.info("🔄 BiRefNet unavailable/failed, trying rembg...")
        try:
            logger.info("🤖 Starting rembg foreground extraction")

            # Try u2net first (better for cartoons/objects like Snoopy)
            try:
                session = new_session('u2net')
                logger.info("✅ Using u2net model")
            except Exception as e:
                logger.warning(f"u2net failed ({e}), trying u2net_human_seg")
                try:
                    session = new_session('u2net_human_seg')
                    logger.info("✅ Using u2net_human_seg model")
                except Exception as e2:
                    logger.error(f"All rembg models failed: {e2}")
                    return None

            # Convert image to bytes for rembg
            img_byte_arr = io.BytesIO()
            original_image.save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()
            logger.info(f"📷 Image size: {len(img_byte_arr)} bytes")

            # Perform background removal
            result = remove(img_byte_arr, session=session)
            result_img = Image.open(io.BytesIO(result)).convert('RGBA')
            alpha_channel = result_img.split()[-1]
            alpha_array = np.array(alpha_channel)

            logger.info(f"📊 Raw alpha stats - Mean: {alpha_array.mean():.1f}, Min: {alpha_array.min()}, Max: {alpha_array.max()}")

            # Step 1: Light smoothing to reduce noise but preserve edges
            alpha_smoothed = cv2.GaussianBlur(alpha_array, (3, 3), 0.8)

            # Step 2: Contrast stretching to utilize full range
            alpha_stretched = cv2.normalize(alpha_smoothed, None, 0, 255, cv2.NORM_MINMAX)

            # Step 3: CRITICAL FIX - More aggressive foreground preservation
            # Instead of hard threshold, use adaptive approach

            # Find the main subject area (high confidence regions)
            high_confidence = alpha_stretched > 180
            medium_confidence = (alpha_stretched > 60) & (alpha_stretched <= 180)
            low_confidence = (alpha_stretched > 15) & (alpha_stretched <= 60)

            # Create final mask with better extremity handling
            final_alpha = np.zeros_like(alpha_stretched)

            # High confidence areas - keep at full opacity
            final_alpha[high_confidence] = 255

            # Medium confidence - boost significantly
            final_alpha[medium_confidence] = np.clip(alpha_stretched[medium_confidence] * 1.8, 200, 255)

            # Low confidence - moderate boost (catches faint extremities)
            final_alpha[low_confidence] = np.clip(alpha_stretched[low_confidence] * 2.5, 120, 199)

            # Morphological operations to connect disconnected parts (hands, feet, tail)
            kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

            # Close small gaps (helps connect separated body parts)
            final_alpha = cv2.morphologyEx(final_alpha, cv2.MORPH_CLOSE, kernel_small, iterations=1)

            # Light dilation to ensure nothing gets cut off
            final_alpha = cv2.dilate(final_alpha, kernel_small, iterations=1)

            logger.info(f"📊 Final alpha stats - Mean: {final_alpha.mean():.1f}, Min: {final_alpha.min()}, Max: {final_alpha.max()}")

            # Quality check - but be more lenient for cartoon characters
            if final_alpha.mean() < 10:
                logger.warning("⚠️ Alpha still too weak, falling back to traditional method")
                return None

            # Enhanced post-processing for cartoon characters
            is_cartoon = self._detect_cartoon_character(original_image, final_alpha)

            if is_cartoon:
                logger.info("🎭 Detected cartoon/character image, applying specialized processing")
                final_alpha = self._enhance_cartoon_mask(original_image, final_alpha)

            # Count non-zero pixels to ensure we have substantial foreground
            foreground_pixels = np.count_nonzero(final_alpha > 50)
            total_pixels = final_alpha.size
            foreground_ratio = foreground_pixels / total_pixels
            logger.info(f"📊 Foreground coverage: {foreground_ratio:.1%} of image")

            if foreground_ratio < 0.05:  # Less than 5% is probably too little
                logger.warning("⚠️ Very low foreground coverage, falling back to traditional method")
                return None

            mask = Image.fromarray(final_alpha.astype(np.uint8), mode='L')
            logger.info("✅ Enhanced rembg mask generation successful!")
            return mask

        except Exception as e:
            logger.error(f"❌ Deep learning mask extraction failed: {e}")
            return None

    def _detect_cartoon_character(self, original_image: Image.Image, alpha_mask: np.ndarray) -> bool:
        """
        Detect if image is cartoon/line art (heuristic approach)
        """
        try:
            img_array = np.array(original_image.convert('RGB'))
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

            # Calculate edge density (cartoons usually have more clear edges)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.count_nonzero(edges) / max(edges.size, 1)  # Avoid division by zero

            # Calculate color complexity (cartoons usually have fewer colors) - optimize memory usage
            h, w, c = img_array.shape
            if h * w > 100000:  # If image is too large, resize for processing
                small_img = cv2.resize(img_array, (200, 200))
            else:
                small_img = img_array

            unique_colors = len(np.unique(small_img.reshape(-1, 3), axis=0))
            total_pixels = small_img.shape[0] * small_img.shape[1]
            color_simplicity = unique_colors < (total_pixels * 0.1)

            # Check for obvious black outlines
            dark_pixels_ratio = np.count_nonzero(gray < 50) / max(gray.size, 1)  # Avoid division by zero
            has_black_outline = dark_pixels_ratio > 0.05

            # Comprehensive judgment: high edge density + color simplicity + black outline = likely cartoon
            is_cartoon = (edge_density > 0.05) and (color_simplicity or has_black_outline)

            logger.info(f"🔍 Cartoon detection - Edge density: {edge_density:.3f}, Color simplicity: {color_simplicity}, Black outline: {has_black_outline} -> Cartoon: {is_cartoon}")
            return is_cartoon

        except Exception as e:
            logger.error(f"❌ Cartoon detection failed: {e}")
            logger.error(f"📍 Traceback: {traceback.format_exc()}")
            print(f"❌ CARTOON DETECTION ERROR: {e}")
            print(f"Traceback: {traceback.format_exc()}")
            return False

    def _enhance_cartoon_mask(self, original_image: Image.Image, alpha_mask: np.ndarray) -> np.ndarray:
        """
        Enhanced mask processing for cartoon characters
        """
        try:
            img_array = np.array(original_image.convert('RGB'))
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            enhanced_alpha = alpha_mask.copy()

            # Step 1: Black outline enhancement - find black outlines and enhance their alpha
            th_dark = 80  # Adjustable parameter: black threshold
            black_outline = gray < th_dark

            # Dilate black outline region by 1px
            kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))  # Adjustable parameter: dilation kernel size
            black_outline_dilated = cv2.dilate(black_outline.astype(np.uint8), kernel_dilate, iterations=1)

            # Set black outline region alpha directly to 255
            enhanced_alpha[black_outline_dilated > 0] = 255
            logger.info(f"🖤 Black outline enhanced: {np.count_nonzero(black_outline_dilated)} pixels")

            # Step 2: Simplified internal enhancement - process white fill areas within outlines
            # Find high confidence regions (alpha ≥ 160)
            high_confidence = enhanced_alpha >= 160

            # Apply close operation on high confidence regions to connect separated parts
            kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # Adjustable parameter: close kernel size
            high_confidence_closed = cv2.morphologyEx(high_confidence.astype(np.uint8), cv2.MORPH_CLOSE, kernel_close, iterations=1)

            # Simplified approach: directly enhance medium confidence regions without complex flood fill
            # Find medium/low confidence regions surrounded by high confidence regions
            medium_confidence = (enhanced_alpha >= 80) & (enhanced_alpha < 160)

            # Dilate high confidence region to include more internal areas
            kernel_dilate_internal = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            high_confidence_expanded = cv2.dilate(high_confidence_closed, kernel_dilate_internal, iterations=1)

            # Medium confidence pixels within expanded high confidence areas are considered internal fill
            internal_fill_regions = medium_confidence & (high_confidence_expanded > 0)

            # Enhance alpha of these internal fill regions to at least 220
            min_alpha_for_fill = 220  # Adjustable parameter: minimum alpha for internal fill
            enhanced_alpha[internal_fill_regions] = np.maximum(enhanced_alpha[internal_fill_regions], min_alpha_for_fill)

            logger.info(f"🤍 Internal fill regions enhanced: {np.count_nonzero(internal_fill_regions)} pixels")
            logger.info(f"📊 Enhanced alpha stats - Mean: {enhanced_alpha.mean():.1f}, Min: {enhanced_alpha.min()}, Max: {enhanced_alpha.max()}")

            return enhanced_alpha

        except Exception as e:
            logger.error(f"❌ Cartoon mask enhancement failed: {e}")
            logger.error(f"📍 Traceback: {traceback.format_exc()}")
            print(f"❌ CARTOON MASK ENHANCEMENT ERROR: {e}")
            print(f"Traceback: {traceback.format_exc()}")
            return alpha_mask

    def _adjust_mask_for_scene_focus(self, mask: Image.Image, original_image: Image.Image) -> Image.Image:
        """
        Adjust mask for scene focus mode to include nearby objects like chairs, furniture
        """
        try:
            logger.info("🏠 Adjusting mask for scene focus mode...")

            mask_array = np.array(mask)
            img_array = np.array(original_image.convert('RGB'))

            # Expand mask to include nearby objects
            # Use larger dilation kernel to include furniture/objects
            kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            expanded_mask = cv2.dilate(mask_array, kernel_large, iterations=2)

            # Find contours in the expanded area to detect objects
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 30, 100)

            # Apply edge detection only in the expanded region
            expanded_region = (expanded_mask > 0) & (mask_array == 0)
            object_edges = np.zeros_like(edges)
            object_edges[expanded_region] = edges[expanded_region]

            # Close gaps to form complete objects
            kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            object_mask = cv2.morphologyEx(object_edges, cv2.MORPH_CLOSE, kernel_close)
            object_mask = cv2.dilate(object_mask, kernel_close, iterations=1)

            # Combine with original mask
            final_mask = np.maximum(mask_array, object_mask)

            logger.info("✅ Scene focus adjustment completed")
            return Image.fromarray(final_mask)

        except Exception as e:
            logger.error(f"❌ Scene focus adjustment failed: {e}")
            return mask

    def create_gradient_based_mask(
        self,
        original_image: Image.Image,
        mode: str = "center",
        focus_mode: str = "person",
        enhance_dark_edges: bool = False
    ) -> Image.Image:
        """
        Intelligent foreground extraction: prioritize deep learning models, fallback to traditional methods
        Focus mode: 'person' for tight crop around person, 'scene' for including nearby objects

        Args:
            original_image: Input PIL Image
            mode: Composition mode (center, left_half, right_half, full)
            focus_mode: 'person' for tight crop, 'scene' for including nearby objects
            enhance_dark_edges: User toggle to enhance mask for dark backgrounds
        """
        width, height = original_image.size
        logger.info(f"🎯 Creating mask for {width}x{height} image, mode: {mode}, focus: {focus_mode}, enhance_dark: {enhance_dark_edges}")

        if mode == "center":
            # Try using deep learning models for intelligent foreground extraction
            logger.info("🤖 Attempting deep learning mask generation...")
            dl_mask = self.try_deep_learning_mask(original_image)
            if dl_mask is not None:
                logger.info("✅ Using deep learning generated mask")

                # Apply focus mode adjustments to deep learning mask
                if focus_mode == "scene":
                    dl_mask = self._adjust_mask_for_scene_focus(dl_mask, original_image)

                # === Dark background detection and enhancement ===
                mask_array = np.array(dl_mask)
                is_dark_bg, avg_luminance = self.detect_dark_background(original_image, mask_array)

                if is_dark_bg or enhance_dark_edges:
                    # Determine dilation amount
                    if enhance_dark_edges:
                        # User explicitly enabled - use stronger dilation
                        dilation = DARK_BG_ENHANCED_DILATION
                        logger.info(f"🌙 User enabled dark edge enhancement (dilation: {dilation}px)")
                    else:
                        # Auto-detected dark background - use moderate dilation
                        dilation = DARK_BG_DILATION_PIXELS
                        logger.info(f"🌙 Auto-detected dark background (luminance: {avg_luminance:.1f}), applying enhancement")

                    dl_mask = self.enhance_mask_for_dark_background(
                        dl_mask,
                        original_image,
                        dilation_pixels=dilation,
                        enhance_gray_areas=True
                    )

                return dl_mask

            # Fallback to traditional method
            logger.info("🔄 Deep learning failed, using traditional gradient-based method")
            img_array = np.array(original_image.convert('RGB'))
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

            # First-order derivatives: use Sobel operator for edge detection
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)

            # Second-order derivatives: use Laplacian operator for texture change detection
            laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
            laplacian_abs = np.abs(laplacian)

            # Combine first and second order derivatives
            combined_edges = gradient_magnitude * 0.7 + laplacian_abs * 0.3
            combined_edges = (combined_edges / np.max(combined_edges)) * 255

            # Threshold processing to find strong edges
            _, edge_binary = cv2.threshold(combined_edges.astype(np.uint8), 20, 255, cv2.THRESH_BINARY)

            # Morphological operations to connect edges
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            edge_binary = cv2.morphologyEx(edge_binary, cv2.MORPH_CLOSE, kernel)

            # Find contours and create mask
            contours, _ = cv2.findContours(edge_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                # Find largest contour (main subject)
                largest_contour = max(contours, key=cv2.contourArea)
                contour_mask = np.zeros((height, width), dtype=np.uint8)
                cv2.fillPoly(contour_mask, [largest_contour], 255)

                # Create foreground enhancement mask: specially protect dark regions
                dark_mask = (gray < 90).astype(np.uint8) * 255
                morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
                dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, morph_kernel, iterations=1)
                dark_mask = cv2.dilate(dark_mask, morph_kernel, iterations=2)
                contour_mask = cv2.bitwise_or(contour_mask, dark_mask)

                # Get core foreground: clean holes and fill gaps
                close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
                core_mask = cv2.morphologyEx(contour_mask, cv2.MORPH_CLOSE, close_kernel, iterations=1)

                open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                core_mask = cv2.morphologyEx(core_mask, cv2.MORPH_OPEN, open_kernel, iterations=1)

                # Convert to binary core (0/255)
                _, core_binary = cv2.threshold(core_mask, 127, 255, cv2.THRESH_BINARY)

                # Keep only slight dilation to avoid foreground being eaten
                dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                core_binary = cv2.dilate(core_binary, dilate_kernel, iterations=1)

                # Distance transform feathering: shrink feathering range for sharp edges
                FEATHER_PX = 4

                # Calculate distance transform
                core_float = core_binary.astype(np.float32) / 255.0
                distances = cv2.distanceTransform((1 - core_float).astype(np.uint8), cv2.DIST_L2, 5)

                # Create feathering mask: 0→FEATHER_PX linear mapping to 1→0
                feather_mask = np.ones_like(distances)
                edge_region = (distances > 0) & (distances <= FEATHER_PX)
                feather_mask[edge_region] = 1.0 - (distances[edge_region] / FEATHER_PX)
                feather_mask[distances > FEATHER_PX] = 0.0

                # Apply double-smoothstep curve: make transition steeper, reduce semi-transparent halos
                def double_smoothstep(t):
                    t = np.clip(t, 0, 1)
                    s1 = t * t * (3 - 2 * t)
                    return s1 * s1 * (3 - 2 * s1)  # Equivalent to t^3 (10 - 15t + 6t^2)

                # Combine core with feathering: core area keeps 255, edges use double_smoothstep feathering
                final_alpha = np.zeros_like(distances)
                final_alpha[core_binary > 127] = 1.0  # Core area
                final_alpha[edge_region] = double_smoothstep(feather_mask[edge_region])  # Feathering area

                # Convert to 0-255 range
                final_mask = (final_alpha * 255).astype(np.uint8)

                # Apply guided filter for edge-preserving smoothing
                final_mask = self.apply_guided_filter(final_mask, original_image, radius=8, eps=0.01)

                mask = Image.fromarray(final_mask)
            else:
                # Backup plan: use large ellipse
                mask = Image.new('L', (width, height), 0)
                draw = ImageDraw.Draw(mask)
                center_x, center_y = width // 2, height // 2
                width_radius = int(width * 0.45)
                height_radius = int(width * 0.48)
                draw.ellipse([
                    center_x - width_radius, center_y - height_radius,
                    center_x + width_radius, center_y + height_radius
                ], fill=255)
                # Apply guided filter instead of Gaussian blur
                mask_array = np.array(mask)
                mask_array = self.apply_guided_filter(mask_array, original_image, radius=10, eps=0.02)
                mask = Image.fromarray(mask_array)

        elif mode == "left_half":
            # Keep original logic unchanged - ensure Snoopy and other functions work normally
            mask = Image.new('L', (width, height), 0)
            mask_array = np.array(mask)
            mask_array[:, :width//2] = 255

            transition_zone = width // 10
            for i in range(transition_zone):
                x_pos = width//2 + i
                if x_pos < width:
                    alpha = 255 * (1 - i / transition_zone)
                    mask_array[:, x_pos] = int(alpha)

            mask = Image.fromarray(mask_array)

        elif mode == "right_half":
            # Keep original logic unchanged - ensure Snoopy and other functions work normally
            mask = Image.new('L', (width, height), 0)
            mask_array = np.array(mask)
            mask_array[:, width//2:] = 255

            transition_zone = width // 10
            for i in range(transition_zone):
                x_pos = width//2 - i - 1
                if x_pos >= 0:
                    alpha = 255 * (1 - i / transition_zone)
                    mask_array[:, x_pos] = int(alpha)

            mask = Image.fromarray(mask_array)

        elif mode == "full":
            mask = Image.new('L', (width, height), 0)
            draw = ImageDraw.Draw(mask)
            center_x, center_y = width // 2, height // 2
            radius = min(width, height) // 8

            draw.ellipse([
                center_x - radius, center_y - radius,
                center_x + radius, center_y + radius
            ], fill=255)

            mask = mask.filter(ImageFilter.GaussianBlur(radius=5))

        return mask
