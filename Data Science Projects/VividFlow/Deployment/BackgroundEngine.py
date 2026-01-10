import torch
import numpy as np
import cv2
from PIL import Image
import logging
import gc
import time
import os
from typing import Optional, Dict, Any, Callable
import warnings
warnings.filterwarnings("ignore")

from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
import open_clip
from mask_generator import MaskGenerator
from image_blender import ImageBlender

try:
    import spaces
    SPACES_AVAILABLE = True
except ImportError:
    SPACES_AVAILABLE = False

logger = logging.getLogger(__name__)


class BackgroundEngine:
    """
    Background generation engine for VividFlow.

    Integrates SDXL pipeline, OpenCLIP analysis, mask generation,
    and advanced image blending.
    """

    def __init__(self, device: str = "auto"):
        self.device = self._setup_device(device)
        self.base_model_id = "stabilityai/stable-diffusion-xl-base-1.0"
        self.clip_model_name = "ViT-B-32"
        self.clip_pretrained = "openai"

        self.pipeline = None
        self.clip_model = None
        self.clip_preprocess = None
        self.clip_tokenizer = None
        self.is_initialized = False

        self.max_image_size = 1024
        self.default_steps = 25
        self.use_fp16 = True

        self.mask_generator = MaskGenerator(self.max_image_size)
        self.image_blender = ImageBlender()

        logger.info(f"BackgroundEngine initialized on {self.device}")

    def _setup_device(self, device: str) -> str:
        """Setup computation device (ZeroGPU compatible)"""
        if os.getenv('SPACE_ID') is not None:
            return "cpu"

        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        return device

    def _memory_cleanup(self):
        """Memory cleanup"""
        for _ in range(3):
            gc.collect()

        is_spaces = os.getenv('SPACE_ID') is not None
        if not is_spaces and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def load_models(self, progress_callback: Optional[Callable] = None):
        """Load SDXL and OpenCLIP models"""
        if self.is_initialized:
            logger.info("Models already loaded")
            return

        logger.info("Loading background generation models...")

        try:
            self._memory_cleanup()

            # Detect actual device (in ZeroGPU, CUDA becomes available after @spaces.GPU allocation)
            actual_device = "cuda" if torch.cuda.is_available() else self.device
            logger.info(f"Loading models to device: {actual_device}")

            if progress_callback:
                progress_callback("Loading OpenCLIP...", 20)

            # Load OpenCLIP
            self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
                self.clip_model_name,
                pretrained=self.clip_pretrained,
                device=actual_device
            )
            self.clip_tokenizer = open_clip.get_tokenizer(self.clip_model_name)
            self.clip_model.eval()

            logger.info("OpenCLIP loaded")

            if progress_callback:
                progress_callback("Loading SDXL pipeline...", 60)

            # Load SDXL
            self.pipeline = StableDiffusionXLPipeline.from_pretrained(
                self.base_model_id,
                torch_dtype=torch.float16 if self.use_fp16 else torch.float32,
                use_safetensors=True,
                variant="fp16" if self.use_fp16 else None
            )

            # DPM solver for faster generation
            self.pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
                self.pipeline.scheduler.config
            )

            self.pipeline = self.pipeline.to(actual_device)

            if progress_callback:
                progress_callback("Applying optimizations...", 90)

            # Memory optimizations
            try:
                self.pipeline.enable_xformers_memory_efficient_attention()
                logger.info("xformers enabled")
            except Exception:
                try:
                    self.pipeline.enable_attention_slicing()
                    logger.info("Attention slicing enabled")
                except Exception:
                    pass

            if hasattr(self.pipeline, 'enable_vae_tiling'):
                self.pipeline.enable_vae_tiling()

            if hasattr(self.pipeline, 'enable_vae_slicing'):
                self.pipeline.enable_vae_slicing()

            self.pipeline.unet.eval()
            if hasattr(self.pipeline, 'vae'):
                self.pipeline.vae.eval()

            self.is_initialized = True

            if progress_callback:
                progress_callback("Models loaded!", 100)

            logger.info("Background models loaded successfully")

        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            raise RuntimeError(f"Failed to load models: {str(e)}")

    def analyze_image_with_clip(self, image: Image.Image) -> str:
        """Analyze image using OpenCLIP"""
        if not self.clip_model:
            return "Unknown"

        try:
            # Use actual device
            actual_device = "cuda" if torch.cuda.is_available() else self.device

            image_input = self.clip_preprocess(image).unsqueeze(0).to(actual_device)

            categories = [
                "a photo of a person",
                "a photo of an animal",
                "a photo of an object",
                "a photo of nature",
                "a photo of a building"
            ]

            text_inputs = self.clip_tokenizer(categories).to(actual_device)

            with torch.no_grad():
                image_features = self.clip_model.encode_image(image_input)
                text_features = self.clip_model.encode_text(text_inputs)

                image_features /= image_features.norm(dim=-1, keepdim=True)
                text_features /= text_features.norm(dim=-1, keepdim=True)

                similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                best_match_idx = similarity.argmax().item()

                category = categories[best_match_idx].replace("a photo of ", "")
                return category

        except Exception as e:
            logger.error(f"CLIP analysis failed: {e}")
            return "unknown"

    def enhance_prompt(self, user_prompt: str, foreground_image: Image.Image) -> str:
        """Smart prompt enhancement based on image analysis"""
        try:
            img_array = np.array(foreground_image.convert('RGB'))

            # Analyze color temperature
            lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
            avg_b = np.mean(lab[:, :, 2])
            is_warm = avg_b > 128

            # Analyze brightness
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            avg_brightness = np.mean(gray)
            is_bright = avg_brightness > 127

            # Get subject type
            clip_analysis = self.analyze_image_with_clip(foreground_image)
            subject_type = clip_analysis

            # Build lighting descriptors
            if is_warm and is_bright:
                lighting = "warm golden hour lighting, soft natural light"
            elif is_warm and not is_bright:
                lighting = "warm ambient lighting, cozy atmosphere"
            elif not is_warm and is_bright:
                lighting = "bright daylight, clear sky lighting"
            else:
                lighting = "soft diffused light, gentle shadows"

            # Build atmosphere based on subject
            atmosphere_map = {
                "person": "professional, elegant composition",
                "animal": "natural, harmonious setting",
                "object": "clean product photography style",
                "nature": "scenic, peaceful atmosphere",
                "building": "architectural, balanced composition"
            }
            atmosphere = atmosphere_map.get(subject_type, "balanced composition")

            quality_modifiers = "high quality, detailed, sharp focus, photorealistic"

            # Avoid conflicts
            user_prompt_lower = user_prompt.lower()
            if "sunset" in user_prompt_lower or "golden" in user_prompt_lower:
                lighting = ""
            if "dark" in user_prompt_lower or "night" in user_prompt_lower:
                lighting = lighting.replace("bright", "").replace("daylight", "")

            # Combine
            fragments = [user_prompt]
            if lighting:
                fragments.append(lighting)
            fragments.append(atmosphere)
            fragments.append(quality_modifiers)

            enhanced_prompt = ", ".join(filter(None, fragments))

            logger.debug(f"Enhanced: {enhanced_prompt[:80]}...")
            return enhanced_prompt

        except Exception as e:
            logger.warning(f"Prompt enhancement failed: {e}")
            return f"{user_prompt}, high quality, detailed, photorealistic"

    def _prepare_image(self, image: Image.Image) -> Image.Image:
        """Prepare image for processing"""
        if image.mode != 'RGB':
            image = image.convert('RGB')

        width, height = image.size
        max_size = self.max_image_size

        if width > max_size or height > max_size:
            ratio = min(max_size/width, max_size/height)
            new_width = int(width * ratio)
            new_height = int(height * ratio)
            image = image.resize((new_width, new_height), Image.LANCZOS)

        width, height = image.size
        new_width = (width // 8) * 8
        new_height = (height // 8) * 8

        if new_width != width or new_height != height:
            image = image.resize((new_width, new_height), Image.LANCZOS)

        return image

    def generate_background(
        self,
        prompt: str,
        width: int,
        height: int,
        negative_prompt: str = "blurry, low quality, distorted",
        num_inference_steps: int = 25,
        guidance_scale: float = 7.5
    ) -> Image.Image:
        """Generate background using SDXL"""
        if not self.is_initialized:
            raise RuntimeError("Models not loaded")

        logger.info(f"Generating background: {prompt[:50]}...")

        try:
            # Use actual device
            actual_device = "cuda" if torch.cuda.is_available() else self.device

            with torch.inference_mode():
                result = self.pipeline(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    width=width,
                    height=height,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=torch.Generator(device=actual_device).manual_seed(42)
                )

                generated_image = result.images[0]
                logger.info("Background generation completed")
                return generated_image

        except torch.cuda.OutOfMemoryError:
            logger.error("GPU memory exhausted")
            self._memory_cleanup()
            raise RuntimeError("GPU memory insufficient")

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise RuntimeError(f"Generation failed: {str(e)}")

    def generate_and_combine(
        self,
        original_image: Image.Image,
        prompt: str,
        combination_mode: str = "center",
        focus_mode: str = "person",
        negative_prompt: str = "blurry, low quality, distorted",
        num_inference_steps: int = 25,
        guidance_scale: float = 7.5,
        progress_callback: Optional[Callable] = None,
        enable_prompt_enhancement: bool = True,
        feather_radius: int = 0
    ) -> Dict[str, Any]:
        """
        Generate background and combine with foreground.

        Args:
            feather_radius: Gaussian blur radius for mask edge softening (0-20, default 0)

        Returns dict with: combined_image, generated_scene, original_image, mask, success
        """
        if not self.is_initialized:
            raise RuntimeError("Models not loaded")

        logger.info("Starting background generation and combination...")

        try:
            if progress_callback:
                progress_callback("Analyzing image...", 5)

            # Prepare image
            processed_original = self._prepare_image(original_image)
            target_width, target_height = processed_original.size

            if progress_callback:
                progress_callback("Enhancing prompt...", 15)

            # Enhance prompt
            if enable_prompt_enhancement:
                enhanced_prompt = self.enhance_prompt(prompt, processed_original)
            else:
                enhanced_prompt = f"{prompt}, high quality, detailed, photorealistic"

            enhanced_negative = f"{negative_prompt}, people, characters, cartoons, logos"

            if progress_callback:
                progress_callback("Generating background...", 30)

            # Generate background
            generated_background = self.generate_background(
                prompt=enhanced_prompt,
                width=target_width,
                height=target_height,
                negative_prompt=enhanced_negative,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale
            )

            if progress_callback:
                progress_callback("Creating mask...", 80)

            # Generate mask
            logger.info("Generating mask...")
            combination_mask = self.mask_generator.create_gradient_based_mask(
                processed_original,
                combination_mode,
                focus_mode
            )

            if progress_callback:
                progress_callback("Blending images...", 90)

            # Blend images with feather_radius
            logger.info("Blending images...")
            combined_image = self.image_blender.simple_blend_images(
                processed_original,
                generated_background,
                combination_mask,
                feather_radius=feather_radius
            )

            # Cleanup
            self._memory_cleanup()

            if progress_callback:
                progress_callback("Complete!", 100)

            logger.info("Background generation completed successfully")

            # Build result dict (always include mask for diagnostics)
            return {
                "combined_image": combined_image,
                "generated_scene": generated_background,
                "original_image": processed_original,
                "mask": combination_mask,
                "success": True
            }

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            self._memory_cleanup()
            return {
                "success": False,
                "error": str(e)
            }
