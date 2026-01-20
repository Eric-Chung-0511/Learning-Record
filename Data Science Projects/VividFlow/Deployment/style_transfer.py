import os
import gc
from typing import Tuple, Optional, Dict, Any

from PIL import Image
import torch

try:
    import spaces
    HAS_SPACES = True
except ImportError:
    HAS_SPACES = False


# Identity preservation keywords (added to all styles) - kept short for CLIP 77 token limit
IDENTITY_PRESERVE = "same person, same face, same ethnicity, same age"
IDENTITY_NEGATIVE = "different person, altered face, changed ethnicity, age change, distorted features"

# Enhanced face restore mode - concise weighted keywords
FACE_RESTORE_PRESERVE = "(same person:1.4), (preserve face:1.3), (same ethnicity:1.2), same pose, same lighting"
FACE_RESTORE_NEGATIVE = "(different person:1.4), (deformed face:1.3), wrong ethnicity, age change, western features"

# IP-Adapter settings for stronger identity preservation
# Using standard IP-Adapter (not face-specific) to avoid image encoder dependency
IP_ADAPTER_REPO = "h94/IP-Adapter"
IP_ADAPTER_SUBFOLDER = "sdxl_models"
IP_ADAPTER_WEIGHT = "ip-adapter_sdxl.bin"  # Standard model, no extra encoder needed
IP_ADAPTER_SCALE_DEFAULT = 0.5  # Balance between identity and style

# Style-specific face_restore settings (some styles are more transformative)
FACE_RESTORE_STYLE_SETTINGS = {
    "3d_cartoon": {"max_strength": 0.45, "lora_scale_mult": 0.7, "ip_scale": 0.4},
    "anime": {"max_strength": 0.45, "lora_scale_mult": 0.7, "ip_scale": 0.4},
    "illustrated_fantasy": {"max_strength": 0.42, "lora_scale_mult": 0.65, "ip_scale": 0.45},
    "watercolor": {"max_strength": 0.40, "lora_scale_mult": 0.6, "ip_scale": 0.5},
    "oil_painting": {"max_strength": 0.35, "lora_scale_mult": 0.5, "ip_scale": 0.6},  # Most transformative
    "pixel_art": {"max_strength": 0.50, "lora_scale_mult": 0.8, "ip_scale": 0.3},
}

# Style configurations
STYLE_CONFIGS = {
    "3d_cartoon": {
        "name": "3D Cartoon",
        "emoji": "🎬",
        "lora_repo": "imagepipeline/Samaritan-3d-Cartoon-SDXL",
        "lora_weight": "Samaritan 3d Cartoon.safetensors",
        "prompt": "3D cartoon style, smooth rounded features, soft ambient lighting, CGI quality, vibrant colors, cel-shaded, studio render",
        "negative_prompt": "ugly, deformed, noisy, blurry, low quality, flat, sketch",
        "lora_scale": 0.75,
        "recommended_strength": 0.55,
    },
    "anime": {
        "name": "Anime Illustration",
        "emoji": "🌸",
        "lora_repo": None,
        "lora_weight": None,
        "prompt": "anime illustration, soft lighting, rich colors, delicate linework, smooth gradients, expressive eyes, cel shading, masterpiece",
        "negative_prompt": "ugly, deformed, bad anatomy, bad hands, blurry, low quality",
        "lora_scale": 0.0,
        "recommended_strength": 0.50,
    },
    "illustrated_fantasy": {
        "name": "Illustrated Fantasy",
        "emoji": "🍃",
        "lora_repo": "ntc-ai/SDXL-LoRA-slider.Studio-Ghibli-style",
        "lora_weight": "Studio Ghibli style.safetensors",
        "prompt": "Ghibli style illustration, hand-painted look, soft watercolor textures, dreamy atmosphere, pastel colors, golden hour lighting, storybook quality",
        "negative_prompt": "ugly, dark, horror, scary, blurry, low quality, modern",
        "lora_scale": 1.0,
        "recommended_strength": 0.50,
    },
    "watercolor": {
        "name": "Watercolor Art",
        "emoji": "🌊",
        "lora_repo": "ostris/watercolor_style_lora_sdxl",
        "lora_weight": "watercolor_style_lora.safetensors",
        "prompt": "watercolor painting, wet-on-wet technique, soft color bleeds, paper texture, transparent washes, feathered edges, hand-painted",
        "negative_prompt": "sharp edges, solid flat colors, harsh lines, vector art, airbrushed",
        "lora_scale": 1.0,
        "recommended_strength": 0.50,
    },
    "oil_painting": {
        "name": "Classic Oil Paint",
        "emoji": "🖼️",
        "lora_repo": "EldritchAdam/ClassipeintXL",
        "lora_weight": "ClassipeintXL.safetensors",
        "prompt": "oil painting style, impasto technique, palette knife strokes, visible canvas texture, rich saturated pigments, masterful lighting, museum quality",
        "negative_prompt": "flat, smooth, cartoon, anime, blurry, low quality, modern, airbrushed",
        "lora_scale": 0.9,
        "recommended_strength": 0.50,
    },
    "pixel_art": {
        "name": "Pixel Art",
        "emoji": "👾",
        "lora_repo": "nerijs/pixel-art-xl",
        "lora_weight": "pixel-art-xl.safetensors",
        "prompt": "pixel art style, crisp blocky pixels, limited color palette, 16-bit aesthetic, retro game vibes, dithering effects, sprite art",
        "negative_prompt": "smooth, blurry, anti-aliased, soft gradient, painterly",
        "lora_scale": 0.9,
        "recommended_strength": 0.60,
    },
}

# Style Blend Presets - combining multiple styles (prompts kept short for CLIP 77 token limit)
STYLE_BLENDS = {
    "cartoon_anime": {
        "name": "3D Anime Fusion",
        "emoji": "🎭",
        "description": "70% 3D Cartoon + 30% Anime linework",
        "primary_style": "3d_cartoon",
        "secondary_style": "anime",
        "primary_weight": 0.7,
        "secondary_weight": 0.3,
        "prompt": "3D cartoon with anime linework, smooth features, soft lighting, CGI quality, vibrant colors, cel-shaded",
        "negative_prompt": "ugly, deformed, noisy, blurry, low quality",
        "strength": 0.52,
    },
    "fantasy_watercolor": {
        "name": "Dreamy Watercolor",
        "emoji": "🌈",
        "description": "60% Illustrated Fantasy + 40% Watercolor",
        "primary_style": "illustrated_fantasy",
        "secondary_style": "watercolor",
        "primary_weight": 0.6,
        "secondary_weight": 0.4,
        "prompt": "Ghibli style with watercolor washes, soft color bleeds, storybook atmosphere, paper texture, warm golden lighting",
        "negative_prompt": "dark, horror, harsh lines, solid colors",
        "strength": 0.50,
    },
    "anime_fantasy": {
        "name": "Anime Storybook",
        "emoji": "📖",
        "description": "50% Anime + 50% Illustrated Fantasy",
        "primary_style": "anime",
        "secondary_style": "illustrated_fantasy",
        "primary_weight": 0.5,
        "secondary_weight": 0.5,
        "prompt": "Ghibli anime illustration, hand-painted storybook, soft lighting, pastel colors, expressive eyes, warm glow",
        "negative_prompt": "ugly, deformed, bad anatomy, dark, horror, blurry",
        "strength": 0.48,
    },
    "oil_classical": {
        "name": "Renaissance Portrait",
        "emoji": "👑",
        "description": "Classical oil painting style",
        "primary_style": "oil_painting",
        "secondary_style": "oil_painting",
        "primary_weight": 1.0,
        "secondary_weight": 0.0,
        "prompt": "classical oil portrait, impasto technique, palette knife strokes, chiaroscuro lighting, canvas texture, museum quality",
        "negative_prompt": "flat, cartoon, anime, modern, minimalist, overexposed",
        "strength": 0.50,
    },
    "pixel_retro": {
        "name": "Retro Game Art",
        "emoji": "🕹️",
        "description": "Pixel art with enhanced retro feel",
        "primary_style": "pixel_art",
        "secondary_style": "pixel_art",
        "primary_weight": 1.0,
        "secondary_weight": 0.0,
        "prompt": "retro pixel art, crisp blocky pixels, limited palette, arcade aesthetic, dithering, 16-bit charm, sprite art",
        "negative_prompt": "smooth, blurry, anti-aliased, modern, gradient",
        "strength": 0.58,
    },
}


class StyleTransferEngine:
    """
    Multi-style image transformation engine using SDXL + LoRAs.
    Supports: 3D Cartoon, Anime, Watercolor, Oil Painting, Pixel Art styles.
    With IP-Adapter support for identity preservation.
    """

    BASE_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipe = None
        self.current_lora = None
        self.is_loaded = False
        self.ip_adapter_loaded = False

    def load_model(self) -> None:
        """Load SDXL base pipeline."""
        if self.is_loaded:
            return

        print("→ Loading SDXL base model...")

        from diffusers import AutoPipelineForImage2Image

        actual_device = "cuda" if torch.cuda.is_available() else self.device

        self.pipe = AutoPipelineForImage2Image.from_pretrained(
            self.BASE_MODEL,
            torch_dtype=torch.float16 if actual_device == "cuda" else torch.float32,
            variant="fp16" if actual_device == "cuda" else None,
            use_safetensors=True,
        )

        self.pipe.to(actual_device)

        # Enable memory optimizations
        if actual_device == "cuda":
            try:
                self.pipe.enable_xformers_memory_efficient_attention()
            except Exception:
                pass

        self.is_loaded = True
        self.device = actual_device
        print(f"✓ SDXL base loaded ({actual_device})")

    def _load_lora(self, style_key: str) -> None:
        """Load LoRA for the specified style."""
        config = STYLE_CONFIGS.get(style_key)
        if not config:
            return

        lora_repo = config.get("lora_repo")

        # Skip if no LoRA needed or already loaded
        if lora_repo is None:
            if self.current_lora is not None:
                print("→ Unloading previous LoRA...")
                self.pipe.unload_lora_weights()
                self.current_lora = None
            return

        if self.current_lora == lora_repo:
            return

        # Unload previous LoRA if different
        if self.current_lora is not None:
            print(f"→ Unloading previous LoRA: {self.current_lora}")
            self.pipe.unload_lora_weights()

        # Load new LoRA
        print(f"→ Loading LoRA: {config['name']}...")
        try:
            lora_weight = config.get("lora_weight")
            if lora_weight:
                self.pipe.load_lora_weights(lora_repo, weight_name=lora_weight)
            else:
                self.pipe.load_lora_weights(lora_repo)

            self.current_lora = lora_repo
            print(f"✓ LoRA loaded: {config['name']}")
        except Exception as e:
            print(f"⚠ LoRA loading failed: {e}, continuing without LoRA")
            self.current_lora = None

    def _load_ip_adapter(self) -> bool:
        """Load IP-Adapter for identity preservation."""
        if self.ip_adapter_loaded:
            return True

        if self.pipe is None:
            return False

        print("→ Loading IP-Adapter for face preservation...")
        try:
            self.pipe.load_ip_adapter(
                IP_ADAPTER_REPO,
                subfolder=IP_ADAPTER_SUBFOLDER,
                weight_name=IP_ADAPTER_WEIGHT
            )
            self.ip_adapter_loaded = True
            print("✓ IP-Adapter loaded")
            return True
        except Exception as e:
            print(f"⚠ IP-Adapter loading failed: {e}")
            self.ip_adapter_loaded = False
            return False

    def _unload_ip_adapter(self) -> None:
        """Unload IP-Adapter to free memory."""
        if not self.ip_adapter_loaded or self.pipe is None:
            return

        try:
            self.pipe.unload_ip_adapter()
            self.ip_adapter_loaded = False
            print("✓ IP-Adapter unloaded")
        except Exception as e:
            print(f"⚠ IP-Adapter unload failed: {e}")

    def unload_model(self) -> None:
        """Unload model and free memory."""
        if not self.is_loaded:
            return

        # Unload IP-Adapter first if loaded
        if self.ip_adapter_loaded:
            self._unload_ip_adapter()

        if self.pipe is not None:
            del self.pipe
            self.pipe = None

        self.current_lora = None
        self.ip_adapter_loaded = False

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.is_loaded = False
        print("✓ Model unloaded")

    def _preprocess_image(self, image: Image.Image) -> Image.Image:
        """Preprocess image for SDXL - resize to appropriate dimensions."""
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # SDXL works best with 1024x1024, maintain aspect ratio
        max_size = 1024
        width, height = image.size

        if width > height:
            new_width = max_size
            new_height = int(height * (max_size / width))
        else:
            new_height = max_size
            new_width = int(width * (max_size / height))

        # Round to nearest 8 (SDXL requirement)
        new_width = (new_width // 8) * 8
        new_height = (new_height // 8) * 8

        # Ensure minimum size
        new_width = max(new_width, 512)
        new_height = max(new_height, 512)

        image = image.resize((new_width, new_height), Image.LANCZOS)
        return image

    def generate_styled_image(
        self,
        image: Image.Image,
        style_key: str = "3d_cartoon",
        strength: float = 0.65,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 30,
        custom_prompt: str = "",
        seed: int = -1,
        face_restore: bool = False
    ) -> Tuple[Image.Image, int]:
        """
        Convert image to the specified style.

        Args:
            image: Input PIL Image
            style_key: One of: 3d_cartoon, anime, illustrated_fantasy, watercolor, oil_painting, pixel_art
            strength: How much to transform (0.0-1.0)
            guidance_scale: How closely to follow the prompt
            num_inference_steps: Number of denoising steps
            custom_prompt: Additional prompt text
            seed: Random seed (-1 for random)
            face_restore: Enable enhanced face preservation mode

        Returns:
            Tuple of (Stylized PIL Image, seed used)
        """
        if not self.is_loaded:
            self.load_model()

        # Get style config
        config = STYLE_CONFIGS.get(style_key, STYLE_CONFIGS["3d_cartoon"])

        # Load appropriate LoRA
        self._load_lora(style_key)

        # Preprocess
        print("→ Preprocessing image...")
        processed_image = self._preprocess_image(image)

        # Get style-specific face_restore settings
        face_settings = FACE_RESTORE_STYLE_SETTINGS.get(style_key, {
            "max_strength": 0.45, "lora_scale_mult": 0.7, "ip_scale": 0.5
        })

        # Build prompt based on face_restore mode
        base_prompt = config["prompt"]
        ip_adapter_image = None
        ip_scale = 0.0

        if face_restore:
            # Enhanced face preservation mode with style-specific settings
            preserve_prompt = FACE_RESTORE_PRESERVE
            negative_base = FACE_RESTORE_NEGATIVE

            # Apply style-specific strength cap
            max_str = face_settings["max_strength"]
            strength = min(strength, max_str)
            print(f"→ Face Restore enabled: strength capped at {strength} (style: {style_key})")

            # Load IP-Adapter for stronger identity preservation
            if self._load_ip_adapter():
                ip_adapter_image = processed_image
                ip_scale = face_settings["ip_scale"]
                print(f"→ IP-Adapter scale: {ip_scale}")
        else:
            preserve_prompt = IDENTITY_PRESERVE
            negative_base = IDENTITY_NEGATIVE
            # Unload IP-Adapter if not using face_restore (save memory)
            if self.ip_adapter_loaded:
                self._unload_ip_adapter()

        if custom_prompt:
            prompt = f"{preserve_prompt}, {base_prompt}, {custom_prompt}"
        else:
            prompt = f"{preserve_prompt}, {base_prompt}"

        # Build negative prompt
        negative_prompt = f"{negative_base}, {config['negative_prompt']}"

        # Set LoRA scale (reduce for face restore mode with style-specific multiplier)
        lora_scale = config.get("lora_scale", 1.0)
        if face_restore:
            lora_scale = lora_scale * face_settings["lora_scale_mult"]

        # Handle seed
        if seed == -1:
            seed = torch.randint(0, 2147483647, (1,)).item()
        generator = torch.Generator(device=self.device).manual_seed(seed)

        # Generate
        print(f"→ Generating {config['name']} style (strength: {strength}, steps: {num_inference_steps}, seed: {seed})...")

        # Build generation kwargs
        gen_kwargs = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "image": processed_image,
            "strength": strength,
            "guidance_scale": guidance_scale,
            "num_inference_steps": num_inference_steps,
            "generator": generator,
        }

        # Add cross_attention_kwargs only if LoRA is loaded
        if self.current_lora is not None:
            gen_kwargs["cross_attention_kwargs"] = {"scale": lora_scale}

        # Add IP-Adapter settings for face restoration
        if ip_adapter_image is not None and self.ip_adapter_loaded:
            self.pipe.set_ip_adapter_scale(ip_scale)
            gen_kwargs["ip_adapter_image"] = ip_adapter_image

        result = self.pipe(**gen_kwargs).images[0]

        print(f"✓ {config['name']} style generated (seed: {seed})")

        # Cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return result, seed

    def generate_blended_style(
        self,
        image: Image.Image,
        blend_key: str,
        custom_prompt: str = "",
        seed: int = -1,
        face_restore: bool = False
    ) -> Tuple[Image.Image, int]:
        """
        Generate image using a style blend preset.

        Args:
            image: Input PIL Image
            blend_key: Key from STYLE_BLENDS
            custom_prompt: Additional prompt text
            seed: Random seed (-1 for random)
            face_restore: Enable enhanced face preservation mode

        Returns:
            Tuple of (Stylized PIL Image, seed used)
        """
        if not self.is_loaded:
            self.load_model()

        blend_config = STYLE_BLENDS.get(blend_key)
        if not blend_config:
            return self.generate_styled_image(image, "3d_cartoon", seed=seed, face_restore=face_restore)

        # Get primary style for LoRA
        primary_style = blend_config["primary_style"]
        self._load_lora(primary_style)

        # Preprocess
        print("→ Preprocessing image...")
        processed_image = self._preprocess_image(image)

        # Get style-specific face_restore settings (use primary style)
        face_settings = FACE_RESTORE_STYLE_SETTINGS.get(primary_style, {
            "max_strength": 0.45, "lora_scale_mult": 0.7, "ip_scale": 0.5
        })

        # Build prompt based on face_restore mode
        base_prompt = blend_config["prompt"]
        ip_adapter_image = None
        ip_scale = 0.0

        if face_restore:
            preserve_prompt = FACE_RESTORE_PRESERVE
            negative_base = FACE_RESTORE_NEGATIVE

            # Apply style-specific strength cap
            max_str = face_settings["max_strength"]
            strength = min(blend_config["strength"], max_str)
            print(f"→ Face Restore enabled: strength capped at {strength} (blend: {blend_key})")

            # Load IP-Adapter for stronger identity preservation
            if self._load_ip_adapter():
                ip_adapter_image = processed_image
                ip_scale = face_settings["ip_scale"]
                print(f"→ IP-Adapter scale: {ip_scale}")
        else:
            preserve_prompt = IDENTITY_PRESERVE
            negative_base = IDENTITY_NEGATIVE
            strength = blend_config["strength"]
            # Unload IP-Adapter if not using face_restore
            if self.ip_adapter_loaded:
                self._unload_ip_adapter()

        if custom_prompt:
            prompt = f"{preserve_prompt}, {base_prompt}, {custom_prompt}"
        else:
            prompt = f"{preserve_prompt}, {base_prompt}"

        # Build negative prompt
        negative_prompt = f"{negative_base}, {blend_config['negative_prompt']}"

        # Get LoRA scale from primary style (reduce for face restore with style-specific multiplier)
        primary_config = STYLE_CONFIGS.get(primary_style, {})
        lora_scale = primary_config.get("lora_scale", 1.0) * blend_config["primary_weight"]
        if face_restore:
            lora_scale = lora_scale * face_settings["lora_scale_mult"]

        # Handle seed
        if seed == -1:
            seed = torch.randint(0, 2147483647, (1,)).item()
        generator = torch.Generator(device=self.device).manual_seed(seed)

        # Generate
        print(f"→ Generating {blend_config['name']} blend (seed: {seed})...")

        gen_kwargs = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "image": processed_image,
            "strength": strength,
            "guidance_scale": 7.5,
            "num_inference_steps": 30,
            "generator": generator,
        }

        if self.current_lora is not None:
            gen_kwargs["cross_attention_kwargs"] = {"scale": lora_scale}

        # Add IP-Adapter settings for face restoration
        if ip_adapter_image is not None and self.ip_adapter_loaded:
            self.pipe.set_ip_adapter_scale(ip_scale)
            gen_kwargs["ip_adapter_image"] = ip_adapter_image

        result = self.pipe(**gen_kwargs).images[0]

        print(f"✓ {blend_config['name']} blend generated (seed: {seed})")

        # Cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return result, seed

    def generate_all_outputs(
        self,
        image: Image.Image,
        style_key: str = "3d_cartoon",
        strength: float = 0.65,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 30,
        custom_prompt: str = "",
        seed: int = -1,
        is_blend: bool = False,
        face_restore: bool = False
    ) -> dict:
        """
        Generate styled image output.

        Returns dict with success status, stylized image, and seed used.
        """
        result = {
            "success": False,
            "stylized_image": None,
            "preview_image": None,
            "style_name": "",
            "seed_used": 0,
            "error": None
        }

        try:
            if is_blend:
                # Use blend preset
                blend_config = STYLE_BLENDS.get(style_key, {})
                result["style_name"] = blend_config.get("name", "Unknown Blend")

                stylized, seed_used = self.generate_blended_style(
                    image=image,
                    blend_key=style_key,
                    custom_prompt=custom_prompt,
                    seed=seed,
                    face_restore=face_restore
                )
            else:
                # Use single style
                config = STYLE_CONFIGS.get(style_key, STYLE_CONFIGS["3d_cartoon"])
                result["style_name"] = config["name"]

                stylized, seed_used = self.generate_styled_image(
                    image=image,
                    style_key=style_key,
                    strength=strength,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    custom_prompt=custom_prompt,
                    seed=seed,
                    face_restore=face_restore
                )

            result["stylized_image"] = stylized
            result["preview_image"] = stylized
            result["seed_used"] = seed_used
            result["success"] = True
            print(f"✓ {result['style_name']} conversion completed (seed: {seed_used})")

        except Exception as e:
            result["error"] = str(e)
            print(f"✗ Style conversion failed: {e}")

        return result

    @staticmethod
    def get_available_styles() -> Dict[str, Dict[str, Any]]:
        """Return available style configurations."""
        return {
            key: {
                "name": config["name"],
                "emoji": config["emoji"],
            }
            for key, config in STYLE_CONFIGS.items()
        }

    @staticmethod
    def get_style_choices() -> list:
        """Return style choices for UI dropdown."""
        return [
            f"{config['emoji']} {config['name']}"
            for config in STYLE_CONFIGS.values()
        ]

    @staticmethod
    def get_style_key_from_choice(choice: str) -> str:
        """Convert UI choice back to style key."""
        for key, config in STYLE_CONFIGS.items():
            if config["name"] in choice:
                return key
        return "3d_cartoon"

    @staticmethod
    def get_blend_choices() -> list:
        """Return blend preset choices for UI dropdown."""
        return [
            f"{config['emoji']} {config['name']} - {config['description']}"
            for config in STYLE_BLENDS.values()
        ]

    @staticmethod
    def get_blend_key_from_choice(choice: str) -> str:
        """Convert UI blend choice back to blend key."""
        for key, config in STYLE_BLENDS.items():
            if config["name"] in choice:
                return key
        return "cartoon_anime"

    @staticmethod
    def get_all_choices() -> dict:
        """Return both style and blend choices for UI."""
        styles = [
            f"{config['emoji']} {config['name']}"
            for config in STYLE_CONFIGS.values()
        ]
        blends = [
            f"{config['emoji']} {config['name']}"
            for config in STYLE_BLENDS.values()
        ]
        return {
            "styles": styles,
            "blends": blends,
            "all": styles + ["─── Style Blends ───"] + blends
        }
