import os
import torch
import numpy as np
from PIL import Image
from typing import Tuple, Optional
from VideoEngine_optimized import VideoEngine
from TextProcessor import TextProcessor

try:
    import spaces
    HAS_SPACES = True
except ImportError:
    HAS_SPACES = False
    class spaces:
        @staticmethod
        def GPU(duration=120):
            def decorator(func):
                return func
            return decorator


class FlowFacade:
    def __init__(self):
        self.is_spaces = os.environ.get('SPACE_ID') is not None
        self.video_engine = VideoEngine()
        self.text_processor = TextProcessor(resource_manager=None)
        print("✓ DeltaFlow initialized")

    @spaces.GPU(duration=300)
    def generate_video_from_image(self, image: Image.Image, user_instruction: str,
                                  duration_seconds: float = 3.0, num_inference_steps: int = 4,
                                  guidance_scale: float = 1.0, guidance_scale_2: float = 1.0,
                                  seed: int = 42, randomize_seed: bool = False,
                                  enable_prompt_expansion: bool = False,
                                  progress=None) -> Tuple[str, str, int]:
        if image is None:
            raise ValueError("No image provided")
        if not user_instruction or user_instruction.strip() == "":
            raise ValueError("Please provide a motion instruction")

        try:
            if randomize_seed:
                seed = np.random.randint(0, 2147483647)

            if enable_prompt_expansion:
                if progress:
                    progress(0.1, desc="AI expanding your prompt...")
                final_prompt = self.text_processor.process(user_instruction, auto_unload=True)
            else:
                final_prompt = user_instruction

            if progress:
                progress(0.2, desc="Preparing GPU memory...")

            if not self.video_engine.is_loaded:
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()

                if progress:
                    progress(0.25, desc="Loading video generation model...")
                self.video_engine.load_model()

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if progress:
                progress(0.3, desc=f"Generating video ({num_inference_steps} steps)...")

            video_path = self.video_engine.generate_video(
                image=image, prompt=final_prompt, duration_seconds=duration_seconds,
                num_inference_steps=num_inference_steps, guidance_scale=guidance_scale,
                guidance_scale_2=guidance_scale_2, seed=seed
            )

            if progress:
                progress(1.0, desc="Complete!")

            return video_path, final_prompt, seed

        except Exception as e:
            import traceback
            print(f"\n✗ Generation error: {type(e).__name__}: {str(e)}")
            if os.environ.get('DEBUG'):
                print(traceback.format_exc())
            raise RuntimeError(f"Generation failed: {type(e).__name__}: {str(e)}")

    def cleanup(self) -> None:
        try:
            if hasattr(self.text_processor, 'is_loaded') and self.text_processor.is_loaded:
                self.text_processor.unload_model()
            torch.cuda.empty_cache()
        except Exception as e:
            if os.environ.get('DEBUG'):
                print(f"⚠ Cleanup warning: {str(e)}")

    def get_system_info(self) -> dict:
        quantization_type = "None"
        if torch.cuda.is_available():
            cuda_cap = torch.cuda.get_device_capability()
            fp8_supported = cuda_cap[0] > 8 or (cuda_cap[0] == 8 and cuda_cap[1] >= 9)
            quantization_type = "FP8" if fp8_supported else "INT8"

        return {
            "device": self.video_engine.device,
            "video_model": VideoEngine.MODEL_ID,
            "text_model": TextProcessor.MODEL_ID,
            "lightning_lora": "Enabled",
            "quantization": quantization_type,
            "optimizations": [
                "Lightning LoRA (4-8 steps)",
                f"{quantization_type} Quantization"
            ]
        }

    def validate_image(self, image: Image.Image) -> bool:
        if image is None:
            return False

        min_dim, max_dim = 256, 4096

        if image.width < min_dim or image.height < min_dim:
            print(f"⚠ Image too small: {image.width}x{image.height}")
            return False

        if image.width > max_dim or image.height > max_dim:
            print(f"⚠ Image too large: {image.width}x{image.height}")
            return False

        return True

    def __del__(self):
        try:
            self.cleanup()
        except:
            pass
