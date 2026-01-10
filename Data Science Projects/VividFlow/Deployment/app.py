import os
import sys

# CRITICAL: Import spaces FIRST before any CUDA initialization
try:
    import spaces
except ImportError:
    pass

sys.stdout.flush()
import functools
print = functools.partial(print, flush=True)

import ftfy
import sentencepiece

from FlowFacade import FlowFacade
from BackgroundEngine import BackgroundEngine
from ui_manager import UIManager


def preload_models():
    """
    Pre-download models to cache on HF Spaces startup.
    Backup method if YAML preload_from_hub doesn't work.
    Only runs in HF Spaces environment.
    """
    if not os.environ.get('SPACE_ID'):
        return

    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    if os.path.exists(cache_dir):
        cached_models = os.listdir(cache_dir)
        if any("wan2.2" in m.lower() or "models--kijai" in m.lower() for m in cached_models):
            print("✓ Models already cached (YAML preload worked)")
            return

    print("→ Pre-caching models to disk (first-time setup)...")
    print("  This may take 2-3 minutes, please wait...")

    try:
        from diffusers import WanTransformer3DModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from huggingface_hub import hf_hub_download
        import torch

        print("  [1/4] Downloading video model transformer...")
        WanTransformer3DModel.from_pretrained(
            "cbensimon/Wan2.2-I2V-A14B-bf16-Diffusers",
            subfolder='transformer',
            torch_dtype=torch.bfloat16,
        )

        print("  [2/4] Downloading video model transformer_2...")
        WanTransformer3DModel.from_pretrained(
            "cbensimon/Wan2.2-I2V-A14B-bf16-Diffusers",
            subfolder='transformer_2',
            torch_dtype=torch.bfloat16,
        )

        print("  [3/4] Downloading Lightning LoRA...")
        hf_hub_download(
            "Kijai/WanVideo_comfy",
            "Lightx2v/lightx2v_I2V_14B_480p_cfg_step_distill_rank128_bf16.safetensors"
        )

        print("  [4/4] Downloading text model (optional)...")
        AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-0.5B-Instruct",
            torch_dtype=torch.bfloat16,
        )
        AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

        print("✓ All models cached successfully!")
        print("  Future users will load instantly from cache")

    except Exception as e:
        print(f"⚠ Pre-cache warning: {e}")
        print("  Models will download on first generation instead")


def check_environment():
    required_packages = [
        "torch", "transformers", "diffusers", "gradio", "PIL",
        "accelerate", "numpy", "ftfy", "sentencepiece"
    ]

    optional_packages = {
        "torchao": "INT8/FP8 quantization",
        "xformers": "Memory efficient attention",
        "aoti": "AoT compilation"
    }

    missing_packages = []
    missing_optional = []

    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)

    for package, description in optional_packages.items():
        try:
            __import__(package)
        except ImportError:
            missing_optional.append(f"{package} ({description})")

    if missing_packages:
        print("\n❌ Missing required packages:", ", ".join(missing_packages))
        print("\nInstall commands:")
        print("!pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/cu126")
        print("!pip install diffusers>=0.32.0 transformers>=4.46.0 accelerate gradio pillow numpy spaces ftfy sentencepiece protobuf imageio-ffmpeg")
        print("!pip install torchao xformers")
        sys.exit(1)

    # Only show missing optional in debug mode
    if missing_optional and os.environ.get('DEBUG'):
        print("⚠ Optional packages missing:", ", ".join(missing_optional))


def main():
    check_environment()
    preload_models()

    try:
        facade = FlowFacade()
        background_engine = BackgroundEngine()
        ui_manager = UIManager(facade, background_engine)
        interface = ui_manager.create_interface()
        is_colab = 'google.colab' in sys.modules

        print("✓ Ready")
        interface.launch(
            share=is_colab,
            server_name="0.0.0.0",
            server_port=None,
            show_error=True
        )

    except KeyboardInterrupt:
        print("\n⚠ Shutdown requested")
        if 'facade' in locals():
            facade.cleanup()
        sys.exit(0)

    except Exception as e:
        print(f"\n❌ Startup error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
