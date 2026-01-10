import gc
import torch
from typing import Optional


class ResourceManager:
    """
    Centralized GPU resource management for model lifecycle.
    Ensures efficient memory usage and prevents OOM errors.
    """

    def __init__(self):
        """Initialize resource manager and detect device."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.current_model = None
        self.model_registry = {}

    def get_device(self) -> str:
        """
        Get current compute device.

        Returns:
            str: 'cuda' or 'cpu'
        """
        return self.device

    def register_model(self, model_name: str, model_instance: object) -> None:
        """
        Register a model instance for tracking.

        Args:
            model_name (str): Identifier for the model
            model_instance (object): The model object
        """
        self.model_registry[model_name] = model_instance
        self.current_model = model_name
        print(f"✓ Model registered: {model_name}")

    def unregister_model(self, model_name: str) -> None:
        """
        Unregister and cleanup a specific model.

        Args:
            model_name (str): Identifier of model to remove
        """
        if model_name in self.model_registry:
            del self.model_registry[model_name]
            if self.current_model == model_name:
                self.current_model = None
            print(f"✓ Model unregistered: {model_name}")

    def clear_cache(self, aggressive: bool = False) -> None:
        """
        Clear CUDA cache and run garbage collection.

        Args:
            aggressive (bool): If True, performs additional cleanup steps
        """
        if self.device == "cuda":
            # Standard cache clearing
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

            if aggressive:
                # Aggressive cleanup for critical memory situations
                gc.collect()
                with torch.cuda.device(self.device):
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()

            print(f"✓ CUDA cache cleared (aggressive={aggressive})")
        else:
            gc.collect()
            print("✓ CPU memory garbage collected")

    def cleanup_model(self, model_instance: Optional[object] = None) -> None:
        """
        Safely cleanup a model instance and free GPU memory.

        Args:
            model_instance (Optional[object]): Specific model to cleanup.
                                               If None, cleans all registered models.
        """
        if model_instance is not None:
            # Cleanup specific model
            if hasattr(model_instance, 'to'):
                model_instance.to('cpu')
            del model_instance
        else:
            # Cleanup all registered models
            for name, model in list(self.model_registry.items()):
                if hasattr(model, 'to'):
                    model.to('cpu')
                del model
                self.unregister_model(name)

        # Force cleanup
        gc.collect()
        self.clear_cache(aggressive=True)
        print("✓ Model cleanup completed")

    def get_memory_stats(self) -> dict:
        """
        Get current GPU memory statistics.

        Returns:
            dict: Memory statistics (allocated, reserved, free)
        """
        if self.device == "cuda" and torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            reserved = torch.cuda.memory_reserved() / 1024**3    # GB
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3

            return {
                "allocated_gb": round(allocated, 2),
                "reserved_gb": round(reserved, 2),
                "total_gb": round(total, 2),
                "free_gb": round(total - allocated, 2)
            }
        else:
            return {
                "allocated_gb": 0,
                "reserved_gb": 0,
                "total_gb": 0,
                "free_gb": 0
            }

    def ensure_memory_available(self, required_gb: float = 2.0) -> bool:
        """
        Check if sufficient GPU memory is available.

        Args:
            required_gb (float): Required memory in GB

        Returns:
            bool: True if memory is available, False otherwise
        """
        stats = self.get_memory_stats()
        available = stats["free_gb"]

        if available < required_gb:
            print(f"⚠ Low memory: {available:.2f}GB available, {required_gb:.2f}GB required")
            # Attempt cleanup
            self.clear_cache(aggressive=True)
            stats = self.get_memory_stats()
            available = stats["free_gb"]

        return available >= required_gb

    def switch_model_context(self, from_model: str, to_model: str) -> None:
        """
        Handle model switching with proper cleanup.

        Args:
            from_model (str): Current model to unload
            to_model (str): Next model to prepare for
        """
        print(f"→ Switching context: {from_model} → {to_model}")

        # Unregister old model
        self.unregister_model(from_model)

        # Aggressive cleanup before loading new model
        self.clear_cache(aggressive=True)

        # Update current model tracker
        self.current_model = to_model
        print(f"✓ Context switched to {to_model}")
