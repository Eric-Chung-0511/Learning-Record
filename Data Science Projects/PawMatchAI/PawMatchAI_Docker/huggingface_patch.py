import huggingface_hub
import sys
import warnings

# Check if cached_download exists
if not hasattr(huggingface_hub, 'cached_download'):
    # Provide a fallback function
    def cached_download(*args, **kwargs):
        """
        Simulates the behavior of the deprecated cached_download function.
        Internally uses the newer hf_hub_download function.
        """
        warnings.warn(
            "Using a simulated version of cached_download. This function has been removed in the newer version of huggingface_hub.",
            DeprecationWarning,
            stacklevel=2
        )
        # Call the new equivalent function
        return huggingface_hub.hf_hub_download(*args, **kwargs)

    # Add the simulated function to the huggingface_hub module
    huggingface_hub.cached_download = cached_download
    print("Successfully added simulated cached_download function to huggingface_hub")
else:
    print("huggingface_hub already includes the cached_download function")

# For more comprehensive patching, check and add the model_info function
if not hasattr(huggingface_hub, 'model_info') and hasattr(huggingface_hub, 'api'):
    def model_info(*args, **kwargs):
        """Simulated version of the deprecated model_info function"""
        warnings.warn(
            "Using a simulated version of model_info. This function may have been moved or renamed in the newer version of huggingface_hub.",
            DeprecationWarning,
            stacklevel=2
        )
        # Use the newer equivalent API
        return huggingface_hub.api.model_info(*args, **kwargs)

    # Add to the module
    huggingface_hub.model_info = model_info
    print("Successfully added simulated model_info function to huggingface_hub")
