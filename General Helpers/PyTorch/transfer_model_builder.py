"""
Contains PyTorch model code to instantiate a Transfer Learning model with flexible layer freezing.
"""
import torch
from torch import nn
from torchvision import models

def create_transfer_model(model_name: str, num_classes: int, num_unfrozen_layers: int = 0):
    """
    Creates a transfer learning model based on a specified architecture with flexible layer unfreezing.

    Args:
    model_name: The name of the pretrained model to use (e.g., 'resnet50', 'efficientnet_b0').
    num_classes: The number of output classes for the model.
    num_unfrozen_layers: The number of final layers to unfreeze for fine-tuning. 
                         If 0, the entire model except the classifier will be frozen.

    Returns:
    A PyTorch model ready for training or fine-tuning.
    """
    # Select the model based on the input model name
    if model_name == "resnet50":
        model = models.resnet50(weights="IMAGENET1K_V2")  # Load the pre-trained ResNet50 model
        in_features = model.fc.in_features  # Get the number of input features for the classifier
        model.fc = nn.Linear(in_features, num_classes)  # Replace the classifier head

        # Freeze all layers
        for param in model.parameters():
            param.requires_grad = False
        
        # Unfreeze the last num_unfrozen_layers layers
        if num_unfrozen_layers > 0:
            layers = list(model.children())[-(num_unfrozen_layers + 1):]  # +1 to include the classifier
            for layer in layers:
                for param in layer.parameters():
                    param.requires_grad = True

    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights="IMAGENET1K_V2")  # Load the pre-trained EfficientNet-B0 model
        in_features = model.classifier[1].in_features  # Get the number of input features for the classifier
        model.classifier[1] = nn.Linear(in_features, num_classes)  # Replace the classifier head

        # Freeze all layers
        for param in model.parameters():
            param.requires_grad = False
        
        # Unfreeze the last num_unfrozen_layers layers
        if num_unfrozen_layers > 0:
            layers = list(model.features.children())[-num_unfrozen_layers:] + [model.classifier]
            for layer in layers:
                for param in layer.parameters():
                    param.requires_grad = True

    else:
        raise ValueError(f"Model {model_name} is not supported. Please choose 'resnet50' or 'efficientnet_b0'.")

    return model

# Example usage:
if __name__ == "__main__":
    # Set the model name, number of classes, and number of layers to unfreeze
    MODEL_NAME = "resnet50"  # Choose between 'resnet50' or 'efficientnet_b0'
    NUM_CLASSES = 10  # Number of classes in your dataset
    NUM_UNFROZEN_LAYERS = 3  # Number of layers to unfreeze for fine-tuning

    # Create the model
    model = create_transfer_model(MODEL_NAME, NUM_CLASSES, NUM_UNFROZEN_LAYERS)

    # Print the model architecture
    print(model)
