import torch
import torch.nn as nn

class ModelManager:
  
    def __init__(self, backbone_fn, backbone_out_dim, num_classes, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Handles the backbone model and its customization for transfer learning.
        Args:
            backbone_fn (Callable): A function or class to initialize the backbone model.
            backbone_out_dim (int): Output feature dimension of the backbone.
            num_classes (int): Number of output classes for classification.
            device (str): Device to run the model on ('cuda' or 'cpu').
        """
        self.device = device
        self.backbone = backbone_fn(pretrained=True)  # Initialize backbone with pretrained weights
        self.feature_dim = backbone_out_dim
        self.backbone.classifier = nn.Identity()  # Remove classifier layer
        self.num_classes = num_classes

        # Define a classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Dropout(0.3),
            nn.Linear(self.feature_dim, self.num_classes)
        )

    def freeze_backbone(self):
        """Freezes all layers of the backbone."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self, num_layers=None):
        """
        Unfreezes layers of the backbone for fine-tuning.
        Args:
            num_layers (int, optional): Number of layers to unfreeze. If None, unfreeze all layers.
        """
        layers = list(self.backbone.children())
        if num_layers is None:
            for param in self.backbone.parameters():
                param.requires_grad = True
        else:
            for layer in layers[-num_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True

    def forward(self, x):
        """
        Performs a forward pass through the backbone and classifier.
        Args:
            x (Tensor): Input tensor.
        Returns:
            Tensor: Output logits.
        """
        features = self.backbone(x.to(self.device))
        logits = self.classifier(features)
        return logits
      

# Usage Example
# from torchvision.models import resnet50, ResNet50_Weights
# manager = ModelManager(
#     backbone_fn=lambda pretrained: resnet50(weights=ResNet50_Weights.IMAGENET1K_V1),
#     backbone_out_dim=2048,
#     num_classes=10
# )
# manager.freeze_backbone()

