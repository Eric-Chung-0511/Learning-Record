import torch
import torch.nn as nn
from model_manager import ModelManager
from multi_head_attention import MultiHeadAttention

class BaseModel(nn.Module):

    def __init__(self, backbone_fn, backbone_out_dim, num_classes, num_heads=8, device='cuda'):
        """
        Combines ModelManager and MultiHeadAttention into a single model.
        Args:
            backbone_fn (Callable): A function to initialize the backbone.
            backbone_out_dim (int): Output dimension of the backbone.
            num_classes (int): Number of classes for classification.
            num_heads (int): Number of attention heads.
            device (str): Device to run the model on.
        """
        super().__init__()
        self.manager = ModelManager(backbone_fn, backbone_out_dim, num_classes, device)
        self.attention = MultiHeadAttention(backbone_out_dim, num_heads)

    def forward(self, x):
        """
        Forward pass through the backbone and attention layers.
        Args:
            x (Tensor): Input tensor.
        Returns:
            Tuple[Tensor, Tensor]: Classification logits and attended features.
        """
        logits = self.manager.forward(x)
        attended_features = self.attention(logits)
        return logits, attended_features


# Usage Example
# from torchvision.models import resnet50, ResNet50_Weights
# model = BaseModel(
#     backbone_fn=lambda pretrained: resnet50(weights=ResNet50_Weights.IMAGENET1K_V1),
#     backbone_out_dim=2048,
#     num_classes=10,
#     num_heads=8
# )
