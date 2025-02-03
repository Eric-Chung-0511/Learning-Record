import torch
import torch.nn as nn
import timm
import torchvision.models as tv_models
from typing import Optional, Tuple, Union, List, Dict

class ModelManager(nn.Module):
    """
    進階模型管理器：支援自動模型載入、特徵提取和訓練優化
    
    主要特色：
    1. 自動檢測模型來源（timm、torchvision）
    2. 自動特徵維度檢測
    3. 可選的 MultiHeadAttention 支援
    4. 進階的層級凍結/解凍機制
    5. 統一的模型接口
    """
    def __init__(
        self,
        model_name: str,                # 模型名稱 (如 'convnextv2_base')
        num_classes: int,               # 分類數量
        use_attention: bool = False,    # 是否使用 attention
        num_heads: int = 8,            # attention heads 數量
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        super().__init__()
        self.device = device
        self.model_name = model_name
        self.num_classes = num_classes
        
        # 初始化骨幹網路
        self.backbone = self._initialize_backbone()
        
        # 自動檢測特徵維度
        self.feature_dim = self._detect_feature_dim()
        
        # 設置 attention 層（如果需要）
        self.use_attention = use_attention
        if use_attention:
            self.attention = MultiHeadAttention(
                in_dim=self.feature_dim,
                num_heads=num_heads
            )
        
        # 建立分類器
        self.classifier = self._build_classifier()
        
        # 保存層級信息用於漸進式訓練
        self.layers = self._organize_layers()
        
        # 初始時凍結所有層
        self.freeze_backbone()
        
    def _initialize_backbone(self) -> nn.Module:
        """初始化並配置骨幹網路"""
        try:
            # 嘗試從 timm 加載
            if self.model_name in timm.list_models():
                model = timm.create_model(self.model_name, pretrained=True)
                # 移除分類層
                if hasattr(model, 'head'):
                    model.head = nn.Identity()
                elif hasattr(model, 'fc'):
                    model.fc = nn.Identity()
                return model.to(self.device)
                
            # 嘗試從 torchvision 加載
            if hasattr(tv_models, self.model_name):
                model = getattr(tv_models, self.model_name)(weights='DEFAULT')
                if hasattr(model, 'fc'):
                    model.fc = nn.Identity()
                elif hasattr(model, 'classifier'):
                    model.classifier = nn.Identity()
                return model.to(self.device)
                
        except Exception as e:
            raise ValueError(f"Error loading model {self.model_name}: {str(e)}")
            
        raise ValueError(f"Model {self.model_name} not found in supported sources")
        
    def _detect_feature_dim(self) -> int:
        """使用 dummy input 自動檢測特徵維度"""
        self.backbone.eval()
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
            features = self.backbone(dummy_input)
            return features.shape[1]
            
    def _build_classifier(self) -> nn.Sequential:
        """建立分類器頭部"""
        return nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Dropout(0.3),
            nn.Linear(self.feature_dim, self.num_classes)
        ).to(self.device)
        
    def _organize_layers(self) -> List[nn.Module]:
        """組織模型層級，用於漸進式解凍"""
        layers = []
        
        # 處理不同類型的模型架構
        if hasattr(self.backbone, 'stages'):
            # ConvNeXt 類型的模型
            layers = list(self.backbone.stages)
        elif hasattr(self.backbone, 'layer1'):
            # ResNet 類型的模型
            layers = [
                self.backbone.layer1,
                self.backbone.layer2,
                self.backbone.layer3,
                self.backbone.layer4
            ]
        else:
            # 其他類型模型，使用 children 分層
            layers = list(self.backbone.children())
            
        return layers
        
    def freeze_backbone(self):
        """凍結所有骨幹網路層"""
        for param in self.backbone.parameters():
            param.requires_grad = False
            
    def unfreeze_layers(self, num_layers: Optional[int] = None):
        """
        解凍指定數量的層
        Args:
            num_layers: 要解凍的層數，None 表示解凍所有層
        """
        if num_layers is None:
            # 解凍所有層
            for param in self.backbone.parameters():
                param.requires_grad = True
        else:
            # 解凍指定數量的後部層
            for layer in self.layers[-num_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True
                    
    def get_trainable_params(self) -> List[Dict]:
        """
        獲取需要訓練的參數，用於優化器
        為不同組件設置不同的學習率
        """
        params = []
        
        # 骨幹網路參數
        backbone_params = [p for p in self.backbone.parameters() if p.requires_grad]
        if backbone_params:
            params.append({'params': backbone_params, 'lr': 1e-4})
            
        # Attention 層參數（如果有）
        if self.use_attention:
            params.append({'params': self.attention.parameters(), 'lr': 1e-4})
            
        # 分類器參數
        params.append({'params': self.classifier.parameters(), 'lr': 1e-3})
        
        return params
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向傳播
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (features, logits)
        """
        x = x.to(self.device)
        features = self.backbone(x)
        
        if self.use_attention:
            features = self.attention(features)
            
        logits = self.classifier(features)
        return features, logits

