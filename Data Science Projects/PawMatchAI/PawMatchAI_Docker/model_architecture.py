import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import numpy as np

dog_breeds = ["Afghan_Hound", "African_Hunting_Dog", "Airedale", "American_Staffordshire_Terrier",
              "Appenzeller", "Australian_Terrier", "Bedlington_Terrier", "Bernese_Mountain_Dog", "Bichon_Frise",
              "Blenheim_Spaniel", "Border_Collie", "Border_Terrier", "Boston_Bull", "Bouvier_Des_Flandres",
              "Brabancon_Griffon", "Brittany_Spaniel", "Cardigan", "Chesapeake_Bay_Retriever",
              "Chihuahua", "Dachshund", "Dandie_Dinmont", "Doberman", "English_Foxhound", "English_Setter",
              "English_Springer", "EntleBucher", "Eskimo_Dog", "French_Bulldog", "German_Shepherd",
              "German_Short-Haired_Pointer", "Gordon_Setter", "Great_Dane", "Great_Pyrenees",
              "Greater_Swiss_Mountain_Dog","Havanese", "Ibizan_Hound", "Irish_Setter", "Irish_Terrier",
              "Irish_Water_Spaniel", "Irish_Wolfhound", "Italian_Greyhound", "Japanese_Spaniel",
              "Kerry_Blue_Terrier", "Labrador_Retriever", "Lakeland_Terrier", "Leonberg", "Lhasa",
              "Maltese_Dog", "Mexican_Hairless", "Newfoundland", "Norfolk_Terrier", "Norwegian_Elkhound",
              "Norwich_Terrier", "Old_English_Sheepdog", "Pekinese", "Pembroke", "Pomeranian",
              "Rhodesian_Ridgeback", "Rottweiler", "Saint_Bernard", "Saluki", "Samoyed",
              "Scotch_Terrier", "Scottish_Deerhound", "Sealyham_Terrier", "Shetland_Sheepdog", "Shiba_Inu",
              "Shih-Tzu", "Siberian_Husky", "Staffordshire_Bullterrier", "Sussex_Spaniel",
              "Tibetan_Mastiff", "Tibetan_Terrier", "Walker_Hound", "Weimaraner",
              "Welsh_Springer_Spaniel", "West_Highland_White_Terrier", "Yorkshire_Terrier",
              "Affenpinscher", "Basenji", "Basset", "Beagle", "Black-and-Tan_Coonhound", "Bloodhound",
              "Bluetick", "Borzoi", "Boxer", "Briard", "Bull_Mastiff", "Cairn", "Chow", "Clumber",
              "Cocker_Spaniel", "Collie", "Curly-Coated_Retriever", "Dhole", "Dingo",
              "Flat-Coated_Retriever", "Giant_Schnauzer", "Golden_Retriever", "Groenendael", "Keeshond",
              "Kelpie", "Komondor", "Kuvasz", "Malamute", "Malinois", "Miniature_Pinscher",
              "Miniature_Poodle", "Miniature_Schnauzer", "Otterhound", "Papillon", "Pug", "Redbone",
              "Schipperke", "Silky_Terrier", "Soft-Coated_Wheaten_Terrier", "Standard_Poodle",
              "Standard_Schnauzer", "Toy_Poodle", "Toy_Terrier", "Vizsla", "Whippet",
              "Wire-Haired_Fox_Terrier"]


class MorphologicalFeatureExtractor(nn.Module):

    def __init__(self, in_features):
        super().__init__()

        # 基礎特徵維度設置
        self.reduced_dim = in_features // 4
        self.spatial_size = max(7, int(np.sqrt(self.reduced_dim // 64)))

        # 1. 特徵空間轉換器：將一維特徵轉換為二維空間表示
        self.dimension_transformer = nn.Sequential(
            nn.Linear(in_features, self.spatial_size * self.spatial_size * 64),
            nn.LayerNorm(self.spatial_size * self.spatial_size * 64),
            nn.ReLU()
        )

        # 2. 形態特徵分析器：分析具體的形態特徵
        self.morphological_analyzers = nn.ModuleDict({
            # 體型分析器：分析整體比例和大小
            'body_proportion': nn.Sequential(
                # 使用大卷積核捕捉整體體型特徵
                nn.Conv2d(64, 128, kernel_size=7, padding=3),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                # 使用較小的卷積核精煉特徵
                nn.Conv2d(128, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU()
            ),

            # 頭部特徵分析器：關注耳朵、臉部等
            'head_features': nn.Sequential(
                # 中等大小的卷積核，適合分析頭部結構
                nn.Conv2d(64, 128, kernel_size=5, padding=2),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                # 小卷積核捕捉細節
                nn.Conv2d(128, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU()
            ),

            # 尾部特徵分析器
            'tail_features': nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=5, padding=2),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Conv2d(128, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU()
            ),

            # 毛髮特徵分析器：分析毛髮長度、質地等
            'fur_features': nn.Sequential(
                # 使用多個小卷積核捕捉毛髮紋理
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Conv2d(128, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU()
            ),

            # 顏色特徵分析器：分析顏色分佈
            'color_pattern': nn.Sequential(
                # 第一層：捕捉基本顏色分布
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),

                # 第二層：分析顏色模式和花紋
                nn.Conv2d(128, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),

                # 第三層：整合顏色信息
                nn.Conv2d(128, 128, kernel_size=1),
                nn.BatchNorm2d(128),
                nn.ReLU()
            )
        })

        # 3. 特徵注意力機制：動態關注不同特徵
        self.feature_attention = nn.MultiheadAttention(
            embed_dim=128,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )

        # 4. 特徵關係分析器：分析不同特徵之間的關係
        self.relation_analyzer = nn.Sequential(
            nn.Linear(128 * 5, 256),  # 4個特徵分析器的輸出
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU()
        )

        # 5. 特徵整合器：將所有特徵智能地組合在一起
        self.feature_integrator = nn.Sequential(
            nn.Linear(128 * 6, in_features),  # 5個原始特徵 + 1個關係特徵
            nn.LayerNorm(in_features),
            nn.ReLU()
        )

    def forward(self, x):
        batch_size = x.size(0)

        # 1. 將特徵轉換為空間形式
        spatial_features = self.dimension_transformer(x).view(
            batch_size, 64, self.spatial_size, self.spatial_size
        )

        # 2. 分析各種形態特徵
        morphological_features = {}
        for name, analyzer in self.morphological_analyzers.items():
            # 提取特定形態特徵
            features = analyzer(spatial_features)
            # 使用自適應池化統一特徵大小
            pooled_features = F.adaptive_avg_pool2d(features, (1, 1))
            # 重塑特徵為向量形式
            morphological_features[name] = pooled_features.view(batch_size, -1)

        # 3. 特徵注意力處理
        # 將所有特徵堆疊成序列
        stacked_features = torch.stack(list(morphological_features.values()), dim=1)
        # 應用注意力機制
        attended_features, _ = self.feature_attention(
            stacked_features, stacked_features, stacked_features
        )

        # 4. 分析特徵之間的關係
        # 將所有特徵連接起來
        combined_features = torch.cat(list(morphological_features.values()), dim=1)
        # 提取特徵間的關係
        relation_features = self.relation_analyzer(combined_features)

        # 5. 特徵整合
        # 將原始特徵和關係特徵結合
        final_features = torch.cat([
            *morphological_features.values(),
            relation_features
        ], dim=1)

        # 6. 最終整合
        integrated_features = self.feature_integrator(final_features)

        # 添加殘差連接
        return integrated_features + x


class MultiHeadAttention(nn.Module):

    def __init__(self, in_dim, num_heads=8):
        """
        Initializes the MultiHeadAttention module.
        Args:
            in_dim (int): Dimension of the input features.
            num_heads (int): Number of attention heads. Defaults to 8.
        """
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = max(1, in_dim // num_heads)  
        self.scaled_dim = self.head_dim * num_heads  
        self.fc_in = nn.Linear(in_dim, self.scaled_dim)  
        self.query = nn.Linear(self.scaled_dim, self.scaled_dim)  # Query projection
        self.key = nn.Linear(self.scaled_dim, self.scaled_dim)  # Key projection
        self.value = nn.Linear(self.scaled_dim, self.scaled_dim)  # Value projection
        self.fc_out = nn.Linear(self.scaled_dim, in_dim)  # Linear layer to project output back to in_dim

    def forward(self, x):
        """
        Forward pass for multi-head attention mechanism.
        Args:
            x (Tensor): Input tensor of shape (batch_size, input_dim).
            x 是 (N,D), N:批次大小, D:輸入特徵維度
        Returns:
            Tensor: Output tensor after applying attention mechanism.
        """
        N = x.shape[0]  # Batch size
        x = self.fc_in(x)  # Project input to scaled_dim
        q = self.query(x).view(N, self.num_heads, self.head_dim)  # Compute queries
        k = self.key(x).view(N, self.num_heads, self.head_dim)  # Compute keys
        v = self.value(x).view(N, self.num_heads, self.head_dim)  # Compute values

        # Calculate attention scores
        energy = torch.einsum("nqd,nkd->nqk", [q, k])  
        attention = F.softmax(energy / (self.head_dim ** 0.5), dim=2)  # Apply softmax with scaling

        # Compute weighted sum of values based on attention scores
        out = torch.einsum("nqk,nvd->nqd", [attention, v]) 
        out = out.reshape(N, self.scaled_dim)  # Concatenate all heads
        out = self.fc_out(out)  # Project back to original input dimension
        return out


class BaseModel(nn.Module):

    def __init__(self, num_classes, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()
        self.device = device

        # 1. Initialize backbone
        self.backbone = timm.create_model(
                'convnextv2_base',
                pretrained=True,
                num_classes=0
        )

        # 2. 使用測試數據來確定實際的特徵維度
        with torch.no_grad():  
            dummy_input = torch.randn(1, 3, 224, 224)
            features = self.backbone(dummy_input)

            if len(features.shape) > 2:
                features = features.mean([-2, -1])

            self.feature_dim = features.shape[1]

        print(f"Feature Dimension from V2 backbone: {self.feature_dim}")

        # 3. Setup multi-head attention layer
        self.num_heads = max(1, min(8, self.feature_dim // 64))
        self.attention = MultiHeadAttention(self.feature_dim, num_heads=self.num_heads)

        # 4. Setup classifier
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Dropout(0.3),
            nn.Linear(self.feature_dim, num_classes)
        )

        self.morphological_extractor = MorphologicalFeatureExtractor(
            in_features=self.feature_dim
        )

        self.feature_fusion = nn.Sequential(
            nn.Linear(self.feature_dim * 3, self.feature_dim),  
            nn.LayerNorm(self.feature_dim),
            nn.ReLU(),
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.LayerNorm(self.feature_dim),
            nn.ReLU()
        )

    def forward(self, x):
        """
        Forward propagation process, combining V2's FCCA and multi-head attention mechanism
        Args:
            x (Tensor): Input image tensor of shape [batch_size, channels, height, width]
        Returns:
            Tuple[Tensor, Tensor]: Classification logits and attention features
        """
        x = x.to(self.device)

        # 1. Extract base features
        features = self.backbone(x)
        if len(features.shape) > 2:
            features = features.mean([-2, -1])

        # 2. Extract morphological features (including all detail features)
        morphological_features = self.morphological_extractor(features)

        # 3. Feature fusion (note dimension alignment with new fusion layer)
        combined_features = torch.cat([
            features,  # Original features
            morphological_features,  # Morphological features
            features * morphological_features  # Feature interaction information
        ], dim=1)
        fused_features = self.feature_fusion(combined_features)

        # 4. Apply attention mechanism
        attended_features = self.attention(fused_features)

        # 5. Final classifier
        logits = self.classifier(attended_features)

        return logits, attended_features
