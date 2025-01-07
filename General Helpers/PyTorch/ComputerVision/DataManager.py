import os
import random
from typing import Tuple, Dict, Any
from torch.utils.data import DataLoader, Subset
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np

class DataManager:
    """
    進階的資料管理器，用於處理資料集的分割和載入
    特色：
    1. 支援彈性的資料集分割（訓練/驗證/測試）
    2. 提供完整的資料集統計資訊
    3. 自動處理資料平衡和批次載入
    4. 整合錯誤處理和資料驗證
    """
    def __init__(self, dataset: Any, train_ratio: float = 0.7, val_ratio: float = 0.15, seed: int = 42) -> None:
        """
        初始化資料管理器
        
        參數:
            dataset: FlexibleImageDataset 實例
            train_ratio: 訓練集比例（預設 0.7）
            val_ratio: 驗證集比例（預設 0.15）
            seed: 隨機種子
        """
        # 驗證輸入參數
        self._validate_ratios(train_ratio, val_ratio)
        
        self.dataset = dataset
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = 1 - train_ratio - val_ratio
        
        # 設定隨機種子
        self._set_random_seed(seed)
        
        # 初始化基本資訊
        self.dataset_size = len(dataset)
        self.num_classes = len(dataset.classes) if hasattr(dataset, 'has_subdirs') and dataset.has_subdirs else 1
        
        # 初始化資料索引
        self.train_indices: list = []
        self.val_indices: list = []
        self.test_indices: list = []
        
        # 執行資料分割
        self._split_indices()
        
    def _validate_ratios(self, train_ratio: float, val_ratio: float) -> None:
        """驗證分割比例的合法性"""
        if not 0 < train_ratio < 1:
            raise ValueError("Train ratio must be between 0 and 1")
        if not 0 < val_ratio < 1:
            raise ValueError("Validation ratio must be between 0 and 1")
        if train_ratio + val_ratio >= 1:
            raise ValueError("Sum of train and validation ratios must be less than 1")

    def _set_random_seed(self, seed: int) -> None:
        """設定隨機種子確保可重現性"""
        random.seed(seed)
        np.random.seed(seed)
        
    def _split_indices(self) -> None:
        """執行三向資料分割"""
        indices = list(range(self.dataset_size))
        random.shuffle(indices)
        
        # 計算分割點
        train_split = int(self.dataset_size * self.train_ratio)
        val_split = int(self.dataset_size * (self.train_ratio + self.val_ratio))
        
        # 分配索引
        self.train_indices = indices[:train_split]
        self.val_indices = indices[train_split:val_split]
        self.test_indices = indices[val_split:]
        
    def get_loaders(self, batch_size: int = 32, num_workers: int = 4, pin_memory: bool = True) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        獲取所有資料載入器
        
        參數:
            batch_size: 批次大小
            num_workers: 資料載入的工作程序數
            pin_memory: 是否將資料釘在 GPU 記憶體
            
        返回:
            tuple: (訓練集載入器, 驗證集載入器, 測試集載入器)
        """
        try:
            train_loader = DataLoader(
                dataset=self.dataset,
                batch_size=batch_size,
                sampler=SubsetRandomSampler(self.train_indices),
                num_workers=num_workers,
                pin_memory=pin_memory
            )
            
            val_loader = DataLoader(
                dataset=self.dataset,
                batch_size=batch_size,
                sampler=SubsetRandomSampler(self.val_indices),
                num_workers=num_workers,
                pin_memory=pin_memory
            )
            
            test_loader = DataLoader(
                dataset=self.dataset,
                batch_size=batch_size,
                sampler=SubsetRandomSampler(self.test_indices),
                num_workers=num_workers,
                pin_memory=pin_memory
            )
            
            return train_loader, val_loader, test_loader
            
        except Exception as e:
            raise RuntimeError(f"Error creating data loaders: {str(e)}")
    
    def get_dataset_info(self) -> Dict:
        """
        獲取資料集的詳細資訊
        
        返回:
            dict: 包含資料集完整資訊的字典
        """
        info = {
            "Dataset Statistics": {
                "Total samples": self.dataset_size,
                "Number of classes": self.num_classes,
                "Available classes": self.dataset.classes,
            },
            "Split Information": {
                "Training set size": len(self.train_indices),
                "Validation set size": len(self.val_indices),
                "Testing set size": len(self.test_indices)
            },
            "Split Ratios": {
                "Training": f"{self.train_ratio:.1%}",
                "Validation": f"{self.val_ratio:.1%}",
                "Testing": f"{self.test_ratio:.1%}"
            }
        }
        
        print("\nDataset Information:")
        for category, details in info.items():
            print(f"\n{category}:")
            for key, value in details.items():
                print(f"  {key}: {value}")
        print()
        
        return info



# # 第一步：建立資料集（使用 FlexibleImageDataset）
# dataset = FlexibleImageDataset(
#     root_dir='path/to/images',
#     is_train=True,
#     image_size=224,
#     recursive_search=True,  # 如果有巢狀資料夾結構
#     verbose=True
# )

# # 第二步：建立資料管理器
# # 如果數據資料非常大，可使用 train=0.8, val=0.1
# data_manager = DataManager(
#     dataset=dataset,
#     train_ratio=0.7,   # 70% 訓練集
#     val_ratio=0.15     # 15% 驗證集，15% 測試集
# )

# # 第三步：檢查資料集資訊
# data_info = data_manager.get_dataset_info()

# # 第四步：獲取資料載入器
# train_loader, val_loader, test_loader = data_manager.get_loaders(
#     batch_size=32,
#     num_workers=4,
#     pin_memory=True  # 使用 GPU 時建議設為 True
# )
