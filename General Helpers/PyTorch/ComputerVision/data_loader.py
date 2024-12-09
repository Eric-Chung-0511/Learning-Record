import os
import random
from torch.utils.data import DataLoader, Subset
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np

class ImageDataLoader:
    """
    通用的影像資料載入器，負責將 FlexibleImageDataset 轉換成可訓練用的 DataLoader
    支援：
    1. 訓練/測試集的分割
    2. 批次載入
    3. 自動偵測資料集類型（單一資料夾或多類別資料夾）
    """
    def __init__(self, dataset, train_ratio=0.8, seed=42):
        """
        初始化資料載入器
        
        Args:
            dataset: FlexibleImageDataset 實例
            train_ratio (float): 訓練集比例，預設 0.8
            seed (int): 隨機種子，確保可重複性
        """
        self.dataset = dataset
        self.train_ratio = train_ratio
        
        # 設定隨機種子以確保可重複性
        random.seed(seed)
        np.random.seed(seed)
        
        # 儲存資料集相關資訊
        self.dataset_size = len(dataset)
        self.num_classes = len(dataset.classes) if dataset.has_subdirs else 1
        
        # 初始化資料索引
        self.train_indices = []
        self.test_indices = []
        
        # 分割資料集
        self._split_indices()
        
    def _split_indices(self):
        """
        將資料集分割成訓練和測試集
        使用索引方式分割，避免實際複製資料
        """
        indices = list(range(self.dataset_size))
        random.shuffle(indices)
        
        # 計算訓練集大小
        train_size = int(self.dataset_size * self.train_ratio)
        
        # 分配索引
        self.train_indices = indices[:train_size]
        self.test_indices = indices[train_size:]
        
    def get_loaders(self, batch_size=32, num_workers=4, pin_memory=True):
        """
        取得訓練和測試用的 DataLoader
        
        Args:
            batch_size (int): 批次大小
            num_workers (int): 資料載入的工作程序數
            pin_memory (bool): 是否將資料釘在 GPU 記憶體，用於加速
            
        Returns:
            tuple: (訓練用 DataLoader, 測試用 DataLoader)
        """
        train_loader = DataLoader(
            dataset=self.dataset,
            batch_size=batch_size,
            sampler=SubsetRandomSampler(self.train_indices),
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
        
        return train_loader, test_loader
    
    def get_single_loader(self, indices=None, batch_size=32, shuffle=True, 
                         num_workers=4, pin_memory=True):
        """
        取得單一特定用途的 DataLoader
        
        Args:
            indices (list): 要使用的資料索引，如果為 None 則使用全部資料
            batch_size (int): 批次大小
            shuffle (bool): 是否打亂資料順序
            num_workers (int): 資料載入的工作程序數
            pin_memory (bool): 是否將資料釘在 GPU 記憶體
            
        Returns:
            DataLoader: 指定用途的資料載入器
        """
        if indices is None:
            # 如果沒有指定索引，使用整個資料集
            return DataLoader(
                dataset=self.dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=pin_memory
            )
        else:
            # 使用指定的索引建立子集
            subset = Subset(self.dataset, indices)
            return DataLoader(
                dataset=subset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=pin_memory
            )
    
    def get_dataset_info(self):
        """
        取得資料集的基本資訊
        
        Returns:
            dict: 包含資料集資訊的字典
        """
        info = {
            "Total samples": self.dataset_size,
            "Number of classes": self.num_classes,
            "Classes": self.dataset.classes,
            "Training set size": len(self.train_indices),
            "Testing set size": len(self.test_indices)
        }
        
        # 列印資訊
        print("\nDataset Information:")
        print(f"Total samples: {info['Total samples']}")
        print(f"Number of classes: {info['Number of classes']}")
        print(f"Training set size: {info['Training set size']}")
        print(f"Testing set size: {info['Testing set size']}")
        print(f"Available classes: {info['Classes']}\n")
        
        return info

# # Usage
# dataset = your_dataset
# data_loader = ImageDataLoader(
#     dataset=dataset,
#     train_ratio=0.8
# )

# # Can start begin training
# train_loader, test_loader = data_loader.get_loaders(
#     batch_size=32,
#     num_workers=4
# )
