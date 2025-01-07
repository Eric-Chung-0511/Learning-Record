import os
import warnings
from collections import defaultdict
from typing import List, Dict, Tuple, Optional, Any
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image
from torchvision.transforms import RandAugment

class FlexibleImageDataset(Dataset):
    """
    靈活的圖像數據集類別，支援多種數據結構和圖像格式。
    可以處理：
    1. 單一資料夾中的圖像文件
    2. 具有類別子資料夾的層級結構
    """
    
    # PIL 支援的圖像格式列表
    SUPPORTED_FORMATS = [
        'jpg', 'jpeg', 'png', 'bmp', 'gif', 'tiff', 
        'webp', 'ppm', 'pgm', 'pbm', 'pnm'
    ]

    def __init__(
        self,
        root_dir: str,
        transform: Optional[Any] = None,
        is_train: bool = True,
        allowed_extensions: Optional[List[str]] = None,
        recursive_search: bool = False,
        min_samples_per_class: int = 1,
        image_size: int = 224,
        verbose: bool = True
    ):
        """
        初始化數據集
        
        參數:
            root_dir (str): 數據根目錄路徑
            transform: 自定義的數據轉換函數
            is_train (bool): 是否為訓練模式
            allowed_extensions (List[str]): 允許的文件擴展名列表
            recursive_search (bool): 是否遞迴搜索子目錄
            min_samples_per_class (int): 每個類別的最小樣本數
            image_size (int): 圖像的目標大小
            verbose (bool): 是否顯示詳細信息
        """
        self.root_dir = root_dir
        self.is_train = is_train
        self.allowed_extensions = allowed_extensions or self.SUPPORTED_FORMATS
        self.recursive_search = recursive_search
        self.min_samples_per_class = min_samples_per_class
        self.image_size = image_size
        self.verbose = verbose

        # 初始化統計信息
        self.stats = {
            'total_images': 0,
            'valid_images': 0,
            'corrupt_images': 0,
            'skipped_images': 0,
            'class_distribution': {}
        }

        # 設置數據轉換
        if transform is None:
            self.transform = self._get_default_transform()
        else:
            self.transform = transform

        # 檢查並載入數據
        self.has_subdirs = self._check_directory_structure()
        self._load_and_validate_data()

        if self.verbose:
            self._print_dataset_info()

    def _check_directory_structure(self) -> bool:
        """檢查數據集的目錄結構"""
        for item in os.listdir(self.root_dir):
            item_path = os.path.join(self.root_dir, item)
            if os.path.isdir(item_path):
                for sub_item in os.listdir(item_path):
                    if self._is_valid_image(os.path.join(item_path, sub_item)):
                        return True
        return False

    def _is_valid_image(self, img_path: str) -> bool:
        """驗證圖像文件的有效性"""
        if not any(img_path.lower().endswith(ext) for ext in self.allowed_extensions):
            self.stats['skipped_images'] += 1
            return False

        try:
            with Image.open(img_path) as img:
                img.verify()
            self.stats['valid_images'] += 1
            return True
        except Exception as e:
            self.stats['corrupt_images'] += 1
            if self.verbose:
                print(f"Warning: Corrupted image found at {img_path}: {str(e)}")
            return False

    def _get_default_transform(self):
        """獲取默認的數據轉換流程"""
        if self.is_train:
            return transforms.Compose([
                transforms.RandomResizedCrop(self.image_size),
                transforms.RandomHorizontalFlip(),
                RandAugment(num_ops=2, magnitude=9),
                transforms.ColorJitter(0.3, 0.3, 0.3, 0.1),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                transforms.RandomErasing(p=0.2)
            ])
        else:
            return transforms.Compose([
                transforms.Resize(int(self.image_size * 1.14)),
                transforms.CenterCrop(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

    def _load_and_validate_data(self):
        """載入並驗證數據"""
        if self.has_subdirs:
            self.dataset = ImageFolder(self.root_dir, transform=self.transform)
            self.classes = self.dataset.classes
            self.class_to_idx = self.dataset.class_to_idx
            self.samples = self.dataset.samples
            
            # 更新類別分佈統計
            for _, class_idx in self.samples:
                class_name = self.classes[class_idx]
                self.stats['class_distribution'][class_name] = \
                    self.stats['class_distribution'].get(class_name, 0) + 1
        else:
            self.classes = ['all']
            self.class_to_idx = {'all': 0}
            self.samples = [(os.path.join(self.root_dir, f), 0)
                          for f in os.listdir(self.root_dir)
                          if self._is_valid_image(os.path.join(self.root_dir, f))]
            self.stats['class_distribution']['all'] = len(self.samples)

        self.stats['total_images'] = len(self.samples)

    def _print_dataset_info(self):
        """輸出數據集信息"""
        print(f"Dataset Statistics:")
        print(f"- Total images: {self.stats['total_images']}")
        print(f"- Valid images: {self.stats['valid_images']}")
        print(f"- Corrupted images: {self.stats['corrupt_images']}")
        print(f"- Skipped images: {self.stats['skipped_images']}")
        print("\nClass distribution:")
        for class_name, count in self.stats['class_distribution'].items():
            print(f"- {class_name}: {count}")

    def __len__(self) -> int:
        """返回數據集長度"""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """獲取指定索引的數據項"""
        if self.has_subdirs:
            return self.dataset[idx]

        img_path, class_idx = self.samples[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, torch.tensor(class_idx, dtype=torch.long)
        except Exception as e:
            if self.verbose:
                print(f"Error loading image {img_path}: {e}")
            # 返回一個隨機的有效樣本
            return self[torch.randint(len(self), (1,)).item()]

# # 創建數據集的基本用法
# dataset = FlexibleImageDataset(
#     root_dir='path/to/your/image/folder',  # 圖像資料夾的路徑
#     transform=None,  # 使用默認的數據增強，如果需要自定義，可以傳入自己的 transform
#     is_train=True,  # True 表示訓練模式（會應用數據增強），False 表示評估模式
#     allowed_extensions=['jpg', 'jpeg', 'png'],  # 指定要處理的文件格式，不指定則使用所有支援的格式
#     recursive_search=False,  # 是否遞迴搜索子資料夾，只有一個母資料夾或一層類別子資料夾就用False，如果有巢狀或大於一層子資料夾就用True
#     min_samples_per_class=1,  # 每個類別最少需要的樣本數
#     image_size=224,  # 輸出圖像的大小
#     verbose=True  # 是否顯示詳細的數據集信息
# )
