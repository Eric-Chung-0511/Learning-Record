import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.transforms import RandAugment
from PIL import Image

class FlexibleImageDataset(Dataset):
    def __init__(self, root_dir, transform=None, is_train=True, allowed_extensions=None):
        """
        靈活的影像資料集類別，可以同時處理兩種資料結構：
        1. 有類別子資料夾的結構（使用 ImageFolder）
        2. 單一資料夾直接存放圖片的結構
        """
        self.root_dir = root_dir
        self.is_train = is_train
        self.allowed_extensions = allowed_extensions or ['jpg', 'jpeg', 'png']
        
        # 設定資料轉換
        if transform is None:
            self.transform = self._get_default_transform()
        else:
            self.transform = transform

        # 檢查並設置資料結構
        self.has_subdirs = self._check_directory_structure()
        
        if self.has_subdirs:
            # 使用 ImageFolder 處理有子資料夾的情況
            self.dataset = ImageFolder(root_dir, transform=self.transform)
            self.classes = self.dataset.classes
            self.class_to_idx = self.dataset.class_to_idx
            self.samples = self.dataset.samples
        else:
            # 處理單一資料夾的情況
            self.classes = ['all']
            self.class_to_idx = {'all': 0}
            self.samples = self._load_flat_directory()

    def _check_directory_structure(self):
        """檢查資料集的目錄結構"""
        for item in os.listdir(self.root_dir):
            item_path = os.path.join(self.root_dir, item)
            if os.path.isdir(item_path):
                for sub_item in os.listdir(item_path):
                    if self._is_valid_image(os.path.join(item_path, sub_item)):
                        return True
        return False

    def _load_flat_directory(self):
        """載入單一資料夾中的所有圖片"""
        return [(os.path.join(self.root_dir, img_name), 0)
                for img_name in os.listdir(self.root_dir)
                if self._is_valid_image(os.path.join(self.root_dir, img_name))]

    def _is_valid_image(self, img_path):
        """檢查檔案是否為允許的圖片格式"""
        return img_path.split('.')[-1].lower() in self.allowed_extensions

    def _get_default_transform(self):
        """提供預設的資料增強變換"""
        if self.is_train:
            transform = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.08, 1.0), ratio=(3/4, 4/3)),
                transforms.RandomHorizontalFlip(),
                RandAugment(num_ops=2, magnitude=9),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                transforms.RandomErasing(p=0.2, scale=(0.02, 0.33), ratio=(0.3, 3.3))
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        return transform

    def __len__(self):
        """
        統一的長度取得方法
        - 對於 ImageFolder 結構：使用 ImageFolder 的 __len__
        - 對於單一資料夾：使用 samples 的長度
        """
        if self.has_subdirs:
            return len(self.dataset)
        return len(self.samples)

    def __getitem__(self, idx):
        """
        統一的資料取得方法
        - 對於 ImageFolder 結構：直接使用 ImageFolder 的 __getitem__
        - 對於單一資料夾：自行處理圖片載入和轉換
        """
        if self.has_subdirs:
            return self.dataset[idx]
            
        # 處理單一資料夾的情況
        img_path, class_idx = self.samples[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, torch.tensor(class_idx, dtype=torch.long)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return None, None

# # Usage
# # 對於有子資料夾的資料集（例如：分類資料夾）
# dataset = FlexibleImageDataset(
#     root_dir='path/to/data_folder',
#     is_train=True
# )

# # 對於單一資料夾的資料集
# dataset = FlexibleImageDataset(
#     root_dir='path/to/single_folder',
#     is_train=False
# )
