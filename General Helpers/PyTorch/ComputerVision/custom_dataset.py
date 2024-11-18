import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from torchvision.transforms import RandAugment

class CustomImageDataset(Dataset):
    
    def __init__(self, root_dir, transform=None, is_train=True, use_subdirectories=True, allowed_extensions=None):
        """
        通用影像資料集類別，適用於各類影像分類專案。
        Args:
            root_dir (str): 資料集根目錄。
            transform (callable, optional): 資料增強變換，預設為 None。
            is_train (bool): 是否為訓練資料集，決定資料增強策略。
            use_subdirectories (bool): 是否使用子目錄作為類別目錄，預設為 True。
            allowed_extensions (list): 可接受的影像檔案格式，預設為 ['jpg', 'jpeg', 'png']。
        """
        self.root_dir = root_dir
        self.is_train = is_train
        self.use_subdirectories = use_subdirectories
        self.allowed_extensions = allowed_extensions or ['jpg', 'jpeg', 'png']
        self.classes = [] # 存儲所有類別名稱的列表
        self.class_to_idx = {} # 建立類別名稱與類別索引之間的映射字典, 
        self.samples = [] # 存儲 (image_path, class_index) 配對的列表
        self._load_dataset() # 初始化此部分是為了準備好 self.classes, self.class_to_idx, self.samples,並避免重複讀取資料及確保後續 __getitem__()能正常運作

        # 使用預設變換或指定的變換
        if transform is None:
            self.transform = self._get_default_transform()
        else:
            self.transform = transform

    def _load_dataset(self):
        """
        讀取資料集並建立 (影像路徑, 類別索引) 的配對。
        """
        if self.use_subdirectories:
            for class_idx, class_name in enumerate(sorted(os.listdir(self.root_dir))):
                class_dir = os.path.join(self.root_dir, class_name)
                if os.path.isdir(class_dir):
                    self.classes.append(class_name)
                    self.class_to_idx[class_name] = class_idx
                    for img_name in os.listdir(class_dir):
                        img_path = os.path.join(class_dir, img_name)
                        if self._is_valid_image(img_path):
                            self.samples.append((img_path, class_idx))
        else:
            # 不使用子目錄時，所有影像被視為同一類別
            class_name = "all"
            self.classes.append(class_name)
            self.class_to_idx[class_name] = 0
            for img_name in os.listdir(self.root_dir):
                img_path = os.path.join(self.root_dir, img_name)
                if self._is_valid_image(img_path):
                    self.samples.append((img_path, 0))

    def _is_valid_image(self, img_path):
        """
        檢查影像文件是否為允許的格式。
        """
        return img_path.split('.')[-1].lower() in self.allowed_extensions

    def _get_default_transform(self):
        """
        提供預設的資料增強變換。
        """
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
        回傳資料集中樣本的總數。
        """
        return len(self.samples)

    def __getitem__(self, idx):
        """
        回傳指定索引的影像和對應的類別索引。
        Args:
            idx (int): 索引。
        Returns:
            Tuple[Tensor, Tensor]: 影像張量和類別索引。
        """
        img_path, class_idx = self.samples[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return None, None
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(class_idx, dtype=torch.long) # CrossEntropy needs long(int64) format


# Example Usage
# import torch
# from torch.utils.data import DataLoader
# from torchvision.transforms import RandAugment
# from custom_dataset import CustomImageDataset

# # 指定資料集根目錄
# root_dir = "/path/to/your/data"

# # 建立訓練資料集
# train_dataset = CustomImageDataset(
#     root_dir=root_dir,
#     transform=None,  # 使用預設資料增強
#     is_train=True,   # 設定為訓練資料集
#     use_subdirectories=True,  # 使用子目錄作為類別
#     allowed_extensions=['jpg', 'jpeg', 'png']  # 允許的影像格式
# )

# # 建立驗證資料集
# val_dataset = CustomImageDataset(
#     root_dir=root_dir,
#     transform=None,  # 使用預設資料增強
#     is_train=False,  # 設定為驗證資料集
#     use_subdirectories=True,  # 使用子目錄作為類別
#     allowed_extensions=['jpg', 'jpeg', 'png']  # 允許的影像格式
# )

# # 建立 DataLoader
# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# # 檢視資料集資訊
# print(f"Train dataset size: {len(train_dataset)}")
# print(f"Validation dataset size: {len(val_dataset)}")

# # 範例迭代 DataLoader
# for images, labels in train_loader:
#     print(f"Images batch shape: {images.shape}")
#     print(f"Labels batch shape: {labels.shape}")
#     break

